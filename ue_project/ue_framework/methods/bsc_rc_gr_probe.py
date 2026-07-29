from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics import YOLO

from ..data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)
from ..io_utils import atomic_write_json
from ..support import build_support_mask
from ..ultra.hijacked_loss import HijackedV8Loss
from .alce_acgt import build_pag_gate
from .background_spectral_basis import (
    BackgroundSpectralBasis,
    band_mask,
    build_background_spectral_basis,
    deterministic_two_crops,
    spectrum_energy_ratios,
    validate_repository_manifest,
)
from .cicr import (
    CICRPrototypeBank,
    ClassificationResiduals,
    classification_residuals,
)
from .constraint_gradient_router import (
    ConstraintTerm,
    backtracking_candidate,
    route_coefficient_gradient,
)
from .detector_tower_hooks import YOLODetectTowerCapture
from .fourier import build_fourier_pattern, sample_bandfreq_coords
from .shadow_tal import (
    DifferentiableShadowTAL,
    NonTargetConstraintSet,
    TargetRouteResult,
    build_non_target_constraints,
    compute_target_route,
)


CARRIER_IDS = ("C0", "C1-L", "C2-L", "C2-LM")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_probe_json(path: Path, value: Any) -> None:
    atomic_write_json(str(path), _json_safe(value))


def validate_probe_config(config: Mapping[str, Any]) -> None:
    required_sections = {
        "spec",
        "dataset",
        "model",
        "background",
        "carrier",
        "split",
        "phase_a",
        "phase_b",
        "phase_c",
        "runtime",
    }
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"Missing probe config sections: {missing}")
    if str(config["spec"].get("spec_id")) != "TAUSB-BSC-RC-GR-v1":
        raise ValueError("Probe config spec_id mismatch.")
    if str(config["spec"].get("exp_id")) != "TAUSB-BSC-RC-GR-MECH-S0":
        raise ValueError("Probe config exp_id mismatch.")
    if int(config["spec"].get("seed", -1)) != 0:
        raise ValueError("Probe seed must remain 0.")
    if int(config["dataset"].get("target_class_id", -1)) != 14:
        raise ValueError("Probe target_class_id must remain 14.")
    if int(config["model"].get("num_classes", -1)) != 20:
        raise ValueError("Probe num_classes must remain 20.")
    if abs(
        float(config["carrier"].get("epsilon", -1)) - pytest_approx_16_255()
    ) > 1e-9:
        raise ValueError("Probe epsilon must remain 16/255.")
    if int(config["carrier"].get("num_bases", -1)) != 16:
        raise ValueError("Probe coefficient space must remain 16x3=48.")
    if int(config["model"].get("image_size", -1)) != 640:
        raise ValueError("Probe image_size must remain 640.")
    if int(config["carrier"].get("resolution", -1)) != 640:
        raise ValueError("Probe carrier resolution must remain 640.")
    for field in (
        ("dataset", "root"),
        ("model", "surrogate_checkpoint"),
        ("background", "source_manifest"),
        ("background", "source_local_map"),
        ("split", "manifest"),
        ("runtime", "artifact_root"),
    ):
        section, key = field
        if not str(config[section].get(key, "")).strip():
            raise ValueError(f"Probe config requires {section}.{key}.")
    if (
        str(config["split"].get("required_protocol_prefix", ""))
        != "TAUSB-ALCE-CTX-AUDIT-v1"
    ):
        raise ValueError("Probe must reuse the frozen ALCE context-audit split.")
    if not all(
        bool(config[phase].get("enabled", False))
        for phase in ("phase_a", "phase_b", "phase_c")
    ):
        raise ValueError("Approved probe requires Phase A, B, and C enabled.")
    if int(config["phase_c"].get("max_backtracks", -1)) != 5:
        raise ValueError("Phase C max_backtracks must remain 5.")
    if bool(config["phase_c"].get("enabled", False)) and not bool(
        config["phase_b"].get("enabled", False)
    ):
        raise ValueError("Phase C cannot be enabled while Phase B is disabled.")


def pytest_approx_16_255() -> float:
    return 16.0 / 255.0


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_background_sources(
    manifest_path: Path,
    local_map_path: Path,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], str]:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("Background source manifest must be a JSON list.")
    validate_repository_manifest(manifest)
    local_map = _load_json(local_map_path)
    if not isinstance(local_map, dict):
        raise ValueError("Background local map must be a JSON object.")

    images: list[torch.Tensor] = []
    for entry in manifest:
        source_id = str(entry["source_id"])
        local_value = local_map.get(source_id)
        if not isinstance(local_value, str) or not local_value:
            raise ValueError(f"Missing local path for source_id={source_id}.")
        local_path = Path(local_value)
        if not local_path.is_absolute() or not local_path.is_file():
            raise ValueError(
                f"Local source path must be an existing absolute file: {source_id}"
            )
        digest = _file_sha256(local_path)
        if digest.lower() != str(entry["sha256"]).lower():
            raise ValueError(f"SHA256 mismatch for source_id={source_id}.")
        image = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode background source: {source_id}")
        actual_height, actual_width = image.shape[:2]
        if (
            actual_width != int(entry["width"])
            or actual_height != int(entry["height"])
        ):
            raise ValueError(
                f"Image dimensions mismatch for source_id={source_id}: "
                f"manifest={entry['width']}x{entry['height']} "
                f"actual={actual_width}x{actual_height}."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(image).permute(2, 0, 1).contiguous())
    return images, [dict(item) for item in manifest], canonical_hash(manifest)


def build_background_basis_registry(
    source_images: Sequence[torch.Tensor],
    *,
    resolution: int,
    num_bases: int,
    seed: int,
) -> dict[str, BackgroundSpectralBasis]:
    return {
        "C1-L": build_background_spectral_basis(
            source_images,
            resolution=resolution,
            num_bases=num_bases,
            bands=((2.0, 8.0),),
            phase_mode="raw",
            seed=seed,
        ),
        "C2-L": build_background_spectral_basis(
            source_images,
            resolution=resolution,
            num_bases=num_bases,
            bands=((2.0, 8.0),),
            phase_mode="scrambled",
            seed=seed,
        ),
        "C2-LM": build_background_spectral_basis(
            source_images,
            resolution=resolution,
            num_bases=num_bases,
            bands=((2.0, 8.0), (8.0, 24.0)),
            phase_mode="scrambled",
            seed=seed,
        ),
    }


class ProbeCarrier(nn.Module):
    def __init__(
        self,
        *,
        carrier_id: str,
        resolution: int,
        epsilon: float,
        num_bases: int,
        seed: int,
        basis: torch.Tensor | None = None,
        synthetic_coords: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        if carrier_id not in CARRIER_IDS:
            raise ValueError(f"Unknown carrier_id={carrier_id}.")
        self.carrier_id = carrier_id
        self.resolution = int(resolution)
        self.epsilon = float(epsilon)
        self.num_bases = int(num_bases)
        self.seed = int(seed)
        if carrier_id == "C0":
            if synthetic_coords is None or len(synthetic_coords) != num_bases:
                raise ValueError("C0 requires exactly num_bases synthetic coords.")
            self.synthetic_coords = tuple(
                (int(y), int(x)) for y, x in synthetic_coords
            )
            self.register_buffer("basis", torch.empty((0, resolution, resolution)))
        else:
            if basis is None or basis.shape != (
                num_bases,
                resolution,
                resolution,
            ):
                raise ValueError("Background carrier basis shape mismatch.")
            self.synthetic_coords = ()
            self.register_buffer("basis", basis.detach().float().clone())
        self.coefficients = nn.Parameter(torch.zeros((num_bases, 3)))

    def reset_common_coefficients(self, *, scale: float = 0.25) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 701)
        values = torch.randn(
            self.coefficients.shape,
            generator=generator,
            dtype=torch.float32,
        )
        values = values / values.abs().amax().clamp_min(1e-12) * float(scale)
        self.coefficients.data.copy_(values.to(self.coefficients))

    def pattern(self, height: int, width: int) -> torch.Tensor:
        channels: list[torch.Tensor] = []
        for channel in range(3):
            amplitudes = torch.tanh(self.coefficients[:, channel]) * self.epsilon
            if self.carrier_id == "C0":
                base = build_fourier_pattern(
                    self.resolution,
                    self.resolution,
                    self.synthetic_coords,
                    amplitudes,
                    self.coefficients.device,
                ).squeeze(0).squeeze(0)
            else:
                base = (
                    self.basis
                    * amplitudes.view(-1, 1, 1)
                ).sum(dim=0)
                base = base / base.abs().amax().clamp_min(1e-6)
            resized = F.interpolate(
                base.view(1, 1, self.resolution, self.resolution),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            channels.append(resized)
        return torch.cat(channels, dim=1)

    @staticmethod
    def _jnd(images: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
        gray = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )
        kx = images.new_tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]
        )
        ky = images.new_tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]
        )
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        magnitude = magnitude - magnitude.amin(dim=(2, 3), keepdim=True)
        magnitude = magnitude / magnitude.amax(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-6)
        return floor + (1.0 - floor) * magnitude

    def apply(
        self,
        images: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support.shape != (images.shape[0], 1, *images.shape[-2:]):
            raise ValueError("Carrier support must have shape [B,1,H,W].")
        pattern = torch.tanh(self.pattern(*images.shape[-2:]) * 4.0)
        raw = pattern * self._jnd(images).repeat(1, 3, 1, 1)
        raw = raw * support.repeat(1, 3, 1, 1)
        perturbation = raw.clamp(-self.epsilon, self.epsilon)
        return (images + perturbation).clamp(0, 1), perturbation


@dataclass
class ProbeBatch:
    images: torch.Tensor
    support: torch.Tensor
    yolo_batch: dict[str, Any]
    image_ids: list[str]
    person_cooccur: list[bool]
    person_scale_group: list[str]


@dataclass
class ProbeObservation:
    target_residuals: ClassificationResiduals
    non_target_residuals: ClassificationResiduals
    box_residuals: ClassificationResiduals
    clean_target_vectors: tuple[torch.Tensor, ...]
    adv_target_vectors: tuple[torch.Tensor, ...]
    route: TargetRouteResult
    constraints: NonTargetConstraintSet
    materialized_energy: dict[str, float]
    image_ids: list[str]
    person_cooccur: list[bool]
    person_scale_group: list[str]


def _split_flat_gate(
    gate: torch.Tensor,
    features: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    expected = sum(feature.shape[-2] * feature.shape[-1] for feature in features)
    if gate.shape[1] != expected:
        raise ValueError(
            f"Flattened gate anchors={gate.shape[1]} != feature cells={expected}."
        )
    result: list[torch.Tensor] = []
    offset = 0
    for feature in features:
        height, width = feature.shape[-2:]
        count = height * width
        result.append(gate[:, offset : offset + count].reshape(-1, 1, height, width))
        offset += count
    return tuple(result)


def _masked_vectors(
    features: Sequence[torch.Tensor],
    gates: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    zeros = [torch.zeros_like(feature) for feature in features]
    return classification_residuals(zeros, features, gates).vectors


def _assigned_gt_boxes(
    target_gt_index: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    if target_gt_index.ndim == 3 and target_gt_index.shape[-1] == 1:
        target_gt_index = target_gt_index[..., 0]
    if gt_boxes.shape[1] == 0:
        raise ValueError("Real assignment has no GT boxes.")
    index = target_gt_index.long().clamp(0, gt_boxes.shape[1] - 1)
    return gt_boxes.gather(1, index.unsqueeze(-1).expand(-1, -1, 4))


class ProbeEngine:
    def __init__(
        self,
        config: Mapping[str, Any],
        device: torch.device,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.target_class_id = int(config["dataset"]["target_class_id"])
        self.num_classes = int(config["model"]["num_classes"])
        checkpoint = str(
            checkpoint_path
            if checkpoint_path is not None
            else config["model"]["surrogate_checkpoint"]
        )
        wrapper = YOLO(checkpoint)
        self.model = wrapper.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.capture = YOLODetectTowerCapture(
            self.model,
            expected_num_classes=self.num_classes,
        )
        self.hijacked = HijackedV8Loss.from_surrogate(
            self.model,
            num_classes=self.num_classes,
            target_class_id=self.target_class_id,
        )
        self.shadow_tal = DifferentiableShadowTAL(
            target_class_id=self.target_class_id,
            alpha=float(config["phase_b"]["align_alpha"]),
            beta=float(config["phase_b"]["align_beta"]),
            topk=int(config["phase_b"]["assignment_topk"]),
        )

    def close(self) -> None:
        self.capture.close()

    def observe(
        self,
        batch: ProbeBatch,
        carrier: ProbeCarrier,
        *,
        route: str,
        view_gain: float = 1.0,
    ) -> ProbeObservation:
        images = batch.images
        adv_images, perturbation = carrier.apply(images, batch.support)
        clean_view = (images * float(view_gain)).clamp(0, 1)
        adv_view = (adv_images * float(view_gain)).clamp(0, 1)
        with torch.no_grad():
            with self.capture.record("clean"):
                clean_output = self.model(clean_view)
            clean_features = self.capture.take("clean")
            clean_predictions = (
                clean_output[0] if isinstance(clean_output, (tuple, list)) else clean_output
            )
            self.hijacked.last_real_assign = {}
            self.hijacked.get_assigned_targets_and_loss(
                clean_predictions,
                batch.yolo_batch,
            )
            raw_assign = self.hijacked.last_real_assign
            required = ("fg_mask", "target_labels", "target_scores", "target_gt_idx")
            if any(not torch.is_tensor(raw_assign.get(name)) for name in required):
                raise RuntimeError("Clean real TAL assignment is incomplete.")
            real_assign = {
                name: raw_assign[name].detach().clone() for name in required
            }
            clean_cache = self.hijacked.cache_assign_inputs_only(
                clean_predictions,
                batch.yolo_batch,
                image_shape=images.shape[-2:],
                assignment_topk=int(self.config["phase_b"]["assignment_topk"]),
            )

        with self.capture.record("adv"):
            adv_output = self.model(adv_view)
        adv_features = self.capture.take("adv")
        adv_predictions = (
            adv_output[0] if isinstance(adv_output, (tuple, list)) else adv_output
        )
        adv_cache = self.hijacked.cache_assign_inputs_only(
            adv_predictions,
            batch.yolo_batch,
            image_shape=images.shape[-2:],
            assignment_topk=int(self.config["phase_b"]["assignment_topk"]),
        )

        real_foreground = real_assign["fg_mask"].bool()
        labels = real_assign["target_labels"].long()
        if labels.ndim == 3:
            labels = labels[..., 0]
        strict_target = real_foreground & (labels == self.target_class_id)
        layer_sizes = [
            feature.shape[-2] * feature.shape[-1]
            for feature in adv_features.classification
        ]
        pag_gate, _ = build_pag_gate(
            strict_gate_1d=strict_target,
            target_scores=real_assign["target_scores"],
            target_class_id=self.target_class_id,
            top_ratio=self.config["phase_b"]["pag_layer_ratios"],
            min_keep=self.config["phase_b"]["pag_min_pos"],
            layer_sizes=layer_sizes,
        )
        target_gates = _split_flat_gate(
            pag_gate.bool(),
            adv_features.classification,
        )
        non_target_gates = _split_flat_gate(
            real_foreground & (labels != self.target_class_id),
            adv_features.classification,
        )
        target_residuals = classification_residuals(
            clean_features.classification,
            adv_features.classification,
            target_gates,
        )
        non_target_residuals = classification_residuals(
            clean_features.classification,
            adv_features.classification,
            non_target_gates,
        )
        box_residuals = classification_residuals(
            clean_features.box,
            adv_features.box,
            target_gates,
        )

        route_result = compute_target_route(
            route=route,
            adv_class_logits=adv_cache["pred_scores_logits"],
            adv_boxes=adv_cache["pred_bboxes"],
            clean_boxes=clean_cache["pred_bboxes"],
            target_gate=pag_gate,
            target_class_id=self.target_class_id,
            num_classes=self.num_classes,
            box_teacher_weight=float(self.config["phase_b"]["box_teacher_weight"]),
            shadow_tal=self.shadow_tal,
            gt_labels=clean_cache["gt_labels"],
            gt_bboxes=clean_cache["gt_bboxes"],
            mask_gt=clean_cache["mask_gt"],
        )
        assigned_gt = _assigned_gt_boxes(
            real_assign["target_gt_idx"],
            clean_cache["gt_bboxes"],
        )
        constraints = build_non_target_constraints(
            clean_class_logits=clean_cache["pred_scores_logits"],
            adv_class_logits=adv_cache["pred_scores_logits"],
            clean_boxes=clean_cache["pred_bboxes"],
            adv_boxes=adv_cache["pred_bboxes"],
            assigned_gt_boxes=assigned_gt,
            assigned_labels=labels,
            real_foreground=real_foreground,
            target_class_id=self.target_class_id,
            num_classes=self.num_classes,
            tau_cls=float(self.config["phase_b"]["tau_cls"]),
            tau_box=float(self.config["phase_b"]["tau_box"]),
        )
        energy = spectrum_energy_ratios(
            perturbation.detach()
            .reshape(-1, *perturbation.shape[-2:])
            .cpu()
        )
        return ProbeObservation(
            target_residuals=target_residuals,
            non_target_residuals=non_target_residuals,
            box_residuals=box_residuals,
            clean_target_vectors=_masked_vectors(
                clean_features.classification,
                target_gates,
            ),
            adv_target_vectors=_masked_vectors(
                adv_features.classification,
                target_gates,
            ),
            route=route_result,
            constraints=constraints,
            materialized_energy=energy,
            image_ids=batch.image_ids,
            person_cooccur=batch.person_cooccur,
            person_scale_group=batch.person_scale_group,
        )


def _scale_group(annotations: Sequence[Mapping[str, Any]], target_class_id: int) -> str:
    areas = [
        float(item["bbox"][2]) * float(item["bbox"][3])
        for item in annotations
        if int(item.get("cls", -1)) == target_class_id
    ]
    area = max(areas, default=0.0)
    if area < 0.02:
        return "small"
    if area < 0.15:
        return "medium"
    return "large"


def load_probe_batch(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    image_size: int,
    target_class_id: int,
    device: torch.device,
) -> ProbeBatch:
    images: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    classes: list[list[float]] = []
    boxes: list[list[float]] = []
    batch_indices: list[int] = []
    image_ids: list[str] = []
    cooccur: list[bool] = []
    scale_groups: list[str] = []
    for batch_index, image_path in enumerate(image_paths):
        image = load_image_rgb_float(str(image_path))
        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        annotations = read_yolo_annotations(
            label_path_for_image(str(image_path), str(label_dir))
        )
        support = build_support_mask(
            image_shape=image.shape,
            annotations=annotations,
            target_class_id=target_class_id,
            support_type="mask",
            ring_width=4,
            mask_path=None,
        )
        images.append(torch.from_numpy(image).permute(2, 0, 1).float())
        supports.append(torch.from_numpy(support).unsqueeze(0).float())
        image_ids.append(image_path.stem)
        cooccur.append(
            any(int(item.get("cls", -1)) != target_class_id for item in annotations)
        )
        scale_groups.append(_scale_group(annotations, target_class_id))
        for item in annotations:
            classes.append([float(item["cls"])])
            boxes.append([float(value) for value in item["bbox"]])
            batch_indices.append(batch_index)

    batch_size = len(image_paths)
    images_tensor = torch.stack(images).to(device)
    yolo_batch = {
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long, device=device),
        "cls": torch.tensor(classes, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "batch_size": batch_size,
        "img": images_tensor,
    }
    return ProbeBatch(
        images=images_tensor,
        support=torch.stack(supports).to(device),
        yolo_batch=yolo_batch,
        image_ids=image_ids,
        person_cooccur=cooccur,
        person_scale_group=scale_groups,
    )


def load_required_shared_split(
    split_path: Path,
    *,
    target_images: Sequence[Path],
    required_protocol_prefix: str,
) -> dict[str, Any]:
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Required shared split manifest is missing: {split_path}"
        )
    manifest = _load_json(split_path)
    if not isinstance(manifest, dict):
        raise ValueError("Shared split manifest must be a JSON object.")
    expected_hash = str(manifest.get("split_hash", ""))
    without_hash = {
        key: value for key, value in manifest.items() if key != "split_hash"
    }
    if not expected_hash or expected_hash != canonical_hash(without_hash):
        raise ValueError("Shared split hash is invalid.")
    protocol_id = str(manifest.get("protocol_id", ""))
    if not protocol_id.startswith(required_protocol_prefix):
        raise ValueError("Shared split protocol_id mismatch.")
    calibration = manifest.get("calibration")
    heldout = manifest.get("heldout")
    if (
        not isinstance(calibration, list)
        or not calibration
        or not isinstance(heldout, list)
        or not heldout
        or not all(isinstance(item, str) and item for item in calibration + heldout)
    ):
        raise ValueError(
            "Shared split requires non-empty string calibration/heldout entries."
        )
    if set(calibration) & set(heldout):
        raise ValueError("Calibration and heldout image IDs overlap.")

    by_id: dict[str, Path] = {}
    by_resolved: dict[str, Path] = {}
    for path in target_images:
        resolved = path.resolve()
        if path.stem in by_id:
            raise ValueError(f"Duplicate target image id: {path.stem}")
        by_id[path.stem] = resolved
        by_resolved[str(resolved)] = resolved

    def resolve_entries(entries: Sequence[str]) -> list[str]:
        resolved_entries: list[str] = []
        for entry in entries:
            path = Path(entry)
            candidate = by_resolved.get(str(path.resolve()))
            if candidate is None:
                candidate = by_id.get(path.stem)
            if candidate is None:
                raise ValueError(
                    f"Shared split contains a missing/non-target image: {entry}"
                )
            resolved_entries.append(str(candidate))
        return resolved_entries

    runtime = dict(manifest)
    runtime["calibration"] = resolve_entries(calibration)
    runtime["heldout"] = resolve_entries(heldout)
    runtime["split_hash"] = expected_hash
    runtime["shared_split_manifest"] = manifest
    return runtime


def make_batches(
    paths: Sequence[str],
    *,
    batch_size: int,
) -> list[list[Path]]:
    return [
        [Path(value) for value in paths[index : index + batch_size]]
        for index in range(0, len(paths), batch_size)
    ]


def _concat_residuals(
    observations: Sequence[ProbeObservation],
    attribute: str,
) -> ClassificationResiduals:
    selected = [getattr(item, attribute) for item in observations]
    num_scales = len(selected[0].vectors)
    return ClassificationResiduals(
        vectors=tuple(
            torch.cat([item.vectors[scale] for item in selected], dim=0)
            for scale in range(num_scales)
        ),
        gate_valid=tuple(
            torch.cat([item.gate_valid[scale] for item in selected], dim=0)
            for scale in range(num_scales)
        ),
        gate_mass=tuple(
            torch.cat([item.gate_mass[scale] for item in selected], dim=0)
            for scale in range(num_scales)
        ),
    )


def fit_prototype_bank(
    observations: Sequence[ProbeObservation],
    *,
    momentum: float,
) -> CICRPrototypeBank:
    residuals = _concat_residuals(observations, "target_residuals")
    bank = CICRPrototypeBank(num_scales=len(residuals.vectors), momentum=momentum)
    warmup = [
        vector[valid].detach()
        for vector, valid in zip(residuals.vectors, residuals.gate_valid)
    ]
    bank.calibrate_energy_floors(warmup)
    bank.update(residuals, split="train")
    return bank


def _residual_cosines(
    observations: Sequence[ProbeObservation],
    bank: CICRPrototypeBank,
) -> tuple[list[float], list[dict[str, Any]]]:
    all_values: list[float] = []
    records: list[dict[str, Any]] = []
    for observation in observations:
        per_image: list[list[float]] = [[] for _ in observation.image_ids]
        for scale, (vectors, valid) in enumerate(
            zip(
                observation.target_residuals.vectors,
                observation.target_residuals.gate_valid,
            )
        ):
            prototype = bank.prototype(scale)
            if prototype is None:
                continue
            floor = float(bank.energy_floors[scale] or 0.0)
            norms = vectors.detach().norm(dim=1)
            eligible = valid & (norms >= floor)
            cosine = F.cosine_similarity(
                vectors.detach(),
                prototype.to(vectors).unsqueeze(0),
                dim=1,
                eps=1e-8,
            )
            for index in range(vectors.shape[0]):
                if bool(eligible[index]):
                    value = float(cosine[index].item())
                    all_values.append(value)
                    per_image[index].append(value)
        for index, image_id in enumerate(observation.image_ids):
            records.append(
                {
                    "image_id": image_id,
                    "person_cooccur": observation.person_cooccur[index],
                    "person_scale_group": observation.person_scale_group[index],
                    "cicr": (
                        float(np.median(per_image[index]))
                        if per_image[index]
                        else float("nan")
                    ),
                }
            )
    return all_values, records


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _quantile(values: Iterable[float], quantile: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, quantile)) if finite else float("nan")


def _mean_residual_energy(
    observations: Sequence[ProbeObservation],
    attribute: str,
) -> float:
    values: list[float] = []
    for observation in observations:
        residuals = getattr(observation, attribute)
        for vector, valid in zip(residuals.vectors, residuals.gate_valid):
            if bool(valid.any()):
                values.extend(
                    float(value)
                    for value in vector.detach()[valid].norm(dim=1).cpu()
                )
    return float(np.mean(values)) if values else float("nan")


def _linear_centroid_accuracy(
    calibration: Sequence[ProbeObservation],
    heldout: Sequence[ProbeObservation],
) -> float:
    def vectors(items: Sequence[ProbeObservation], adv: bool) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        name = "adv_target_vectors" if adv else "clean_target_vectors"
        for item in items:
            scales = getattr(item, name)
            rows.append(torch.cat([value.detach() for value in scales], dim=1))
        return torch.cat(rows, dim=0)

    clean_train = vectors(calibration, False)
    adv_train = vectors(calibration, True)
    direction = adv_train.mean(dim=0) - clean_train.mean(dim=0)
    if float(direction.norm()) <= 1e-12:
        return 0.5
    direction = direction / direction.norm()
    threshold = 0.5 * (
        float((clean_train @ direction).mean())
        + float((adv_train @ direction).mean())
    )
    clean_test = vectors(heldout, False) @ direction
    adv_test = vectors(heldout, True) @ direction
    correct = (clean_test < threshold).sum() + (adv_test >= threshold).sum()
    return float(correct.item() / (clean_test.numel() + adv_test.numel()))


def summarize_observations(
    calibration: Sequence[ProbeObservation],
    heldout: Sequence[ProbeObservation],
    bank: CICRPrototypeBank,
) -> dict[str, Any]:
    cosine_values, records = _residual_cosines(heldout, bank)
    target_energy = _mean_residual_energy(heldout, "target_residuals")
    non_target_energy = _mean_residual_energy(heldout, "non_target_residuals")
    box_energy = _mean_residual_energy(heldout, "box_residuals")
    valid_gate_count = 0
    zero_norm_count = 0
    for observation in heldout:
        for vector, valid in zip(
            observation.target_residuals.vectors,
            observation.target_residuals.gate_valid,
        ):
            valid_gate_count += int(valid.sum().item())
            if bool(valid.any()):
                zero_norm_count += int(
                    (vector.detach()[valid].norm(dim=1) <= 1e-12).sum().item()
                )
    group_values: dict[str, list[float]] = {}
    group_target_energy: dict[str, list[float]] = {}
    group_non_target_energy: dict[str, list[float]] = {}
    for record in records:
        value = float(record["cicr"])
        if not math.isfinite(value):
            continue
        cooccur_key = "person_cooccur" if record["person_cooccur"] else "person_only"
        group_values.setdefault(cooccur_key, []).append(value)
        group_values.setdefault(str(record["person_scale_group"]), []).append(value)
    for observation in heldout:
        for image_index, cooccur in enumerate(observation.person_cooccur):
            group = "person_cooccur" if cooccur else "person_only"
            target_values: list[float] = []
            non_target_values: list[float] = []
            for target_vector, target_valid, non_target_vector, non_target_valid in zip(
                observation.target_residuals.vectors,
                observation.target_residuals.gate_valid,
                observation.non_target_residuals.vectors,
                observation.non_target_residuals.gate_valid,
            ):
                if bool(target_valid[image_index]):
                    target_values.append(
                        float(target_vector[image_index].detach().norm())
                    )
                if bool(non_target_valid[image_index]):
                    non_target_values.append(
                        float(non_target_vector[image_index].detach().norm())
                    )
            if target_values:
                group_target_energy.setdefault(group, []).append(
                    float(np.mean(target_values))
                )
            if non_target_values:
                group_non_target_energy.setdefault(group, []).append(
                    float(np.mean(non_target_values))
                )
    energy_keys = ("low", "mid", "high", "dc")
    materialized = {
        key: float(
            np.mean(
                [
                    observation.materialized_energy[key]
                    for observation in heldout
                ]
            )
        )
        for key in energy_keys
    }
    return {
        "heldout_cicr_median": _median(cosine_values),
        "heldout_cicr_q25": _quantile(cosine_values, 0.25),
        "target_residual_energy": target_energy,
        "non_target_residual_energy": non_target_energy,
        "non_target_target_energy_ratio": non_target_energy
        / max(target_energy, 1e-12),
        "box_residual_energy": box_energy,
        "linear_centroid_accuracy": _linear_centroid_accuracy(
            calibration,
            heldout,
        ),
        "group_cicr_median": {
            name: _median(values) for name, values in group_values.items()
        },
        "group_non_target_target_energy_ratio": {
            name: float(np.mean(group_non_target_energy.get(name, [])))
            / max(float(np.mean(target_values)), 1e-12)
            for name, target_values in group_target_energy.items()
            if target_values and group_non_target_energy.get(name)
        },
        "materialized_spectrum_energy": materialized,
        "valid_residual_count": len(cosine_values),
        "target_residual_zero_norm_ratio": zero_norm_count
        / max(valid_gate_count, 1),
        "per_image": records,
    }


def _constraint_terms(
    constraint_set: NonTargetConstraintSet,
    *,
    tau_cls: float,
    tau_box: float,
) -> list[ConstraintTerm]:
    terms: list[ConstraintTerm] = []
    for item in constraint_set.constraints:
        terms.append(
            ConstraintTerm(
                f"class_{item.class_id}_cls",
                item.cls_margin,
                tau_cls,
            )
        )
        terms.append(
            ConstraintTerm(
                f"class_{item.class_id}_box",
                item.box_margin,
                tau_box,
            )
        )
    return terms


def evaluate_phase_a(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    split_hash: str = "",
    source_manifest_hash: str = "",
) -> dict[str, Any]:
    baseline = metrics["C0"]
    raw = metrics["C1-L"]
    scrambled_low = metrics["C2-L"]
    semantic_dependence_failure = (
        raw["heldout_cicr_median"] - scrambled_low["heldout_cicr_median"] > 0.10
        and raw["source_max_abs_correlation"]
        > scrambled_low["source_max_abs_correlation"]
    )
    low_only_failure = (
        scrambled_low["materialized_spectrum_energy"]["high"] > 0.30
        or scrambled_low["heldout_cicr_median"] < 0.10
    )
    candidates: dict[str, dict[str, Any]] = {}
    for carrier_id in ("C2-L", "C2-LM"):
        candidate = metrics[carrier_id]
        intended = candidate["materialized_spectrum_energy"]["low"]
        if carrier_id == "C2-LM":
            intended += candidate["materialized_spectrum_energy"]["mid"]
        checks = {
            "cicr_improvement": candidate["heldout_cicr_median"]
            - baseline["heldout_cicr_median"]
            >= 0.10,
            "cicr_q25_positive": candidate["heldout_cicr_q25"] > 0,
            "non_target_ratio": candidate["non_target_target_energy_ratio"]
            <= baseline["non_target_target_energy_ratio"] * 1.05,
            "box_leakage": candidate["box_residual_energy"]
            <= baseline["box_residual_energy"] * 1.05,
            "intended_band": intended >= 0.70,
            "finite": bool(candidate.get("finite", False)),
            "protocol_hashes": bool(
                split_hash
                and source_manifest_hash
                and candidate.get("basis_hash")
                and candidate.get("coefficient_hash")
            ),
            "zero_norm_ratio": float(
                candidate.get("target_residual_zero_norm_ratio", 1.0)
            )
            <= 0.25,
        }
        if carrier_id == "C2-L":
            checks["low_only_stable"] = not low_only_failure
        candidates[carrier_id] = {
            "checks": checks,
            "pass": all(checks.values()),
        }
    passing = [
        carrier_id
        for carrier_id, result in candidates.items()
        if result["pass"]
    ]
    best = (
        max(passing, key=lambda name: metrics[name]["heldout_cicr_median"])
        if passing
        else None
    )
    return {
        "pass": best is not None and not semantic_dependence_failure,
        "best_background_carrier": best,
        "candidates": candidates,
        "failure_signals": {
            "semantic_dependence": semantic_dependence_failure,
            "low_only_unstable": low_only_failure,
        },
    }


def evaluate_phase_b(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    best_background_arm: str,
) -> dict[str, Any]:
    baseline = arms["A1"]
    candidate = arms[best_background_arm]
    groups = candidate["group_cicr_median"]
    cooccur_gap = abs(
        float(groups.get("person_only", float("nan")))
        - float(groups.get("person_cooccur", float("nan")))
    )
    scale_values = [
        float(groups[name])
        for name in ("small", "medium", "large")
        if name in groups and math.isfinite(float(groups[name]))
    ]
    scale_gap = max(scale_values) - min(scale_values) if scale_values else float("inf")
    leakage_groups = candidate.get(
        "group_non_target_target_energy_ratio",
        {},
    )
    person_only_leakage = float(
        leakage_groups.get("person_only", float("nan"))
    )
    person_cooccur_leakage = float(
        leakage_groups.get("person_cooccur", float("nan"))
    )
    collateral_leakage = (
        math.isfinite(person_only_leakage)
        and math.isfinite(person_cooccur_leakage)
        and person_cooccur_leakage > 1.5 * person_only_leakage
    )
    checks = {
        "background_gain": candidate["heldout_cicr_median"]
        - baseline["heldout_cicr_median"]
        >= 0.05,
        "route_effect": candidate["route_effect"] >= 0.10,
        "cooccur_gap": cooccur_gap <= 0.15,
        "scale_gap": scale_gap <= 0.20,
        "retention_median": candidate["attack_retention_median"] >= 0.30,
        "retention_q25": candidate["attack_retention_q25"] >= 0.10,
        "first_order_violation": candidate["projected_violation_ratio"] <= 0.02,
        "gradient_alignment_finite": math.isfinite(
            float(candidate.get("gradient_alignment_median", float("nan")))
        ),
        "low_retention_ratio": float(
            candidate.get("low_retention_ratio", 1.0)
        )
        < 0.80,
        "zero_norm_ratio": float(
            candidate.get("target_residual_zero_norm_ratio", 1.0)
        )
        <= 0.25,
        "no_collateral_leakage_failure": not collateral_leakage,
    }
    return {
        "pass": all(checks.values()),
        "best_arm": best_background_arm,
        "checks": checks,
        "person_cooccur_gap": cooccur_gap,
        "scale_gap": scale_gap,
        "failure_signals": {
            "person_cooccur_collateral_leakage": collateral_leakage,
            "low_retention": not checks["low_retention_ratio"],
        },
    }


def evaluate_phase_c(off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, Any]:
    off_rate = float(off["actual_violation_rate"])
    on_rate = float(on["actual_violation_rate"])
    reduction = (off_rate - on_rate) / max(off_rate, 1e-12)
    checks = {
        "cicr_retained": float(on["heldout_cicr_median"])
        >= 0.95 * float(off["heldout_cicr_median"]),
        "violation_reduction": reduction >= 0.50,
        "repair_skip_ratio": float(on["repair_skip_ratio"]) < 0.50,
        "null_dimension": float(on["null_dimension_median"]) >= 8,
        "finite": bool(on["finite"]),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "violation_reduction": reduction,
    }


class BSCProbeWorkflow:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        source_manifest: str | None = None,
        source_local_map: str | None = None,
        device_override: str | None = None,
    ) -> None:
        validate_probe_config(config)
        self.config = dict(config)
        self.config_path = config_path.resolve()
        self.project_root = self.config_path.parents[2]
        self.seed = int(config["spec"]["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        runtime_device = (
            str(device_override)
            if device_override is not None
            else str(config["runtime"]["device"])
        )
        if runtime_device.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA probe but CUDA is unavailable.")
            self.device = torch.device(
                runtime_device
                if runtime_device.startswith("cuda")
                else f"cuda:{runtime_device}"
            )

        self.artifact_root = _resolve_path(
            self.project_root,
            str(config["runtime"]["artifact_root"]),
        )
        if self.artifact_root.exists():
            raise FileExistsError(
                f"Fresh probe refuses existing artifact root: {self.artifact_root}"
            )

        dataset_root = _resolve_path(
            self.project_root,
            str(config["dataset"]["root"]),
        )
        self.train_image_dir = _resolve_path(
            dataset_root,
            str(config["dataset"]["train_images"]),
        )
        self.train_label_dir = _resolve_path(
            dataset_root,
            str(config["dataset"]["train_labels"]),
        )
        if not self.train_image_dir.is_dir() or not self.train_label_dir.is_dir():
            raise FileNotFoundError("VOC train image/label directories are missing.")

        target_class_id = int(config["dataset"]["target_class_id"])
        self.target_images = [
            Path(path)
            for path in list_images(str(self.train_image_dir))
            if image_has_target(
                read_yolo_annotations(
                    label_path_for_image(path, str(self.train_label_dir))
                ),
                target_class_id,
            )
        ]
        split_path = _resolve_path(
            self.project_root,
            str(config["split"]["manifest"]),
        )
        self.split = load_required_shared_split(
            split_path,
            target_images=self.target_images,
            required_protocol_prefix=str(
                config["split"]["required_protocol_prefix"]
            ),
        )

        manifest_value = source_manifest or str(
            config["background"]["source_manifest"]
        )
        local_map_value = source_local_map or str(
            config["background"]["source_local_map"]
        )
        self.source_images, self.source_manifest, self.source_manifest_hash = (
            load_background_sources(
                _resolve_path(self.project_root, manifest_value),
                _resolve_path(self.project_root, local_map_value),
            )
        )
        carrier_cfg = config["carrier"]
        self.basis_registry = build_background_basis_registry(
            self.source_images,
            resolution=int(carrier_cfg["resolution"]),
            num_bases=int(carrier_cfg["num_bases"]),
            seed=int(carrier_cfg["seed"]),
        )
        self.synthetic_coords, self.synthetic_source = (
            self._load_synthetic_coords()
        )
        self.surrogate_checkpoint_path = _resolve_path(
            self.project_root,
            str(config["model"]["surrogate_checkpoint"]),
        )
        if not self.surrogate_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Surrogate checkpoint is missing: {self.surrogate_checkpoint_path}"
            )
        self.surrogate_checkpoint_hash = _file_sha256(
            self.surrogate_checkpoint_path
        )
        self.engine = ProbeEngine(
            config,
            self.device,
            checkpoint_path=self.surrogate_checkpoint_path,
        )
        try:
            # Create the fresh run root only after all external inputs and the
            # surrogate have validated. Missing sources/checkpoints therefore
            # do not leave an empty root that blocks a corrected rerun.
            self.artifact_root.mkdir(parents=True)
            self.basis_dir = self.artifact_root / "bases"
            self.basis_dir.mkdir()
            self.logs_dir = self.artifact_root / "logs"
            self.logs_dir.mkdir()
            self._write_initial_metadata()
        except Exception:
            self.engine.close()
            raise

    def close(self) -> None:
        self.engine.close()

    def _write_initial_metadata(self) -> None:
        _write_probe_json(
            self.artifact_root / "config_resolved.json",
            self.config,
        )
        _write_probe_json(
            self.artifact_root / "source_manifest.json",
            self.source_manifest,
        )
        _write_probe_json(
            self.artifact_root / "split_manifest.json",
            self.split["shared_split_manifest"],
        )
        _write_probe_json(
            self.artifact_root / "protocol.json",
            {
                "spec_id": self.config["spec"]["spec_id"],
                "exp_id": self.config["spec"]["exp_id"],
                "seed": self.seed,
                "split_hash": self.split["split_hash"],
                "source_manifest_hash": self.source_manifest_hash,
                "surrogate_checkpoint": str(self.surrogate_checkpoint_path),
                "surrogate_checkpoint_sha256": self.surrogate_checkpoint_hash,
                "background_basis_hashes": {
                    carrier_id: basis.basis_hash
                    for carrier_id, basis in self.basis_registry.items()
                },
                "background_basis_ranks": {
                    carrier_id: basis.rank
                    for carrier_id, basis in self.basis_registry.items()
                },
                "synthetic_source": self.synthetic_source,
                "config_hash": canonical_hash(self.config),
                "claim_boundary": "surrogate-only mechanism probe; not victim UE evidence",
            },
        )

    def _load_synthetic_coords(self) -> tuple[list[tuple[int, int]], str]:
        carrier_cfg = self.config["carrier"]
        path_value = str(carrier_cfg.get("synthetic_global_params_path", "") or "")
        if path_value:
            path = _resolve_path(self.project_root, path_value)
            if path.is_file():
                pack = torch.load(path, map_location="cpu")
                coords = [tuple(map(int, item)) for item in pack.get("coords", [])]
                if len(coords) != int(carrier_cfg["num_bases"]):
                    raise ValueError(
                        "Synthetic global params active coord count is not 16."
                    )
                return coords, str(path)

        coords, _ = sample_bandfreq_coords(
            h=int(carrier_cfg["resolution"]),
            w=int(carrier_cfg["resolution"]),
            band_names=("low", "mid", "high"),
            band_num_bases=(1, 13, 2),
            band_radius_ranges={
                "low": (2, 8),
                "mid": (8, 24),
                "high": (24, 48),
            },
            seed=int(carrier_cfg["seed"]),
            enable_search=True,
        )
        return coords, "deterministic_band_fallback"

    def _carrier(self, carrier_id: str, *, initialize: bool) -> ProbeCarrier:
        carrier_cfg = self.config["carrier"]
        basis = (
            None
            if carrier_id == "C0"
            else self.basis_registry[carrier_id].bases
        )
        carrier = ProbeCarrier(
            carrier_id=carrier_id,
            resolution=int(carrier_cfg["resolution"]),
            epsilon=float(carrier_cfg["epsilon"]),
            num_bases=int(carrier_cfg["num_bases"]),
            seed=int(carrier_cfg["seed"]),
            basis=basis,
            synthetic_coords=self.synthetic_coords if carrier_id == "C0" else None,
        ).to(self.device)
        if initialize:
            carrier.reset_common_coefficients(
                scale=float(carrier_cfg["frozen_coefficient_scale"])
            )
        return carrier

    def _save_basis_registry(self) -> None:
        for carrier_id, basis in self.basis_registry.items():
            mode_map = {
                "C1-L": "background_raw_low",
                "C2-L": "background_scrambled_low",
                "C2-LM": "background_scrambled_low_mid",
            }
            torch.save(
                {
                    "bases": basis.bases,
                    "metadata": {
                        "carrier_id": carrier_id,
                        "carrier_basis_mode": mode_map[carrier_id],
                        "basis_hash": basis.basis_hash,
                        "source_hash": basis.source_hash,
                        "source_manifest_hash": self.source_manifest_hash,
                        "bands": basis.bands,
                        "phase_mode": basis.phase_mode,
                        "resolution": basis.resolution,
                        "seed": basis.seed,
                        "rank": basis.rank,
                        "singular_values": basis.singular_values.tolist(),
                    },
                },
                self.basis_dir / f"{carrier_id}.pt",
            )

    def _load_batch(self, paths: Sequence[Path]) -> ProbeBatch:
        return load_probe_batch(
            paths,
            label_dir=self.train_label_dir,
            image_size=int(self.config["model"]["image_size"]),
            target_class_id=int(self.config["dataset"]["target_class_id"]),
            device=self.device,
        )

    def _collect(
        self,
        paths: Sequence[str],
        carrier: ProbeCarrier,
        *,
        route: str,
        gradients: bool,
        view_gain: float = 1.0,
    ) -> list[ProbeObservation]:
        observations: list[ProbeObservation] = []
        batches = make_batches(
            paths,
            batch_size=int(self.config["runtime"]["batch_size"]),
        )
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context:
            for path_batch in batches:
                observations.append(
                    self.engine.observe(
                        self._load_batch(path_batch),
                        carrier,
                        route=route,
                        view_gain=view_gain,
                    )
                )
        return observations

    def _source_correlation(
        self,
        carrier: ProbeCarrier,
        carrier_id: str,
    ) -> float:
        pattern = carrier.pattern(
            carrier.resolution,
            carrier.resolution,
        ).detach().mean(dim=1).reshape(-1)
        pattern = F.normalize(pattern, dim=0)
        bands = (
            ((2.0, 8.0), (8.0, 24.0))
            if carrier_id in {"C0", "C2-LM"}
            else ((2.0, 8.0),)
        )
        mask = band_mask(
            carrier.resolution,
            carrier.resolution,
            bands,
        )
        correlations: list[float] = []
        for source_index, image in enumerate(self.source_images):
            for crop in deterministic_two_crops(
                image,
                resolution=carrier.resolution,
                source_index=source_index,
            ):
                luminance = (
                    0.299 * crop[0]
                    + 0.587 * crop[1]
                    + 0.114 * crop[2]
                )
                spectrum = torch.fft.fft2(luminance.double()) * mask
                source = torch.fft.ifft2(spectrum).real.float()
                source = F.normalize((source - source.mean()).reshape(-1), dim=0)
                correlations.append(float(torch.dot(pattern.cpu(), source)))
        return max(abs(value) for value in correlations)

    def run_phase_a(self) -> dict[str, Any]:
        self._save_basis_registry()
        metrics: dict[str, Any] = {}
        heldout_by_carrier: dict[str, list[ProbeObservation]] = {}
        bank_by_carrier: dict[str, CICRPrototypeBank] = {}
        for carrier_id in CARRIER_IDS:
            carrier = self._carrier(carrier_id, initialize=True)
            calibration = self._collect(
                self.split["calibration"],
                carrier,
                route="easy_cls",
                gradients=False,
            )
            bank = fit_prototype_bank(
                calibration,
                momentum=float(self.config["phase_a"]["prototype_momentum"]),
            )
            heldout = self._collect(
                self.split["heldout"],
                carrier,
                route="easy_cls",
                gradients=False,
            )
            summary = summarize_observations(calibration, heldout, bank)
            eot_heldout = self._collect(
                self.split["heldout"],
                carrier,
                route="easy_cls",
                gradients=False,
                view_gain=float(self.config["phase_a"]["eot_gain"]),
            )
            eot_summary = summarize_observations(calibration, eot_heldout, bank)
            source_correlation = self._source_correlation(
                carrier,
                carrier_id,
            )
            finite_keys = (
                "heldout_cicr_median",
                "heldout_cicr_q25",
                "non_target_target_energy_ratio",
                "box_residual_energy",
                "linear_centroid_accuracy",
                "target_residual_zero_norm_ratio",
            )
            core_finite = all(
                math.isfinite(float(summary[key])) for key in finite_keys
            ) and all(
                math.isfinite(float(value))
                for value in summary["materialized_spectrum_energy"].values()
            )
            summary.update(
                {
                    "carrier_id": carrier_id,
                    "source_max_abs_correlation": source_correlation,
                    "basis_hash": (
                        None
                        if carrier_id == "C0"
                        else self.basis_registry[carrier_id].basis_hash
                    ),
                    "coefficient_hash": hashlib.sha256(
                        carrier.coefficients.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "deterministic_eot": {
                        "gain": float(self.config["phase_a"]["eot_gain"]),
                        "heldout_cicr_median": eot_summary[
                            "heldout_cicr_median"
                        ],
                        "non_target_target_energy_ratio": eot_summary[
                            "non_target_target_energy_ratio"
                        ],
                    },
                    "finite": core_finite
                    and math.isfinite(source_correlation),
                }
            )
            metrics[carrier_id] = summary
            heldout_by_carrier[carrier_id] = heldout
            bank_by_carrier[carrier_id] = bank

        for carrier_id in CARRIER_IDS:
            wrong: dict[str, float] = {}
            for other_id in CARRIER_IDS:
                if other_id == carrier_id:
                    continue
                values, _ = _residual_cosines(
                    heldout_by_carrier[other_id],
                    bank_by_carrier[carrier_id],
                )
                wrong[other_id] = _median(values)
            metrics[carrier_id]["wrong_carrier_cicr_median"] = wrong
        decision = evaluate_phase_a(
            metrics,
            split_hash=self.split["split_hash"],
            source_manifest_hash=self.source_manifest_hash,
        )
        result = {
            "phase": "A",
            "split_hash": self.split["split_hash"],
            "source_manifest_hash": self.source_manifest_hash,
            "metrics": metrics,
            "decision": decision,
        }
        _write_probe_json(self.artifact_root / "phase_a_metrics.json", result)
        return result

    def _phase_b_arm(
        self,
        *,
        arm_id: str,
        carrier_id: str,
        route: str,
    ) -> tuple[dict[str, Any], Path]:
        carrier = self._carrier(carrier_id, initialize=False)
        optimizer = torch.optim.Adam(
            [carrier.coefficients],
            lr=float(self.config["phase_b"]["learning_rate"]),
        )
        calibration_batches = make_batches(
            self.split["calibration"],
            batch_size=int(self.config["runtime"]["batch_size"]),
        )
        heldout_initial = self._collect(
            self.split["heldout"],
            carrier,
            route=route,
            gradients=False,
        )
        initial_route = _median(
            float(item.route.loss.detach()) for item in heldout_initial
        )

        warmup_steps = int(self.config["phase_b"]["warmup_steps"])
        for step in range(warmup_steps):
            batch = self._load_batch(calibration_batches[step % len(calibration_batches)])
            observation = self.engine.observe(batch, carrier, route=route)
            optimizer.zero_grad(set_to_none=True)
            observation.route.loss.backward()
            if carrier.coefficients.grad is None or not torch.isfinite(
                carrier.coefficients.grad
            ).all():
                raise RuntimeError(f"{arm_id} warmup coefficient gradient invalid.")
            optimizer.step()

        calibration = self._collect(
            self.split["calibration"],
            carrier,
            route=route,
            gradients=False,
        )
        bank = fit_prototype_bank(
            calibration,
            momentum=float(self.config["phase_b"]["prototype_momentum"]),
        )
        retention: list[float] = []
        projected_ratios: list[float] = []
        gradient_alignments: list[float] = []
        diagnostic_rows: list[dict[str, Any]] = []
        steps = int(self.config["phase_b"]["optimization_steps"])
        for step in range(steps):
            batch = self._load_batch(calibration_batches[step % len(calibration_batches)])
            observation = self.engine.observe(batch, carrier, route=route)
            cicr_result = bank.loss(observation.target_residuals)
            target_loss = (
                float(self.config["phase_b"]["lambda_cicr"]) * cicr_result.loss
                + float(self.config["phase_b"]["lambda_route"])
                * observation.route.loss
            )
            cicr_gradient = torch.autograd.grad(
                cicr_result.loss,
                carrier.coefficients,
                retain_graph=True,
                allow_unused=True,
            )[0]
            route_gradient = torch.autograd.grad(
                observation.route.loss,
                carrier.coefficients,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if (
                cicr_gradient is None
                or route_gradient is None
                or float(cicr_gradient.norm().detach()) <= 1e-12
                or float(route_gradient.norm().detach()) <= 1e-12
            ):
                gradient_alignment = float("nan")
            else:
                gradient_alignment = float(
                    F.cosine_similarity(
                        cicr_gradient.reshape(1, -1),
                        route_gradient.reshape(1, -1),
                        dim=1,
                    )
                    .detach()
                    .item()
                )
            gradient_alignments.append(gradient_alignment)
            terms = _constraint_terms(
                observation.constraints,
                tau_cls=float(self.config["phase_b"]["tau_cls"]),
                tau_box=float(self.config["phase_b"]["tau_box"]),
            )
            route_diagnostic = route_coefficient_gradient(
                parameter=carrier.coefficients,
                target_loss=target_loss,
                constraints=terms,
                near_boundary=float(self.config["phase_b"]["near_boundary"]),
            )
            target_norm = float(route_diagnostic.target_gradient.norm().detach())
            retention.append(route_diagnostic.attack_retention)
            projected_ratios.append(
                route_diagnostic.max_projected_row_dot / max(target_norm, 1e-12)
            )
            optimizer.zero_grad(set_to_none=True)
            target_loss.backward()
            if carrier.coefficients.grad is None or not torch.isfinite(
                carrier.coefficients.grad
            ).all():
                raise RuntimeError(f"{arm_id} optimization gradient invalid.")
            optimizer.step()
            bank.update(observation.target_residuals, split="train")
            diagnostic_rows.append(
                {
                    "step": step,
                    "loss_cicr": float(cicr_result.loss.detach()),
                    "loss_route": float(observation.route.loss.detach()),
                    "route_mode_diagnostic": route_diagnostic.mode,
                    "constraint_rank": route_diagnostic.rank,
                    "null_dimension": route_diagnostic.null_dimension,
                    "attack_retention": route_diagnostic.attack_retention,
                    "projected_violation_ratio": projected_ratios[-1],
                    "gradient_alignment_cicr_route": gradient_alignment,
                    "target_gradient_norm": target_norm,
                    "active_constraints": route_diagnostic.active_constraints,
                    "violated_constraints": route_diagnostic.violated_constraints,
                }
            )

        final_calibration = self._collect(
            self.split["calibration"],
            carrier,
            route=route,
            gradients=False,
        )
        heldout = self._collect(
            self.split["heldout"],
            carrier,
            route=route,
            gradients=False,
        )
        summary = summarize_observations(final_calibration, heldout, bank)
        final_route = _median(float(item.route.loss.detach()) for item in heldout)
        summary.update(
            {
                "arm_id": arm_id,
                "carrier_id": carrier_id,
                "target_route": route,
                "initial_route_loss": initial_route,
                "final_route_loss": final_route,
                "route_effect": (initial_route - final_route)
                / max(abs(initial_route), 1e-12),
                "attack_retention_median": _median(retention),
                "attack_retention_q25": _quantile(retention, 0.25),
                "projected_violation_ratio": _median(projected_ratios),
                "gradient_alignment_median": _median(gradient_alignments),
                "low_retention_ratio": sum(
                    value < 0.10 for value in retention
                )
                / max(len(retention), 1),
                "coefficient_l2": float(
                    carrier.coefficients.detach().norm().item()
                ),
                "basis_usage_l2": [
                    float(value)
                    for value in carrier.coefficients.detach().norm(dim=1).cpu()
                ],
                "active_basis_fraction": float(
                    (
                        carrier.coefficients.detach().norm(dim=1) > 1e-6
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "diagnostics": diagnostic_rows,
            }
        )
        state_path = self.artifact_root / f"phase_b_{arm_id}_state.pt"
        torch.save(
            {
                "arm_id": arm_id,
                "carrier_id": carrier_id,
                "target_route": route,
                "coefficients": carrier.coefficients.detach().cpu(),
                "prototype_bank": bank.state_dict(),
            },
            state_path,
        )
        return summary, state_path

    def run_phase_b(self, phase_a: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(phase_a["decision"]["pass"]):
            return {
                "phase": "B",
                "decision": {"pass": False, "status": "blocked_by_phase_a"},
                "arms": {},
            }
        best_carrier = str(phase_a["decision"]["best_background_carrier"])
        definitions = {
            "A1": ("C0", "easy_cls"),
            "A2": ("C2-L", "easy_cls"),
            "A3": ("C2-LM", "easy_cls"),
            "A4": (best_carrier, "tal_evasion"),
        }
        arms: dict[str, Any] = {}
        state_paths: dict[str, str] = {}
        for arm_id, (carrier_id, route) in definitions.items():
            summary, state_path = self._phase_b_arm(
                arm_id=arm_id,
                carrier_id=carrier_id,
                route=route,
            )
            arms[arm_id] = summary
            state_paths[arm_id] = str(state_path)
        best_arm = "A2" if best_carrier == "C2-L" else "A3"
        decision = evaluate_phase_b(arms, best_background_arm=best_arm)
        r_minus = arms["A4"]
        r_plus = arms[best_arm]
        r_minus_only = (
            float(r_minus["route_effect"]) >= 0.10
            and (
                float(r_minus["heldout_cicr_median"]) < 0.10
                or float(r_plus["route_effect"]) < 0.10
            )
        )
        decision["failure_signals"]["r_minus_only_evasion"] = r_minus_only
        decision["pass"] = bool(decision["pass"]) and not r_minus_only
        result = {
            "phase": "B",
            "phase_a_best_carrier": best_carrier,
            "arms": arms,
            "state_paths": state_paths,
            "decision": decision,
        }
        _write_probe_json(self.artifact_root / "phase_b_metrics.json", result)
        return result

    @staticmethod
    def _constraint_value_map(
        observation: ProbeObservation,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for item in observation.constraints.constraints:
            values[f"class_{item.class_id}_cls"] = float(item.cls_margin.detach())
            values[f"class_{item.class_id}_box"] = float(item.box_margin.detach())
        return values

    def _phase_c_arm(
        self,
        *,
        state_path: Path,
        routing_on: bool,
    ) -> dict[str, Any]:
        state = torch.load(state_path, map_location="cpu")
        carrier = self._carrier(str(state["carrier_id"]), initialize=False)
        carrier.coefficients.data.copy_(
            state["coefficients"].to(carrier.coefficients)
        )
        bank = CICRPrototypeBank(
            num_scales=3,
            momentum=float(self.config["phase_b"]["prototype_momentum"]),
        )
        bank.load_state_dict(state["prototype_bank"])
        route = str(state["target_route"])
        calibration_batches = make_batches(
            self.split["calibration"],
            batch_size=int(self.config["runtime"]["batch_size"]),
        )
        mode_counts: dict[str, int] = {}
        null_dimensions: list[int] = []
        finite = True
        step_rows: list[dict[str, Any]] = []
        step_size = float(self.config["phase_c"]["learning_rate"])
        for step in range(int(self.config["phase_c"]["optimization_steps"])):
            batch = self._load_batch(calibration_batches[step % len(calibration_batches)])
            observation = self.engine.observe(batch, carrier, route=route)
            cicr = bank.loss(observation.target_residuals)
            target_loss = (
                float(self.config["phase_b"]["lambda_cicr"]) * cicr.loss
                + float(self.config["phase_b"]["lambda_route"])
                * observation.route.loss
            )
            terms = _constraint_terms(
                observation.constraints,
                tau_cls=float(self.config["phase_b"]["tau_cls"]),
                tau_box=float(self.config["phase_b"]["tau_box"]),
            )
            routed = route_coefficient_gradient(
                parameter=carrier.coefficients,
                target_loss=target_loss,
                constraints=terms,
                near_boundary=float(self.config["phase_b"]["near_boundary"]),
            )
            constraint_values_before = self._constraint_value_map(observation)
            active_class_ids = [
                item.class_id for item in observation.constraints.constraints
            ]
            null_dimensions.append(routed.null_dimension)
            if not routing_on:
                selected_mode = "routing_off"
                candidate = (
                    carrier.coefficients.detach()
                    - step_size * routed.target_gradient.detach()
                )
                accepted = True
                attempts = 1
            elif routed.mode == "target":
                selected_mode = "target"
                candidate = (
                    carrier.coefficients.detach()
                    - step_size * routed.gradient.detach()
                )
                accepted = True
                attempts = 1
            elif routed.mode == "skip":
                selected_mode = "skip"
                candidate = carrier.coefficients.detach().clone()
                accepted = False
                attempts = 0
            else:
                limits = {
                    term.name: float(term.tolerance)
                    for term in terms
                    if term.name in routed.active_constraints
                    or term.name in routed.violated_constraints
                }
                baseline = self._constraint_value_map(observation)
                original = carrier.coefficients.detach().clone()

                def evaluate(candidate_value: torch.Tensor) -> Mapping[str, float]:
                    carrier.coefficients.data.copy_(candidate_value)
                    try:
                        with torch.no_grad():
                            current = self.engine.observe(batch, carrier, route=route)
                        values = self._constraint_value_map(current)
                        return {name: values[name] for name in limits}
                    finally:
                        carrier.coefficients.data.copy_(original)

                backtracked = backtracking_candidate(
                    parameter=carrier.coefficients,
                    gradient=routed.gradient,
                    step_size=step_size,
                    evaluate_constraints=evaluate,
                    limits=limits,
                    mode=(
                        "repair"
                        if routed.mode == "repair_only"
                        else "feasible"
                    ),
                    baseline_values=(
                        {name: baseline[name] for name in limits}
                        if routed.mode == "repair_only"
                        else None
                    ),
                    max_backtracks=int(self.config["phase_c"]["max_backtracks"]),
                )
                selected_mode = (
                    routed.mode if backtracked.accepted else "skip"
                )
                candidate = backtracked.candidate
                accepted = backtracked.accepted
                attempts = backtracked.attempts
            carrier.coefficients.data.copy_(candidate)
            with torch.no_grad():
                post_observation = self.engine.observe(
                    batch,
                    carrier,
                    route=route,
                )
            constraint_values_after = self._constraint_value_map(post_observation)
            shared_constraint_names = sorted(
                set(constraint_values_before).intersection(
                    constraint_values_after
                )
            )
            constraint_changes = {
                name: (
                    constraint_values_after[name]
                    - constraint_values_before[name]
                )
                for name in shared_constraint_names
            }
            mode_counts[selected_mode] = mode_counts.get(selected_mode, 0) + 1
            finite = finite and bool(torch.isfinite(candidate).all())
            step_rows.append(
                {
                    "step": step,
                    "mode": selected_mode,
                    "accepted": accepted,
                    "backtracking_attempts": attempts,
                    "constraint_rank": routed.rank,
                    "null_dimension": routed.null_dimension,
                    "attack_retention": routed.attack_retention,
                    "active_class_ids": active_class_ids,
                    "constraint_values_before": constraint_values_before,
                    "constraint_values_after": constraint_values_after,
                    "constraint_changes": constraint_changes,
                    "max_constraint_increase": max(
                        constraint_changes.values(),
                        default=float("nan"),
                    ),
                }
            )

        calibration = self._collect(
            self.split["calibration"],
            carrier,
            route=route,
            gradients=False,
        )
        heldout = self._collect(
            self.split["heldout"],
            carrier,
            route=route,
            gradients=False,
        )
        summary = summarize_observations(calibration, heldout, bank)
        total_constraints = 0
        violated_constraints = 0
        for observation in heldout:
            for item in observation.constraints.constraints:
                total_constraints += 2
                violated_constraints += int(
                    float(item.cls_margin.detach())
                    > float(self.config["phase_b"]["tau_cls"])
                )
                violated_constraints += int(
                    float(item.box_margin.detach())
                    > float(self.config["phase_b"]["tau_box"])
                )
        steps = max(int(self.config["phase_c"]["optimization_steps"]), 1)
        summary.update(
            {
                "routing_on": routing_on,
                "actual_violation_rate": violated_constraints
                / max(total_constraints, 1),
                "repair_skip_ratio": (
                    mode_counts.get("repair_only", 0)
                    + mode_counts.get("skip", 0)
                )
                / steps,
                "null_dimension_median": _median(null_dimensions),
                "finite": finite,
                "mode_counts": mode_counts,
                "steps": step_rows,
            }
        )
        return summary

    def run_phase_c(self, phase_b: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(phase_b["decision"]["pass"]):
            return {
                "phase": "C",
                "decision": {"pass": False, "status": "blocked_by_phase_b"},
                "arms": {},
            }
        best_arm = str(phase_b["decision"]["best_arm"])
        state_path = Path(str(phase_b["state_paths"][best_arm]))
        off = self._phase_c_arm(state_path=state_path, routing_on=False)
        on = self._phase_c_arm(state_path=state_path, routing_on=True)
        decision = evaluate_phase_c(off, on)
        result = {
            "phase": "C",
            "source_arm": best_arm,
            "arms": {"routing_off": off, "routing_on": on},
            "decision": decision,
        }
        _write_probe_json(self.artifact_root / "phase_c_metrics.json", result)
        return result

    def run(self, phase: str) -> dict[str, Any]:
        if phase not in {"A", "B", "C", "all"}:
            raise ValueError("phase must be A, B, C, or all.")
        status: dict[str, Any] = {
            "spec_id": self.config["spec"]["spec_id"],
            "exp_id": self.config["spec"]["exp_id"],
            "state": "running",
            "phase": phase,
            "split_hash": self.split["split_hash"],
            "source_manifest_hash": self.source_manifest_hash,
            "surrogate_checkpoint_sha256": self.surrogate_checkpoint_hash,
            "config_hash": canonical_hash(self.config),
            "background_basis_hashes": {
                carrier_id: basis.basis_hash
                for carrier_id, basis in self.basis_registry.items()
            },
        }
        _write_probe_json(self.artifact_root / "status.json", status)
        try:
            phase_a = self.run_phase_a()
            if phase == "A" or not phase_a["decision"]["pass"]:
                status["state"] = (
                    "completed" if phase_a["decision"]["pass"] else "stopped"
                )
                status["stop_reason"] = (
                    None
                    if phase_a["decision"]["pass"]
                    else "phase_a_failure_signal"
                )
                status["phase_a_pass"] = phase_a["decision"]["pass"]
                _write_probe_json(self.artifact_root / "status.json", status)
                return status

            phase_b = self.run_phase_b(phase_a)
            if phase == "B" or not phase_b["decision"]["pass"]:
                status["state"] = (
                    "completed" if phase_b["decision"]["pass"] else "stopped"
                )
                status["stop_reason"] = (
                    None
                    if phase_b["decision"]["pass"]
                    else "phase_b_failure_signal"
                )
                status["phase_a_pass"] = True
                status["phase_b_pass"] = phase_b["decision"]["pass"]
                _write_probe_json(self.artifact_root / "status.json", status)
                return status

            phase_c = self.run_phase_c(phase_b)
            status.update(
                {
                    "state": (
                        "completed"
                        if phase_c["decision"]["pass"]
                        else "stopped"
                    ),
                    "stop_reason": (
                        None
                        if phase_c["decision"]["pass"]
                        else "phase_c_failure_signal"
                    ),
                    "phase_a_pass": True,
                    "phase_b_pass": True,
                    "phase_c_pass": phase_c["decision"]["pass"],
                }
            )
            _write_probe_json(self.artifact_root / "status.json", status)
            return status
        except Exception as error:
            status.update(
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_probe_json(self.artifact_root / "status.json", status)
            raise
