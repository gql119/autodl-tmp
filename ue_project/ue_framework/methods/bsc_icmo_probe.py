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

from ..data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)
from ..support import (
    _bbox_to_pixels,
    build_forced_pseudo_instance_masks,
    build_support_mask,
)
from .alce_acgt import build_pag_gate
from .background_spectral_basis import (
    band_mask,
    build_background_spectral_basis,
    deterministic_two_crops,
    spectrum_energy_ratios,
)
from .bsc_rc_gr_probe import (
    ProbeEngine,
    _file_sha256,
    _resolve_path,
    _split_flat_gate,
    _write_probe_json,
    canonical_hash,
    load_background_sources,
    load_required_shared_split,
    make_batches,
)
from .cicr import ClassificationResiduals, classification_residuals
from .instance_canonical_carrier import (
    MatchedCanonicalCarrier,
    SharedGammaCalibration,
    affine_canonical_pattern,
    apply_canonical_pattern,
    build_synthetic_fourier_bases,
    calibrate_shared_gamma,
    canonicalize_explicit_bases,
    common_initial_coefficients,
    tensor_sha256,
    warp_canonical_patch,
)
from .instance_cicr import (
    InstanceClassificationResiduals,
    fit_instance_prototype_bank,
    instance_cicr,
    instance_classification_residuals,
    target_gt_indices_from_labels,
)
from .shadow_tal import TargetRouteResult, compute_target_route


ARM_DEFINITIONS = {
    "G-C0": ("C0", "global"),
    "G-C2LM": ("C2-LM", "global"),
    "I-C0": ("C0", "instance"),
    "I-C2LM": ("C2-LM", "instance"),
}
AFFINE_AUDITS = {
    "scale_0.90": {"scale": 0.90},
    "scale_1.10": {"scale": 1.10},
    "translate_x_-0.05": {"translate_x": -0.05},
    "translate_x_+0.05": {"translate_x": 0.05},
    "translate_y_-0.05": {"translate_y": -0.05},
    "translate_y_+0.05": {"translate_y": 0.05},
}


def validate_icmo_config(config: Mapping[str, Any]) -> None:
    required = {
        "spec",
        "dataset",
        "model",
        "background",
        "carrier",
        "split",
        "optimization",
        "bootstrap",
        "runtime",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing ICMO config sections: {missing}")
    if config["spec"].get("spec_id") != "TAUSB-BSC-ICMO-v1":
        raise ValueError("ICMO spec_id mismatch.")
    if config["spec"].get("exp_id") != "TAUSB-BSC-ICMO-MECH-S0":
        raise ValueError("ICMO exp_id mismatch.")
    if int(config["spec"].get("seed", -1)) != 0:
        raise ValueError("ICMO experiment seed must remain 0.")
    if int(config["dataset"].get("target_class_id", -1)) != 14:
        raise ValueError("ICMO target class must remain person=14.")
    if int(config["model"].get("num_classes", -1)) != 20:
        raise ValueError("ICMO detector must remain VOC20.")
    if int(config["model"].get("image_size", -1)) != 640:
        raise ValueError("ICMO image_size must remain 640.")

    carrier = config["carrier"]
    if abs(float(carrier.get("epsilon", -1)) - 16.0 / 255.0) > 1e-9:
        raise ValueError("ICMO epsilon must remain 16/255.")
    frozen_carrier_values = {
        "resolution": 640,
        "num_bases": 16,
        "basis_seed": 0,
        "gamma_seed": 2032,
        "gamma_directions": 256,
        "initial_seed": 2033,
    }
    for key, expected in frozen_carrier_values.items():
        if int(carrier.get(key, -1)) != expected:
            raise ValueError(f"ICMO carrier.{key} must remain {expected}.")
    if abs(float(carrier.get("coefficient_max_abs", -1)) - 0.25) > 1e-12:
        raise ValueError("ICMO coefficient_max_abs must remain 0.25.")
    if abs(float(carrier.get("target_rms_ratio", -1)) - 0.35) > 1e-12:
        raise ValueError("ICMO target_rms_ratio must remain 0.35.")
    for key in ("gamma_bisection_iterations", "gamma_chunk_size"):
        if int(carrier.get(key, 0)) <= 0:
            raise ValueError(f"ICMO carrier.{key} must be positive.")

    optimization = config["optimization"]
    frozen_optimization = {
        "warmup_steps": 4,
        "optimization_steps": 40,
        "batch_size": 4,
    }
    for key, expected in frozen_optimization.items():
        if int(optimization.get(key, -1)) != expected:
            raise ValueError(f"ICMO optimization.{key} must remain {expected}.")
    if abs(float(optimization.get("learning_rate", -1)) - 0.01) > 1e-12:
        raise ValueError("ICMO learning_rate must remain 0.01.")
    if str(optimization.get("target_route")) != "easy_cls":
        raise ValueError("ICMO target route must remain easy_cls.")
    for key in ("lambda_cicr", "lambda_route", "lambda_rms"):
        if abs(float(optimization.get(key, -1)) - 1.0) > 1e-12:
            raise ValueError(f"ICMO {key} must remain 1.0.")
    if not 0 <= float(optimization.get("prototype_momentum", -1)) < 1:
        raise ValueError("ICMO prototype_momentum must lie in [0,1).")
    for key in ("box_teacher_weight", "align_alpha", "align_beta"):
        if float(optimization.get(key, 0)) <= 0:
            raise ValueError(f"ICMO optimization.{key} must be positive.")
    if int(optimization.get("assignment_topk", 0)) <= 0:
        raise ValueError("ICMO assignment_topk must be positive.")
    if (
        len(optimization.get("pag_layer_ratios", ())) != 3
        or len(optimization.get("pag_min_pos", ())) != 3
    ):
        raise ValueError("ICMO PAG settings must define P3/P4/P5.")

    bootstrap = config["bootstrap"]
    if int(bootstrap.get("seed", -1)) != 2040:
        raise ValueError("ICMO bootstrap seed must remain 2040.")
    if int(bootstrap.get("iterations", -1)) != 10000:
        raise ValueError("ICMO bootstrap iterations must remain 10000.")
    if (
        str(config["split"].get("required_protocol_prefix", ""))
        != "TAUSB-ALCE-CTX-AUDIT-v1"
    ):
        raise ValueError("ICMO must reuse the frozen ALCE shared split.")
    for section, key in (
        ("dataset", "root"),
        ("dataset", "train_images"),
        ("dataset", "train_labels"),
        ("model", "surrogate_checkpoint"),
        ("background", "source_manifest"),
        ("background", "source_local_map"),
        ("carrier", "synthetic_global_params_path"),
        ("split", "manifest"),
        ("runtime", "artifact_root"),
        ("runtime", "device"),
    ):
        if not str(config[section].get(key, "")).strip():
            raise ValueError(f"ICMO config requires {section}.{key}.")
    for key in (
        "source_manifest_sha256",
        "shared_split_sha256",
        "label_sha256",
        "surrogate_checkpoint_sha256",
        "synthetic_global_params_sha256",
        "c2lm_basis_sha256",
    ):
        value = str(config["spec"].get(key, ""))
        if len(value) != 64:
            raise ValueError(f"ICMO spec.{key} must be a SHA256.")


def _scale_group_for_annotation(annotation: Mapping[str, Any]) -> str:
    area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
    if area < 0.02:
        return "small"
    if area < 0.15:
        return "medium"
    return "large"


@dataclass
class ICMOBatch:
    images: torch.Tensor
    yolo_batch: dict[str, Any]
    boxes_by_image: list[list[tuple[int, int, int, int]]]
    supports_by_image: list[torch.Tensor]
    instance_scale_groups_by_image: list[list[str]]
    image_ids: list[str]
    person_cooccur: list[bool]


@dataclass
class ICMOObservation:
    target_residuals: InstanceClassificationResiduals
    non_target_residuals: ClassificationResiduals
    box_residuals: InstanceClassificationResiduals
    route: TargetRouteResult
    image_ids: list[str]
    person_cooccur: list[bool]
    instance_scale_groups_by_image: list[list[str]]
    active_rms_by_image: list[float]
    linf_by_image: list[float]
    pattern_saturation_by_image: list[float]
    target_logit_energy_by_image: list[float]
    non_target_logit_energy_by_image: list[float]
    outside_support_max: float


def load_icmo_batch(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    image_size: int,
    target_class_id: int,
    device: torch.device,
) -> ICMOBatch:
    images = []
    classes: list[list[float]] = []
    yolo_boxes: list[list[float]] = []
    batch_indices: list[int] = []
    boxes_by_image = []
    supports_by_image = []
    scale_groups_by_image = []
    image_ids = []
    person_cooccur = []

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
        target_annotations = [
            item
            for item in annotations
            if int(item.get("cls", -1)) == int(target_class_id)
        ]
        target_boxes = [
            _bbox_to_pixels(item["bbox"], image_size, image_size)
            for item in target_annotations
        ]
        if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in target_boxes):
            raise ValueError(f"Degenerate target box in image {image_path.stem}.")
        target_masks = build_forced_pseudo_instance_masks(
            image.shape,
            annotations,
            target_class_id,
        )
        if len(target_masks) != len(target_boxes):
            raise RuntimeError("Target boxes and forced pseudo supports diverged.")
        legacy_forced = build_support_mask(
            image_shape=image.shape,
            annotations=[
                {
                    key: value
                    for key, value in item.items()
                    if key != "polygon"
                }
                for item in annotations
            ],
            target_class_id=target_class_id,
            support_type="mask",
            ring_width=4,
            mask_path=None,
        )
        forced_union = np.maximum.reduce(target_masks)
        if not np.array_equal(forced_union, legacy_forced):
            raise RuntimeError("Forced pseudo support area changed from legacy.")

        images.append(torch.from_numpy(image).permute(2, 0, 1).float())
        boxes_by_image.append(target_boxes)
        supports_by_image.append(
            torch.from_numpy(np.stack(target_masks))
            .unsqueeze(1)
            .float()
            .to(device)
        )
        scale_groups_by_image.append(
            [_scale_group_for_annotation(item) for item in target_annotations]
        )
        image_ids.append(image_path.stem)
        person_cooccur.append(
            any(
                int(item.get("cls", -1)) != int(target_class_id)
                for item in annotations
            )
        )
        for item in annotations:
            classes.append([float(item["cls"])])
            yolo_boxes.append([float(value) for value in item["bbox"]])
            batch_indices.append(batch_index)

    images_tensor = torch.stack(images).to(device)
    yolo_batch = {
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long, device=device),
        "cls": torch.tensor(classes, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(yolo_boxes, dtype=torch.float32, device=device),
        "batch_size": len(image_paths),
        "img": images_tensor,
    }
    return ICMOBatch(
        images=images_tensor,
        yolo_batch=yolo_batch,
        boxes_by_image=boxes_by_image,
        supports_by_image=supports_by_image,
        instance_scale_groups_by_image=scale_groups_by_image,
        image_ids=image_ids,
        person_cooccur=person_cooccur,
    )


class ICMOProbeEngine(ProbeEngine):
    def __init__(
        self,
        config: Mapping[str, Any],
        device: torch.device,
        *,
        checkpoint_path: Path,
    ) -> None:
        engine_config = dict(config)
        engine_config["phase_b"] = dict(config["optimization"])
        super().__init__(
            engine_config,
            device,
            checkpoint_path=checkpoint_path,
        )

    def observe(
        self,
        batch: ICMOBatch,
        carrier: MatchedCanonicalCarrier,
        *,
        render_mode: str,
        affine: Mapping[str, float] | None = None,
    ) -> ICMOObservation:
        canonical = carrier()
        if affine:
            canonical = affine_canonical_pattern(canonical, **affine)
        adv_images, perturbation, rendered = apply_canonical_pattern(
            batch.images,
            canonical,
            boxes_by_image=batch.boxes_by_image,
            supports_by_image=batch.supports_by_image,
            mode=render_mode,
            epsilon=carrier.epsilon,
        )
        with torch.no_grad():
            with self.capture.record("clean"):
                clean_output = self.model(batch.images)
            clean_features = self.capture.take("clean")
            clean_predictions = (
                clean_output[0]
                if isinstance(clean_output, (tuple, list))
                else clean_output
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
                image_shape=batch.images.shape[-2:],
                assignment_topk=int(self.config["optimization"]["assignment_topk"]),
            )

        with self.capture.record("adv"):
            adv_output = self.model(adv_images)
        adv_features = self.capture.take("adv")
        adv_predictions = (
            adv_output[0] if isinstance(adv_output, (tuple, list)) else adv_output
        )
        adv_cache = self.hijacked.cache_assign_inputs_only(
            adv_predictions,
            batch.yolo_batch,
            image_shape=batch.images.shape[-2:],
            assignment_topk=int(self.config["optimization"]["assignment_topk"]),
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
            top_ratio=self.config["optimization"]["pag_layer_ratios"],
            min_keep=self.config["optimization"]["pag_min_pos"],
            layer_sizes=layer_sizes,
        )
        target_gt_indices = target_gt_indices_from_labels(
            clean_cache["gt_labels"],
            clean_cache["mask_gt"],
            target_class_id=self.target_class_id,
        )
        expected_counts = tuple(
            len(values) for values in batch.instance_scale_groups_by_image
        )
        if tuple(len(values) for values in target_gt_indices) != expected_counts:
            raise RuntimeError("Renderer instances and clean GT indices diverged.")
        target_residuals = instance_classification_residuals(
            clean_features.classification,
            adv_features.classification,
            pag_gate,
            real_assign["target_gt_idx"],
            target_gt_indices,
        )
        box_residuals = instance_classification_residuals(
            clean_features.box,
            adv_features.box,
            pag_gate,
            real_assign["target_gt_idx"],
            target_gt_indices,
        )
        non_target_gates = _split_flat_gate(
            real_foreground & (labels != self.target_class_id),
            adv_features.classification,
        )
        non_target_residuals = classification_residuals(
            clean_features.classification,
            adv_features.classification,
            non_target_gates,
        )
        route = compute_target_route(
            route="easy_cls",
            adv_class_logits=adv_cache["pred_scores_logits"],
            adv_boxes=adv_cache["pred_bboxes"],
            clean_boxes=clean_cache["pred_bboxes"],
            target_gate=pag_gate,
            target_class_id=self.target_class_id,
            num_classes=self.num_classes,
            box_teacher_weight=float(
                self.config["optimization"]["box_teacher_weight"]
            ),
            shadow_tal=self.shadow_tal,
            gt_labels=clean_cache["gt_labels"],
            gt_bboxes=clean_cache["gt_bboxes"],
            mask_gt=clean_cache["mask_gt"],
        )
        score_delta = (
            adv_cache["pred_scores_logits"]
            - clean_cache["pred_scores_logits"].detach()
        )
        non_target_class_mask = torch.ones(
            self.num_classes,
            device=score_delta.device,
            dtype=torch.bool,
        )
        non_target_class_mask[self.target_class_id] = False
        target_logit_energy = []
        non_target_logit_energy = []
        for image_index in range(score_delta.shape[0]):
            selected = pag_gate[image_index].bool()
            if bool(selected.any()):
                target_values = score_delta[
                    image_index,
                    selected,
                    self.target_class_id,
                ]
                non_target_values = score_delta[
                    image_index,
                    selected,
                ][:, non_target_class_mask]
                target_logit_energy.append(
                    float(target_values.square().mean().sqrt().detach())
                )
                non_target_logit_energy.append(
                    float(non_target_values.square().mean().sqrt().detach())
                )
            else:
                target_logit_energy.append(float("nan"))
                non_target_logit_energy.append(float("nan"))

        active_rms = []
        linf = []
        pattern_saturation = []
        outside_support_max = 0.0
        for image_index, item in enumerate(rendered):
            active_mask = item.union_support.expand_as(perturbation[image_index]).bool()
            active = perturbation[image_index][active_mask]
            active_rms.append(
                float(active.square().mean().sqrt().detach())
                if active.numel()
                else 0.0
            )
            linf.append(float(perturbation[image_index].abs().amax().detach()))
            pattern_saturation.append(
                float(
                    (
                        active.abs()
                        >= 0.95 * float(carrier.epsilon)
                    )
                    .float()
                    .mean()
                    .detach()
                )
                if active.numel()
                else 0.0
            )
            outside = perturbation[image_index] * (1.0 - item.union_support)
            outside_support_max = max(
                outside_support_max,
                float(outside.abs().amax().detach()),
            )
        return ICMOObservation(
            target_residuals=target_residuals,
            non_target_residuals=non_target_residuals,
            box_residuals=box_residuals,
            route=route,
            image_ids=batch.image_ids,
            person_cooccur=batch.person_cooccur,
            instance_scale_groups_by_image=batch.instance_scale_groups_by_image,
            active_rms_by_image=active_rms,
            linf_by_image=linf,
            pattern_saturation_by_image=pattern_saturation,
            target_logit_energy_by_image=target_logit_energy,
            non_target_logit_energy_by_image=non_target_logit_energy,
            outside_support_max=outside_support_max,
        )


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _quantile(values: Iterable[float], q: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, q)) if finite else float("nan")


def _concat_instance_residuals(
    observations: Sequence[ICMOObservation],
    attribute: str,
) -> InstanceClassificationResiduals:
    selected = [getattr(item, attribute) for item in observations]
    if not selected:
        raise ValueError("Cannot concatenate an empty observation sequence.")
    num_scales = len(selected[0].vectors)
    vectors = tuple(
        torch.cat([item.vectors[scale] for item in selected], dim=0)
        for scale in range(num_scales)
    )
    gate_valid = tuple(
        torch.cat([item.gate_valid[scale] for item in selected], dim=0)
        for scale in range(num_scales)
    )
    gate_mass = tuple(
        torch.cat([item.gate_mass[scale] for item in selected], dim=0)
        for scale in range(num_scales)
    )
    total = vectors[0].shape[0]
    return InstanceClassificationResiduals(
        vectors=vectors,
        gate_valid=gate_valid,
        gate_mass=gate_mass,
        image_indices=torch.arange(total, device=vectors[0].device),
        gt_indices=torch.zeros(total, device=vectors[0].device, dtype=torch.long),
    )


def _mean_instance_energy(
    observations: Sequence[ICMOObservation],
    attribute: str,
) -> float:
    values = []
    for observation in observations:
        residuals = getattr(observation, attribute)
        for vectors, valid in zip(residuals.vectors, residuals.gate_valid):
            if bool(valid.any()):
                values.extend(
                    float(value)
                    for value in vectors.detach()[valid].norm(dim=1).cpu()
                )
    return float(np.mean(values)) if values else float("nan")


def _mean_non_target_energy(
    observations: Sequence[ICMOObservation],
) -> float:
    values = []
    for observation in observations:
        for vectors, valid in zip(
            observation.non_target_residuals.vectors,
            observation.non_target_residuals.gate_valid,
        ):
            if bool(valid.any()):
                values.extend(
                    float(value)
                    for value in vectors.detach()[valid].norm(dim=1).cpu()
                )
    return float(np.mean(values)) if values else 0.0


def fit_icmo_bank(
    observations: Sequence[ICMOObservation],
    *,
    momentum: float,
) -> Any:
    return fit_instance_prototype_bank(
        _concat_instance_residuals(observations, "target_residuals"),
        momentum=momentum,
    )


def summarize_icmo_observations(
    observations: Sequence[ICMOObservation],
    bank: Any,
) -> dict[str, Any]:
    image_records = []
    instance_records = []
    total_instances = 0
    valid_instances = 0
    missing_assignments = 0
    low_energy = 0
    zero_norm = 0
    active_rms = []
    linf = []
    pattern_saturation = []
    target_logit_energy = []
    non_target_logit_energy = []
    outside_support_max = 0.0
    route_losses = []

    for observation in observations:
        result = instance_cicr(observation.target_residuals, bank)
        total_instances += result.total_instance_count
        valid_instances += result.valid_instance_count
        missing_assignments += int(
            round(result.missing_assignment_ratio * result.total_instance_count)
        )
        low_energy += int(
            round(result.low_energy_ratio * result.total_instance_count)
        )
        zero_norm += int(
            round(result.zero_norm_ratio * result.total_instance_count)
        )
        route_losses.append(float(observation.route.loss.detach()))
        active_rms.extend(observation.active_rms_by_image)
        linf.extend(observation.linf_by_image)
        pattern_saturation.extend(observation.pattern_saturation_by_image)
        target_logit_energy.extend(observation.target_logit_energy_by_image)
        non_target_logit_energy.extend(
            observation.non_target_logit_energy_by_image
        )
        outside_support_max = max(
            outside_support_max,
            observation.outside_support_max,
        )

        per_image_values = [[] for _ in observation.image_ids]
        scale_offsets = [0 for _ in observation.image_ids]
        for instance_index in range(result.total_instance_count):
            image_index = int(
                observation.target_residuals.image_indices[instance_index]
            )
            value = float(result.per_instance_cosine[instance_index])
            local_index = scale_offsets[image_index]
            scale_offsets[image_index] += 1
            scale_group = observation.instance_scale_groups_by_image[
                image_index
            ][local_index]
            instance_records.append(
                {
                    "image_id": observation.image_ids[image_index],
                    "gt_index": int(
                        observation.target_residuals.gt_indices[instance_index]
                    ),
                    "person_cooccur": observation.person_cooccur[image_index],
                    "person_scale_group": scale_group,
                    "cicr": value,
                    "valid_scale_count": int(
                        result.per_instance_valid_scale_count[instance_index]
                    ),
                }
            )
            if math.isfinite(value):
                per_image_values[image_index].append(value)

        for image_index, image_id in enumerate(observation.image_ids):
            values = per_image_values[image_index]
            image_records.append(
                {
                    "image_id": image_id,
                    "person_cooccur": observation.person_cooccur[image_index],
                    "cicr": (
                        float(np.median(values))
                        if values
                        else float("nan")
                    ),
                    "valid_instance_count": len(values),
                    "total_instance_count": len(
                        observation.instance_scale_groups_by_image[image_index]
                    ),
                }
            )

    image_values = [float(record["cicr"]) for record in image_records]
    valid_image_count = sum(math.isfinite(value) for value in image_values)
    group_cicr: dict[str, list[float]] = {}
    group_target_logit: dict[str, list[float]] = {}
    group_non_target_logit: dict[str, list[float]] = {}
    for record in image_records:
        value = float(record["cicr"])
        if math.isfinite(value):
            key = (
                "person_cooccur"
                if bool(record["person_cooccur"])
                else "person_only"
            )
            group_cicr.setdefault(key, []).append(value)
    for observation in observations:
        for image_index, cooccur in enumerate(observation.person_cooccur):
            key = "person_cooccur" if cooccur else "person_only"
            target_value = observation.target_logit_energy_by_image[image_index]
            non_target_value = observation.non_target_logit_energy_by_image[
                image_index
            ]
            if math.isfinite(target_value):
                group_target_logit.setdefault(key, []).append(target_value)
            if math.isfinite(non_target_value):
                group_non_target_logit.setdefault(key, []).append(
                    non_target_value
                )
    for record in instance_records:
        value = float(record["cicr"])
        if math.isfinite(value):
            group_cicr.setdefault(
                str(record["person_scale_group"]),
                [],
            ).append(value)

    target_energy = _mean_instance_energy(observations, "target_residuals")
    non_target_feature_energy = _mean_non_target_energy(observations)
    box_energy = _mean_instance_energy(observations, "box_residuals")
    finite_target_logit = [
        value for value in target_logit_energy if math.isfinite(value)
    ]
    finite_non_target_logit = [
        value for value in non_target_logit_energy if math.isfinite(value)
    ]
    target_logit_mean = (
        float(np.mean(finite_target_logit))
        if finite_target_logit
        else float("nan")
    )
    non_target_logit_mean = (
        float(np.mean(finite_non_target_logit))
        if finite_non_target_logit
        else float("nan")
    )
    return {
        "heldout_cicr_median": _median(image_values),
        "heldout_cicr_q25": _quantile(image_values, 0.25),
        "valid_image_count": valid_image_count,
        "total_image_count": len(image_records),
        "valid_image_coverage": valid_image_count / max(len(image_records), 1),
        "valid_instance_count": valid_instances,
        "total_instance_count": total_instances,
        "valid_instance_coverage": valid_instances / max(total_instances, 1),
        "missing_assignment_ratio": missing_assignments
        / max(total_instances, 1),
        "low_energy_ratio": low_energy / max(total_instances, 1),
        "zero_norm_ratio": zero_norm / max(total_instances, 1),
        "target_residual_energy": target_energy,
        "non_target_feature_residual_energy": non_target_feature_energy,
        "target_logit_residual_energy": target_logit_mean,
        "non_target_logit_residual_energy": non_target_logit_mean,
        "non_target_target_energy_ratio": non_target_logit_mean
        / max(target_logit_mean, 1e-12),
        "preservation_energy_definition": (
            "RMS non-target-class logit residual / RMS person-logit "
            "residual on clean TAL/PAG person positives"
        ),
        "group_non_target_target_energy_ratio": {
            key: float(np.mean(group_non_target_logit[key]))
            / max(float(np.mean(target_values)), 1e-12)
            for key, target_values in group_target_logit.items()
            if target_values and group_non_target_logit.get(key)
        },
        "box_residual_energy": box_energy,
        "route_loss": _median(route_losses),
        "group_cicr_median": {
            key: _median(values) for key, values in group_cicr.items()
        },
        "active_pixel_rms": _median(active_rms),
        "active_pixel_linf": max(linf, default=0.0),
        "pattern_saturation_ratio": _median(pattern_saturation),
        "outside_support_max": outside_support_max,
        "per_image": image_records,
        "per_instance": instance_records,
    }


def stratified_paired_bootstrap(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 2040,
    iterations: int = 10000,
) -> dict[str, Any]:
    left = {
        str(record["image_id"]): record
        for record in left_records
        if math.isfinite(float(record["cicr"]))
    }
    right = {
        str(record["image_id"]): record
        for record in right_records
        if math.isfinite(float(record["cicr"]))
    }
    paired = []
    for image_id in sorted(set(left) & set(right)):
        left_record = left[image_id]
        right_record = right[image_id]
        if bool(left_record["person_cooccur"]) != bool(
            right_record["person_cooccur"]
        ):
            raise ValueError("Paired records disagree on cooccurrence stratum.")
        paired.append(
            (
                image_id,
                "person_cooccur"
                if bool(left_record["person_cooccur"])
                else "person_only",
                float(left_record["cicr"]) - float(right_record["cicr"]),
            )
        )
    strata = {
        key: np.asarray(
            [value for _, stratum, value in paired if stratum == key],
            dtype=np.float64,
        )
        for key in ("person_only", "person_cooccur")
    }
    rng = np.random.default_rng(int(seed))
    bootstrapped = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sampled = []
        for values in strata.values():
            if values.size:
                sampled.append(
                    values[rng.integers(0, values.size, size=values.size)]
                )
        bootstrapped[index] = (
            float(np.median(np.concatenate(sampled)))
            if sampled
            else float("nan")
        )
    finite_bootstrap = bootstrapped[np.isfinite(bootstrapped)]
    return {
        "paired_image_count": len(paired),
        "stratum_counts": {
            key: int(values.size) for key, values in strata.items()
        },
        "paired_median_delta": _median(value for _, _, value in paired),
        "ci95": (
            [
                float(np.quantile(finite_bootstrap, 0.025)),
                float(np.quantile(finite_bootstrap, 0.975)),
            ]
            if finite_bootstrap.size
            else [float("nan"), float("nan")]
        ),
        "seed": int(seed),
        "iterations": int(iterations),
    }


def evaluate_icmo_result(
    arms: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    *,
    initial_rms_ratio: float,
    active_rms_ratio: float,
    mechanical: Mapping[str, Any],
    hashes_complete: bool,
) -> dict[str, Any]:
    candidate = arms["I-C2LM"]
    synthetic = arms["I-C0"]
    coordinate = contrasts["I-C2LM_vs_G-C2LM"]
    natural = contrasts["I-C2LM_vs_I-C0"]
    group = candidate["group_cicr_median"]
    scale_values = [
        float(group[key])
        for key in ("small", "medium", "large")
        if key in group and math.isfinite(float(group[key]))
    ]
    affine_values = candidate["affine_audit"]
    identity = float(candidate["heldout_cicr_median"])
    checks = {
        "candidate_level": identity >= 0.60
        and float(candidate["heldout_cicr_q25"]) >= 0.20,
        "coordinate_contrast": float(coordinate["arm_median_delta"]) >= 0.08,
        "natural_basis_contrast": float(natural["arm_median_delta"]) >= 0.05,
        "bootstrap": all(
            int(item["paired_image_count"]) >= 80
            and float(item["ci95"][0]) > 0
            for item in (coordinate, natural)
        ),
        "coverage_energy": float(candidate["valid_image_coverage"]) >= 0.90
        and max(
            float(candidate["low_energy_ratio"]),
            float(candidate["zero_norm_ratio"]),
        )
        <= 0.20,
        "route_effect": float(candidate["route_effect"]) >= 0.10,
        "non_target_preservation": float(
            candidate["non_target_target_energy_ratio"]
        )
        <= 0.40
        and float(candidate["non_target_target_energy_ratio"])
        <= 1.05 * float(synthetic["non_target_target_energy_ratio"]),
        "box_preservation": float(candidate["box_residual_energy"])
        <= 1.05 * float(synthetic["box_residual_energy"]),
        "cooccur_gap": (
            "person_only" in group
            and "person_cooccur" in group
            and abs(
                float(group["person_only"])
                - float(group["person_cooccur"])
            )
            <= 0.15
        ),
        "scale_gap": len(scale_values) == 3
        and max(scale_values) - min(scale_values) <= 0.20,
        "affine_retention": all(
            math.isfinite(float(value))
            and float(value) >= 0.90 * identity
            for value in affine_values.values()
        ),
        "frequency_semantics": float(
            candidate["canonical_spectrum_energy"]["low"]
            + candidate["canonical_spectrum_energy"]["mid"]
        )
        >= 0.70
        and float(candidate["source_max_abs_correlation"]) <= 0.30,
        "matched_active_amplitude": active_rms_ratio <= 1.05
        and all(
            float(arm["active_pixel_linf"]) <= 16.0 / 255.0 + 1e-8
            for arm in arms.values()
        ),
        "basis_usage": all(
            float(arm["coefficient_saturation_ratio"]) < 0.25
            and float(arm["active_basis_fraction"]) >= 0.25
            and float(arm["top1_basis_energy_share"]) < 0.80
            for arm in arms.values()
        ),
        "finite_hashes": hashes_complete
        and all(bool(arm["finite"]) for arm in arms.values()),
    }
    failure_signals = {
        "renderer_ncc": float(mechanical["ncc_median"]) < 0.98
        or float(mechanical["ncc_q25"]) < 0.95
        or not bool(mechanical["render_paths_distinct"]),
        "amplitude_mismatch": not 0.98 <= initial_rms_ratio <= 1.02
        or active_rms_ratio > 1.05,
        "coefficient_overfit": float(candidate["calibration_cicr_gain"]) >= 0.10
        and (
            float(candidate["heldout_cicr_gain"]) < 0.02
            or float(candidate["calibration_heldout_gap"]) > 0.15
        ),
        "coordinate_only": not checks["natural_basis_contrast"]
        and all(
            float(contrasts[key]["arm_median_delta"]) >= 0.08
            and float(contrasts[key]["ci95"][0]) > 0
            for key in ("I-C0_vs_G-C0", "I-C2LM_vs_G-C2LM")
        ),
        "source_semantic_dependence": float(
            candidate["source_max_abs_correlation"]
        )
        > 0.30,
        "cooccur_collateral_leakage": (
            "person_only"
            in candidate["group_non_target_target_energy_ratio"]
            and "person_cooccur"
            in candidate["group_non_target_target_energy_ratio"]
            and float(
                candidate["group_non_target_target_energy_ratio"][
                    "person_cooccur"
                ]
            )
            > 1.5
            * float(
                candidate["group_non_target_target_energy_ratio"][
                    "person_only"
                ]
            )
        ),
        "basis_or_amplitude_shortcut": not checks["basis_usage"],
        "route_only": float(candidate["route_effect"]) >= 0.10
        and identity < 0.20,
        "insufficient_assignment": float(
            candidate["valid_instance_coverage"]
        )
        < 0.70,
        "non_finite_rank_hash_split": not checks["finite_hashes"],
    }
    return {
        "pass": all(checks.values()) and not any(failure_signals.values()),
        "checks": checks,
        "failure_signals": failure_signals,
        "status": (
            "mechanism_pass"
            if all(checks.values()) and not any(failure_signals.values())
            else "mechanism_fail"
        ),
    }


def _pairwise_cosines(values: Sequence[torch.Tensor]) -> list[float]:
    result = []
    for index, first in enumerate(values):
        first_flat = first.flatten() - first.mean()
        for second in values[index + 1 :]:
            second_flat = second.flatten() - second.mean()
            result.append(
                float(
                    F.cosine_similarity(
                        first_flat.unsqueeze(0),
                        second_flat.unsqueeze(0),
                        dim=1,
                        eps=1e-12,
                    ).item()
                )
            )
    return result


def renderer_mechanical_audit(pattern: torch.Tensor) -> dict[str, Any]:
    resolution = pattern.shape[-1]
    boxes = [
        (
            int(0.05 * resolution),
            int(0.08 * resolution),
            int(0.45 * resolution),
            int(0.68 * resolution),
        ),
        (
            int(0.55 * resolution),
            int(0.10 * resolution),
            int(0.85 * resolution),
            int(0.50 * resolution),
        ),
        (
            int(0.18 * resolution),
            int(0.62 * resolution),
            int(0.78 * resolution),
            int(0.92 * resolution),
        ),
    ]
    reconstructed = [
        F.interpolate(
            warp_canonical_patch(pattern, box).unsqueeze(0),
            size=pattern.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        for box in boxes
    ]
    global_pattern = F.interpolate(
        pattern.unsqueeze(0),
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )[0]
    global_crops = [
        F.interpolate(
            global_pattern[:, box[1] : box[3], box[0] : box[2]].unsqueeze(0),
            size=pattern.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        for box in boxes
    ]
    instance_values = _pairwise_cosines(reconstructed)
    global_values = _pairwise_cosines(global_crops)
    return {
        "ncc_median": _median(instance_values),
        "ncc_q25": _quantile(instance_values, 0.25),
        "global_crop_ncc_median": _median(global_values),
        "render_paths_distinct": _median(global_values)
        < _median(instance_values) - 0.05,
        "forced_support_semantics": "forced_pseudo_fallback_not_instance_mask",
    }


class ICMOProbeWorkflow:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        source_manifest: str | None = None,
        source_local_map: str | None = None,
        device_override: str | None = None,
    ) -> None:
        validate_icmo_config(config)
        self.config = dict(config)
        self.config_path = config_path.resolve()
        self.project_root = self.config_path.parents[2]
        self.seed = int(config["spec"]["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        runtime_device = str(
            device_override
            if device_override is not None
            else config["runtime"]["device"]
        )
        if runtime_device.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA ICMO probe but CUDA is unavailable.")
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
                f"Fresh ICMO probe refuses existing artifact root: {self.artifact_root}"
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
        self.split = load_required_shared_split(
            _resolve_path(self.project_root, str(config["split"]["manifest"])),
            target_images=self.target_images,
            required_protocol_prefix=str(
                config["split"]["required_protocol_prefix"]
            ),
        )
        if (
            self.split["split_hash"]
            != str(config["spec"]["shared_split_sha256"])
        ):
            raise ValueError("Frozen shared split hash mismatch.")
        if (
            str(self.split["shared_split_manifest"].get("label_hash"))
            != str(config["spec"]["label_sha256"])
        ):
            raise ValueError("Frozen shared split label hash mismatch.")

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
        if (
            self.source_manifest_hash
            != str(config["spec"]["source_manifest_sha256"])
        ):
            raise ValueError("Frozen source manifest hash mismatch.")

        carrier_cfg = config["carrier"]
        resolution = int(carrier_cfg["resolution"])
        num_bases = int(carrier_cfg["num_bases"])
        natural = build_background_spectral_basis(
            self.source_images,
            resolution=resolution,
            num_bases=num_bases,
            bands=((2.0, 8.0), (8.0, 24.0)),
            phase_mode="scrambled",
            seed=int(carrier_cfg["basis_seed"]),
        )
        if natural.basis_hash != str(config["spec"]["c2lm_basis_sha256"]):
            raise ValueError("Frozen C2-LM source basis hash mismatch.")
        self.natural_source_basis = natural

        params_path = _resolve_path(
            self.project_root,
            str(carrier_cfg["synthetic_global_params_path"]),
        )
        if not params_path.is_file():
            raise FileNotFoundError(f"C0 coordinate pack is missing: {params_path}")
        self.synthetic_params_path = params_path
        self.synthetic_params_hash = _file_sha256(params_path)
        if self.synthetic_params_hash != str(
            config["spec"]["synthetic_global_params_sha256"]
        ):
            raise ValueError("Frozen C0 coordinate pack hash mismatch.")
        params = torch.load(params_path, map_location="cpu")
        coords = [tuple(map(int, value)) for value in params.get("coords", [])]
        if len(coords) != num_bases:
            raise ValueError("Frozen C0 coordinate pack must contain 16 coords.")

        self.bases = {
            "C0": build_synthetic_fourier_bases(resolution, coords),
            "C2-LM": canonicalize_explicit_bases(natural.bases),
        }
        self.basis_hashes = {
            family: tensor_sha256(bases)
            for family, bases in self.bases.items()
        }
        self.basis_ranks = {
            family: int(torch.linalg.matrix_rank(bases.flatten(1)).item())
            for family, bases in self.bases.items()
        }
        self.gamma_calibration: SharedGammaCalibration = calibrate_shared_gamma(
            self.bases,
            epsilon=float(carrier_cfg["epsilon"]),
            device=self.device,
            seed=int(carrier_cfg["gamma_seed"]),
            num_directions=int(carrier_cfg["gamma_directions"]),
            coefficient_max_abs=float(carrier_cfg["coefficient_max_abs"]),
            target_rms_ratio=float(carrier_cfg["target_rms_ratio"]),
            iterations=int(carrier_cfg["gamma_bisection_iterations"]),
            chunk_size=int(carrier_cfg["gamma_chunk_size"]),
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.initial_coefficients = common_initial_coefficients(
            num_bases,
            seed=int(carrier_cfg["initial_seed"]),
            max_abs=float(carrier_cfg["coefficient_max_abs"]),
        )
        initial_patterns = {
            family: self._carrier(family)().detach()
            for family in ("C0", "C2-LM")
        }
        initial_rms = {
            family: float(pattern.square().mean().sqrt())
            for family, pattern in initial_patterns.items()
        }
        self.initial_rms_ratio = max(initial_rms.values()) / max(
            min(initial_rms.values()),
            1e-12,
        )
        self.mechanical = renderer_mechanical_audit(
            initial_patterns["C2-LM"]
        )
        self.mechanical.update(
            {
                "initial_family_rms": initial_rms,
                "initial_rms_ratio": self.initial_rms_ratio,
                "basis_ranks": self.basis_ranks,
            }
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
        if self.surrogate_checkpoint_hash != str(
            config["spec"]["surrogate_checkpoint_sha256"]
        ):
            raise ValueError("Frozen surrogate checkpoint hash mismatch.")
        self.engine = ICMOProbeEngine(
            config,
            self.device,
            checkpoint_path=self.surrogate_checkpoint_path,
        )
        try:
            self.artifact_root.mkdir(parents=True)
            (self.artifact_root / "arms").mkdir()
            (self.artifact_root / "bases").mkdir()
            self._write_initial_artifacts()
        except Exception:
            self.engine.close()
            raise

    def close(self) -> None:
        self.engine.close()

    def _carrier(self, family: str) -> MatchedCanonicalCarrier:
        carrier_cfg = self.config["carrier"]
        return MatchedCanonicalCarrier(
            self.bases[family],
            epsilon=float(carrier_cfg["epsilon"]),
            gamma=self.gamma_calibration.gamma,
            initial_coefficients=self.initial_coefficients,
        ).to(self.device)

    def _write_initial_artifacts(self) -> None:
        for family, bases in self.bases.items():
            torch.save(
                {
                    "family": family,
                    "bases": bases,
                    "matched_basis_hash": self.basis_hashes[family],
                    "source_basis_hash": (
                        self.natural_source_basis.basis_hash
                        if family == "C2-LM"
                        else None
                    ),
                },
                self.artifact_root / "bases" / f"{family}.pt",
            )
        _write_probe_json(
            self.artifact_root / "config_resolved.json",
            self.config,
        )
        _write_probe_json(
            self.artifact_root / "split_manifest.json",
            self.split["shared_split_manifest"],
        )
        _write_probe_json(
            self.artifact_root / "source_manifest.json",
            self.source_manifest,
        )
        _write_probe_json(
            self.artifact_root / "mechanical_audit.json",
            self.mechanical,
        )
        _write_probe_json(
            self.artifact_root / "gamma_calibration.json",
            {
                "gamma": self.gamma_calibration.gamma,
                "target_rms": self.gamma_calibration.target_rms,
                "pooled_median_rms": (
                    self.gamma_calibration.pooled_median_rms
                ),
                "family_median_rms": (
                    self.gamma_calibration.family_median_rms
                ),
                "family_rms_ratio": (
                    self.gamma_calibration.family_rms_ratio
                ),
                "direction_hash": self.gamma_calibration.direction_hash,
                "num_directions": self.gamma_calibration.num_directions,
                "iterations": self.gamma_calibration.iterations,
            },
        )
        _write_probe_json(
            self.artifact_root / "protocol.json",
            {
                "spec_id": self.config["spec"]["spec_id"],
                "exp_id": self.config["spec"]["exp_id"],
                "seed": self.seed,
                "split_hash": self.split["split_hash"],
                "label_hash": self.split["shared_split_manifest"]["label_hash"],
                "source_manifest_hash": self.source_manifest_hash,
                "surrogate_checkpoint_sha256": self.surrogate_checkpoint_hash,
                "synthetic_global_params_sha256": self.synthetic_params_hash,
                "c2lm_source_basis_sha256": (
                    self.natural_source_basis.basis_hash
                ),
                "matched_basis_hashes": self.basis_hashes,
                "basis_ranks": self.basis_ranks,
                "config_hash": canonical_hash(self.config),
                "claim_boundary": (
                    "surrogate-only matched mechanism probe; forced pseudo "
                    "fallback support; not victim UE or instance-mask evidence"
                ),
            },
        )

    def _load_batch(self, paths: Sequence[Path]) -> ICMOBatch:
        return load_icmo_batch(
            paths,
            label_dir=self.train_label_dir,
            image_size=int(self.config["model"]["image_size"]),
            target_class_id=int(self.config["dataset"]["target_class_id"]),
            device=self.device,
        )

    def _collect(
        self,
        paths: Sequence[str],
        carrier: MatchedCanonicalCarrier,
        *,
        render_mode: str,
        gradients: bool,
        affine: Mapping[str, float] | None = None,
    ) -> list[ICMOObservation]:
        observations = []
        batches = make_batches(
            paths,
            batch_size=int(self.config["optimization"]["batch_size"]),
        )
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context:
            for path_batch in batches:
                observations.append(
                    self.engine.observe(
                        self._load_batch(path_batch),
                        carrier,
                        render_mode=render_mode,
                        affine=affine,
                    )
                )
        return observations

    def _source_correlation(
        self,
        pattern: torch.Tensor,
    ) -> float:
        channels = pattern.detach().cpu()
        mask = band_mask(
            channels.shape[-2],
            channels.shape[-1],
            ((2.0, 8.0), (8.0, 24.0)),
        )
        vectors = [
            F.normalize(channel.reshape(-1), dim=0)
            for channel in channels
        ]
        correlations = []
        for source_index, image in enumerate(self.source_images):
            for crop in deterministic_two_crops(
                image,
                resolution=int(pattern.shape[-1]),
                source_index=source_index,
            ):
                luminance = (
                    0.299 * crop[0]
                    + 0.587 * crop[1]
                    + 0.114 * crop[2]
                )
                spectrum = torch.fft.fft2(luminance.double()) * mask
                source = torch.fft.ifft2(spectrum).real.float()
                source = F.normalize(
                    (source - source.mean()).reshape(-1),
                    dim=0,
                )
                correlations.extend(
                    float(torch.dot(vector, source))
                    for vector in vectors
                )
        return max(abs(value) for value in correlations)

    def _run_arm(
        self,
        *,
        arm_id: str,
        family: str,
        render_mode: str,
    ) -> dict[str, Any]:
        carrier = self._carrier(family)
        optimizer = torch.optim.Adam(
            [carrier.coefficients],
            lr=float(self.config["optimization"]["learning_rate"]),
        )
        calibration_batches = make_batches(
            self.split["calibration"],
            batch_size=int(self.config["optimization"]["batch_size"]),
        )
        prewarm_heldout = self._collect(
            self.split["heldout"],
            carrier,
            render_mode=render_mode,
            gradients=False,
        )
        initial_route_loss = _median(
            float(observation.route.loss.detach())
            for observation in prewarm_heldout
        )
        diagnostic_rows = []
        gradient_norms = []

        warmup_steps = int(self.config["optimization"]["warmup_steps"])
        for step in range(warmup_steps):
            batch = self._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self.engine.observe(
                batch,
                carrier,
                render_mode=render_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            observation.route.loss.backward()
            gradient = carrier.coefficients.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm_id} warmup gradient is invalid.")
            gradient_norm = float(gradient.norm().detach())
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"{arm_id} warmup gradient norm is invalid.")
            gradient_norms.append(gradient_norm)
            optimizer.step()
            diagnostic_rows.append(
                {
                    "stage": "warmup",
                    "step": step,
                    "loss_route": float(observation.route.loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )

        initial_calibration = self._collect(
            self.split["calibration"],
            carrier,
            render_mode=render_mode,
            gradients=False,
        )
        bank = fit_icmo_bank(
            initial_calibration,
            momentum=float(self.config["optimization"]["prototype_momentum"]),
        )
        initial_heldout = self._collect(
            self.split["heldout"],
            carrier,
            render_mode=render_mode,
            gradients=False,
        )
        initial_calibration_summary = summarize_icmo_observations(
            initial_calibration,
            bank,
        )
        initial_heldout_summary = summarize_icmo_observations(
            initial_heldout,
            bank,
        )

        optimization_steps = int(
            self.config["optimization"]["optimization_steps"]
        )
        epsilon = float(self.config["carrier"]["epsilon"])
        target_rms = (
            epsilon * float(self.config["carrier"]["target_rms_ratio"])
        )
        for step in range(optimization_steps):
            batch = self._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self.engine.observe(
                batch,
                carrier,
                render_mode=render_mode,
            )
            cicr_result = instance_cicr(
                observation.target_residuals,
                bank,
            )
            pattern = carrier()
            rms_loss = (
                pattern.square().mean().sqrt() / target_rms - 1.0
            ).square()
            total_loss = (
                float(self.config["optimization"]["lambda_cicr"])
                * cicr_result.loss
                + float(self.config["optimization"]["lambda_route"])
                * observation.route.loss
                + float(self.config["optimization"]["lambda_rms"])
                * rms_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient = carrier.coefficients.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm_id} optimization gradient is invalid.")
            gradient_norm = float(gradient.norm().detach())
            if not math.isfinite(gradient_norm):
                raise RuntimeError(
                    f"{arm_id} optimization gradient norm is invalid."
                )
            gradient_norms.append(gradient_norm)
            optimizer.step()
            diagnostic_rows.append(
                {
                    "stage": "matched",
                    "step": step,
                    "loss_total": float(total_loss.detach()),
                    "loss_cicr": float(cicr_result.loss.detach()),
                    "loss_route": float(observation.route.loss.detach()),
                    "loss_canonical_rms": float(rms_loss.detach()),
                    "valid_instance_coverage": (
                        cicr_result.valid_instance_coverage
                    ),
                    "gradient_norm": gradient_norm,
                }
            )

        final_calibration = self._collect(
            self.split["calibration"],
            carrier,
            render_mode=render_mode,
            gradients=False,
        )
        final_heldout = self._collect(
            self.split["heldout"],
            carrier,
            render_mode=render_mode,
            gradients=False,
        )
        calibration_summary = summarize_icmo_observations(
            final_calibration,
            bank,
        )
        summary = summarize_icmo_observations(final_heldout, bank)
        final_route_loss = float(summary["route_loss"])
        final_pattern = carrier().detach()
        coefficient_activation = torch.tanh(carrier.coefficients.detach())
        basis_energy = coefficient_activation.square().sum(dim=1)
        coefficient_saturation = float(
            (coefficient_activation.abs() >= 0.95).float().mean()
        )
        active_basis_fraction = float(
            (coefficient_activation.norm(dim=1) > 1e-3).float().mean()
        )
        top1_share = float(
            basis_energy.max() / basis_energy.sum().clamp_min(1e-12)
        )
        affine_audit = {}
        for audit_id, affine in AFFINE_AUDITS.items():
            audited = self._collect(
                self.split["heldout"],
                carrier,
                render_mode=render_mode,
                gradients=False,
                affine=affine,
            )
            affine_audit[audit_id] = float(
                summarize_icmo_observations(audited, bank)[
                    "heldout_cicr_median"
                ]
            )

        prototype_state = bank.state_dict()
        prototypes = [
            value
            for value in prototype_state["prototypes"]
            if torch.is_tensor(value)
        ]
        prototype_hash = (
            tensor_sha256(torch.cat([value.flatten() for value in prototypes]))
            if prototypes
            else ""
        )
        summary.update(
            {
                "arm_id": arm_id,
                "family": family,
                "render_mode": render_mode,
                "initial_route_loss": initial_route_loss,
                "final_route_loss": final_route_loss,
                "route_effect": (initial_route_loss - final_route_loss)
                / max(abs(initial_route_loss), 1e-12),
                "initial_calibration_cicr": initial_calibration_summary[
                    "heldout_cicr_median"
                ],
                "final_calibration_cicr": calibration_summary[
                    "heldout_cicr_median"
                ],
                "initial_heldout_cicr": initial_heldout_summary[
                    "heldout_cicr_median"
                ],
                "calibration_cicr_gain": float(
                    calibration_summary["heldout_cicr_median"]
                    - initial_calibration_summary["heldout_cicr_median"]
                ),
                "heldout_cicr_gain": float(
                    summary["heldout_cicr_median"]
                    - initial_heldout_summary["heldout_cicr_median"]
                ),
                "calibration_heldout_gap": abs(
                    float(calibration_summary["heldout_cicr_median"])
                    - float(summary["heldout_cicr_median"])
                ),
                "canonical_rms": float(
                    final_pattern.square().mean().sqrt()
                ),
                "canonical_linf": float(final_pattern.abs().amax()),
                "canonical_spectrum_energy": spectrum_energy_ratios(
                    final_pattern.cpu()
                ),
                "source_max_abs_correlation": self._source_correlation(
                    final_pattern
                ),
                "coefficient_saturation_ratio": coefficient_saturation,
                "active_basis_fraction": active_basis_fraction,
                "top1_basis_energy_share": top1_share,
                "coefficient_hash": tensor_sha256(
                    carrier.coefficients.detach()
                ),
                "prototype_hash": prototype_hash,
                "basis_hash": self.basis_hashes[family],
                "affine_audit": affine_audit,
                "gradient_norms": gradient_norms,
                "diagnostics": diagnostic_rows,
            }
        )
        finite_values = [
            summary["heldout_cicr_median"],
            summary["heldout_cicr_q25"],
            summary["valid_image_coverage"],
            summary["valid_instance_coverage"],
            summary["non_target_target_energy_ratio"],
            summary["box_residual_energy"],
            summary["route_effect"],
            summary["active_pixel_rms"],
            summary["active_pixel_linf"],
            summary["canonical_rms"],
            summary["canonical_linf"],
            summary["source_max_abs_correlation"],
            *gradient_norms,
            *affine_audit.values(),
        ]
        summary["finite"] = (
            all(math.isfinite(float(value)) for value in finite_values)
            and bool(torch.isfinite(carrier.coefficients).all())
            and all(bool(torch.isfinite(value).all()) for value in prototypes)
            and bool(summary["outside_support_max"] == 0.0)
        )
        state_path = self.artifact_root / "arms" / f"{arm_id}_state.pt"
        torch.save(
            {
                "arm_id": arm_id,
                "family": family,
                "render_mode": render_mode,
                "coefficients": carrier.coefficients.detach().cpu(),
                "prototype_bank": prototype_state,
                "gamma": self.gamma_calibration.gamma,
                "basis_hash": self.basis_hashes[family],
            },
            state_path,
        )
        _write_probe_json(
            self.artifact_root / "arms" / f"{arm_id}_metrics.json",
            summary,
        )
        return summary

    def run(self) -> dict[str, Any]:
        status = {
            "spec_id": self.config["spec"]["spec_id"],
            "exp_id": self.config["spec"]["exp_id"],
            "state": "running",
            "config_hash": canonical_hash(self.config),
            "split_hash": self.split["split_hash"],
            "source_manifest_hash": self.source_manifest_hash,
            "surrogate_checkpoint_sha256": self.surrogate_checkpoint_hash,
        }
        _write_probe_json(self.artifact_root / "status.json", status)
        try:
            mechanical_checks = {
                "instance_ncc": (
                    float(self.mechanical["ncc_median"]) >= 0.98
                    and float(self.mechanical["ncc_q25"]) >= 0.95
                ),
                "render_paths_distinct": bool(
                    self.mechanical["render_paths_distinct"]
                ),
                "initial_rms_ratio": 0.98
                <= self.initial_rms_ratio
                <= 1.02,
                "basis_rank": all(
                    rank >= 8 for rank in self.basis_ranks.values()
                ),
            }
            if not all(mechanical_checks.values()):
                result = {
                    "spec_id": self.config["spec"]["spec_id"],
                    "exp_id": self.config["spec"]["exp_id"],
                    "arms": {},
                    "contrasts": {},
                    "mechanical": self.mechanical,
                    "mechanical_checks": mechanical_checks,
                    "decision": {
                        "pass": False,
                        "status": "mechanical_fail",
                        "failure_signals": {
                            "mechanical_precondition": True,
                        },
                    },
                    "claim_boundary": "mechanical failure; no mechanism evidence",
                }
                _write_probe_json(self.artifact_root / "metrics.json", result)
                status.update(
                    {
                        "state": "stopped",
                        "stop_reason": "mechanical_precondition_failure",
                        "mechanism_pass": False,
                    }
                )
                _write_probe_json(self.artifact_root / "status.json", status)
                return status
            arms = {}
            for arm_id, (family, render_mode) in ARM_DEFINITIONS.items():
                arms[arm_id] = self._run_arm(
                    arm_id=arm_id,
                    family=family,
                    render_mode=render_mode,
                )
            bootstrap_cfg = self.config["bootstrap"]

            def contrast(left: str, right: str) -> dict[str, Any]:
                result = stratified_paired_bootstrap(
                    arms[left]["per_image"],
                    arms[right]["per_image"],
                    seed=int(bootstrap_cfg["seed"]),
                    iterations=int(bootstrap_cfg["iterations"]),
                )
                result["arm_median_delta"] = float(
                    arms[left]["heldout_cicr_median"]
                    - arms[right]["heldout_cicr_median"]
                )
                return result

            contrasts = {
                "I-C2LM_vs_G-C2LM": contrast("I-C2LM", "G-C2LM"),
                "I-C2LM_vs_I-C0": contrast("I-C2LM", "I-C0"),
                "I-C0_vs_G-C0": contrast("I-C0", "G-C0"),
            }
            active_rms_values = [
                float(arm["active_pixel_rms"]) for arm in arms.values()
            ]
            active_rms_ratio = max(active_rms_values) / max(
                min(active_rms_values),
                1e-12,
            )
            hashes_complete = all(
                len(value) == 64
                for value in (
                    self.split["split_hash"],
                    self.source_manifest_hash,
                    self.surrogate_checkpoint_hash,
                    self.synthetic_params_hash,
                    self.natural_source_basis.basis_hash,
                    canonical_hash(self.config),
                    *self.basis_hashes.values(),
                    self.gamma_calibration.direction_hash,
                )
            ) and all(rank >= 8 for rank in self.basis_ranks.values())
            decision = evaluate_icmo_result(
                arms,
                contrasts,
                initial_rms_ratio=self.initial_rms_ratio,
                active_rms_ratio=active_rms_ratio,
                mechanical=self.mechanical,
                hashes_complete=hashes_complete,
            )
            result = {
                "spec_id": self.config["spec"]["spec_id"],
                "exp_id": self.config["spec"]["exp_id"],
                "arms": arms,
                "contrasts": contrasts,
                "mechanical": self.mechanical,
                "initial_rms_ratio": self.initial_rms_ratio,
                "active_rms_ratio": active_rms_ratio,
                "hashes_complete": hashes_complete,
                "decision": decision,
                "not_applicable": [
                    "clean_voc_map",
                    "victim_ap",
                    "dataset_psnr_lpips",
                ],
                "claim_boundary": (
                    "surrogate-only mechanism evidence; no fresh-victim "
                    "unlearnability or instance-mask claim"
                ),
            }
            _write_probe_json(self.artifact_root / "metrics.json", result)
            status.update(
                {
                    "state": (
                        "completed"
                        if decision["pass"]
                        else "stopped"
                    ),
                    "stop_reason": (
                        None
                        if decision["pass"]
                        else "icmo_failure_signal"
                    ),
                    "mechanism_pass": decision["pass"],
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
