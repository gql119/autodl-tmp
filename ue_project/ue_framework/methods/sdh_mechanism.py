from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ..support import _bbox_to_pixels
from ..ultra.hijacked_loss import HijackedV8Loss
from .alce_acgt import build_pag_gate
from .bsc_rc_gr_probe import _assigned_gt_boxes
from .detector_lfc import DetectorLFCExtractor, DetectorLFCFeatures
from .detector_tower_hooks import YOLODetectTowerCapture
from .dgcaip import DGCAIPResult, dgcaip_instance_preservation
from .instance_cicr import (
    InstanceClassificationResiduals,
    instance_classification_residuals,
    target_gt_indices_from_labels,
)
from .malc import target_class_assignment_scores
from .non_target_logit_alignment import (
    NonTargetLogitAlignmentResult,
    class_balanced_non_target_logit_alignment,
)
from .semantic_hiding_carrier import (
    RenderedSemanticCarrier,
    SemanticHidingCarrier,
    render_person_box_carrier,
)
from .semantic_hiding_validation import reveal_loss
from .shadow_tal import DifferentiableShadowTAL, TargetRouteResult, compute_target_route


@dataclass(frozen=True)
class SDHBatch:
    images: torch.Tensor
    yolo_batch: Dict[str, Any]
    boxes_by_image: Tuple[torch.Tensor, ...]
    image_ids: Tuple[str, ...]
    person_cooccur: Tuple[bool, ...]


@dataclass(frozen=True)
class SDHObservation:
    image_ids: Tuple[str, ...]
    rendered: RenderedSemanticCarrier
    canonical_dlfc_features: DetectorLFCFeatures
    target_residuals: InstanceClassificationResiduals
    route: TargetRouteResult
    reveal_loss: torch.Tensor
    rms_loss: torch.Tensor
    nla: NonTargetLogitAlignmentResult
    dgcaip: Optional[DGCAIPResult]
    per_class_probability_drop: Dict[int, float]
    real_foreground_count: int
    target_positive_count: int


@dataclass(frozen=True)
class SDHTargetObjective:
    loss: torch.Tensor
    active_components: Tuple[str, ...]
    weighted_components: Dict[str, torch.Tensor]


class FrozenTargetGradientCalibration:
    """One-shot median gradient-norm balancing for target components."""

    def __init__(
        self,
        *,
        reference: str = "easy",
        minimum: float = 1.0e-3,
        maximum: float = 100.0,
    ) -> None:
        if minimum <= 0 or maximum < minimum:
            raise ValueError("Invalid target calibration bounds.")
        self.reference = str(reference)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self._weights: Optional[Dict[str, float]] = None
        self._clipped: Optional[Dict[str, bool]] = None

    @property
    def weights(self) -> Dict[str, float]:
        if self._weights is None:
            raise RuntimeError("Target loss weights are not calibrated.")
        return dict(self._weights)

    def calibrate(
        self,
        component_gradient_norms: Mapping[str, Sequence[float]],
        *,
        split: str,
    ) -> Dict[str, float]:
        if split not in ("warmup", "train_calibration"):
            raise ValueError("Target weights may only use warm-up calibration.")
        if self._weights is not None:
            raise RuntimeError("Target loss weight calibration is already frozen.")
        if self.reference not in component_gradient_norms:
            raise ValueError("Target calibration reference component is missing.")
        medians: Dict[str, float] = {}
        inactive = set()
        for name, values in component_gradient_norms.items():
            tensor = torch.as_tensor(values, dtype=torch.float64)
            valid = torch.isfinite(tensor) & (tensor > 0)
            if not bool(valid.any()):
                if torch.isfinite(tensor).all() and bool((tensor == 0).all()):
                    medians[str(name)] = 0.0
                    inactive.add(str(name))
                    continue
                raise ValueError("Component %s has no finite non-negative gradient norm." % name)
            medians[str(name)] = float(tensor[valid].median())
        if self.reference in inactive:
            raise ValueError("Target calibration reference gradient is inactive.")
        reference = medians[self.reference]
        weights = {}
        clipped = {}
        for name, median in medians.items():
            raw = 1.0 if name in inactive else reference / median
            value = min(max(raw, self.minimum), self.maximum)
            weights[name] = float(value)
            clipped[name] = bool(value != raw or name in inactive)
        self._weights = weights
        self._clipped = clipped
        return dict(weights)

    def state_dict(self) -> Dict[str, object]:
        if self._weights is None or self._clipped is None:
            raise RuntimeError("Cannot serialize uncalibrated target weights.")
        return {
            "reference": self.reference,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "weights": dict(self._weights),
            "clipped": dict(self._clipped),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._weights is not None:
            raise RuntimeError("Cannot overwrite frozen target weights.")
        if str(state["reference"]) != self.reference:
            raise ValueError("Target calibration reference mismatch.")
        weights = {str(name): float(value) for name, value in dict(state["weights"]).items()}
        if any(not self.minimum <= value <= self.maximum for value in weights.values()):
            raise ValueError("Serialized target weight is outside configured bounds.")
        self._weights = weights
        self._clipped = {
            str(name): bool(value) for name, value in dict(state["clipped"]).items()
        }


def adapter_parameters(carrier: SemanticHidingCarrier) -> Tuple[torch.Tensor, ...]:
    parameters = tuple(carrier.adapter.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("All residual-adapter omega parameters must be trainable.")
    frozen_ids = {id(value) for value in carrier.hiding_trunk.parameters()}
    frozen_ids.update(id(value) for value in carrier.reveal_decoder.parameters())
    if any(id(parameter) in frozen_ids for parameter in parameters):
        raise RuntimeError("Omega overlaps a frozen hiding/reveal parameter.")
    return parameters


def compose_sdh_target_objective(
    *,
    easy: torch.Tensor,
    reveal: torch.Tensor,
    rms: torch.Tensor,
    dlfc: Optional[torch.Tensor],
    cicr: Optional[torch.Tensor],
    floor: Optional[torch.Tensor],
    weights: Mapping[str, float],
    enable_dlfc: bool,
    enable_cicr: bool,
) -> SDHTargetObjective:
    required = {"easy": easy, "reveal": reveal, "rms": rms}
    if enable_dlfc:
        if dlfc is None:
            raise ValueError("D-LFC is enabled but its loss is missing.")
        required["dlfc"] = dlfc
    elif dlfc is not None:
        raise ValueError("D-LFC loss was supplied while its switch is off.")
    if enable_cicr:
        if cicr is None or floor is None:
            raise ValueError("CICR is enabled but direction/floor loss is missing.")
        required["cicr"] = cicr
        required["floor"] = floor
    elif cicr is not None or floor is not None:
        raise ValueError("CICR loss was supplied while its switch is off.")
    missing = sorted(set(required).difference(weights))
    if missing:
        raise ValueError("Missing frozen target loss weights: %s" % missing)
    weighted = {name: float(weights[name]) * value for name, value in required.items()}
    loss = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(loss):
        raise ValueError("SDH target objective is non-finite.")
    return SDHTargetObjective(
        loss=loss,
        active_components=tuple(weighted),
        weighted_components=weighted,
    )


def load_sdh_batch(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    image_size: int,
    target_class_id: int,
    device: torch.device,
) -> SDHBatch:
    if not image_paths:
        raise ValueError("SDH batch cannot be empty.")
    images = []
    classes = []
    yolo_boxes = []
    batch_indices = []
    boxes_by_image = []
    image_ids = []
    cooccur = []
    for batch_index, image_path in enumerate(image_paths):
        image = load_image_rgb_float(str(image_path))
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        annotations = read_yolo_annotations(
            label_path_for_image(str(image_path), str(label_dir))
        )
        target_boxes = [
            _bbox_to_pixels(item["bbox"], image_size, image_size)
            for item in annotations
            if int(item["cls"]) == int(target_class_id)
        ]
        if not target_boxes:
            raise ValueError("SDH mechanism batches must contain person: %s" % image_path)
        boxes_tensor = torch.tensor(target_boxes, device=device, dtype=torch.float32)
        if bool((boxes_tensor[:, 2:] <= boxes_tensor[:, :2]).any()):
            raise ValueError("SDH batch contains a degenerate person box.")
        images.append(torch.from_numpy(image).permute(2, 0, 1).float())
        boxes_by_image.append(boxes_tensor)
        image_ids.append(image_path.stem)
        cooccur.append(any(int(item["cls"]) != int(target_class_id) for item in annotations))
        for item in annotations:
            classes.append([float(item["cls"])])
            yolo_boxes.append([float(value) for value in item["bbox"]])
            batch_indices.append(batch_index)
    images_tensor = torch.stack(images).to(device)
    yolo_batch = {
        "batch_idx": torch.tensor(batch_indices, device=device, dtype=torch.long),
        "cls": torch.tensor(classes, device=device, dtype=torch.float32),
        "bboxes": torch.tensor(yolo_boxes, device=device, dtype=torch.float32),
        "batch_size": len(image_paths),
        "img": images_tensor,
    }
    return SDHBatch(
        images=images_tensor,
        yolo_batch=yolo_batch,
        boxes_by_image=tuple(boxes_by_image),
        image_ids=tuple(image_ids),
        person_cooccur=tuple(cooccur),
    )


class SDHObservationEngine:
    """One differentiable clean/poison observation using real clean TAL."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_class_id: int = 14,
        num_classes: int = 20,
        epsilon: float = 16.0 / 255.0,
        assignment_topk: int = 100,
        pag_layer_ratios: Sequence[float] = (0.25, 0.25, 0.25),
        pag_min_pos: Sequence[int] = (1, 1, 1),
        box_teacher_weight: float = 1.0,
        dgcaip_temperature: float = 2.0,
        dgcaip_classification_tolerance: float = 0.005,
        dgcaip_box_tolerance: float = 0.02,
        dgcaip_alignment_tolerance: float = 0.05,
        dgcaip_minimum_rank_instances: int = 4,
    ) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.target_class_id = int(target_class_id)
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)
        self.assignment_topk = int(assignment_topk)
        self.pag_layer_ratios = tuple(float(value) for value in pag_layer_ratios)
        self.pag_min_pos = tuple(int(value) for value in pag_min_pos)
        self.box_teacher_weight = float(box_teacher_weight)
        self.dgcaip_temperature = float(dgcaip_temperature)
        self.dgcaip_classification_tolerance = float(
            dgcaip_classification_tolerance
        )
        self.dgcaip_box_tolerance = float(dgcaip_box_tolerance)
        self.dgcaip_alignment_tolerance = float(dgcaip_alignment_tolerance)
        self.dgcaip_minimum_rank_instances = int(dgcaip_minimum_rank_instances)
        self.capture = YOLODetectTowerCapture(
            self.model, expected_num_classes=self.num_classes
        )
        self.dlfc_extractor = DetectorLFCExtractor(
            self.model, eps=self.epsilon, expected_num_classes=self.num_classes
        )
        self.hijacked = HijackedV8Loss.from_surrogate(
            self.model,
            num_classes=self.num_classes,
            target_class_id=self.target_class_id,
        )
        self.shadow_tal = DifferentiableShadowTAL(
            target_class_id=self.target_class_id,
            alpha=0.5,
            beta=6.0,
            topk=self.assignment_topk,
        )
        self._counter = 0

    @staticmethod
    def _decoded(output: object) -> torch.Tensor:
        return output[0] if isinstance(output, (tuple, list)) else output

    def observe(
        self,
        batch: SDHBatch,
        carrier: SemanticHidingCarrier,
        secret: torch.Tensor,
        *,
        target_rms_ratio: float = 0.35,
        dgcaip_mode: str = "off",
        dgcaip_component_weights: Optional[Mapping[str, float]] = None,
    ) -> SDHObservation:
        if dgcaip_mode not in {"off", "caip", "dist", "dgcaip"}:
            raise ValueError("Unknown DG-CAIP observation mode.")
        rendered = render_person_box_carrier(
            batch.images, batch.boxes_by_image, carrier, secret
        )
        clean_tag = "sdh_clean_%d" % self._counter
        adv_tag = "sdh_adv_%d" % self._counter
        self._counter += 1
        with torch.no_grad():
            with self.capture.record(clean_tag):
                clean_output = self.model(batch.images)
            clean_features = self.capture.take(clean_tag)
            clean_predictions = self._decoded(clean_output)
            self.hijacked.last_real_assign = {}
            self.hijacked.get_assigned_targets_and_loss(clean_predictions, batch.yolo_batch)
            raw_assign = self.hijacked.last_real_assign
            required = ("fg_mask", "target_labels", "target_scores", "target_gt_idx")
            if any(not torch.is_tensor(raw_assign.get(name)) for name in required):
                raise RuntimeError("SDH requires complete clean real TAL assignments.")
            real_assign = {name: raw_assign[name].detach().clone() for name in required}
            clean_cache = self.hijacked.cache_assign_inputs_only(
                clean_predictions,
                batch.yolo_batch,
                image_shape=batch.images.shape[-2:],
                assignment_topk=self.assignment_topk,
            )
        with self.capture.record(adv_tag):
            adv_output = self.model(rendered.poisoned)
        adv_features = self.capture.take(adv_tag)
        adv_predictions = self._decoded(adv_output)
        adv_cache = self.hijacked.cache_assign_inputs_only(
            adv_predictions,
            batch.yolo_batch,
            image_shape=batch.images.shape[-2:],
            assignment_topk=self.assignment_topk,
        )

        labels = real_assign["target_labels"].long()
        if labels.ndim == 3:
            labels = labels[..., 0]
        foreground = real_assign["fg_mask"].bool()
        target_positive = foreground & (labels == self.target_class_id)
        layer_sizes = [
            value.shape[-2] * value.shape[-1]
            for value in adv_features.classification
        ]
        pag_gate, _ = build_pag_gate(
            strict_gate_1d=target_positive,
            target_scores=real_assign["target_scores"],
            target_class_id=self.target_class_id,
            top_ratio=self.pag_layer_ratios,
            min_keep=self.pag_min_pos,
            layer_sizes=layer_sizes,
        )
        target_gt_indices = target_gt_indices_from_labels(
            clean_cache["gt_labels"],
            clean_cache["mask_gt"],
            target_class_id=self.target_class_id,
        )
        target_residuals = instance_classification_residuals(
            clean_features.classification,
            adv_features.classification,
            pag_gate,
            real_assign["target_gt_idx"],
            target_gt_indices,
            assigned_scores=target_class_assignment_scores(
                real_assign["target_scores"], target_class_id=self.target_class_id
            ),
        )
        route = compute_target_route(
            route="easy_cls",
            adv_class_logits=adv_cache["pred_scores_logits"],
            adv_boxes=adv_cache["pred_bboxes"],
            clean_boxes=clean_cache["pred_bboxes"],
            target_gate=pag_gate,
            target_class_id=self.target_class_id,
            num_classes=self.num_classes,
            box_teacher_weight=self.box_teacher_weight,
            shadow_tal=self.shadow_tal,
            gt_labels=clean_cache["gt_labels"],
            gt_bboxes=clean_cache["gt_bboxes"],
            mask_gt=clean_cache["mask_gt"],
        )
        nla = class_balanced_non_target_logit_alignment(
            clean_cache["pred_scores_logits"],
            adv_cache["pred_scores_logits"],
            labels,
            foreground,
            target_class_id=self.target_class_id,
            assignment_source="clean_real_tal",
        )
        dgcaip = None
        if dgcaip_mode != "off":
            component_weights = {
                "classification": 1.0,
                "box": 1.0,
                "alignment": 1.0,
                "distribution": 1.0,
            }
            if dgcaip_component_weights is not None:
                component_weights.update(
                    {
                        str(name): float(value)
                        for name, value in dgcaip_component_weights.items()
                    }
                )
            if dgcaip_mode == "caip":
                component_weights["distribution"] = 0.0
            dgcaip = dgcaip_instance_preservation(
                clean_cache["pred_scores_logits"],
                adv_cache["pred_scores_logits"],
                clean_cache["pred_bboxes"],
                adv_cache["pred_bboxes"],
                labels,
                foreground,
                real_assign["target_gt_idx"],
                clean_cache["gt_labels"],
                clean_cache["gt_bboxes"],
                clean_cache["mask_gt"],
                target_class_id=self.target_class_id,
                assignment_source="clean_real_tal",
                component_weights=component_weights,
                enable_geometry_risk=True,
                enable_divergence_hardness=dgcaip_mode == "dgcaip",
                temperature=self.dgcaip_temperature,
                classification_tolerance=self.dgcaip_classification_tolerance,
                box_tolerance=self.dgcaip_box_tolerance,
                alignment_tolerance=self.dgcaip_alignment_tolerance,
                minimum_rank_instances=self.dgcaip_minimum_rank_instances,
            )
        probability_drop = {}
        for class_id in nla.active_classes:
            class_mask = foreground & (labels == class_id)
            clean_probability = clean_cache["pred_scores_logits"][..., class_id][
                class_mask
            ].detach().sigmoid()
            poison_probability = adv_cache["pred_scores_logits"][..., class_id][
                class_mask
            ].detach().sigmoid()
            probability_drop[class_id] = float(
                (clean_probability - poison_probability).mean().cpu()
            )
        canonical = torch.cat(rendered.canonical_deltas, dim=0)
        dlfc_features = self.dlfc_extractor.extract(canonical)
        recovered = torch.cat(rendered.recovered_secrets, dim=0)
        expanded_secret = secret.expand(recovered.shape[0], -1, -1, -1)
        reveal, _, _ = reveal_loss(recovered, expanded_secret)
        target_rms = self.epsilon * float(target_rms_ratio)
        rms = (canonical.square().mean().sqrt() / target_rms - 1.0).square()
        return SDHObservation(
            image_ids=batch.image_ids,
            rendered=rendered,
            canonical_dlfc_features=dlfc_features,
            target_residuals=target_residuals,
            route=route,
            reveal_loss=reveal,
            rms_loss=rms,
            nla=nla,
            dgcaip=dgcaip,
            per_class_probability_drop=probability_drop,
            real_foreground_count=int(foreground.sum().item()),
            target_positive_count=int(pag_gate.sum().item()),
        )

    def close(self) -> None:
        self.dlfc_extractor.close()
        self.capture.close()

    def __enter__(self) -> "SDHObservationEngine":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
