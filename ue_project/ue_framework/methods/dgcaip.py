from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cooccurrence_instance_preservation import (
    collect_cooccurring_non_target_instances,
)


_COMPONENTS = ("classification", "box", "alignment", "distribution")


@dataclass(frozen=True)
class DivergenceWeightResult:
    weights: torch.Tensor
    percentile_ranks: torch.Tensor
    hardness: torch.Tensor


@dataclass(frozen=True)
class DGCAIPInstanceTerm:
    batch_index: int
    gt_index: int
    class_id: int
    positive_count: int
    geometry_risk: float
    divergence_rank: float
    weight: float
    classification_damage: torch.Tensor
    box_damage: torch.Tensor
    alignment_damage: torch.Tensor
    classification_loss: torch.Tensor
    box_loss: torch.Tensor
    alignment_loss: torch.Tensor
    distribution_loss: torch.Tensor
    clean_to_poison_kl: torch.Tensor


@dataclass(frozen=True)
class DGCAIPResult:
    loss: torch.Tensor
    instances: Tuple[DGCAIPInstanceTerm, ...]
    active_classes: Tuple[int, ...]
    per_class_loss: Dict[int, torch.Tensor]
    per_class_instance_count: Dict[int, int]
    eligible_instance_count: int
    covered_instance_count: int
    coverage: float


def _bounded_mean_one(
    values: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Weight values must be a non-empty vector.")
    if lower <= 0 or lower >= 1 or upper <= 1:
        raise ValueError("Weight bounds must satisfy 0 < lower < 1 < upper.")
    if not torch.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("Unnormalized weights must be finite and positive.")
    detached = values.detach()
    low = 0.0
    high = float(upper / detached.min().item())
    for _ in range(64):
        middle = 0.5 * (low + high)
        mean = detached.mul(middle).clamp(lower, upper).mean().item()
        if mean < 1.0:
            low = middle
        else:
            high = middle
    return detached.mul(0.5 * (low + high)).clamp(lower, upper)


def divergence_guided_weights(
    divergences: torch.Tensor,
    geometry_risks: torch.Tensor,
    *,
    minimum_rank_instances: int = 4,
    lower: float = 0.5,
    upper: float = 2.0,
) -> DivergenceWeightResult:
    """Build detached, bounded, mean-one protection weights."""

    if divergences.ndim != 1 or geometry_risks.shape != divergences.shape:
        raise ValueError("Divergences and geometry risks must align as vectors.")
    if divergences.numel() == 0:
        empty = divergences.detach().clone()
        return DivergenceWeightResult(empty, empty, empty)
    if minimum_rank_instances < 2:
        raise ValueError("minimum_rank_instances must be at least two.")
    detached_divergence = divergences.detach()
    detached_geometry = geometry_risks.detach()
    if not torch.isfinite(detached_divergence).all() or bool(
        (detached_divergence < 0).any()
    ):
        raise ValueError("Divergences must be finite and non-negative.")
    if not torch.isfinite(detached_geometry).all() or bool(
        (detached_geometry <= 0).any()
    ):
        raise ValueError("Geometry risks must be finite and positive.")
    if divergences.numel() < minimum_rank_instances:
        ranks = torch.zeros_like(detached_divergence)
        hardness = torch.ones_like(detached_divergence)
    else:
        lower_counts = (
            detached_divergence[:, None] > detached_divergence[None, :]
        ).sum(dim=1)
        ranks = lower_counts.to(detached_divergence.dtype) / float(
            divergences.numel() - 1
        )
        hardness = 1.0 + 2.0 * ranks.square()
    weights = _bounded_mean_one(
        detached_geometry * hardness,
        lower=lower,
        upper=upper,
    )
    return DivergenceWeightResult(
        weights=weights,
        percentile_ranks=ranks,
        hardness=hardness,
    )


def dataset_rank_guided_weights(
    percentile_ranks: torch.Tensor,
    geometry_risks: torch.Tensor,
    *,
    lower: float = 0.5,
    upper: float = 2.0,
) -> DivergenceWeightResult:
    """Convert frozen dataset-level ranks into bounded per-batch weights."""

    if percentile_ranks.ndim != 1 or geometry_risks.shape != percentile_ranks.shape:
        raise ValueError("Dataset ranks and geometry risks must align as vectors.")
    detached_ranks = percentile_ranks.detach()
    detached_geometry = geometry_risks.detach()
    if not torch.isfinite(detached_ranks).all() or bool(
        ((detached_ranks < 0) | (detached_ranks > 1)).any()
    ):
        raise ValueError("Dataset percentile ranks must be finite in [0,1].")
    if not torch.isfinite(detached_geometry).all() or bool(
        (detached_geometry <= 0).any()
    ):
        raise ValueError("Geometry risks must be finite and positive.")
    hardness = 1.0 + 2.0 * detached_ranks.square()
    weights = _bounded_mean_one(
        detached_geometry * hardness,
        lower=lower,
        upper=upper,
    )
    return DivergenceWeightResult(
        weights=weights,
        percentile_ranks=detached_ranks,
        hardness=hardness,
    )


def _rectangle_union_area(rectangles: Sequence[Sequence[float]]) -> float:
    if not rectangles:
        return 0.0
    x_coordinates = sorted({value for box in rectangles for value in (box[0], box[2])})
    area = 0.0
    for left, right in zip(x_coordinates[:-1], x_coordinates[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        intervals = sorted(
            (box[1], box[3])
            for box in rectangles
            if box[0] <= midpoint < box[2] and box[3] > box[1]
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        area += (right - left) * covered
    return area


def cooccurrence_geometry_risk(
    non_target_box: torch.Tensor,
    person_boxes: torch.Tensor,
) -> float:
    if non_target_box.shape != (4,) or person_boxes.ndim != 2 or person_boxes.shape[-1] != 4:
        raise ValueError("Geometry risk expects one [4] box and person boxes [N,4].")
    if person_boxes.shape[0] == 0:
        return 1.0
    if not torch.isfinite(non_target_box).all() or not torch.isfinite(person_boxes).all():
        raise ValueError("Geometry-risk boxes must be finite.")
    target = [float(value) for value in non_target_box.detach().cpu().tolist()]
    if target[2] <= target[0] or target[3] <= target[1]:
        raise ValueError("Non-target GT box must have positive area.")
    person = [
        [float(value) for value in row]
        for row in person_boxes.detach().cpu().tolist()
    ]
    if any(box[2] <= box[0] or box[3] <= box[1] for box in person):
        raise ValueError("Person GT boxes must have positive area.")
    clipped = []
    for box in person:
        intersection = [
            max(target[0], box[0]),
            max(target[1], box[1]),
            min(target[2], box[2]),
            min(target[3], box[3]),
        ]
        if intersection[2] > intersection[0] and intersection[3] > intersection[1]:
            clipped.append(intersection)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    overlap = _rectangle_union_area(clipped) / target_area
    distances = []
    for box in person:
        horizontal = max(box[0] - target[2], target[0] - box[2], 0.0)
        vertical = max(box[1] - target[3], target[1] - box[3], 0.0)
        distances.append(math.hypot(horizontal, vertical))
    normalized_distance = min(distances) / math.sqrt(target_area)
    raw = 1.0 + min(overlap / 0.25, 1.0) + math.exp(-normalized_distance)
    return float(min(max(raw, 1.0), 3.0))


def _aligned_iou(boxes: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], gt_box[:2])
    bottom_right = torch.minimum(boxes[:, 2:], gt_box[2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    box_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    gt_area = (gt_box[2:] - gt_box[:2]).clamp_min(0).prod()
    return intersection / (box_area + gt_area - intersection).clamp_min(1.0e-9)


def dgcaip_instance_preservation(
    clean_logits: torch.Tensor,
    poison_logits: torch.Tensor,
    clean_boxes: torch.Tensor,
    poison_boxes: torch.Tensor,
    assigned_labels: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    target_class_id: int,
    assignment_source: str,
    component_weights: Optional[Mapping[str, float]] = None,
    enable_geometry_risk: bool = True,
    enable_divergence_hardness: bool = True,
    temperature: float = 2.0,
    classification_tolerance: float = 0.005,
    box_tolerance: float = 0.02,
    alignment_tolerance: float = 0.05,
    minimum_rank_instances: int = 4,
    image_ids: Optional[Sequence[str]] = None,
    dataset_percentile_ranks: Optional[
        Mapping[Tuple[str, int, int], float]
    ] = None,
) -> DGCAIPResult:
    if clean_boxes.shape != poison_boxes.shape or clean_boxes.shape[:2] != clean_logits.shape[:2] or clean_boxes.shape[-1] != 4:
        raise ValueError("Clean/poison boxes must align with logits as [B,A,4].")
    if gt_bboxes.ndim != 3 or gt_bboxes.shape[:2] != gt_labels.shape[:2] or gt_bboxes.shape[-1] != 4:
        raise ValueError("gt_bboxes must align with gt_labels as [B,M,4].")
    if not torch.isfinite(clean_boxes).all() or not torch.isfinite(poison_boxes).all():
        raise ValueError("DG-CAIP decoded boxes must be finite.")
    if min(classification_tolerance, box_tolerance, alignment_tolerance) < 0:
        raise ValueError("DG-CAIP tolerances must be non-negative.")
    if int(minimum_rank_instances) < 1:
        raise ValueError("minimum_rank_instances must be positive.")
    weights_by_component = {name: 1.0 for name in _COMPONENTS}
    if component_weights is not None:
        if set(component_weights) != set(_COMPONENTS):
            raise ValueError("component_weights must define all DG-CAIP components.")
        weights_by_component.update(
            {name: float(value) for name, value in component_weights.items()}
        )
    if any(value < 0 or not math.isfinite(value) for value in weights_by_component.values()):
        raise ValueError("DG-CAIP component weights must be finite and non-negative.")

    collection = collect_cooccurring_non_target_instances(
        clean_logits,
        poison_logits,
        assigned_labels,
        foreground_mask,
        target_gt_idx,
        gt_labels,
        mask_gt,
        target_class_id=target_class_id,
        assignment_source=assignment_source,
        temperature=temperature,
    )
    if not collection.instances:
        return DGCAIPResult(
            loss=poison_logits.sum() * 0.0 + poison_boxes.sum() * 0.0,
            instances=(),
            active_classes=(),
            per_class_loss={},
            per_class_instance_count={},
            eligible_instance_count=collection.eligible_instance_count,
            covered_instance_count=0,
            coverage=collection.coverage,
        )

    assigned_labels = assigned_labels[..., 0] if assigned_labels.ndim == 3 else assigned_labels
    foreground_mask = foreground_mask[..., 0] if foreground_mask.ndim == 3 else foreground_mask
    target_gt_idx = target_gt_idx[..., 0] if target_gt_idx.ndim == 3 else target_gt_idx
    gt_labels_2d = gt_labels[..., 0] if gt_labels.ndim == 3 else gt_labels
    mask_gt_2d = mask_gt[..., 0] if mask_gt.ndim == 3 else mask_gt
    gt_labels_2d = gt_labels_2d.to(device=poison_logits.device).long()
    mask_gt_2d = mask_gt_2d.to(device=poison_logits.device).bool()
    foreground_mask = foreground_mask.to(device=poison_logits.device).bool()
    target_gt_idx = target_gt_idx.to(device=poison_logits.device).long()
    geometry_risks = []
    component_terms = []
    for record in collection.instances:
        batch_index = record.batch_index
        gt_index = record.gt_index
        class_id = record.class_id
        positive_mask = foreground_mask[batch_index] & (
            target_gt_idx[batch_index] == gt_index
        )
        clean_assigned = clean_logits[batch_index, positive_mask, class_id].detach()
        poison_assigned = poison_logits[batch_index, positive_mask, class_id]
        classification_drop = clean_assigned.sigmoid() - poison_assigned.sigmoid()
        classification_damage = F.relu(classification_drop).mean()
        classification_loss = F.relu(
            classification_drop - float(classification_tolerance)
        ).mean()
        gt_box = gt_bboxes[batch_index, gt_index].to(device=poison_boxes.device)
        clean_iou = _aligned_iou(
            clean_boxes[batch_index, positive_mask].detach(), gt_box
        )
        poison_iou = _aligned_iou(
            poison_boxes[batch_index, positive_mask], gt_box
        )
        box_drop = clean_iou - poison_iou
        box_damage = F.relu(box_drop).mean()
        box_loss = F.relu(box_drop - float(box_tolerance)).mean()
        clean_alignment = clean_assigned.sigmoid().pow(0.5) * clean_iou.pow(6.0)
        poison_alignment = poison_assigned.sigmoid().pow(0.5) * poison_iou.pow(6.0)
        alignment_drop = (clean_alignment - poison_alignment) / clean_alignment.clamp_min(
            1.0e-6
        )
        alignment_damage = F.relu(alignment_drop).mean()
        alignment_loss = F.relu(
            alignment_drop - float(alignment_tolerance)
        ).mean()
        valid_person = mask_gt_2d[batch_index] & (
            gt_labels_2d[batch_index] == int(target_class_id)
        )
        geometry_risk = (
            cooccurrence_geometry_risk(
                gt_box.detach(),
                gt_bboxes[batch_index, valid_person].to(device=gt_box.device),
            )
            if enable_geometry_risk
            else 1.0
        )
        geometry_risks.append(geometry_risk)
        component_terms.append(
            (
                classification_damage,
                box_damage,
                alignment_damage,
                classification_loss,
                box_loss,
                alignment_loss,
                record.js_divergence,
            )
        )

    divergence_values = torch.stack(
        [record.js_divergence.detach() for record in collection.instances]
    )
    geometry_tensor = divergence_values.new_tensor(geometry_risks)
    if dataset_percentile_ranks is not None:
        if image_ids is None or len(image_ids) != clean_logits.shape[0]:
            raise ValueError(
                "Dataset-ranked DG-CAIP requires one image_id per batch item."
            )
        frozen_ranks = []
        for record in collection.instances:
            key = (
                str(image_ids[record.batch_index]),
                record.gt_index,
                record.class_id,
            )
            if key not in dataset_percentile_ranks:
                raise ValueError("Dataset risk bank is missing an active DG-CAIP instance.")
            rank = float(dataset_percentile_ranks[key])
            if not math.isfinite(rank) or not 0.0 <= rank <= 1.0:
                raise ValueError("Dataset risk-bank rank must be finite in [0,1].")
            frozen_ranks.append(rank)
        weight_result = dataset_rank_guided_weights(
            divergence_values.new_tensor(frozen_ranks),
            geometry_tensor,
        )
    elif enable_divergence_hardness:
        weight_result = divergence_guided_weights(
            divergence_values,
            geometry_tensor,
            minimum_rank_instances=int(minimum_rank_instances),
        )
    else:
        weight_result = divergence_guided_weights(
            torch.zeros_like(divergence_values),
            geometry_tensor,
            minimum_rank_instances=divergence_values.numel() + 1,
        )

    terms = []
    per_class_values: Dict[int, list[torch.Tensor]] = {}
    for index, (record, values) in enumerate(zip(collection.instances, component_terms)):
        (
            classification_damage,
            box_damage,
            alignment_damage,
            classification_loss,
            box_loss,
            alignment_loss,
            distribution_loss,
        ) = values
        combined = weight_result.weights[index] * (
            weights_by_component["classification"] * classification_loss
            + weights_by_component["box"] * box_loss
            + weights_by_component["alignment"] * alignment_loss
            + weights_by_component["distribution"] * distribution_loss
        )
        per_class_values.setdefault(record.class_id, []).append(combined)
        terms.append(
            DGCAIPInstanceTerm(
                batch_index=record.batch_index,
                gt_index=record.gt_index,
                class_id=record.class_id,
                positive_count=record.positive_count,
                geometry_risk=float(geometry_risks[index]),
                divergence_rank=float(weight_result.percentile_ranks[index].cpu()),
                weight=float(weight_result.weights[index].cpu()),
                classification_damage=classification_damage,
                box_damage=box_damage,
                alignment_damage=alignment_damage,
                classification_loss=classification_loss,
                box_loss=box_loss,
                alignment_loss=alignment_loss,
                distribution_loss=distribution_loss,
                clean_to_poison_kl=record.clean_to_poison_kl,
            )
        )
    per_class_loss = {
        class_id: torch.stack(values).mean()
        for class_id, values in sorted(per_class_values.items())
    }
    per_class_count = {
        class_id: len(values)
        for class_id, values in sorted(per_class_values.items())
    }
    return DGCAIPResult(
        loss=torch.stack(tuple(per_class_loss.values())).mean(),
        instances=tuple(terms),
        active_classes=tuple(per_class_loss),
        per_class_loss=per_class_loss,
        per_class_instance_count=per_class_count,
        eligible_instance_count=collection.eligible_instance_count,
        covered_instance_count=collection.covered_instance_count,
        coverage=collection.coverage,
    )


class FrozenDGCAIPGradientCalibration:
    def __init__(
        self,
        *,
        min_weight: float = 1.0e-4,
        max_weight: float = 100.0,
    ) -> None:
        if min_weight <= 0 or not min_weight <= 1.0 <= max_weight:
            raise ValueError("Invalid DG-CAIP calibration bounds.")
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._weights: Optional[Dict[str, float]] = None
        self._warmup_count = 0

    @property
    def value(self) -> Dict[str, float]:
        if self._weights is None:
            raise RuntimeError("DG-CAIP component weights are not calibrated.")
        return dict(self._weights)

    def calibrate(
        self,
        component_gradient_norms: Mapping[str, Sequence[float]],
        *,
        split: str,
    ) -> Dict[str, float]:
        if split not in ("warmup", "train_calibration"):
            raise ValueError("DG-CAIP calibration may only use the warm-up split.")
        if self._weights is not None:
            raise RuntimeError("DG-CAIP calibration is already frozen.")
        if set(component_gradient_norms) != set(_COMPONENTS):
            raise ValueError("Calibration must provide every DG-CAIP component.")
        lengths = {len(values) for values in component_gradient_norms.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("Calibration norm sequences must be non-empty and aligned.")
        stacked = torch.stack(
            [
                torch.as_tensor(component_gradient_norms[name], dtype=torch.float64)
                for name in _COMPONENTS
            ]
        )
        valid = torch.isfinite(stacked).all(dim=0) & (stacked > 0).all(dim=0)
        if not bool(valid.any()):
            raise ValueError("Calibration has no shared finite positive batches.")
        medians = stacked[:, valid].median(dim=1).values
        reference = float(medians[0])
        weights = {"classification": 1.0}
        for index, name in enumerate(_COMPONENTS[1:], start=1):
            raw = reference / float(medians[index])
            weights[name] = min(max(raw, self.min_weight), self.max_weight)
        self._weights = weights
        self._warmup_count = int(valid.sum().item())
        return dict(weights)

    def state_dict(self) -> Dict[str, object]:
        if self._weights is None:
            raise RuntimeError("Cannot serialize uncalibrated DG-CAIP weights.")
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weights": dict(self._weights),
            "warmup_count": self._warmup_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._weights is not None:
            raise RuntimeError("Cannot overwrite frozen DG-CAIP calibration.")
        raw_weights = state.get("weights")
        if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(_COMPONENTS):
            raise ValueError("Serialized DG-CAIP weights are incomplete.")
        weights = {name: float(raw_weights[name]) for name in _COMPONENTS}
        if any(
            not self.min_weight <= value <= self.max_weight
            for value in weights.values()
        ):
            raise ValueError("Serialized DG-CAIP weight is outside configured bounds.")
        self._weights = weights
        self._warmup_count = int(state["warmup_count"])
