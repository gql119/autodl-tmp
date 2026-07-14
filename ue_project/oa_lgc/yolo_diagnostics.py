from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from oa_lgc.yolo_adapter import ReferenceAssignment, YOLOFunctionalAdapter, YOLOVirtualTrajectory


TARGET_DIAGNOSTIC_FIELDS = (
    "target_gt_count",
    "target_reference_positive_count",
    "target_clean_positive_count",
    "target_poison_positive_count",
    "target_positive_coverage",
    "target_assignment_overlap",
    "target_cls_loss",
    "target_box_loss",
    "target_dfl_loss",
    "target_matched_score",
    "target_localization_recall",
    "target_valid_reason",
    "valid",
)

CLASSWISE_DIAGNOSTIC_FIELDS = (
    "class_id",
    "class_name",
    "gt_count",
    "reference_positive_count",
    "clean_positive_count",
    "poison_positive_count",
    "classification_gain_clean",
    "classification_gain_poison",
    "authorized_gain_gap",
    "assignment_drift",
    "box_drift",
    "dfl_drift",
    "valid",
    "invalid_reason",
)


@dataclass(frozen=True)
class TargetDiagnostics:
    target_gt_count: int
    target_reference_positive_count: int
    target_clean_positive_count: int
    target_poison_positive_count: int
    target_positive_coverage: float
    target_assignment_overlap: float
    target_cls_loss: float
    target_box_loss: float
    target_dfl_loss: float
    target_matched_score: float
    target_localization_recall: float
    target_valid_reason: str
    valid: bool


@dataclass(frozen=True)
class ClasswiseDiagnostics:
    class_id: int
    class_name: str
    gt_count: int
    reference_positive_count: int
    clean_positive_count: int
    poison_positive_count: int
    classification_gain_clean: float | None
    classification_gain_poison: float | None
    authorized_gain_gap: float | None
    assignment_drift: float
    box_drift: float | None
    dfl_drift: float | None
    valid: bool
    invalid_reason: str


@dataclass
class EpisodeDiagnostics:
    target: TargetDiagnostics
    classes: dict[int, ClasswiseDiagnostics]
    reference: ReferenceAssignment
    clean_assignment: ReferenceAssignment
    poison_assignment: ReferenceAssignment
    target_fixed_losses: dict[str, torch.Tensor]
    class_fixed_losses: dict[int, dict[str, torch.Tensor]]

    def target_dict(self) -> dict:
        return asdict(self.target)

    def class_rows(self) -> list[dict]:
        return [asdict(self.classes[class_id]) for class_id in sorted(self.classes)]


def positive_mask(assignment: ReferenceAssignment, class_id: int) -> torch.Tensor:
    return assignment.target_scores[..., int(class_id)] > 0


def positive_coverage(reference_count: int, poison_count: int) -> float:
    return float(poison_count) / max(int(reference_count), 1)


def coverage_valid_reason(reference_count: int, poison_count: int, minimum: float = 0.5) -> str:
    if int(reference_count) <= 0:
        return "no_reference_positive"
    if positive_coverage(reference_count, poison_count) < float(minimum):
        return "target_positive_coverage_below_0.50"
    return ""


def assignment_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    union = (first | second).sum().item()
    if union == 0:
        return 1.0
    return float((first & second).sum().item() / union)


def _gt_count(reference: ReferenceAssignment, class_id: int) -> int:
    labels = reference.gt_labels[..., 0].to(torch.long)
    valid = reference.mask_gt[..., 0].bool()
    return int(((labels == int(class_id)) & valid).sum().item())


def _localization_recall(assignment: ReferenceAssignment, class_id: int) -> float:
    matched = 0
    total = 0
    class_mask = positive_mask(assignment, class_id)
    for batch_index in range(assignment.gt_labels.shape[0]):
        labels = assignment.gt_labels[batch_index, :, 0].to(torch.long)
        valid = assignment.mask_gt[batch_index, :, 0].bool()
        gt_indices = set(torch.nonzero((labels == int(class_id)) & valid, as_tuple=False).flatten().tolist())
        total += len(gt_indices)
        if gt_indices:
            assigned = set(
                assignment.target_gt_idx[batch_index][class_mask[batch_index]].detach().cpu().tolist()
            )
            matched += len(gt_indices & assigned)
    return float(matched / max(total, 1))


def _fixed_box_dfl_losses(
    adapter: YOLOFunctionalAdapter,
    raw_predictions: Mapping[str, torch.Tensor],
    reference: ReferenceAssignment,
    class_id: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mask = positive_mask(reference, class_id)
    if not torch.any(mask):
        return None, None
    criterion = adapter.model.criterion
    pred_distri = raw_predictions["boxes"].permute(0, 2, 1).contiguous()
    pred_bboxes = criterion.bbox_decode(reference.anchor_points, pred_distri)
    target_scores = torch.zeros_like(reference.target_scores)
    target_scores[..., int(class_id)] = reference.target_scores[..., int(class_id)]
    mass = target_scores.sum().clamp_min(1e-12)
    image_hw = torch.tensor(
        reference.image_size, device=pred_distri.device, dtype=pred_distri.dtype
    )
    box, dfl = criterion.bbox_loss(
        pred_distri,
        pred_bboxes,
        reference.anchor_points,
        reference.target_bboxes / reference.stride_tensor,
        target_scores,
        mass,
        mask,
        image_hw,
        reference.stride_tensor,
    )
    return box * float(criterion.hyp.box), dfl * float(criterion.hyp.dfl)


def build_episode_diagnostics(
    adapter: YOLOFunctionalAdapter,
    query_images: torch.Tensor,
    query_batch: Mapping[str, torch.Tensor],
    clean_trajectory: YOLOVirtualTrajectory,
    poison_trajectory: YOLOVirtualTrajectory,
    class_names: Mapping[int, str] | Mapping[str, str] | None = None,
    minimum_target_coverage: float = 0.5,
) -> EpisodeDiagnostics:
    reference = adapter.reference_assignment(query_images, query_batch)
    base_raw = adapter.forward(query_images, adapter.base_parameters(), adapter.clone_buffers())
    clean_raw = adapter.forward(
        query_images, clean_trajectory.parameters, adapter.clone_buffers(clean_trajectory.buffers)
    )
    poison_raw = adapter.forward(
        query_images, poison_trajectory.parameters, adapter.clone_buffers(poison_trajectory.buffers)
    )
    clean_assignment = adapter.extract_tal_diagnostics(clean_raw, query_batch)
    poison_assignment = adapter.extract_tal_diagnostics(poison_raw, query_batch)
    base_cls = adapter.compute_classwise_query_loss(
        query_images,
        query_batch,
        adapter.base_parameters(),
        adapter.clone_buffers(),
        reference,
    )
    clean_cls = adapter.compute_classwise_query_loss(
        query_images,
        query_batch,
        clean_trajectory.parameters,
        clean_trajectory.buffers,
        reference,
    )
    poison_cls = adapter.compute_classwise_query_loss(
        query_images,
        query_batch,
        poison_trajectory.parameters,
        poison_trajectory.buffers,
        reference,
    )

    fixed_losses: dict[int, dict[str, torch.Tensor]] = {}
    diagnostics: dict[int, ClasswiseDiagnostics] = {}
    for class_id in range(adapter.num_classes):
        reference_mask = positive_mask(reference, class_id)
        clean_mask = positive_mask(clean_assignment, class_id)
        poison_mask = positive_mask(poison_assignment, class_id)
        reference_count = int(reference_mask.sum().item())
        clean_count = int(clean_mask.sum().item())
        poison_count = int(poison_mask.sum().item())
        gt_count = _gt_count(reference, class_id)
        reason = ""
        if gt_count <= 0:
            reason = "class_absent_from_query_gt"
        elif reference_count <= 0:
            reason = "no_reference_positive"
        elif not all(result.valid[class_id] for result in (base_cls, clean_cls, poison_cls)):
            reason = "fixed_classification_loss_unavailable"
        base_box, base_dfl = _fixed_box_dfl_losses(adapter, base_raw, reference, class_id)
        clean_box, clean_dfl = _fixed_box_dfl_losses(adapter, clean_raw, reference, class_id)
        poison_box, poison_dfl = _fixed_box_dfl_losses(adapter, poison_raw, reference, class_id)
        if not reason and any(value is None for value in (base_box, base_dfl, clean_box, clean_dfl, poison_box, poison_dfl)):
            reason = "fixed_box_dfl_loss_unavailable"
        if not reason:
            fixed_losses[class_id] = {
                "base_cls": base_cls.losses[class_id],
                "clean_cls": clean_cls.losses[class_id],
                "poison_cls": poison_cls.losses[class_id],
                "base_box": base_box,
                "clean_box": clean_box,
                "poison_box": poison_box,
                "base_dfl": base_dfl,
                "clean_dfl": clean_dfl,
                "poison_dfl": poison_dfl,
            }
            clean_gain = base_cls.losses[class_id] - clean_cls.losses[class_id]
            poison_gain = base_cls.losses[class_id] - poison_cls.losses[class_id]
            gap = (poison_gain - clean_gain).abs() / clean_gain.abs().clamp_min(1e-8)
            box_drift = (poison_box - clean_box).abs()
            dfl_drift = (poison_dfl - clean_dfl).abs()
            clean_gain_value = float(clean_gain.detach())
            poison_gain_value = float(poison_gain.detach())
            gap_value = float(gap.detach())
            box_drift_value = float(box_drift.detach())
            dfl_drift_value = float(dfl_drift.detach())
        else:
            clean_gain_value = poison_gain_value = gap_value = None
            box_drift_value = dfl_drift_value = None
        name = str(class_id)
        if class_names is not None:
            name = str(class_names.get(class_id, class_names.get(str(class_id), name)))
        diagnostics[class_id] = ClasswiseDiagnostics(
            class_id=class_id,
            class_name=name,
            gt_count=gt_count,
            reference_positive_count=reference_count,
            clean_positive_count=clean_count,
            poison_positive_count=poison_count,
            classification_gain_clean=clean_gain_value,
            classification_gain_poison=poison_gain_value,
            authorized_gain_gap=gap_value,
            assignment_drift=1.0 - assignment_overlap(clean_mask, poison_mask),
            box_drift=box_drift_value,
            dfl_drift=dfl_drift_value,
            valid=not reason,
            invalid_reason=reason,
        )

    target_id = adapter.target_class_id
    target_row = diagnostics[target_id]
    coverage = positive_coverage(
        target_row.reference_positive_count, target_row.poison_positive_count
    )
    target_reason = coverage_valid_reason(
        target_row.reference_positive_count,
        target_row.poison_positive_count,
        minimum=minimum_target_coverage,
    )
    if not target_reason and not target_row.valid:
        target_reason = target_row.invalid_reason
    poison_target_mask = positive_mask(poison_assignment, target_id)
    matched_score = 0.0
    if torch.any(poison_target_mask):
        matched_score = float(
            poison_assignment.target_scores[..., target_id][poison_target_mask].mean().detach()
        )
    target_losses = fixed_losses.get(target_id, {})
    target = TargetDiagnostics(
        target_gt_count=target_row.gt_count,
        target_reference_positive_count=target_row.reference_positive_count,
        target_clean_positive_count=target_row.clean_positive_count,
        target_poison_positive_count=target_row.poison_positive_count,
        target_positive_coverage=coverage,
        target_assignment_overlap=assignment_overlap(
            positive_mask(reference, target_id), poison_target_mask
        ),
        target_cls_loss=float(target_losses["poison_cls"].detach()) if target_losses else float("nan"),
        target_box_loss=float(target_losses["poison_box"].detach()) if target_losses else float("nan"),
        target_dfl_loss=float(target_losses["poison_dfl"].detach()) if target_losses else float("nan"),
        target_matched_score=matched_score,
        target_localization_recall=_localization_recall(poison_assignment, target_id),
        target_valid_reason=target_reason,
        valid=not target_reason,
    )
    return EpisodeDiagnostics(
        target=target,
        classes=diagnostics,
        reference=reference,
        clean_assignment=clean_assignment,
        poison_assignment=poison_assignment,
        target_fixed_losses=target_losses,
        class_fixed_losses=fixed_losses,
    )
