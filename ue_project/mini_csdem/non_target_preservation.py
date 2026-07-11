from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from ue_framework.core.assignment_parser import infer_fpn_level_ids


@dataclass
class MatchedUnits:
    batch_indices: torch.Tensor
    level_indices: torch.Tensor
    anchor_indices: torch.Tensor
    matched_gt_indices: torch.Tensor
    matched_classes: torch.Tensor
    clean_non_target_count: int
    poison_non_target_count: int
    gt_index_mismatch_count: int
    class_alignment_counts: Dict[int, int]

    @property
    def matched_count(self) -> int:
        return int(self.anchor_indices.numel())

    @property
    def coverage(self) -> float:
        return self.matched_count / max(self.clean_non_target_count, 1)


@dataclass
class PreservationResult:
    total_loss: torch.Tensor
    logits_loss: torch.Tensor
    box_loss: torch.Tensor
    dfl_loss: torch.Tensor
    assignment_loss: torch.Tensor
    raw_logits_drift: torch.Tensor
    normalized_logits_drift: torch.Tensor
    clean_soft_mean: torch.Tensor
    clean_soft_std: torch.Tensor
    poison_soft_mean: torch.Tensor
    poison_soft_std: torch.Tensor
    alignment: MatchedUnits
    classwise_drift: Dict[int, torch.Tensor]


def _matched_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp_min(eps)


def align_non_target_units(
    clean_state: Dict[str, torch.Tensor], poison_state: Dict[str, torch.Tensor], target_class_id: int
) -> MatchedUnits:
    clean_fg = clean_state["fg_mask"].bool()
    poison_fg = poison_state["fg_mask"].bool()
    clean_labels = clean_state["target_labels"].long()
    poison_labels = poison_state["target_labels"].long()
    clean_non_target = clean_fg & (clean_labels != int(target_class_id))
    poison_non_target = poison_fg & (poison_labels != int(target_class_id))
    same_class = poison_labels == clean_labels
    same_gt = poison_state["target_gt_idx"].long() == clean_state["target_gt_idx"].long()
    gt_mismatch = clean_non_target & poison_non_target & same_class & ~same_gt
    matched = clean_non_target & poison_non_target & same_class & same_gt
    positions = torch.nonzero(matched, as_tuple=False)
    if positions.numel():
        batch_indices, anchor_indices = positions[:, 0], positions[:, 1]
        levels = infer_fpn_level_ids(clean_fg.shape[1], clean_fg.device)[anchor_indices]
        gt_indices = clean_state["target_gt_idx"][batch_indices, anchor_indices].long()
        classes = clean_labels[batch_indices, anchor_indices]
    else:
        empty = torch.empty(0, dtype=torch.long, device=clean_fg.device)
        batch_indices = anchor_indices = levels = gt_indices = classes = empty
    class_counts = {int(class_id): int((classes == class_id).sum()) for class_id in classes.unique().tolist()}
    return MatchedUnits(
        batch_indices=batch_indices,
        level_indices=levels,
        anchor_indices=anchor_indices,
        matched_gt_indices=gt_indices,
        matched_classes=classes,
        clean_non_target_count=int(clean_non_target.sum()),
        poison_non_target_count=int(poison_non_target.sum()),
        gt_index_mismatch_count=int(gt_mismatch.sum()),
        class_alignment_counts=class_counts,
    )


def compute_non_target_preservation(
    clean_state: Dict[str, torch.Tensor],
    poison_state: Dict[str, torch.Tensor],
    target_class_id: int,
    weights: Dict[str, float],
) -> PreservationResult:
    alignment = align_non_target_units(clean_state, poison_state, target_class_id)
    zero = poison_state["pred_scores"].sum() * 0.0
    if alignment.matched_count == 0:
        return PreservationResult(zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, alignment, {})

    batch_indices = alignment.batch_indices
    anchor_indices = alignment.anchor_indices
    labels = alignment.matched_classes

    clean_logits = clean_state["pred_scores"][batch_indices, anchor_indices].detach()
    poison_logits = poison_state["pred_scores"][batch_indices, anchor_indices]
    raw_logits_per_unit = (poison_logits - clean_logits).square().mean(dim=1)
    clean_logits_normalized = F.layer_norm(clean_logits, (clean_logits.shape[-1],))
    poison_logits_normalized = F.layer_norm(poison_logits, (poison_logits.shape[-1],))
    logits_per_unit = (poison_logits_normalized - clean_logits_normalized).square().mean(dim=1)

    stride = poison_state["stride_tensor"][anchor_indices]
    clean_boxes = (clean_state["pred_bboxes"][batch_indices, anchor_indices] * stride).detach()
    poison_boxes = poison_state["pred_bboxes"][batch_indices, anchor_indices] * stride
    scale = poison_state["imgsz"][[1, 0, 1, 0]].to(poison_boxes)
    box_per_unit = F.smooth_l1_loss(poison_boxes / scale, clean_boxes / scale, reduction="none").mean(dim=1)

    clean_dfl_logits = clean_state["pred_distri"][batch_indices, anchor_indices].detach()
    poison_dfl_logits = poison_state["pred_distri"][batch_indices, anchor_indices]
    reg_max = clean_dfl_logits.shape[-1] // 4
    clean_dfl = clean_dfl_logits.reshape(-1, 4, reg_max).softmax(dim=-1)
    poison_dfl = poison_dfl_logits.reshape(-1, 4, reg_max).softmax(dim=-1)
    dfl_per_unit = (poison_dfl - clean_dfl).square().mean(dim=(1, 2))

    clean_gt = clean_state["target_bboxes"][batch_indices, anchor_indices].detach()
    poison_gt = poison_state["target_bboxes"][batch_indices, anchor_indices].detach()
    clean_score = clean_logits.gather(1, labels.unsqueeze(1)).squeeze(1).sigmoid()
    poison_score = poison_logits.gather(1, labels.unsqueeze(1)).squeeze(1).sigmoid()
    clean_soft = clean_score * _matched_iou(clean_boxes, clean_gt)
    poison_soft = poison_score * _matched_iou(poison_boxes, poison_gt)
    assignment_per_unit = (poison_soft - clean_soft.detach()).square()

    raw_logits_drift = raw_logits_per_unit.mean()
    logits_loss = logits_per_unit.mean()
    box_loss = box_per_unit.mean()
    dfl_loss = dfl_per_unit.mean()
    assignment_loss = assignment_per_unit.mean()
    total = (
        float(weights.get("logits", 0.0)) * logits_loss
        + float(weights.get("box", 0.0)) * box_loss
        + float(weights.get("dfl", 0.0)) * dfl_loss
        + float(weights.get("assignment", 0.0)) * assignment_loss
    )
    classwise = {}
    for class_id in labels.unique().tolist():
        selected = labels == int(class_id)
        classwise[int(class_id)] = (
            float(weights.get("logits", 0.0)) * logits_per_unit[selected].mean()
            + float(weights.get("box", 0.0)) * box_per_unit[selected].mean()
            + float(weights.get("dfl", 0.0)) * dfl_per_unit[selected].mean()
            + float(weights.get("assignment", 0.0)) * assignment_per_unit[selected].mean()
        )
    return PreservationResult(
        total_loss=total,
        logits_loss=logits_loss,
        box_loss=box_loss,
        dfl_loss=dfl_loss,
        assignment_loss=assignment_loss,
        raw_logits_drift=raw_logits_drift,
        normalized_logits_drift=logits_loss,
        clean_soft_mean=clean_soft.mean(),
        clean_soft_std=clean_soft.std(unbiased=False),
        poison_soft_mean=poison_soft.mean(),
        poison_soft_std=poison_soft.std(unbiased=False),
        alignment=alignment,
        classwise_drift=classwise,
    )
