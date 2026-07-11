from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .gt_conditioned_partition import TargetPartition


def target_only_detection_loss(adapter, state: Dict[str, torch.Tensor], partition: TargetPartition) -> Dict[str, torch.Tensor]:
    selected = partition.unit_mask.bool()
    zero = state["pred_scores"].sum() * 0.0
    if not selected.any():
        return {"total_loss": zero, "cls_loss": zero, "box_loss": zero, "dfl_loss": zero}

    positions = torch.nonzero(selected, as_tuple=False)
    batch_indices, anchor_indices = positions[:, 0], positions[:, 1]
    labels = partition.target_labels[batch_indices, anchor_indices].long()
    logits = state["pred_scores"][batch_indices, anchor_indices].gather(1, labels.unsqueeze(1)).squeeze(1)
    assigned_scores = partition.target_scores[batch_indices, anchor_indices].to(logits).clamp(0.0, 1.0)
    score_sum = assigned_scores.sum().clamp_min(1.0)
    cls_loss = F.binary_cross_entropy_with_logits(logits, assigned_scores, reduction="sum") / score_sum

    target_scores = torch.zeros_like(state["target_scores"])
    target_scores[batch_indices, anchor_indices, labels] = assigned_scores
    box_loss, dfl_loss = adapter.criterion.bbox_loss(
        state["pred_distri"],
        state["pred_bboxes"],
        state["anchor_points"],
        partition.target_bboxes / state["stride_tensor"],
        target_scores,
        score_sum,
        selected,
        state["imgsz"],
        state["stride_tensor"],
    )
    box_loss = box_loss * adapter._hyp_gain("box")
    cls_loss = cls_loss * adapter._hyp_gain("cls")
    dfl_loss = dfl_loss * adapter._hyp_gain("dfl")
    return {
        "total_loss": cls_loss + box_loss + dfl_loss,
        "cls_loss": cls_loss,
        "box_loss": box_loss,
        "dfl_loss": dfl_loss,
    }

