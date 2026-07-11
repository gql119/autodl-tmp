from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from ue_framework.core.assignment_parser import infer_fpn_level_ids


@dataclass
class TargetPartition:
    unit_mask: torch.Tensor
    target_labels: torch.Tensor
    target_bboxes: torch.Tensor
    target_scores: torch.Tensor
    target_gt_idx: torch.Tensor
    tal_positive_count: int
    fallback_positive_count: int

    @property
    def fallback_ratio(self) -> float:
        total = self.tal_positive_count + self.fallback_positive_count
        return self.fallback_positive_count / max(total, 1)


def build_target_partition(
    adapter,
    state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    target_class_id: int,
    minimum_target_positives: int,
    fallback_min_per_level: int,
) -> TargetPartition:
    fg_mask = state["fg_mask"].bool()
    labels = state["target_labels"].long().clone()
    target_mask = fg_mask & (labels == int(target_class_id))
    bboxes = state["target_bboxes"].clone()
    class_scores = state["target_scores"].gather(2, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    scores = class_scores.clone()
    gt_indices = state["target_gt_idx"].long().clone()
    tal_count = int(target_mask.sum().item())
    fallback_count = 0

    if tal_count < int(minimum_target_positives):
        pred_scores = state["pred_scores"]
        batch_size, num_units, _ = pred_scores.shape
        anchor_pixels = state["anchor_points"] * state["stride_tensor"]
        level_ids = infer_fpn_level_ids(num_units, pred_scores.device)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = adapter.criterion.preprocess(
            targets.to(adapter.criterion.device),
            batch_size,
            scale_tensor=state["imgsz"][[1, 0, 1, 0]],
        )
        gt_labels, gt_boxes = targets.split((1, 4), 2)
        for batch_index in range(batch_size):
            for gt_index in range(gt_labels.shape[1]):
                if int(gt_labels[batch_index, gt_index, 0].item()) != int(target_class_id):
                    continue
                box = gt_boxes[batch_index, gt_index]
                if float(box.sum()) <= 0:
                    continue
                inside = (
                    (anchor_pixels[:, 0] >= box[0])
                    & (anchor_pixels[:, 0] <= box[2])
                    & (anchor_pixels[:, 1] >= box[1])
                    & (anchor_pixels[:, 1] <= box[3])
                    & ~fg_mask[batch_index]
                )
                center = 0.5 * (box[:2] + box[2:])
                distance = (anchor_pixels - center).square().sum(dim=1)
                for level in (3, 4, 5):
                    candidates = torch.nonzero(inside & (level_ids == level), as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        continue
                    count = min(int(fallback_min_per_level), int(candidates.numel()))
                    selected = candidates[torch.topk(distance[candidates], k=count, largest=False).indices]
                    target_mask[batch_index, selected] = True
                    labels[batch_index, selected] = int(target_class_id)
                    bboxes[batch_index, selected] = box
                    scores[batch_index, selected] = 1.0
                    gt_indices[batch_index, selected] = gt_index
                    fallback_count += int(selected.numel())

    return TargetPartition(
        unit_mask=target_mask,
        target_labels=labels,
        target_bboxes=bboxes,
        target_scores=scores,
        target_gt_idx=gt_indices,
        tal_positive_count=tal_count,
        fallback_positive_count=fallback_count,
    )

