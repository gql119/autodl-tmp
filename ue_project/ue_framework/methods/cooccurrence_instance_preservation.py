from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .non_target_distribution_divergence import (
    non_target_bernoulli_divergence,
)


@dataclass(frozen=True)
class CooccurringNonTargetInstance:
    batch_index: int
    gt_index: int
    class_id: int
    positive_count: int
    js_divergence: torch.Tensor
    clean_to_poison_kl: torch.Tensor


@dataclass(frozen=True)
class CooccurringInstanceCollection:
    loss: torch.Tensor
    instances: Tuple[CooccurringNonTargetInstance, ...]
    active_classes: Tuple[int, ...]
    per_class_loss: Dict[int, torch.Tensor]
    per_class_instance_count: Dict[int, int]
    eligible_instance_count: int
    covered_instance_count: int
    skipped_no_positive_count: int
    coverage: float


def _squeeze_last(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must align as a two-dimensional tensor.")
    return tensor


def collect_cooccurring_non_target_instances(
    clean_logits: torch.Tensor,
    poison_logits: torch.Tensor,
    assigned_labels: torch.Tensor,
    foreground_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    gt_labels: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    target_class_id: int,
    assignment_source: str,
    temperature: float = 2.0,
    epsilon: float = 1.0e-6,
) -> CooccurringInstanceCollection:
    """Collect non-target GT instances from person-cooccurring images.

    Clean real-TAL assignments define one immutable anchor set for both views.
    The poison branch is never reassigned.
    """

    if assignment_source != "clean_real_tal":
        raise ValueError("Cooccurrence preservation requires clean_real_tal assignments.")
    if clean_logits.shape != poison_logits.shape or clean_logits.ndim != 3:
        raise ValueError("Clean/poison logits must align as [B,A,C].")
    assigned_labels = _squeeze_last(assigned_labels, "assigned_labels")
    foreground_mask = _squeeze_last(foreground_mask, "foreground_mask")
    target_gt_idx = _squeeze_last(target_gt_idx, "target_gt_idx")
    gt_labels = _squeeze_last(gt_labels, "gt_labels")
    mask_gt = _squeeze_last(mask_gt, "mask_gt")
    if assigned_labels.shape != clean_logits.shape[:2]:
        raise ValueError("assigned_labels must align with logits as [B,A].")
    if foreground_mask.shape != clean_logits.shape[:2]:
        raise ValueError("foreground_mask must align with logits as [B,A].")
    if target_gt_idx.shape != clean_logits.shape[:2]:
        raise ValueError("target_gt_idx must align with logits as [B,A].")
    if gt_labels.shape != mask_gt.shape or gt_labels.shape[0] != clean_logits.shape[0]:
        raise ValueError("gt_labels and mask_gt must align as [B,M].")
    class_count = int(clean_logits.shape[-1])
    if not 0 <= int(target_class_id) < class_count:
        raise ValueError("target_class_id is outside the detector class range.")

    device = poison_logits.device
    assigned_labels = assigned_labels.to(device=device).long()
    foreground_mask = foreground_mask.to(device=device).bool()
    target_gt_idx = target_gt_idx.to(device=device).long()
    gt_labels = gt_labels.to(device=device).long()
    mask_gt = mask_gt.to(device=device).bool()
    valid_gt_labels = gt_labels[mask_gt]
    if bool(
        ((valid_gt_labels < 0) | (valid_gt_labels >= class_count)).any()
    ):
        raise ValueError("Valid GT contains a class outside the detector range.")
    if bool(foreground_mask.any()):
        foreground_gt_idx = target_gt_idx[foreground_mask]
        if bool(
            ((foreground_gt_idx < 0) | (foreground_gt_idx >= gt_labels.shape[1])).any()
        ):
            raise ValueError("Foreground assignment contains an invalid GT index.")
        batch_indices = torch.arange(
            clean_logits.shape[0], device=device
        ).unsqueeze(1).expand_as(target_gt_idx)
        safe_gt_idx = target_gt_idx.clamp(0, gt_labels.shape[1] - 1)
        assigned_gt_is_valid = mask_gt[batch_indices, safe_gt_idx]
        if not bool(assigned_gt_is_valid[foreground_mask].all()):
            raise ValueError("Foreground assignment points to a masked GT slot.")

    divergence = non_target_bernoulli_divergence(
        clean_logits,
        poison_logits,
        target_class_id=target_class_id,
        temperature=temperature,
        epsilon=epsilon,
    )
    records = []
    eligible_count = 0
    skipped_count = 0
    for batch_index in range(clean_logits.shape[0]):
        valid_indices = torch.nonzero(mask_gt[batch_index], as_tuple=False).flatten()
        valid_classes = gt_labels[batch_index, valid_indices]
        if not bool((valid_classes == int(target_class_id)).any()):
            continue
        for gt_index_tensor in valid_indices:
            gt_index = int(gt_index_tensor.item())
            class_id = int(gt_labels[batch_index, gt_index].item())
            if class_id == int(target_class_id):
                continue
            eligible_count += 1
            positive_mask = foreground_mask[batch_index] & (
                target_gt_idx[batch_index] == gt_index
            )
            positive_count = int(positive_mask.sum().item())
            if positive_count == 0:
                skipped_count += 1
                continue
            positive_labels = assigned_labels[batch_index, positive_mask]
            if not bool((positive_labels == class_id).all()):
                raise ValueError("Clean TAL assigned class disagrees with its GT instance.")
            records.append(
                CooccurringNonTargetInstance(
                    batch_index=batch_index,
                    gt_index=gt_index,
                    class_id=class_id,
                    positive_count=positive_count,
                    js_divergence=divergence.js_per_anchor[
                        batch_index, positive_mask
                    ].mean(),
                    clean_to_poison_kl=divergence.clean_to_poison_kl_per_anchor[
                        batch_index, positive_mask
                    ].mean(),
                )
            )

    per_class_values: Dict[int, list[torch.Tensor]] = {}
    for record in records:
        per_class_values.setdefault(record.class_id, []).append(record.js_divergence)
    per_class_loss = {
        class_id: torch.stack(values).mean()
        for class_id, values in sorted(per_class_values.items())
    }
    per_class_count = {
        class_id: len(values)
        for class_id, values in sorted(per_class_values.items())
    }
    if per_class_loss:
        loss = torch.stack(tuple(per_class_loss.values())).mean()
    else:
        loss = poison_logits.sum() * 0.0
    covered_count = len(records)
    coverage = float(covered_count / eligible_count) if eligible_count else 1.0
    return CooccurringInstanceCollection(
        loss=loss,
        instances=tuple(records),
        active_classes=tuple(per_class_loss),
        per_class_loss=per_class_loss,
        per_class_instance_count=per_class_count,
        eligible_instance_count=eligible_count,
        covered_instance_count=covered_count,
        skipped_no_positive_count=skipped_count,
        coverage=coverage,
    )
