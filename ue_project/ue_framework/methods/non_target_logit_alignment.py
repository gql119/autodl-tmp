from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NonTargetLogitAlignmentResult:
    loss: torch.Tensor
    active_classes: Tuple[int, ...]
    per_class_loss: Dict[int, torch.Tensor]
    per_class_count: Dict[int, int]
    assigned_logit_abs_drift: float
    full_non_target_vector_rms_drift: float


def class_balanced_non_target_logit_alignment(
    clean_logits: torch.Tensor,
    poison_logits: torch.Tensor,
    assigned_labels: torch.Tensor,
    foreground_mask: torch.Tensor,
    *,
    target_class_id: int,
    assignment_source: str,
    beta: float = 1.0,
) -> NonTargetLogitAlignmentResult:
    """Macro-average raw assigned-class logit alignment over real TAL positives."""

    if assignment_source != "clean_real_tal":
        raise ValueError("NLA requires clean_real_tal assignments; pseudo is forbidden.")
    if clean_logits.shape != poison_logits.shape or clean_logits.ndim != 3:
        raise ValueError("Clean/poison logits must align as [B,A,C].")
    if assigned_labels.ndim == 3 and assigned_labels.shape[-1] == 1:
        assigned_labels = assigned_labels[..., 0]
    if foreground_mask.ndim == 3 and foreground_mask.shape[-1] == 1:
        foreground_mask = foreground_mask[..., 0]
    if assigned_labels.shape != clean_logits.shape[:2]:
        raise ValueError("Assigned labels must align with logits as [B,A].")
    if foreground_mask.shape != clean_logits.shape[:2]:
        raise ValueError("Foreground mask must align with logits as [B,A].")
    if not 0 <= int(target_class_id) < clean_logits.shape[-1]:
        raise ValueError("target_class_id is outside the detector class range.")
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive.")
    if not torch.isfinite(clean_logits).all() or not torch.isfinite(poison_logits).all():
        raise ValueError("NLA logits must be finite.")

    labels = assigned_labels.to(device=poison_logits.device).long()
    foreground = foreground_mask.to(device=poison_logits.device).bool()
    if bool(foreground.any()):
        foreground_labels = labels[foreground]
        if bool(
            ((foreground_labels < 0) | (foreground_labels >= clean_logits.shape[-1])).any()
        ):
            raise ValueError("Real TAL foreground contains an invalid assigned class.")
    active_mask = foreground & (labels != int(target_class_id))
    active_classes = tuple(
        int(value)
        for value in torch.unique(labels[active_mask], sorted=True).tolist()
    )
    per_class_loss: Dict[int, torch.Tensor] = {}
    per_class_count: Dict[int, int] = {}
    assigned_drifts = []
    vector_drifts = []
    non_target_indices = torch.arange(
        clean_logits.shape[-1], device=poison_logits.device
    ) != int(target_class_id)

    for class_id in active_classes:
        class_mask = active_mask & (labels == class_id)
        clean_assigned = clean_logits[..., class_id][class_mask].detach()
        poison_assigned = poison_logits[..., class_id][class_mask]
        class_loss = F.smooth_l1_loss(
            poison_assigned,
            clean_assigned,
            reduction="mean",
            beta=float(beta),
        )
        per_class_loss[class_id] = class_loss
        per_class_count[class_id] = int(class_mask.sum().item())
        assigned_drifts.append((poison_assigned.detach() - clean_assigned).abs().mean())
        clean_vector = clean_logits[class_mask][:, non_target_indices].detach()
        poison_vector = poison_logits[class_mask][:, non_target_indices].detach()
        vector_drifts.append(
            (poison_vector - clean_vector).square().mean().sqrt()
        )

    if per_class_loss:
        loss = torch.stack(tuple(per_class_loss.values())).mean()
        assigned_drift = float(torch.stack(assigned_drifts).mean().cpu())
        vector_drift = float(torch.stack(vector_drifts).mean().cpu())
    else:
        loss = poison_logits.sum() * 0.0
        assigned_drift = 0.0
        vector_drift = 0.0
    return NonTargetLogitAlignmentResult(
        loss=loss,
        active_classes=active_classes,
        per_class_loss=per_class_loss,
        per_class_count=per_class_count,
        assigned_logit_abs_drift=assigned_drift,
        full_non_target_vector_rms_drift=vector_drift,
    )


class FrozenNLAGradientCalibration:
    """One-shot warm-up calibration for lambda_NLA.

    The scale targets a fixed fraction of the already-projected target gradient
    norm and is never adapted from AP50 or held-out observations.
    """

    def __init__(
        self,
        *,
        target_ratio: float = 0.25,
        min_lambda: float = 1.0e-4,
        max_lambda: float = 100.0,
        epsilon: float = 1.0e-12,
    ) -> None:
        if target_ratio <= 0 or min_lambda <= 0 or max_lambda < min_lambda:
            raise ValueError("Invalid NLA calibration bounds.")
        self.target_ratio = float(target_ratio)
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self.epsilon = float(epsilon)
        self._value: Optional[float] = None
        self._was_clipped: Optional[bool] = None
        self._warmup_count = 0

    @property
    def value(self) -> float:
        if self._value is None:
            raise RuntimeError("NLA lambda is not calibrated.")
        return self._value

    @property
    def was_clipped(self) -> bool:
        if self._was_clipped is None:
            raise RuntimeError("NLA lambda is not calibrated.")
        return self._was_clipped

    def calibrate(
        self,
        projected_target_grad_norms: Sequence[float],
        nla_grad_norms: Sequence[float],
        *,
        split: str,
    ) -> float:
        if split not in ("warmup", "train_calibration"):
            raise ValueError("NLA lambda may only use the warm-up calibration split.")
        if self._value is not None:
            raise RuntimeError("NLA lambda calibration is already frozen.")
        if len(projected_target_grad_norms) != len(nla_grad_norms) or not nla_grad_norms:
            raise ValueError("NLA calibration norm sequences must be non-empty and aligned.")
        target = torch.as_tensor(projected_target_grad_norms, dtype=torch.float64)
        nla = torch.as_tensor(nla_grad_norms, dtype=torch.float64)
        valid = torch.isfinite(target) & torch.isfinite(nla) & (target > 0) & (nla > 0)
        if not bool(valid.any()):
            raise ValueError("NLA calibration has no finite positive gradient pairs.")
        raw = self.target_ratio * float(target[valid].median() / nla[valid].median())
        clipped = min(max(raw, self.min_lambda), self.max_lambda)
        self._value = float(clipped)
        self._was_clipped = bool(clipped != raw)
        self._warmup_count = int(valid.sum().item())
        return self._value

    def state_dict(self) -> Dict[str, object]:
        if self._value is None or self._was_clipped is None:
            raise RuntimeError("Cannot serialize uncalibrated NLA lambda.")
        return {
            "target_ratio": self.target_ratio,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "value": self._value,
            "was_clipped": self._was_clipped,
            "warmup_count": self._warmup_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if self._value is not None:
            raise RuntimeError("Cannot overwrite frozen NLA calibration.")
        value = float(state["value"])
        if not self.min_lambda <= value <= self.max_lambda:
            raise ValueError("Serialized NLA lambda is outside configured bounds.")
        self._value = value
        self._was_clipped = bool(state["was_clipped"])
        self._warmup_count = int(state["warmup_count"])
