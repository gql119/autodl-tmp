from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass
class TargetGainResult:
    clean_gain: torch.Tensor
    poison_gain: torch.Tensor
    ratio: torch.Tensor | None
    protect_loss: torch.Tensor
    valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class ClassGainInput:
    initial_loss: torch.Tensor
    clean_updated_loss: torch.Tensor
    poison_updated_loss: torch.Tensor
    support_count: int
    query_count: int


@dataclass
class ClassGainResult:
    clean_gain: torch.Tensor
    poison_gain: torch.Tensor
    normalized_gap: torch.Tensor | None
    loss: torch.Tensor
    valid: bool
    invalid_reason: str
    support_count: int
    query_count: int


@dataclass
class AuthorizedGainResult:
    loss: torch.Tensor
    classes: dict[int, ClassGainResult]
    valid_class_ids: tuple[int, ...]
    invalid_class_ids: tuple[int, ...]


def _finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in values)


def target_learning_gain(
    initial_loss: torch.Tensor,
    clean_updated_loss: torch.Tensor,
    poison_updated_loss: torch.Tensor,
    rho_t: float,
    min_valid_clean_gain: float,
    eps: float = 1e-8,
) -> TargetGainResult:
    clean_gain = initial_loss - clean_updated_loss
    poison_gain = initial_loss - poison_updated_loss
    zero = (initial_loss + clean_updated_loss + poison_updated_loss) * 0.0
    if not _finite(clean_gain, poison_gain):
        return TargetGainResult(clean_gain, poison_gain, None, zero, False, "non_finite_gain")
    if float(clean_gain.detach().item()) <= float(min_valid_clean_gain):
        return TargetGainResult(clean_gain, poison_gain, None, zero, False, "invalid_clean_gain")
    ratio = poison_gain / (clean_gain.abs() + float(eps))
    protect = F.relu(ratio - float(rho_t))
    return TargetGainResult(clean_gain, poison_gain, ratio, protect, True, "")


def authorized_learning_gain(
    class_inputs: Mapping[int, ClassGainInput],
    target_class_id: int,
    rho_k: float | Mapping[int, float],
    min_valid_class_gain: float,
    minimum_class_samples: int,
    eps: float = 1e-8,
) -> AuthorizedGainResult:
    results: dict[int, ClassGainResult] = {}
    losses: list[torch.Tensor] = []
    zero_reference: torch.Tensor | None = None
    for class_id, values in sorted(class_inputs.items()):
        if int(class_id) == int(target_class_id):
            continue
        zero = (values.initial_loss + values.clean_updated_loss + values.poison_updated_loss) * 0.0
        zero_reference = zero if zero_reference is None else zero_reference + zero
        clean_gain = values.initial_loss - values.clean_updated_loss
        poison_gain = values.initial_loss - values.poison_updated_loss
        reason = ""
        if values.support_count < int(minimum_class_samples):
            reason = "insufficient_support_samples"
        elif values.query_count < int(minimum_class_samples):
            reason = "insufficient_query_samples"
        elif not _finite(clean_gain, poison_gain):
            reason = "non_finite_gain"
        elif abs(float(clean_gain.detach().item())) < float(min_valid_class_gain):
            reason = "clean_gain_below_threshold"
        if reason:
            results[int(class_id)] = ClassGainResult(
                clean_gain, poison_gain, None, zero, False, reason, values.support_count, values.query_count
            )
            continue
        tolerance = float(rho_k.get(int(class_id), 0.0)) if isinstance(rho_k, Mapping) else float(rho_k)
        gap = (poison_gain - clean_gain).abs() / (clean_gain.abs() + float(eps))
        loss = F.relu(gap - tolerance)
        losses.append(loss)
        results[int(class_id)] = ClassGainResult(
            clean_gain, poison_gain, gap, loss, True, "", values.support_count, values.query_count
        )
    if losses:
        total = torch.stack(losses).sum()
    elif zero_reference is not None:
        total = zero_reference
    else:
        total = torch.tensor(0.0)
    valid_ids = tuple(class_id for class_id, result in results.items() if result.valid)
    invalid_ids = tuple(class_id for class_id, result in results.items() if not result.valid)
    return AuthorizedGainResult(total, results, valid_ids, invalid_ids)


def carrier_query_loss(target_poison_query_loss: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(target_poison_query_loss.detach()).all()):
        raise FloatingPointError("non-finite carrier query loss")
    return target_poison_query_loss

