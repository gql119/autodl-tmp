from __future__ import annotations

from typing import Dict

import torch

from .gradient_extractor import GradientVector


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sum(a * b) / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))


def build_p1_trajectory_loss(
    protected_clean: GradientVector,
    protected_poison: GradientVector,
    authorized_clean: GradientVector,
    authorized_poison: GradientVector,
    lambda_protected: float,
    lambda_authorized: float,
    use_protected: bool = True,
    use_authorized: bool = True,
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    zero = protected_poison.vector.sum() * 0.0 + authorized_poison.vector.sum() * 0.0

    if use_protected:
        cos_protected = _cosine(protected_poison.vector, protected_clean.vector.detach(), eps)
        protected_loss = cos_protected
    else:
        cos_protected = zero
        protected_loss = zero

    if use_authorized:
        cos_authorized = _cosine(authorized_poison.vector, authorized_clean.vector.detach(), eps)
        authorized_loss = 1.0 - cos_authorized
    else:
        cos_authorized = zero
        authorized_loss = zero

    total = float(lambda_protected) * protected_loss + float(lambda_authorized) * authorized_loss
    return {
        "loss": total,
        "protected_traj_loss": protected_loss,
        "authorized_traj_loss": authorized_loss,
        "cos_protected_clean_poison": cos_protected,
        "cos_authorized_clean_poison": cos_authorized,
    }
