from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _vectors(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"expected [N,C], got {tuple(values.shape)}")
    return values


def subspace_energies(shift: torch.Tensor, basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift = _vectors(shift)
    q = basis.to(device=shift.device, dtype=shift.dtype)
    coefficients = shift @ q
    projected = coefficients @ q.T
    projected_energy = coefficients.square().sum(dim=1)
    outside_energy = (shift - projected).square().sum(dim=1)
    total_energy = shift.square().sum(dim=1)
    return projected_energy, outside_energy, total_energy


def target_energy_margin_loss(shift: torch.Tensor, basis: torch.Tensor, margin: float) -> torch.Tensor:
    projected, _, _ = subspace_energies(shift, basis)
    return F.relu(float(margin) - projected.clamp_min(1e-12).sqrt()).mean()


def non_target_leakage(shift: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projected, _, _ = subspace_energies(shift, basis)
    return projected.mean()


def symmetric_bernoulli_kl(clean_logits: torch.Tensor, adv_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clean_p = clean_logits.detach().sigmoid().clamp(eps, 1.0 - eps)
    adv_p = adv_logits.sigmoid().clamp(eps, 1.0 - eps)
    kl_ca = clean_p * (clean_p / adv_p).log() + (1 - clean_p) * ((1 - clean_p) / (1 - adv_p)).log()
    kl_ac = adv_p * (adv_p / clean_p).log() + (1 - adv_p) * ((1 - adv_p) / (1 - clean_p)).log()
    return 0.5 * (kl_ca + kl_ac).mean()


def dcss_stage1_loss(
    target_shift: torch.Tensor,
    non_target_shift: torch.Tensor,
    basis: torch.Tensor,
    margin: float,
    clean_non_target_logits: Optional[torch.Tensor] = None,
    adv_non_target_logits: Optional[torch.Tensor] = None,
    regularizer: Optional[torch.Tensor] = None,
    lambda_energy: float = 1.0,
    lambda_outside: float = 1.0,
    lambda_leakage: float = 1.0,
    lambda_logits: float = 1.0,
    lambda_regularizer: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    energy = target_energy_margin_loss(target_shift, basis, margin)
    target_projected, target_outside, target_total = subspace_energies(target_shift, basis)
    leakage = non_target_leakage(non_target_shift, basis) if non_target_shift.numel() else target_shift.new_zeros(())
    if clean_non_target_logits is None or adv_non_target_logits is None:
        logit_loss = target_shift.new_zeros(())
    else:
        logit_loss = symmetric_bernoulli_kl(clean_non_target_logits, adv_non_target_logits)
    reg = regularizer if regularizer is not None else target_shift.new_zeros(())
    total = (
        lambda_energy * energy
        + lambda_outside * target_outside.mean()
        + lambda_leakage * leakage
        + lambda_logits * logit_loss
        + lambda_regularizer * reg
    )
    metrics = {
        "energy_loss": energy,
        "target_projected_energy": target_projected.mean(),
        "target_outside_energy": target_outside.mean(),
        "target_total_energy": target_total.mean(),
        "non_target_leakage": leakage,
        "non_target_logit_loss": logit_loss,
    }
    return total, metrics
