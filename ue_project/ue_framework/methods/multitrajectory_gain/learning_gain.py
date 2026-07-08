from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LearningGainMetrics:
    protected_clean_gain: torch.Tensor
    protected_poison_gain: torch.Tensor
    authorized_clean_gain: torch.Tensor
    authorized_poison_gain: torch.Tensor
    shared_clean_gain: torch.Tensor
    shared_poison_gain: torch.Tensor
    d_protected: torch.Tensor
    e_authorized: torch.Tensor
    e_shared: torch.Tensor
    protected_valid: bool
    authorized_valid: bool
    shared_valid: bool
    protected_loss: torch.Tensor
    authorized_loss: torch.Tensor
    shared_loss: torch.Tensor
    total_loss: torch.Tensor
    statistics: Dict[str, float]


def compute_learning_gain_objective(
    initial_losses: Dict[str, torch.Tensor],
    clean_losses: Dict[str, torch.Tensor],
    poison_losses: Dict[str, torch.Tensor],
    query_counts: Dict[str, float],
    protected_margin: float = 0.10,
    protected_clean_gain_min: float = 1.0e-4,
    gain_denominator_floor: float = 1.0e-4,
    lambda_protected: float = 1.0,
    lambda_authorized: float = 1.0,
    lambda_shared: float = 1.0,
) -> LearningGainMetrics:
    zero = poison_losses["protected"] * 0.0

    g_t_c = (initial_losses["protected"] - clean_losses["protected"]).detach()
    g_a_c = (initial_losses["authorized"] - clean_losses["authorized"]).detach()
    g_s_c = (initial_losses["shared"] - clean_losses["shared"]).detach()
    g_t_p = initial_losses["protected"].detach() - poison_losses["protected"]
    g_a_p = initial_losses["authorized"].detach() - poison_losses["authorized"]
    g_s_p = initial_losses["shared"].detach() - poison_losses["shared"]

    floor = float(gain_denominator_floor)
    d_t = (g_t_c - g_t_p) / torch.maximum(g_t_c.detach().abs(), g_t_c.new_tensor(floor))
    e_a_signed = (g_a_p - g_a_c) / torch.maximum(g_a_c.detach().abs(), g_a_c.new_tensor(floor))
    e_s_signed = (g_s_p - g_s_c) / torch.maximum(g_s_c.detach().abs(), g_s_c.new_tensor(floor))
    e_a = e_a_signed.abs()
    e_s = e_s_signed.abs()

    protected_valid = bool(query_counts.get("protected_positive_count", 0.0) > 0 and float(g_t_c.detach().item()) > protected_clean_gain_min)
    authorized_valid = bool(query_counts.get("authorized_positive_count", 0.0) > 0)
    shared_valid = bool(query_counts.get("shared_positive_count", 0.0) > 0 or query_counts.get("background_count", 0.0) > 0)

    protected_loss = F.relu(d_t.new_tensor(float(protected_margin)) - d_t) if protected_valid else zero
    authorized_loss = F.smooth_l1_loss(e_a_signed, torch.zeros_like(e_a_signed), reduction="mean") if authorized_valid else zero
    shared_loss = F.smooth_l1_loss(e_s_signed, torch.zeros_like(e_s_signed), reduction="mean") if shared_valid else zero
    total = float(lambda_protected) * protected_loss + float(lambda_authorized) * authorized_loss + float(lambda_shared) * shared_loss
    stats = {
        "protected_clean_gain": float(g_t_c.detach().item()),
        "protected_poison_gain": float(g_t_p.detach().item()),
        "authorized_clean_gain": float(g_a_c.detach().item()),
        "authorized_poison_gain": float(g_a_p.detach().item()),
        "shared_clean_gain": float(g_s_c.detach().item()),
        "shared_poison_gain": float(g_s_p.detach().item()),
        "d_protected": float(d_t.detach().item()),
        "e_authorized": float(e_a.detach().item()),
        "e_shared": float(e_s.detach().item()),
        "s_gain": float((d_t - e_a - e_s).detach().item()),
        "protected_valid": float(protected_valid),
        "authorized_valid": float(authorized_valid),
        "shared_valid": float(shared_valid),
        "protected_loss": float(protected_loss.detach().item()),
        "authorized_loss": float(authorized_loss.detach().item()),
        "shared_loss": float(shared_loss.detach().item()),
    }
    return LearningGainMetrics(
        protected_clean_gain=g_t_c,
        protected_poison_gain=g_t_p,
        authorized_clean_gain=g_a_c,
        authorized_poison_gain=g_a_p,
        shared_clean_gain=g_s_c,
        shared_poison_gain=g_s_p,
        d_protected=d_t,
        e_authorized=e_a,
        e_shared=e_s,
        protected_valid=protected_valid,
        authorized_valid=authorized_valid,
        shared_valid=shared_valid,
        protected_loss=protected_loss,
        authorized_loss=authorized_loss,
        shared_loss=shared_loss,
        total_loss=total,
        statistics=stats,
    )
