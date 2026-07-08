from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

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
    stats = _raw_gain_statistics(
        initial_losses,
        clean_losses,
        poison_losses,
        g_t_c,
        g_t_p,
        g_a_c,
        g_a_p,
        g_s_c,
        g_s_p,
        denominators={"protected": torch.maximum(g_t_c.detach().abs(), g_t_c.new_tensor(floor)), "authorized": torch.maximum(g_a_c.detach().abs(), g_a_c.new_tensor(floor)), "shared": torch.maximum(g_s_c.detach().abs(), g_s_c.new_tensor(floor))},
    )
    stats.update(
        {
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
        "objective_version": "v1",
        }
    )
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


def _raw_gain_statistics(
    initial_losses: Mapping[str, torch.Tensor],
    clean_losses: Mapping[str, torch.Tensor],
    poison_losses: Mapping[str, torch.Tensor],
    g_t_c: torch.Tensor,
    g_t_p: torch.Tensor,
    g_a_c: torch.Tensor,
    g_a_p: torch.Tensor,
    g_s_c: torch.Tensor,
    g_s_p: torch.Tensor,
    denominators: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        "L_t_before": float(initial_losses["protected"].detach().item()),
        "L_t_after_clean": float(clean_losses["protected"].detach().item()),
        "L_t_after_poison": float(poison_losses["protected"].detach().item()),
        "G_t_clean": float(g_t_c.detach().item()),
        "G_t_poison": float(g_t_p.detach().item()),
        "G_t_clean_minus_poison": float((g_t_c - g_t_p).detach().item()),
        "L_a_before": float(initial_losses["authorized"].detach().item()),
        "L_a_after_clean": float(clean_losses["authorized"].detach().item()),
        "L_a_after_poison": float(poison_losses["authorized"].detach().item()),
        "G_a_clean": float(g_a_c.detach().item()),
        "G_a_poison": float(g_a_p.detach().item()),
        "G_a_poison_minus_clean": float((g_a_p - g_a_c).detach().item()),
        "L_s_before": float(initial_losses["shared"].detach().item()),
        "L_s_after_clean": float(clean_losses["shared"].detach().item()),
        "L_s_after_poison": float(poison_losses["shared"].detach().item()),
        "G_s_clean": float(g_s_c.detach().item()),
        "G_s_poison": float(g_s_p.detach().item()),
        "G_s_poison_minus_clean": float((g_s_p - g_s_c).detach().item()),
        "denominator_t": float(denominators["protected"].detach().item()),
        "denominator_a": float(denominators["authorized"].detach().item()),
        "denominator_s": float(denominators["shared"].detach().item()),
    }


def compute_learning_gain_objective_v2(
    initial_losses: Dict[str, torch.Tensor],
    clean_losses: Dict[str, torch.Tensor],
    poison_losses: Dict[str, torch.Tensor],
    query_counts: Dict[str, float],
    support_counts: Dict[str, float],
    robust_scales: Mapping[str, torch.Tensor | float],
    protected_margin: float = 0.10,
    authorized_tolerance: float = 0.10,
    shared_tolerance: float = 0.10,
    protected_clean_gain_min: float = 1.0e-4,
    authorized_clean_gain_min: float = 1.0e-4,
    shared_clean_gain_min: float = 1.0e-4,
    lambda_protected: float = 1.0,
    lambda_authorized: float = 2.0,
    lambda_shared: float = 2.0,
    protected_support_min_batches: int = 2,
) -> LearningGainMetrics:
    zero = poison_losses["protected"] * 0.0

    g_t_c = (initial_losses["protected"] - clean_losses["protected"]).detach()
    g_a_c = (initial_losses["authorized"] - clean_losses["authorized"]).detach()
    g_s_c = (initial_losses["shared"] - clean_losses["shared"]).detach()
    g_t_p = initial_losses["protected"].detach() - poison_losses["protected"]
    g_a_p = initial_losses["authorized"].detach() - poison_losses["authorized"]
    g_s_p = initial_losses["shared"].detach() - poison_losses["shared"]

    scale_t = _scale_tensor(robust_scales["protected"], g_t_c)
    scale_a = _scale_tensor(robust_scales["authorized"], g_a_c)
    scale_s = _scale_tensor(robust_scales["shared"], g_s_c)
    delta_t = (g_t_c - g_t_p) / scale_t
    delta_a_signed = (g_a_p - g_a_c) / scale_a
    delta_s_signed = (g_s_p - g_s_c) / scale_s
    delta_a = delta_a_signed.abs()
    delta_s = delta_s_signed.abs()

    protected_valid = _finite_positive(query_counts.get("protected_positive_count", 0.0)) and support_counts.get("protected_support_batches", 0.0) >= float(protected_support_min_batches) and _finite_gain(g_t_c) and float(g_t_c.detach().item()) > float(protected_clean_gain_min)
    authorized_valid = _finite_positive(query_counts.get("authorized_positive_count", 0.0)) and support_counts.get("authorized_support_batches", 0.0) > 0.0 and _finite_gain(g_a_c) and abs(float(g_a_c.detach().item())) > float(authorized_clean_gain_min)
    shared_valid = (query_counts.get("shared_positive_count", 0.0) > 0.0 or query_counts.get("background_count", 0.0) > 0.0) and _finite_gain(g_s_c) and abs(float(g_s_c.detach().item())) > float(shared_clean_gain_min)

    protected_loss = torch.nn.functional.softplus(delta_t.new_tensor(float(protected_margin)) - delta_t) if protected_valid else zero
    authorized_loss = torch.relu(delta_a - delta_a.new_tensor(float(authorized_tolerance))) if authorized_valid else zero
    shared_loss = torch.relu(delta_s - delta_s.new_tensor(float(shared_tolerance))) if shared_valid else zero
    total = float(lambda_protected) * protected_loss + float(lambda_authorized) * authorized_loss + float(lambda_shared) * shared_loss

    stats = _raw_gain_statistics(
        initial_losses,
        clean_losses,
        poison_losses,
        g_t_c,
        g_t_p,
        g_a_c,
        g_a_p,
        g_s_c,
        g_s_p,
        denominators={"protected": scale_t, "authorized": scale_a, "shared": scale_s},
    )
    stats.update(
        {
            "protected_clean_gain": float(g_t_c.detach().item()),
            "protected_poison_gain": float(g_t_p.detach().item()),
            "authorized_clean_gain": float(g_a_c.detach().item()),
            "authorized_poison_gain": float(g_a_p.detach().item()),
            "shared_clean_gain": float(g_s_c.detach().item()),
            "shared_poison_gain": float(g_s_p.detach().item()),
            "d_protected": float(delta_t.detach().item()),
            "e_authorized": float(delta_a.detach().item()),
            "e_shared": float(delta_s.detach().item()),
            "normalized_D_t": float(delta_t.detach().item()),
            "normalized_E_a": float(delta_a.detach().item()),
            "normalized_E_s": float(delta_s.detach().item()),
            "delta_a_signed": float(delta_a_signed.detach().item()),
            "delta_s_signed": float(delta_s_signed.detach().item()),
            "s_gain": float((delta_t - torch.relu(delta_a - delta_a.new_tensor(float(authorized_tolerance))) - torch.relu(delta_s - delta_s.new_tensor(float(shared_tolerance)))).detach().item()),
            "protected_valid": float(protected_valid),
            "authorized_valid": float(authorized_valid),
            "shared_valid": float(shared_valid),
            "protected_support_batches": float(support_counts.get("protected_support_batches", 0.0)),
            "authorized_support_batches": float(support_counts.get("authorized_support_batches", 0.0)),
            "protected_loss": float(protected_loss.detach().item()),
            "authorized_loss": float(authorized_loss.detach().item()),
            "shared_loss": float(shared_loss.detach().item()),
            "objective_version": "v2",
        }
    )
    return LearningGainMetrics(
        protected_clean_gain=g_t_c,
        protected_poison_gain=g_t_p,
        authorized_clean_gain=g_a_c,
        authorized_poison_gain=g_a_p,
        shared_clean_gain=g_s_c,
        shared_poison_gain=g_s_p,
        d_protected=delta_t,
        e_authorized=delta_a,
        e_shared=delta_s,
        protected_valid=protected_valid,
        authorized_valid=authorized_valid,
        shared_valid=shared_valid,
        protected_loss=protected_loss,
        authorized_loss=authorized_loss,
        shared_loss=shared_loss,
        total_loss=total,
        statistics=stats,
    )


def _scale_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(float(value), device=reference.device, dtype=reference.dtype).detach()


def _finite_gain(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _finite_positive(value: float) -> bool:
    return bool(np_isfinite(value) and float(value) > 0.0)


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
