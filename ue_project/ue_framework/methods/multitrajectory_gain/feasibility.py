from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


def summarize_clean_gains(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    materialized = list(rows)
    gains = np.array([float(row["protected_raw_gain"]) for row in materialized], dtype=np.float64)
    authorized = np.array([float(row["authorized_raw_gain"]) for row in materialized], dtype=np.float64)
    shared = np.array([float(row["shared_raw_gain"]) for row in materialized], dtype=np.float64)
    valid = np.array([float(row["protected_valid"]) for row in materialized], dtype=np.float64)
    return {
        "trajectory_count": int(len(materialized)),
        "protected_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "protected_gain_positive_ratio": float((gains > 0.0).mean()) if gains.size else 0.0,
        "protected_gain_mean": float(gains.mean()) if gains.size else 0.0,
        "protected_gain_median": float(np.median(gains)) if gains.size else 0.0,
        "protected_gain_std": float(gains.std()) if gains.size else 0.0,
        "authorized_gain_positive_ratio": float((authorized > 0.0).mean()) if authorized.size else 0.0,
        "shared_gain_std": float(shared.std()) if shared.size else 0.0,
    }


def natural_variation_thresholds(rows: Iterable[Mapping[str, float]], quantile: float = 90.0, kappa_t: float = 2.0) -> Dict[str, float]:
    materialized = list(rows)
    nt = np.array([float(row["N_t"]) for row in materialized], dtype=np.float64)
    na = np.array([float(row["N_a"]) for row in materialized], dtype=np.float64)
    ns = np.array([float(row["N_s"]) for row in materialized], dtype=np.float64)
    tau_t = float(np.percentile(nt, quantile)) if nt.size else 0.0
    tau_a = float(np.percentile(na, quantile)) if na.size else 0.0
    tau_s = float(np.percentile(ns, quantile)) if ns.size else 0.0
    return {
        "tau_t": tau_t,
        "tau_a": tau_a,
        "tau_s": tau_s,
        "protected_margin": float(kappa_t) * tau_t,
        "p50_t": float(np.percentile(nt, 50)) if nt.size else 0.0,
        "p75_t": float(np.percentile(nt, 75)) if nt.size else 0.0,
        "p90_t": tau_t,
        "p95_t": float(np.percentile(nt, 95)) if nt.size else 0.0,
        "p50_a": float(np.percentile(na, 50)) if na.size else 0.0,
        "p75_a": float(np.percentile(na, 75)) if na.size else 0.0,
        "p90_a": tau_a,
        "p95_a": float(np.percentile(na, 95)) if na.size else 0.0,
        "p50_s": float(np.percentile(ns, 50)) if ns.size else 0.0,
        "p75_s": float(np.percentile(ns, 75)) if ns.size else 0.0,
        "p90_s": tau_s,
        "p95_s": float(np.percentile(ns, 95)) if ns.size else 0.0,
    }


def raw_counterfactual_gap(clean_losses: Mapping[str, torch.Tensor], poison_losses: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "Delta_t": poison_losses["protected"] - clean_losses["protected"],
        "Delta_a": poison_losses["authorized"] - clean_losses["authorized"],
        "Delta_s": poison_losses["shared"] - clean_losses["shared"],
    }


def constraint_violations(gaps: Mapping[str, torch.Tensor], thresholds: Mapping[str, float]) -> Dict[str, torch.Tensor]:
    delta_t = gaps["Delta_t"]
    delta_a = gaps["Delta_a"]
    delta_s = gaps["Delta_s"]
    v_t = torch.relu(delta_t.new_tensor(float(thresholds["protected_margin"])) - delta_t)
    v_a = torch.relu(delta_a.abs() - delta_a.new_tensor(float(thresholds["tau_a"])))
    v_s = torch.relu(delta_s.abs() - delta_s.new_tensor(float(thresholds["tau_s"])))
    return {"v_t": v_t, "v_a": v_a, "v_s": v_s}


@dataclass
class DualState:
    mu_authorized: float = 1.0
    mu_shared: float = 1.0

    def update(self, v_a: torch.Tensor | float, v_s: torch.Tensor | float, dual_learning_rate: float, mu_max: float) -> "DualState":
        va = float(v_a.detach().item()) if torch.is_tensor(v_a) else float(v_a)
        vs = float(v_s.detach().item()) if torch.is_tensor(v_s) else float(v_s)
        return DualState(
            mu_authorized=min(float(mu_max), max(0.0, self.mu_authorized + float(dual_learning_rate) * max(0.0, va))),
            mu_shared=min(float(mu_max), max(0.0, self.mu_shared + float(dual_learning_rate) * max(0.0, vs))),
        )


def constrained_objective(
    gaps: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, float],
    mu_authorized: float,
    mu_shared: float,
    regularization: torch.Tensor,
    lambda_regularization: float = 1.0,
) -> Dict[str, torch.Tensor]:
    violations = constraint_violations(gaps, thresholds)
    total = (
        violations["v_t"]
        + float(mu_authorized) * violations["v_a"]
        + float(mu_shared) * violations["v_s"]
        + float(lambda_regularization) * regularization
    )
    return {**violations, "total": total}


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    xs = np.array(list(x), dtype=np.float64)
    ys = np.array(list(y), dtype=np.float64)
    if xs.size != ys.size or xs.size < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(xs.size)}
    return {"rho": pearson_correlation(_rank(xs), _rank(ys))["r"], "p_value": float("nan"), "n": int(xs.size)}


def pearson_correlation(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    xs = np.array(list(x), dtype=np.float64)
    ys = np.array(list(y), dtype=np.float64)
    if xs.size != ys.size or xs.size < 2 or float(xs.std()) == 0.0 or float(ys.std()) == 0.0:
        return {"r": float("nan"), "n": int(xs.size)}
    return {"r": float(np.corrcoef(xs, ys)[0, 1]), "n": int(xs.size)}


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, _value in enumerate(unique):
        if counts[idx] > 1:
            ranks[inverse == idx] = ranks[inverse == idx].mean()
    return ranks
