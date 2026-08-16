from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .dgcaip import DGCAIPResult


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().cpu()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < order.numel():
        end = start + 1
        while end < order.numel() and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float((first_centered @ second_centered) / denominator)


def build_dgcaip_locator_report(
    results: Sequence[DGCAIPResult],
    *,
    correlation_success: float = 0.35,
    correlation_failure: float = 0.20,
    q4_q1_success: float = 1.5,
    q4_q1_failure: float = 1.2,
    coverage_success: float = 0.95,
) -> Dict[str, Any]:
    instances = [term for result in results for term in result.instances]
    eligible = sum(result.eligible_instance_count for result in results)
    covered = sum(result.covered_instance_count for result in results)
    coverage = float(covered / eligible) if eligible else 1.0
    if not instances:
        raise ValueError("DG-CAIP locator requires at least one covered instance.")

    divergence = torch.tensor(
        [float(term.distribution_loss.detach()) for term in instances],
        dtype=torch.float64,
    )
    damage = torch.tensor(
        [
            [
                float(term.classification_loss.detach()),
                float(term.box_loss.detach()),
                float(term.alignment_loss.detach()),
            ]
            for term in instances
        ],
        dtype=torch.float64,
    )
    if not torch.isfinite(divergence).all() or not torch.isfinite(damage).all():
        raise ValueError("DG-CAIP locator inputs must be finite.")
    means = damage.mean(dim=0)
    standard_deviation = damage.std(dim=0, unbiased=False)
    active = standard_deviation > 1.0e-12
    if bool(active.any()):
        standardized = torch.zeros_like(damage)
        standardized[:, active] = (
            damage[:, active] - means[active]
        ) / standard_deviation[active]
        composite = standardized[:, active].mean(dim=1)
    else:
        composite = torch.zeros(damage.shape[0], dtype=torch.float64)
    correlation = _spearman(divergence, composite)

    order = torch.argsort(divergence, stable=True)
    quartile_indices = torch.tensor_split(order, 4)
    quartiles = {}
    for index, indices in enumerate(quartile_indices, start=1):
        if indices.numel() == 0:
            raise ValueError("DG-CAIP locator requires at least four instances.")
        quartiles[f"Q{index}"] = {
            "count": int(indices.numel()),
            "divergence_mean": float(divergence[indices].mean()),
            "classification_damage_mean": float(damage[indices, 0].mean()),
            "box_damage_mean": float(damage[indices, 1].mean()),
            "alignment_damage_mean": float(damage[indices, 2].mean()),
            "composite_damage_mean": float(composite[indices].mean()),
        }
    q1_shifted = quartiles["Q1"]["composite_damage_mean"] - float(composite.min())
    q4_shifted = quartiles["Q4"]["composite_damage_mean"] - float(composite.min())
    if q1_shifted <= 1.0e-12 and q4_shifted <= 1.0e-12:
        q4_q1_ratio = 1.0
    else:
        q4_q1_ratio = q4_shifted / max(q1_shifted, 1.0e-12)

    per_class: Dict[str, Any] = {}
    for class_id in sorted({term.class_id for term in instances}):
        indices = [
            index for index, term in enumerate(instances) if term.class_id == class_id
        ]
        values = divergence[indices]
        per_class[str(class_id)] = {
            "count": len(indices),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(np.quantile(values.numpy(), 0.90)),
        }

    checks = {
        "spearman": correlation >= float(correlation_success),
        "q4_q1_ratio": q4_q1_ratio >= float(q4_q1_success),
        "coverage": coverage >= float(coverage_success),
    }
    failure_checks = {
        "spearman": correlation < float(correlation_failure),
        "q4_q1_ratio": q4_q1_ratio < float(q4_q1_failure),
    }
    if all(checks.values()):
        decision = "pass"
    elif any(failure_checks.values()):
        decision = "fail"
    else:
        decision = "inconclusive"
    return {
        "schema": "tausb.dgcaip-locator.v1",
        "eligible_instance_count": eligible,
        "covered_instance_count": covered,
        "coverage": coverage,
        "spearman_divergence_composite_damage": correlation,
        "q4_q1_composite_damage_ratio": q4_q1_ratio,
        "quartiles": quartiles,
        "per_class": per_class,
        "checks": checks,
        "failure_checks": failure_checks,
        "decision": decision,
        "finite": bool(
            math.isfinite(correlation)
            and math.isfinite(coverage)
            and math.isfinite(q4_q1_ratio)
        ),
    }
