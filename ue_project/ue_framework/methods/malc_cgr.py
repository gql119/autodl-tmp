from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from .constraint_gradient_router import (
    ConstraintTerm,
    GradientRouteResult,
    backtracking_candidate,
    route_coefficient_gradient,
)
from .shadow_tal import NonTargetConstraintSet


@dataclass(frozen=True)
class MALCCGRUpdateResult:
    candidate: torch.Tensor
    accepted: bool
    selected_mode: str
    attempts: int
    route: GradientRouteResult
    class_values_before: dict[str, float]
    class_values_after: dict[str, float]
    box_margin_monitor: dict[str, float]


def class_probability_constraint_terms(
    constraint_set: NonTargetConstraintSet,
    *,
    tolerance: float = 0.005,
) -> tuple[ConstraintTerm, ...]:
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("CGR class tolerance must be finite and non-negative.")
    terms = []
    for item in constraint_set.constraints:
        if item.class_id < 0:
            raise ValueError("Non-target constraint class ids must be non-negative.")
        terms.append(
            ConstraintTerm(
                name=f"class_{item.class_id}_cls",
                margin=item.cls_margin,
                tolerance=float(tolerance),
            )
        )
    if len({term.name for term in terms}) != len(terms):
        raise ValueError("Non-target constraints contain duplicate classes.")
    return tuple(terms)


def route_malc_cgr_update(
    *,
    parameter: torch.Tensor,
    target_loss: torch.Tensor,
    constraint_set: NonTargetConstraintSet,
    step_size: float,
    evaluate_class_margins: Callable[[torch.Tensor], Mapping[str, float]],
    tolerance: float = 0.005,
    near_boundary: float = 0.005,
    svd_relative_tolerance: float = 1e-4,
    max_backtracks: int = 5,
) -> MALCCGRUpdateResult:
    if not math.isfinite(float(step_size)) or step_size <= 0:
        raise ValueError("CGR step_size must be positive and finite.")
    if max_backtracks != 5:
        raise ValueError("MALC-CGR max_backtracks must remain exactly 5.")
    terms = class_probability_constraint_terms(
        constraint_set,
        tolerance=tolerance,
    )
    routed = route_coefficient_gradient(
        parameter=parameter,
        target_loss=target_loss,
        constraints=terms,
        near_boundary=near_boundary,
        svd_relative_tolerance=svd_relative_tolerance,
    )
    before = {
        term.name: float(term.margin.detach())
        for term in terms
    }
    box_monitor = {
        f"class_{item.class_id}_box": float(item.box_margin.detach())
        for item in constraint_set.constraints
    }
    if routed.mode == "skip":
        return MALCCGRUpdateResult(
            candidate=parameter.detach().clone(),
            accepted=False,
            selected_mode="skip",
            attempts=0,
            route=routed,
            class_values_before=before,
            class_values_after=before.copy(),
            box_margin_monitor=box_monitor,
        )

    if not terms:
        candidate = parameter.detach() - step_size * routed.gradient.detach()
        if not torch.isfinite(candidate).all():
            raise ValueError("CGR candidate contains non-finite coefficients.")
        return MALCCGRUpdateResult(
            candidate=candidate,
            accepted=True,
            selected_mode=routed.mode,
            attempts=1,
            route=routed,
            class_values_before={},
            class_values_after={},
            box_margin_monitor=box_monitor,
        )

    limits = {term.name: float(term.tolerance) for term in terms}
    backtracked = backtracking_candidate(
        parameter=parameter,
        gradient=routed.gradient,
        step_size=step_size,
        evaluate_constraints=evaluate_class_margins,
        limits=limits,
        mode="repair" if routed.mode == "repair_only" else "feasible",
        baseline_values=before if routed.mode == "repair_only" else None,
        max_backtracks=max_backtracks,
    )
    return MALCCGRUpdateResult(
        candidate=backtracked.candidate,
        accepted=backtracked.accepted,
        selected_mode=routed.mode if backtracked.accepted else "skip",
        attempts=backtracked.attempts,
        route=routed,
        class_values_before=before,
        class_values_after=backtracked.values,
        box_margin_monitor=box_monitor,
    )
