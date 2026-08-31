from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import torch

from .constraint_gradient_router import (
    MultiParameterBacktrackingResult,
    StrictConstrainedRouteResult,
    backtrack_mixed_multi_parameter_constraints,
    route_strict_final_update,
)
from .sdh_mechanism import SDHObservation


@dataclass(frozen=True)
class StrictDGCAIPStepResult:
    route: StrictConstrainedRouteResult
    backtracking: MultiParameterBacktrackingResult
    safe_limits: Mapping[str, float]
    violated_baselines: Mapping[str, float]


def strict_constraint_losses(
    observation: SDHObservation,
    *,
    epsilon: float = 1.0e-12,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split class protection losses using pre-update structural violations."""

    if observation.dgcaip is None:
        raise ValueError("Strict DG-CAIP routing requires an instance observation.")
    class_ids = sorted(
        set(observation.nla.per_class_loss).union(
            observation.dgcaip.per_class_loss
        )
    )
    zero = observation.nla.loss * 0.0 + observation.dgcaip.loss * 0.0
    safe: Dict[str, torch.Tensor] = {}
    violated: Dict[str, torch.Tensor] = {}
    for class_id in class_ids:
        combined = (
            observation.nla.per_class_loss.get(class_id, zero)
            + observation.dgcaip.per_class_loss.get(class_id, zero)
        )
        class_terms = tuple(
            term
            for term in observation.dgcaip.instances
            if term.class_id == class_id
        )
        structurally_violated = any(
            float(value.detach()) > epsilon
            for term in class_terms
            for value in (
                term.classification_loss,
                term.box_loss,
                term.alignment_loss,
            )
        )
        destination = violated if structurally_violated else safe
        destination[str(class_id)] = combined
    return safe, violated


def multi_snapshot_constraint_losses(
    observations: Mapping[str, SDHObservation],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Keep each frozen protection snapshot as an independently auditable row."""

    if not observations or any(not str(name).strip() for name in observations):
        raise ValueError("Strict DG-CAIP requires named protection snapshots.")
    safe: Dict[str, torch.Tensor] = {}
    violated: Dict[str, torch.Tensor] = {}
    for snapshot_id in sorted(observations):
        snapshot_safe, snapshot_violated = strict_constraint_losses(
            observations[snapshot_id]
        )
        for destination, rows in (
            (safe, snapshot_safe),
            (violated, snapshot_violated),
        ):
            for class_id, loss in rows.items():
                name = "%s/%s" % (snapshot_id, class_id)
                if name in safe or name in violated:
                    raise ValueError("Duplicate strict protection constraint row.")
                destination[name] = loss
    if not safe and not violated:
        raise ValueError("Protection snapshots produced no strict constraint rows.")
    return safe, violated


def partition_nonlinear_constraints(
    current_metrics: Mapping[str, float],
    *,
    js_epsilon: float,
    epsilon: float = 1.0e-12,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Use fixed safe limits and pre-update baselines for active repairs."""

    if not current_metrics:
        raise ValueError("Strict DG-CAIP step requires nonlinear constraints.")
    safe = {}
    violated = {}
    for name, raw_value in sorted(current_metrics.items()):
        value = float(raw_value)
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError("Strict DG-CAIP nonlinear metric is invalid.")
        family = name.rsplit(":", 1)[-1]
        if family == "js":
            safe[name] = value + float(js_epsilon)
        elif value > epsilon:
            violated[name] = value
        else:
            safe[name] = 0.0
    return safe, violated


def run_strict_dgcaip_step(
    *,
    parameters: Sequence[torch.Tensor],
    target_loss: torch.Tensor,
    observation: SDHObservation,
    protection_observations: Mapping[str, SDHObservation] | None = None,
    current_metrics: Mapping[str, float],
    evaluate_constraints: Callable[
        [Tuple[torch.Tensor, ...]], Mapping[str, float]
    ],
    step_size: float,
    js_epsilon: float,
    repair_floor_fraction: float = 0.05,
    max_repair_norm_ratio: float = 0.25,
    max_projection_iterations: int = 64,
    max_backtracks: int = 5,
    record_trace: bool = True,
) -> StrictDGCAIPStepResult:
    safe_losses, violated_losses = (
        multi_snapshot_constraint_losses(protection_observations)
        if protection_observations is not None
        else strict_constraint_losses(observation)
    )
    safe_limits, violated_baselines = partition_nonlinear_constraints(
        current_metrics, js_epsilon=js_epsilon
    )
    route = route_strict_final_update(
        parameters=parameters,
        target_loss=target_loss,
        safe_constraint_losses=safe_losses,
        violated_constraint_losses=violated_losses,
        repair_floor_fraction=repair_floor_fraction,
        max_repair_norm_ratio=max_repair_norm_ratio,
        max_projection_iterations=max_projection_iterations,
    )
    if not route.feasible:
        originals = tuple(parameter.detach().clone() for parameter in parameters)
        backtracking = MultiParameterBacktrackingResult(
            candidate=originals,
            accepted=False,
            attempts=0,
            step_size=0.0,
            values={},
            status="skip_infeasible_route",
            trace=(),
        )
    else:
        backtracking = backtrack_mixed_multi_parameter_constraints(
            parameters=parameters,
            flattened_gradient=route.gradient,
            step_size=step_size,
            evaluate_constraints=evaluate_constraints,
            safe_limits=safe_limits,
            violated_baselines=violated_baselines,
            max_backtracks=max_backtracks,
            record_trace=record_trace,
        )
    return StrictDGCAIPStepResult(
        route=route,
        backtracking=backtracking,
        safe_limits=safe_limits,
        violated_baselines=violated_baselines,
    )
