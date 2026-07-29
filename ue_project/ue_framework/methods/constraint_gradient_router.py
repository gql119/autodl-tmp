from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ConstraintTerm:
    name: str
    margin: torch.Tensor
    tolerance: float


@dataclass(frozen=True)
class GradientRouteResult:
    mode: str
    gradient: torch.Tensor
    target_gradient: torch.Tensor
    projected_target_gradient: torch.Tensor
    constraint_matrix: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    null_dimension: int
    attack_retention: float
    max_projected_row_dot: float
    active_constraints: tuple[str, ...]
    violated_constraints: tuple[str, ...]


@dataclass(frozen=True)
class BacktrackingResult:
    candidate: torch.Tensor
    accepted: bool
    attempts: int
    step_size: float
    values: dict[str, float]
    status: str


def _gradient(
    value: torch.Tensor,
    parameter: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor | None:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ValueError("Losses and constraint margins must be scalar tensors.")
    if not value.requires_grad:
        return None
    gradient = torch.autograd.grad(
        value,
        parameter,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        return None
    if not torch.isfinite(gradient).all():
        raise ValueError("Non-finite coefficient gradient.")
    return gradient


def route_coefficient_gradient(
    *,
    parameter: torch.Tensor,
    target_loss: torch.Tensor,
    constraints: Sequence[ConstraintTerm],
    near_boundary: float = 0.005,
    svd_relative_tolerance: float = 1e-4,
    epsilon: float = 1e-12,
) -> GradientRouteResult:
    if not parameter.requires_grad:
        raise ValueError("The routed coefficient parameter must require gradients.")
    if near_boundary < 0:
        raise ValueError("near_boundary must be non-negative.")
    target_gradient = _gradient(
        target_loss,
        parameter,
        retain_graph=True,
    )
    if target_gradient is None:
        raise ValueError("Target loss is disconnected from the coefficient parameter.")
    target_flat = target_gradient.reshape(-1)
    dimension = int(target_flat.numel())

    active: list[ConstraintTerm] = []
    violated: list[ConstraintTerm] = []
    for constraint in constraints:
        value = float(constraint.margin.detach().item())
        if value > float(constraint.tolerance):
            violated.append(constraint)
            active.append(constraint)
        elif value >= float(constraint.tolerance) - near_boundary:
            active.append(constraint)

    if not active:
        empty_matrix = parameter.new_zeros((0, dimension))
        return GradientRouteResult(
            mode="target",
            gradient=target_gradient,
            target_gradient=target_gradient,
            projected_target_gradient=target_gradient,
            constraint_matrix=empty_matrix,
            singular_values=parameter.new_zeros((0,)),
            rank=0,
            null_dimension=dimension,
            attack_retention=1.0,
            max_projected_row_dot=0.0,
            active_constraints=(),
            violated_constraints=(),
        )

    rows: list[torch.Tensor] = []
    row_names: list[str] = []
    for constraint in active:
        gradient = _gradient(
            constraint.margin,
            parameter,
            retain_graph=True,
        )
        if gradient is None:
            continue
        flat = gradient.reshape(-1)
        norm = flat.norm()
        if float(norm.detach().item()) <= epsilon:
            continue
        rows.append(flat / norm)
        row_names.append(constraint.name)

    if not rows:
        zeros = torch.zeros_like(target_gradient)
        return GradientRouteResult(
            mode="skip",
            gradient=zeros,
            target_gradient=target_gradient,
            projected_target_gradient=zeros,
            constraint_matrix=parameter.new_zeros((0, dimension)),
            singular_values=parameter.new_zeros((0,)),
            rank=0,
            null_dimension=dimension,
            attack_retention=0.0,
            max_projected_row_dot=0.0,
            active_constraints=tuple(item.name for item in active),
            violated_constraints=tuple(item.name for item in violated),
        )

    matrix = torch.stack(rows, dim=0)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if not torch.isfinite(singular_values).all():
        raise ValueError("Non-finite singular values in constraint projection.")
    if singular_values.numel() == 0 or float(singular_values[0]) <= epsilon:
        rank = 0
    else:
        rank = int(
            (
                singular_values / singular_values[0]
                >= float(svd_relative_tolerance)
            )
            .sum()
            .item()
        )
    if rank == 0:
        projected_flat = target_flat
    else:
        row_space = vh[:rank]
        projected_flat = target_flat - row_space.T @ (row_space @ target_flat)
    projected = projected_flat.reshape_as(parameter)
    target_norm = target_flat.norm()
    retention = float(
        (
            projected_flat.norm()
            / target_norm.clamp_min(epsilon)
        )
        .detach()
        .item()
    )
    row_dot = matrix @ projected_flat
    max_row_dot = float(row_dot.abs().max().detach().item())

    if violated:
        repair_loss = torch.stack(
            [
                F.relu(item.margin - float(item.tolerance))
                for item in violated
            ]
        ).sum()
        repair_gradient = _gradient(
            repair_loss,
            parameter,
            retain_graph=True,
        )
        if repair_gradient is None or float(repair_gradient.norm()) <= epsilon:
            mode = "skip"
            selected = torch.zeros_like(parameter)
        else:
            mode = "repair_only"
            selected = repair_gradient
    else:
        mode = "projected_target"
        selected = projected

    return GradientRouteResult(
        mode=mode,
        gradient=selected,
        target_gradient=target_gradient,
        projected_target_gradient=projected,
        constraint_matrix=matrix,
        singular_values=singular_values,
        rank=rank,
        null_dimension=dimension - rank,
        attack_retention=retention,
        max_projected_row_dot=max_row_dot,
        active_constraints=tuple(row_names),
        violated_constraints=tuple(item.name for item in violated),
    )


def backtracking_candidate(
    *,
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    evaluate_constraints: Callable[[torch.Tensor], Mapping[str, float]],
    limits: Mapping[str, float],
    mode: str,
    baseline_values: Mapping[str, float] | None = None,
    max_backtracks: int = 5,
    epsilon: float = 1e-9,
) -> BacktrackingResult:
    if mode not in {"feasible", "repair"}:
        raise ValueError("Backtracking mode must be 'feasible' or 'repair'.")
    if step_size <= 0 or max_backtracks < 0:
        raise ValueError("step_size must be positive and max_backtracks non-negative.")
    if gradient.shape != parameter.shape:
        raise ValueError("Gradient and parameter shapes must match.")
    if set(limits) == set():
        raise ValueError("At least one nonlinear constraint is required.")
    if mode == "repair":
        if baseline_values is None or set(baseline_values) != set(limits):
            raise ValueError(
                "Repair backtracking requires baseline values for every constraint."
            )

    original = parameter.detach().clone()
    current_step = float(step_size)
    last_values: dict[str, float] = {}
    for attempt in range(max_backtracks + 1):
        candidate = original - current_step * gradient.detach()
        evaluated = dict(evaluate_constraints(candidate))
        if set(evaluated) != set(limits):
            raise ValueError("Constraint evaluator returned unexpected keys.")
        if not all(torch.isfinite(torch.tensor(value)) for value in evaluated.values()):
            raise ValueError("Constraint evaluator returned non-finite values.")
        last_values = {name: float(value) for name, value in evaluated.items()}

        if mode == "feasible":
            accepted = all(
                last_values[name] <= float(limit) + epsilon
                for name, limit in limits.items()
            )
        else:
            assert baseline_values is not None
            nonincreasing = all(
                last_values[name] <= float(baseline_values[name]) + epsilon
                for name in limits
            )
            improved = any(
                last_values[name] < float(baseline_values[name]) - epsilon
                for name in limits
            )
            accepted = nonincreasing and improved
        if accepted:
            return BacktrackingResult(
                candidate=candidate,
                accepted=True,
                attempts=attempt + 1,
                step_size=current_step,
                values=last_values,
                status="accepted",
            )
        current_step *= 0.5

    return BacktrackingResult(
        candidate=original,
        accepted=False,
        attempts=max_backtracks + 1,
        step_size=0.0,
        values=last_values,
        status="skip",
    )
