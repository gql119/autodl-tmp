from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

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


@dataclass(frozen=True)
class MultiParameterGradientRouteResult:
    mode: str
    gradient: torch.Tensor
    parameter_gradients: Tuple[torch.Tensor, ...]
    target_gradient: torch.Tensor
    projected_target_gradient: torch.Tensor
    nla_gradient: torch.Tensor
    constraint_matrix: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    null_dimension: int
    attack_retention: float
    max_projected_row_dot: float
    max_final_row_dot: float
    active_classes: Tuple[str, ...]
    target_norm: float
    projected_target_norm: float
    nla_norm: float
    combined_norm: float


@dataclass(frozen=True)
class MultiParameterBacktrackingResult:
    candidate: Tuple[torch.Tensor, ...]
    accepted: bool
    attempts: int
    step_size: float
    values: Dict[str, float]
    status: str
    trace: Tuple["ConstraintAttemptTrace", ...] = ()


@dataclass(frozen=True)
class ConstraintValueTrace:
    name: str
    family: str
    value: float
    limit: float
    margin: float
    violated: bool


@dataclass(frozen=True)
class ConstraintAttemptTrace:
    attempt: int
    step_size: float
    finite: bool
    constraints: Tuple[ConstraintValueTrace, ...]
    group_max_margin: Dict[str, float | None]
    group_violation_count: Dict[str, int]
    accepted: bool
    reason: str


@dataclass(frozen=True)
class BudgetedProtectionRouteResult:
    mode: str
    gradient: torch.Tensor
    parameter_gradients: Tuple[torch.Tensor, ...]
    target_gradient: torch.Tensor
    projected_target_gradient: torch.Tensor
    protection_gradient: torch.Tensor
    scaled_protection_gradient: torch.Tensor
    constraint_matrix: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    null_dimension: int
    attack_retention: float
    max_projected_row_dot: float
    max_final_row_dot: float
    active_classes: Tuple[str, ...]
    target_norm: float
    projected_target_norm: float
    protection_norm: float
    scaled_protection_norm: float
    explicit_protection_norm_ratio: float
    combined_norm: float


def _validate_parameters(parameters: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    frozen = tuple(parameters)
    if not frozen:
        raise ValueError("At least one omega parameter is required.")
    if any(not parameter.requires_grad for parameter in frozen):
        raise ValueError("Every routed omega parameter must require gradients.")
    if len({id(parameter) for parameter in frozen}) != len(frozen):
        raise ValueError("Routed omega parameters must be unique.")
    devices = {parameter.device for parameter in frozen}
    dtypes = {parameter.dtype for parameter in frozen}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("All routed omega parameters must share device and dtype.")
    return frozen


def flatten_parameter_tensors(
    tensors: Sequence[torch.Tensor],
    parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(tensors) != len(parameters):
        raise ValueError("Tensor and parameter sequences must have equal length.")
    pieces = []
    for tensor, parameter in zip(tensors, parameters):
        if tensor.shape != parameter.shape:
            raise ValueError("Gradient shape does not match its omega parameter.")
        pieces.append(tensor.reshape(-1))
    return torch.cat(pieces)


def unflatten_parameter_tensor(
    flattened: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    expected = sum(parameter.numel() for parameter in parameters)
    if flattened.ndim != 1 or flattened.numel() != expected:
        raise ValueError("Flattened gradient dimension does not match omega parameters.")
    output = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        output.append(flattened[offset : offset + count].reshape_as(parameter))
        offset += count
    return tuple(output)


def _multi_parameter_gradient(
    value: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise ValueError("Routed losses must be scalar tensors.")
    if not value.requires_grad:
        return torch.cat([torch.zeros_like(item).reshape(-1) for item in parameters])
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    materialized = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )
    if any(not torch.isfinite(gradient).all() for gradient in materialized):
        raise ValueError("Non-finite omega gradient.")
    return flatten_parameter_tensors(materialized, parameters)


def route_multi_parameter_gradients(
    *,
    parameters: Sequence[torch.Tensor],
    target_loss: torch.Tensor,
    per_class_nla_losses: Mapping[str, torch.Tensor],
    nla_loss: torch.Tensor,
    nla_weight: float,
    svd_relative_tolerance: float = 1.0e-4,
    epsilon: float = 1.0e-12,
) -> MultiParameterGradientRouteResult:
    """Project the target component, then add explicit NLA descent.

    All active non-target classes participate in the row space. Orthogonality
    applies only to ``projected_target_gradient``; the final update deliberately
    contains the NLA component and therefore need not be orthogonal.
    """

    omega = _validate_parameters(parameters)
    if nla_weight < 0:
        raise ValueError("nla_weight must be non-negative.")
    if svd_relative_tolerance <= 0:
        raise ValueError("svd_relative_tolerance must be positive.")
    target = _multi_parameter_gradient(target_loss, omega)
    target_norm_tensor = target.norm()
    if float(target_norm_tensor.detach()) <= epsilon:
        raise ValueError("Target loss has a zero omega gradient.")

    rows = []
    row_names = []
    for name in sorted(per_class_nla_losses):
        gradient = _multi_parameter_gradient(per_class_nla_losses[name], omega)
        norm = gradient.norm()
        if float(norm.detach()) <= epsilon:
            continue
        rows.append(gradient / norm)
        row_names.append(str(name))
    dimension = int(target.numel())
    if rows:
        matrix = torch.stack(rows)
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        if not torch.isfinite(singular_values).all():
            raise ValueError("Non-finite singular values in multi-parameter CGR.")
        rank = int(
            (
                singular_values / singular_values[0].clamp_min(epsilon)
                >= float(svd_relative_tolerance)
            ).sum().item()
        )
        row_space = vh[:rank]
        projected = target - row_space.T @ (row_space @ target)
        max_projected_dot = float((matrix @ projected).abs().max().detach())
    else:
        matrix = target.new_zeros((0, dimension))
        singular_values = target.new_zeros((0,))
        rank = 0
        projected = target
        max_projected_dot = 0.0
    nla_gradient = _multi_parameter_gradient(nla_loss, omega)
    combined = projected + float(nla_weight) * nla_gradient
    max_final_dot = (
        float((matrix @ combined).abs().max().detach()) if rows else 0.0
    )
    projected_norm = projected.norm()
    retention = float((projected_norm / target_norm_tensor.clamp_min(epsilon)).detach())
    if float(combined.norm().detach()) <= epsilon:
        mode = "skip"
    elif float(projected_norm.detach()) <= epsilon:
        mode = "protection_only"
    elif rows:
        mode = "projected_target_plus_nla"
    else:
        mode = "target_plus_nla"
    return MultiParameterGradientRouteResult(
        mode=mode,
        gradient=combined,
        parameter_gradients=unflatten_parameter_tensor(combined, omega),
        target_gradient=target,
        projected_target_gradient=projected,
        nla_gradient=nla_gradient,
        constraint_matrix=matrix,
        singular_values=singular_values,
        rank=rank,
        null_dimension=dimension - rank,
        attack_retention=retention,
        max_projected_row_dot=max_projected_dot,
        max_final_row_dot=max_final_dot,
        active_classes=tuple(row_names),
        target_norm=float(target_norm_tensor.detach()),
        projected_target_norm=float(projected_norm.detach()),
        nla_norm=float(nla_gradient.norm().detach()),
        combined_norm=float(combined.norm().detach()),
    )


def route_budgeted_protection_gradients(
    *,
    parameters: Sequence[torch.Tensor],
    target_loss: torch.Tensor,
    per_class_protection_losses: Mapping[str, torch.Tensor],
    protection_loss: torch.Tensor,
    protection_ratio: float = 0.25,
    svd_relative_tolerance: float = 1.0e-4,
    epsilon: float = 1.0e-12,
) -> BudgetedProtectionRouteResult:
    """Project target attack and add an exactly norm-budgeted protection step."""

    omega = _validate_parameters(parameters)
    if protection_ratio < 0:
        raise ValueError("protection_ratio must be non-negative.")
    if svd_relative_tolerance <= 0:
        raise ValueError("svd_relative_tolerance must be positive.")
    target = _multi_parameter_gradient(target_loss, omega)
    target_norm_tensor = target.norm()
    if float(target_norm_tensor.detach()) <= epsilon:
        raise ValueError("Target loss has a zero omega gradient.")

    rows = []
    row_names = []
    for name in sorted(per_class_protection_losses):
        gradient = _multi_parameter_gradient(
            per_class_protection_losses[name], omega
        )
        norm = gradient.norm()
        if float(norm.detach()) <= epsilon:
            continue
        rows.append(gradient / norm)
        row_names.append(str(name))
    dimension = int(target.numel())
    if rows:
        matrix = torch.stack(rows)
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        if not torch.isfinite(singular_values).all():
            raise ValueError("Non-finite singular values in budgeted CGR.")
        rank = int(
            (
                singular_values / singular_values[0].clamp_min(epsilon)
                >= float(svd_relative_tolerance)
            ).sum().item()
        )
        row_space = vh[:rank]
        projected = target - row_space.T @ (row_space @ target)
        max_projected_dot = float((matrix @ projected).abs().max().detach())
    else:
        matrix = target.new_zeros((0, dimension))
        singular_values = target.new_zeros((0,))
        rank = 0
        projected = target
        max_projected_dot = 0.0

    protection = _multi_parameter_gradient(protection_loss, omega)
    projected_norm = projected.norm()
    protection_norm = protection.norm()
    if (
        float(projected_norm.detach()) > epsilon
        and float(protection_norm.detach()) > epsilon
        and protection_ratio > 0
    ):
        scaled_protection = protection * (
            float(protection_ratio)
            * projected_norm
            / protection_norm.clamp_min(epsilon)
        )
    else:
        scaled_protection = torch.zeros_like(protection)
    combined = projected + scaled_protection
    scaled_norm = scaled_protection.norm()
    actual_ratio = (
        float((scaled_norm / projected_norm.clamp_min(epsilon)).detach())
        if float(projected_norm.detach()) > epsilon
        else 0.0
    )
    max_final_dot = (
        float((matrix @ combined).abs().max().detach()) if rows else 0.0
    )
    retention = float(
        (projected_norm / target_norm_tensor.clamp_min(epsilon)).detach()
    )
    if float(combined.norm().detach()) <= epsilon:
        mode = "skip"
    elif float(projected_norm.detach()) <= epsilon:
        mode = "skip_no_attack_nullspace"
    elif float(scaled_norm.detach()) <= epsilon:
        mode = "projected_target"
    elif rows:
        mode = "projected_target_plus_budgeted_protection"
    else:
        mode = "target_plus_budgeted_protection"
    return BudgetedProtectionRouteResult(
        mode=mode,
        gradient=combined,
        parameter_gradients=unflatten_parameter_tensor(combined, omega),
        target_gradient=target,
        projected_target_gradient=projected,
        protection_gradient=protection,
        scaled_protection_gradient=scaled_protection,
        constraint_matrix=matrix,
        singular_values=singular_values,
        rank=rank,
        null_dimension=dimension - rank,
        attack_retention=retention,
        max_projected_row_dot=max_projected_dot,
        max_final_row_dot=max_final_dot,
        active_classes=tuple(row_names),
        target_norm=float(target_norm_tensor.detach()),
        projected_target_norm=float(projected_norm.detach()),
        protection_norm=float(protection_norm.detach()),
        scaled_protection_norm=float(scaled_norm.detach()),
        explicit_protection_norm_ratio=actual_ratio,
        combined_norm=float(combined.norm().detach()),
    )


def backtrack_multi_parameter_update(
    *,
    parameters: Sequence[torch.Tensor],
    flattened_gradient: torch.Tensor,
    step_size: float,
    evaluate_probability_drops: Callable[[Tuple[torch.Tensor, ...]], Mapping[str, float]],
    tolerance: float = 0.005,
    max_backtracks: int = 5,
    epsilon: float = 1.0e-9,
) -> MultiParameterBacktrackingResult:
    """Accept only updates whose nonlinear per-class probability drops are safe."""

    omega = _validate_parameters(parameters)
    if step_size <= 0 or tolerance < 0:
        raise ValueError("Invalid CGR backtracking step or tolerance.")
    if max_backtracks != 5:
        raise ValueError("SDH-CGR requires exactly five nonlinear backtracks.")
    gradients = unflatten_parameter_tensor(flattened_gradient, omega)
    originals = tuple(parameter.detach().clone() for parameter in omega)
    current_step = float(step_size)
    last_values: Dict[str, float] = {}
    for attempt in range(max_backtracks + 1):
        candidate = tuple(
            original - current_step * gradient.detach()
            for original, gradient in zip(originals, gradients)
        )
        evaluated = dict(evaluate_probability_drops(candidate))
        if not evaluated:
            return MultiParameterBacktrackingResult(
                candidate=candidate,
                accepted=True,
                attempts=attempt + 1,
                step_size=current_step,
                values={},
                status="accepted_no_active_class",
            )
        if not all(torch.isfinite(torch.tensor(value)) for value in evaluated.values()):
            raise ValueError("CGR probability-drop evaluator returned non-finite values.")
        last_values = {str(name): float(value) for name, value in evaluated.items()}
        if all(value <= float(tolerance) + epsilon for value in last_values.values()):
            return MultiParameterBacktrackingResult(
                candidate=candidate,
                accepted=True,
                attempts=attempt + 1,
                step_size=current_step,
                values=last_values,
                status="accepted",
            )
        current_step *= 0.5
    return MultiParameterBacktrackingResult(
        candidate=originals,
        accepted=False,
        attempts=max_backtracks + 1,
        step_size=0.0,
        values=last_values,
        status="skip",
    )


def backtrack_multi_parameter_constraints(
    *,
    parameters: Sequence[torch.Tensor],
    flattened_gradient: torch.Tensor,
    step_size: float,
    evaluate_constraints: Callable[[Tuple[torch.Tensor, ...]], Mapping[str, float]],
    limits: Mapping[str, float],
    max_backtracks: int = 5,
    epsilon: float = 1.0e-9,
    record_trace: bool = False,
) -> MultiParameterBacktrackingResult:
    """Backtrack a multi-parameter update against heterogeneous limits."""

    omega = _validate_parameters(parameters)
    if step_size <= 0 or not limits:
        raise ValueError("A positive step and at least one limit are required.")
    if max_backtracks != 5:
        raise ValueError("SDH-CGR requires exactly five nonlinear backtracks.")
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in limits.values()):
        raise ValueError("Constraint limits must be finite and non-negative.")
    gradients = unflatten_parameter_tensor(flattened_gradient, omega)
    originals = tuple(parameter.detach().clone() for parameter in omega)
    current_step = float(step_size)
    last_values: Dict[str, float] = {}
    trace = []
    for attempt in range(max_backtracks + 1):
        candidate = tuple(
            original - current_step * gradient.detach()
            for original, gradient in zip(originals, gradients)
        )
        evaluated = dict(evaluate_constraints(candidate))
        if set(evaluated) != set(limits):
            raise ValueError("Constraint evaluator returned unexpected keys.")
        if not all(torch.isfinite(torch.tensor(value)) for value in evaluated.values()):
            raise ValueError("Constraint evaluator returned non-finite values.")
        last_values = {str(name): float(value) for name, value in evaluated.items()}
        if record_trace:
            constraint_rows = tuple(
                ConstraintValueTrace(
                    name=str(name),
                    family=str(name).rsplit(":", 1)[-1],
                    value=last_values[name],
                    limit=float(limits[name]),
                    margin=last_values[name] - float(limits[name]),
                    violated=last_values[name] > float(limits[name]) + epsilon,
                )
                for name in sorted(limits)
            )
            group_max_margin: Dict[str, float | None] = {}
            group_violation_count: Dict[str, int] = {}
            for family in ("probability", "iou", "alignment", "js"):
                family_rows = [
                    row for row in constraint_rows if row.family == family
                ]
                group_max_margin[family] = (
                    max(row.margin for row in family_rows) if family_rows else None
                )
                group_violation_count[family] = sum(
                    row.violated for row in family_rows
                )
            trace_accepted = all(not row.violated for row in constraint_rows)
            trace.append(
                ConstraintAttemptTrace(
                    attempt=attempt,
                    step_size=current_step,
                    finite=True,
                    constraints=constraint_rows,
                    group_max_margin=group_max_margin,
                    group_violation_count=group_violation_count,
                    accepted=trace_accepted,
                    reason=(
                        "accepted"
                        if trace_accepted
                        else "constraint_limit_exceeded"
                    ),
                )
            )
        if all(
            last_values[name] <= float(limits[name]) + epsilon
            for name in limits
        ):
            return MultiParameterBacktrackingResult(
                candidate=candidate,
                accepted=True,
                attempts=attempt + 1,
                step_size=current_step,
                values=last_values,
                status="accepted",
                trace=tuple(trace),
            )
        current_step *= 0.5
    return MultiParameterBacktrackingResult(
        candidate=originals,
        accepted=False,
        attempts=max_backtracks + 1,
        step_size=0.0,
        values=last_values,
        status="skip",
        trace=tuple(trace),
    )


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
