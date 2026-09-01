from __future__ import annotations

import pytest
import torch

from ue_framework.methods.constraint_gradient_router import (
    ConstraintTerm,
    backtrack_mixed_multi_parameter_constraints,
    backtracking_candidate,
    flatten_loss_gradient,
    route_strict_final_update,
    route_coefficient_gradient,
)


def test_no_constraints_uses_target_gradient() -> None:
    parameter = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = route_coefficient_gradient(
        parameter=parameter,
        target_loss=parameter.sum(),
        constraints=[],
    )
    assert result.mode == "target"
    assert torch.equal(result.gradient, torch.ones_like(parameter))
    assert result.rank == 0
    assert result.null_dimension == 3
    assert result.attack_retention == pytest.approx(1.0)


def test_near_boundary_projects_target_into_nullspace() -> None:
    parameter = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    result = route_coefficient_gradient(
        parameter=parameter,
        target_loss=parameter.sum(),
        constraints=[
            ConstraintTerm("class_7_cls", parameter[0], tolerance=0.0),
        ],
    )
    assert result.mode == "projected_target"
    assert result.rank == 1
    assert result.null_dimension == 2
    assert result.gradient[0] == pytest.approx(0.0, abs=1e-7)
    assert result.max_projected_row_dot <= 1e-6


def test_full_rank_projection_and_near_singular_rank() -> None:
    full = torch.zeros(3, requires_grad=True)
    full_result = route_coefficient_gradient(
        parameter=full,
        target_loss=full.sum(),
        constraints=[
            ConstraintTerm(f"row_{index}", full[index], tolerance=0.0)
            for index in range(3)
        ],
    )
    assert full_result.rank == 3
    assert full_result.null_dimension == 0
    assert full_result.attack_retention == pytest.approx(0.0, abs=1e-6)

    near = torch.zeros(2, requires_grad=True)
    near_result = route_coefficient_gradient(
        parameter=near,
        target_loss=near.sum(),
        constraints=[
            ConstraintTerm("row_a", near[0], tolerance=0.0),
            ConstraintTerm("row_b", near[0] + 1e-8 * near[1], tolerance=0.0),
        ],
        svd_relative_tolerance=1e-4,
    )
    assert near_result.rank == 1
    assert near_result.null_dimension == 1


def test_zero_rank_constraint_skips() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    result = route_coefficient_gradient(
        parameter=parameter,
        target_loss=parameter.sum(),
        constraints=[
            ConstraintTerm("zero", parameter.sum() * 0.0, tolerance=0.0),
        ],
    )
    assert result.mode == "skip"
    assert torch.equal(result.gradient, torch.zeros_like(parameter))


def test_violated_constraint_switches_to_repair_only() -> None:
    parameter = torch.tensor([0.5, 0.0], requires_grad=True)
    result = route_coefficient_gradient(
        parameter=parameter,
        target_loss=-parameter[1],
        constraints=[
            ConstraintTerm("class_3_cls", parameter[0], tolerance=0.1),
        ],
    )
    assert result.mode == "repair_only"
    assert result.violated_constraints == ("class_3_cls",)
    assert torch.equal(result.gradient, torch.tensor([1.0, 0.0]))
    assert result.projected_target_gradient[0] == pytest.approx(0.0, abs=1e-7)


def test_backtracking_accepts_feasible_smaller_step_and_skips_failure() -> None:
    parameter = torch.tensor([0.0])
    accepted = backtracking_candidate(
        parameter=parameter,
        gradient=torch.tensor([-1.0]),
        step_size=1.0,
        evaluate_constraints=lambda candidate: {
            "quadratic": float(candidate[0].square())
        },
        limits={"quadratic": 0.1},
        mode="feasible",
        max_backtracks=5,
    )
    assert accepted.accepted
    assert accepted.attempts == 3
    assert accepted.candidate[0] == pytest.approx(0.25)

    failed = backtracking_candidate(
        parameter=parameter,
        gradient=torch.tensor([-1.0]),
        step_size=1.0,
        evaluate_constraints=lambda _candidate: {"fixed": 1.0},
        limits={"fixed": 0.0},
        mode="feasible",
        max_backtracks=2,
    )
    assert not failed.accepted
    assert failed.status == "skip"
    assert torch.equal(failed.candidate, parameter)


def test_repair_backtracking_requires_actual_improvement() -> None:
    parameter = torch.tensor([1.0])
    result = backtracking_candidate(
        parameter=parameter,
        gradient=torch.tensor([1.0]),
        step_size=0.5,
        evaluate_constraints=lambda candidate: {"margin": float(candidate[0])},
        limits={"margin": 0.0},
        mode="repair",
        baseline_values={"margin": 1.0},
    )
    assert result.accepted
    assert result.values["margin"] == pytest.approx(0.5)


def test_strict_route_constrains_the_complete_final_update() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    result = route_strict_final_update(
        parameters=(parameter,),
        target_loss=parameter[0] + parameter[1],
        safe_constraint_losses={"safe": parameter[0]},
        violated_constraint_losses={},
    )
    assert result.feasible
    assert result.mode == "strict_projected_target"
    assert result.gradient.tolist() == pytest.approx([0.0, 1.0], abs=1e-7)
    assert result.max_safe_final_row_dot <= 1.0e-6


def test_precomputed_strict_gradients_match_loss_based_route() -> None:
    parameter = torch.zeros(3, requires_grad=True)
    target_loss = parameter[0] + 2.0 * parameter[1] + parameter[2]
    safe_loss = parameter[0] + parameter[1]
    violated_loss = parameter[2]
    loss_based = route_strict_final_update(
        parameters=(parameter,),
        target_loss=target_loss,
        safe_constraint_losses={"safe": safe_loss},
        violated_constraint_losses={"violated": violated_loss},
    )
    target_gradient = flatten_loss_gradient(
        target_loss, (parameter,), retain_graph=True
    )
    safe_gradient = flatten_loss_gradient(
        safe_loss, (parameter,), retain_graph=True
    )
    violated_gradient = flatten_loss_gradient(
        violated_loss, (parameter,), retain_graph=False
    )
    precomputed = route_strict_final_update(
        parameters=(parameter,),
        target_gradient=target_gradient,
        safe_constraint_gradients={"safe": safe_gradient},
        violated_constraint_gradients={"violated": violated_gradient},
    )
    assert precomputed.mode == loss_based.mode
    assert precomputed.feasible == loss_based.feasible
    assert precomputed.gradient.tolist() == pytest.approx(
        loss_based.gradient.tolist(), abs=1.0e-7
    )
    assert precomputed.max_safe_final_row_dot == pytest.approx(
        loss_based.max_safe_final_row_dot, abs=1.0e-7
    )
    assert precomputed.min_violated_final_row_dot == pytest.approx(
        loss_based.min_violated_final_row_dot, abs=1.0e-7
    )


def test_strict_route_repairs_inside_safe_nullspace() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    result = route_strict_final_update(
        parameters=(parameter,),
        target_loss=parameter[0] + 0.01 * parameter[1],
        safe_constraint_losses={"safe": parameter[0]},
        violated_constraint_losses={"violated": parameter[1]},
    )
    assert result.feasible
    assert result.mode == "strict_projected_target_with_repair"
    assert result.max_safe_final_row_dot <= 1.0e-6
    assert result.min_violated_final_row_dot >= result.repair_floor - 1.0e-6
    assert result.gradient[0] == pytest.approx(0.0, abs=1e-7)


def test_strict_route_skips_incompatible_safe_and_repair_rows() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    result = route_strict_final_update(
        parameters=(parameter,),
        target_loss=parameter[1],
        safe_constraint_losses={"safe": parameter[0]},
        violated_constraint_losses={"violated": parameter[0]},
    )
    assert not result.feasible
    assert result.mode == "skip_infeasible_constraints"
    assert torch.equal(result.gradient, torch.zeros_like(parameter))


def test_strict_route_v2_replaces_the_observed_repair_budget_failure() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    target = torch.tensor([1.0, -1.0])
    violated = torch.tensor([0.0, 1.0])
    legacy = route_strict_final_update(
        parameters=(parameter,),
        target_gradient=target,
        safe_constraint_gradients={},
        violated_constraint_gradients={"violated": violated},
        repair_floor_fraction=0.05,
        max_repair_norm_ratio=0.25,
    )
    revised = route_strict_final_update(
        parameters=(parameter,),
        target_gradient=target,
        safe_constraint_gradients={},
        violated_constraint_gradients={"violated": violated},
        route_mode="nonworsening_target_progress_v2",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert not legacy.feasible
    assert legacy.repair_norm / legacy.target_norm > 0.25
    assert revised.feasible
    assert revised.mode == "strict_nonworsening_target_progress_v2"
    assert revised.min_violated_final_row_dot >= -1.0e-6
    assert revised.target_progress >= 0.60 - 1.0e-6
    assert revised.solver_dtype == "float64"


def test_strict_route_v2_audits_complete_postcast_update() -> None:
    parameter = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    result = route_strict_final_update(
        parameters=(parameter,),
        target_gradient=torch.tensor([1.0, -1.0, 1.0]),
        safe_constraint_gradients={"safe": torch.tensor([0.0, 0.0, 1.0])},
        violated_constraint_gradients={
            "violated": torch.tensor([0.0, 1.0, 0.0])
        },
        route_mode="nonworsening_target_progress_v2",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert result.feasible
    assert result.gradient.dtype == torch.float32
    assert result.max_safe_final_row_dot <= 1.0e-5
    assert result.precast_max_safe_row_dot <= 1.0e-8
    assert result.min_violated_final_row_dot >= -1.0e-6
    assert result.target_progress >= 0.60 - 1.0e-6
    assert result.target_cosine > 0.0
    assert torch.dot(result.safe_constraint_matrix[0], result.gradient).abs() <= 1.0e-5
    assert torch.dot(result.violated_constraint_matrix[0], result.gradient) >= -1.0e-6


def test_strict_route_v2_skips_infeasible_target_progress_without_mutation() -> None:
    parameter = torch.zeros(1, requires_grad=True)
    result = route_strict_final_update(
        parameters=(parameter,),
        target_gradient=torch.ones(1),
        safe_constraint_gradients={"safe": torch.ones(1)},
        violated_constraint_gradients={},
        route_mode="nonworsening_target_progress_v2",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert not result.feasible
    assert result.mode == "skip_infeasible_constraints_v2"
    assert torch.equal(result.gradient, torch.zeros_like(parameter))
    assert torch.equal(parameter, torch.zeros_like(parameter))


def test_strict_route_v2_is_deterministic_for_precomputed_rows() -> None:
    parameter = torch.zeros(3, requires_grad=True)
    kwargs = {
        "parameters": (parameter,),
        "target_gradient": torch.tensor([1.0, -1.0, 1.0]),
        "safe_constraint_gradients": {"safe": torch.tensor([0.0, 0.0, 1.0])},
        "violated_constraint_gradients": {
            "violated": torch.tensor([0.0, 1.0, 0.0])
        },
        "route_mode": "nonworsening_target_progress_v2",
        "minimum_target_progress": 0.60,
        "max_projection_iterations": 128,
        "svd_relative_tolerance": 1.0e-6,
    }
    first = route_strict_final_update(**kwargs)
    second = route_strict_final_update(**kwargs)
    assert torch.equal(first.gradient, second.gradient)
    assert first.target_progress == second.target_progress
    assert first.max_safe_final_row_dot == second.max_safe_final_row_dot


def test_mixed_backtracking_requires_safe_feasibility_and_real_repair() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)

    def evaluate(candidate):
        value = float(candidate[0][0])
        return {"safe": value * value, "violated": value}

    accepted = backtrack_mixed_multi_parameter_constraints(
        parameters=(parameter,),
        flattened_gradient=torch.tensor([1.0]),
        step_size=0.5,
        evaluate_constraints=evaluate,
        safe_limits={"safe": 0.8},
        violated_baselines={"violated": 1.0},
        record_trace=True,
    )
    assert accepted.accepted
    assert accepted.values == pytest.approx({"safe": 0.25, "violated": 0.5})
    assert accepted.trace[-1].accepted

    skipped = backtrack_mixed_multi_parameter_constraints(
        parameters=(parameter,),
        flattened_gradient=torch.tensor([-1.0]),
        step_size=0.5,
        evaluate_constraints=evaluate,
        safe_limits={"safe": 4.0},
        violated_baselines={"violated": 1.0},
    )
    assert not skipped.accepted
    assert skipped.status == "skip"
