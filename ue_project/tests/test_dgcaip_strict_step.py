from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ue_framework.methods.dgcaip_strict_step import (
    multi_snapshot_constraint_losses,
    partition_nonlinear_constraints,
    run_strict_dgcaip_step,
    strict_component_constraint_losses,
    strict_constraint_losses,
)


def _observation(parameter: torch.Tensor, *, violated: bool):
    structural = (
        parameter[1].square() + 0.1
        if violated
        else parameter[1] * 0.0
    )
    term = SimpleNamespace(
        class_id=1,
        weight=2.0,
        classification_loss=structural,
        box_loss=parameter[1] * 0.0,
        alignment_loss=parameter[1] * 0.0,
        distribution_loss=parameter[0].square(),
    )
    return SimpleNamespace(
        nla=SimpleNamespace(
            loss=parameter[0].square(),
            per_class_loss={1: parameter[0].square()},
        ),
        dgcaip=SimpleNamespace(
            loss=structural,
            per_class_loss={1: structural},
            instances=(term,),
        ),
    )


def test_constraint_partition_uses_structural_tolerance_and_js_baseline() -> None:
    parameter = torch.tensor([0.0, 0.0], requires_grad=True)
    safe, violated = strict_constraint_losses(
        _observation(parameter, violated=True)
    )
    assert not safe and set(violated) == {"1"}
    safe_limits, violated_baselines = partition_nonlinear_constraints(
        {"1:probability": 0.02, "1:iou": 0.0, "1:js": 0.03},
        js_epsilon=1.0e-9,
    )
    assert safe_limits == pytest.approx({"1:iou": 0.0, "1:js": 0.030000001})
    assert violated_baselines == pytest.approx({"1:probability": 0.02})


def test_multi_snapshot_rows_remain_independent_and_named() -> None:
    parameter = torch.tensor([0.1, 0.1], requires_grad=True)
    safe, violated = multi_snapshot_constraint_losses(
        {
            "e1": _observation(parameter, violated=False),
            "e5": _observation(parameter, violated=True),
        }
    )
    assert set(safe) == {"e1/1"}
    assert set(violated) == {"e5/1"}


def test_strict_step_never_adds_an_unconstrained_protection_gradient() -> None:
    parameter = torch.tensor([0.1, 0.1], requires_grad=True)
    observation = _observation(parameter, violated=False)

    def evaluate(candidate):
        value = float(candidate[0][0].square())
        return {"1:probability": 0.0, "1:js": value}

    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=parameter[0] + parameter[1],
        observation=observation,
        current_metrics={"1:probability": 0.0, "1:js": 0.02},
        evaluate_constraints=evaluate,
        step_size=0.01,
        js_epsilon=1.0e-9,
    )
    assert result.route.feasible
    assert result.route.max_safe_final_row_dot <= 1.0e-5
    assert result.backtracking.accepted


def test_strict_step_accepts_sequentially_precomputed_gradient_rows() -> None:
    parameter = torch.tensor([0.1, 0.1], requires_grad=True)

    def evaluate(candidate):
        value = float(candidate[0][0].square())
        return {"e1/1:probability": 0.0, "e1/1:js": value}

    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=None,
        observation=None,
        target_gradient=torch.tensor([1.0, 1.0]),
        safe_constraint_gradients={"e1/1": torch.tensor([1.0, 0.0])},
        violated_constraint_gradients={},
        current_metrics={"e1/1:probability": 0.0, "e1/1:js": 0.02},
        evaluate_constraints=evaluate,
        step_size=0.01,
        js_epsilon=1.0e-9,
    )
    assert result.route.feasible
    assert result.route.gradient.tolist() == pytest.approx([0.0, 1.0])
    assert result.route.max_safe_final_row_dot <= 1.0e-6
    assert result.backtracking.accepted


def test_strict_step_v2_routes_and_backtracks_the_complete_update() -> None:
    parameter = torch.zeros(2, requires_grad=True)

    def evaluate(candidate):
        return {"e1/1:probability": 1.0 + float(candidate[0][0])}

    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=None,
        observation=None,
        target_gradient=torch.tensor([1.0, -1.0]),
        safe_constraint_gradients={},
        violated_constraint_gradients={"e1/1": torch.tensor([1.0, 0.0])},
        current_metrics={"e1/1:probability": 1.0},
        evaluate_constraints=evaluate,
        step_size=0.1,
        js_epsilon=1.0e-9,
        route_mode="nonworsening_target_progress_v2",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert result.route.feasible
    assert result.route.target_progress >= 0.60
    assert result.route.min_violated_final_row_dot >= -1.0e-6
    assert result.backtracking.accepted
    assert result.backtracking.values["e1/1:probability"] < 1.0


def test_component_rows_preserve_nla_and_risk_weighted_families() -> None:
    parameter = torch.tensor([0.2, 0.3], requires_grad=True)
    observation = _observation(parameter, violated=True)
    observation.nla.per_class_loss[2] = parameter[1].square()
    rows = strict_component_constraint_losses(observation)
    assert set(rows) == {
        "1:nla",
        "2:nla",
        "1:probability",
        "1:iou",
        "1:alignment",
        "1:js",
    }
    assert float(rows["1:nla"].detach()) == pytest.approx(0.04)
    assert float(rows["1:probability"].detach()) == pytest.approx(0.38)
    assert float(rows["1:js"].detach()) == pytest.approx(0.08)
    assert "2:probability" not in rows


def test_component_aligned_step_fails_closed_on_row_key_mismatch() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="safe gradient and metric keys differ"):
        run_strict_dgcaip_step(
            parameters=(parameter,),
            target_loss=None,
            observation=None,
            target_gradient=torch.tensor([1.0, -1.0]),
            safe_constraint_gradients={"e1/1": torch.tensor([1.0, 0.0])},
            violated_constraint_gradients={},
            current_metrics={"e1/1:probability": 0.0},
            evaluate_constraints=lambda _: {"e1/1:probability": 0.0},
            step_size=0.1,
            js_epsilon=1.0e-9,
            route_mode="component_aligned_target_progress_v3",
        )


def test_v2_combined_row_can_hide_a_worsening_component() -> None:
    parameter = torch.zeros(2, requires_grad=True)

    def evaluate(candidate):
        coordinate = float(candidate[0][0])
        return {
            "e1/1:probability": 1.0 + coordinate,
            "e1/1:iou": 1.0 - coordinate,
        }

    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=None,
        observation=None,
        target_gradient=torch.tensor([1.0, 0.0]),
        safe_constraint_gradients={},
        violated_constraint_gradients={"e1/1": torch.tensor([0.0, 1.0])},
        current_metrics={"e1/1:probability": 1.0, "e1/1:iou": 1.0},
        evaluate_constraints=evaluate,
        step_size=0.1,
        js_epsilon=1.0e-9,
        route_mode="nonworsening_target_progress_v2",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert result.route.feasible
    assert float(torch.dot(torch.tensor([0.0, 1.0]), result.route.gradient)) == 0.0
    assert not result.backtracking.accepted
    assert all(
        attempt.reason == "mixed_constraint_failed"
        for attempt in result.backtracking.trace
    )


def test_component_aligned_step_routes_the_exact_component_row() -> None:
    parameter = torch.zeros(2, requires_grad=True)

    def evaluate(candidate):
        return {"e1/1:probability": 1.0 + float(candidate[0][0])}

    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=None,
        observation=None,
        target_gradient=torch.tensor([1.0, -1.0]),
        safe_constraint_gradients={},
        violated_constraint_gradients={
            "e1/1:probability": torch.tensor([1.0, 0.0])
        },
        current_metrics={"e1/1:probability": 1.0},
        evaluate_constraints=evaluate,
        step_size=0.1,
        js_epsilon=1.0e-9,
        route_mode="component_aligned_target_progress_v3",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert result.route.feasible
    assert result.route.mode == "strict_nonworsening_target_progress_v3"
    assert result.route.active_violated_constraints == ("e1/1:probability",)
    assert result.backtracking.accepted


def test_component_aligned_route_skips_infeasible_progress_without_mutation() -> None:
    parameter = torch.zeros(1, requires_grad=True)
    result = run_strict_dgcaip_step(
        parameters=(parameter,),
        target_loss=None,
        observation=None,
        target_gradient=torch.ones(1),
        safe_constraint_gradients={"e1/1:nla": torch.ones(1)},
        violated_constraint_gradients={},
        current_metrics={"e1/1:nla": 0.0},
        evaluate_constraints=lambda _: {"e1/1:nla": 0.0},
        step_size=0.1,
        js_epsilon=1.0e-9,
        route_mode="component_aligned_target_progress_v3",
        minimum_target_progress=0.60,
        max_projection_iterations=128,
        svd_relative_tolerance=1.0e-6,
    )
    assert not result.route.feasible
    assert result.route.mode == "skip_infeasible_constraints_v3"
    assert not result.backtracking.accepted
    assert torch.equal(result.backtracking.candidate[0], parameter.detach())
