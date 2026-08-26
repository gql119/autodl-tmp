from __future__ import annotations

import pytest
import torch

from ue_framework.methods.constraint_gradient_router import (
    backtrack_multi_parameter_constraints,
    route_budgeted_protection_gradients,
)


def test_budgeted_router_projects_target_and_scales_protection_exactly() -> None:
    omega = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    target_loss = omega @ torch.tensor([1.0, 2.0, 3.0])
    class_zero = omega[0].square()
    class_one = omega[1].square()
    result = route_budgeted_protection_gradients(
        parameters=(omega,),
        target_loss=target_loss,
        per_class_protection_losses={"0": class_zero, "1": class_one},
        protection_loss=0.5 * (class_zero + class_one),
        protection_ratio=0.25,
    )
    assert result.rank == 2
    assert result.null_dimension == 1
    assert result.max_projected_row_dot <= 1.0e-6
    assert result.projected_target_gradient.tolist() == pytest.approx([0.0, 0.0, 3.0])
    assert result.explicit_protection_norm_ratio == pytest.approx(0.25)
    assert result.mode == "projected_target_plus_budgeted_protection"
    assert torch.isfinite(result.gradient).all()


def test_full_rank_constraints_skip_when_no_attack_nullspace_remains() -> None:
    omega = torch.tensor([1.0, 1.0], requires_grad=True)
    result = route_budgeted_protection_gradients(
        parameters=(omega,),
        target_loss=omega.sum(),
        per_class_protection_losses={"0": omega[0], "1": omega[1]},
        protection_loss=omega.square().sum(),
    )
    assert result.rank == 2
    assert result.null_dimension == 0
    assert result.mode == "skip"
    assert result.gradient.abs().sum().item() == pytest.approx(0.0)
    assert result.explicit_protection_norm_ratio == 0.0


def test_no_active_protection_keeps_target_route() -> None:
    omega = torch.tensor([1.0, 2.0], requires_grad=True)
    result = route_budgeted_protection_gradients(
        parameters=(omega,),
        target_loss=omega.sum(),
        per_class_protection_losses={},
        protection_loss=omega.sum() * 0.0,
    )
    assert result.rank == 0
    assert result.null_dimension == 2
    assert result.mode == "projected_target"
    assert result.gradient.tolist() == pytest.approx([1.0, 1.0])
    assert result.explicit_protection_norm_ratio == 0.0


def test_heterogeneous_backtracking_accepts_after_step_reduction() -> None:
    omega = torch.tensor([0.0], requires_grad=True)

    def evaluate(candidate):
        value = float(candidate[0].abs().item())
        return {
            "class1:probability": value,
            "class1:iou": value / 2.0,
            "class1:alignment": value / 4.0,
            "class1:js": value / 8.0,
        }

    result = backtrack_multi_parameter_constraints(
        parameters=(omega,),
        flattened_gradient=torch.tensor([-1.0]),
        step_size=0.04,
        evaluate_constraints=evaluate,
        limits={
            "class1:probability": 0.005,
            "class1:iou": 0.02,
            "class1:alignment": 0.05,
            "class1:js": 0.01,
        },
    )
    assert result.accepted is True
    assert result.attempts == 4
    assert result.step_size == pytest.approx(0.005)
    assert result.trace == ()


def test_heterogeneous_backtracking_trace_is_observational_only() -> None:
    omega = torch.tensor([0.0], requires_grad=True)

    def evaluate(candidate):
        value = float(candidate[0].abs().item())
        return {
            "class1:probability": value,
            "class1:iou": value / 2.0,
            "class1:alignment": value / 4.0,
            "class1:js": value / 8.0,
        }

    kwargs = {
        "parameters": (omega,),
        "flattened_gradient": torch.tensor([-1.0]),
        "step_size": 0.04,
        "evaluate_constraints": evaluate,
        "limits": {
            "class1:probability": 0.005,
            "class1:iou": 0.02,
            "class1:alignment": 0.05,
            "class1:js": 0.01,
        },
    }
    untraced = backtrack_multi_parameter_constraints(**kwargs)
    traced = backtrack_multi_parameter_constraints(
        **kwargs,
        record_trace=True,
    )
    assert traced.candidate[0].tolist() == pytest.approx(
        untraced.candidate[0].tolist()
    )
    assert traced.accepted == untraced.accepted
    assert traced.attempts == untraced.attempts
    assert traced.step_size == untraced.step_size
    assert traced.values == untraced.values
    assert traced.status == untraced.status
    assert untraced.trace == ()
    assert traced.accepted is True
    assert traced.attempts == 4
    assert traced.step_size == pytest.approx(0.005)
    assert len(traced.trace) == traced.attempts
    assert [item.attempt for item in traced.trace] == [0, 1, 2, 3]
    assert traced.trace[-1].accepted is True
    assert traced.trace[-1].reason == "accepted"
    assert traced.trace[-1].group_violation_count == {
        "probability": 0,
        "iou": 0,
        "alignment": 0,
        "js": 0,
    }
    first_probability = next(
        row
        for row in traced.trace[0].constraints
        if row.name == "class1:probability"
    )
    assert first_probability.value == pytest.approx(0.04)
    assert first_probability.limit == pytest.approx(0.005)
    assert first_probability.margin == pytest.approx(0.035)
    assert first_probability.violated is True


def test_heterogeneous_backtracking_skips_after_exactly_five_reductions() -> None:
    omega = torch.tensor([0.0], requires_grad=True)
    result = backtrack_multi_parameter_constraints(
        parameters=(omega,),
        flattened_gradient=torch.tensor([1.0]),
        step_size=1.0,
        evaluate_constraints=lambda _: {"class1:js": 1.0},
        limits={"class1:js": 0.0},
    )
    assert result.accepted is False
    assert result.attempts == 6
    assert result.status == "skip"
    assert result.candidate[0].item() == 0.0


def test_heterogeneous_backtracking_trace_records_terminal_rejection() -> None:
    omega = torch.tensor([0.0], requires_grad=True)
    result = backtrack_multi_parameter_constraints(
        parameters=(omega,),
        flattened_gradient=torch.tensor([1.0]),
        step_size=1.0,
        evaluate_constraints=lambda _: {"class1:js": 1.0},
        limits={"class1:js": 0.0},
        record_trace=True,
    )
    assert result.accepted is False
    assert len(result.trace) == 6
    assert result.trace[-1].step_size == pytest.approx(1.0 / 32.0)
    assert result.trace[-1].group_max_margin == {
        "probability": None,
        "iou": None,
        "alignment": None,
        "js": pytest.approx(1.0),
    }
    assert result.trace[-1].group_violation_count["js"] == 1
    assert result.trace[-1].reason == "constraint_limit_exceeded"
