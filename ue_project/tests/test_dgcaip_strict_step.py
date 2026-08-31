from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ue_framework.methods.dgcaip_strict_step import (
    multi_snapshot_constraint_losses,
    partition_nonlinear_constraints,
    run_strict_dgcaip_step,
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
        classification_loss=structural,
        box_loss=parameter[1] * 0.0,
        alignment_loss=parameter[1] * 0.0,
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
