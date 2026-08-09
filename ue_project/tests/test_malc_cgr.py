from __future__ import annotations

import pytest
import torch

from ue_framework.methods.malc_cgr import (
    class_probability_constraint_terms,
    route_malc_cgr_update,
)
from ue_framework.methods.shadow_tal import (
    NonTargetClassConstraint,
    NonTargetConstraintSet,
)


def _constraint(
    class_id: int,
    cls_margin: torch.Tensor,
    box_margin: torch.Tensor,
) -> NonTargetClassConstraint:
    zero = cls_margin * 0.0
    return NonTargetClassConstraint(
        class_id=class_id,
        count=3,
        cls_margin=cls_margin,
        box_margin=box_margin,
        cls_violation=zero,
        box_violation=box_margin * 0.0,
        clean_probability_mean=0.8,
        adv_probability_mean=0.8 - float(cls_margin.detach()),
    )


def test_only_class_probability_margin_enters_cgr_rows() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    constraint_set = NonTargetConstraintSet(
        constraints=(_constraint(7, parameter[0], 100.0 * parameter[1]),),
        status="active",
    )
    terms = class_probability_constraint_terms(constraint_set)
    assert [term.name for term in terms] == ["class_7_cls"]
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=parameter.sum(),
        constraint_set=constraint_set,
        step_size=0.1,
        evaluate_class_margins=lambda candidate: {
            "class_7_cls": float(candidate[0])
        },
    )
    assert result.route.rank == 1
    assert result.route.max_projected_row_dot <= 1e-6
    assert result.candidate[0] == pytest.approx(0.0, abs=1e-7)
    assert result.candidate[1] == pytest.approx(-0.1)
    assert result.box_margin_monitor == {"class_7_box": 0.0}


def test_violated_class_constraint_uses_repair_only() -> None:
    parameter = torch.tensor([0.02, 0.0], requires_grad=True)
    constraint_set = NonTargetConstraintSet(
        constraints=(_constraint(3, parameter[0], parameter[1]),),
        status="active",
    )
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=-parameter[1],
        constraint_set=constraint_set,
        step_size=0.01,
        evaluate_class_margins=lambda candidate: {
            "class_3_cls": float(candidate[0])
        },
    )
    assert result.selected_mode == "repair_only"
    assert result.accepted
    assert result.candidate[0] < parameter.detach()[0]
    assert result.candidate[1] == parameter.detach()[1]


def test_actual_nonlinear_constraint_is_backtracked() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    nonlinear_margin = parameter[0] + parameter[1].square()
    constraint_set = NonTargetConstraintSet(
        constraints=(_constraint(4, nonlinear_margin, parameter[0] * 0.0),),
        status="active",
    )
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=-parameter[1],
        constraint_set=constraint_set,
        step_size=1.0,
        evaluate_class_margins=lambda candidate: {
            "class_4_cls": float(candidate[0] + candidate[1].square())
        },
    )
    assert result.selected_mode == "projected_target"
    assert result.accepted
    assert result.attempts == 5
    assert result.class_values_after["class_4_cls"] <= 0.005


def test_backtracking_failure_skips_without_changing_coefficients() -> None:
    parameter = torch.zeros(2, requires_grad=True)
    constraint_set = NonTargetConstraintSet(
        constraints=(_constraint(9, parameter[0], parameter[1] * 0.0),),
        status="active",
    )
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=-parameter[1],
        constraint_set=constraint_set,
        step_size=1.0,
        evaluate_class_margins=lambda _candidate: {"class_9_cls": 1.0},
    )
    assert not result.accepted
    assert result.selected_mode == "skip"
    assert result.attempts == 6
    assert torch.equal(result.candidate, parameter.detach())


def test_no_non_target_class_uses_unprojected_target_gradient() -> None:
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=parameter.sum(),
        constraint_set=NonTargetConstraintSet(
            constraints=(), status="not_applicable"
        ),
        step_size=0.1,
        evaluate_class_margins=lambda _candidate: {},
    )
    assert result.selected_mode == "target"
    assert result.accepted
    assert torch.allclose(result.candidate, torch.tensor([0.9, 1.9]))


def test_composite_target_loss_is_the_gradient_routed_by_cgr() -> None:
    parameter = torch.tensor([0.2, -0.3], requires_grad=True)
    easy_cls = (2.0 * parameter).sum()
    malc = (parameter - 1.0).square().mean()
    rms = parameter.square().mean()
    composite = easy_cls + 0.5 * malc + 0.25 * rms
    expected = torch.autograd.grad(composite, parameter, retain_graph=True)[0]
    result = route_malc_cgr_update(
        parameter=parameter,
        target_loss=composite,
        constraint_set=NonTargetConstraintSet(
            constraints=(), status="not_applicable"
        ),
        step_size=0.1,
        evaluate_class_margins=lambda _candidate: {},
    )
    assert torch.allclose(result.route.target_gradient, expected)
    assert torch.allclose(
        result.candidate,
        parameter.detach() - 0.1 * expected,
    )


def test_duplicate_class_constraints_fail_closed() -> None:
    parameter = torch.zeros(1, requires_grad=True)
    duplicate = _constraint(2, parameter[0], parameter[0])
    with pytest.raises(ValueError, match="duplicate"):
        class_probability_constraint_terms(
            NonTargetConstraintSet(
                constraints=(duplicate, duplicate), status="active"
            )
        )
