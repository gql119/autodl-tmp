from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ue_framework.methods.shadow_tal import (
    NonTargetClassConstraint,
    NonTargetConstraintSet,
)
from ue_framework.methods.sirc_malc_mechanism import SIRCMALCMechanismWorkflow


class _Carrier:
    def __init__(self) -> None:
        self.coefficients = torch.nn.Parameter(torch.tensor([0.1, -0.2]))


def test_backtracking_margin_observation_runs_without_autograd_graph() -> None:
    workflow = SIRCMALCMechanismWorkflow.__new__(SIRCMALCMechanismWorkflow)
    workflow.optimization = {"cgr_tolerance": 0.005}
    carrier = _Carrier()
    grad_modes: list[bool] = []

    def observe(_batch, active_carrier):
        grad_modes.append(torch.is_grad_enabled())
        margin = active_carrier.coefficients.sum()
        zero = margin * 0.0
        constraint = NonTargetClassConstraint(
            class_id=3,
            count=1,
            cls_margin=margin,
            box_margin=zero,
            cls_violation=zero,
            box_violation=zero,
            clean_probability_mean=0.8,
            adv_probability_mean=0.7,
        )
        return SimpleNamespace(
            constraints=NonTargetConstraintSet(
                constraints=(constraint,),
                status="active",
            )
        )

    workflow._observe = observe
    original = carrier.coefficients.detach().clone()
    margins = workflow._evaluate_class_margins(
        torch.tensor([0.3, 0.4]),
        batch=SimpleNamespace(),
        carrier=carrier,
    )

    assert grad_modes == [False]
    assert margins["class_3_cls"] == pytest.approx(0.7)
    assert torch.equal(carrier.coefficients.detach(), original)
