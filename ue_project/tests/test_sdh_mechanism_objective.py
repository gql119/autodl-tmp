from __future__ import annotations

import pytest
import torch

from ue_framework.methods.sdh_mechanism import (
    FrozenTargetGradientCalibration,
    adapter_parameters,
    compose_sdh_target_objective,
)
from ue_framework.methods.semantic_hiding_carrier import SemanticHidingCarrier


def test_t0_and_t1_objectives_have_independent_switches() -> None:
    parameter = torch.tensor(1.0, requires_grad=True)
    common = {
        "easy": parameter,
        "reveal": 2 * parameter,
        "rms": 3 * parameter,
        "weights": {
            "easy": 1.0,
            "reveal": 0.5,
            "rms": 0.25,
            "dlfc": 2.0,
            "cicr": 3.0,
            "floor": 4.0,
        },
    }
    t0 = compose_sdh_target_objective(
        **common,
        dlfc=None,
        cicr=None,
        floor=None,
        enable_dlfc=False,
        enable_cicr=False,
    )
    assert t0.active_components == ("easy", "reveal", "rms")
    assert t0.loss.item() == pytest.approx(2.75)
    t1 = compose_sdh_target_objective(
        **common,
        dlfc=4 * parameter,
        cicr=5 * parameter,
        floor=6 * parameter,
        enable_dlfc=True,
        enable_cicr=True,
    )
    assert t1.active_components == (
        "easy", "reveal", "rms", "dlfc", "cicr", "floor"
    )
    assert t1.loss.item() == pytest.approx(49.75)


def test_enabled_component_cannot_silently_disappear() -> None:
    value = torch.tensor(1.0)
    with pytest.raises(ValueError, match="enabled but its loss is missing"):
        compose_sdh_target_objective(
            easy=value,
            reveal=value,
            rms=value,
            dlfc=None,
            cicr=None,
            floor=None,
            weights={"easy": 1, "reveal": 1, "rms": 1},
            enable_dlfc=True,
            enable_cicr=False,
        )


def test_target_gradient_calibration_is_one_shot_and_median_based() -> None:
    calibration = FrozenTargetGradientCalibration()
    weights = calibration.calibrate(
        {
            "easy": [2.0, 4.0, 6.0],
            "reveal": [1.0, 2.0, 3.0],
            "rms": [4.0, 8.0, 12.0],
        },
        split="warmup",
    )
    assert weights == pytest.approx({"easy": 1.0, "reveal": 2.0, "rms": 0.5})
    with pytest.raises(RuntimeError, match="already frozen"):
        calibration.calibrate({"easy": [1.0]}, split="warmup")


def test_inactive_hinge_gets_frozen_neutral_weight() -> None:
    calibration = FrozenTargetGradientCalibration()
    weights = calibration.calibrate(
        {"easy": [2.0, 3.0], "floor": [0.0, 0.0]}, split="warmup"
    )
    assert weights["floor"] == 1.0
    assert calibration.state_dict()["clipped"]["floor"] is True


def test_omega_is_the_complete_adapter_and_excludes_frozen_trunk() -> None:
    carrier = SemanticHidingCarrier(input_size=32, width=8, coupling_blocks=2)
    carrier.freeze_for_detector_optimization()
    omega = adapter_parameters(carrier)
    assert sum(value.numel() for value in omega) == sum(
        value.numel() for value in carrier.adapter.parameters()
    )
    assert all(value.requires_grad for value in omega)
    assert all(not value.requires_grad for value in carrier.hiding_trunk.parameters())
    assert all(not value.requires_grad for value in carrier.reveal_decoder.parameters())
