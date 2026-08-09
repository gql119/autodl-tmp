from __future__ import annotations

import pytest
import torch

from ue_framework.methods.malc import MALCInstanceResiduals
from ue_framework.methods.malc_calibration import (
    MALCGradientNormCalibrator,
    MALCPrototypeCalibrator,
)


def _residuals() -> MALCInstanceResiduals:
    vectors = (
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[0.0, 2.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    flags = (torch.ones(3, dtype=torch.bool),) * 2
    return MALCInstanceResiduals(
        vectors=vectors,
        assigned=flags,
        pooling_valid=flags,
        assignment_count=(torch.ones(3), torch.ones(3)),
        score_mass=(torch.ones(3), torch.ones(3)),
        image_indices=torch.tensor([0, 0, 1]),
        gt_indices=torch.tensor([0, 1, 0]),
    )


def test_prototype_calibration_is_deterministic_and_uses_q25_floor() -> None:
    outputs = []
    for _ in range(2):
        calibrator = MALCPrototypeCalibrator(
            num_scales=2,
            split_hash="fixed-s0",
            energy_floor_multiplier=0.5,
        )
        calibrator.update(_residuals(), split="calibration")
        outputs.append(calibrator.finalize())
    first, second = outputs
    assert first.calibration_hash == second.calibration_hash
    assert first.per_scale_vector_count == (3, 3)
    for left, right in zip(
        first.bank.direction_prototypes,
        second.bank.direction_prototypes,
    ):
        assert torch.equal(left, right)
        assert torch.allclose(left.norm(), torch.tensor(1.0))
    scale_zero_rms = _residuals().vectors[0].double().square().mean(dim=1).sqrt()
    expected_floor = 0.5 * float(torch.quantile(scale_zero_rms, 0.25))
    assert abs(first.bank.energy_floors[0] - expected_floor) < 1e-8


def test_prototype_calibration_rejects_heldout_update_and_second_finalize() -> None:
    calibrator = MALCPrototypeCalibrator(num_scales=2, split_hash="fixed-s0")
    with pytest.raises(ValueError, match="split='calibration'"):
        calibrator.update(_residuals(), split="heldout")
    calibrator.update(_residuals(), split="calibration")
    calibrator.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        calibrator.finalize()


def test_gradient_norm_calibration_matches_reference_and_is_repeatable() -> None:
    calibrations = []
    for _ in range(2):
        theta = torch.ones(4, requires_grad=True)
        calibrator = MALCGradientNormCalibrator(
            component_names=("easy_cls", "malc", "rms")
        )
        for factor in (1.0, 2.0, 3.0):
            calibrator.update(
                {
                    "easy_cls": (2.0 * factor * theta).sum(),
                    "malc": (1.0 * factor * theta).sum(),
                    "rms": (4.0 * factor * theta).sum(),
                },
                (theta,),
            )
        calibrations.append(calibrator.finalize())
    first, second = calibrations
    assert first.weights == {"malc": pytest.approx(2.0), "rms": pytest.approx(0.5)}
    assert first.calibration_hash == second.calibration_hash
    assert first.clipped_components == ()
    with pytest.raises(TypeError):
        first.weights["malc"] = 1.0


def test_gradient_calibration_fails_on_disconnected_component() -> None:
    theta = torch.ones(2, requires_grad=True)
    calibrator = MALCGradientNormCalibrator(
        component_names=("easy_cls", "malc")
    )
    with pytest.raises(RuntimeError, match="disconnected"):
        calibrator.update(
            {
                "easy_cls": theta.sum(),
                "malc": torch.tensor(1.0, requires_grad=True),
            },
            (theta,),
        )


def test_gradient_calibration_fails_if_over_half_components_clip() -> None:
    theta = torch.ones(2, requires_grad=True)
    calibrator = MALCGradientNormCalibrator(
        component_names=("easy_cls", "malc", "rms")
    )
    calibrator.update(
        {
            "easy_cls": theta.sum(),
            "malc": (1e-6 * theta).sum(),
            "rms": (1e6 * theta).sum(),
        },
        (theta,),
    )
    with pytest.raises(RuntimeError, match="clipping boundary"):
        calibrator.finalize()
