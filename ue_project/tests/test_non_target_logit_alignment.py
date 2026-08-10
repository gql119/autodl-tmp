from __future__ import annotations

import pytest
import torch

from ue_framework.methods.non_target_logit_alignment import (
    FrozenNLAGradientCalibration,
    class_balanced_non_target_logit_alignment,
)


def _inputs():
    clean = torch.zeros((1, 6, 20))
    poison = clean.clone().requires_grad_(True)
    labels = torch.tensor([[14, 1, 1, 1, 7, 3]])
    foreground = torch.tensor([[True, True, True, True, True, False]])
    return clean, poison, labels, foreground


def test_exact_clean_alignment_is_zero_and_excludes_target_and_background() -> None:
    clean, poison, labels, foreground = _inputs()
    result = class_balanced_non_target_logit_alignment(
        clean,
        poison,
        labels,
        foreground,
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.loss.item() == 0.0
    assert result.active_classes == (1, 7)
    assert result.per_class_count == {1: 3, 7: 1}
    result.loss.backward()
    assert poison.grad is not None


def test_macro_average_prevents_frequent_class_from_dominating() -> None:
    clean, poison, labels, foreground = _inputs()
    poison = poison.detach()
    poison[0, 1:4, 1] = 1.0
    poison[0, 4, 7] = 2.0
    poison.requires_grad_(True)
    result = class_balanced_non_target_logit_alignment(
        clean,
        poison,
        labels,
        foreground,
        target_class_id=14,
        assignment_source="clean_real_tal",
        beta=1.0,
    )
    # SmoothL1(1)=0.5 and SmoothL1(2)=1.5; class macro mean is 1.0.
    assert result.per_class_loss[1].item() == pytest.approx(0.5)
    assert result.per_class_loss[7].item() == pytest.approx(1.5)
    assert result.loss.item() == pytest.approx(1.0)
    result.loss.backward()
    # Three class-1 anchors jointly receive the same total class weight as the
    # one class-7 anchor, modulo their different SmoothL1 derivative magnitude.
    assert poison.grad[0, 1:4, 1].abs().sum().item() == pytest.approx(0.5)
    assert poison.grad[0, 4, 7].abs().item() == pytest.approx(0.5)


def test_clean_teacher_is_detached_and_pseudo_assignments_fail_closed() -> None:
    clean, poison, labels, foreground = _inputs()
    clean.requires_grad_(True)
    with pytest.raises(ValueError, match="pseudo is forbidden"):
        class_balanced_non_target_logit_alignment(
            clean,
            poison,
            labels,
            foreground,
            target_class_id=14,
            assignment_source="pseudo",
        )
    result = class_balanced_non_target_logit_alignment(
        clean,
        poison,
        labels,
        foreground,
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    result.loss.backward()
    assert clean.grad is None


def test_no_active_non_target_is_differentiable_zero() -> None:
    clean = torch.zeros((1, 2, 20))
    poison = clean.clone().requires_grad_(True)
    result = class_balanced_non_target_logit_alignment(
        clean,
        poison,
        torch.tensor([[14, 0]]),
        torch.tensor([[True, False]]),
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.active_classes == ()
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert poison.grad is not None
    assert poison.grad.abs().sum() == 0


def test_lambda_calibration_is_one_shot_median_scaled_and_serializable() -> None:
    calibration = FrozenNLAGradientCalibration(target_ratio=0.25)
    value = calibration.calibrate(
        [2.0, 4.0, 100.0],
        [1.0, 2.0, 10.0],
        split="warmup",
    )
    assert value == pytest.approx(0.5)
    assert calibration.was_clipped is False
    with pytest.raises(RuntimeError, match="already frozen"):
        calibration.calibrate([1.0], [1.0], split="warmup")
    restored = FrozenNLAGradientCalibration(target_ratio=0.25)
    restored.load_state_dict(calibration.state_dict())
    assert restored.value == pytest.approx(0.5)
