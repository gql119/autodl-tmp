from __future__ import annotations

import math

import pytest
import torch

from ue_framework.methods.dgcaip import (
    FrozenDGCAIPGradientCalibration,
    cooccurrence_geometry_risk,
    dgcaip_instance_preservation,
    divergence_guided_weights,
)


def test_geometry_risk_uses_person_union_and_normalized_distance() -> None:
    non_target = torch.tensor([0.0, 0.0, 10.0, 10.0])
    overlapping_people = torch.tensor(
        [[0.0, 0.0, 5.0, 10.0], [5.0, 0.0, 10.0, 5.0]]
    )
    assert cooccurrence_geometry_risk(non_target, overlapping_people) == pytest.approx(3.0)
    far_person = torch.tensor([[20.0, 0.0, 30.0, 10.0]])
    assert cooccurrence_geometry_risk(non_target, far_person) == pytest.approx(
        1.0 + math.exp(-1.0)
    )


def test_divergence_weights_are_detached_monotonic_bounded_and_mean_one() -> None:
    divergences = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    result = divergence_guided_weights(divergences, torch.ones(4))
    assert result.percentile_ranks.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert torch.all(result.weights[1:] > result.weights[:-1])
    assert result.weights.mean().item() == pytest.approx(1.0)
    assert result.weights.min().item() >= 0.5
    assert result.weights.max().item() <= 2.0
    assert result.weights.requires_grad is False


def test_small_batch_disables_divergence_ranking() -> None:
    result = divergence_guided_weights(
        torch.tensor([0.0, 10.0, 20.0]),
        torch.ones(3),
    )
    assert torch.allclose(result.hardness, torch.ones(3))
    assert torch.allclose(result.weights, torch.ones(3))


def test_joint_instance_loss_is_finite_and_backpropagates_only_to_poison() -> None:
    clean_logits = torch.zeros((1, 3, 20), requires_grad=True)
    poison_logits = clean_logits.detach().clone()
    poison_logits[0, 1:, 1] = -2.0
    poison_logits[0, 1:, 7] = 1.0
    poison_logits.requires_grad_(True)
    clean_boxes = torch.tensor(
        [[[0.0, 0.0, 5.0, 10.0], [4.0, 0.0, 10.0, 10.0], [4.0, 0.0, 10.0, 10.0]]],
        requires_grad=True,
    )
    poison_boxes = clean_boxes.detach().clone()
    poison_boxes[0, 1:, 2] = 8.0
    poison_boxes.requires_grad_(True)
    result = dgcaip_instance_preservation(
        clean_logits,
        poison_logits,
        clean_boxes,
        poison_boxes,
        torch.tensor([[14, 1, 1]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 1, 1]]),
        torch.tensor([[[14], [1]]]),
        torch.tensor([[[0.0, 0.0, 5.0, 10.0], [4.0, 0.0, 10.0, 10.0]]]),
        torch.tensor([[[True], [True]]]),
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.active_classes == (1,)
    assert result.coverage == 1.0
    term = result.instances[0]
    assert term.classification_damage.item() > term.classification_loss.item()
    assert term.box_damage.item() > term.box_loss.item()
    assert term.alignment_damage.item() > term.alignment_loss.item()
    assert term.classification_loss.item() > 0
    assert term.box_loss.item() > 0
    assert term.alignment_loss.item() > 0
    assert term.distribution_loss.item() > 0
    assert term.geometry_risk > 1.0
    assert term.weight == pytest.approx(1.0)
    result.loss.backward()
    assert clean_logits.grad is None
    assert clean_boxes.grad is None
    assert poison_logits.grad is not None and torch.isfinite(poison_logits.grad).all()
    assert poison_boxes.grad is not None and torch.isfinite(poison_boxes.grad).all()


def test_feature_switches_remove_geometry_and_hardness_without_removing_loss() -> None:
    clean_logits = torch.zeros((1, 5, 20))
    poison_logits = clean_logits.clone().requires_grad_(True)
    poison_logits.data[0, 1:, 1] = -1.0
    boxes = torch.tensor(
        [[[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0], [2.0, 0.0, 4.0, 2.0], [4.0, 0.0, 6.0, 2.0], [4.0, 0.0, 6.0, 2.0]]]
    )
    result = dgcaip_instance_preservation(
        clean_logits,
        poison_logits,
        boxes,
        boxes.clone().requires_grad_(True),
        torch.tensor([[14, 1, 1, 2, 2]]),
        torch.tensor([[True, True, True, True, True]]),
        torch.tensor([[0, 1, 1, 2, 2]]),
        torch.tensor([[[14], [1], [2]]]),
        torch.tensor([[[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0], [4.0, 0.0, 6.0, 2.0]]]),
        torch.tensor([[[True], [True], [True]]]),
        target_class_id=14,
        assignment_source="clean_real_tal",
        enable_geometry_risk=False,
        enable_divergence_hardness=False,
    )
    assert all(term.geometry_risk == 1.0 for term in result.instances)
    assert all(term.weight == pytest.approx(1.0) for term in result.instances)
    assert result.loss.item() > 0


def test_component_calibration_is_one_shot_warmup_only_and_serializable() -> None:
    calibration = FrozenDGCAIPGradientCalibration()
    weights = calibration.calibrate(
        {
            "classification": [2.0, 4.0, 6.0],
            "box": [1.0, 2.0, 3.0],
            "alignment": [4.0, 8.0, 12.0],
            "distribution": [0.5, 1.0, 1.5],
        },
        split="warmup",
    )
    assert weights == pytest.approx(
        {"classification": 1.0, "box": 2.0, "alignment": 0.5, "distribution": 4.0}
    )
    with pytest.raises(RuntimeError, match="already frozen"):
        calibration.calibrate(
            {name: [1.0] for name in weights},
            split="warmup",
        )
    restored = FrozenDGCAIPGradientCalibration()
    restored.load_state_dict(calibration.state_dict())
    assert restored.value == pytest.approx(weights)
