from __future__ import annotations

import pytest
import torch

from ue_framework.methods.cooccurrence_instance_preservation import (
    collect_cooccurring_non_target_instances,
)


def _base_inputs():
    clean = torch.zeros((2, 7, 20))
    poison = clean.clone()
    assigned = torch.tensor(
        [
            [14, 1, 1, 1, 7, 7, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ]
    )
    foreground = torch.tensor(
        [
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, True, False],
        ]
    )
    target_gt_idx = torch.tensor(
        [
            [0, 1, 1, 1, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    gt_labels = torch.tensor(
        [
            [[14], [1], [7]],
            [[1], [0], [0]],
        ]
    )
    mask_gt = torch.tensor(
        [
            [[True], [True], [True]],
            [[True], [False], [False]],
        ]
    )
    return (
        clean,
        poison,
        assigned,
        foreground,
        target_gt_idx,
        gt_labels,
        mask_gt,
    )


def test_collects_only_non_target_instances_from_person_cooccurring_images() -> None:
    inputs = _base_inputs()
    poison = inputs[1]
    poison[0, 1:4, 1] = 2.0
    poison[0, 4:6, 7] = -3.0
    poison.requires_grad_(True)
    result = collect_cooccurring_non_target_instances(
        inputs[0],
        poison,
        *inputs[2:],
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert [(item.gt_index, item.class_id, item.positive_count) for item in result.instances] == [
        (1, 1, 3),
        (2, 7, 2),
    ]
    assert result.active_classes == (1, 7)
    assert result.per_class_instance_count == {1: 1, 7: 1}
    assert result.eligible_instance_count == 2
    assert result.covered_instance_count == 2
    assert result.coverage == 1.0
    result.loss.backward()
    assert poison.grad is not None
    assert torch.isfinite(poison.grad).all()


def test_instance_then_class_average_ignores_positive_anchor_count() -> None:
    inputs = _base_inputs()
    poison = inputs[1]
    poison[0, 1:4, 1] = 2.0
    poison[0, 4:6, 7] = 2.0
    poison.requires_grad_(True)
    result = collect_cooccurring_non_target_instances(
        inputs[0],
        poison,
        *inputs[2:],
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.per_class_loss[1].item() == pytest.approx(
        result.per_class_loss[7].item()
    )
    result.loss.backward()
    class_one_gradient = poison.grad[0, 1:4].abs().sum().item()
    class_seven_gradient = poison.grad[0, 4:6].abs().sum().item()
    assert class_one_gradient == pytest.approx(class_seven_gradient)


def test_missing_clean_positive_is_reported_in_coverage() -> None:
    inputs = list(_base_inputs())
    inputs[3][0, 4:6] = False
    poison = inputs[1].requires_grad_(True)
    result = collect_cooccurring_non_target_instances(
        inputs[0],
        poison,
        *inputs[2:],
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.eligible_instance_count == 2
    assert result.covered_instance_count == 1
    assert result.skipped_no_positive_count == 1
    assert result.coverage == pytest.approx(0.5)


def test_no_person_cooccurrence_is_differentiable_zero() -> None:
    inputs = _base_inputs()
    poison = inputs[1][1:].clone().requires_grad_(True)
    result = collect_cooccurring_non_target_instances(
        inputs[0][1:],
        poison,
        inputs[2][1:],
        inputs[3][1:],
        inputs[4][1:],
        inputs[5][1:],
        inputs[6][1:],
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.instances == ()
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert poison.grad is not None
    assert poison.grad.abs().sum().item() == 0.0


def test_invalid_assignment_source_and_gt_index_fail_closed() -> None:
    inputs = list(_base_inputs())
    with pytest.raises(ValueError, match="clean_real_tal"):
        collect_cooccurring_non_target_instances(
            *inputs,
            target_class_id=14,
            assignment_source="poison_tal",
        )
    inputs[4][0, 1] = 9
    with pytest.raises(ValueError, match="invalid GT index"):
        collect_cooccurring_non_target_instances(
            *inputs,
            target_class_id=14,
            assignment_source="clean_real_tal",
        )


def test_assigned_class_must_match_clean_gt() -> None:
    inputs = list(_base_inputs())
    inputs[2][0, 1] = 2
    with pytest.raises(ValueError, match="disagrees"):
        collect_cooccurring_non_target_instances(
            *inputs,
            target_class_id=14,
            assignment_source="clean_real_tal",
        )


def test_background_assignment_indices_are_not_dereferenced() -> None:
    inputs = list(_base_inputs())
    inputs[4][~inputs[3]] = 999
    result = collect_cooccurring_non_target_instances(
        *inputs,
        target_class_id=14,
        assignment_source="clean_real_tal",
    )
    assert result.covered_instance_count == 2
