from __future__ import annotations

import torch

from ue_framework.methods.instance_cicr import (
    fit_instance_prototype_bank,
    instance_cicr,
    instance_classification_residuals,
    target_gt_indices_from_labels,
)


def test_two_instances_remain_separate() -> None:
    clean = [torch.zeros((1, 2, 1, 4))]
    adv = [
        torch.tensor(
            [[[[1.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]]]
        ).requires_grad_()
    ]
    residuals = instance_classification_residuals(
        clean,
        adv,
        pag_gate=torch.ones((1, 4), dtype=torch.bool),
        target_gt_idx=torch.tensor([[0, 0, 1, 1]]),
        target_gt_indices_by_image=((0, 1),),
    )
    assert torch.equal(
        residuals.vectors[0],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.equal(residuals.gate_mass[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(residuals.image_indices, torch.tensor([0, 0]))
    assert torch.equal(residuals.gt_indices, torch.tensor([0, 1]))
    bank = fit_instance_prototype_bank(residuals, momentum=0.9)
    result = instance_cicr(residuals, bank)
    result.loss.backward()
    assert adv[0].grad is not None
    assert torch.isfinite(adv[0].grad).all()


def test_zero_assignment_is_retained_in_coverage_denominator() -> None:
    clean = [torch.zeros((1, 2, 1, 2))]
    adv = [torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])]
    residuals = instance_classification_residuals(
        clean,
        adv,
        pag_gate=torch.tensor([[True, False]]),
        target_gt_idx=torch.tensor([[0, 1]]),
        target_gt_indices_by_image=((0, 1),),
    )
    bank = fit_instance_prototype_bank(residuals, momentum=0.9)
    result = instance_cicr(residuals, bank)
    assert result.total_instance_count == 2
    assert result.valid_instance_count == 1
    assert result.valid_instance_coverage == 0.5
    assert result.missing_assignment_ratio == 0.5
    assert torch.isnan(result.per_instance_cosine[1])
    assert result.per_instance_cosine.device == residuals.vectors[0].device
    assert result.per_instance_cosine.dtype == residuals.vectors[0].dtype


def test_multi_scale_grouping_and_label_index_extraction() -> None:
    gt_indices = target_gt_indices_from_labels(
        torch.tensor([[[14], [11], [14]], [[11], [14], [0]]]),
        torch.tensor([[[1], [1], [1]], [[1], [1], [0]]]),
        target_class_id=14,
    )
    assert gt_indices == ((0, 2), (1,))

    clean = [
        torch.zeros((2, 1, 1, 2)),
        torch.zeros((2, 1, 1, 1)),
    ]
    adv = [
        torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]]),
        torch.tensor([[[[5.0]]], [[[6.0]]]]),
    ]
    residuals = instance_classification_residuals(
        clean,
        adv,
        pag_gate=torch.ones((2, 3), dtype=torch.bool),
        target_gt_idx=torch.tensor([[0, 2, 2], [1, 1, 1]]),
        target_gt_indices_by_image=gt_indices,
    )
    assert len(residuals.vectors) == 2
    assert residuals.vectors[0].shape == (3, 1)
    assert residuals.vectors[1].shape == (3, 1)
    assert torch.equal(residuals.gate_valid[0], torch.tensor([True, True, True]))
    assert torch.equal(residuals.gate_valid[1], torch.tensor([False, True, True]))


def test_prototype_update_is_train_only_and_heldout_is_frozen() -> None:
    clean = [torch.zeros((1, 2, 1, 2))]
    adv = [torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])]
    residuals = instance_classification_residuals(
        clean,
        adv,
        pag_gate=torch.ones((1, 2), dtype=torch.bool),
        target_gt_idx=torch.tensor([[0, 1]]),
        target_gt_indices_by_image=((0, 1),),
    )
    bank = fit_instance_prototype_bank(residuals, momentum=0.9)
    before = bank.prototype(0)
    try:
        bank.update(residuals.as_classification_residuals(), split="heldout")
    except ValueError as error:
        assert "split='train'" in str(error)
    else:
        raise AssertionError("Held-out prototype update must fail closed.")
    assert torch.equal(before, bank.prototype(0))
