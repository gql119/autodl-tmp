from __future__ import annotations

import pytest
import torch

from ue_framework.methods.instance_cicr import (
    FrozenInstanceCICRBank,
    InstanceClassificationResiduals,
    instance_classification_residuals,
)


def _residuals(
    values=(1.0, 1.0),
    valid=(True, True),
) -> InstanceClassificationResiduals:
    vectors = []
    flags = []
    masses = []
    for channels in (2, 3, 4):
        vector = torch.zeros((2, channels), dtype=torch.float32)
        vector[0, 0] = values[0]
        vector[1, 0] = values[1]
        vectors.append(vector)
        flags.append(torch.tensor(valid, dtype=torch.bool))
        masses.append(torch.tensor([2.0, 1.0]))
    return InstanceClassificationResiduals(
        vectors=tuple(vectors),
        gate_valid=tuple(flags),
        gate_mass=tuple(masses),
        image_indices=torch.tensor([0, 0]),
        gt_indices=torch.tensor([1, 3]),
    )


def test_real_gt_assignment_produces_separate_instance_residuals() -> None:
    clean = [torch.zeros((1, 2, 1, 4))]
    adv = [torch.tensor([[[[1.0, 3.0, 7.0, 9.0]], [[2.0, 4.0, 8.0, 10.0]]]])]
    pag = torch.tensor([[True, True, True, False]])
    gt_idx = torch.tensor([[1, 1, 3, 3]])
    result = instance_classification_residuals(
        clean,
        adv,
        pag,
        gt_idx,
        ((1, 3),),
    )
    assert result.vectors[0] == pytest.approx(
        torch.tensor([[2.0, 3.0], [7.0, 8.0]])
    )
    assert result.gate_mass[0].tolist() == [2.0, 1.0]


def test_real_assignment_scores_normalize_each_instance_pool() -> None:
    clean = [torch.zeros((1, 1, 1, 3))]
    adv = [torch.tensor([[[[1.0, 5.0, 9.0]]]])]
    result = instance_classification_residuals(
        clean,
        adv,
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 0, 1]]),
        ((0, 1),),
        assigned_scores=torch.tensor([[1.0, 3.0, 2.0]]),
    )
    assert result.vectors[0][:, 0].tolist() == pytest.approx([4.0, 9.0])
    assert result.gate_mass[0].tolist() == pytest.approx([4.0, 2.0])


def test_frozen_bank_is_calibration_only() -> None:
    bank = FrozenInstanceCICRBank()
    with pytest.raises(ValueError, match="calibration split"):
        bank.fit([_residuals()], split="heldout")
    bank.fit([_residuals()], split="calibration")
    assert bank.calibration_instance_count == 2
    with pytest.raises(RuntimeError, match="already fitted"):
        bank.fit([_residuals()], split="calibration")


def test_cicr_direction_and_energy_floor_are_separate_and_differentiable() -> None:
    bank = FrozenInstanceCICRBank(energy_floor_multiplier=0.5)
    bank.fit([_residuals(values=(2.0, 2.0))], split="calibration")
    active = _residuals(values=(0.2, 2.0))
    active = InstanceClassificationResiduals(
        vectors=tuple(value.clone().requires_grad_(True) for value in active.vectors),
        gate_valid=active.gate_valid,
        gate_mass=active.gate_mass,
        image_indices=active.image_indices,
        gt_indices=active.gt_indices,
    )
    result = bank.compute(active, energy_weight=1.0)
    assert result.energy_floor_loss.item() > 0
    assert result.valid_instance_count == 1
    assert result.low_energy_ratio == pytest.approx(0.5)
    result.loss.backward()
    assert all(value.grad is not None for value in active.vectors)
    assert all(torch.isfinite(value.grad).all() for value in active.vectors)
    assert sum(value.grad.abs().sum() for value in active.vectors) > 0


def test_instance_equal_weighting_does_not_overweight_more_scales() -> None:
    bank = FrozenInstanceCICRBank(energy_floor_multiplier=0.1)
    bank.fit([_residuals()], split="calibration")
    active = _residuals()
    vectors = list(active.vectors)
    vectors[0] = vectors[0].clone()
    vectors[0][1, 0] = -1.0
    flags = list(active.gate_valid)
    flags[1] = torch.tensor([True, False])
    flags[2] = torch.tensor([True, False])
    changed = InstanceClassificationResiduals(
        vectors=tuple(vectors),
        gate_valid=tuple(flags),
        gate_mass=active.gate_mass,
        image_indices=active.image_indices,
        gt_indices=active.gt_indices,
    )
    result = bank.compute(changed, energy_weight=0.0)
    # Instance 0 has loss 0 across three scales; instance 1 has loss 2 at one
    # valid scale. Equal instance weighting therefore yields (0 + 2) / 2 = 1.
    assert result.direction_loss.item() == pytest.approx(1.0, abs=1e-6)


def test_frozen_bank_state_roundtrip() -> None:
    source = FrozenInstanceCICRBank()
    source.fit([_residuals()], split="calibration")
    restored = FrozenInstanceCICRBank()
    restored.load_state_dict(source.state_dict())
    assert restored.energy_floors == pytest.approx(source.energy_floors)
    assert restored.compute(_residuals()).loss.item() == pytest.approx(0.0, abs=1e-6)
