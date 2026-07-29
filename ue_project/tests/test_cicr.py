from __future__ import annotations

import math

import pytest
import torch

from ue_framework.methods.cicr import (
    CICRPrototypeBank,
    classification_residuals,
)


def _features(
    *,
    batch: int = 4,
    channels: int = 6,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    clean: list[torch.Tensor] = []
    adv: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    for scale, size in enumerate((8, 4, 2)):
        generator = torch.Generator().manual_seed(20 + scale)
        clean_scale = torch.randn(
            (batch, channels, size, size),
            generator=generator,
            requires_grad=True,
        )
        direction = torch.linspace(0.1, 0.6, channels).view(1, channels, 1, 1)
        adv_scale = (clean_scale.detach() + direction).requires_grad_()
        gate = torch.ones((batch, 1, size, size))
        clean.append(clean_scale)
        adv.append(adv_scale)
        gates.append(gate)
    return clean, adv, gates


def test_train_prototype_update_and_heldout_freeze() -> None:
    clean, adv, gates = _features()
    residuals = classification_residuals(clean, adv, gates)
    bank = CICRPrototypeBank(num_scales=3, momentum=0.5)
    bank.calibrate_energy_floors(
        [vector.detach() for vector in residuals.vectors]
    )
    bank.update(residuals, split="train")
    before = [bank.prototype(scale) for scale in range(3)]

    result = bank.loss(residuals)
    assert torch.isfinite(result.loss)
    assert result.per_scale_valid_count == (4, 4, 4)
    result.loss.backward()
    assert all(item.grad is not None for item in adv)
    assert all(item.grad is None for item in clean)

    with pytest.raises(ValueError, match="split='train'"):
        bank.update(residuals, split="heldout")
    after = [bank.prototype(scale) for scale in range(3)]
    assert all(
        torch.equal(left, right)
        for left, right in zip(before, after)
        if left is not None and right is not None
    )


def test_zero_mask_and_zero_residual_do_not_fake_consistency() -> None:
    clean = [torch.zeros((2, 4, size, size)) for size in (8, 4, 2)]
    adv = [item.clone().requires_grad_() for item in clean]
    gates = [torch.zeros((2, 1, size, size)) for size in (8, 4, 2)]
    residuals = classification_residuals(clean, adv, gates)
    bank = CICRPrototypeBank(num_scales=3)
    bank._prototypes = [torch.ones(4) for _ in range(3)]

    result = bank.loss(residuals)
    assert result.per_scale_valid_count == (0, 0, 0)
    assert all(math.isnan(value) for value in result.per_scale_cosine)
    assert float(result.loss.detach()) == 0.0
    result.loss.backward()
    assert all(item.grad is not None for item in adv)


def test_energy_floor_uses_half_warmup_q25() -> None:
    bank = CICRPrototypeBank(num_scales=3)
    scale = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    floors = bank.calibrate_energy_floors([scale, scale * 2, scale * 3])
    assert floors == pytest.approx((0.875, 1.75, 2.625))


def test_gate_shape_mismatch_fails_closed() -> None:
    clean, adv, gates = _features()
    gates[1] = torch.ones((4, 1, 5, 5))
    with pytest.raises(ValueError, match="does not align"):
        classification_residuals(clean, adv, gates)
