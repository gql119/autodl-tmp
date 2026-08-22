from __future__ import annotations

import math

import pytest
import torch

from ue_framework.methods.non_target_distribution_divergence import (
    _clamp_nonnegative_roundoff,
    non_target_bernoulli_divergence,
)


def test_identical_logits_have_zero_divergence_and_finite_backward() -> None:
    clean = torch.randn((2, 5, 20))
    poison = clean.clone().requires_grad_(True)
    result = non_target_bernoulli_divergence(
        clean,
        poison,
        target_class_id=14,
    )
    assert result.js_per_anchor.shape == (2, 5)
    assert result.clean_to_poison_kl_per_anchor.shape == (2, 5)
    assert torch.allclose(result.js_per_anchor, torch.zeros_like(result.js_per_anchor))
    assert torch.allclose(
        result.clean_to_poison_kl_per_anchor,
        torch.zeros_like(result.clean_to_poison_kl_per_anchor),
    )
    result.js_per_anchor.mean().backward()
    assert poison.grad is not None
    assert torch.isfinite(poison.grad).all()


def test_nearly_identical_float32_logits_clamp_negative_roundoff() -> None:
    torch.manual_seed(0)
    clean = torch.randn((2, 32, 20))
    poison = (clean + 1.0e-4 * torch.randn_like(clean)).requires_grad_(True)
    result = non_target_bernoulli_divergence(
        clean,
        poison,
        target_class_id=14,
    )
    assert torch.isfinite(result.js_per_anchor).all()
    assert torch.isfinite(result.clean_to_poison_kl_per_anchor).all()
    assert bool((result.js_per_anchor >= 0).all())
    assert bool((result.clean_to_poison_kl_per_anchor >= 0).all())
    result.js_per_anchor.mean().backward()
    assert poison.grad is not None
    assert torch.isfinite(poison.grad).all()


def test_materially_negative_divergence_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="Materially negative"):
        _clamp_nonnegative_roundoff(
            torch.tensor([-1.0e-3]),
            name="test divergence",
        )


def test_js_is_symmetric_bounded_and_one_way_kl_is_diagnostic() -> None:
    first = torch.tensor([[[3.0, -2.0, 1.0]]])
    second = torch.tensor([[[-1.0, 4.0, -3.0]]])
    forward = non_target_bernoulli_divergence(
        first,
        second,
        target_class_id=2,
    )
    reverse = non_target_bernoulli_divergence(
        second,
        first,
        target_class_id=2,
    )
    assert forward.js_per_anchor.item() == pytest.approx(
        reverse.js_per_anchor.item()
    )
    assert 0.0 <= forward.js_per_anchor.item() <= math.log(2.0)
    assert forward.clean_to_poison_kl_per_anchor.item() != pytest.approx(
        reverse.clean_to_poison_kl_per_anchor.item()
    )


def test_target_dimension_is_excluded_and_clean_teacher_is_detached() -> None:
    clean = torch.zeros((1, 2, 20), requires_grad=True)
    poison = clean.detach().clone()
    poison[..., 14] = 100.0
    poison.requires_grad_(True)
    result = non_target_bernoulli_divergence(
        clean,
        poison,
        target_class_id=14,
    )
    assert result.js_per_anchor.abs().sum().item() == 0.0
    result.js_per_anchor.sum().backward()
    assert clean.grad is None
    assert poison.grad is not None
    assert poison.grad[..., 14].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    ("clean", "poison", "message"),
    [
        (torch.zeros((1, 3)), torch.zeros((1, 4)), "same"),
        (torch.tensor([[float("nan"), 0.0]]), torch.zeros((1, 2)), "finite"),
    ],
)
def test_invalid_inputs_fail_closed(
    clean: torch.Tensor,
    poison: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        non_target_bernoulli_divergence(
            clean,
            poison,
            target_class_id=0,
        )


def test_temperature_and_probability_epsilon_are_validated() -> None:
    logits = torch.zeros((1, 20))
    with pytest.raises(ValueError, match="temperature"):
        non_target_bernoulli_divergence(
            logits,
            logits,
            target_class_id=14,
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="epsilon"):
        non_target_bernoulli_divergence(
            logits,
            logits,
            target_class_id=14,
            epsilon=0.5,
        )
