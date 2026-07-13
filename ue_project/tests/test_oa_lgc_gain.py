import pytest
import torch

from oa_lgc.gains import ClassGainInput, authorized_learning_gain, carrier_query_loss, target_learning_gain


def test_target_learning_gain_sign():
    result = target_learning_gain(torch.tensor(1.0), torch.tensor(0.4), torch.tensor(0.9), 0.5, 1e-4)
    assert result.clean_gain > result.poison_gain
    assert result.valid


def test_target_gain_ratio():
    result = target_learning_gain(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.9), 0.1, 1e-4)
    assert result.ratio is not None
    assert torch.allclose(result.ratio, torch.tensor(0.2))
    assert torch.allclose(result.protect_loss, torch.tensor(0.1))


def test_invalid_clean_gain_handling():
    result = target_learning_gain(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.9), 0.1, 1e-3)
    assert not result.valid and result.ratio is None
    assert result.invalid_reason == "invalid_clean_gain"
    assert result.protect_loss == 0


def test_per_class_authorized_gain():
    inputs = {1: ClassGainInput(torch.tensor(1.0), torch.tensor(0.6), torch.tensor(0.6), 2, 2)}
    result = authorized_learning_gain(inputs, 14, rho_k=0.05, min_valid_class_gain=1e-4, minimum_class_samples=1)
    assert result.valid_class_ids == (1,)
    assert torch.allclose(result.classes[1].normalized_gap, torch.tensor(0.0))
    assert result.loss == 0


def test_missing_class_not_averaged():
    inputs = {
        1: ClassGainInput(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.6), 2, 2),
        2: ClassGainInput(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5), 0, 0),
    }
    result = authorized_learning_gain(inputs, 14, 0.0, 1e-4, 1)
    assert result.valid_class_ids == (1,)
    assert result.invalid_class_ids == (2,)
    assert torch.allclose(result.loss, torch.tensor(0.2))


def test_negative_gain_logging():
    inputs = {3: ClassGainInput(torch.tensor(1.0), torch.tensor(1.5), torch.tensor(1.4), 1, 1)}
    result = authorized_learning_gain(inputs, 14, 0.0, 1e-4, 1)
    assert result.classes[3].valid
    assert result.classes[3].clean_gain < 0
    assert result.classes[3].normalized_gap is not None


def test_carrier_query_is_disjoint():
    support_ids = {"s0", "s1"}
    query_ids = {"q0", "q1"}
    assert support_ids.isdisjoint(query_ids)
    assert carrier_query_loss(torch.tensor(0.7)) == pytest.approx(0.7)


def test_gain_metric_gradient_flow():
    delta = torch.tensor(0.1, requires_grad=True)
    initial = torch.tensor(1.0)
    result = target_learning_gain(initial, torch.tensor(0.5), 0.9 + delta.square(), 0.0, 1e-4)
    assert result.valid
    result.protect_loss.backward()
    assert delta.grad is not None and torch.isfinite(delta.grad) and delta.grad.abs() > 0


def test_denominator_too_small_is_invalid_not_clipped():
    inputs = {1: ClassGainInput(torch.tensor(1.0), torch.tensor(1.0 - 1e-9), torch.tensor(0.5), 1, 1)}
    result = authorized_learning_gain(inputs, 14, 0.0, min_valid_class_gain=1e-4, minimum_class_samples=1)
    assert not result.classes[1].valid
    assert result.classes[1].normalized_gap is None

