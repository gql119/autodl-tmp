from __future__ import annotations

import pytest
import torch

from ue_framework.methods.constraint_gradient_router import (
    backtrack_multi_parameter_update,
    flatten_parameter_tensors,
    route_multi_parameter_gradients,
    unflatten_parameter_tensor,
)


def test_flatten_unflatten_covers_every_omega_parameter_exactly() -> None:
    first = torch.zeros((2, 3), requires_grad=True)
    second = torch.zeros((4,), requires_grad=True)
    tensors = (torch.arange(6.0).reshape(2, 3), torch.arange(4.0))
    flattened = flatten_parameter_tensors(tensors, (first, second))
    restored = unflatten_parameter_tensor(flattened, (first, second))
    assert flattened.numel() == 10
    assert torch.equal(restored[0], tensors[0])
    assert torch.equal(restored[1], tensors[1])


def test_known_subspace_projection_and_explicit_protection_sign() -> None:
    first = torch.tensor([1.0, 1.0], requires_grad=True)
    second = torch.tensor([1.0], requires_grad=True)
    target = 2.0 * first[0] + 3.0 * first[1] + 4.0 * second[0]
    class_a = first[0]
    class_b = second[0]
    nla = (class_a + class_b) / 2.0
    result = route_multi_parameter_gradients(
        parameters=(first, second),
        target_loss=target,
        per_class_nla_losses={"1": class_a, "7": class_b},
        nla_loss=nla,
        nla_weight=0.25,
    )
    assert result.rank == 2
    assert result.projected_target_gradient.tolist() == pytest.approx([0.0, 3.0, 0.0])
    assert result.max_projected_row_dot <= 1.0e-5
    assert result.gradient.tolist() == pytest.approx([0.125, 3.0, 0.125])
    assert result.max_final_row_dot > 0  # final update includes NLA descent
    assert [tuple(item.shape) for item in result.parameter_gradients] == [(2,), (1,)]


def test_rank_deficiency_and_empty_active_classes_are_safe() -> None:
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    target = parameter[0] + parameter[1]
    same_a = parameter[0]
    same_b = 2.0 * parameter[0]
    deficient = route_multi_parameter_gradients(
        parameters=(parameter,),
        target_loss=target,
        per_class_nla_losses={"a": same_a, "b": same_b},
        nla_loss=(same_a + same_b) / 2.0,
        nla_weight=0.1,
    )
    assert deficient.rank == 1
    assert deficient.projected_target_gradient.tolist() == pytest.approx([0.0, 1.0])

    empty = route_multi_parameter_gradients(
        parameters=(parameter,),
        target_loss=target,
        per_class_nla_losses={},
        nla_loss=parameter.sum() * 0.0,
        nla_weight=0.1,
    )
    assert empty.rank == 0
    assert empty.gradient.tolist() == pytest.approx([1.0, 1.0])


def test_projection_collapse_becomes_protection_only() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    target = 2.0 * parameter[0]
    protect = parameter[0]
    result = route_multi_parameter_gradients(
        parameters=(parameter,),
        target_loss=target,
        per_class_nla_losses={"1": protect},
        nla_loss=protect,
        nla_weight=0.25,
    )
    assert result.mode == "protection_only"
    assert result.projected_target_norm == pytest.approx(0.0)
    assert result.gradient.item() == pytest.approx(0.25)


def test_nonlinear_backtracking_accepts_safe_step_or_skips_without_update() -> None:
    first = torch.tensor([1.0], requires_grad=True)
    second = torch.tensor([2.0], requires_grad=True)
    gradient = torch.tensor([1.0, 1.0])

    accepted = backtrack_multi_parameter_update(
        parameters=(first, second),
        flattened_gradient=gradient,
        step_size=0.04,
        evaluate_probability_drops=lambda candidate: {
            "1": float((1.0 - candidate[0]).abs().item())
        },
        tolerance=0.005,
        max_backtracks=5,
    )
    assert accepted.accepted
    assert accepted.attempts == 4
    assert accepted.step_size == pytest.approx(0.005)

    failed = backtrack_multi_parameter_update(
        parameters=(first, second),
        flattened_gradient=gradient,
        step_size=0.04,
        evaluate_probability_drops=lambda _candidate: {"1": 0.1},
        tolerance=0.005,
        max_backtracks=5,
    )
    assert not failed.accepted
    assert failed.status == "skip"
    assert torch.equal(failed.candidate[0], first.detach())
    assert torch.equal(failed.candidate[1], second.detach())
