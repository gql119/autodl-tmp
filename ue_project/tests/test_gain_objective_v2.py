import torch

from ue_framework.methods.multitrajectory_gain.learning_gain import compute_learning_gain_objective_v2


def _losses(p, a, s):
    return {
        "protected": torch.tensor(float(p), requires_grad=True),
        "authorized": torch.tensor(float(a), requires_grad=True),
        "shared": torch.tensor(float(s), requires_grad=True),
    }


def test_gain_objective_v2_has_zero_preservation_penalty_inside_tolerance():
    initial = _losses(10, 10, 10)
    clean = _losses(8, 8, 8)
    poison = _losses(9, 7.95, 8.04)
    out = compute_learning_gain_objective_v2(
        initial,
        clean,
        poison,
        {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1},
        {"protected_support_batches": 3, "authorized_support_batches": 1},
        {"protected": 2.0, "authorized": 2.0, "shared": 2.0},
        authorized_tolerance=0.1,
        shared_tolerance=0.1,
    )
    assert out.authorized_loss.item() == 0.0
    assert out.shared_loss.item() == 0.0


def test_gain_objective_v2_penalty_increases_outside_tolerance():
    initial = _losses(10, 10, 10)
    clean = _losses(8, 8, 8)
    small = _losses(9, 7.95, 8.0)
    large = _losses(9, 7.0, 8.0)
    counts = {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1}
    support = {"protected_support_batches": 3, "authorized_support_batches": 1}
    scales = {"protected": 2.0, "authorized": 2.0, "shared": 2.0}
    out_small = compute_learning_gain_objective_v2(initial, clean, small, counts, support, scales, authorized_tolerance=0.1)
    out_large = compute_learning_gain_objective_v2(initial, clean, large, counts, support, scales, authorized_tolerance=0.1)
    assert out_large.authorized_loss.item() > out_small.authorized_loss.item()


def test_gain_objective_v2_protected_only_finite_difference_direction():
    delta = torch.tensor(0.0, requires_grad=True)
    initial = {"protected": delta * 0 + 10, "authorized": delta * 0 + 10, "shared": delta * 0 + 10}
    clean = {"protected": delta * 0 + 8, "authorized": delta * 0 + 8, "shared": delta * 0 + 8}
    poison = {"protected": 8.0 + delta, "authorized": delta * 0 + 8, "shared": delta * 0 + 8}
    out = compute_learning_gain_objective_v2(
        initial,
        clean,
        poison,
        {"protected_positive_count": 1, "authorized_positive_count": 0, "background_count": 0},
        {"protected_support_batches": 3, "authorized_support_batches": 0},
        {"protected": 2.0, "authorized": 2.0, "shared": 2.0},
    )
    grad = torch.autograd.grad(out.total_loss, delta)[0]
    next_delta = (delta - 0.1 * grad).detach().requires_grad_(True)
    next_poison = {"protected": 8.0 + next_delta, "authorized": next_delta * 0 + 8, "shared": next_delta * 0 + 8}
    out_next = compute_learning_gain_objective_v2(
        {"protected": next_delta * 0 + 10, "authorized": next_delta * 0 + 10, "shared": next_delta * 0 + 10},
        {"protected": next_delta * 0 + 8, "authorized": next_delta * 0 + 8, "shared": next_delta * 0 + 8},
        next_poison,
        {"protected_positive_count": 1, "authorized_positive_count": 0, "background_count": 0},
        {"protected_support_batches": 3, "authorized_support_batches": 0},
        {"protected": 2.0, "authorized": 2.0, "shared": 2.0},
    )
    assert out_next.d_protected.item() > out.d_protected.item()
    assert out_next.total_loss.item() < out.total_loss.item()


def test_gain_objective_v2_authorized_only_direction_reduces_deviation():
    delta = torch.tensor(0.5, requires_grad=True)
    initial = {"protected": delta * 0 + 10, "authorized": delta * 0 + 10, "shared": delta * 0 + 10}
    clean = {"protected": delta * 0 + 8, "authorized": delta * 0 + 8, "shared": delta * 0 + 8}
    poison = {"protected": delta * 0 + 8, "authorized": 8.0 - delta, "shared": delta * 0 + 8}
    counts = {"protected_positive_count": 0, "authorized_positive_count": 1, "background_count": 0}
    support = {"protected_support_batches": 0, "authorized_support_batches": 1}
    out = compute_learning_gain_objective_v2(initial, clean, poison, counts, support, {"protected": 2.0, "authorized": 2.0, "shared": 2.0}, authorized_tolerance=0.1)
    grad = torch.autograd.grad(out.total_loss, delta)[0]
    next_delta = (delta - 0.1 * grad).detach().requires_grad_(True)
    next_poison = {"protected": next_delta * 0 + 8, "authorized": 8.0 - next_delta, "shared": next_delta * 0 + 8}
    out_next = compute_learning_gain_objective_v2(
        {"protected": next_delta * 0 + 10, "authorized": next_delta * 0 + 10, "shared": next_delta * 0 + 10},
        {"protected": next_delta * 0 + 8, "authorized": next_delta * 0 + 8, "shared": next_delta * 0 + 8},
        next_poison,
        counts,
        support,
        {"protected": 2.0, "authorized": 2.0, "shared": 2.0},
        authorized_tolerance=0.1,
    )
    assert out_next.e_authorized.item() < out.e_authorized.item()


def test_gain_objective_v2_shared_only_direction_reduces_deviation():
    delta = torch.tensor(0.5, requires_grad=True)
    initial = {"protected": delta * 0 + 10, "authorized": delta * 0 + 10, "shared": delta * 0 + 10}
    clean = {"protected": delta * 0 + 8, "authorized": delta * 0 + 8, "shared": delta * 0 + 8}
    poison = {"protected": delta * 0 + 8, "authorized": delta * 0 + 8, "shared": 8.0 - delta}
    counts = {"protected_positive_count": 0, "authorized_positive_count": 0, "background_count": 1}
    support = {"protected_support_batches": 0, "authorized_support_batches": 0}
    out = compute_learning_gain_objective_v2(initial, clean, poison, counts, support, {"protected": 2.0, "authorized": 2.0, "shared": 2.0}, shared_tolerance=0.1)
    grad = torch.autograd.grad(out.total_loss, delta)[0]
    next_delta = (delta - 0.1 * grad).detach().requires_grad_(True)
    next_poison = {"protected": next_delta * 0 + 8, "authorized": next_delta * 0 + 8, "shared": 8.0 - next_delta}
    out_next = compute_learning_gain_objective_v2(
        {"protected": next_delta * 0 + 10, "authorized": next_delta * 0 + 10, "shared": next_delta * 0 + 10},
        {"protected": next_delta * 0 + 8, "authorized": next_delta * 0 + 8, "shared": next_delta * 0 + 8},
        next_poison,
        counts,
        support,
        {"protected": 2.0, "authorized": 2.0, "shared": 2.0},
        shared_tolerance=0.1,
    )
    assert out_next.e_shared.item() < out.e_shared.item()
