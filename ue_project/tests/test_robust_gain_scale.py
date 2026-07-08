import torch

from ue_framework.methods.multitrajectory_gain.gain_scale import compute_gain_scales_from_rows, robust_scale_from_clean_gains
from ue_framework.methods.multitrajectory_gain.learning_gain import compute_learning_gain_objective_v2


def test_robust_scale_dampens_tiny_per_trajectory_denominator():
    old = (0.00001 - 0.00002) / max(abs(0.00001), 1.0e-6)
    scale = robust_scale_from_clean_gains([0.00001, 1.0, 1.1, 1.2], quantile=0.5, epsilon=1.0e-4, minimum=1.0e-4)
    new = (0.00001 - 0.00002) / scale
    assert abs(old) > 0.9
    assert abs(new) < 1.0e-3


def test_robust_scale_uses_only_passed_training_rows():
    train_rows = [{"G_t_clean": 1.0, "G_a_clean": 2.0, "G_s_clean": 3.0}]
    heldout_rows = [{"G_t_clean": 100.0, "G_a_clean": 200.0, "G_s_clean": 300.0}]
    train_scales = compute_gain_scales_from_rows(train_rows)
    mixed_scales = compute_gain_scales_from_rows(train_rows + heldout_rows)
    assert train_scales.protected != mixed_scales.protected
    assert train_scales.protected < 2.0


def test_scale_does_not_backpropagate_to_delta():
    delta = torch.tensor(0.2, requires_grad=True)
    initial = {"protected": delta * 0 + 10, "authorized": delta * 0 + 10, "shared": delta * 0 + 10}
    clean = {"protected": delta * 0 + 8, "authorized": delta * 0 + 8, "shared": delta * 0 + 8}
    poison = {"protected": 8 + delta, "authorized": 8.0 - delta * 0, "shared": 8.0 - delta * 0}
    scales = {"protected": torch.tensor(2.0, requires_grad=True), "authorized": torch.tensor(2.0), "shared": torch.tensor(2.0)}
    out = compute_learning_gain_objective_v2(
        initial,
        clean,
        poison,
        {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1},
        {"protected_support_batches": 3, "authorized_support_batches": 1},
        scales,
    )
    out.total_loss.backward()
    assert delta.grad is not None
    assert scales["protected"].grad is None
