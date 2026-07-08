import torch

from ue_framework.methods.multitrajectory_gain.learning_gain import compute_learning_gain_objective


def _losses(p, a, s):
    return {
        "protected": torch.tensor(float(p), requires_grad=True),
        "authorized": torch.tensor(float(a), requires_grad=True),
        "shared": torch.tensor(float(s), requires_grad=True),
    }


def test_learning_gain_protected_suppression_has_positive_d():
    initial = _losses(10.0, 10.0, 10.0)
    clean = _losses(8.0, 8.0, 8.0)
    poison = _losses(9.5, 8.0, 8.0)
    out = compute_learning_gain_objective(initial, clean, poison, {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1})
    assert out.d_protected.item() > 0.0
    assert out.protected_loss.item() == 0.0


def test_learning_gain_equal_poison_clean_keeps_margin_penalty():
    initial = _losses(10.0, 10.0, 10.0)
    clean = _losses(8.0, 8.0, 8.0)
    poison = _losses(8.0, 8.0, 8.0)
    out = compute_learning_gain_objective(initial, clean, poison, {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1})
    assert abs(out.d_protected.item()) < 1.0e-6
    assert out.protected_loss.item() > 0.0


def test_learning_gain_authorized_deviation_increases_loss():
    initial = _losses(10.0, 10.0, 10.0)
    clean = _losses(8.0, 8.0, 8.0)
    same = _losses(9.0, 8.0, 8.0)
    deviated = _losses(9.0, 9.0, 8.0)
    counts = {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1}
    out_same = compute_learning_gain_objective(initial, clean, same, counts)
    out_dev = compute_learning_gain_objective(initial, clean, deviated, counts)
    assert out_dev.authorized_loss.item() > out_same.authorized_loss.item()


def test_learning_gain_invalid_protected_clean_gain_skips_term():
    initial = _losses(10.0, 10.0, 10.0)
    clean = _losses(9.99999, 8.0, 8.0)
    poison = _losses(10.0, 8.0, 8.0)
    out = compute_learning_gain_objective(initial, clean, poison, {"protected_positive_count": 1, "authorized_positive_count": 1, "background_count": 1})
    assert not out.protected_valid
    assert out.protected_loss.item() == 0.0
