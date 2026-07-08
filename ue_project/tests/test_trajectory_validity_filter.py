from ue_framework.methods.multitrajectory_gain.learning_gain import compute_learning_gain_objective
from tests.test_learning_gain import _losses


def test_authorized_term_skips_when_query_has_no_authorized_positives():
    out = compute_learning_gain_objective(
        _losses(10.0, 10.0, 10.0),
        _losses(8.0, 8.0, 8.0),
        _losses(9.0, 9.0, 8.0),
        {"protected_positive_count": 1, "authorized_positive_count": 0, "background_count": 1},
    )
    assert not out.authorized_valid
    assert out.authorized_loss.item() == 0.0


def test_protected_term_skips_when_query_has_no_protected_positives():
    out = compute_learning_gain_objective(
        _losses(10.0, 10.0, 10.0),
        _losses(8.0, 8.0, 8.0),
        _losses(9.0, 8.0, 8.0),
        {"protected_positive_count": 0, "authorized_positive_count": 1, "background_count": 1},
    )
    assert not out.protected_valid
    assert out.protected_loss.item() == 0.0
