from ue_framework.methods.multitrajectory_gain.feasibility import summarize_clean_gains


def test_checkpoint_screening_uses_clean_gain_rows_only():
    rows = [
        {"protected_raw_gain": 1.0, "authorized_raw_gain": 0.5, "shared_raw_gain": 0.1, "protected_valid": 1.0},
        {"protected_raw_gain": -1.0, "authorized_raw_gain": -0.5, "shared_raw_gain": 0.2, "protected_valid": 0.0},
    ]
    out = summarize_clean_gains(rows)
    assert out["trajectory_count"] == 2
    assert out["protected_valid_ratio"] == 0.5
    assert out["protected_gain_positive_ratio"] == 0.5
