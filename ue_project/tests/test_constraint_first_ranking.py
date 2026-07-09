from ue_framework.methods.mtepi import ChannelScoreThresholds, constraint_first_rank


def test_constraint_first_ranking_filters_before_ratio_score():
    rows = [
        {"layer": "P3", "channel": 1, "Delta_t": 0.02, "Delta_a": 0.0001, "Delta_s": 0.0001, "clean_energy": 1.0},
        {"layer": "P3", "channel": 1, "Delta_t": 0.01, "Delta_a": 0.0001, "Delta_s": 0.0001, "clean_energy": 1.0},
        {"layer": "P3", "channel": 2, "Delta_t": 0.02, "Delta_a": 0.5, "Delta_s": 0.0001, "clean_energy": 1.0},
        {"layer": "P3", "channel": 2, "Delta_t": 0.02, "Delta_a": 0.5, "Delta_s": 0.0001, "clean_energy": 1.0},
    ]
    thresholds = ChannelScoreThresholds(0.005, 0.01, 0.01, 0.6, 2, 1.0e-6)
    ranked = constraint_first_rank(rows, thresholds)
    assert ranked[0]["channel"] == 1
    assert ranked[0]["hard_pass"] is True
    assert ranked[1]["hard_pass"] is False
