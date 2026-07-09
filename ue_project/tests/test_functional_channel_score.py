from ue_framework.methods.mtepi import ablation_delta


def test_functional_channel_score_uses_ablation_minus_base_loss():
    out = ablation_delta(
        {"protected": 2.0, "authorized": 3.0, "shared": 4.0},
        {"protected": 2.5, "authorized": 2.5, "shared": 4.25},
    )
    assert out["Delta_t"] == 0.5
    assert out["Delta_a"] == -0.5
    assert out["Delta_s"] == 0.25
