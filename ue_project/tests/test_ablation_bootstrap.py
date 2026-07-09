from ue_framework.methods.mtepi import bootstrap_mean_ci


def test_ablation_bootstrap_returns_reproducible_ci():
    a = bootstrap_mean_ci([1.0, 2.0, 3.0], num_bootstrap=50, seed=7)
    b = bootstrap_mean_ci([1.0, 2.0, 3.0], num_bootstrap=50, seed=7)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
