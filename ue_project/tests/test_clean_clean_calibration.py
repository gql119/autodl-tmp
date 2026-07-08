from ue_framework.methods.multitrajectory_gain.feasibility import natural_variation_thresholds


def test_clean_clean_thresholds_use_train_rows_only():
    train = [{"N_t": 1.0, "N_a": 2.0, "N_s": 3.0}, {"N_t": 2.0, "N_a": 3.0, "N_s": 4.0}]
    heldout = [{"N_t": 100.0, "N_a": 100.0, "N_s": 100.0}]
    train_thresholds = natural_variation_thresholds(train, quantile=90, kappa_t=2.0)
    mixed_thresholds = natural_variation_thresholds(train + heldout, quantile=90, kappa_t=2.0)
    assert train_thresholds["tau_a"] != mixed_thresholds["tau_a"]
    assert train_thresholds["protected_margin"] == 2.0 * train_thresholds["tau_t"]
