import torch

from ue_framework.methods.multitrajectory_gain.feasibility import constraint_violations


def test_constraint_violation_zero_inside_tolerance():
    gaps = {"Delta_t": torch.tensor(3.0), "Delta_a": torch.tensor(0.5), "Delta_s": torch.tensor(-0.25)}
    thresholds = {"protected_margin": 2.0, "tau_a": 1.0, "tau_s": 1.0}
    out = constraint_violations(gaps, thresholds)
    assert out["v_t"].item() == 0.0
    assert out["v_a"].item() == 0.0
    assert out["v_s"].item() == 0.0


def test_constraint_violation_positive_outside_tolerance():
    gaps = {"Delta_t": torch.tensor(1.0), "Delta_a": torch.tensor(2.5), "Delta_s": torch.tensor(-3.0)}
    thresholds = {"protected_margin": 2.0, "tau_a": 1.0, "tau_s": 1.0}
    out = constraint_violations(gaps, thresholds)
    assert out["v_t"].item() == 1.0
    assert out["v_a"].item() == 1.5
    assert out["v_s"].item() == 2.0
