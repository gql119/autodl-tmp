import torch

from ue_framework.methods.multitrajectory_gain.feasibility import raw_counterfactual_gap


def test_raw_counterfactual_gap_signs_are_poison_minus_clean():
    clean = {"protected": torch.tensor(2.0), "authorized": torch.tensor(5.0), "shared": torch.tensor(7.0)}
    poison = {"protected": torch.tensor(3.5), "authorized": torch.tensor(4.0), "shared": torch.tensor(10.0)}
    out = raw_counterfactual_gap(clean, poison)
    assert out["Delta_t"].item() == 1.5
    assert out["Delta_a"].item() == -1.0
    assert out["Delta_s"].item() == 3.0
