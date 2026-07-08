import torch

from ue_framework.methods.multitrajectory_gain.gradient_diagnostics import gradient_conflict_diagnostics


def test_outer_gradient_diagnostics_reports_norms_and_cosines():
    delta = torch.tensor([1.0, -2.0, 0.0], requires_grad=True)
    losses = {
        "protected": (delta[0] + delta[1]).pow(2),
        "authorized": (delta[0] - delta[1]).pow(2),
        "shared": delta[2] * 0.0,
    }
    out = gradient_conflict_diagnostics(losses, delta)
    assert out["names"] == ["protected", "authorized", "shared"]
    assert len(out["matrix"]) == 3
    assert out["rows"][0]["gradient_norm"] > 0.0
    assert out["rows"][2]["zero_gradient_ratio"] == 1.0
