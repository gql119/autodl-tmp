import torch

from ue_framework.methods.multitrajectory_gain.feasibility import DualState


def test_dual_multiplier_increases_only_on_violation():
    state = DualState(1.0, 1.0)
    same = state.update(torch.tensor(0.0), torch.tensor(0.0), dual_learning_rate=0.1, mu_max=10.0)
    bigger = same.update(torch.tensor(2.0), torch.tensor(3.0), dual_learning_rate=0.1, mu_max=10.0)
    assert same.mu_authorized == 1.0
    assert same.mu_shared == 1.0
    assert bigger.mu_authorized == 1.2
    assert bigger.mu_shared == 1.3


def test_dual_multiplier_detaches_violation_from_delta_gradient():
    delta = torch.tensor(1.0, requires_grad=True)
    violation = delta * 2
    state = DualState(1.0, 1.0).update(violation, violation, 0.1, 10.0)
    assert state.mu_authorized == 1.2
    assert delta.grad is None
