import torch

from tests.test_j3_rollout import make_engine, make_sequence


def test_finite_difference_step_reduces_j3_objective():
    torch.manual_seed(1)
    engine = make_engine(steps=3)
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    grad = torch.autograd.grad(out.loss, delta, allow_unused=False)[0]
    with torch.no_grad():
        delta_next = (delta - 1.0e-2 * grad).detach().requires_grad_(True)
    out_next = engine.run(make_sequence(), delta_next, create_graph=True)
    assert out_next.loss.detach().item() < out.loss.detach().item()
