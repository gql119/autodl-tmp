import torch

from tests.test_j3_rollout import make_engine, make_sequence


def test_j3_meta_gradient_to_delta_is_finite_and_nonzero():
    engine = make_engine(steps=3)
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    grad = torch.autograd.grad(out.loss, delta, allow_unused=False)[0]
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0
