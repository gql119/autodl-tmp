import torch

from tests.test_j3_rollout import make_engine, make_sequence


def test_j3_poison_path_uses_localized_support_and_zero_outside():
    engine = make_engine(steps=3)
    delta = torch.ones((1, 3, 8, 8), requires_grad=True) * 0.01
    out = engine.run(make_sequence(), delta, create_graph=True)
    assert max(step["valid_support_ratio"] for step in out.per_step) < 1.0
    assert max(step["outside_support_max_abs_delta"] for step in out.per_step) == 0.0
