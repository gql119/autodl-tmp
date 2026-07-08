import torch

from tests.test_j3_rollout import make_engine, make_sequence


def test_j3_surrogate_parameters_do_not_change():
    engine = make_engine(steps=3)
    snapshot = {name: param.detach().clone() for name, param in engine.adapter.model.named_parameters()}
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    assert out.logs["surrogate_parameter_max_abs_diff"] == 0.0
    for name, param in engine.adapter.model.named_parameters():
        assert torch.allclose(snapshot[name], param.detach())
