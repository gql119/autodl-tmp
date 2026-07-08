import torch

from ue_framework.methods.multitrajectory_gain.functional_optimizer import (
    clone_parameter_dict,
    functional_sgd_step,
    init_functional_sgd_state,
)


def test_functional_sgd_matches_torch_sgd_single_step():
    torch.manual_seed(0)
    layer = torch.nn.Linear(3, 1, bias=False)
    ref = torch.nn.Linear(3, 1, bias=False)
    ref.load_state_dict(layer.state_dict())
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)

    params = {"weight": layer.weight.detach().clone().requires_grad_(True)}
    state = init_functional_sgd_state(params)
    loss = torch.nn.functional.mse_loss(x @ params["weight"].t(), y)
    updated, next_state, _ = functional_sgd_step(
        params, loss, state, learning_rate=0.1, momentum=0.9, weight_decay=0.01, create_graph=False
    )

    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    ref_loss = torch.nn.functional.mse_loss(ref(x), y)
    ref_loss.backward()
    opt.step()
    assert torch.allclose(updated["weight"], ref.weight, atol=1.0e-6)
    assert next_state.momentum_buffers["weight"].abs().sum().item() > 0.0
    assert torch.allclose(layer.weight.detach(), params["weight"].detach())


def test_functional_sgd_state_clone_has_no_shared_references():
    params = {"w": torch.ones(2, requires_grad=True)}
    state = init_functional_sgd_state(params)
    clone = state.clone()
    clone.momentum_buffers["w"].add_(1.0)
    assert state.momentum_buffers["w"].sum().item() == 0.0


def test_clone_parameter_dict_does_not_alias_original():
    params = {"w": torch.ones(2, requires_grad=True)}
    cloned = clone_parameter_dict(params, detach=True)
    cloned["w"].data.add_(1.0)
    assert params["w"].sum().item() == 2.0
