import copy

import pytest
import torch

from oa_lgc.objective import (
    LOSS_LOG_FIELDS,
    CoreObjectiveConfig,
    compose_core_objective,
    load_delta_checkpoint,
    project_delta_,
    save_delta_checkpoint,
    update_delta,
)


def test_core_objective_components():
    delta = torch.ones(2)
    config = CoreObjectiveConfig(lambda_carrier=2.0, lambda_auth=3.0, lambda_reg=0.5)
    result = compose_core_objective(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), delta, config)
    assert torch.allclose(result.loss, torch.tensor(14.5))
    assert set(result.components) == {"L_protect", "L_carrier", "L_auth", "L_delta", "weighted_carrier", "weighted_auth", "weighted_reg"}


def test_core_objective_delta_update():
    delta = torch.nn.Parameter(torch.tensor([0.0]))
    config = CoreObjectiveConfig(eps=0.2, gradient_clip_norm=1.0)
    optimizer = torch.optim.SGD([delta], lr=0.1)
    result = compose_core_objective(torch.tensor(0.0), (delta - 0.15).square().mean(), torch.tensor(0.0), delta, config)
    before = delta.detach().clone()
    gradient = update_delta(result, delta, optimizer, config)
    assert gradient > 0 and not torch.equal(before, delta.detach())


def test_core_objective_model_frozen():
    model = torch.nn.Linear(2, 2)
    model.requires_grad_(False)
    before = copy.deepcopy(model.state_dict())
    delta = torch.nn.Parameter(torch.tensor([0.1]))
    config = CoreObjectiveConfig()
    result = compose_core_objective(delta.square(), delta.square(), delta.square() * 0, delta, config)
    update_delta(result, delta, torch.optim.SGD([delta], lr=0.01), config)
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_core_objective_empty_authorized_classes():
    delta = torch.tensor([0.1], requires_grad=True)
    result = compose_core_objective(delta.square(), delta.square(), delta.sum() * 0, delta, CoreObjectiveConfig())
    assert torch.isfinite(result.loss)


def test_core_objective_invalid_episode_skip():
    delta = torch.tensor([0.1], requires_grad=True)
    invalid_protect = delta.sum() * 0
    result = compose_core_objective(invalid_protect, delta.square(), delta.sum() * 0, delta, CoreObjectiveConfig())
    result.loss.backward()
    assert delta.grad is not None and torch.isfinite(delta.grad)


def test_delta_projection_budget():
    delta = torch.tensor([-1.0, 0.1, 1.0])
    project_delta_(delta, 0.2)
    assert float(delta.abs().max()) <= 0.2 + 1e-7


def test_delta_checkpoint_roundtrip(tmp_path):
    delta = torch.tensor([[[0.1, -0.1]]])
    config = CoreObjectiveConfig(eps=0.2)
    path = tmp_path / "delta.pt"
    save_delta_checkpoint(path, delta, config, {"step": 1})
    loaded = load_delta_checkpoint(path)
    assert torch.equal(loaded["delta_obj"], delta)
    assert loaded["metadata"]["step"] == 1
    with pytest.raises(FileExistsError):
        save_delta_checkpoint(path, delta, config)


def test_loss_logging_schema():
    required = {"L_core", "L_protect", "L_carrier", "L_auth", "L_delta", "gradient_norm", "valid_target_gain"}
    assert required.issubset(LOSS_LOG_FIELDS)
