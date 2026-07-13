import copy

import torch

from oa_lgc.model import ObjectCropDetector, class_loss
from oa_lgc.virtual_update import functional_forward, model_state_unchanged, select_parameter_names, virtual_update


def _batch():
    torch.manual_seed(4)
    images = torch.rand(2, 3, 20, 20)
    annotations = (
        ({"cls": 14, "bbox": [0.4, 0.5, 0.4, 0.5]}, {"cls": 1, "bbox": [0.8, 0.2, 0.2, 0.2]}),
        ({"cls": 14, "bbox": [0.6, 0.5, 0.3, 0.4]}, {"cls": 1, "bbox": [0.2, 0.8, 0.2, 0.2]}),
    )
    return images, annotations


def _model():
    torch.manual_seed(3)
    model = ObjectCropDetector(hidden_dim=12)
    model.requires_grad_(False)
    return model


def test_virtual_update_does_not_mutate_base_model():
    model = _model()
    before = copy.deepcopy(model.state_dict())
    virtual_update(model, *_batch(), steps=3, learning_rate=0.1)
    assert model_state_unchanged(model, before)


def test_virtual_update_clean_poison_separate():
    model = _model()
    images, annotations = _batch()
    clean = virtual_update(model, images, annotations, 1, 0.1)
    poison = virtual_update(model, (images + 0.05).clamp(0, 1), annotations, 1, 0.1)
    assert any(not torch.equal(clean.parameters[name], poison.parameters[name]) for name in clean.selected_names)


def test_virtual_update_j1():
    result = virtual_update(_model(), *_batch(), steps=1, learning_rate=0.1)
    assert len(result.step_losses) == len(result.parameter_delta_norms) == 1


def test_virtual_update_j3():
    result = virtual_update(_model(), *_batch(), steps=3, learning_rate=0.1)
    assert len(result.step_losses) == 3 and all(value > 0 for value in result.parameter_delta_norms)


def test_virtual_update_j5_single_episode():
    result = virtual_update(_model(), *_batch(), steps=5, learning_rate=0.05)
    assert len(result.step_losses) == 5 and all(torch.isfinite(torch.tensor(result.step_losses)))


def test_virtual_update_gradient_reaches_delta():
    model = _model()
    images, annotations = _batch()
    delta = torch.zeros_like(images, requires_grad=True)
    trajectory = virtual_update(model, (images + delta).clamp(0, 1), annotations, 3, 0.1, first_order=True)
    query_outputs = functional_forward(model, trajectory.parameters, trajectory.buffers, images, annotations)
    query_loss, _ = class_loss(query_outputs, 14)
    gradient = torch.autograd.grad(query_loss, delta)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_virtual_update_parameter_subset():
    model = _model()
    assert all(name.startswith("cls_head.") for name in select_parameter_names(model, "head_only"))
    assert all(name.startswith(("cls_head.", "box_head.")) for name in select_parameter_names(model, "detection_head"))
    selected = select_parameter_names(model, "selected_modules", ["feature_proj."])
    assert selected and all(name.startswith("feature_proj.") for name in selected)
    assert len(select_parameter_names(model, "full_model")) == len(tuple(model.parameters()))


def test_virtual_update_reproducibility():
    images, annotations = _batch()
    first = virtual_update(_model(), images, annotations, 3, 0.1)
    second = virtual_update(_model(), images, annotations, 3, 0.1)
    assert first.step_losses == second.step_losses
    assert all(torch.equal(first.parameters[name], second.parameters[name]) for name in first.parameters)


def test_virtual_update_no_optimizer_state_leak():
    model = _model()
    images, annotations = _batch()
    first = virtual_update(model, images, annotations, 1, 0.1)
    second = virtual_update(model, images, annotations, 1, 0.1)
    assert first.step_losses == second.step_losses
    assert not hasattr(first, "optimizer_state")

