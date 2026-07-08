import torch

from ue_framework.methods.learning_trajectory.virtual_update import make_virtual_parameters, parameter_leak_max_abs_diff, snapshot_parameters

from .helpers import make_batch, make_images, make_toy_components


def test_virtual_update_changes_functional_parameters_and_meta_loss_reaches_delta():
    model, adapter, method, _config = make_toy_components(seed=7)
    support_images = make_images(seed=8)
    query_images = make_images(seed=9)
    batch = make_batch()
    delta = (torch.randn_like(support_images) * 0.01).requires_grad_()

    snapshot = snapshot_parameters(model)
    support_pred = adapter.forward((support_images + delta).clamp(0, 1))
    support_loss = adapter.compute_detection_loss(support_pred, batch)
    selected = adapter.get_named_trainable_parameters("head")
    virtual = make_virtual_parameters(model, selected, support_loss, lr=0.2, create_graph=True)

    assert virtual.update_norm.detach().item() > 0.0
    assert any(not torch.allclose(dict(model.named_parameters())[name], value) for name, value in virtual.updated_parameters.items())
    assert parameter_leak_max_abs_diff(model, snapshot) == 0.0

    result = method.compute_p2_step(support_images, query_images, batch, batch, delta)
    logs = result["logs"]
    assert logs["meta_gradient_norm_to_delta"] > 0.0
    assert logs["parameter_leak_max_abs_diff"] == 0.0
    result["loss"].backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
