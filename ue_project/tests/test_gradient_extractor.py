import torch

from .helpers import make_batch, make_images, make_toy_components


def test_p1_loss_is_differentiable_to_delta_and_does_not_update_surrogate():
    model, _adapter, method, _config = make_toy_components(seed=5)
    images = make_images(seed=6)
    batch = make_batch()
    delta = (torch.randn_like(images) * 0.01).requires_grad_()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    result = method.compute_p1_step(images, delta, batch)
    result["loss"].backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert float(delta.grad.norm().item()) > 0.0
    for name, param in model.named_parameters():
        assert torch.allclose(param.detach(), before[name])

    logs = result["logs"]
    assert "cos_protected_clean_poison" in logs
    assert "cos_authorized_clean_poison" in logs
    assert logs["effective_gradient_parameter_count"] > 0
