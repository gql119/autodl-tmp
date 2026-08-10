import torch
from torch import nn

from ue_framework.methods.semantic_hiding_carrier import (
    FixedHaarDWT,
    SemanticHidingCarrier,
    render_person_box_carrier,
)


def _small_carrier() -> SemanticHidingCarrier:
    torch.manual_seed(7)
    return SemanticHidingCarrier(input_size=32, width=8, coupling_blocks=4)


def test_haar_round_trip_is_exact_to_float_tolerance():
    torch.manual_seed(1)
    image = torch.rand(2, 3, 16, 20)
    dwt = FixedHaarDWT()
    recovered = dwt.inverse(dwt(image))
    assert torch.allclose(recovered, image, atol=1e-6, rtol=1e-6)


def test_formal_architecture_has_four_blocks_and_no_bn_or_dropout():
    carrier = SemanticHidingCarrier()
    descriptor = carrier.architecture_descriptor()
    assert descriptor["input_size"] == 256
    assert descriptor["coupling_blocks"] == 4
    assert descriptor["coupling_width"] == 64
    assert descriptor["batch_norm"] is False
    assert descriptor["dropout"] is False
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.Dropout)) for m in carrier.modules())
    assert len(carrier.architecture_sha256()) == 64


def test_same_host_is_deterministic_and_host_or_secret_changes_delta():
    carrier = _small_carrier().eval()
    host_a = torch.rand(1, 3, 32, 32)
    host_b = torch.rand(1, 3, 32, 32)
    secret_a = torch.rand(1, 3, 32, 32)
    secret_b = torch.rand(1, 3, 32, 32)
    out_1 = carrier(host_a, secret_a)
    out_2 = carrier(host_a, secret_a)
    out_host = carrier(host_b, secret_a)
    out_secret = carrier(host_a, secret_b)
    assert torch.equal(out_1.delta, out_2.delta)
    assert not torch.allclose(out_1.delta, out_host.delta)
    assert not torch.allclose(out_1.delta, out_secret.delta)
    assert float(out_1.delta.detach().abs().max()) <= carrier.epsilon + 1e-6


def test_freeze_boundary_leaves_only_adapter_trainable():
    carrier = _small_carrier()
    carrier.freeze_for_detector_optimization()
    assert all(not p.requires_grad for p in carrier.hiding_trunk.parameters())
    assert all(not p.requires_grad for p in carrier.reveal_decoder.parameters())
    assert all(p.requires_grad for p in carrier.adapter.parameters())


def test_bbox_render_has_exact_union_support_and_finite_adapter_backward():
    carrier = _small_carrier()
    carrier.freeze_for_detector_optimization()
    images = torch.full((1, 3, 48, 56), 0.5)
    boxes = (torch.tensor([[4.2, 5.1, 28.4, 34.6], [20.0, 18.0, 50.0, 44.0]]),)
    secret = torch.rand(1, 3, 32, 32)
    rendered = render_person_box_carrier(images, boxes, carrier, secret)
    union = rendered.union_support[0, 0]
    assert union[5:35, 4:29].all()
    assert union[18:44, 20:50].all()
    assert not union[:5].any()
    outside = rendered.perturbation * (~rendered.union_support).expand_as(
        rendered.perturbation
    )
    assert torch.count_nonzero(outside).item() == 0
    assert (
        float(rendered.perturbation.detach().abs().max())
        <= carrier.epsilon + 1e-6
    )
    loss = rendered.poisoned.square().mean()
    loss.backward()
    gradients = [p.grad for p in carrier.adapter.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0.0
    assert all(p.grad is None for p in carrier.hiding_trunk.parameters())
    assert len(rendered.canonical_deltas) == 2


def test_identical_overlapping_boxes_are_averaged_not_summed():
    carrier = _small_carrier().eval()
    images = torch.full((1, 3, 40, 44), 0.5)
    box = torch.tensor([[5.0, 6.0, 32.0, 35.0]])
    secret = torch.rand(1, 3, 32, 32)
    single = render_person_box_carrier(images, (box,), carrier, secret)
    duplicate = render_person_box_carrier(
        images, (torch.cat((box, box), dim=0),), carrier, secret
    )
    assert torch.allclose(single.perturbation, duplicate.perturbation, atol=1e-7)
