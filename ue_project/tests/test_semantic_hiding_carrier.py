import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ue_framework.methods.semantic_hiding_carrier import (
    FixedHaarDWT,
    SemanticHidingCarrier,
    deterministic_bilinear_resize_2d,
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
    assert "hf_subband_scale" not in descriptor
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.Dropout)) for m in carrier.modules())
    assert len(carrier.architecture_sha256()) == 64


@pytest.mark.parametrize("scale", [-0.01, 1.01, float("nan")])
def test_hf_subband_scale_rejects_values_outside_unit_interval(scale):
    with pytest.raises(ValueError, match="hf_subband_scale"):
        SemanticHidingCarrier(
            input_size=16,
            width=4,
            coupling_blocks=2,
            hf_subband_scale=scale,
        )


def test_scale_one_is_exact_output_and_gradient_rollback():
    torch.manual_seed(19)
    legacy = SemanticHidingCarrier(input_size=16, width=4, coupling_blocks=2)
    rollback = SemanticHidingCarrier(
        input_size=16,
        width=4,
        coupling_blocks=2,
        hf_subband_scale=1.0,
    )
    rollback.load_state_dict(legacy.state_dict(), strict=True)
    host_legacy = torch.rand(2, 3, 16, 16, requires_grad=True)
    secret_legacy = torch.rand(2, 3, 16, 16, requires_grad=True)
    host_rollback = host_legacy.detach().clone().requires_grad_(True)
    secret_rollback = secret_legacy.detach().clone().requires_grad_(True)
    out_legacy = legacy(host_legacy, secret_legacy)
    out_rollback = rollback(host_rollback, secret_rollback)
    assert torch.equal(out_legacy.delta, out_rollback.delta)
    assert legacy.architecture_sha256() == rollback.architecture_sha256()
    out_legacy.delta.square().sum().backward()
    out_rollback.delta.square().sum().backward()
    assert torch.equal(host_legacy.grad, host_rollback.grad)
    assert torch.equal(secret_legacy.grad, secret_rollback.grad)
    for legacy_parameter, rollback_parameter in zip(
        legacy.parameters(), rollback.parameters()
    ):
        if legacy_parameter.grad is None or rollback_parameter.grad is None:
            assert legacy_parameter.grad is None and rollback_parameter.grad is None
        else:
            assert torch.equal(legacy_parameter.grad, rollback_parameter.grad)


def test_quarter_scale_attenuates_only_haar_high_subbands_with_finite_gradient():
    carrier = SemanticHidingCarrier(
        input_size=16,
        width=4,
        coupling_blocks=2,
        hf_subband_scale=0.25,
    )
    raw = torch.randn(2, 3, 16, 16, requires_grad=True)
    filtered = carrier._filter_residual_subbands(raw)
    raw_ll, *raw_high = carrier.dwt(raw).chunk(4, dim=1)
    filtered_ll, *filtered_high = carrier.dwt(filtered).chunk(4, dim=1)
    assert torch.allclose(filtered_ll, raw_ll, atol=1e-6, rtol=1e-6)
    for before, after in zip(raw_high, filtered_high):
        assert torch.allclose(after, before * 0.25, atol=1e-6, rtol=1e-6)
    filtered.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    descriptor = carrier.architecture_descriptor()
    assert descriptor["hf_subband_scale"] == 0.25


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


@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [
        ((7, 11), (13, 5)),
        ((13, 5), (7, 11)),
        ((7, 11), (1, 19)),
        ((7, 11), (19, 1)),
        ((1, 11), (17, 3)),
        ((9, 1), (3, 15)),
        ((8, 8), (8, 8)),
    ],
)
def test_deterministic_resize_matches_align_corners_false_bilinear_forward(
    source_size, target_size
):
    torch.manual_seed(23)
    inputs = torch.randn(2, 3, *source_size)
    expected = F.interpolate(
        inputs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    actual = deterministic_bilinear_resize_2d(inputs, target_size)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
    assert float((actual - expected).abs().max()) <= 2e-6


def test_deterministic_resize_matches_cpu_bilinear_input_gradient():
    torch.manual_seed(29)
    legacy_input = torch.randn(2, 3, 9, 13, requires_grad=True)
    fixed_input = legacy_input.detach().clone().requires_grad_(True)
    probe = torch.randn(2, 3, 17, 6)
    legacy = F.interpolate(
        legacy_input,
        size=(17, 6),
        mode="bilinear",
        align_corners=False,
    )
    fixed = deterministic_bilinear_resize_2d(fixed_input, (17, 6))
    legacy_gradient = torch.autograd.grad((legacy * probe).sum(), legacy_input)[0]
    fixed_gradient = torch.autograd.grad((fixed * probe).sum(), fixed_input)[0]
    assert torch.allclose(fixed_gradient, legacy_gradient, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "target_size",
    [(372, 394), (399, 175), (599, 577), (425, 160)],
)
def test_deterministic_resize_matches_real_box_forward_in_production_range(
    target_size,
):
    torch.manual_seed(0)
    epsilon = 16.0 / 255.0
    inputs = (torch.rand(1, 3, 256, 256) * 2.0 - 1.0) * epsilon
    expected = F.interpolate(
        inputs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    actual = deterministic_bilinear_resize_2d(inputs, target_size)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
    assert float((actual - expected).abs().max()) <= 2e-6


def test_render_host_is_no_grad_but_patch_reaches_adapter():
    carrier = _small_carrier()
    carrier.freeze_for_detector_optimization()
    images = torch.full((1, 3, 48, 56), 0.5, requires_grad=True)
    boxes = (torch.tensor([[4.2, 5.1, 28.4, 34.6]]),)
    secret = torch.rand(1, 3, 32, 32)
    trace = {}

    def capture(stage, tensors):
        trace[stage] = tensors

    rendered = render_person_box_carrier(
        images,
        boxes,
        carrier,
        secret,
        trace_callback=capture,
    )
    assert trace["render"]["hosts"][0].requires_grad is False
    assert trace["render"]["resized_patches"][0].requires_grad is True
    rendered.perturbation.square().mean().backward()
    gradients = [parameter.grad for parameter in carrier.adapter.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


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
