from __future__ import annotations

import torch

from ue_framework.methods.semantic_residual_carrier import (
    VariantMatchedCanonicalCarrier,
    build_semantic_carrier_bank,
    calibrate_variant_shared_gamma,
    center_square_resize,
    radial_orientation_masks,
    stable_variant_index,
)


def _images() -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(42)
    anchor = torch.rand((3, 35, 51), generator=generator)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 35),
        torch.linspace(-1, 1, 51),
        indexing="ij",
    )
    tree_like = torch.exp(-10 * xx.square()) * (yy > -0.2)
    anchor = (0.25 * anchor + 0.75 * tree_like).clamp(0, 1)
    donors = [
        torch.rand((3, 36 + index, 52 - index), generator=generator)
        for index in range(4)
    ]
    return anchor, donors


def test_center_crop_and_variant_hash_are_deterministic() -> None:
    anchor, _ = _images()
    first = center_square_resize(anchor, 32)
    second = center_square_resize(anchor, 32)
    assert first.shape == (3, 32, 32)
    assert torch.equal(first, second)
    assert stable_variant_index("2008_000001") == stable_variant_index(
        "2008_000001"
    )
    assert 0 <= stable_variant_index("2008_000001") < 4


def test_masks_partition_the_frozen_low_mid_band() -> None:
    masks = radial_orientation_masks(48)
    assert masks.shape == (16, 48, 48)
    assert int(masks.sum(dim=0).amax()) == 1
    assert not bool(masks[:, 0, 0].any())
    for mask in masks:
        conjugate = torch.roll(
            torch.flip(mask, (0, 1)),
            shifts=(1, 1),
            dims=(0, 1),
        )
        assert torch.equal(mask, conjugate)


def test_semantic_and_scrambled_families_are_exactly_amplitude_matched() -> None:
    anchor, donors = _images()
    bank = build_semantic_carrier_bank(anchor, donors, resolution=48)
    assert bank.variants.shape == (4, 48, 48)
    assert bank.semantic_bases.shape == (4, 16, 48, 48)
    assert bank.control_bases.shape == bank.semantic_bases.shape
    reference = torch.fft.fft2(bank.variants[0])
    reference_phase = reference / reference.abs().clamp_min(1e-8)
    for variant in bank.variants[1:]:
        spectrum = torch.fft.fft2(variant)
        active = (reference.abs() > 1e-5) & (spectrum.abs() > 1e-5)
        phase = spectrum / spectrum.abs().clamp_min(1e-8)
        assert torch.allclose(
            phase[active],
            reference_phase[active],
            atol=2e-5,
            rtol=2e-5,
        )
    for semantic, control in zip(bank.variants, bank.controls):
        assert torch.allclose(
            torch.fft.fft2(semantic).abs(),
            torch.fft.fft2(control).abs(),
            atol=2e-5,
            rtol=2e-5,
        )
    for semantic, control in zip(bank.semantic_bases, bank.control_bases):
        assert torch.allclose(
            torch.fft.fft2(semantic).abs(),
            torch.fft.fft2(control).abs(),
            atol=3e-5,
            rtol=3e-5,
        )


def test_basis_invariants_rank_hash_and_gradients() -> None:
    anchor, donors = _images()
    first = build_semantic_carrier_bank(anchor, donors, resolution=48)
    second = build_semantic_carrier_bank(anchor, donors, resolution=48)
    assert first.bank_hash == second.bank_hash
    assert torch.equal(first.semantic_bases, second.semantic_bases)
    assert torch.equal(first.control_bases, second.control_bases)
    assert torch.equal(first.semantic_ranks, torch.full((4,), 16))
    assert torch.equal(first.control_ranks, torch.full((4,), 16))
    for bases in (first.semantic_bases, first.control_bases):
        assert torch.allclose(
            bases.mean(dim=(-2, -1)),
            torch.zeros((4, 16)),
            atol=1e-7,
        )
        assert torch.allclose(
            bases.flatten(2).norm(dim=2),
            torch.ones((4, 16)),
            atol=1e-6,
        )
    coefficients = torch.randn((16, 3), requires_grad=True)
    pattern = torch.einsum("kc,khw->chw", coefficients, first.semantic_bases[0])
    pattern.square().mean().backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert float(coefficients.grad.norm()) > 0


def test_variant_carrier_shares_one_coefficient_matrix() -> None:
    anchor, donors = _images()
    bank = build_semantic_carrier_bank(anchor, donors, resolution=48)
    initial = torch.zeros((16, 3))
    initial[:, 0] = torch.linspace(-0.2, 0.2, 16)
    carrier = VariantMatchedCanonicalCarrier(
        bank.semantic_bases,
        bank.semantic_scales,
        epsilon=16 / 255,
        gamma=80.0,
        initial_coefficients=initial,
    )
    patterns = carrier()
    assert patterns.shape == (4, 3, 48, 48)
    assert sum(parameter.numel() for parameter in carrier.parameters()) == 48
    patterns.square().mean().backward()
    assert carrier.coefficients.grad is not None
    assert torch.isfinite(carrier.coefficients.grad).all()
    assert float(carrier.coefficients.grad.norm()) > 0


def test_positive_modulation_preserves_initial_semantic_structure() -> None:
    anchor, donors = _images()
    bank = build_semantic_carrier_bank(anchor, donors, resolution=48)
    families = {
        "semantic": (bank.semantic_bases, bank.semantic_scales),
        "control": (bank.control_bases, bank.control_scales),
    }
    calibration = calibrate_variant_shared_gamma(
        families,
        epsilon=16 / 255,
        num_directions=24,
        iterations=24,
    )
    initial = torch.randn(
        (16, 3),
        generator=torch.Generator(device="cpu").manual_seed(2103),
    )
    initial *= 0.25 / initial.abs().amax()
    semantic = VariantMatchedCanonicalCarrier(
        bank.semantic_bases,
        bank.semantic_scales,
        epsilon=16 / 255,
        gamma=calibration.gamma,
        initial_coefficients=initial,
    )()
    control = VariantMatchedCanonicalCarrier(
        bank.control_bases,
        bank.control_scales,
        epsilon=16 / 255,
        gamma=calibration.gamma,
        initial_coefficients=initial,
    )()
    semantic_rms = semantic.square().mean(dim=(1, 2, 3)).sqrt()
    control_rms = control.square().mean(dim=(1, 2, 3)).sqrt()
    initial_median = torch.cat((semantic_rms, control_rms)).detach().median()
    assert abs(float(initial_median) / (16 / 255) - 0.35) < 0.01
    assert abs(calibration.pooled_median_rms / (16 / 255) - 0.35) < 1e-5
    assert calibration.family_rms_ratio <= 1.10
