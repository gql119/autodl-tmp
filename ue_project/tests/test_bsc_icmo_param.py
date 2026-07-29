from __future__ import annotations

import torch

from ue_framework.methods.instance_canonical_carrier import (
    MatchedCanonicalCarrier,
    build_synthetic_fourier_bases,
    calibrate_shared_gamma,
    canonical_pattern,
    canonicalize_explicit_bases,
    common_initial_coefficients,
    tensor_sha256,
)


def _natural_like_bases(num_bases: int, resolution: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(91)
    raw = torch.randn(
        (num_bases, resolution, resolution),
        generator=generator,
    )
    spectrum = torch.fft.fft2(raw)
    frequencies = torch.fft.fftfreq(resolution) * resolution
    yy, xx = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    mask = (radius >= 2) & (radius < 8)
    return torch.fft.ifft2(spectrum * mask).real


def test_explicit_bases_are_symmetric_and_canonical() -> None:
    synthetic = build_synthetic_fourier_bases(
        32,
        ((2, 3), (4, 1), (5, 6), (7, 2)),
    )
    natural = canonicalize_explicit_bases(_natural_like_bases(4, 32))
    for bases in (synthetic, natural):
        assert bases.shape == (4, 32, 32)
        assert torch.allclose(
            bases.mean(dim=(-2, -1)),
            torch.zeros(4),
            atol=1e-7,
        )
        assert torch.allclose(
            bases.flatten(1).norm(dim=1),
            torch.ones(4),
            atol=1e-6,
        )
        flat = bases.flatten(1)
        indices = flat.abs().argmax(dim=1, keepdim=True)
        assert bool((flat.gather(1, indices) > 0).all())


def test_common_initialization_and_hash_are_deterministic() -> None:
    first = common_initial_coefficients(16)
    second = common_initial_coefficients(16)
    assert torch.equal(first, second)
    assert torch.isclose(first.abs().amax(), torch.tensor(0.25))
    assert tensor_sha256(first) == tensor_sha256(second)


def test_shared_gamma_matches_pooled_rms_without_stepwise_normalization() -> None:
    epsilon = 16.0 / 255.0
    synthetic = build_synthetic_fourier_bases(
        24,
        ((2, 3), (4, 1), (5, 6), (7, 2)),
    )
    natural = canonicalize_explicit_bases(_natural_like_bases(4, 24))
    calibration = calibrate_shared_gamma(
        {"C0": synthetic, "C2-LM": natural},
        epsilon=epsilon,
        num_directions=24,
        iterations=24,
        chunk_size=6,
    )
    assert calibration.gamma > 0
    assert abs(calibration.pooled_median_rms / epsilon - 0.35) < 1e-5
    assert set(calibration.family_median_rms) == {"C0", "C2-LM"}
    assert len(calibration.direction_hash) == 64

    coefficients = common_initial_coefficients(4)
    c0 = canonical_pattern(
        synthetic,
        coefficients,
        epsilon=epsilon,
        gamma=calibration.gamma,
    )
    c2lm = canonical_pattern(
        natural,
        coefficients,
        epsilon=epsilon,
        gamma=calibration.gamma,
    )
    assert c0.abs().amax() <= epsilon
    assert c2lm.abs().amax() <= epsilon
    assert torch.isfinite(c0).all() and torch.isfinite(c2lm).all()


def test_coefficient_scale_mapping_is_continuous_and_differentiable() -> None:
    bases = build_synthetic_fourier_bases(
        24,
        ((2, 3), (4, 1), (5, 6), (7, 2)),
    )
    initial = common_initial_coefficients(4)
    carrier = MatchedCanonicalCarrier(
        bases,
        epsilon=16.0 / 255.0,
        gamma=80.0,
        initial_coefficients=initial,
    )
    rms_values = []
    for scale in (1e-4, 1e-3, 1e-2, 0.1, 0.5):
        carrier.coefficients.data.copy_(initial * scale)
        pattern = carrier()
        rms_values.append(float(pattern.square().mean().sqrt().detach()))
    assert all(
        later > earlier for earlier, later in zip(rms_values, rms_values[1:])
    )

    carrier.coefficients.data.copy_(initial * 0.1)
    carrier().square().mean().backward()
    assert carrier.coefficients.grad is not None
    assert torch.isfinite(carrier.coefficients.grad).all()
    assert float(carrier.coefficients.grad.norm()) > 0
