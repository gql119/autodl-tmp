from __future__ import annotations

import hashlib

import pytest
import torch

from ue_framework.methods.background_spectral_basis import (
    build_background_spectral_basis,
    deterministic_two_crops,
    spectrum_energy_ratios,
    validate_repository_manifest,
)
from ue_framework.methods.fourier import build_fourier_pattern
from ue_framework.methods.tausb_universal import _TAUSBCommon


def _source_images() -> list[torch.Tensor]:
    images: list[torch.Tensor] = []
    for index in range(8):
        generator = torch.Generator().manual_seed(100 + index)
        image = torch.rand((3, 40 + index, 48 + index), generator=generator)
        yy = torch.linspace(0, 1, image.shape[1]).view(1, -1, 1)
        xx = torch.linspace(0, 1, image.shape[2]).view(1, 1, -1)
        images.append((0.6 * image + 0.2 * yy + 0.2 * xx).clamp(0, 1))
    return images


def _manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(8):
        digest = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        rows.append(
            {
                "source_id": f"licensed-background-{index:02d}",
                "sha256": digest,
                "width": 1920,
                "height": 1080,
                "license_note": "owned research asset",
                "person_free": True,
            }
        )
    return rows


def test_repository_manifest_rejects_local_paths() -> None:
    rows = _manifest()
    validate_repository_manifest(rows)

    rows[0]["absolute_path"] = "D:/private/sky.jpg"
    with pytest.raises(ValueError, match="local path"):
        validate_repository_manifest(rows)


def test_scrambled_basis_is_deterministic_orthonormal_and_band_limited() -> None:
    kwargs = {
        "resolution": 32,
        "num_bases": 16,
        "bands": ((2.0, 8.0), (8.0, 12.0)),
        "phase_mode": "scrambled",
        "seed": 7,
        "min_rank": 8,
    }
    first = build_background_spectral_basis(_source_images(), **kwargs)
    second = build_background_spectral_basis(_source_images(), **kwargs)

    assert first.rank >= 8
    assert first.bases.shape == (16, 32, 32)
    assert torch.equal(first.bases, second.bases)
    assert first.source_hash == second.source_hash
    assert first.basis_hash == second.basis_hash
    assert torch.allclose(
        first.bases.mean(dim=(-2, -1)),
        torch.zeros(16),
        atol=1e-6,
    )
    assert torch.allclose(
        first.bases.flatten(1).norm(dim=1),
        torch.ones(16),
        atol=1e-5,
    )
    gram = first.bases.flatten(1) @ first.bases.flatten(1).T
    assert torch.allclose(gram, torch.eye(16), atol=1e-5)

    energy = spectrum_energy_ratios(
        first.bases[0],
        low=(2.0, 8.0),
        mid=(8.0, 12.0),
        high=(12.0, float("inf")),
    )
    assert energy["low"] + energy["mid"] > 0.999
    assert energy["high"] < 1e-8
    assert energy["dc"] < 1e-10


def test_raw_and_scrambled_phase_produce_distinct_bases() -> None:
    shared = {
        "resolution": 32,
        "num_bases": 8,
        "bands": ((2.0, 8.0),),
        "seed": 3,
        "min_rank": 8,
    }
    raw = build_background_spectral_basis(
        _source_images(),
        phase_mode="raw",
        **shared,
    )
    scrambled = build_background_spectral_basis(
        _source_images(),
        phase_mode="scrambled",
        **shared,
    )
    assert raw.basis_hash != scrambled.basis_hash


def test_recipe_hash_binds_sources_and_parameters() -> None:
    provenance = {
        "manifest_sha256": "b" * 64,
        "ordered_sources": [
            {"source_id": f"source-{index}", "sha256": f"{index:x}" * 64}
            for index in range(8)
        ],
    }
    kwargs = {
        "resolution": 24,
        "num_bases": 8,
        "bands": ((2.0, 8.0),),
        "phase_mode": "scrambled",
        "min_rank": 8,
        "hash_mode": "recipe-v1",
        "source_provenance": provenance,
    }
    first = build_background_spectral_basis(_source_images(), seed=3, **kwargs)
    second = build_background_spectral_basis(_source_images(), seed=3, **kwargs)
    changed = build_background_spectral_basis(_source_images(), seed=4, **kwargs)
    assert first.hash_mode == "recipe-v1"
    assert first.basis_hash == second.basis_hash
    assert first.basis_hash != changed.basis_hash


def test_background_recipe_hash_requires_source_provenance() -> None:
    with pytest.raises(ValueError, match="source provenance"):
        build_background_spectral_basis(
            _source_images(),
            resolution=24,
            num_bases=8,
            bands=((2.0, 8.0),),
            phase_mode="scrambled",
            seed=3,
            min_rank=8,
            hash_mode="recipe-v1",
        )


def test_square_source_produces_two_distinct_deterministic_crops() -> None:
    image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
    first = deterministic_two_crops(image, resolution=24, source_index=0)
    second = deterministic_two_crops(image, resolution=24, source_index=0)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0], first[1])


def test_multichannel_spectrum_energy_does_not_cancel_opposite_channels() -> None:
    yy = torch.arange(32, dtype=torch.float32).view(-1, 1)
    wave = torch.cos(2 * torch.pi * 4 * yy / 32).repeat(1, 32)
    pattern = torch.stack((wave, -wave, torch.zeros_like(wave)))
    energy = spectrum_energy_ratios(pattern)
    assert energy["low"] > 0.999
    assert energy["mid"] < 1e-8
    assert energy["high"] < 1e-8


def _common_stub(mode: str, *, num_bases: int) -> _TAUSBCommon:
    common = object.__new__(_TAUSBCommon)
    common.carrier_basis_mode = mode
    common.background_bases = None
    common.background_basis_meta = {}
    common.shortcut_num_bases = num_bases
    common.device = torch.device("cpu")
    common.imgsz = 32
    common.tanh_temp = 1.0
    common.eps = 16 / 255
    common.freq_amp_buffer = 1.0
    common.lambda_freq = 1.0
    common.jnd_ceiling = 1.0
    common.is_universal_training = False
    return common


def test_synthetic_dispatch_is_numerically_identical() -> None:
    common = _common_stub("synthetic_fourier", num_bases=3)
    coords = [(2, 3), (5, 4), (7, 2)]
    coeff = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.2]],
        dtype=torch.float32,
    )
    actual = common._build_global_freq_pattern(28, 24, coords, coeff)

    expected_channels = []
    for channel in range(3):
        amplitudes = torch.tanh(coeff[:, channel]) * common.eps
        base = build_fourier_pattern(
            common.imgsz,
            common.imgsz,
            coords,
            amplitudes,
            common.device,
        )
        expected_channels.append(
            torch.nn.functional.interpolate(
                base,
                size=(28, 24),
                mode="bilinear",
                align_corners=False,
            )
        )
    expected = torch.cat(expected_channels, dim=1)
    assert torch.equal(actual, expected)


def test_background_dispatch_is_rgb_differentiable_and_reports_spectrum() -> None:
    basis = build_background_spectral_basis(
        _source_images(),
        resolution=32,
        num_bases=8,
        bands=((2.0, 8.0),),
        phase_mode="scrambled",
        seed=11,
        min_rank=8,
    )
    common = _common_stub("background_scrambled_low", num_bases=8)
    common._load_background_basis_pack(
        {
            "bases": basis.bases,
            "metadata": {
                "carrier_basis_mode": "background_scrambled_low",
                "basis_hash": basis.basis_hash,
            },
        }
    )
    coeff = torch.linspace(-0.4, 0.5, 24).reshape(8, 3).requires_grad_()
    pattern = common._build_global_freq_pattern(32, 32, [], coeff)
    assert pattern.shape == (1, 3, 32, 32)
    pattern.square().mean().backward()
    assert coeff.grad is not None
    assert torch.isfinite(coeff.grad).all()

    image = torch.full((1, 3, 32, 32), 0.5)
    support = torch.ones((1, 1, 32, 32))
    _, perturb, adv, _, _ = common._compose_delta_batched(
        image,
        support,
        torch.zeros_like(support),
        [],
        coeff.detach(),
        torch.zeros((1, 3, 4, 4)),
    )
    assert perturb.shape == image.shape
    assert float(perturb.abs().max()) <= common.eps + 1e-7
    assert torch.all((adv >= 0) & (adv <= 1))
    energy = spectrum_energy_ratios(perturb.squeeze(0))
    assert set(energy) == {"low", "mid", "high", "dc"}
    assert all(torch.isfinite(torch.tensor(value)) for value in energy.values())
