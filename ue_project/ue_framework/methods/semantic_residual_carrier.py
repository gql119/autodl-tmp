from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .instance_canonical_carrier import (
    SharedGammaCalibration,
    canonicalize_explicit_bases,
    tensor_sha256,
)


DEFAULT_RADIAL_EDGES = (2.0, 5.5, 10.0, 16.0, 24.0)


def _validate_rgb_image(image: torch.Tensor, *, name: str) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{name} must have shape [3,H,W].")
    if min(image.shape[-2:]) <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions.")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values.")


def center_square_resize(image: torch.Tensor, resolution: int) -> torch.Tensor:
    _validate_rgb_image(image, name="image")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if image.is_floating_point():
        image = image.float()
        if float(image.max()) > 1.0:
            if float(image.max()) > 255.0 or float(image.min()) < 0.0:
                raise ValueError("Floating RGB source must lie in [0,1] or [0,255].")
            image = image / 255.0
    else:
        image = image.float() / 255.0
    height, width = image.shape[-2:]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = image[:, top : top + side, left : left + side]
    return F.interpolate(
        crop.unsqueeze(0),
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )[0]


def rgb_luminance(image: torch.Tensor) -> torch.Tensor:
    _validate_rgb_image(image, name="image")
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (image * weights).sum(dim=0)


def radial_orientation_masks(
    resolution: int,
    *,
    radial_edges: Sequence[float] = DEFAULT_RADIAL_EDGES,
    num_orientations: int = 4,
) -> torch.Tensor:
    if resolution <= 0 or num_orientations <= 0:
        raise ValueError("resolution and num_orientations must be positive.")
    edges = tuple(float(value) for value in radial_edges)
    if len(edges) < 2 or any(
        not math.isfinite(value) for value in edges
    ):
        raise ValueError("radial_edges must contain finite boundaries.")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("radial_edges must be strictly increasing.")

    frequency = torch.fft.fftfreq(resolution) * resolution
    yy, xx = torch.meshgrid(frequency, frequency, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    # Orientation modulo pi assigns conjugate Fourier coordinates to the same
    # wedge, so every mask preserves a real-valued inverse transform.
    canonical_half = (yy > 0) | ((yy == 0) & (xx >= 0))
    canonical_y = torch.where(canonical_half, yy, -yy)
    canonical_x = torch.where(canonical_half, xx, -xx)
    orientation = torch.atan2(canonical_y, canonical_x)
    orientation_width = math.pi / num_orientations

    masks = []
    for radial_low, radial_high in zip(edges, edges[1:]):
        radial = (radius >= radial_low) & (radius < radial_high)
        for index in range(num_orientations):
            angular_low = index * orientation_width
            angular_high = (index + 1) * orientation_width
            angular = (orientation >= angular_low) & (
                orientation < angular_high
            )
            masks.append(radial & angular)
    return torch.stack(masks)


def construct_phase_amplitude_variants(
    anchor: torch.Tensor,
    donors: Sequence[torch.Tensor],
    *,
    resolution: int,
    radial_edges: Sequence[float] = DEFAULT_RADIAL_EDGES,
) -> torch.Tensor:
    if not donors:
        raise ValueError("donors must be non-empty.")
    anchor_luma = rgb_luminance(center_square_resize(anchor, resolution))
    anchor_phase = torch.angle(torch.fft.fft2(anchor_luma.double()))
    masks = radial_orientation_masks(
        resolution,
        radial_edges=radial_edges,
        num_orientations=1,
    )
    band = masks.any(dim=0)

    variants = []
    for index, donor in enumerate(donors):
        _validate_rgb_image(donor, name=f"donors[{index}]")
        donor_luma = rgb_luminance(center_square_resize(donor, resolution))
        amplitude = torch.fft.fft2(donor_luma.double()).abs()
        spectrum = amplitude * torch.exp(1j * anchor_phase)
        variants.append(torch.fft.ifft2(spectrum * band).real.float())
    return torch.stack(variants)


def construct_phase_scrambled_controls(
    variants: torch.Tensor,
    *,
    seed: int = 2101,
) -> torch.Tensor:
    if variants.ndim != 3 or min(variants.shape) <= 0:
        raise ValueError("variants must have shape [M,H,W].")
    if variants.shape[-2] != variants.shape[-1]:
        raise ValueError("variants must be square.")
    if not torch.isfinite(variants).all():
        raise ValueError("variants must contain only finite values.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    controls = []
    for variant in variants.detach().to(device="cpu", dtype=torch.float64):
        random_image = torch.randn(
            variant.shape,
            generator=generator,
            dtype=torch.float64,
        )
        random_phase = torch.angle(torch.fft.fft2(random_image))
        amplitude = torch.fft.fft2(variant).abs()
        controls.append(
            torch.fft.ifft2(amplitude * torch.exp(1j * random_phase))
            .real.float()
        )
    return torch.stack(controls).to(variants.device)


def decompose_radial_orientation_bases(
    variants: torch.Tensor,
    *,
    radial_edges: Sequence[float] = DEFAULT_RADIAL_EDGES,
    num_orientations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variants.ndim != 3 or variants.shape[-2] != variants.shape[-1]:
        raise ValueError("variants must have shape [M,R,R].")
    if not torch.isfinite(variants).all():
        raise ValueError("variants must contain only finite values.")
    resolution = variants.shape[-1]
    masks = radial_orientation_masks(
        resolution,
        radial_edges=radial_edges,
        num_orientations=num_orientations,
    ).to(variants.device)
    decomposed = []
    ranks = []
    reconstruction_scales = []
    for variant in variants.double():
        spectrum = torch.fft.fft2(variant)
        components = torch.stack(
            [torch.fft.ifft2(spectrum * mask).real for mask in masks]
        )
        bases = canonicalize_explicit_bases(components)
        scales = torch.einsum(
            "khw,khw->k",
            components.float(),
            bases,
        )
        scales /= scales.norm().clamp_min(1e-12)
        rank = int(torch.linalg.matrix_rank(bases.flatten(1).double()).item())
        if rank != masks.shape[0]:
            raise ValueError(
                f"Carrier basis rank {rank} is below {masks.shape[0]}."
            )
        decomposed.append(bases)
        ranks.append(rank)
        reconstruction_scales.append(scales)
    return (
        torch.stack(decomposed),
        torch.tensor(ranks, dtype=torch.int64),
        torch.stack(reconstruction_scales),
    )


def stable_variant_index(
    image_id: str,
    *,
    seed: int = 2102,
    num_variants: int = 4,
) -> int:
    if not image_id:
        raise ValueError("image_id must be non-empty.")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive.")
    payload = f"{int(seed)}:{image_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % num_variants


@dataclass(frozen=True)
class SemanticCarrierBank:
    variants: torch.Tensor
    controls: torch.Tensor
    semantic_bases: torch.Tensor
    control_bases: torch.Tensor
    semantic_ranks: torch.Tensor
    control_ranks: torch.Tensor
    semantic_scales: torch.Tensor
    control_scales: torch.Tensor
    bank_hash: str
    resolution: int
    radial_edges: tuple[float, ...]
    phase_seed: int


class VariantMatchedCanonicalCarrier(nn.Module):
    def __init__(
        self,
        bases: torch.Tensor,
        reconstruction_scales: torch.Tensor,
        *,
        epsilon: float,
        gamma: float,
        initial_coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        if bases.ndim not in (3, 4):
            raise ValueError("bases must have shape [K,H,W] or [V,K,H,W].")
        variant_bases = bases.unsqueeze(0) if bases.ndim == 3 else bases
        canonical = torch.stack(
            [canonicalize_explicit_bases(value) for value in variant_bases]
        )
        if reconstruction_scales.ndim == 1:
            reconstruction_scales = reconstruction_scales.unsqueeze(0)
        if reconstruction_scales.shape != canonical.shape[:2]:
            raise ValueError("reconstruction_scales must have shape [V,K].")
        if not torch.isfinite(reconstruction_scales).all():
            raise ValueError("reconstruction_scales must be finite.")
        if initial_coefficients.shape != (canonical.shape[1], 3):
            raise ValueError("initial_coefficients must have shape [K,3].")
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.register_buffer("bases", canonical)
        self.register_buffer(
            "reconstruction_scales",
            reconstruction_scales.detach().float().clone(),
        )
        self.coefficients = nn.Parameter(
            initial_coefficients.detach().float().clone()
        )

    @property
    def num_variants(self) -> int:
        return int(self.bases.shape[0])

    def forward(self) -> torch.Tensor:
        positive_modulation = 1.0 + torch.tanh(self.coefficients)
        pre_activation = torch.einsum(
            "kc,vk,vkhw->vchw",
            positive_modulation,
            self.reconstruction_scales,
            self.bases,
        )
        return self.epsilon * torch.tanh(self.gamma * pre_activation)


def calibrate_variant_shared_gamma(
    families: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    epsilon: float,
    seed: int = 2104,
    num_directions: int = 256,
    coefficient_max_abs: float = 0.25,
    target_rms_ratio: float = 0.35,
    iterations: int = 32,
    chunk_size: int = 4,
    device: torch.device | str | None = None,
) -> SharedGammaCalibration:
    if len(families) < 2:
        raise ValueError("At least two carrier families are required.")
    if num_directions <= 0 or iterations <= 0 or chunk_size <= 0:
        raise ValueError("Calibration counts must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first_bases = next(iter(families.values()))[0]
    num_bases = first_bases.shape[-3]
    directions = torch.randn(
        (num_directions, num_bases, 3),
        generator=generator,
    )
    directions *= float(coefficient_max_abs) / directions.abs().amax(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1e-12)
    calibration_device = torch.device(device or "cpu")
    prepared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for family, (bases, scales) in families.items():
        variant_bases = bases.unsqueeze(0) if bases.ndim == 3 else bases
        variant_scales = scales.unsqueeze(0) if scales.ndim == 1 else scales
        if variant_bases.shape[:2] != variant_scales.shape:
            raise ValueError(f"Carrier family {family} basis/scale mismatch.")
        if variant_bases.shape[1] != num_bases:
            raise ValueError("All carrier families must share K.")
        prepared[family] = (
            variant_bases.to(calibration_device),
            variant_scales.to(calibration_device),
        )

    target_rms = float(epsilon) * float(target_rms_ratio)

    def family_rms(gamma: float) -> dict[str, torch.Tensor]:
        result = {}
        for family, (bases, scales) in prepared.items():
            chunks = []
            for start in range(0, num_directions, chunk_size):
                modulation = 1.0 + torch.tanh(
                    directions[start : start + chunk_size].to(
                        calibration_device
                    )
                )
                pre_activation = torch.einsum(
                    "dkc,vk,vkhw->dvchw",
                    modulation,
                    scales,
                    bases,
                )
                chunks.append(
                    (
                        float(epsilon)
                        * torch.tanh(float(gamma) * pre_activation)
                    )
                    .square()
                    .mean(dim=(2, 3, 4))
                    .sqrt()
                    .flatten()
                    .detach()
                    .cpu()
                )
            result[family] = torch.cat(chunks)
        return result

    def pooled_median(gamma: float) -> float:
        pooled = torch.cat(list(family_rms(gamma).values()))
        return float(torch.quantile(pooled, 0.5))

    # The sub-band bases are orthonormal because their Fourier masks are
    # disjoint. Use that exact energy identity for a near-linear starting
    # point, then refine against the actual tanh patterns. This keeps the
    # frozen 256 directions while avoiding dozens of full 640x640 passes.
    analytic_rms = []
    for bases, scales in prepared.values():
        gram = torch.einsum(
            "vkhw,vlhw->vkl",
            bases,
            bases,
        )
        identity = torch.eye(
            bases.shape[1],
            device=gram.device,
            dtype=gram.dtype,
        ).expand_as(gram)
        if float((gram - identity).abs().amax()) > 1e-5:
            raise ValueError("Variant bases must be orthonormal for calibration.")
        modulation = 1.0 + torch.tanh(directions.to(calibration_device))
        coefficient_energy = (
            modulation.unsqueeze(1) * scales.unsqueeze(0).unsqueeze(-1)
        ).square().sum(dim=(2, 3))
        analytic_rms.append(
            torch.sqrt(
                coefficient_energy
                / (3.0 * bases.shape[-2] * bases.shape[-1])
            )
            .flatten()
            .cpu()
        )
    median_pre_activation_rms = float(
        torch.quantile(torch.cat(analytic_rms), 0.5)
    )
    gamma = target_rms / max(
        float(epsilon) * median_pre_activation_rms,
        1e-12,
    )
    used_iterations = 0
    for used_iterations in range(1, iterations + 1):
        current = pooled_median(gamma)
        relative_error = abs(current - target_rms) / target_rms
        if relative_error <= 1e-6:
            break
        gamma *= target_rms / max(current, 1e-12)
    values = family_rms(gamma)
    medians = {
        family: float(torch.quantile(rms, 0.5))
        for family, rms in values.items()
    }
    pooled = torch.cat(list(values.values()))
    return SharedGammaCalibration(
        gamma=gamma,
        target_rms=target_rms,
        pooled_median_rms=float(torch.quantile(pooled, 0.5)),
        family_median_rms=medians,
        family_rms_ratio=max(medians.values())
        / max(min(medians.values()), 1e-12),
        direction_hash=tensor_sha256(directions),
        num_directions=int(num_directions),
        iterations=int(used_iterations),
    )


def build_semantic_carrier_bank(
    anchor: torch.Tensor,
    donors: Sequence[torch.Tensor],
    *,
    resolution: int,
    phase_seed: int = 2101,
    radial_edges: Sequence[float] = DEFAULT_RADIAL_EDGES,
    num_orientations: int = 4,
) -> SemanticCarrierBank:
    variants = construct_phase_amplitude_variants(
        anchor,
        donors,
        resolution=resolution,
        radial_edges=radial_edges,
    )
    controls = construct_phase_scrambled_controls(variants, seed=phase_seed)
    semantic_bases, semantic_ranks, semantic_scales = (
        decompose_radial_orientation_bases(
        variants,
        radial_edges=radial_edges,
        num_orientations=num_orientations,
        )
    )
    control_bases, control_ranks, control_scales = (
        decompose_radial_orientation_bases(
        controls,
        radial_edges=radial_edges,
        num_orientations=num_orientations,
        )
    )
    metadata = {
        "resolution": int(resolution),
        "radial_edges": [float(value) for value in radial_edges],
        "num_orientations": int(num_orientations),
        "phase_seed": int(phase_seed),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for tensor in (
        variants,
        controls,
        semantic_bases,
        control_bases,
        semantic_scales,
        control_scales,
    ):
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return SemanticCarrierBank(
        variants=variants,
        controls=controls,
        semantic_bases=semantic_bases,
        control_bases=control_bases,
        semantic_ranks=semantic_ranks,
        control_ranks=control_ranks,
        semantic_scales=semantic_scales,
        control_scales=control_scales,
        bank_hash=digest.hexdigest(),
        resolution=int(resolution),
        radial_edges=tuple(float(value) for value in radial_edges),
        phase_seed=int(phase_seed),
    )
