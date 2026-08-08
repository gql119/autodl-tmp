from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(payload.numpy().tobytes()).hexdigest()


def canonicalize_explicit_bases(bases: torch.Tensor) -> torch.Tensor:
    if bases.ndim != 3:
        raise ValueError("bases must have shape [K,H,W].")
    if min(bases.shape) <= 0:
        raise ValueError("bases must have non-empty dimensions.")
    if not torch.isfinite(bases).all():
        raise ValueError("bases must contain only finite values.")

    canonical = bases.detach().to(device="cpu", dtype=torch.float64).clone()
    canonical -= canonical.mean(dim=(-2, -1), keepdim=True)
    norms = canonical.flatten(1).norm(dim=1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("bases must have non-zero energy after DC removal.")
    canonical /= norms.view(-1, 1, 1)

    canonical = canonical.float()
    flat = canonical.flatten(1)
    max_indices = flat.abs().argmax(dim=1, keepdim=True)
    signs = flat.gather(1, max_indices).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    canonical *= signs.view(-1, 1, 1)
    return canonical


def build_synthetic_fourier_bases(
    resolution: int,
    coords: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if not coords:
        raise ValueError("coords must be non-empty.")

    bases = []
    for y_value, x_value in coords:
        y = int(y_value) % resolution
        x = int(x_value) % resolution
        if y == 0 and x == 0:
            raise ValueError("The DC Fourier coordinate is not allowed.")
        spectrum = torch.zeros(
            (resolution, resolution),
            dtype=torch.complex128,
        )
        spectrum[y, x] += 1.0
        spectrum[-y % resolution, -x % resolution] += 1.0
        bases.append(torch.fft.ifft2(spectrum).real)
    return canonicalize_explicit_bases(torch.stack(bases))


def common_initial_coefficients(
    num_bases: int,
    *,
    seed: int = 2033,
    max_abs: float = 0.25,
) -> torch.Tensor:
    if num_bases <= 0:
        raise ValueError("num_bases must be positive.")
    if max_abs <= 0:
        raise ValueError("max_abs must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    coefficients = torch.randn(
        (num_bases, 3),
        generator=generator,
        dtype=torch.float32,
    )
    coefficients *= float(max_abs) / coefficients.abs().amax().clamp_min(1e-12)
    return coefficients


def canonical_pattern(
    bases: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    epsilon: float,
    gamma: float | torch.Tensor,
) -> torch.Tensor:
    if bases.ndim != 3:
        raise ValueError("bases must have shape [K,H,W].")
    if coefficients.shape != (bases.shape[0], 3):
        raise ValueError("coefficients must have shape [K,3].")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")
    if not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must contain only finite values.")

    bases_on_device = bases.to(
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    pre_activation = torch.einsum(
        "kc,khw->chw",
        coefficients,
        bases_on_device,
    )
    gamma_tensor = torch.as_tensor(
        gamma,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    if gamma_tensor.numel() != 1 or not torch.isfinite(gamma_tensor):
        raise ValueError("gamma must be one finite scalar.")
    if bool(gamma_tensor <= 0):
        raise ValueError("gamma must be positive.")
    return float(epsilon) * torch.tanh(gamma_tensor * pre_activation)


def _pooled_pattern_rms(
    pre_activations: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    gamma: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    family_values: dict[str, torch.Tensor] = {}
    for family, values in pre_activations.items():
        rms_parts = []
        for start in range(0, values.shape[0], chunk_size):
            chunk = values[start : start + chunk_size]
            patterns = float(epsilon) * torch.tanh(float(gamma) * chunk)
            rms_parts.append(
                patterns.square().mean(dim=(1, 2, 3)).sqrt().detach().cpu()
            )
        family_values[family] = torch.cat(rms_parts)
    pooled = torch.cat(list(family_values.values()))
    return pooled, family_values


@dataclass(frozen=True)
class SharedGammaCalibration:
    gamma: float
    target_rms: float
    pooled_median_rms: float
    family_median_rms: dict[str, float]
    family_rms_ratio: float
    direction_hash: str
    num_directions: int
    iterations: int


def calibrate_shared_gamma(
    bases_by_family: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    device: torch.device | str | None = None,
    seed: int = 2032,
    num_directions: int = 256,
    coefficient_max_abs: float = 0.25,
    target_rms_ratio: float = 0.35,
    iterations: int = 32,
    chunk_size: int = 16,
) -> SharedGammaCalibration:
    if len(bases_by_family) < 2:
        raise ValueError("Shared calibration requires at least two basis families.")
    if num_directions <= 0 or iterations <= 0 or chunk_size <= 0:
        raise ValueError("Calibration counts must be positive.")
    if (
        not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
        or coefficient_max_abs <= 0
        or not 0 < target_rms_ratio < 1
    ):
        raise ValueError("Invalid calibration scale or target ratio.")

    family_names = sorted(bases_by_family)
    canonical_bases = {
        family: canonicalize_explicit_bases(bases_by_family[family])
        for family in family_names
    }
    shapes = {tuple(bases.shape) for bases in canonical_bases.values()}
    if len(shapes) != 1:
        raise ValueError("Basis families must have identical [K,H,W] shapes.")
    num_bases = next(iter(canonical_bases.values())).shape[0]
    calibration_device = torch.device(device or "cpu")
    canonical_bases = {
        family: bases.to(calibration_device)
        for family, bases in canonical_bases.items()
    }

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(
        (num_directions, num_bases, 3),
        generator=generator,
        dtype=torch.float32,
    )
    direction_norms = directions.abs().amax(dim=(1, 2), keepdim=True)
    directions *= float(coefficient_max_abs) / direction_norms.clamp_min(1e-12)

    pre_activations = {}
    for family in family_names:
        bases = canonical_bases[family]
        device_directions = directions.to(device=bases.device, dtype=bases.dtype)
        pre_activations[family] = torch.einsum(
            "dkc,khw->dchw",
            device_directions,
            bases,
        )

    target_rms = float(epsilon) * float(target_rms_ratio)

    def pooled_median(gamma: float) -> float:
        pooled, _ = _pooled_pattern_rms(
            pre_activations,
            epsilon=epsilon,
            gamma=gamma,
            chunk_size=chunk_size,
        )
        return float(torch.quantile(pooled, 0.5).item())

    lower = 0.0
    upper = 1.0
    while pooled_median(upper) < target_rms:
        upper *= 2.0
        if upper > 1e8:
            raise RuntimeError("Unable to bracket shared gamma.")
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if pooled_median(midpoint) < target_rms:
            lower = midpoint
        else:
            upper = midpoint
    gamma = (lower + upper) / 2.0

    pooled, family_values = _pooled_pattern_rms(
        pre_activations,
        epsilon=epsilon,
        gamma=gamma,
        chunk_size=chunk_size,
    )
    medians = {
        family: float(torch.quantile(values, 0.5).item())
        for family, values in family_values.items()
    }
    median_values = [medians[family] for family in family_names]
    ratio = max(median_values) / max(min(median_values), 1e-12)
    return SharedGammaCalibration(
        gamma=float(gamma),
        target_rms=target_rms,
        pooled_median_rms=float(torch.quantile(pooled, 0.5).item()),
        family_median_rms=medians,
        family_rms_ratio=float(ratio),
        direction_hash=tensor_sha256(directions),
        num_directions=int(num_directions),
        iterations=int(iterations),
    )


class MatchedCanonicalCarrier(nn.Module):
    def __init__(
        self,
        bases: torch.Tensor,
        *,
        epsilon: float,
        gamma: float,
        initial_coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        canonical_bases = canonicalize_explicit_bases(bases)
        if initial_coefficients.shape != (canonical_bases.shape[0], 3):
            raise ValueError("initial_coefficients must have shape [K,3].")
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.register_buffer("bases", canonical_bases)
        self.coefficients = nn.Parameter(
            initial_coefficients.detach().float().clone()
        )

    def forward(self) -> torch.Tensor:
        return canonical_pattern(
            self.bases,
            self.coefficients,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )


@dataclass(frozen=True)
class RenderedCanonicalPattern:
    spatial_pattern: torch.Tensor
    union_support: torch.Tensor
    overlap_count: torch.Tensor


def warp_canonical_patch(
    pattern: torch.Tensor,
    box: tuple[int, int, int, int],
) -> torch.Tensor:
    if pattern.ndim != 3 or pattern.shape[0] != 3:
        raise ValueError("pattern must have shape [3,Hc,Wc].")
    x1, y1, x2, y2 = (int(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height.")
    return F.interpolate(
        pattern.unsqueeze(0),
        size=(y2 - y1, x2 - x1),
        mode="bilinear",
        align_corners=False,
    )[0]


def affine_canonical_pattern(
    pattern: torch.Tensor,
    *,
    scale: float = 1.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> torch.Tensor:
    if pattern.ndim != 3 or pattern.shape[0] != 3:
        raise ValueError("pattern must have shape [3,Hc,Wc].")
    if not all(
        math.isfinite(float(value))
        for value in (scale, translate_x, translate_y)
    ):
        raise ValueError("Affine parameters must be finite.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    theta = pattern.new_tensor(
        [
            [
                [1.0 / float(scale), 0.0, -2.0 * float(translate_x)],
                [0.0, 1.0 / float(scale), -2.0 * float(translate_y)],
            ]
        ]
    )
    grid = F.affine_grid(
        theta,
        size=(1, 3, *pattern.shape[-2:]),
        align_corners=False,
    )
    return F.grid_sample(
        pattern.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]


def render_canonical_pattern(
    pattern: torch.Tensor,
    *,
    image_size: tuple[int, int],
    boxes: Sequence[tuple[int, int, int, int]],
    instance_supports: torch.Tensor,
    mode: str,
) -> RenderedCanonicalPattern:
    if pattern.ndim != 3 or pattern.shape[0] != 3:
        raise ValueError("pattern must have shape [3,Hc,Wc].")
    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive.")
    if mode not in {"global", "instance"}:
        raise ValueError("mode must be 'global' or 'instance'.")
    expected_support_shape = (len(boxes), 1, height, width)
    if instance_supports.shape != expected_support_shape:
        raise ValueError(
            f"instance_supports must have shape {expected_support_shape}."
        )
    supports = instance_supports.to(device=pattern.device, dtype=pattern.dtype)
    if not torch.isfinite(supports).all():
        raise ValueError("instance_supports must contain only finite values.")
    if bool(((supports < 0) | (supports > 1)).any()):
        raise ValueError("instance_supports must lie in [0,1].")
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(value) for value in box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("boxes must be clipped inside image_size.")
        support = supports[index, 0]
        if any(
            bool(torch.count_nonzero(region))
            for region in (
                support[:y1],
                support[y2:],
                support[y1:y2, :x1],
                support[y1:y2, x2:],
            )
        ):
            raise ValueError("Each instance support must stay inside its box.")

    overlap_count = supports.sum(dim=0)
    union_support = (overlap_count > 0).to(pattern.dtype)
    if not boxes:
        return RenderedCanonicalPattern(
            spatial_pattern=pattern.new_zeros((3, height, width)),
            union_support=union_support,
            overlap_count=overlap_count,
        )

    if mode == "global":
        global_pattern = F.interpolate(
            pattern.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        weighted_sum = (
            global_pattern.unsqueeze(0) * supports
        ).sum(dim=0)
    else:
        warped_canvases = []
        for box in boxes:
            x1, y1, x2, y2 = (int(value) for value in box)
            patch = warp_canonical_patch(pattern, box)
            warped_canvases.append(
                F.pad(
                    patch,
                    (x1, width - x2, y1, height - y2),
                )
            )
        weighted_sum = (
            torch.stack(warped_canvases) * supports
        ).sum(dim=0)

    spatial_pattern = weighted_sum / overlap_count.clamp_min(1.0)
    spatial_pattern *= union_support
    return RenderedCanonicalPattern(
        spatial_pattern=spatial_pattern,
        union_support=union_support,
        overlap_count=overlap_count,
    )


def jnd_map(images: torch.Tensor, *, floor: float = 0.5) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W].")
    if not 0 <= floor <= 1:
        raise ValueError("floor must lie in [0,1].")
    gray = (
        0.299 * images[:, 0:1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )
    kx = images.new_tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
    ky = images.new_tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    magnitude -= magnitude.amin(dim=(2, 3), keepdim=True)
    magnitude /= magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return float(floor) + (1.0 - float(floor)) * magnitude


def apply_canonical_pattern(
    images: torch.Tensor,
    pattern: torch.Tensor,
    *,
    boxes_by_image: Sequence[Sequence[tuple[int, int, int, int]]],
    supports_by_image: Sequence[torch.Tensor],
    mode: str,
    epsilon: float,
    jnd_floor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[RenderedCanonicalPattern]]:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W].")
    if len(boxes_by_image) != images.shape[0] or len(supports_by_image) != images.shape[0]:
        raise ValueError("Per-image boxes/supports must match batch size.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")

    rendered = [
        render_canonical_pattern(
            pattern,
            image_size=images.shape[-2:],
            boxes=boxes,
            instance_supports=supports,
            mode=mode,
        )
        for boxes, supports in zip(boxes_by_image, supports_by_image)
    ]
    spatial_patterns = torch.stack(
        [item.spatial_pattern for item in rendered]
    )
    raw = spatial_patterns * jnd_map(images, floor=jnd_floor)
    raw = raw.clamp(-float(epsilon), float(epsilon))
    poisoned = (images + raw).clamp(0, 1)
    return poisoned, poisoned - images, rendered


def apply_variant_canonical_patterns(
    images: torch.Tensor,
    patterns: torch.Tensor,
    *,
    variant_indices: Sequence[int],
    boxes_by_image: Sequence[Sequence[tuple[int, int, int, int]]],
    supports_by_image: Sequence[torch.Tensor],
    mode: str,
    epsilon: float,
    jnd_floor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[RenderedCanonicalPattern]]:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W].")
    if patterns.ndim != 4 or patterns.shape[1] != 3:
        raise ValueError("patterns must have shape [V,3,Hc,Wc].")
    batch_size = images.shape[0]
    if not (
        len(variant_indices)
        == len(boxes_by_image)
        == len(supports_by_image)
        == batch_size
    ):
        raise ValueError("All per-image inputs must match batch size.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")

    rendered = []
    for image_index, variant_index in enumerate(variant_indices):
        selected = int(variant_index)
        if not 0 <= selected < patterns.shape[0]:
            raise ValueError("variant_indices contains an out-of-range index.")
        pattern = patterns[selected]
        rendered.append(
            render_canonical_pattern(
                pattern,
                image_size=images.shape[-2:],
                boxes=boxes_by_image[image_index],
                instance_supports=supports_by_image[image_index],
                mode=mode,
            )
        )

    spatial_patterns = torch.stack(
        [item.spatial_pattern for item in rendered]
    )
    raw = spatial_patterns * jnd_map(images, floor=jnd_floor)
    raw = raw.clamp(-float(epsilon), float(epsilon))
    poisoned = (images + raw).clamp(0, 1)
    return poisoned, poisoned - images, rendered
