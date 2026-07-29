from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


BandRange = tuple[float, float]


@dataclass(frozen=True)
class BackgroundSpectralBasis:
    bases: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    source_hash: str
    basis_hash: str
    phase_mode: str
    bands: tuple[BandRange, ...]
    resolution: int
    seed: int


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _looks_absolute(value: str) -> bool:
    if not value:
        return False
    paths: tuple[PurePath, ...] = (
        PureWindowsPath(value),
        PurePosixPath(value),
    )
    return any(path.is_absolute() for path in paths)


def validate_repository_manifest(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_sources: int = 8,
) -> None:
    if len(entries) != expected_sources:
        raise ValueError(
            f"Expected {expected_sources} source entries, got {len(entries)}."
        )

    stable_ids: set[str] = set()
    for index, entry in enumerate(entries):
        stable_id = str(entry.get("source_id", "")).strip()
        if not stable_id or _looks_absolute(stable_id):
            raise ValueError(
                f"Manifest entry {index} must use a non-absolute stable source_id."
            )
        if stable_id in stable_ids:
            raise ValueError(f"Duplicate source_id: {stable_id}")
        stable_ids.add(stable_id)

        if any(
            key in entry
            for key in ("absolute_path", "local_path", "source_path", "path")
        ):
            raise ValueError(
                f"Manifest entry {stable_id} contains a local path field."
            )

        digest = str(entry.get("sha256", "")).strip()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Manifest entry {stable_id} has an invalid SHA256.")
        if int(entry.get("width", 0)) <= 0 or int(entry.get("height", 0)) <= 0:
            raise ValueError(f"Manifest entry {stable_id} has invalid dimensions.")
        if not str(entry.get("license_note", "")).strip():
            raise ValueError(f"Manifest entry {stable_id} lacks a license note.")
        if entry.get("person_free") is not True:
            raise ValueError(
                f"Manifest entry {stable_id} is not confirmed person-free."
            )


def _to_float_chw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(
            "Each source image must have shape [H,W], [1,H,W], or [3,H,W]."
        )
    image = image.detach().to(device="cpu")
    if not image.dtype.is_floating_point:
        image = image.float().div(255.0)
    else:
        image = image.float()
    if not torch.isfinite(image).all():
        raise ValueError("Source images must contain only finite values.")
    return image


def _square_crop(image: torch.Tensor, y0: int, x0: int, side: int) -> torch.Tensor:
    return image[:, y0 : y0 + side, x0 : x0 + side]


def deterministic_two_crops(
    image: torch.Tensor,
    *,
    resolution: int,
    source_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = _to_float_chw(image)
    _, height, width = image.shape
    side = min(height, width)
    center_y = (height - side) // 2
    center_x = (width - side) // 2
    center = _square_crop(image, center_y, center_x, side)

    # Use a smaller second crop so square source images do not yield two
    # identical samples.  The alternating corner remains deterministic.
    corner_side = max(1, int(round(side * 0.875)))
    if source_index % 2 == 0:
        corner_y, corner_x = 0, 0
    else:
        corner_y = height - corner_side
        corner_x = width - corner_side
    corner = _square_crop(image, corner_y, corner_x, corner_side)

    resized = tuple(
        F.interpolate(
            crop.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )[0]
        for crop in (center, corner)
    )
    return resized


def _to_luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 1:
        return image[0]
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (image * weights).sum(dim=0)


def radial_frequency_grid(height: int, width: int) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0) * float(height)
    fx = torch.fft.fftfreq(width, d=1.0) * float(width)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(xx.square() + yy.square())


def band_mask(
    height: int,
    width: int,
    bands: Sequence[BandRange],
) -> torch.Tensor:
    if not bands:
        raise ValueError("At least one frequency band is required.")
    radius = radial_frequency_grid(height, width)
    mask = torch.zeros((height, width), dtype=torch.bool)
    for low, high in bands:
        low_f, high_f = float(low), float(high)
        if low_f < 0 or high_f <= low_f:
            raise ValueError(f"Invalid band range: ({low_f}, {high_f})")
        mask |= (radius >= low_f) & (radius < high_f)
    mask[0, 0] = False
    if not bool(mask.any()):
        raise ValueError("The requested bands contain no FFT bins.")
    return mask


def _band_limited_sample(
    luminance: torch.Tensor,
    *,
    mask: torch.Tensor,
    phase_mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    spectrum = torch.fft.fft2(luminance.double())
    magnitude = spectrum.abs()
    if phase_mode == "raw":
        filtered = spectrum * mask
    elif phase_mode == "scrambled":
        noise = torch.randn(
            luminance.shape,
            generator=generator,
            dtype=torch.float64,
        )
        random_phase = torch.angle(torch.fft.fft2(noise))
        filtered = magnitude * torch.polar(
            torch.ones_like(random_phase),
            random_phase,
        )
        filtered = filtered * mask
    else:
        raise ValueError("phase_mode must be 'raw' or 'scrambled'.")
    filtered[0, 0] = 0
    sample = torch.fft.ifft2(filtered).real
    return sample - sample.mean()


def _canonicalize_basis_sign(bases: torch.Tensor) -> torch.Tensor:
    canonical = bases.clone()
    flat = canonical.flatten(1)
    max_indices = flat.abs().argmax(dim=1)
    signs = flat.gather(1, max_indices.unsqueeze(1)).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return canonical * signs.view(-1, 1, 1)


def _tensor_digest(tensor: torch.Tensor, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    payload = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest.update(payload.numpy().tobytes())
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def build_background_spectral_basis(
    source_images: Sequence[torch.Tensor],
    *,
    resolution: int = 640,
    num_bases: int = 16,
    bands: Sequence[BandRange] = ((2.0, 8.0),),
    phase_mode: str = "scrambled",
    seed: int = 0,
    min_rank: int = 8,
) -> BackgroundSpectralBasis:
    if len(source_images) != 8:
        raise ValueError(f"Expected 8 source images, got {len(source_images)}.")
    if resolution <= 0 or num_bases <= 0:
        raise ValueError("resolution and num_bases must be positive.")

    samples: list[torch.Tensor] = []
    mask = band_mask(resolution, resolution, bands)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    source_tensors: list[torch.Tensor] = []

    for source_index, image in enumerate(source_images):
        crops = deterministic_two_crops(
            image,
            resolution=resolution,
            source_index=source_index,
        )
        for crop in crops:
            source_tensors.append(crop)
            sample = _band_limited_sample(
                _to_luminance(crop),
                mask=mask,
                phase_mode=phase_mode,
                generator=generator,
            )
            norm = sample.norm().clamp_min(1e-12)
            samples.append(sample / norm)

    matrix = torch.stack(samples, dim=0).flatten(1).double()
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        raise ValueError("SVD returned no singular values.")
    tolerance = singular_values[0] * 1e-6
    rank = int((singular_values > tolerance).sum().item())
    if rank < min_rank:
        raise ValueError(f"Background basis rank {rank} is below {min_rank}.")
    if num_bases > rank:
        raise ValueError(
            f"Requested {num_bases} bases, but the usable rank is {rank}."
        )

    bases = vh[:num_bases].reshape(num_bases, resolution, resolution)
    bases = bases - bases.mean(dim=(-2, -1), keepdim=True)
    bases = bases / bases.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    bases = _canonicalize_basis_sign(bases).float()

    frozen_bands = tuple((float(low), float(high)) for low, high in bands)
    metadata = {
        "bands": frozen_bands,
        "num_bases": int(num_bases),
        "phase_mode": phase_mode,
        "resolution": int(resolution),
        "seed": int(seed),
    }
    source_hash = _tensor_digest(torch.stack(source_tensors), metadata)
    basis_hash = _tensor_digest(bases, metadata)
    return BackgroundSpectralBasis(
        bases=bases,
        singular_values=singular_values.float(),
        rank=rank,
        source_hash=source_hash,
        basis_hash=basis_hash,
        phase_mode=phase_mode,
        bands=frozen_bands,
        resolution=int(resolution),
        seed=int(seed),
    )


def spectrum_energy_ratios(
    pattern: torch.Tensor,
    *,
    low: BandRange = (2.0, 8.0),
    mid: BandRange = (8.0, 24.0),
    high: BandRange = (24.0, float("inf")),
) -> dict[str, float]:
    if pattern.ndim == 2:
        pattern = pattern.unsqueeze(0)
    if pattern.ndim != 3:
        raise ValueError("pattern must have shape [H,W] or [C,H,W].")
    spectrum = torch.fft.fft2(pattern.double(), dim=(-2, -1))
    # Sum per-channel energy. Averaging RGB in the spatial domain can cancel
    # equal-and-opposite channels and report a false band distribution.
    energy = spectrum.abs().square().sum(dim=0)
    radius = radial_frequency_grid(*pattern.shape[-2:])
    total = energy.sum().clamp_min(1e-12)

    def ratio(bounds: BandRange) -> float:
        lower, upper = bounds
        selected = (radius >= float(lower)) & (radius < float(upper))
        return float((energy[selected].sum() / total).item())

    return {
        "low": ratio(low),
        "mid": ratio(mid),
        "high": ratio(high),
        "dc": float((energy[0, 0] / total).item()),
    }
