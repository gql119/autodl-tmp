from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .semantic_hiding_carrier import SemanticHidingCarrier


def ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM inputs must align as [B,C,H,W].")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be a positive odd integer.")
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(
        prediction.square(), window_size, stride=1, padding=padding
    ) - mu_x.square()
    sigma_y = F.avg_pool2d(
        target.square(), window_size, stride=1, padding=padding
    ) - mu_y.square()
    sigma_xy = F.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding
    ) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    score = numerator / denominator.clamp_min(1e-12)
    return score.flatten(1).mean(dim=1)


def reveal_loss(
    recovered: torch.Tensor, secret: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = (recovered - secret).abs().flatten(1).mean(dim=1)
    ssim = ssim_per_sample(recovered, secret)
    per_sample = l1 + 0.2 * (1.0 - ssim)
    return per_sample.mean(), l1, ssim


def retrieval_statistics(
    recovered: torch.Tensor,
    secret_bank: torch.Tensor,
    true_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if recovered.ndim != 4 or secret_bank.ndim != 4:
        raise ValueError("recovered and secret_bank must be image batches.")
    if recovered.shape[1:] != secret_bank.shape[1:]:
        raise ValueError("recovered and secret_bank image shapes differ.")
    if true_indices.shape != (recovered.shape[0],):
        raise ValueError("true_indices must be [B].")
    if secret_bank.shape[0] < 2:
        raise ValueError("retrieval requires at least two candidate secrets.")
    if torch.any(true_indices < 0) or torch.any(true_indices >= secret_bank.shape[0]):
        raise ValueError("true_indices contains an invalid secret id.")
    pair_l1 = (
        recovered[:, None] - secret_bank[None]
    ).abs().flatten(2).mean(dim=2)
    predicted = pair_l1.argmin(dim=1)
    row = torch.arange(recovered.shape[0], device=recovered.device)
    true_l1 = pair_l1[row, true_indices]
    wrong = pair_l1.clone()
    wrong[row, true_indices] = float("inf")
    best_wrong_l1 = wrong.min(dim=1).values
    margin = (best_wrong_l1 - true_l1) / best_wrong_l1.clamp_min(1e-12)
    return {
        "pair_l1": pair_l1,
        "predicted_indices": predicted,
        "top1_accuracy": (predicted == true_indices).float().mean(),
        "true_l1": true_l1,
        "best_wrong_l1": best_wrong_l1,
        "relative_l1_margin": margin,
    }


def pairwise_pixel_cosine_median(deltas: torch.Tensor) -> torch.Tensor:
    if deltas.ndim != 4 or deltas.shape[0] < 2:
        raise ValueError("pixel diversity requires at least two [C,H,W] deltas.")
    flat = deltas.flatten(1)
    norms = flat.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError("pixel diversity is undefined for zero-norm deltas.")
    normalized = flat / norms[:, None]
    matrix = normalized @ normalized.transpose(0, 1)
    mask = torch.triu(
        torch.ones_like(matrix, dtype=torch.bool), diagonal=1
    )
    return matrix[mask].median()


def channel_rms_cv(deltas: torch.Tensor) -> torch.Tensor:
    if deltas.ndim != 4 or deltas.shape[0] < 2:
        raise ValueError("RMS diversity requires at least two deltas.")
    rms = deltas.square().mean(dim=(-2, -1)).sqrt()
    mean = rms.mean(dim=0)
    if torch.any(mean <= 1e-12):
        raise ValueError("RMS diversity is undefined for a zero-energy channel.")
    return rms.std(dim=0, unbiased=False) / mean


def high_frequency_energy_ratio(
    deltas: torch.Tensor, high_radius: float = 64.0
) -> torch.Tensor:
    if deltas.ndim != 4 or deltas.shape[1] != 3:
        raise ValueError("spectrum input must be [B,3,H,W].")
    if high_radius <= 0 or not math.isfinite(high_radius):
        raise ValueError("high_radius must be positive and finite.")
    luminance = (
        0.2126 * deltas[:, 0]
        + 0.7152 * deltas[:, 1]
        + 0.0722 * deltas[:, 2]
    )
    luminance = luminance - luminance.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fftshift(
        torch.fft.fft2(luminance), dim=(-2, -1)
    ).abs().square()
    height, width = luminance.shape[-2:]
    yy = torch.arange(height, device=deltas.device, dtype=deltas.dtype)
    xx = torch.arange(width, device=deltas.device, dtype=deltas.dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(
        (yy - (height - 1) / 2.0).square()
        + (xx - (width - 1) / 2.0).square()
    )
    non_dc = radius >= 1.0
    high = radius >= high_radius
    denominator = power[:, non_dc].sum(dim=1)
    if torch.any(denominator <= 1e-20):
        raise ValueError("spectrum ratio is undefined for zero-energy deltas.")
    return power[:, high].sum(dim=1) / denominator


def phase_scramble(
    images: torch.Tensor, seed: int
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("phase_scramble expects [B,3,H,W].")
    generator = torch.Generator(device=images.device)
    generator.manual_seed(int(seed))
    spectrum = torch.fft.fft2(images)
    random_image = torch.rand(
        images.shape, generator=generator, device=images.device, dtype=images.dtype
    )
    # The phase of a real random image already satisfies Hermitian symmetry, so
    # replacing phase preserves a real inverse and the source amplitude spectrum.
    phase = torch.angle(torch.fft.fft2(random_image))
    phase[..., 0, 0] = torch.angle(spectrum[..., 0, 0])
    scrambled = torch.fft.ifft2(
        spectrum.abs() * torch.exp(1j * phase)
    ).real
    low = scrambled.amin(dim=(-2, -1), keepdim=True)
    high = scrambled.amax(dim=(-2, -1), keepdim=True)
    return (scrambled - low) / (high - low).clamp_min(1e-12)


@dataclass
class HidingMetrics:
    retrieval_top1: float
    primary_recovery_ssim_median: float
    primary_relative_l1_margin_median: float
    pairwise_pixel_cosine_median: float
    channel_rms_cv: Tuple[float, float, float]
    high_frequency_energy_median: float
    linf: float
    support_outside_max: float
    all_finite: bool


def compute_hiding_metrics(
    recovered: torch.Tensor,
    true_secrets: torch.Tensor,
    secret_bank: torch.Tensor,
    true_indices: torch.Tensor,
    final_deltas: torch.Tensor,
    primary_index: int,
    support_outside_max: float = 0.0,
    high_radius: float = 64.0,
) -> HidingMetrics:
    retrieval = retrieval_statistics(recovered, secret_bank, true_indices)
    _, _, ssim = reveal_loss(recovered, true_secrets)
    primary = true_indices == int(primary_index)
    if not torch.any(primary):
        raise ValueError("metrics require at least one primary-secret sample.")
    cosine = pairwise_pixel_cosine_median(final_deltas)
    rms_cv = channel_rms_cv(final_deltas)
    high = high_frequency_energy_ratio(final_deltas, high_radius=high_radius)
    values = torch.stack(
        (
            retrieval["top1_accuracy"],
            ssim[primary].median(),
            retrieval["relative_l1_margin"][primary].median(),
            cosine,
            rms_cv.min(),
            high.median(),
            final_deltas.abs().max(),
        )
    )
    return HidingMetrics(
        retrieval_top1=float(retrieval["top1_accuracy"].detach()),
        primary_recovery_ssim_median=float(ssim[primary].median().detach()),
        primary_relative_l1_margin_median=float(
            retrieval["relative_l1_margin"][primary].median().detach()
        ),
        pairwise_pixel_cosine_median=float(cosine.detach()),
        channel_rms_cv=tuple(float(value) for value in rms_cv.detach()),
        high_frequency_energy_median=float(high.median().detach()),
        linf=float(final_deltas.detach().abs().max()),
        support_outside_max=float(support_outside_max),
        all_finite=bool(torch.isfinite(values).all()),
    )


def evaluate_hiding_gate(
    metrics: HidingMetrics,
    epsilon: float = 16.0 / 255.0,
) -> Dict[str, object]:
    checks = {
        "retrieval_top1": metrics.retrieval_top1 >= 0.90,
        "primary_recovery_ssim": metrics.primary_recovery_ssim_median >= 0.50,
        "primary_l1_margin": metrics.primary_relative_l1_margin_median >= 0.20,
        "pixel_diversity": metrics.pairwise_pixel_cosine_median < 0.98,
        "rms_diversity": min(metrics.channel_rms_cv) >= 0.05,
        "delta_high_frequency": metrics.high_frequency_energy_median <= 0.40,
        "linf": metrics.linf <= epsilon + 1.0 / 255.0,
        "support": metrics.support_outside_max == 0.0,
        "finite": metrics.all_finite,
    }
    return {
        "schema": "tausb.sdh-hiding-gate.v1",
        "metrics": asdict(metrics),
        "checks": checks,
        "pass": all(checks.values()),
        "claim_boundary": "hiding mechanics only; no detector or victim efficacy",
    }


def hiding_pretrain_step(
    carrier: SemanticHidingCarrier,
    optimizer: torch.optim.Optimizer,
    hosts: torch.Tensor,
    secrets: torch.Tensor,
    cover_weight: float = 0.01,
) -> Dict[str, float]:
    if cover_weight < 0 or not math.isfinite(cover_weight):
        raise ValueError("cover_weight must be finite and non-negative.")
    carrier.unfreeze_for_hiding_pretrain()
    optimizer.zero_grad(set_to_none=True)
    output = carrier(hosts, secrets)
    reveal, _, _ = reveal_loss(output.recovered_secret, secrets)
    cover = output.delta.square().mean()
    total = reveal + float(cover_weight) * cover
    if not torch.isfinite(total):
        raise ValueError("Non-finite hiding pretrain loss.")
    total.backward()
    gradients = [
        parameter.grad
        for parameter in carrier.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("Missing or non-finite hiding pretrain gradients.")
    optimizer.step()
    return {
        "total": float(total.detach()),
        "reveal": float(reveal.detach()),
        "cover": float(cover.detach()),
        "gradient_l1": float(
            sum(gradient.detach().abs().sum() for gradient in gradients)
        ),
    }
