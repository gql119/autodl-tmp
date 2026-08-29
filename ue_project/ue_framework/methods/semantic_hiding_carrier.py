from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class FixedHaarDWT(nn.Module):
    """One-level orthonormal Haar transform with no trainable state."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[-2] % 2 or image.shape[-1] % 2:
            raise ValueError("Haar DWT expects [B,C,H,W] with even H and W.")
        x00 = image[..., 0::2, 0::2]
        x01 = image[..., 0::2, 1::2]
        x10 = image[..., 1::2, 0::2]
        x11 = image[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat((ll, lh, hl, hh), dim=1)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 4 or coeffs.shape[1] % 4:
            raise ValueError("Haar inverse expects [B,4C,H,W].")
        ll, lh, hl, hh = coeffs.chunk(4, dim=1)
        x00 = (ll - lh - hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll + lh + hl + hh) * 0.5
        batch, channels, height, width = x00.shape
        output = coeffs.new_empty(batch, channels, height * 2, width * 2)
        output[..., 0::2, 0::2] = x00
        output[..., 0::2, 1::2] = x01
        output[..., 1::2, 0::2] = x10
        output[..., 1::2, 1::2] = x11
        return output


class AffineCouplingBlock(nn.Module):
    def __init__(self, channels: int, width: int, swap: bool) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("Affine coupling channels must be even.")
        half = channels // 2
        self.swap = bool(swap)
        self.net = nn.Sequential(
            nn.Conv2d(half, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(width, half * 2, 3, padding=1),
        )
        final = self.net[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=1)
        if self.swap:
            first, second = second, first
        scale, shift = self.net(first).chunk(2, dim=1)
        transformed = second * torch.exp(0.5 * torch.tanh(scale)) + shift
        if self.swap:
            return torch.cat((transformed, first), dim=1)
        return torch.cat((first, transformed), dim=1)


class ResidualAdapter(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SecretRevealDecoder(nn.Module):
    def __init__(self, width: int, dwt: FixedHaarDWT) -> None:
        super().__init__()
        self.dwt = dwt
        self.layers = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(width, 12, 3, padding=1),
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        coeffs = self.layers(self.dwt(stego))
        return torch.sigmoid(self.dwt.inverse(coeffs))


@dataclass
class SemanticHidingOutput:
    raw_residual: torch.Tensor
    delta: torch.Tensor
    stego: torch.Tensor
    recovered_secret: torch.Tensor


class SemanticHidingCarrier(nn.Module):
    """Host-conditioned single-secret carrier used on canonical person crops."""

    def __init__(
        self,
        input_size: int = 256,
        width: int = 64,
        coupling_blocks: int = 4,
        epsilon: float = 16.0 / 255.0,
        hf_subband_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_size <= 0 or input_size % 2:
            raise ValueError("input_size must be a positive even integer.")
        if width <= 0 or coupling_blocks <= 0:
            raise ValueError("width and coupling_blocks must be positive.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")
        if (
            not math.isfinite(hf_subband_scale)
            or not 0.0 <= hf_subband_scale <= 1.0
        ):
            raise ValueError("hf_subband_scale must be finite in [0,1].")
        self.input_size = int(input_size)
        self.width = int(width)
        self.coupling_blocks = int(coupling_blocks)
        self.epsilon = float(epsilon)
        self.hf_subband_scale = float(hf_subband_scale)
        self.dwt = FixedHaarDWT()
        self.hiding_trunk = nn.ModuleList(
            AffineCouplingBlock(24, self.width, swap=bool(index % 2))
            for index in range(self.coupling_blocks)
        )
        self.adapter = ResidualAdapter(self.width)
        self.reveal_decoder = SecretRevealDecoder(self.width, self.dwt)

    def _filter_residual_subbands(self, raw_residual: torch.Tensor) -> torch.Tensor:
        # The identity branch deliberately avoids a DWT round trip so scale=1.0
        # is an exact output/gradient rollback for existing checkpoints.
        if self.hf_subband_scale == 1.0:
            return raw_residual
        ll, lh, hl, hh = self.dwt(raw_residual).chunk(4, dim=1)
        scale = self.hf_subband_scale
        return self.dwt.inverse(
            torch.cat((ll, lh * scale, hl * scale, hh * scale), dim=1)
        )

    def _validate_inputs(
        self, host: torch.Tensor, secret: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = (self.input_size, self.input_size)
        if host.ndim != 4 or host.shape[1] != 3 or tuple(host.shape[-2:]) != expected:
            raise ValueError("host must be [B,3,input_size,input_size].")
        if secret.ndim != 4 or secret.shape[1] != 3 or tuple(secret.shape[-2:]) != expected:
            raise ValueError("secret must be [B or 1,3,input_size,input_size].")
        if secret.shape[0] == 1 and host.shape[0] != 1:
            secret = secret.expand(host.shape[0], -1, -1, -1)
        if host.shape[0] != secret.shape[0]:
            raise ValueError("host and secret batch sizes do not align.")
        if not torch.isfinite(host).all() or not torch.isfinite(secret).all():
            raise ValueError("host and secret must be finite.")
        return host, secret

    def forward(
        self, host: torch.Tensor, secret: torch.Tensor
    ) -> SemanticHidingOutput:
        host, secret = self._validate_inputs(host, secret)
        features = torch.cat((self.dwt(host), self.dwt(secret)), dim=1)
        for block in self.hiding_trunk:
            features = block(features)
        mixed_spatial = self.dwt.inverse(features[:, :12])
        raw_residual = self.adapter(mixed_spatial)
        filtered_residual = self._filter_residual_subbands(raw_residual)
        delta = self.epsilon * torch.tanh(filtered_residual)
        stego = torch.clamp(host + delta, 0.0, 1.0)
        recovered = self.reveal_decoder(stego)
        return SemanticHidingOutput(raw_residual, delta, stego, recovered)

    def freeze_for_detector_optimization(self) -> None:
        for parameter in self.hiding_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.reveal_decoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(True)

    def unfreeze_for_hiding_pretrain(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def architecture_descriptor(self) -> Dict[str, object]:
        descriptor = {
            "input_size": self.input_size,
            "dwt": "fixed_haar_one_level",
            "host_wavelet_channels": 12,
            "secret_wavelet_channels": 12,
            "coupling_channels": 24,
            "coupling_blocks": self.coupling_blocks,
            "coupling_width": self.width,
            "coupling_kernel": 3,
            "activation": "LeakyReLU(0.2)",
            "batch_norm": False,
            "dropout": False,
            "adapter": "Conv3x3(3,width)-LeakyReLU-Conv3x3(width,3)",
            "epsilon": self.epsilon,
            "parameter_counts": {
                "hiding_trunk": sum(p.numel() for p in self.hiding_trunk.parameters()),
                "adapter": sum(p.numel() for p in self.adapter.parameters()),
                "reveal_decoder": sum(
                    p.numel() for p in self.reveal_decoder.parameters()
                ),
                "total": sum(p.numel() for p in self.parameters()),
            },
        }
        # Preserve the legacy architecture hash for the exact scale=1 rollback.
        if self.hf_subband_scale != 1.0:
            descriptor["hf_subband_scale"] = self.hf_subband_scale
        return descriptor

    def architecture_sha256(self) -> str:
        payload = json.dumps(
            self.architecture_descriptor(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class RenderedSemanticCarrier:
    poisoned: torch.Tensor
    perturbation: torch.Tensor
    union_support: torch.Tensor
    canonical_deltas: Tuple[torch.Tensor, ...]
    recovered_secrets: Tuple[torch.Tensor, ...]


def _integer_box(
    box: torch.Tensor, image_height: int, image_width: int
) -> Optional[Tuple[int, int, int, int]]:
    if box.numel() != 4 or not torch.isfinite(box).all():
        return None
    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
    left = max(0, min(image_width, int(math.floor(x1))))
    top = max(0, min(image_height, int(math.floor(y1))))
    right = max(0, min(image_width, int(math.ceil(x2))))
    bottom = max(0, min(image_height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _linear_interpolation_matrix(
    source_size: int,
    target_size: int,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if source_size <= 0 or target_size <= 0:
        raise ValueError("Resize dimensions must be positive.")
    positions = (
        (
            torch.arange(
                target_size,
                device=reference.device,
                dtype=reference.dtype,
            )
            + 0.5
        )
        * (float(source_size) / float(target_size))
        - 0.5
    )
    lower_unclamped = torch.floor(positions)
    upper_weight = positions - lower_unclamped
    lower_weight = 1.0 - upper_weight
    lower = lower_unclamped.clamp(0, source_size - 1).to(torch.long)
    upper = (lower_unclamped + 1.0).clamp(0, source_size - 1).to(torch.long)
    source_indices = torch.arange(source_size, device=reference.device).unsqueeze(0)
    matrix = (
        (source_indices == lower.unsqueeze(1)) * lower_weight.unsqueeze(1)
        + (source_indices == upper.unsqueeze(1)) * upper_weight.unsqueeze(1)
    )
    return matrix.to(dtype=reference.dtype)


def deterministic_bilinear_resize_2d(
    inputs: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """Bilinear resize with a matmul-only gradient path.

    The coordinate rule matches ``align_corners=False``. Fixed interpolation
    weights keep CUDA autograd away from nondeterministic bilinear-upsample
    backward kernels while preserving gradients with respect to ``inputs``.
    """

    if inputs.ndim != 4:
        raise ValueError("deterministic_bilinear_resize_2d expects [B,C,H,W].")
    if not inputs.is_floating_point():
        raise ValueError("deterministic_bilinear_resize_2d expects floating input.")
    target_height, target_width = (int(size[0]), int(size[1]))
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Resize dimensions must be positive.")
    source_height, source_width = inputs.shape[-2:]
    if (source_height, source_width) == (target_height, target_width):
        return inputs
    height_matrix = _linear_interpolation_matrix(
        source_height,
        target_height,
        reference=inputs,
    )
    width_matrix = _linear_interpolation_matrix(
        source_width,
        target_width,
        reference=inputs,
    )
    resized_height = torch.matmul(height_matrix, inputs)
    return torch.matmul(resized_height, width_matrix.transpose(0, 1))


def render_person_box_carrier(
    images: torch.Tensor,
    boxes_by_image: Sequence[torch.Tensor],
    carrier: SemanticHidingCarrier,
    secret: torch.Tensor,
    *,
    trace_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
) -> RenderedSemanticCarrier:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [B,3,H,W].")
    if len(boxes_by_image) != images.shape[0]:
        raise ValueError("boxes_by_image length must match image batch size.")
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite.")
    batch, _, height, width = images.shape
    accumulated = torch.zeros_like(images)
    counts = images.new_zeros(batch, 1, height, width)
    canonical: List[torch.Tensor] = []
    recovered: List[torch.Tensor] = []
    traced_hosts: List[torch.Tensor] = []
    traced_patches: List[torch.Tensor] = []
    for batch_index, boxes in enumerate(boxes_by_image):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("each boxes tensor must be [N,4] pixel xyxy.")
        for box in boxes:
            coords = _integer_box(box, height, width)
            if coords is None:
                continue
            left, top, right, bottom = coords
            crop = images[batch_index : batch_index + 1, :, top:bottom, left:right]
            with torch.no_grad():
                host = F.interpolate(
                    crop,
                    size=(carrier.input_size, carrier.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            output = carrier(host, secret)
            patch = deterministic_bilinear_resize_2d(
                output.delta,
                (bottom - top, right - left),
            )
            accumulated[batch_index, :, top:bottom, left:right] += patch[0]
            counts[batch_index, :, top:bottom, left:right] += 1.0
            canonical.append(output.delta)
            recovered.append(output.recovered_secret)
            if trace_callback is not None:
                traced_hosts.append(host)
                traced_patches.append(patch)
    union = counts > 0
    perturbation = torch.where(
        union.expand_as(accumulated), accumulated / counts.clamp_min(1.0), accumulated
    )
    poisoned = torch.clamp(images + perturbation, 0.0, 1.0)
    perturbation = poisoned - images
    if float(perturbation.detach().abs().max()) > carrier.epsilon + 1e-6:
        raise RuntimeError("Rendered semantic carrier exceeded epsilon.")
    outside = perturbation * (~union).expand_as(perturbation)
    if torch.count_nonzero(outside).item() != 0:
        raise RuntimeError("Rendered semantic carrier leaked outside person boxes.")
    if trace_callback is not None:
        trace_callback(
            "render",
            {
                "hosts": tuple(traced_hosts),
                "canonical_deltas": tuple(canonical),
                "resized_patches": tuple(traced_patches),
                "perturbation": perturbation,
                "poisoned": poisoned,
            },
        )
    return RenderedSemanticCarrier(
        poisoned=poisoned,
        perturbation=perturbation,
        union_support=union,
        canonical_deltas=tuple(canonical),
        recovered_secrets=tuple(recovered),
    )
