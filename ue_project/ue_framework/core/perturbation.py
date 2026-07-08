from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from ue_framework.methods.fourier import build_fourier_pattern


class UniversalPerturbation(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], max_norm: float) -> None:
        super().__init__()
        channels, height, width = shape
        self.delta = nn.Parameter(torch.zeros(1, channels, height, width))
        self.max_norm = float(max_norm)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images + self.delta.clamp(-self.max_norm, self.max_norm)).clamp(0.0, 1.0)

    @torch.no_grad()
    def project_(self) -> None:
        self.delta.clamp_(min=-self.max_norm, max=self.max_norm)


class UniversalFourierPerturbation(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        coords: Sequence[Tuple[int, int]],
        max_norm: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.coords = [(int(y), int(x)) for y, x in coords]
        self.max_norm = float(max_norm)
        self.amplitudes = nn.Parameter(torch.zeros(len(self.coords)))
        self.channel_weights = nn.Parameter(torch.ones(1, self.channels, 1, 1))

    def delta(self, device: torch.device) -> torch.Tensor:
        pattern = build_fourier_pattern(self.height, self.width, self.coords, self.amplitudes, device=device)
        delta = pattern.expand(1, self.channels, self.height, self.width) * self.channel_weights.to(device)
        return delta.clamp(-self.max_norm, self.max_norm)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images + self.delta(images.device)).clamp(0.0, 1.0)
