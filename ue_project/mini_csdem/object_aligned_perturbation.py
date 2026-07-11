from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PerturbationApplication:
    images: torch.Tensor
    raw_target_pixels: int
    effective_target_pixels: int
    target_instances: int

    @property
    def excluded_pixels(self) -> int:
        return self.raw_target_pixels - self.effective_target_pixels


class ObjectAlignedPerturbation(nn.Module):
    def __init__(self, object_size: int, epsilon: float, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        delta = torch.empty(1, 3, object_size, object_size)
        delta.uniform_(-0.25 * epsilon, 0.25 * epsilon, generator=generator)
        self.delta_object = nn.Parameter(delta)
        self.epsilon = float(epsilon)

    def forward(
        self,
        images: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        target_class_id: int,
        exclude_non_target_overlap: bool = True,
        non_target_dilation: int = 2,
    ) -> PerturbationApplication:
        if images.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        batch_size, _, height, width = images.shape
        classes = batch["cls"].reshape(-1).long()
        boxes = batch["bboxes"].reshape(-1, 4)
        batch_indices = batch["batch_idx"].reshape(-1).long()
        delta = self.delta_object.clamp(-self.epsilon, self.epsilon).to(images)
        canvases = []
        raw_total = 0
        effective_total = 0
        target_instances = 0

        for image_index in range(batch_size):
            image_canvas = torch.zeros_like(images[image_index])
            image_mask = torch.zeros((1, height, width), dtype=torch.bool, device=images.device)
            rows = batch_indices == image_index
            image_classes = classes[rows]
            image_boxes = boxes[rows]
            non_target_mask = torch.zeros_like(image_mask)
            if exclude_non_target_overlap:
                for box in image_boxes[image_classes != int(target_class_id)]:
                    x1, y1, x2, y2 = self._pixel_box(box, width, height, non_target_dilation)
                    non_target_mask[:, y1:y2, x1:x2] = True

            for box in image_boxes[image_classes == int(target_class_id)]:
                x1, y1, x2, y2 = self._pixel_box(box, width, height, 0)
                if x2 <= x1 or y2 <= y1:
                    continue
                target_instances += 1
                raw_total += (x2 - x1) * (y2 - y1)
                patch = F.interpolate(delta, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0]
                valid = ~non_target_mask[:, y1:y2, x1:x2]
                effective_total += int(valid.sum().item())
                padded_patch = F.pad(patch * valid.to(patch), (x1, width - x2, y1, height - y2))
                padded_valid = F.pad(valid, (x1, width - x2, y1, height - y2))
                image_canvas = torch.where(padded_valid.expand_as(image_canvas), padded_patch, image_canvas)
                image_mask = image_mask | padded_valid
            canvases.append(image_canvas * image_mask.to(image_canvas))

        perturbation = torch.stack(canvases, dim=0)
        return PerturbationApplication(
            images=(images + perturbation).clamp(0.0, 1.0),
            raw_target_pixels=raw_total,
            effective_target_pixels=effective_total,
            target_instances=target_instances,
        )

    @staticmethod
    def _pixel_box(box: torch.Tensor, width: int, height: int, dilation: int) -> Tuple[int, int, int, int]:
        x, y, w, h = [float(value) for value in box.detach().cpu()]
        x1 = max(0, int(round((x - 0.5 * w) * width)) - dilation)
        y1 = max(0, int(round((y - 0.5 * h) * height)) - dilation)
        x2 = min(width, int(round((x + 0.5 * w) * width)) + dilation)
        y2 = min(height, int(round((y + 0.5 * h) * height)) + dilation)
        return x1, y1, x2, y2

    @torch.no_grad()
    def project_(self) -> None:
        self.delta_object.clamp_(-self.epsilon, self.epsilon)

    def statistics(self) -> Dict[str, float]:
        delta = self.delta_object.detach()
        return {
            "perturbation_linf": float(delta.abs().max()),
            "perturbation_l2": float(delta.square().sum().sqrt()),
            "perturbation_mean_abs": float(delta.abs().mean()),
            "perturbation_saturation_ratio": float((delta.abs() >= self.epsilon - 1.0e-7).float().mean()),
        }

