from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CarrierConfig:
    target_class_id: int = 14
    eps: float = 16.0 / 255.0
    non_target_dilation: int = 4
    min_valid_fraction: float = 0.01
    interpolation: str = "bilinear"
    soft_mask: bool = True
    soft_edge_pixels: float = 2.0
    box_jitter: float = 0.0


@dataclass
class CarrierResult:
    poisoned: torch.Tensor
    perturbation: torch.Tensor
    target_support: torch.Tensor
    valid_support: torch.Tensor
    non_target_mask: torch.Tensor
    metrics: dict[str, Any]
    instance_metrics: list[dict[str, Any]]


def _clipped_box(annotation: dict[str, Any], height: int, width: int) -> tuple[int, int, int, int]:
    xc, yc, box_width, box_height = [float(value) for value in annotation["bbox"]]
    x1 = max(0, min(width, int(torch.floor(torch.tensor((xc - box_width / 2.0) * width)).item())))
    y1 = max(0, min(height, int(torch.floor(torch.tensor((yc - box_height / 2.0) * height)).item())))
    x2 = max(0, min(width, int(torch.ceil(torch.tensor((xc + box_width / 2.0) * width)).item())))
    y2 = max(0, min(height, int(torch.ceil(torch.tensor((yc + box_height / 2.0) * height)).item())))
    return x1, y1, x2, y2


def _soft_box_mask(
    height: int,
    width: int,
    edge_pixels: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if edge_pixels <= 0:
        return reference.new_ones((1, height, width))
    y = torch.arange(height, device=reference.device, dtype=reference.dtype)
    x = torch.arange(width, device=reference.device, dtype=reference.dtype)
    y_distance = torch.minimum(y + 1.0, height - y)
    x_distance = torch.minimum(x + 1.0, width - x)
    y_weight = (y_distance / float(edge_pixels)).clamp(0.0, 1.0)
    x_weight = (x_distance / float(edge_pixels)).clamp(0.0, 1.0)
    return (y_weight[:, None] * x_weight[None, :]).unsqueeze(0)


def _resize_delta(delta_obj: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    allowed = {"nearest", "bilinear", "bicubic"}
    if mode not in allowed:
        raise ValueError(f"unsupported interpolation: {mode}; expected one of {sorted(allowed)}")
    options: dict[str, Any] = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        options["align_corners"] = False
    return F.interpolate(delta_obj.unsqueeze(0), **options)[0]


def apply_object_aligned_carrier(
    image: torch.Tensor,
    annotations: Iterable[dict[str, Any]],
    delta_obj: torch.Tensor,
    config: CarrierConfig,
) -> CarrierResult:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be [3,H,W]")
    if delta_obj.ndim != 3 or delta_obj.shape[0] != 3:
        raise ValueError("delta_obj must be [3,h0,w0]")
    if image.device != delta_obj.device:
        raise ValueError("image and delta_obj must be on the same device")
    if config.box_jitter != 0.0:
        raise NotImplementedError("box_jitter is configurable but disabled in the local smoke protocol")
    if not 0.0 <= config.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0,1]")

    height, width = int(image.shape[1]), int(image.shape[2])
    annotations = list(annotations)
    target_boxes: list[tuple[int, int, int, int]] = []
    non_target = delta_obj.new_zeros((1, height, width))
    for annotation in annotations:
        box = _clipped_box(annotation, height, width)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        if int(annotation["cls"]) == int(config.target_class_id):
            target_boxes.append(box)
        else:
            non_target[:, y1:y2, x1:x2] = 1.0

    undilated_non_target = non_target.clone()
    if config.non_target_dilation > 0:
        radius = int(config.non_target_dilation)
        non_target = F.max_pool2d(
            non_target.unsqueeze(0), kernel_size=2 * radius + 1, stride=1, padding=radius
        )[0]

    canvas = delta_obj.new_zeros((3, height, width))
    target_support = delta_obj.new_zeros((1, height, width))
    valid_support = delta_obj.new_zeros((1, height, width))
    raw_overlap = delta_obj.new_zeros((1, height, width))
    instance_rows: list[dict[str, Any]] = []
    applied_instances = 0

    for instance_index, (x1, y1, x2, y2) in enumerate(target_boxes):
        box_height, box_width = y2 - y1, x2 - x1
        blend = _soft_box_mask(
            box_height,
            box_width,
            config.soft_edge_pixels if config.soft_mask else 0.0,
            delta_obj,
        )
        overlap = blend * non_target[:, y1:y2, x1:x2]
        valid = blend * (1.0 - non_target[:, y1:y2, x1:x2])
        support_pixels = float(blend.detach().sum().item())
        valid_pixels = float(valid.detach().sum().item())
        overlap_pixels = float(overlap.detach().sum().item())
        valid_fraction = valid_pixels / max(support_pixels, 1e-12)
        skipped_reason = ""
        if support_pixels <= 0:
            skipped_reason = "empty_target_support"
        elif valid_fraction < config.min_valid_fraction:
            skipped_reason = "valid_fraction_below_threshold"

        target_support[:, y1:y2, x1:x2] = torch.maximum(
            target_support[:, y1:y2, x1:x2], blend
        )
        raw_overlap[:, y1:y2, x1:x2] = torch.maximum(
            raw_overlap[:, y1:y2, x1:x2], overlap
        )
        if not skipped_reason:
            pattern = _resize_delta(delta_obj, (box_height, box_width), config.interpolation)
            canvas[:, y1:y2, x1:x2] = canvas[:, y1:y2, x1:x2] + pattern * valid
            valid_support[:, y1:y2, x1:x2] = torch.maximum(
                valid_support[:, y1:y2, x1:x2], valid
            )
            applied_instances += 1
        instance_rows.append(
            {
                "instance_index": instance_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": box_width,
                "box_height": box_height,
                "support_pixels": support_pixels,
                "valid_pixels": valid_pixels,
                "overlap_pixels": overlap_pixels,
                "valid_fraction": valid_fraction,
                "skipped": int(bool(skipped_reason)),
                "invalid_reason": skipped_reason,
            }
        )

    perturbation = canvas.clamp(min=-float(config.eps), max=float(config.eps))
    perturbation = perturbation * (1.0 - non_target)
    poisoned = (image + perturbation).clamp(0.0, 1.0)
    changed = perturbation.detach().abs().amax(dim=0, keepdim=True) > 0
    metrics = {
        "image_height": height,
        "image_width": width,
        "target_instances": len(target_boxes),
        "applied_instances": applied_instances,
        "skipped_instances": len(target_boxes) - applied_instances,
        "actual_support_area": float((target_support.detach() > 0).float().mean().item()),
        "support_weight_mass": float(target_support.detach().mean().item()),
        "perturbed_area": float(changed.float().mean().item()),
        "valid_support_area": float((valid_support.detach() > 0).float().mean().item()),
        "valid_support_weight_mass": float(valid_support.detach().mean().item()),
        "non_target_area": float(undilated_non_target.detach().mean().item()),
        "dilated_non_target_area": float(non_target.detach().mean().item()),
        "non_target_overlap_area": float(raw_overlap.detach().mean().item()),
        "non_target_overlap_ratio": float(
            raw_overlap.detach().sum().item() / max(target_support.detach().sum().item(), 1e-12)
        ),
        "direct_non_target_perturbation_max": float(
            (perturbation.detach() * non_target).abs().max().item()
        ),
        "max_abs_perturbation": float(perturbation.detach().abs().max().item()),
        "mean_abs_perturbation": float(perturbation.detach().abs().mean().item()),
        "finite": bool(torch.isfinite(poisoned).all() and torch.isfinite(perturbation).all()),
        "interpolation": config.interpolation,
        "soft_mask": bool(config.soft_mask),
    }
    return CarrierResult(
        poisoned=poisoned,
        perturbation=perturbation,
        target_support=target_support,
        valid_support=valid_support,
        non_target_mask=non_target,
        metrics=metrics,
        instance_metrics=instance_rows,
    )
