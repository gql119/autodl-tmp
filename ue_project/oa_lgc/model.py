from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _pixel_box(annotation: dict, height: int, width: int) -> tuple[int, int, int, int]:
    xc, yc, box_width, box_height = [float(value) for value in annotation["bbox"]]
    x1 = max(0, min(width - 1, int((xc - box_width / 2.0) * width)))
    y1 = max(0, min(height - 1, int((yc - box_height / 2.0) * height)))
    x2 = max(x1 + 1, min(width, int((xc + box_width / 2.0) * width + 0.999999)))
    y2 = max(y1 + 1, min(height, int((yc + box_height / 2.0) * height + 0.999999)))
    return x1, y1, x2, y2


class ObjectCropDetector(torch.nn.Module):
    """Small detector-shaped proxy used only for local engineering validation."""

    def __init__(self, num_classes: int = 20, pool_size: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.pool_size = int(pool_size)
        self.feature_proj = nn.Linear(3 * self.pool_size * self.pool_size, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, self.num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

    def forward(self, images: torch.Tensor, annotations: Iterable[Iterable[dict]]) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [B,3,H,W]")
        crops = []
        labels = []
        boxes = []
        source_indices = []
        height, width = int(images.shape[2]), int(images.shape[3])
        for batch_index, image_annotations in enumerate(annotations):
            for annotation in image_annotations:
                x1, y1, x2, y2 = _pixel_box(annotation, height, width)
                crop = images[batch_index : batch_index + 1, :, y1:y2, x1:x2]
                crops.append(F.adaptive_avg_pool2d(crop, (self.pool_size, self.pool_size))[0])
                labels.append(int(annotation["cls"]))
                boxes.append([float(value) for value in annotation["bbox"]])
                source_indices.append(batch_index)
        if not crops:
            empty_features = images.new_zeros((0, self.feature_proj.out_features))
            return {
                "logits": images.new_zeros((0, self.num_classes)),
                "boxes": images.new_zeros((0, 4)),
                "features": empty_features,
                "labels": torch.zeros((0,), dtype=torch.long, device=images.device),
                "target_boxes": images.new_zeros((0, 4)),
                "source_indices": torch.zeros((0,), dtype=torch.long, device=images.device),
            }
        flattened = torch.stack(crops).flatten(1)
        features = torch.tanh(self.feature_proj(flattened))
        return {
            "logits": self.cls_head(features),
            "boxes": torch.sigmoid(self.box_head(features)),
            "features": features,
            "labels": torch.tensor(labels, dtype=torch.long, device=images.device),
            "target_boxes": torch.tensor(boxes, dtype=images.dtype, device=images.device),
            "source_indices": torch.tensor(source_indices, dtype=torch.long, device=images.device),
        }


def detection_loss(outputs: dict[str, torch.Tensor], box_weight: float = 0.25) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, labels = outputs["logits"], outputs["labels"]
    if labels.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, {"classification": zero, "box": zero}
    classification = F.cross_entropy(logits, labels)
    box = F.smooth_l1_loss(outputs["boxes"], outputs["target_boxes"])
    return classification + float(box_weight) * box, {"classification": classification, "box": box}


def class_loss(outputs: dict[str, torch.Tensor], class_id: int) -> tuple[torch.Tensor, int]:
    mask = outputs["labels"] == int(class_id)
    count = int(mask.sum().item())
    if count == 0:
        return outputs["logits"].sum() * 0.0, 0
    return F.cross_entropy(outputs["logits"][mask], outputs["labels"][mask]), count

