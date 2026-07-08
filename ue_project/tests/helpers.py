from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ue_framework.core import DetectorAdapter
from ue_framework.methods.learning_trajectory import LearningTrajectoryMethod


class ToyDetector(nn.Module):
    def __init__(self, num_classes: int = 20, num_units: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_units = int(num_units)
        self.head = nn.Linear(3, self.num_units * self.num_classes)
        anchors = torch.tensor(
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
                [0.25, 0.75, 0.20, 0.20],
                [0.75, 0.25, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("anchors_xywh", anchors[: self.num_units])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        logits = self.head(pooled).reshape(images.shape[0], self.num_units, self.num_classes)
        boxes = self.anchors_xywh.to(images.device, images.dtype).unsqueeze(0).expand(images.shape[0], -1, -1)
        return torch.cat([boxes, logits], dim=-1)


def make_toy_components(seed: int = 0) -> Tuple[ToyDetector, DetectorAdapter, LearningTrajectoryMethod, Dict]:
    torch.manual_seed(seed)
    model = ToyDetector()
    adapter = DetectorAdapter(model, num_classes=20, protected_class_id=14, assignment_topk=1)
    config = {
        "protected_class_id": 14,
        "authorized_class_ids": "auto",
        "num_classes": 20,
        "trajectory": {
            "parameter_scope": "head",
            "normalize_per_parameter": True,
            "lambda_protected": 1.0,
            "lambda_authorized": 1.0,
            "eps": 1.0e-8,
        },
        "class_routing": {
            "exclude_ambiguous": True,
            "include_background_negatives": False,
        },
        "virtual_update": {
            "parameter_scope": "head",
            "steps": 1,
            "lr": 0.2,
        },
        "meta": {
            "use_p1_regularizer": False,
            "lambda_meta": 1.0,
            "lambda_p1": 0.2,
            "lambda_protected_query": 1.0,
            "enable_clean_counterfactual": True,
        },
    }
    return model, adapter, LearningTrajectoryMethod(adapter, config), config


def make_images(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(2, 3, 8, 8)


def make_batch() -> Dict[str, torch.Tensor]:
    cls = torch.tensor([14, 1, 14, 1], dtype=torch.long)
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.20, 0.20],
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return {"cls": cls, "bboxes": bboxes, "batch_idx": batch_idx, "batch_size": 2}
