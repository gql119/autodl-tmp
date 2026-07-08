from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AssignmentResult:
    fg_mask: torch.Tensor
    target_gt_idx: torch.Tensor
    target_labels: torch.Tensor
    target_scores: torch.Tensor
    assignment_counts: torch.Tensor
    level_ids: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.fg_mask.ndim != 2:
            raise ValueError(f"fg_mask must be [B,N], got {tuple(self.fg_mask.shape)}")
        shape = self.fg_mask.shape
        for name in ["target_gt_idx", "target_labels", "assignment_counts"]:
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape must match fg_mask {tuple(shape)}, got {tuple(value.shape)}")
        if self.target_scores.ndim != 3 or self.target_scores.shape[:2] != shape:
            raise ValueError(
                "target_scores must be [B,N,C] with B,N matching fg_mask, "
                f"got {tuple(self.target_scores.shape)}"
            )
        if self.level_ids is not None and self.level_ids.shape != shape:
            raise ValueError(f"level_ids shape must match fg_mask {tuple(shape)}, got {tuple(self.level_ids.shape)}")


def infer_fpn_level_ids(num_units: int, device: torch.device) -> torch.Tensor:
    """Infer P3/P4/P5 ids for a YOLO-style flattened prediction tensor."""
    if num_units <= 0:
        raise ValueError(f"num_units must be positive, got {num_units}")

    # Common YOLOv8 640 layout is 80^2 + 40^2 + 20^2 = 8400.
    # For other square inputs, solve s^2 + (s/2)^2 + (s/4)^2 ~= num_units.
    base = int(round((num_units / 1.3125) ** 0.5))
    p3 = base * base
    p4 = max(1, (base // 2) * (base // 2))
    p5 = max(1, num_units - p3 - p4)
    if p3 + p4 >= num_units:
        p3 = int(num_units * 0.76)
        p4 = int(num_units * 0.19)
        p5 = num_units - p3 - p4

    level = torch.empty(num_units, device=device, dtype=torch.long)
    level[:p3] = 3
    level[p3 : p3 + p4] = 4
    level[p3 + p4 :] = 5
    return level
