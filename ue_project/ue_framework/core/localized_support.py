from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LocalizedSupportOutput:
    valid_support_mask: torch.Tensor
    protected_support_mask: torch.Tensor
    authorized_core_mask: torch.Tensor
    ambiguous_mask: torch.Tensor
    statistics: Dict[str, float]


class LocalizedSupportBuilder:
    def __init__(
        self,
        protected_class_id: int = 14,
        authorized_class_ids: Optional[Iterable[int]] = None,
        num_classes: int = 20,
        dilation_pixels: int = 0,
        expansion_ratio: float = 0.0,
        exclude_authorized_core: bool = True,
        authorized_core_scale: float = 1.0,
        exclude_ambiguous: bool = True,
        ambiguous_iou_threshold: float = 0.5,
    ) -> None:
        self.protected_class_id = int(protected_class_id)
        if authorized_class_ids is None or authorized_class_ids == "auto":
            self.authorized_class_ids = [i for i in range(int(num_classes)) if i != self.protected_class_id]
        else:
            self.authorized_class_ids = [int(v) for v in authorized_class_ids]
        self.dilation_pixels = int(dilation_pixels)
        self.expansion_ratio = float(expansion_ratio)
        self.exclude_authorized_core = bool(exclude_authorized_core)
        self.authorized_core_scale = float(authorized_core_scale)
        self.exclude_ambiguous = bool(exclude_ambiguous)
        self.ambiguous_iou_threshold = float(ambiguous_iou_threshold)

    def build(self, images: torch.Tensor, targets: Dict, assignments: object | None = None) -> LocalizedSupportOutput:
        if images.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        batch_size, _channels, height, width = images.shape
        device = images.device
        dtype = images.dtype
        protected = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        authorized = torch.zeros_like(protected)
        ambiguous = torch.zeros_like(protected)

        cls = torch.as_tensor(targets.get("cls", []), device=device).reshape(-1).long()
        bboxes = torch.as_tensor(targets.get("bboxes", []), device=device, dtype=dtype).reshape(-1, 4)
        batch_idx = torch.as_tensor(
            targets.get("batch_idx", torch.zeros(cls.numel(), device=device)),
            device=device,
        ).reshape(-1).long()
        if cls.numel() == 0:
            return self._output(protected, authorized, ambiguous)

        boxes_xyxy = self._xywh_to_xyxy_pixels(bboxes, width, height)
        authorized_ids = torch.tensor(self.authorized_class_ids, device=device, dtype=cls.dtype)
        for b in range(batch_size):
            in_batch = batch_idx == b
            p_keep = in_batch & (cls == self.protected_class_id)
            a_keep = in_batch & torch.isin(cls, authorized_ids)
            p_boxes = boxes_xyxy[p_keep]
            a_boxes = boxes_xyxy[a_keep]

            for box in self._expand_boxes(p_boxes, width, height, self.expansion_ratio):
                self._fill_box(protected[b, 0], box)
            for box in self._expand_boxes(a_boxes, width, height, self.authorized_core_scale - 1.0):
                self._fill_box(authorized[b, 0], box)

            if p_boxes.numel() and a_boxes.numel():
                iou = self._box_iou_xyxy(p_boxes, a_boxes)
                pairs = torch.nonzero(iou >= self.ambiguous_iou_threshold, as_tuple=False)
                for p_i, a_i in pairs.detach().cpu().tolist():
                    inter = self._intersection_box(p_boxes[p_i], a_boxes[a_i])
                    if inter is not None:
                        self._fill_box(ambiguous[b, 0], inter)

        if self.dilation_pixels > 0:
            protected = self._dilate_binary(protected, self.dilation_pixels)

        valid = protected.clone()
        if self.exclude_authorized_core:
            valid = valid * (1.0 - authorized)
        if self.exclude_ambiguous:
            valid = valid * (1.0 - ambiguous)
        valid = (valid > 0.5).to(dtype)
        return self._output(valid, authorized, ambiguous, protected_support_mask=protected)

    @staticmethod
    def apply_support(delta: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        if support_mask.ndim == 3:
            support_mask = support_mask.unsqueeze(1)
        return delta * support_mask.to(device=delta.device, dtype=delta.dtype)

    @staticmethod
    def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        resized = F.interpolate(mask.float(), size=size, mode="nearest")
        return (resized > 0.5).to(mask.dtype)

    def _output(
        self,
        valid: torch.Tensor,
        authorized: torch.Tensor,
        ambiguous: torch.Tensor,
        protected_support_mask: Optional[torch.Tensor] = None,
    ) -> LocalizedSupportOutput:
        protected = valid if protected_support_mask is None else (protected_support_mask > 0.5).to(valid.dtype)
        total = float(valid.numel())
        stats = {
            "protected_support_ratio": float(protected.mean().detach().item()) if total else 0.0,
            "authorized_core_ratio": float(authorized.mean().detach().item()) if total else 0.0,
            "ambiguous_ratio": float(ambiguous.mean().detach().item()) if total else 0.0,
            "valid_support_ratio": float(valid.mean().detach().item()) if total else 0.0,
            "support_area_ratio": float(valid.mean().detach().item()) if total else 0.0,
            "support_source": 1.0,
        }
        return LocalizedSupportOutput(
            valid_support_mask=valid,
            protected_support_mask=protected,
            authorized_core_mask=(authorized > 0.5).to(valid.dtype),
            ambiguous_mask=(ambiguous > 0.5).to(valid.dtype),
            statistics=stats,
        )

    @staticmethod
    def _xywh_to_xyxy_pixels(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        boxes = boxes.clone()
        if float(boxes.detach().abs().max().item()) <= 1.5:
            scale = boxes.new_tensor([width, height, width, height])
            boxes = boxes * scale
        x, y, w, h = boxes.unbind(-1)
        out = torch.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], dim=-1)
        out[:, 0::2] = out[:, 0::2].clamp(0, width)
        out[:, 1::2] = out[:, 1::2].clamp(0, height)
        return out

    @staticmethod
    def _expand_boxes(boxes: torch.Tensor, width: int, height: int, ratio: float) -> torch.Tensor:
        if boxes.numel() == 0 or ratio == 0.0:
            return boxes
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
        bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0) * (1.0 + ratio)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0) * (1.0 + ratio)
        out = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
        out[:, 0::2] = out[:, 0::2].clamp(0, width)
        out[:, 1::2] = out[:, 1::2].clamp(0, height)
        return out

    @staticmethod
    def _fill_box(mask: torch.Tensor, box: torch.Tensor) -> None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.detach().cpu().tolist()]
        x1 = max(0, min(mask.shape[1], x1))
        x2 = max(0, min(mask.shape[1], x2))
        y1 = max(0, min(mask.shape[0], y1))
        y2 = max(0, min(mask.shape[0], y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    @staticmethod
    def _dilate_binary(mask: torch.Tensor, pixels: int) -> torch.Tensor:
        kernel = 2 * int(pixels) + 1
        return (F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(pixels)) > 0.5).to(mask.dtype)

    @staticmethod
    def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        lt = torch.maximum(a[:, None, :2], b[None, :, :2])
        rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
        return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(eps)

    @staticmethod
    def _intersection_box(a: torch.Tensor, b: torch.Tensor) -> Optional[torch.Tensor]:
        lt = torch.maximum(a[:2], b[:2])
        rb = torch.minimum(a[2:], b[2:])
        if bool(torch.any(rb <= lt)):
            return None
        return torch.cat([lt, rb], dim=0)
