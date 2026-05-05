from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..support import build_support_mask


@dataclass
class PoisonResult:
    poisoned_image: np.ndarray
    perturbation: np.ndarray
    support_mask: np.ndarray
    ring_mask: np.ndarray
    losses: Dict[str, float] = field(default_factory=dict)
    extras: Dict = field(default_factory=dict)


class BasePoisonGenerator:
    def __init__(self, cfg: Dict, method_cfg: Dict, device: torch.device, surrogate):
        self.cfg = cfg
        self.method_cfg = method_cfg
        self.device = device
        self.surrogate = surrogate
        self.target_class_id = int(cfg["experiment"]["target_class_id"])
        self.num_classes = int(cfg["surrogate"]["num_classes"])
        self.eps = float(cfg["experiment"]["eps"])
        self.imgsz = int(cfg["surrogate"].get("imgsz", 640))
        self.pad_value = 0.447

        self.surrogate.eval()
        for p in self.surrogate.parameters():
            p.requires_grad_(False)

        self._current_image_shape = None

        # Mid-level feature hook state for shape entanglement.
        self._mid_feature_handle = None
        self._mid_feature_module_name: Optional[str] = None
        self._latest_mid_feature: Optional[torch.Tensor] = None

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        self._current_image_shape = image.shape
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def _to_numpy(self, image_t: torch.Tensor) -> np.ndarray:
        return image_t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()

    def _letterbox_tensor(self, image_t: torch.Tensor) -> torch.Tensor:
        if image_t.ndim != 4:
            raise ValueError(f"image_t should be [B,C,H,W], got {tuple(image_t.shape)}")

        _, _, h, w = image_t.shape
        if h == self.imgsz and w == self.imgsz:
            return image_t

        r = min(self.imgsz / float(h), self.imgsz / float(w))
        new_h = max(1, int(round(h * r)))
        new_w = max(1, int(round(w * r)))

        resized = F.interpolate(image_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return F.pad(resized, (left, right, top, bottom), mode="constant", value=float(self.pad_value))

    def _forward_raw(self, image_t: torch.Tensor) -> torch.Tensor:
        image_lb = self._letterbox_tensor(image_t)
        out = self.surrogate(image_lb)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def _target_scores(self, preds: torch.Tensor) -> torch.Tensor:
        cls_start = 4
        cls_idx = cls_start + self.target_class_id
        if cls_idx >= preds.shape[1]:
            raise RuntimeError(
                "Target class index overflow. "
                f"cls_idx={cls_idx}, pred_channels={preds.shape[1]}. "
                "Check VOC/COCO class alignment and surrogate num_classes."
            )
        return preds[:, cls_idx, :]

    def _non_target_logits(self, preds: torch.Tensor) -> torch.Tensor:
        cls_start = 4
        cls_end = cls_start + self.num_classes
        if cls_end > preds.shape[1]:
            cls_end = preds.shape[1]
        logits = preds[:, cls_start:cls_end, :]
        keep = [i for i in range(logits.shape[1]) if i != self.target_class_id]
        if not keep:
            return logits[:, :0, :]
        return logits[:, keep, :]

    @staticmethod
    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        if box1.ndim != 2 or box1.shape[1] != 4:
            raise ValueError(f"box1 shape must be [N,4], got {tuple(box1.shape)}")
        if box2.ndim != 2 or box2.shape[1] != 4:
            raise ValueError(f"box2 shape must be [M,4], got {tuple(box2.shape)}")
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)

        lt = torch.max(box1[:, None, :2], box2[None, :, :2])
        rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0))

        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp_min(1e-6)

    def _collect_target_gt_boxes_xyxy(self, annotations: List[dict], image_shape) -> torch.Tensor:
        h, w = image_shape[:2]
        out = []

        for ann in annotations:
            cls = ann.get("cls", ann.get("class_id", ann.get("category_id", -1)))
            try:
                cls = int(cls)
            except Exception:
                continue
            if cls != self.target_class_id:
                continue

            if "xyxy" in ann:
                box = ann["xyxy"]
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
            else:
                box = ann.get("bbox", ann.get("box", None))
                if box is None or len(box) != 4:
                    continue
                cx, cy, bw, bh = [float(v) for v in box]
                if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                x1 = cx - bw * 0.5
                y1 = cy - bh * 0.5
                x2 = cx + bw * 0.5
                y2 = cy + bh * 0.5

            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])

        if not out:
            return torch.zeros((0, 4), device=self.device, dtype=torch.float32)

        return torch.tensor(out, device=self.device, dtype=torch.float32)

    def _decode_pred_boxes_xyxy(self, preds: torch.Tensor, image_shape=None) -> torch.Tensor:
        if preds.ndim != 3 or preds.shape[0] != 1:
            raise ValueError(f"preds should be [1,C,N], got {tuple(preds.shape)}")

        if image_shape is None:
            image_shape = self._current_image_shape
        if image_shape is None:
            raise RuntimeError("image_shape is not available for decoding predicted boxes.")

        h, w = image_shape[:2]
        pred_xywh = preds[0, :4, :].transpose(0, 1)
        pred_xyxy = self.xywh_to_xyxy(pred_xywh)
        pred_xyxy[:, 0::2] = pred_xyxy[:, 0::2].clamp(0.0, float(w))
        pred_xyxy[:, 1::2] = pred_xyxy[:, 1::2].clamp(0.0, float(h))
        return pred_xyxy

    def _alignment_proxy(
        self,
        preds: torch.Tensor,
        gt_boxes_xyxy: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 6.0,
        image_shape=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if preds.ndim != 3 or preds.shape[0] != 1:
            raise ValueError(f"preds should be [1,C,N], got {tuple(preds.shape)}")

        target_logits = self._target_scores(preds).squeeze(0)
        target_prob = torch.sigmoid(target_logits)

        pred_boxes_xyxy = self._decode_pred_boxes_xyxy(preds, image_shape=image_shape)

        if gt_boxes_xyxy.numel() == 0:
            best_iou = torch.zeros_like(target_prob)
        else:
            iou_mat = self.box_iou_xyxy(pred_boxes_xyxy, gt_boxes_xyxy)
            best_iou = torch.max(iou_mat, dim=1).values

        align = (target_prob.clamp(min=0.0) ** float(alpha)) * (best_iou.clamp(min=0.0) ** float(beta))

        if not torch.isfinite(align).all():
            raise RuntimeError("Non-finite alignment proxy detected.")

        return align, target_prob, best_iou

    def _select_mid_feature_module(self):
        candidates = []
        for name, module in self.surrogate.named_modules():
            low_name = name.lower()
            cls_name = module.__class__.__name__.lower()

            if any(k in low_name for k in ["detect", "head", "dfl"]):
                continue
            if any(k in cls_name for k in ["detect", "head", "dfl"]):
                continue

            if isinstance(module, torch.nn.Conv2d) or ("c2f" in cls_name):
                candidates.append((name, module))

        if not candidates:
            return None

        idx = int(0.6 * (len(candidates) - 1))
        return candidates[idx]

    def _register_mid_feature_hook(self) -> str:
        if self._mid_feature_handle is not None and self._mid_feature_module_name is not None:
            return self._mid_feature_module_name

        picked = self._select_mid_feature_module()
        if picked is None:
            raise RuntimeError("Unable to find a stable mid-level feature module for hook.")

        module_name, module = picked

        def _hook(_module, _inputs, output):
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0] if len(out) > 0 else None
            if torch.is_tensor(out):
                self._latest_mid_feature = out

        self._latest_mid_feature = None
        self._mid_feature_handle = module.register_forward_hook(_hook)
        self._mid_feature_module_name = module_name
        print(f"[BasePoisonGenerator] mid feature hook module: {module_name}")
        return module_name

    def _clear_mid_feature_hook(self) -> None:
        if self._mid_feature_handle is not None:
            self._mid_feature_handle.remove()
        self._mid_feature_handle = None
        self._mid_feature_module_name = None
        self._latest_mid_feature = None

    def _clear_mid_feature_cache(self) -> None:
        self._latest_mid_feature = None

    def _get_mid_feature(self, require: bool = True, detach: bool = False) -> Optional[torch.Tensor]:
        feat = self._latest_mid_feature
        if feat is None:
            if require:
                raise RuntimeError(
                    "Mid feature is empty. Ensure _register_mid_feature_hook() is called and forward has executed."
                )
            return None
        return feat.detach() if detach else feat

    def _build_support(
        self,
        image_shape,
        annotations,
        support_type: str,
        ring_width: int = 4,
        image_path: str = None,
    ):
        mask_path = None
        instance_mask_dir = str(self.cfg.get("data", {}).get("instance_mask_dir", "") or "").strip()
        if instance_mask_dir and os.path.isdir(instance_mask_dir) and image_path:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(instance_mask_dir, stem + ".png")

        support = build_support_mask(
            image_shape=image_shape,
            annotations=annotations,
            target_class_id=self.target_class_id,
            support_type=support_type,
            ring_width=ring_width,
            mask_path=mask_path,
        )
        ring = build_support_mask(
            image_shape=image_shape,
            annotations=annotations,
            target_class_id=self.target_class_id,
            support_type="mask_ring",
            ring_width=ring_width,
            mask_path=mask_path,
        )
        return support, ring

    def generate(
        self,
        image: np.ndarray,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ) -> PoisonResult:
        raise NotImplementedError

