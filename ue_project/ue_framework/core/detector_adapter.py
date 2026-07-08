from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assignment_parser import AssignmentResult, infer_fpn_level_ids


class DetectorAdapter:
    """Small detector-facing adapter used by the new trajectory methods.

    The adapter intentionally avoids importing ALCE/RLCP modules. It exposes a
    differentiable proxy loss that is class-filtered at assigned positive units.
    Full Ultralytics loss plumbing can be added behind the same interface later.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        protected_class_id: int,
        assignment_topk: int = 1,
        eps: float = 1.0e-8,
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.protected_class_id = int(protected_class_id)
        self.assignment_topk = int(assignment_topk)
        self.eps = float(eps)
        self._last_image_shape: Optional[Tuple[int, int]] = None
        self._fpn_features: Dict[str, torch.Tensor] = {}
        self._feature_handles: List[torch.utils.hooks.RemovableHandle] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        self._last_image_shape = (int(images.shape[-2]), int(images.shape[-1]))
        return self.model(images)

    def forward_with_parameters(self, images: torch.Tensor, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        from ue_framework.methods.learning_trajectory.virtual_update import functional_forward

        if images.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        self._last_image_shape = (int(images.shape[-2]), int(images.shape[-1]))
        return functional_forward(self.model, parameters, images)

    def get_named_trainable_parameters(self, scope: str) -> List[Tuple[str, nn.Parameter]]:
        scope = str(scope)
        if scope not in {"head", "neck_and_head", "full"}:
            raise ValueError(f"Unsupported parameter scope: {scope}")

        named = list(self.model.named_parameters())
        if scope == "full":
            return [(n, p) for n, p in named if p.requires_grad]

        if scope == "head":
            picked = [(n, p) for n, p in named if p.requires_grad and self._looks_like_head(n)]
            if picked:
                return picked
            return [(n, p) for n, p in named if p.requires_grad][-2:]

        picked = [(n, p) for n, p in named if p.requires_grad and self._looks_like_neck_or_head(n)]
        if picked:
            return picked
        tail = max(1, len(named) // 2)
        return [(n, p) for n, p in named[-tail:] if p.requires_grad]

    def get_task_aligned_assignments(self, predictions: torch.Tensor, batch: Dict) -> AssignmentResult:
        boxes_xyxy, logits = self.decode_prediction_tensors(predictions)
        gt_labels_by_batch, gt_boxes_by_batch = self.collect_gt(batch, boxes_xyxy)
        bsz, num_units, _ = logits.shape
        device = logits.device

        fg_mask = torch.zeros((bsz, num_units), dtype=torch.bool, device=device)
        target_gt_idx = torch.full((bsz, num_units), -1, dtype=torch.long, device=device)
        target_labels = torch.full((bsz, num_units), -1, dtype=torch.long, device=device)
        target_scores = torch.zeros((bsz, num_units, self.num_classes), dtype=logits.dtype, device=device)
        assignment_counts = torch.zeros((bsz, num_units), dtype=torch.long, device=device)

        for b in range(bsz):
            labels = gt_labels_by_batch[b]
            gt_boxes = gt_boxes_by_batch[b]
            if labels.numel() == 0:
                continue
            iou = self.box_iou_xyxy(boxes_xyxy[b], gt_boxes)
            centers = self._center_distance_score(boxes_xyxy[b], gt_boxes)
            score = torch.where(iou > 0, iou, centers)
            for gt_idx in range(labels.numel()):
                k = max(1, min(self.assignment_topk, num_units))
                idx = torch.topk(score[:, gt_idx], k=k, largest=True).indices
                assignment_counts[b, idx] += 1
                overwrite = target_gt_idx[b, idx] < 0
                write_idx = idx[overwrite]
                if write_idx.numel() == 0:
                    continue
                cls_id = labels[gt_idx].long().clamp(min=0, max=self.num_classes - 1)
                fg_mask[b, write_idx] = True
                target_gt_idx[b, write_idx] = gt_idx
                target_labels[b, write_idx] = cls_id
                target_scores[b, write_idx, cls_id] = score[write_idx, gt_idx].detach().clamp(min=0.0, max=1.0)

        level_ids = infer_fpn_level_ids(num_units, device).unsqueeze(0).expand(bsz, -1)
        return AssignmentResult(
            fg_mask=fg_mask,
            target_gt_idx=target_gt_idx,
            target_labels=target_labels,
            target_scores=target_scores,
            assignment_counts=assignment_counts,
            level_ids=level_ids,
        )

    def compute_detection_loss(
        self,
        predictions: torch.Tensor,
        batch: Dict,
        class_filter: Optional[Iterable[int]] = None,
        return_components: bool = False,
        include_background_negatives: bool = False,
    ):
        assignment = self.get_task_aligned_assignments(predictions, batch)
        mask = assignment.fg_mask
        if class_filter is not None:
            class_ids = torch.tensor([int(c) for c in class_filter], device=mask.device, dtype=torch.long)
            mask = mask & torch.isin(assignment.target_labels.long(), class_ids)
        components = self.compute_masked_detection_loss(
            predictions=predictions,
            batch=batch,
            assignment=assignment,
            unit_mask=mask,
            include_background_negatives=include_background_negatives,
            background_class_filter=class_filter,
        )
        return components if return_components else components["total_loss"]

    def compute_masked_detection_loss(
        self,
        predictions: torch.Tensor,
        batch: Dict,
        assignment: AssignmentResult,
        unit_mask: torch.Tensor,
        include_background_negatives: bool = False,
        background_class_filter: Optional[Iterable[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        boxes_xyxy, logits = self.decode_prediction_tensors(predictions)
        gt_labels_by_batch, gt_boxes_by_batch = self.collect_gt(batch, boxes_xyxy)
        zero = logits.sum() * 0.0
        cls_loss = zero
        box_loss = zero
        dfl_loss = zero

        if unit_mask.any():
            pos = torch.nonzero(unit_mask, as_tuple=False)
            b_idx = pos[:, 0]
            u_idx = pos[:, 1]
            labels = assignment.target_labels[b_idx, u_idx].long().clamp(min=0, max=self.num_classes - 1)
            cls_logits = logits[b_idx, u_idx, :].gather(1, labels.unsqueeze(1)).squeeze(1)
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, torch.ones_like(cls_logits), reduction="mean")

            gt_boxes = []
            for b, gt_idx in zip(b_idx.detach().cpu().tolist(), assignment.target_gt_idx[b_idx, u_idx].detach().cpu().tolist()):
                if gt_idx < 0:
                    gt_boxes.append(torch.zeros(4, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype))
                else:
                    gt_boxes.append(gt_boxes_by_batch[int(b)][int(gt_idx)])
            target_boxes = torch.stack(gt_boxes, dim=0)
            box_loss = F.smooth_l1_loss(boxes_xyxy[b_idx, u_idx, :], target_boxes, reduction="mean")

        if include_background_negatives:
            bg_mask = ~assignment.fg_mask
            if bg_mask.any():
                bg_logits = logits[bg_mask]
                if background_class_filter is not None:
                    class_ids = torch.tensor(
                        [int(c) for c in background_class_filter],
                        device=bg_logits.device,
                        dtype=torch.long,
                    )
                    bg_logits = bg_logits[:, class_ids]
                bg_loss = F.binary_cross_entropy_with_logits(bg_logits, torch.zeros_like(bg_logits), reduction="mean")
                cls_loss = cls_loss + bg_loss

        total = cls_loss + box_loss + dfl_loss
        return {
            "total_loss": total,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
        }

    def get_fpn_features(self) -> Dict[str, torch.Tensor]:
        return dict(self._fpn_features)

    def register_feature_hooks(self, module_names: Sequence[str]) -> None:
        self.clear_feature_hooks()
        modules = dict(self.model.named_modules())
        for name in module_names:
            if name not in modules:
                raise KeyError(f"Module not found for feature hook: {name}")

            def _hook(_module, _inputs, output, key=name):
                if torch.is_tensor(output):
                    self._fpn_features[key] = output

            self._feature_handles.append(modules[name].register_forward_hook(_hook))

    def clear_feature_hooks(self) -> None:
        for handle in self._feature_handles:
            handle.remove()
        self._feature_handles.clear()
        self._fpn_features.clear()

    @contextmanager
    def temporarily_requires_grad(self, names: Iterable[str]) -> Iterator[None]:
        name_set = set(names)
        old = {name: param.requires_grad for name, param in self.model.named_parameters() if name in name_set}
        try:
            for name, param in self.model.named_parameters():
                if name in old:
                    param.requires_grad_(True)
            yield
        finally:
            for name, param in self.model.named_parameters():
                if name in old:
                    param.requires_grad_(old[name])

    def decode_prediction_tensors(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self._unwrap_predictions(predictions)
        if pred.ndim != 3:
            raise ValueError(f"predictions must be [B,C,N] or [B,N,C], got {tuple(pred.shape)}")
        channels = 4 + self.num_classes
        if pred.shape[1] >= channels and pred.shape[2] != channels:
            raw_boxes = pred[:, :4, :].transpose(1, 2).contiguous()
            logits = pred[:, 4 : 4 + self.num_classes, :].transpose(1, 2).contiguous()
        elif pred.shape[2] >= channels:
            raw_boxes = pred[:, :, :4].contiguous()
            logits = pred[:, :, 4 : 4 + self.num_classes].contiguous()
        else:
            raise ValueError(f"Prediction tensor does not contain 4+{self.num_classes} channels: {tuple(pred.shape)}")
        return self.xywh_to_xyxy(raw_boxes), logits

    def collect_gt(self, batch: Dict, pred_boxes_xyxy: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = pred_boxes_xyxy.device
        dtype = pred_boxes_xyxy.dtype
        bsz = int(batch.get("batch_size", pred_boxes_xyxy.shape[0]))
        cls = batch.get("cls")
        bboxes = batch.get("bboxes")
        if cls is None or bboxes is None:
            return (
                [torch.empty(0, device=device, dtype=torch.long) for _ in range(bsz)],
                [torch.empty(0, 4, device=device, dtype=dtype) for _ in range(bsz)],
            )

        cls_t = torch.as_tensor(cls, device=device).reshape(-1).long()
        boxes_t = torch.as_tensor(bboxes, device=device, dtype=dtype).reshape(-1, 4)
        batch_idx = batch.get("batch_idx")
        if batch_idx is None:
            batch_idx_t = torch.zeros(cls_t.numel(), device=device, dtype=torch.long)
        else:
            batch_idx_t = torch.as_tensor(batch_idx, device=device).reshape(-1).long()

        gt_labels: List[torch.Tensor] = []
        gt_boxes: List[torch.Tensor] = []
        scale = self._gt_scale_for(pred_boxes_xyxy, boxes_t)
        for b in range(bsz):
            keep = batch_idx_t == b
            labels_b = cls_t[keep]
            boxes_b = boxes_t[keep]
            if boxes_b.numel() > 0 and scale is not None:
                sx, sy = scale
                boxes_b = boxes_b.clone()
                boxes_b[:, 0] *= sx
                boxes_b[:, 2] *= sx
                boxes_b[:, 1] *= sy
                boxes_b[:, 3] *= sy
            gt_labels.append(labels_b)
            gt_boxes.append(self.xywh_to_xyxy(boxes_b) if boxes_b.numel() else boxes_b.reshape(0, 4))
        return gt_labels, gt_boxes

    @staticmethod
    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(*boxes.shape[:-1], 4)
        x, y, w, h = boxes.unbind(dim=-1)
        return torch.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], dim=-1)

    @staticmethod
    def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)
        lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
        rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0.0) * (box1[:, 3] - box1[:, 1]).clamp(min=0.0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0.0) * (box2[:, 3] - box2[:, 1]).clamp(min=0.0))
        return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(eps)

    @staticmethod
    def _unwrap_predictions(predictions):
        if isinstance(predictions, (list, tuple)):
            for value in predictions:
                if torch.is_tensor(value) and value.ndim == 3:
                    return value
            raise ValueError("No [B,C,N] or [B,N,C] tensor found in predictions.")
        return predictions

    @staticmethod
    def _center_distance_score(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        box_center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        gt_center = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
        dist = torch.cdist(box_center, gt_center, p=2)
        return 1.0 / (1.0 + dist)

    @staticmethod
    def _looks_like_head(name: str) -> bool:
        low = name.lower()
        return "detect" in low or "head" in low or low.startswith("model.22")

    @staticmethod
    def _looks_like_neck_or_head(name: str) -> bool:
        low = name.lower()
        return DetectorAdapter._looks_like_head(low) or "neck" in low or any(tok in low for tok in ["model.15", "model.16", "model.17", "model.18", "model.19", "model.20", "model.21"])

    def _gt_scale_for(self, pred_boxes_xyxy: torch.Tensor, boxes_xywh: torch.Tensor) -> Optional[Tuple[float, float]]:
        if boxes_xywh.numel() == 0 or self._last_image_shape is None:
            return None
        if float(boxes_xywh.detach().abs().max().item()) > 1.5:
            return None
        if float(pred_boxes_xyxy.detach().abs().max().item()) <= 2.0:
            return None
        h, w = self._last_image_shape
        return float(w), float(h)
