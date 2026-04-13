from typing import Dict, Optional, Tuple

import torch

try:
    from ultralytics.utils.loss import v8DetectionLoss as _V8Base
except Exception:  # pragma: no cover
    _V8Base = object


class HijackedV8Loss(_V8Base):
    """
    Side-channel helper for assignment-aware proxy training.

    It keeps compatibility with upstream loss signature when available, while always
    exposing `last_assign_inputs` containing non-detached proxy tensors for ShadowTAL.
    """

    def __init__(self, model=None, num_classes: int = 20, target_class_id: int = 14):
        self._super_ready = False
        if _V8Base is not object and model is not None:
            try:
                super().__init__(model)
                self._super_ready = True
            except Exception:
                self._super_ready = False
        self.num_classes = int(num_classes)
        self.target_class_id = int(target_class_id)
        self.last_assign_inputs: Dict = {}
        self.last_assign_outputs: Dict = {}

    @classmethod
    def from_surrogate(cls, surrogate, num_classes: int, target_class_id: int):
        return cls(model=surrogate, num_classes=num_classes, target_class_id=target_class_id)

    @staticmethod
    def _to_decoded_preds(preds: torch.Tensor) -> torch.Tensor:
        if isinstance(preds, (list, tuple)):
            if len(preds) == 0:
                raise RuntimeError("Empty predictions passed to HijackedV8Loss.")
            if torch.is_tensor(preds[0]) and preds[0].ndim == 3:
                preds = preds[0]
            else:
                raise RuntimeError(
                    "HijackedV8Loss received non-decoded predictions. "
                    "Pass decoded [B,C,N] preds for assignment-aware proxy path."
                )
        if not torch.is_tensor(preds):
            raise RuntimeError("Predictions must be a tensor.")
        if preds.ndim != 3:
            raise RuntimeError(f"Decoded predictions should be [B,C,N], got {tuple(preds.shape)}")
        return preds

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
        x = boxes_xywh[..., 0]
        y = boxes_xywh[..., 1]
        w = boxes_xywh[..., 2]
        h = boxes_xywh[..., 3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)

        lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
        rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]

        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0.0) * (box1[:, 3] - box1[:, 1]).clamp(min=0.0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0.0) * (box2[:, 3] - box2[:, 1]).clamp(min=0.0))
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp_min(eps)

    def _build_gt_tensors(
        self,
        batch: Dict,
        device: torch.device,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = image_shape
        cls = batch.get("cls")
        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")

        if cls is None or bboxes is None or batch_idx is None:
            raise RuntimeError("batch must contain keys: cls, bboxes, batch_idx")

        cls = cls.to(device)
        bboxes = bboxes.to(device)
        batch_idx = batch_idx.to(device).to(torch.long).reshape(-1)

        bsz = int(batch.get("batch_size", int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1))
        gt_count = [int((batch_idx == i).sum().item()) for i in range(bsz)]
        max_gt = max(gt_count) if gt_count else 0

        gt_labels = torch.full((bsz, max_gt, 1), -1.0, device=device)
        gt_bboxes = torch.zeros((bsz, max_gt, 4), device=device)
        mask_gt = torch.zeros((bsz, max_gt, 1), device=device)

        for i in range(bsz):
            pos = torch.nonzero(batch_idx == i, as_tuple=False).reshape(-1)
            if pos.numel() == 0:
                continue
            c = cls[pos]
            b = bboxes[pos]
            bb = b.clone()
            if torch.max(torch.abs(bb)) <= 1.5:
                bb[:, 0] = bb[:, 0] * w
                bb[:, 1] = bb[:, 1] * h
                bb[:, 2] = bb[:, 2] * w
                bb[:, 3] = bb[:, 3] * h
            bb = self._xywh_to_xyxy(bb)
            n = pos.numel()
            gt_labels[i, :n, :] = c
            gt_bboxes[i, :n, :] = bb
            mask_gt[i, :n, :] = 1.0

        return gt_labels, gt_bboxes, mask_gt

    def _build_clean_gate(
        self,
        pred_scores_logits: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        assignment_topk: int,
    ) -> torch.Tensor:
        bsz, n_anchor, _ = pred_scores_logits.shape
        target_prob = torch.sigmoid(pred_scores_logits[:, :, self.target_class_id])
        gate = torch.zeros((bsz, n_anchor), device=pred_scores_logits.device, dtype=torch.bool)

        for b in range(bsz):
            valid = mask_gt[b, :, 0] > 0
            valid = valid & (gt_labels[b, :, 0].to(torch.long) == self.target_class_id)
            if not torch.any(valid):
                continue
            gt_b = gt_bboxes[b, valid]
            iou_mat = self._box_iou(pred_bboxes[b], gt_b)
            best_iou = iou_mat.max(dim=1).values
            align = (target_prob[b].detach() ** 0.5) * (best_iou.detach().clamp(min=0.0) ** 6.0)
            k = max(1, min(int(assignment_topk), int(align.numel())))
            idx = torch.topk(align, k=k, largest=True).indices
            gate[b, idx] = True

        return gate

    def cache_assign_inputs_only(
        self,
        preds: torch.Tensor,
        batch: Dict,
        image_shape: Tuple[int, int],
        assignment_topk: int = 100,
    ) -> Dict:
        preds = self._to_decoded_preds(preds)
        device = preds.device

        cls_start = 4
        cls_end = min(preds.shape[1], cls_start + self.num_classes)
        
        pred_scores_logits = preds[:, cls_start:cls_end, :].permute(0, 2, 1).contiguous()
        pred_bboxes = self._xywh_to_xyxy(preds[:, :4, :].permute(0, 2, 1).contiguous())

        h, w = image_shape
        # --- 修复 In-place Operation 错误 ---
        x1, y1, x2, y2 = pred_bboxes.unbind(-1)
        pred_bboxes = torch.stack([
            x1.clamp(0.0, float(w)),
            y1.clamp(0.0, float(h)),
            x2.clamp(0.0, float(w)),
            y2.clamp(0.0, float(h))
        ], dim=-1)
        # -----------------------------------

        gt_labels, gt_bboxes, mask_gt = self._build_gt_tensors(batch, device=device, image_shape=image_shape)
        gate = self._build_clean_gate(
            pred_scores_logits=pred_scores_logits,
            pred_bboxes=pred_bboxes,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
            assignment_topk=assignment_topk,
        )

        self.last_assign_inputs = {
            "pred_scores_logits": pred_scores_logits,
            "pred_bboxes": pred_bboxes,
            "gt_labels": gt_labels,
            "gt_bboxes": gt_bboxes,
            "mask_gt": mask_gt,
            "fg_mask": gate,
            "anchor_points": None,
            "stride_tensor": None,
        }
        self.last_assign_outputs = {
            "fg_mask": gate,
            "target_gt_idx": None,
        }
        return self.last_assign_inputs

    def get_assigned_targets_and_loss(self, preds, batch):
        out = None
        if self._super_ready:
            try:
                out = super().get_assigned_targets_and_loss(preds, batch)
            except Exception:
                out = None

        try:
            image_shape = batch.get("image_shape", None)
            if image_shape is None:
                img = batch.get("img", None)
                if torch.is_tensor(img) and img.ndim == 4:
                    image_shape = (int(img.shape[2]), int(img.shape[3]))
            if image_shape is not None:
                self.cache_assign_inputs_only(preds, batch, image_shape=image_shape)
        except Exception:
            pass

        return out