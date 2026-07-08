from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import bbox2dist, bbox_iou


@dataclass
class SupervisionMasks:
    protected_positive_mask: torch.Tensor
    authorized_positive_mask: torch.Tensor
    shared_positive_mask: torch.Tensor
    background_mask: torch.Tensor
    ambiguous_mask: torch.Tensor


@dataclass
class DecomposedDetectionLoss:
    protected_total: torch.Tensor
    protected_cls: torch.Tensor
    protected_box: torch.Tensor
    protected_dfl: torch.Tensor
    authorized_total: torch.Tensor
    authorized_cls: torch.Tensor
    authorized_box: torch.Tensor
    authorized_dfl: torch.Tensor
    shared_total: torch.Tensor
    shared_cls: torch.Tensor
    shared_box: torch.Tensor
    shared_dfl: torch.Tensor
    reconstructed_total: torch.Tensor
    original_full_total: torch.Tensor
    reconstruction_error: torch.Tensor
    cls_reconstruction_error: torch.Tensor
    box_reconstruction_error: torch.Tensor
    dfl_reconstruction_error: torch.Tensor
    masks: SupervisionMasks
    statistics: Dict[str, float]


class SupervisionDecomposer:
    def __init__(
        self,
        adapter=None,
        protected_class_id: int = 14,
        authorized_class_ids: Optional[Iterable[int]] = None,
        num_classes: int = 20,
        ambiguous_iou_threshold: float = 0.5,
        target_score_reliability_threshold: float = 0.0,
        eps: float = 1.0e-8,
    ) -> None:
        self.adapter = adapter
        self.protected_class_id = int(protected_class_id)
        self.num_classes = int(num_classes)
        if authorized_class_ids is None or authorized_class_ids == "auto":
            self.authorized_class_ids = [i for i in range(self.num_classes) if i != self.protected_class_id]
        else:
            self.authorized_class_ids = [int(v) for v in authorized_class_ids]
        self.ambiguous_iou_threshold = float(ambiguous_iou_threshold)
        self.target_score_reliability_threshold = float(target_score_reliability_threshold)
        self.eps = float(eps)

    def decompose(self, predictions, batch: Dict, tal_outputs=None, full_loss_outputs=None) -> DecomposedDetectionLoss:
        if self.adapter is None:
            raise ValueError("SupervisionDecomposer.decompose requires an adapter.")
        state = self.adapter._build_assignment_state(predictions, batch)
        batch_size = int(state["pred_scores"].shape[0])
        target_scores_sum = state["target_scores"].sum().clamp_min(1.0)
        ambiguous = self._assignment_ambiguous_mask(state, batch)
        per_box, per_dfl = self._per_unit_box_dfl(state, target_scores_sum)
        return self.decompose_from_tensors(
            pred_scores=state["pred_scores"],
            target_scores=state["target_scores"],
            target_labels=state["target_labels"].long(),
            fg_mask=state["fg_mask"].bool(),
            ambiguous_mask=ambiguous,
            per_unit_box_loss=per_box,
            per_unit_dfl_loss=per_dfl,
            target_scores_sum=target_scores_sum,
            batch_size=batch_size,
            cls_gain=self.adapter._hyp_gain("cls"),
            box_gain=self.adapter._hyp_gain("box"),
            dfl_gain=self.adapter._hyp_gain("dfl"),
            class_weights=getattr(self.adapter.criterion, "class_weights", None),
        )

    def decompose_from_tensors(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        target_labels: torch.Tensor,
        fg_mask: torch.Tensor,
        ambiguous_mask: Optional[torch.Tensor] = None,
        per_unit_box_loss: Optional[torch.Tensor] = None,
        per_unit_dfl_loss: Optional[torch.Tensor] = None,
        target_scores_sum: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        cls_gain: float = 1.0,
        box_gain: float = 1.0,
        dfl_gain: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> DecomposedDetectionLoss:
        if pred_scores.ndim != 3:
            raise ValueError(f"pred_scores must be [B,N,C], got {tuple(pred_scores.shape)}")
        device = pred_scores.device
        dtype = pred_scores.dtype
        fg_mask = fg_mask.bool()
        if ambiguous_mask is None:
            ambiguous_mask = torch.zeros_like(fg_mask)
        ambiguous_mask = ambiguous_mask.bool() & fg_mask
        labels = target_labels.long().clamp(min=0, max=self.num_classes - 1)
        auth_ids = torch.tensor(self.authorized_class_ids, device=device, dtype=labels.dtype)
        protected_pos = fg_mask & ~ambiguous_mask & (labels == self.protected_class_id)
        authorized_pos = fg_mask & ~ambiguous_mask & torch.isin(labels, auth_ids)
        shared_pos = fg_mask & ~(protected_pos | authorized_pos)
        background = ~fg_mask

        if target_scores_sum is None:
            target_scores_sum = target_scores.sum().clamp_min(1.0)
        if not torch.is_tensor(target_scores_sum):
            target_scores_sum = pred_scores.new_tensor(float(target_scores_sum))
        if batch_size is None:
            batch_size = int(pred_scores.shape[0])
        scale_cls = float(cls_gain) * float(batch_size) / target_scores_sum
        scale_box = float(box_gain) * float(batch_size)
        scale_dfl = float(dfl_gain) * float(batch_size)

        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores.to(dtype), reduction="none")
        if class_weights is not None:
            weights = class_weights.to(device=device, dtype=dtype)
            bce = bce * weights.view(1, 1, -1)
        full_cls = bce.sum() * scale_cls

        protected_cls_raw = self._assigned_class_bce_sum(bce, labels, protected_pos)
        authorized_cls_raw = self._assigned_class_bce_sum(bce, labels, authorized_pos)
        protected_cls = protected_cls_raw * scale_cls
        authorized_cls = authorized_cls_raw * scale_cls
        shared_cls = full_cls - protected_cls - authorized_cls

        if per_unit_box_loss is None:
            per_unit_box_loss = torch.zeros_like(fg_mask, dtype=dtype)
        if per_unit_dfl_loss is None:
            per_unit_dfl_loss = torch.zeros_like(fg_mask, dtype=dtype)
        protected_box = per_unit_box_loss[protected_pos].sum() * scale_box
        authorized_box = per_unit_box_loss[authorized_pos].sum() * scale_box
        shared_box = per_unit_box_loss[shared_pos].sum() * scale_box
        protected_dfl = per_unit_dfl_loss[protected_pos].sum() * scale_dfl
        authorized_dfl = per_unit_dfl_loss[authorized_pos].sum() * scale_dfl
        shared_dfl = per_unit_dfl_loss[shared_pos].sum() * scale_dfl

        full_box = (per_unit_box_loss[fg_mask].sum() * scale_box).to(dtype)
        full_dfl = (per_unit_dfl_loss[fg_mask].sum() * scale_dfl).to(dtype)
        protected_total = protected_cls + protected_box + protected_dfl
        authorized_total = authorized_cls + authorized_box + authorized_dfl
        shared_total = shared_cls + shared_box + shared_dfl
        reconstructed = protected_total + authorized_total + shared_total
        original = full_cls + full_box + full_dfl
        cls_recon = (protected_cls + authorized_cls + shared_cls - full_cls).abs()
        box_recon = (protected_box + authorized_box + shared_box - full_box).abs()
        dfl_recon = (protected_dfl + authorized_dfl + shared_dfl - full_dfl).abs()
        err = (reconstructed - original).abs()
        denom = original.detach().abs().clamp_min(self.eps)
        stats = {
            "protected_positive_count": float(protected_pos.sum().detach().item()),
            "authorized_positive_count": float(authorized_pos.sum().detach().item()),
            "shared_positive_count": float(shared_pos.sum().detach().item()),
            "background_count": float(background.sum().detach().item()),
            "ambiguous_positive_count": float(ambiguous_mask.sum().detach().item()),
            "ambiguous_positive_ratio": float(ambiguous_mask.sum().detach().item() / max(float(fg_mask.sum().detach().item()), 1.0)),
            "original_full_total": float(original.detach().item()),
            "reconstructed_total": float(reconstructed.detach().item()),
            "absolute_reconstruction_error": float(err.detach().item()),
            "relative_reconstruction_error": float((err / denom).detach().item()),
            "cls_reconstruction_error": float(cls_recon.detach().item()),
            "box_reconstruction_error": float(box_recon.detach().item()),
            "dfl_reconstruction_error": float(dfl_recon.detach().item()),
            "shared_cls_nonnegative": float(shared_cls.detach().item() >= -1.0e-6),
            "ambiguous_cls_loss": float((bce[ambiguous_mask].sum() * scale_cls).detach().item()) if ambiguous_mask.any() else 0.0,
            "ambiguous_box_loss": float((per_unit_box_loss[ambiguous_mask].sum() * scale_box).detach().item()) if ambiguous_mask.any() else 0.0,
            "ambiguous_dfl_loss": float((per_unit_dfl_loss[ambiguous_mask].sum() * scale_dfl).detach().item()) if ambiguous_mask.any() else 0.0,
        }
        return DecomposedDetectionLoss(
            protected_total=protected_total,
            protected_cls=protected_cls,
            protected_box=protected_box,
            protected_dfl=protected_dfl,
            authorized_total=authorized_total,
            authorized_cls=authorized_cls,
            authorized_box=authorized_box,
            authorized_dfl=authorized_dfl,
            shared_total=shared_total,
            shared_cls=shared_cls,
            shared_box=shared_box,
            shared_dfl=shared_dfl,
            reconstructed_total=reconstructed,
            original_full_total=original,
            reconstruction_error=err,
            cls_reconstruction_error=cls_recon,
            box_reconstruction_error=box_recon,
            dfl_reconstruction_error=dfl_recon,
            masks=SupervisionMasks(protected_pos, authorized_pos, shared_pos, background, ambiguous_mask),
            statistics=stats,
        )

    @staticmethod
    def _assigned_class_bce_sum(bce: torch.Tensor, labels: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        if not unit_mask.any():
            return bce.sum() * 0.0
        pos = torch.nonzero(unit_mask, as_tuple=False)
        b_idx, u_idx = pos[:, 0], pos[:, 1]
        return bce[b_idx, u_idx].gather(1, labels[b_idx, u_idx].unsqueeze(1)).sum()

    def _assignment_ambiguous_mask(self, state: Dict[str, torch.Tensor], batch: Dict) -> torch.Tensor:
        fg = state["fg_mask"].bool()
        ambiguous = torch.zeros_like(fg)
        target_gt_idx = state["target_gt_idx"].long()
        invalid = fg & (target_gt_idx < 0)
        ambiguous |= invalid
        score_max = state["target_scores"].amax(dim=-1)
        ambiguous |= fg & (score_max <= self.target_score_reliability_threshold)

        gt_ambiguous = self._gt_ambiguous_flags(batch, state["pred_scores"].shape[0], state["pred_scores"].device)
        for b, flags in enumerate(gt_ambiguous):
            if flags.numel() == 0:
                continue
            idx = target_gt_idx[b].clamp(min=0, max=max(int(flags.numel()) - 1, 0))
            ambiguous[b] |= fg[b] & flags[idx]
        return ambiguous

    def _gt_ambiguous_flags(self, batch: Dict, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        cls = torch.as_tensor(batch.get("cls", []), device=device).reshape(-1).long()
        bboxes = torch.as_tensor(batch.get("bboxes", []), device=device, dtype=torch.float32).reshape(-1, 4)
        batch_idx = torch.as_tensor(batch.get("batch_idx", torch.zeros(cls.numel(), device=device)), device=device).reshape(-1).long()
        flags_by_batch: list[torch.Tensor] = []
        auth_ids = torch.tensor(self.authorized_class_ids, device=device, dtype=cls.dtype)
        for b in range(batch_size):
            keep = batch_idx == b
            labels = cls[keep]
            boxes = self._xywh_to_xyxy(bboxes[keep])
            flags = torch.zeros(labels.shape[0], device=device, dtype=torch.bool)
            p_idx = torch.nonzero(labels == self.protected_class_id, as_tuple=False).flatten()
            a_idx = torch.nonzero(torch.isin(labels, auth_ids), as_tuple=False).flatten()
            if p_idx.numel() and a_idx.numel():
                ious = self._box_iou_xyxy(boxes[p_idx], boxes[a_idx])
                pairs = torch.nonzero(ious >= self.ambiguous_iou_threshold, as_tuple=False)
                if pairs.numel():
                    flags[p_idx[pairs[:, 0]]] = True
                    flags[a_idx[pairs[:, 1]]] = True
            flags_by_batch.append(flags)
        return flags_by_batch

    def _per_unit_box_dfl(self, state: Dict[str, torch.Tensor], target_scores_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fg = state["fg_mask"].bool()
        dtype = state["pred_scores"].dtype
        per_box = torch.zeros_like(fg, dtype=dtype)
        per_dfl = torch.zeros_like(fg, dtype=dtype)
        if not fg.any():
            return per_box, per_dfl

        target_bboxes = state["target_bboxes"] / state["stride_tensor"]
        weight = state["target_scores"].sum(-1)[fg].to(dtype)
        iou = bbox_iou(state["pred_bboxes"][fg], target_bboxes[fg], xywh=False, CIoU=True).squeeze(-1).to(dtype)
        per_box[fg] = ((1.0 - iou) * weight) / target_scores_sum

        dfl_loss = self.adapter.criterion.bbox_loss.dfl_loss
        if dfl_loss:
            target_ltrb = bbox2dist(state["anchor_points"], target_bboxes, dfl_loss.reg_max - 1)
            raw_dfl = dfl_loss(state["pred_distri"][fg].view(-1, dfl_loss.reg_max), target_ltrb[fg]).squeeze(-1).to(dtype)
            per_dfl[fg] = (raw_dfl * weight) / target_scores_sum
        return per_box, per_dfl

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], dim=-1)

    @staticmethod
    def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        if a.numel() == 0 or b.numel() == 0:
            return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
        lt = torch.maximum(a[:, None, :2], b[None, :, :2])
        rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
        return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(eps)
