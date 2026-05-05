from typing import Dict, Optional

import torch
import torch.nn as nn


class DifferentiableShadowTAL(nn.Module):
    """
    Differentiable assignment-aware proxy that mirrors TAL continuous core:
    alignment = score^alpha * overlap^beta
    """

    def __init__(
        self,
        target_class_id: int,
        alpha: float = 0.5,
        beta: float = 6.0,
        topk: int = 100,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.target_class_id = int(target_class_id)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.topk = int(topk)
        self.eps = float(eps)

    @staticmethod
    def _safe_box(boxes: torch.Tensor) -> torch.Tensor:
        x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
        y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
        x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
        y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
        out = torch.stack([x1, y1, x2, y2], dim=-1)
        return out

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)

        box1 = self._safe_box(box1)
        box2 = self._safe_box(box2)

        lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
        rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]

        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0.0) * (box1[:, 3] - box1[:, 1]).clamp(min=0.0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0.0) * (box2[:, 3] - box2[:, 1]).clamp(min=0.0))
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp_min(self.eps)

    def _safe_ciou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        # [N, 4], [M, 4] -> [N, M]
        iou = self._box_iou(box1, box2)
        if box1.numel() == 0 or box2.numel() == 0:
            return iou

        b1 = self._safe_box(box1)
        b2 = self._safe_box(box2)

        b1_cx = (b1[:, 0] + b1[:, 2]) * 0.5
        b1_cy = (b1[:, 1] + b1[:, 3]) * 0.5
        b2_cx = (b2[:, 0] + b2[:, 2]) * 0.5
        b2_cy = (b2[:, 1] + b2[:, 3]) * 0.5

        rho2 = (b1_cx[:, None] - b2_cx[None, :]) ** 2 + (b1_cy[:, None] - b2_cy[None, :]) ** 2

        enc_x1 = torch.minimum(b1[:, None, 0], b2[None, :, 0])
        enc_y1 = torch.minimum(b1[:, None, 1], b2[None, :, 1])
        enc_x2 = torch.maximum(b1[:, None, 2], b2[None, :, 2])
        enc_y2 = torch.maximum(b1[:, None, 3], b2[None, :, 3])
        c2 = ((enc_x2 - enc_x1).clamp(min=self.eps) ** 2 + (enc_y2 - enc_y1).clamp(min=self.eps) ** 2)

        w1 = (b1[:, 2] - b1[:, 0]).clamp(min=self.eps)
        h1 = (b1[:, 3] - b1[:, 1]).clamp(min=self.eps)
        w2 = (b2[:, 2] - b2[:, 0]).clamp(min=self.eps)
        h2 = (b2[:, 3] - b2[:, 1]).clamp(min=self.eps)
        v = (4.0 / (torch.pi**2)) * (torch.atan(w2[None, :] / h2[None, :]) - torch.atan(w1[:, None] / h1[:, None])) ** 2
        with torch.no_grad():
            alpha_ciou = v / (1.0 - iou + v + self.eps)
        ciou = iou - (rho2 / c2) - alpha_ciou * v
        ciou = torch.where(torch.isfinite(ciou), ciou, iou)
        return ciou.clamp(min=self.eps, max=1.0)

    def _topk_mean(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        k = max(1, min(int(k), int(x.numel())))
        return torch.topk(x.reshape(-1), k=k, largest=True).values.mean()

    def forward(
        self,
        pred_scores_logits: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_scores_logits.ndim != 3:
            raise ValueError(f"pred_scores_logits should be [B,N,C], got {tuple(pred_scores_logits.shape)}")
        if pred_bboxes.ndim != 3 or pred_bboxes.shape[-1] != 4:
            raise ValueError(f"pred_bboxes should be [B,N,4], got {tuple(pred_bboxes.shape)}")
        if pred_scores_logits.shape[:2] != pred_bboxes.shape[:2]:
            raise ValueError("pred_scores_logits and pred_bboxes batch/anchor dims mismatch.")

        bsz, n_anchor, n_cls = pred_scores_logits.shape
        if self.target_class_id >= n_cls:
            raise RuntimeError(
                f"target_class_id overflow in ShadowTAL: target={self.target_class_id}, classes={n_cls}"
            )

        target_logits = pred_scores_logits[:, :, self.target_class_id]
        target_prob = torch.sigmoid(target_logits)
        overlaps = torch.zeros((bsz, n_anchor), device=pred_bboxes.device, dtype=pred_bboxes.dtype)

        for b in range(bsz):
            valid = mask_gt[b].reshape(-1) > 0
            if gt_labels.ndim == 3:
                gl = gt_labels[b, :, 0]
            else:
                gl = gt_labels[b, :]
            valid = valid & (gl.to(torch.long) == self.target_class_id)
            if not torch.any(valid):
                continue

            gt_b = gt_bboxes[b][valid]
            ov = self._safe_ciou(pred_bboxes[b], gt_b)
            if ov.numel() > 0:
                overlaps[b] = ov.max(dim=1).values

        align = (target_prob.clamp(min=self.eps) ** self.alpha) * (overlaps.clamp(min=self.eps) ** self.beta)
        if gate is not None:
            if gate.shape != align.shape:
                raise ValueError(f"gate shape mismatch: expect {tuple(align.shape)}, got {tuple(gate.shape)}")
            gate_f = gate.detach().float()
            align_masked = align * gate_f
            gate_ratio = gate_f.mean()
        else:
            align_masked = align
            gate_ratio = (align > 0).float().mean()

        k = self.topk if topk is None else int(topk)
        topk_per_batch = []
        for b in range(bsz):
            if gate is not None and torch.any(gate[b] > 0):
                v = align_masked[b][gate[b] > 0]
            else:
                v = align_masked[b]
            topk_per_batch.append(self._topk_mean(v, k))
        topk_mean = torch.stack(topk_per_batch).mean()

        out = {
            "align_proxy": align_masked,
            "target_prob": target_prob,
            "overlaps": overlaps,
            "topk_alignment": topk_mean,
            "gate_positive_ratio": gate_ratio,
            "target_prob_mean": target_prob.mean(),
            "overlap_mean": overlaps.mean(),
        }
        return out
