from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Sequence


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


@dataclass(frozen=True)
class TargetRouteResult:
    loss: torch.Tensor
    easy_classification: torch.Tensor
    box_teacher: torch.Tensor
    evasion: torch.Tensor
    positive_count: int
    status: str


@dataclass(frozen=True)
class NonTargetClassConstraint:
    class_id: int
    count: int
    cls_margin: torch.Tensor
    box_margin: torch.Tensor
    cls_violation: torch.Tensor
    box_violation: torch.Tensor
    clean_probability_mean: float
    adv_probability_mean: float


@dataclass(frozen=True)
class NonTargetConstraintSet:
    constraints: tuple[NonTargetClassConstraint, ...]
    status: str


def _class_only_logits(
    logits: torch.Tensor,
    *,
    num_classes: int,
    name: str,
) -> None:
    if logits.ndim != 3 or logits.shape[-1] != num_classes:
        raise ValueError(
            f"{name} must be class-only [B,N,{num_classes}] logits; "
            "YOLOv8 has no separate objectness output in this path."
        )


def _normalize_gate(gate: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if gate.ndim == 3 and gate.shape[-1] == 1:
        gate = gate[..., 0]
    if gate.shape != shape:
        raise ValueError(f"Gate shape {tuple(gate.shape)} != {tuple(shape)}.")
    return gate.detach().bool()


def aligned_ciou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("aligned_ciou expects matching [N,4] tensors.")
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    def safe(boxes: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.minimum(boxes[:, 0], boxes[:, 2]),
                torch.minimum(boxes[:, 1], boxes[:, 3]),
                torch.maximum(boxes[:, 0], boxes[:, 2]),
                torch.maximum(boxes[:, 1], boxes[:, 3]),
            ),
            dim=1,
        )

    box1 = safe(boxes1)
    box2 = safe(boxes2)
    intersection_wh = (
        torch.minimum(box1[:, 2:], box2[:, 2:])
        - torch.maximum(box1[:, :2], box2[:, :2])
    ).clamp(min=0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (
        box1[:, 3] - box1[:, 1]
    ).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (
        box2[:, 3] - box2[:, 1]
    ).clamp(min=0)
    iou = intersection / (area1 + area2 - intersection).clamp_min(eps)

    center1 = (box1[:, :2] + box1[:, 2:]) * 0.5
    center2 = (box2[:, :2] + box2[:, 2:]) * 0.5
    center_distance = (center1 - center2).square().sum(dim=1)
    enclosure_wh = (
        torch.maximum(box1[:, 2:], box2[:, 2:])
        - torch.minimum(box1[:, :2], box2[:, :2])
    ).clamp(min=eps)
    enclosure_diagonal = enclosure_wh.square().sum(dim=1).clamp_min(eps)

    width1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    height1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    width2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    height2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)
    aspect = (4.0 / torch.pi**2) * (
        torch.atan(width2 / height2) - torch.atan(width1 / height1)
    ).square()
    with torch.no_grad():
        aspect_weight = aspect / (1.0 - iou + aspect + eps)
    ciou = iou - center_distance / enclosure_diagonal - aspect_weight * aspect
    return torch.where(torch.isfinite(ciou), ciou, iou).clamp(
        min=eps,
        max=1.0,
    )


def compute_target_route(
    *,
    route: str,
    adv_class_logits: torch.Tensor,
    adv_boxes: torch.Tensor,
    clean_boxes: torch.Tensor,
    target_gate: torch.Tensor,
    target_class_id: int,
    num_classes: int,
    box_teacher_weight: float = 1.0,
    shadow_tal: DifferentiableShadowTAL | None = None,
    gt_labels: torch.Tensor | None = None,
    gt_bboxes: torch.Tensor | None = None,
    mask_gt: torch.Tensor | None = None,
) -> TargetRouteResult:
    if route not in {"easy_cls", "tal_evasion"}:
        raise ValueError("route must be exactly 'easy_cls' or 'tal_evasion'.")
    _class_only_logits(
        adv_class_logits,
        num_classes=num_classes,
        name="adv_class_logits",
    )
    if adv_boxes.shape != clean_boxes.shape or adv_boxes.shape[-1] != 4:
        raise ValueError("adv_boxes and clean_boxes must share shape [B,N,4].")
    if adv_boxes.shape[:2] != adv_class_logits.shape[:2]:
        raise ValueError("Box and class anchor dimensions must match.")
    if not 0 <= target_class_id < num_classes:
        raise ValueError("target_class_id is outside the class-only logits.")
    gate = _normalize_gate(target_gate, adv_class_logits.shape[:2])
    positive_count = int(gate.sum().item())
    zero = adv_class_logits.sum() * 0.0 + adv_boxes.sum() * 0.0
    if positive_count == 0:
        return TargetRouteResult(
            loss=zero,
            easy_classification=zero,
            box_teacher=zero,
            evasion=zero,
            positive_count=0,
            status="not_applicable",
        )

    if route == "easy_cls":
        target_logits = adv_class_logits[..., target_class_id][gate]
        easy = F.softplus(-target_logits).mean()
        box_teacher = F.smooth_l1_loss(
            adv_boxes[gate],
            clean_boxes.detach()[gate],
        )
        evasion = zero
        loss = easy + float(box_teacher_weight) * box_teacher
    else:
        if shadow_tal is None or any(
            value is None for value in (gt_labels, gt_bboxes, mask_gt)
        ):
            raise ValueError(
                "tal_evasion requires shadow_tal and clean GT tensors."
            )
        shadow = shadow_tal(
            pred_scores_logits=adv_class_logits,
            pred_bboxes=adv_boxes,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
            gate=gate,
        )
        easy = zero
        box_teacher = zero
        evasion = shadow["topk_alignment"]
        loss = evasion

    return TargetRouteResult(
        loss=loss,
        easy_classification=easy,
        box_teacher=box_teacher,
        evasion=evasion,
        positive_count=positive_count,
        status="active",
    )


def build_non_target_constraints(
    *,
    clean_class_logits: torch.Tensor,
    adv_class_logits: torch.Tensor,
    clean_boxes: torch.Tensor,
    adv_boxes: torch.Tensor,
    assigned_gt_boxes: torch.Tensor,
    assigned_labels: torch.Tensor,
    real_foreground: torch.Tensor,
    target_class_id: int,
    num_classes: int,
    tau_cls: float = 0.005,
    tau_box: float = 0.02,
) -> NonTargetConstraintSet:
    _class_only_logits(
        clean_class_logits,
        num_classes=num_classes,
        name="clean_class_logits",
    )
    _class_only_logits(
        adv_class_logits,
        num_classes=num_classes,
        name="adv_class_logits",
    )
    if clean_class_logits.shape != adv_class_logits.shape:
        raise ValueError("Clean and adv class logits must have matching shapes.")
    expected_box_shape = (*clean_class_logits.shape[:2], 4)
    if any(
        tensor.shape != expected_box_shape
        for tensor in (clean_boxes, adv_boxes, assigned_gt_boxes)
    ):
        raise ValueError("All box tensors must have shape [B,N,4].")
    if assigned_labels.ndim == 3 and assigned_labels.shape[-1] == 1:
        assigned_labels = assigned_labels[..., 0]
    if assigned_labels.shape != clean_class_logits.shape[:2]:
        raise ValueError("assigned_labels must have shape [B,N].")
    foreground = _normalize_gate(
        real_foreground,
        clean_class_logits.shape[:2],
    )
    labels = assigned_labels.detach().long()
    valid_label = (labels >= 0) & (labels < num_classes)
    non_target = foreground & valid_label & (labels != target_class_id)
    active_ids: Sequence[int] = sorted(
        int(value)
        for value in labels[non_target].unique().detach().cpu().tolist()
    )
    if not active_ids:
        return NonTargetConstraintSet(constraints=(), status="not_applicable")

    clean_probabilities = clean_class_logits.detach().sigmoid()
    adv_probabilities = adv_class_logits.sigmoid()
    constraints: list[NonTargetClassConstraint] = []
    for class_id in active_ids:
        class_mask = non_target & (labels == class_id)
        clean_probability = clean_probabilities[..., class_id][class_mask]
        adv_probability = adv_probabilities[..., class_id][class_mask]
        cls_margin = (clean_probability - adv_probability).mean()
        cls_violation = F.relu(cls_margin - float(tau_cls))

        gt = assigned_gt_boxes.detach()[class_mask]
        clean_quality = aligned_ciou(clean_boxes.detach()[class_mask], gt)
        adv_quality = aligned_ciou(adv_boxes[class_mask], gt)
        box_margin = (clean_quality - adv_quality).mean()
        box_violation = F.relu(box_margin - float(tau_box))
        constraints.append(
            NonTargetClassConstraint(
                class_id=class_id,
                count=int(class_mask.sum().item()),
                cls_margin=cls_margin,
                box_margin=box_margin,
                cls_violation=cls_violation,
                box_violation=box_violation,
                clean_probability_mean=float(clean_probability.mean().item()),
                adv_probability_mean=float(
                    adv_probability.detach().mean().item()
                ),
            )
        )
    return NonTargetConstraintSet(
        constraints=tuple(constraints),
        status="active",
    )
