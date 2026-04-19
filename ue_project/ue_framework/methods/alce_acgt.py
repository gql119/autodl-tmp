import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def renorm_yolo_bbox_after_padding(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    orig_w: int,
    orig_h: int,
    pad_w: int,
    pad_h: int,
) -> Tuple[float, float, float, float]:
    """ACGT: padding-aware YOLO bbox renormalization in padded frame."""
    new_cx = float(cx) * float(orig_w) / float(max(1, pad_w))
    new_cy = float(cy) * float(orig_h) / float(max(1, pad_h))
    new_bw = float(bw) * float(orig_w) / float(max(1, pad_w))
    new_bh = float(bh) * float(orig_h) / float(max(1, pad_h))
    return (
        float(max(0.0, min(new_cx, 1.0))),
        float(max(0.0, min(new_cy, 1.0))),
        float(max(0.0, min(new_bw, 1.0))),
        float(max(0.0, min(new_bh, 1.0))),
    )


def project_strict_gate_to_fpn(
    strict_gate_1d: torch.Tensor,
    shape_layers: List[str],
    features_cache: dict,
) -> dict:
    """
    ACGT: Project 1D assignment gate back to 2D FPN multi-scale grids.
    Returns a dictionary mapping layer_name to [B, 1, H, W].
    """
    gate_dict = {}
    anchor_offset = 0
    for layer_name in shape_layers:
        if layer_name not in features_cache:
            continue
        z = features_cache[layer_name]
        bsz, _, h, w = z.shape
        n = h * w
        gate_1d = strict_gate_1d[:, anchor_offset : anchor_offset + n]
        gate_dict[layer_name] = gate_1d.view(bsz, 1, h, w).float()
        anchor_offset += n
    return gate_dict


def build_non_target_core_mask(
    batch: Dict,
    pad_h: int,
    pad_w: int,
    target_class_id: int,
    device: torch.device,
    core_scale: float = 0.8,
) -> torch.Tensor:
    """
    RLCP: Render non-target object core mask by shrinking bbox around center.
    """
    bsz = int(batch.get("batch_size", 1))
    mask = torch.zeros((bsz, 1, pad_h, pad_w), device=device)
    batch_idx = batch.get("batch_idx", torch.zeros(0, device=device))
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device))
    clss = batch.get("cls", torch.zeros((0, 1), device=device))
    scale = float(max(0.05, min(core_scale, 1.0)))

    for b in range(bsz):
        valid_idx = batch_idx == b
        if valid_idx.sum() <= 0:
            continue
        boxes = bboxes[valid_idx]
        classes = clss[valid_idx].squeeze(-1)
        for box, cls_id in zip(boxes, classes):
            if int(cls_id.item()) == int(target_class_id):
                continue

            cx, cy, bw, bh = box
            bw = float(bw) * scale
            bh = float(bh) * scale
            x1 = max(0, int((float(cx) - bw * 0.5) * pad_w))
            y1 = max(0, int((float(cy) - bh * 0.5) * pad_h))
            x2 = min(pad_w, int((float(cx) + bw * 0.5) * pad_w))
            y2 = min(pad_h, int((float(cy) + bh * 0.5) * pad_h))
            if x2 > x1 and y2 > y1:
                mask[b, 0, y1:y2, x1:x2] = 1.0
    return mask


def build_non_target_objects_mask(
    batch: Dict,
    pad_h: int,
    pad_w: int,
    target_class_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Backward-compatible alias.
    """
    return build_non_target_core_mask(
        batch=batch,
        pad_h=pad_h,
        pad_w=pad_w,
        target_class_id=target_class_id,
        device=device,
        core_scale=0.8,
    )


def build_local_context_mask(
    inner_mask: torch.Tensor, 
    r_inner: int = 12, 
    r_outer: int = 28,
    expand_ratio: float = None, # 🚀 增加向下兼容参数
    ring_width: int = None      # 🚀 增加向下兼容参数
) -> torch.Tensor:
    """
    RLCP context ring: mid-range annulus around target support.
    """
    # 向下兼容旧脚本调用
    if expand_ratio is not None and ring_width is not None:
        r_inner = int(max(1, ring_width))
        r_outer = int(max(r_inner + 1, round(ring_width * expand_ratio)))

    r_inner = int(max(1, r_inner))
    r_outer = int(max(r_inner + 1, r_outer))

    k_inner = r_inner * 2 + 1
    k_outer = r_outer * 2 + 1
    dilated_inner = F.max_pool2d(inner_mask, kernel_size=k_inner, stride=1, padding=r_inner)
    dilated_outer = F.max_pool2d(inner_mask, kernel_size=k_outer, stride=1, padding=r_outer)
    return torch.clamp(dilated_outer - dilated_inner, 0.0, 1.0)

def build_confounder_mask(
    local_ctx_mask: torch.Tensor,
    all_objects_mask: torch.Tensor,
    ring_mask: torch.Tensor,
) -> torch.Tensor:
    m = local_ctx_mask * (1.0 - all_objects_mask) * (1.0 - ring_mask)
    return torch.clamp(m, 0.0, 1.0)

def build_pag_gate(
    strict_gate_1d: torch.Tensor,
    target_scores: torch.Tensor,
    target_class_id: int,
    top_ratio: float = 0.3,
    min_keep: int = 8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PAG: keep only top-ratio units inside strict target-assigned set.
    """
    if strict_gate_1d is None or strict_gate_1d.numel() == 0:
        return strict_gate_1d, {"pag_positive_ratio": 0.0, "pag_threshold": 0.0, "pag_mean_target_score": 0.0, "strict_positive": 0.0, "pag_positive": 0.0, "pag_fallback_count": 0.0}

    strict_gate = strict_gate_1d.bool()
    strict_total = int(strict_gate.sum().item())
    if strict_total <= 0:
        return strict_gate, {"pag_positive_ratio": 0.0, "pag_threshold": 0.0, "pag_mean_target_score": 0.0, "strict_positive": 0.0, "pag_positive": 0.0, "pag_fallback_count": 0.0}

    if not torch.is_tensor(target_scores) or target_scores.numel() == 0:
        return strict_gate, {"pag_positive_ratio": 1.0, "pag_threshold": 0.0, "pag_mean_target_score": 0.0, "strict_positive": float(strict_total), "pag_positive": float(strict_total), "pag_fallback_count": float(strict_gate.shape[0])}

    if target_scores.ndim == 3:
        if target_scores.shape[-1] > int(target_class_id):
            score_t = target_scores[:, :, int(target_class_id)]
        else:
            score_t = target_scores.max(dim=-1).values
    elif target_scores.ndim == 2:
        score_t = target_scores
    else:
        score_t = torch.zeros_like(strict_gate, dtype=torch.float32)

    top_ratio = float(max(0.01, min(top_ratio, 1.0)))
    min_keep = int(max(1, min_keep))
    pag_gate = torch.zeros_like(strict_gate)

    thresholds = []
    mean_scores = []
    pag_total = 0
    fallback_count = 0  # 🚀 记录因为数量太少而退回 strict gate 的 batch 数量

    for b in range(strict_gate.shape[0]):
        idx = torch.nonzero(strict_gate[b], as_tuple=False).view(-1)
        n = int(idx.numel())
        if n <= 0:
            continue
        vals = score_t[b, idx]
        if vals.numel() > 0:
            mean_scores.append(float(vals.mean().item()))

        if n < min_keep:
            pag_gate[b, idx] = True
            pag_total += n
            fallback_count += 1 # 🚀 Fallback 触发
            continue

        k = int(max(1, math.ceil(n * top_ratio)))
        topv, topi = torch.topk(vals, k=k, largest=True)
        keep_idx = idx[topi]
        pag_gate[b, keep_idx] = True
        pag_total += k
        thresholds.append(float(topv.min().item()))

    if pag_total <= 0:
        pag_gate = strict_gate.clone()
        pag_total = strict_total
        fallback_count = strict_gate.shape[0]

    return pag_gate, {
        "pag_positive_ratio": float(pag_total) / float(max(1, strict_total)),
        "pag_threshold": float(sum(thresholds) / max(1, len(thresholds))) if thresholds else 0.0,
        "pag_mean_target_score": float(sum(mean_scores) / max(1, len(mean_scores))) if mean_scores else 0.0,
        "strict_positive": float(strict_total),
        "pag_positive": float(pag_total),
        "pag_fallback_count": float(fallback_count), # 🚀 返回 fallback 统计
    }