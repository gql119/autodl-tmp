import math
from typing import Dict, List, Optional, Tuple, Union
from collections.abc import Sequence

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


def split_gate_by_fpn_layers(
    gate_1d: torch.Tensor,
    shape_layers: List[str],
    layer_sizes: Optional[List[int]] = None,
    features_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Split a [B, N] gate into per-layer slices based on FPN layer sizes.
    Returns:
      {
        layer_name: {
          "start": int,
          "end": int,
          "gate_1d": [B, n_l],
          "gate_2d": [B,1,H,W] (optional when features_cache has layer)
        }
      }
    """
    if gate_1d is None:
        return {}
    if gate_1d.ndim != 2:
        raise ValueError(f"gate_1d must be [B,N], got {tuple(gate_1d.shape)}")

    if layer_sizes is None:
        if features_cache is None:
            raise ValueError("Either layer_sizes or features_cache must be provided.")
        layer_sizes = []
        for layer_name in shape_layers:
            if layer_name not in features_cache:
                raise ValueError(f"Layer {layer_name} missing from features_cache.")
            _, _, h, w = features_cache[layer_name].shape
            layer_sizes.append(int(h * w))

    if len(layer_sizes) != len(shape_layers):
        raise ValueError(
            f"layer_sizes length mismatch: len(layer_sizes)={len(layer_sizes)} vs len(shape_layers)={len(shape_layers)}"
        )

    total = int(sum(layer_sizes))
    if total != int(gate_1d.shape[1]):
        raise ValueError(
            f"layer_sizes sum mismatch: sum(layer_sizes)={total} vs gate_1d width={int(gate_1d.shape[1])}"
        )

    out = {}
    offset = 0
    for layer_name, n_l in zip(shape_layers, layer_sizes):
        end = offset + int(n_l)
        g_slice = gate_1d[:, offset:end]
        item = {
            "start": offset,
            "end": end,
            "gate_1d": g_slice,
        }
        if features_cache is not None and layer_name in features_cache:
            _, _, h, w = features_cache[layer_name].shape
            item["gate_2d"] = g_slice.view(g_slice.shape[0], 1, h, w).float()
        out[layer_name] = item
        offset = end
    return out


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
    """Backward-compatible alias."""
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
    expand_ratio: float = None,
    ring_width: int = None
) -> torch.Tensor:
    """RLCP context ring: mid-range annulus around target support."""
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


def build_scale_adaptive_context_mask(
    batch: Dict,
    pad_h: int,
    pad_w: int,
    target_class_id: int,
    device: torch.device,
    alpha: float = 0.15,
    beta: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Adaptive RLCP (bbox-aware, coordinate-aware):
    - only use target-class bboxes
    - target_scale = sqrt(abs_bw * abs_bh)
    - r_in = alpha * target_scale
    - r_out = beta * target_scale
    - draw outer rect, carve inner rect, and carve target body rect
    Returns:
      mask: [B,1,H,W]
      stats: adaptive radii statistics for logging
    """
    bsz = int(batch.get("batch_size", 1))
    mask = torch.zeros((bsz, 1, pad_h, pad_w), device=device, dtype=torch.float32)

    batch_idx = batch.get("batch_idx", torch.zeros(0, device=device))
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device))
    clss = batch.get("cls", torch.zeros((0, 1), device=device))

    a = float(max(0.0, alpha))
    b = float(max(a + 1e-6, beta))

    inner_vals = []
    outer_vals = []

    for b_id in range(bsz):
        valid_idx = batch_idx == b_id
        if valid_idx.sum() <= 0:
            continue

        boxes = bboxes[valid_idx]
        classes = clss[valid_idx].squeeze(-1)
        for box, cls_id in zip(boxes, classes):
            if int(cls_id.item()) != int(target_class_id):
                continue

            cx, cy, bw, bh = [float(v) for v in box]
            abs_bw = max(1.0, bw * float(pad_w))
            abs_bh = max(1.0, bh * float(pad_h))
            target_scale = math.sqrt(abs_bw * abs_bh)

            r_in = max(1.0, a * target_scale)
            r_out = max(r_in + 1.0, b * target_scale)
            inner_vals.append(float(r_in))
            outer_vals.append(float(r_out))

            x_c = cx * float(pad_w)
            y_c = cy * float(pad_h)
            half_w = abs_bw * 0.5
            half_h = abs_bh * 0.5

            # Outer rectangle
            xo1 = max(0, int(math.floor(x_c - (half_w + r_out))))
            yo1 = max(0, int(math.floor(y_c - (half_h + r_out))))
            xo2 = min(pad_w, int(math.ceil(x_c + (half_w + r_out))))
            yo2 = min(pad_h, int(math.ceil(y_c + (half_h + r_out))))
            if xo2 > xo1 and yo2 > yo1:
                mask[b_id, 0, yo1:yo2, xo1:xo2] = 1.0

            # Inner rectangle (carve)
            xi1 = max(0, int(math.floor(x_c - (half_w + r_in))))
            yi1 = max(0, int(math.floor(y_c - (half_h + r_in))))
            xi2 = min(pad_w, int(math.ceil(x_c + (half_w + r_in))))
            yi2 = min(pad_h, int(math.ceil(y_c + (half_h + r_in))))
            if xi2 > xi1 and yi2 > yi1:
                mask[b_id, 0, yi1:yi2, xi1:xi2] = 0.0

            # Target body rectangle (safety carve)
            xt1 = max(0, int(math.floor(x_c - half_w)))
            yt1 = max(0, int(math.floor(y_c - half_h)))
            xt2 = min(pad_w, int(math.ceil(x_c + half_w)))
            yt2 = min(pad_h, int(math.ceil(y_c + half_h)))
            if xt2 > xt1 and yt2 > yt1:
                mask[b_id, 0, yt1:yt2, xt1:xt2] = 0.0

    if inner_vals:
        stats = {
            "rlcp_adaptive_inner_mean": float(sum(inner_vals) / len(inner_vals)),
            "rlcp_adaptive_outer_mean": float(sum(outer_vals) / len(outer_vals)),
            "rlcp_adaptive_inner_min": float(min(inner_vals)),
            "rlcp_adaptive_outer_max": float(max(outer_vals)),
        }
    else:
        stats = {
            "rlcp_adaptive_inner_mean": 0.0,
            "rlcp_adaptive_outer_mean": 0.0,
            "rlcp_adaptive_inner_min": 0.0,
            "rlcp_adaptive_outer_max": 0.0,
        }

    return torch.clamp(mask, 0.0, 1.0), stats


def build_confounder_mask(
    local_ctx_mask: torch.Tensor,
    all_objects_mask: torch.Tensor,
    ring_mask: torch.Tensor,
) -> torch.Tensor:
    """M_conf = M_local_ctx * (1 - M_non_target_core) * (1 - M_ring)"""
    m = local_ctx_mask * (1.0 - all_objects_mask) * (1.0 - ring_mask)
    return torch.clamp(m, 0.0, 1.0)


def build_pag_gate(
    strict_gate_1d: torch.Tensor,
    target_scores: torch.Tensor,
    target_class_id: int,
    top_ratio: Union[float, Sequence] = 0.3,
    min_keep: Union[int, Sequence] = 8,
    layer_sizes: List[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    FPN-Aware PAG: 根据层级 (P3, P4, P5) 分别应用不同的比例与最小保留数。
    """
    empty_stats = {
        "pag_positive_ratio": 0.0, "pag_threshold": 0.0, "pag_mean_target_score": 0.0,
        "strict_positive": 0.0, "pag_positive": 0.0, "pag_fallback_count": 0.0, "layer_stats": []
    }
    
    if strict_gate_1d is None or strict_gate_1d.numel() == 0:
        return strict_gate_1d, empty_stats

    strict_gate = strict_gate_1d.bool()
    strict_total = int(strict_gate.sum().item())
    if strict_total <= 0:
        return strict_gate, empty_stats

    if not torch.is_tensor(target_scores) or target_scores.numel() == 0:
        empty_stats.update({"pag_positive_ratio": 1.0, "strict_positive": float(strict_total), 
                            "pag_positive": float(strict_total), "pag_fallback_count": float(strict_gate.shape[0])})
        return strict_gate, empty_stats

    if target_scores.ndim == 3:
        if target_scores.shape[-1] > int(target_class_id):
            score_t = target_scores[:, :, int(target_class_id)]
        else:
            score_t = target_scores.max(dim=-1).values
    elif target_scores.ndim == 2:
        score_t = target_scores
    else:
        score_t = torch.zeros_like(strict_gate, dtype=torch.float32)

    pag_gate = torch.zeros_like(strict_gate)
    thresholds = []
    mean_scores = []
    pag_total = 0
    fallback_count = 0
    layer_stats = []

    # 🚀 安全判断：是否使用分层 FPN-Aware PAG
    is_layer_wise = isinstance(top_ratio, Sequence) and not isinstance(top_ratio, (str, bytes))

    if is_layer_wise:
        assert layer_sizes is not None, "layer_sizes must be provided for layer-wise PAG."
        assert len(top_ratio) == len(layer_sizes), f"Mismatch: len(top_ratio)={len(top_ratio)} vs len(layer_sizes)={len(layer_sizes)}"
        assert sum(layer_sizes) == strict_gate.shape[1], f"Mismatch: sum(layer_sizes)={sum(layer_sizes)} vs num_anchors={strict_gate.shape[1]}"

        if isinstance(min_keep, Sequence) and not isinstance(min_keep, (str, bytes)):
            assert len(min_keep) == len(layer_sizes), "Mismatch in min_keep lengths"
            mk_list = [int(max(1, x)) for x in min_keep]
        else:
            mk_list = [int(max(1, min_keep))] * len(layer_sizes)

        tr_list = [float(max(0.01, min(x, 1.0))) for x in top_ratio]
        offset = 0

        for i, (l_size, l_ratio, l_mk) in enumerate(zip(layer_sizes, tr_list, mk_list)):
            slice_gate = strict_gate[:, offset : offset + l_size]
            slice_score = score_t[:, offset : offset + l_size]
            
            l_strict_total = 0
            l_pag_total = 0

            for b in range(strict_gate.shape[0]):
                idx = torch.nonzero(slice_gate[b], as_tuple=False).view(-1)
                n = int(idx.numel())
                l_strict_total += n
                
                if n <= 0:
                    continue
                    
                vals = slice_score[b, idx]
                mean_scores.append(float(vals.mean().item()))

                if n < l_mk:
                    pag_gate[b, offset + idx] = True
                    pag_total += n
                    l_pag_total += n
                    fallback_count += 1
                    continue

                k = int(max(1, math.ceil(n * l_ratio)))
                topv, topi = torch.topk(vals, k=k, largest=True)
                keep_idx = idx[topi]
                
                pag_gate[b, offset + keep_idx] = True
                pag_total += k
                l_pag_total += k
                thresholds.append(float(topv.min().item()))
                
            layer_stats.append({
                "layer_idx": i,
                "strict": l_strict_total,
                "pag": l_pag_total,
                "ratio": float(l_pag_total) / float(max(1, l_strict_total))
            })
            offset += l_size

    else:
        # 兼容单层全局比例
        global_ratio = float(max(0.01, min(top_ratio, 1.0)))
        global_mk = int(max(1, min_keep))

        for b in range(strict_gate.shape[0]):
            idx = torch.nonzero(strict_gate[b], as_tuple=False).view(-1)
            n = int(idx.numel())
            if n <= 0:
                continue
                
            vals = score_t[b, idx]
            mean_scores.append(float(vals.mean().item()))

            if n < global_mk:
                pag_gate[b, idx] = True
                pag_total += n
                fallback_count += 1
                continue

            k = int(max(1, math.ceil(n * global_ratio)))
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
        "pag_fallback_count": float(fallback_count),
        "layer_stats": layer_stats,
    }
