from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn.functional as F

def renorm_yolo_bbox_after_padding(
    cx: float, cy: float, bw: float, bh: float, 
    orig_w: int, orig_h: int, pad_w: int, pad_h: int
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
    shape_layers: list, 
    features_cache: dict
) -> dict:
    """
    ACGT: Project 1D assignment gate back to 2D FPN multi-scale grids.
    Returns a dictionary mapping layer_name to its 2D mask [B, 1, H, W].
    """
    gate_dict = {}
    anchor_offset = 0
    for layer_name in shape_layers:
        if layer_name not in features_cache:
            continue
        Z = features_cache[layer_name]
        B, _, H, W = Z.shape
        n = H * W
        gate_1d = strict_gate_1d[:, anchor_offset : anchor_offset + n]
        gate_dict[layer_name] = gate_1d.view(B, 1, H, W).float()
        anchor_offset += n
    return gate_dict

def build_non_target_objects_mask(batch: Dict, pad_h: int, pad_w: int, target_class_id: int, device: torch.device) -> torch.Tensor:
    """
    ACGT: 只渲染非目标类别（Non-target）的 BBox 掩码。
    释放被 Target 自身庞大的 BBox 错误吞噬的 Local Context 空间。
    """
    B_feat = int(batch.get("batch_size", 1))
    M_non_target = torch.zeros((B_feat, 1, pad_h, pad_w), device=device)
    batch_idx = batch.get("batch_idx", torch.zeros(0, device=device))
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device))
    clss = batch.get("cls", torch.zeros((0, 1), device=device))
    
    for b_id in range(B_feat):
        valid_idx = (batch_idx == b_id)
        if valid_idx.sum() > 0:
            boxes = bboxes[valid_idx]
            classes = clss[valid_idx].squeeze(-1)
            for box, cls_id in zip(boxes, classes):
                # 🚀 核心改动：跳过 Target 本身的框，只遮挡真正的“其他物体”
                if int(cls_id.item()) == target_class_id:
                    continue
                
                cx, cy, bw, bh = box
                x1 = max(0, int((cx - bw/2) * pad_w))
                y1 = max(0, int((cy - bh/2) * pad_h))
                x2 = min(pad_w, int((cx + bw/2) * pad_w))
                y2 = min(pad_h, int((cy + bh/2) * pad_h))
                M_non_target[b_id, 0, y1:y2, x1:x2] = 1.0
    return M_non_target

def build_local_context_mask(inner_mask: torch.Tensor, r_inner: int = 12, r_outer: int = 28) -> torch.Tensor:
    """
    ACGT: 构建中距离外层环带 (Mid-range Outer Annulus)。
    放弃贴边，直接在 Target 周围寻找稍远但安全的共现上下文。
    """
    # 内圈膨胀 (跨过 Ring 的危险区)
    k_inner = r_inner * 2 + 1
    pad_inner = r_inner
    dilated_inner = F.max_pool2d(inner_mask, kernel_size=k_inner, stride=1, padding=pad_inner)
    
    # 外圈膨胀 (定义上下文边界)
    k_outer = r_outer * 2 + 1
    pad_outer = r_outer
    dilated_outer = F.max_pool2d(inner_mask, kernel_size=k_outer, stride=1, padding=pad_outer)
    
    # 🚀 核心改动：中距离环带 = 外圈 - 内圈
    local_ctx = torch.clamp(dilated_outer - dilated_inner, 0.0, 1.0)
    return local_ctx

def build_confounder_mask(
    local_ctx_mask: torch.Tensor, 
    all_objects_mask: torch.Tensor, 
    ring_mask: torch.Tensor, 
) -> torch.Tensor:
    """ALCE confounder mask: M_conf = M_local_ctx * (1 - M_all_objects) * (1 - M_ring)"""
    m = local_ctx_mask * (1.0 - all_objects_mask) * (1.0 - ring_mask)
    return torch.clamp(m, 0.0, 1.0)