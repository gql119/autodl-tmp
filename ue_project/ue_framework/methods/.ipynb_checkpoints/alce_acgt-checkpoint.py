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

def build_all_objects_mask(batch: Dict, pad_h: int, pad_w: int, device: torch.device) -> torch.Tensor:
    """ACGT: Render all GT bounding boxes onto a global absolute mask [B, 1, H, W]."""
    B_feat = int(batch.get("batch_size", 1))
    M_all_objects = torch.zeros((B_feat, 1, pad_h, pad_w), device=device)
    batch_idx = batch.get("batch_idx", torch.zeros(0, device=device))
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device))
    
    for b_id in range(B_feat):
        valid_idx = (batch_idx == b_id)
        if valid_idx.sum() > 0:
            boxes = bboxes[valid_idx]
            for box in boxes:
                cx, cy, bw, bh = box
                x1 = max(0, int((cx - bw/2) * pad_w))
                y1 = max(0, int((cy - bh/2) * pad_h))
                x2 = min(pad_w, int((cx + bw/2) * pad_w))
                y2 = min(pad_h, int((cy + bh/2) * pad_h))
                M_all_objects[b_id, 0, y1:y2, x1:x2] = 1.0
    return M_all_objects

def build_local_context_mask(inner_mask: torch.Tensor, expand_ratio: float = 1.5, ring_width: int = 4) -> torch.Tensor:
    """
    ACGT: local context around target inner region.
    Uses pure PyTorch MaxPool2d to perform binary morphological dilation on GPU.
    """
    k = int(max(3, round((ring_width * max(1.0, float(expand_ratio))) * 2 + 1)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    
    dilated = F.max_pool2d(inner_mask, kernel_size=k, stride=1, padding=pad)
    local_ctx = torch.clamp(dilated - inner_mask, 0.0, 1.0)
    return local_ctx

def build_confounder_mask(
    local_ctx_mask: torch.Tensor, 
    all_objects_mask: torch.Tensor, 
    ring_mask: torch.Tensor, 
) -> torch.Tensor:
    """ALCE confounder mask: M_conf = M_local_ctx * (1 - M_all_objects) * (1 - M_ring)"""
    m = local_ctx_mask * (1.0 - all_objects_mask) * (1.0 - ring_mask)
    return torch.clamp(m, 0.0, 1.0)