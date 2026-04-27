import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def masked_prototype(
    feat: torch.Tensor,
    mask: torch.Tensor,
    min_pixels: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample masked prototype.
    feat: [B,C,H,W], mask: [B,1,H,W]
    returns:
      proto: [B,C]
      valid: [B] bool
    """
    if feat.ndim != 4 or mask.ndim != 4:
        raise ValueError(f"feat/mask dims mismatch: feat={feat.shape}, mask={mask.shape}")
    if feat.shape[0] != mask.shape[0] or feat.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"feat/mask shape mismatch: feat={feat.shape}, mask={mask.shape}")

    weight = mask.clamp(0.0, 1.0)
    denom = weight.sum(dim=(2, 3), keepdim=False).squeeze(1).clamp_min(1e-6)
    proto = (feat * weight).sum(dim=(2, 3)) / denom.unsqueeze(1)
    valid = denom >= float(min_pixels)
    return proto, valid


def robust_masked_prototype(
    feat: torch.Tensor,
    mask: torch.Tensor,
    trim_ratio: float = 0.1,
    min_pixels: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    RLCP robust prototype via trimmed mean:
    - collect tokens in mask
    - sort by L2 norm
    - trim top/bottom ratio
    - average middle tokens
    Returns:
      proto: [B, C]
      valid: [B] bool
      keep_ratio: [B] float, kept_tokens/valid_tokens
    """
    if feat.ndim != 4 or mask.ndim != 4:
        raise ValueError(f"feat/mask dims mismatch: feat={feat.shape}, mask={mask.shape}")
    if feat.shape[0] != mask.shape[0] or feat.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"feat/mask shape mismatch: feat={feat.shape}, mask={mask.shape}")

    bsz, ch, h, w = feat.shape
    trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
    proto = torch.zeros((bsz, ch), device=feat.device, dtype=feat.dtype)
    valid = torch.zeros((bsz,), device=feat.device, dtype=torch.bool)
    keep_ratio = torch.zeros((bsz,), device=feat.device, dtype=feat.dtype)

    feat_flat = feat.permute(0, 2, 3, 1).reshape(bsz, h * w, ch)
    mask_flat = (mask[:, 0] > 0.5).reshape(bsz, h * w)

    for b in range(bsz):
        idx = torch.nonzero(mask_flat[b], as_tuple=False).view(-1)
        n = int(idx.numel())
        if n < int(max(1.0, min_pixels)):
            continue

        tokens = feat_flat[b, idx, :]  # [N, C]
        norms = torch.norm(tokens, dim=1)
        sort_idx = torch.argsort(norms, dim=0)

        k_trim = int(math.floor(n * trim_ratio))
        if n - 2 * k_trim < 1:
            k_trim = 0

        if k_trim > 0:
            keep_idx = sort_idx[k_trim : n - k_trim]
        else:
            keep_idx = sort_idx

        if keep_idx.numel() <= 0:
            keep_idx = sort_idx

        kept = tokens[keep_idx]
        proto[b] = kept.mean(dim=0)
        valid[b] = True
        keep_ratio[b] = float(kept.shape[0]) / float(max(1, n))

    return proto, valid, keep_ratio


def compute_entangle_loss(
    z_adv: torch.Tensor,
    z_clean: torch.Tensor,
    z_conf: torch.Tensor,
    tau: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if valid_mask is None:
        valid_mask = torch.ones((z_adv.shape[0],), dtype=torch.bool, device=z_adv.device)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        zero = torch.zeros((), device=z_adv.device, dtype=z_adv.dtype)
        return zero, {"cos_t_conf": 0.0, "cos_t_clean": 0.0}

    adv = F.normalize(z_adv[valid_idx], dim=1)
    clean = F.normalize(z_clean[valid_idx].detach(), dim=1)
    conf = F.normalize(z_conf[valid_idx].detach(), dim=1)

    tau = max(float(tau), 1e-4)
    sim_pos = F.cosine_similarity(adv, conf, dim=1) / tau
    sim_neg = F.cosine_similarity(adv, clean, dim=1) / tau
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    stats = {
        "cos_t_conf": float((sim_pos * tau).mean().item()),
        "cos_t_clean": float((sim_neg * tau).mean().item()),
    }
    return loss, stats


def compute_anchor_losses(z_adv: torch.Tensor, z_clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    adv = z_adv
    clean = z_clean.detach()
    cos_sim = F.cosine_similarity(adv, clean, dim=1)
    l_cos = (1.0 - cos_sim).mean()
    adv_norm = adv.norm(dim=1)
    clean_norm = clean.norm(dim=1)
    l_energy = ((adv_norm - clean_norm) ** 2).mean()
    return l_cos, l_energy


def compute_collapse_loss(feat_adv: torch.Tensor, mask_al: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    proto, valid = masked_prototype(feat_adv, mask_al, min_pixels=1.0)
    if torch.count_nonzero(valid) == 0:
        zero = torch.zeros((), device=feat_adv.device, dtype=feat_adv.dtype)
        return zero, torch.zeros((feat_adv.shape[0],), device=feat_adv.device, dtype=feat_adv.dtype)

    proto_map = proto.unsqueeze(-1).unsqueeze(-1)
    sq = (feat_adv - proto_map.detach()) ** 2
    denom = mask_al.sum(dim=(2, 3), keepdim=False).squeeze(1).clamp_min(1e-6)
    spatial_var_c = (sq * mask_al).sum(dim=(2, 3)) / denom.unsqueeze(1)
    spatial_var = spatial_var_c.mean(dim=1)
    loss = spatial_var[valid].mean()
    return loss, spatial_var


def compute_alsi_score(
    z_adv: torch.Tensor,
    spatial_var: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    energy = z_adv.norm(dim=1) ** 2
    score = energy / spatial_var.clamp_min(1e-6)
    if valid_mask is None:
        valid_mask = torch.ones_like(score, dtype=torch.bool)
    valid_score = score[valid_mask]
    if valid_score.numel() == 0:
        return 0.0
    return float(valid_score.mean().item())


def compute_non_target_margin_preserve(
    clean_non_target_logits: torch.Tensor,
    adv_non_target_logits: torch.Tensor,
    use_smooth_l1: bool = True,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DSNP-lite margin preserve on non-target logits.
    🚀 修复：增加 valid_mask 支持，仅在真实的非目标前景 Anchor 上计算 Margin，防止海量背景 Anchor 稀释梯度。
    """
    if clean_non_target_logits.ndim != 3 or adv_non_target_logits.ndim != 3:
        zero = torch.zeros((), device=adv_non_target_logits.device, dtype=adv_non_target_logits.dtype)
        return zero, {"margin_clean_mean": 0.0, "margin_adv_mean": 0.0}

    if clean_non_target_logits.shape[-1] < 2:
        zero = torch.zeros((), device=adv_non_target_logits.device, dtype=adv_non_target_logits.dtype)
        return zero, {"margin_clean_mean": 0.0, "margin_adv_mean": 0.0}

    clean_top2 = torch.topk(clean_non_target_logits, k=2, dim=-1).values
    adv_top2 = torch.topk(adv_non_target_logits, k=2, dim=-1).values

    margin_clean = clean_top2[..., 0] - clean_top2[..., 1]
    margin_adv = adv_top2[..., 0] - adv_top2[..., 1]

    # 🚀 施加非目标前景门控过滤
    if valid_mask is not None:
        margin_clean = margin_clean[valid_mask]
        margin_adv = margin_adv[valid_mask]
        
        if margin_clean.numel() == 0:
            zero = torch.zeros((), device=adv_non_target_logits.device, dtype=adv_non_target_logits.dtype)
            return zero, {"margin_clean_mean": 0.0, "margin_adv_mean": 0.0}

    if use_smooth_l1:
        loss = F.smooth_l1_loss(margin_adv, margin_clean.detach())
    else:
        loss = F.l1_loss(margin_adv, margin_clean.detach())

    return loss, {
        "margin_clean_mean": float(margin_clean.mean().item()),
        "margin_adv_mean": float(margin_adv.mean().item()),
    }
