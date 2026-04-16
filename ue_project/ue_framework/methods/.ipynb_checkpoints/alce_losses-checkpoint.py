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
