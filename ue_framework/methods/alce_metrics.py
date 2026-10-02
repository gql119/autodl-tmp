from typing import Iterable

import torch


def compute_overlap_ratio(mask_al: torch.Tensor, mask_assign: torch.Tensor) -> float:
    den = float(mask_assign.sum().item()) + 1e-6
    return float(mask_al.sum().item()) / den


def compute_confounder_purity(mask_conf: torch.Tensor, mask_local_ctx: torch.Tensor) -> float:
    den = float(mask_local_ctx.sum().item()) + 1e-6
    return float(mask_conf.sum().item()) / den


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    seq = list(values)
    if not seq:
        return float(default)
    return float(sum(seq) / max(1, len(seq)))
