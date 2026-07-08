from __future__ import annotations

import torch


def project_delta_linf(delta: torch.Tensor, eps: float) -> None:
    with torch.no_grad():
        delta.clamp_(min=-float(eps), max=float(eps))


def gain_selectivity(d_protected: float, e_authorized: float, e_shared: float) -> float:
    return float(d_protected) - float(e_authorized) - float(e_shared)
