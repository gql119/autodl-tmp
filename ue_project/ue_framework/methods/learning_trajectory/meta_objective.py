from __future__ import annotations

from typing import Dict

import torch


def build_meta_query_loss(
    protected_query_loss: torch.Tensor,
    authorized_query_loss: torch.Tensor,
    lambda_protected_query: float,
) -> Dict[str, torch.Tensor]:
    meta = authorized_query_loss - float(lambda_protected_query) * protected_query_loss
    return {
        "meta_loss": meta,
        "protected_query_loss": protected_query_loss,
        "authorized_query_loss": authorized_query_loss,
    }
