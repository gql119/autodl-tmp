from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GradientLeakageResult:
    matrix: torch.Tensor
    gradient_norms: Dict[str, float]
    effective_parameter_count: int
    none_gradient_count: Dict[str, int]
    per_layer_norms: Dict[str, Dict[str, float]]


def compute_gradient_leakage_matrix(
    losses: Dict[str, torch.Tensor],
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> GradientLeakageResult:
    names = ["protected", "authorized", "shared"]
    params = [(name, param) for name, param in named_parameters if param.requires_grad]
    flat_grads: Dict[str, torch.Tensor] = {}
    none_counts: Dict[str, int] = {}
    per_layer: Dict[str, Dict[str, float]] = {}

    for loss_name in names:
        grads = torch.autograd.grad(
            losses[loss_name],
            [param for _name, param in params],
            retain_graph=True,
            allow_unused=True,
        )
        pieces: List[torch.Tensor] = []
        none_count = 0
        per_layer[loss_name] = {}
        for (param_name, param), grad in zip(params, grads):
            if grad is None:
                none_count += 1
                grad = torch.zeros_like(param)
            per_layer[loss_name][param_name] = float(grad.detach().norm().item())
            pieces.append(grad.reshape(-1))
        flat = torch.cat(pieces) if pieces else torch.zeros(1)
        flat_grads[loss_name] = flat
        none_counts[loss_name] = none_count

    matrix = torch.empty((3, 3), dtype=flat_grads[names[0]].dtype, device=flat_grads[names[0]].device)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            matrix[i, j] = F.cosine_similarity(flat_grads[left], flat_grads[right], dim=0, eps=1.0e-12)

    norms = {name: float(flat_grads[name].detach().norm().item()) for name in names}
    effective = sum(1 for _name, param in params if param.numel() > 0)
    return GradientLeakageResult(matrix, norms, effective, none_counts, per_layer)
