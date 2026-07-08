from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch


@dataclass
class GradientVector:
    vector: torch.Tensor
    norm: torch.Tensor
    total_parameter_count: int
    effective_parameter_count: int
    numel: int


def extract_gradient_vector(
    loss: torch.Tensor,
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    create_graph: bool,
    retain_graph: bool,
    normalize_per_parameter: bool = False,
    allow_unused: bool = True,
    eps: float = 1.0e-8,
) -> GradientVector:
    params = [p for _name, p in named_parameters]
    if not params:
        raise ValueError("No parameters selected for gradient extraction.")

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    pieces: List[torch.Tensor] = []
    effective = 0
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
            continue
        effective += 1
        grad_piece = grad.reshape(-1)
        if normalize_per_parameter:
            grad_piece = grad_piece / grad_piece.norm().clamp_min(eps)
        pieces.append(grad_piece)

    vector = torch.cat(pieces) if pieces else loss.reshape(1) * 0.0
    return GradientVector(
        vector=vector,
        norm=vector.norm(),
        total_parameter_count=len(params),
        effective_parameter_count=effective,
        numel=int(vector.numel()),
    )
