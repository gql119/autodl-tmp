from __future__ import annotations

from typing import Dict, Mapping

import torch


def gradient_conflict_diagnostics(losses: Mapping[str, torch.Tensor], delta: torch.Tensor) -> Dict:
    grads: Dict[str, torch.Tensor] = {}
    rows = []
    for name, loss in losses.items():
        grad = torch.autograd.grad(loss, delta, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(delta)
        flat = grad.detach().reshape(-1)
        grads[name] = flat
        rows.append(
            {
                "component": name,
                "gradient_norm": float(flat.norm().item()),
                "zero_gradient_ratio": float((flat == 0).float().mean().item()),
            }
        )

    names = list(losses.keys())
    matrix = []
    for left in names:
        matrix_row = []
        for right in names:
            denom = grads[left].norm() * grads[right].norm()
            if float(denom.item()) == 0.0:
                value = 0.0 if left != right else 1.0
            else:
                value = float(torch.dot(grads[left], grads[right]).div(denom).item())
            matrix_row.append(value)
        matrix.append(matrix_row)
    return {"names": names, "matrix": matrix, "rows": rows}
