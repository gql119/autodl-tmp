from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

try:
    from torch.func import functional_call as _functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call


@dataclass
class VirtualUpdateResult:
    updated_parameters: Dict[str, torch.Tensor]
    support_grad_norm: torch.Tensor
    update_norm: torch.Tensor


def snapshot_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def parameter_leak_max_abs_diff(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    for name, param in model.named_parameters():
        if name not in snapshot:
            continue
        diff = (param.detach() - snapshot[name].to(param.device)).abs().max()
        max_diff = max(max_diff, float(diff.item()))
    return max_diff


def make_virtual_parameters(
    model: nn.Module,
    selected_named_parameters: Iterable[Tuple[str, nn.Parameter]],
    support_loss: torch.Tensor,
    lr: float,
    create_graph: bool = True,
) -> VirtualUpdateResult:
    selected = list(selected_named_parameters)
    if not selected:
        raise ValueError("No parameters selected for virtual update.")
    names = [name for name, _param in selected]
    params = [param for _name, param in selected]
    grads = torch.autograd.grad(
        support_loss,
        params,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )

    updated: Dict[str, torch.Tensor] = {}
    grad_norm_sq = support_loss.new_zeros(())
    update_norm_sq = support_loss.new_zeros(())
    for name, param, grad in zip(names, params, grads):
        if grad is None:
            grad = torch.zeros_like(param)
        update = float(lr) * grad
        updated[name] = param - update
        grad_norm_sq = grad_norm_sq + grad.pow(2).sum()
        update_norm_sq = update_norm_sq + update.pow(2).sum()

    return VirtualUpdateResult(
        updated_parameters=updated,
        support_grad_norm=grad_norm_sq.sqrt(),
        update_norm=update_norm_sq.sqrt(),
    )


def functional_forward(model: nn.Module, updated_parameters: Dict[str, torch.Tensor], *args, **kwargs):
    state: Dict[str, torch.Tensor] = {}
    state.update(dict(model.named_parameters()))
    state.update(dict(model.named_buffers()))
    state.update(updated_parameters)
    try:
        return _functional_call(model, state, args, kwargs)
    except TypeError:
        return _functional_call(model, state, args)
