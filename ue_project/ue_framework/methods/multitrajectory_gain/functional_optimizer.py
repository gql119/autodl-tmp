from __future__ import annotations

from typing import Dict

import torch

from .trajectory_state import FunctionalOptimizerState


def init_functional_sgd_state(parameters: Dict[str, torch.Tensor]) -> FunctionalOptimizerState:
    return FunctionalOptimizerState({name: torch.zeros_like(param) for name, param in parameters.items()})


def functional_sgd_step(
    parameters: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    state: FunctionalOptimizerState,
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    create_graph: bool = True,
) -> tuple[Dict[str, torch.Tensor], FunctionalOptimizerState, Dict[str, torch.Tensor]]:
    names = list(parameters.keys())
    values = [parameters[name] for name in names]
    grads = torch.autograd.grad(
        loss,
        values,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )
    updated: Dict[str, torch.Tensor] = {}
    next_buffers: Dict[str, torch.Tensor] = {}
    used_grads: Dict[str, torch.Tensor] = {}
    for name, param, grad in zip(names, values, grads):
        if grad is None:
            grad = torch.zeros_like(param)
        if weight_decay:
            grad = grad + float(weight_decay) * param
        if momentum:
            prev = state.momentum_buffers.get(name)
            if prev is None:
                prev = torch.zeros_like(param)
            buf = float(momentum) * prev + grad
            step_grad = grad + float(momentum) * buf if nesterov else buf
        else:
            buf = torch.zeros_like(param)
            step_grad = grad
        updated[name] = param - float(learning_rate) * step_grad
        next_buffers[name] = buf
        used_grads[name] = step_grad
    return updated, FunctionalOptimizerState(next_buffers), used_grads


def clone_parameter_dict(parameters: Dict[str, torch.Tensor], detach: bool = True) -> Dict[str, torch.Tensor]:
    out = {}
    for name, param in parameters.items():
        value = param.detach().clone() if detach else param.clone()
        value.requires_grad_(True)
        out[name] = value
    return out
