from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch.func import functional_call

from .model import detection_loss


@dataclass
class VirtualTrajectory:
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    selected_names: tuple[str, ...]
    step_losses: list[float]
    parameter_delta_norms: list[float]
    step_times_seconds: list[float]
    mode: str
    first_order: bool


def select_parameter_names(
    model: torch.nn.Module,
    mode: str,
    selected_modules: Sequence[str] | None = None,
) -> tuple[str, ...]:
    names = tuple(name for name, _ in model.named_parameters())
    if mode == "head_only":
        selected = tuple(name for name in names if name.startswith("cls_head."))
    elif mode == "detection_head":
        selected = tuple(name for name in names if name.startswith(("cls_head.", "box_head.")))
    elif mode == "selected_modules":
        patterns = tuple(selected_modules or ())
        if not patterns:
            raise ValueError("selected_modules mode requires at least one module prefix")
        selected = tuple(name for name in names if name.startswith(patterns))
    elif mode == "full_model":
        selected = names
    else:
        raise ValueError(f"unknown virtual_update_mode: {mode}")
    if not selected:
        raise ValueError(f"virtual_update_mode selected no parameters: {mode}")
    return selected


def functional_forward(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    annotations: Iterable[Iterable[dict]],
) -> dict[str, torch.Tensor]:
    return functional_call(model, (dict(parameters), dict(buffers)), (images, annotations))


def virtual_update(
    model: torch.nn.Module,
    support_images: torch.Tensor,
    support_annotations: Iterable[Iterable[dict]],
    steps: int,
    learning_rate: float,
    mode: str = "head_only",
    selected_modules: Sequence[str] | None = None,
    first_order: bool = True,
) -> VirtualTrajectory:
    if int(steps) <= 0:
        raise ValueError("virtual update steps must be positive")
    selected_names = select_parameter_names(model, mode, selected_modules)
    base_parameters = {name: value.detach() for name, value in model.named_parameters()}
    parameters = {
        name: value.detach().clone().requires_grad_(name in selected_names)
        for name, value in base_parameters.items()
    }
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    losses: list[float] = []
    delta_norms: list[float] = []
    times: list[float] = []
    selected_set = set(selected_names)

    for step in range(int(steps)):
        if first_order and step > 0:
            parameters = {
                name: value.detach().requires_grad_(name in selected_set) if name in selected_set else value
                for name, value in parameters.items()
            }
        started = time.perf_counter()
        outputs = functional_forward(model, parameters, buffers, support_images, support_annotations)
        loss, _ = detection_loss(outputs)
        selected_values = [parameters[name] for name in selected_names]
        gradients = torch.autograd.grad(loss, selected_values, create_graph=True)
        updated = dict(parameters)
        squared_delta = loss.new_zeros(())
        for name, gradient in zip(selected_names, gradients):
            new_value = parameters[name] - float(learning_rate) * gradient
            squared_delta = squared_delta + (new_value - base_parameters[name]).square().sum()
            updated[name] = new_value
        parameters = updated
        losses.append(float(loss.detach().item()))
        delta_norms.append(float(squared_delta.detach().sqrt().item()))
        times.append(time.perf_counter() - started)
    return VirtualTrajectory(
        parameters=parameters,
        buffers=buffers,
        selected_names=selected_names,
        step_losses=losses,
        parameter_delta_norms=delta_norms,
        step_times_seconds=times,
        mode=mode,
        first_order=bool(first_order),
    )


def model_state_unchanged(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    return all(name in current and torch.equal(current[name], value) for name, value in before.items())

