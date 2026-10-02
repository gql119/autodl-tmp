"""Shared REM/SEM adversarial *surrogate* training machinery."""

from __future__ import annotations

from typing import Sequence

import torch

from .core import eot_flip, target_detection_loss


def freeze_for_defender(model: torch.nn.Module) -> None:
    # Both official REM/SEM defenders use eval-mode, frozen surrogate weights.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_for_surrogate(model: torch.nn.Module) -> None:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def adversarial_ascent(model: torch.nn.Module, base_image: torch.Tensor,
                       annotations: Sequence[dict], support: torch.Tensor,
                       *, radius: float = 4.0 / 255.0, steps: int = 10,
                       step_size: float = 0.8 / 255.0, flip: bool = False,
                       imgsz: int = 640) -> torch.Tensor:
    """Inner PGD max for REM defender and both methods' surrogate update."""
    # Official REM/SEM robust training attacks the *whole image*. The shared
    # person support constrains the final defender poison only, not this
    # ephemeral adversarial inner variable.
    adv = torch.empty_like(base_image).uniform_(-radius, radius).detach()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        attacked = (base_image + adv).clamp(0, 1)
        attacked, attack_annotations = eot_flip(attacked, annotations, flip)
        loss, _ = target_detection_loss(model, attacked, attack_annotations, imgsz)
        grad = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = (adv + step_size * grad.sign()).clamp(-radius, radius)
    return adv.detach()


def adversarial_surrogate_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                               poisoned: torch.Tensor, annotations: Sequence[dict],
                               support: torch.Tensor, *, radius: float, steps: int,
                               step_size: float, imgsz: int = 640) -> float:
    """Official REM and SEM both train theta on adversarially attacked UE."""
    freeze_for_defender(model)
    adv = adversarial_ascent(model, poisoned.detach(), annotations, support,
                             radius=radius, steps=steps, step_size=step_size, imgsz=imgsz)
    unfreeze_for_surrogate(model)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = target_detection_loss(model, (poisoned.detach() + adv).clamp(0, 1), annotations, imgsz)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
