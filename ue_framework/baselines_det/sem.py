"""SEM-Det: sampled-uniform stability in defender, PGD for surrogate only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .core import EPS, eot_flip, poison_image, project_raw_, support_tensor, target_detection_loss
from .robust_common import adversarial_surrogate_step, freeze_for_defender


@dataclass(frozen=True)
class SEMSchedule:
    defender_steps: int = 10
    defender_step_size: float = 1.6 / 255.0
    stability_samples: int = 5
    stability_radius: float = 4.0 / 255.0
    surrogate_adversarial_steps: int = 10
    surrogate_adversarial_step_size: float = 0.8 / 255.0
    surrogate_lr: float = 1e-4


def sem_update(model: torch.nn.Module, clean: torch.Tensor, annotations: Sequence[dict],
               raw_delta: torch.Tensor, surrogate_optimizer: torch.optim.Optimizer,
               schedule: SEMSchedule, *, imgsz: int = 640, seed: int = 0,
               diagnose_fixed: bool = False) -> dict:
    if min(schedule.defender_steps, schedule.stability_samples,
           schedule.surrogate_adversarial_steps) < 1:
        raise ValueError("SEM needs defender sampling and adversarial surrogate training")
    support = support_tensor(clean, annotations)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    freeze_for_defender(model)
    def fixed_stability_loss(delta: torch.Tensor) -> float:
        fixed_rng = torch.Generator(device=clean.device).manual_seed(seed + 10000)
        values = []
        with torch.no_grad():
            poisoned_fixed = poison_image(clean, delta.detach(), support)
            for sample_index in range(schedule.stability_samples):
                stable_noise = torch.empty_like(clean).uniform_(
                    -schedule.stability_radius, schedule.stability_radius,
                    generator=fixed_rng)
                sampled, transformed_anns = eot_flip(
                    (poisoned_fixed + stable_noise).clamp(0, 1), annotations,
                    bool(sample_index % 2),
                )
                value, _ = target_detection_loss(model, sampled, transformed_anns, imgsz)
                values.append(value)
        return float(torch.stack(values).mean())

    fixed_before = fixed_stability_loss(raw_delta) if diagnose_fixed else None
    defender_losses, gradient_norms = [], []
    for _ in range(schedule.defender_steps):
        raw_delta = raw_delta.detach().requires_grad_(True)
        poisoned = poison_image(clean, raw_delta, support)
        sampled_losses = []
        for _ in range(schedule.stability_samples):
            flip = bool(torch.randint(2, (1,), generator=rng).item())
            # This is the defining distinction from REM: a uniform random
            # draw, with NO defender-side adversarial gradient ascent.
            stable_noise = torch.empty_like(clean).uniform_(
                -schedule.stability_radius, schedule.stability_radius)
            sampled, transformed_anns = eot_flip((poisoned + stable_noise).clamp(0, 1), annotations, flip)
            loss, _ = target_detection_loss(model, sampled, transformed_anns, imgsz)
            sampled_losses.append(loss)
        stable_loss = torch.stack(sampled_losses).mean()
        gradient = torch.autograd.grad(stable_loss, raw_delta)[0]
        gradient_norms.append(float(gradient.norm().detach()))
        defender_losses.append(float(stable_loss.detach()))
        with torch.no_grad():
            raw_delta = raw_delta - schedule.defender_step_size * gradient.sign()
            project_raw_(raw_delta, clean, support)
    fixed_after = fixed_stability_loss(raw_delta) if diagnose_fixed else None
    poisoned = poison_image(clean, raw_delta.detach(), support)
    surrogate_loss = adversarial_surrogate_step(
        model, surrogate_optimizer, poisoned, annotations, support,
        radius=schedule.stability_radius, steps=schedule.surrogate_adversarial_steps,
        step_size=schedule.surrogate_adversarial_step_size, imgsz=imgsz,
    )
    return {"raw_delta": raw_delta.detach(), "defender_losses": defender_losses,
            "perturbation_gradient_norms": gradient_norms,
            "surrogate_adversarial_loss": surrogate_loss,
            "objective": "min_delta EOT[E_uniform YOLO_person_GT_loss]; adversarial theta update",
            "defender_inner_adversarial_calls": 0, "stability_draws":
            schedule.defender_steps * schedule.stability_samples,
            "fixed_stability_loss_before": fixed_before,
            "fixed_stability_loss_after": fixed_after, "epsilon": EPS}
