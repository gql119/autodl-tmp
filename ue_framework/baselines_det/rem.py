"""REM-Det: adversarial inner maximization within defender minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .core import EPS, eot_flip, poison_image, project_raw_, support_tensor, target_detection_loss
from .robust_common import adversarial_ascent, adversarial_surrogate_step, freeze_for_defender


@dataclass(frozen=True)
class REMSchedule:
    defender_steps: int = 10
    defender_step_size: float = 1.6 / 255.0
    eot_samples: int = 5
    adversarial_radius: float = 4.0 / 255.0
    adversarial_steps: int = 10
    adversarial_step_size: float = 0.8 / 255.0
    surrogate_lr: float = 1e-4


def rem_update(model: torch.nn.Module, clean: torch.Tensor, annotations: Sequence[dict],
               raw_delta: torch.Tensor, surrogate_optimizer: torch.optim.Optimizer,
               schedule: REMSchedule, *, imgsz: int = 640, seed: int = 0,
               diagnose_fixed: bool = False) -> dict:
    """One official-style defender-regeneration plus adversarial theta update."""
    if min(schedule.defender_steps, schedule.eot_samples, schedule.adversarial_steps) < 1:
        raise ValueError("REM requires defender steps, EOT samples and adversarial PGD")
    support = support_tensor(clean, annotations)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    freeze_for_defender(model)
    def fixed_robust_loss(delta: torch.Tensor) -> float:
        # Paired, fixed-seed *diagnostic* of the same inner-max objective.
        # It is not part of the formal poison update and never uses val data.
        device_ids = [clean.device.index or 0] if clean.is_cuda else []
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(seed + 10000)
            poisoned_fixed = poison_image(clean, delta.detach(), support)
            values = []
            for sample_index in range(schedule.eot_samples):
                flip = bool(sample_index % 2)
                adversarial = adversarial_ascent(
                    model, poisoned_fixed.detach(), annotations, support,
                    radius=schedule.adversarial_radius,
                    steps=schedule.adversarial_steps,
                    step_size=schedule.adversarial_step_size, flip=flip,
                    imgsz=imgsz,
                )
                transformed, transformed_anns = eot_flip(
                    (poisoned_fixed + adversarial).clamp(0, 1), annotations, flip,
                )
                with torch.no_grad():
                    value, _ = target_detection_loss(model, transformed, transformed_anns, imgsz)
                values.append(value)
        return float(torch.stack(values).mean())

    fixed_before = fixed_robust_loss(raw_delta) if diagnose_fixed else None
    defender_losses, gradient_norms = [], []
    for _ in range(schedule.defender_steps):
        raw_delta = raw_delta.detach().requires_grad_(True)
        poisoned = poison_image(clean, raw_delta, support)
        robust_losses = []
        for _ in range(schedule.eot_samples):
            flip = bool(torch.randint(2, (1,), generator=rng).item())
            # The maximizing input is detached by Danskin's theorem; the
            # resulting robust loss still differentiates through the defender.
            adversarial = adversarial_ascent(
                model, poisoned.detach(), annotations, support,
                radius=schedule.adversarial_radius, steps=schedule.adversarial_steps,
                step_size=schedule.adversarial_step_size, flip=flip, imgsz=imgsz,
            )
            attacked, transformed_anns = eot_flip((poisoned + adversarial).clamp(0, 1), annotations, flip)
            loss, _ = target_detection_loss(model, attacked, transformed_anns, imgsz)
            robust_losses.append(loss)
        robust_loss = torch.stack(robust_losses).mean()
        gradient = torch.autograd.grad(robust_loss, raw_delta)[0]
        gradient_norms.append(float(gradient.norm().detach()))
        defender_losses.append(float(robust_loss.detach()))
        with torch.no_grad():
            raw_delta = raw_delta - schedule.defender_step_size * gradient.sign()
            project_raw_(raw_delta, clean, support)
    fixed_after = fixed_robust_loss(raw_delta) if diagnose_fixed else None
    poisoned = poison_image(clean, raw_delta.detach(), support)
    surrogate_loss = adversarial_surrogate_step(
        model, surrogate_optimizer, poisoned, annotations, support,
        radius=schedule.adversarial_radius, steps=schedule.adversarial_steps,
        step_size=schedule.adversarial_step_size, imgsz=imgsz,
    )
    return {"raw_delta": raw_delta.detach(), "defender_losses": defender_losses,
            "perturbation_gradient_norms": gradient_norms,
            "surrogate_adversarial_loss": surrogate_loss,
            "objective": "min_delta EOT[max_adversarial YOLO_person_GT_loss]; adversarial theta update",
            "inner_adversarial_calls": schedule.defender_steps * schedule.eot_samples + 1,
            "fixed_robust_loss_before": fixed_before,
            "fixed_robust_loss_after": fixed_after,
            "epsilon": EPS}
