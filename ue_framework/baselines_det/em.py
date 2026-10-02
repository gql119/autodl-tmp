"""Sample-wise EM-Det: alternate surrogate descent and perturbation descent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .core import EPS, TARGET_ID, _xyxy, letterbox, poison_image, project_raw_, support_tensor, target_detection_loss


@dataclass(frozen=True)
class EMSchedule:
    train_steps_per_cycle: int = 10
    noise_steps_per_cycle: int = 20
    noise_step_size: float = 0.8 / 255.0
    surrogate_lr: float = 1e-4
    max_cycles: int = 3
    stop_error: float = 0.01


def detection_training_error(model: torch.nn.Module,
                             poisoned_train_samples: Sequence[tuple[torch.Tensor, Sequence[dict]]],
                             *, imgsz: int = 640, conf: float = 0.25,
                             iou_threshold: float = 0.5) -> float:
    """One-minus person-GT training recall, for EM's original stop role.

    This reads *training* samples only. It is deliberately a training-error
    proxy and never a clean-validation result or official AP.
    """
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.metrics import box_iou

    model.eval()
    missed, total = 0, 0
    with torch.no_grad():
        for image, annotations in poisoned_train_samples:
            padded, geometry = letterbox(image, imgsz)
            prediction = non_max_suppression(model(padded), conf_thres=conf,
                                              iou_thres=0.7, classes=[TARGET_ID],
                                              max_det=300)[0]
            predicted_boxes = prediction[:, :4]
            for ann in annotations:
                if int(ann["cls"]) != TARGET_ID:
                    continue
                x1, y1, x2, y2 = _xyxy(ann["bbox"], geometry.original_height,
                                        geometry.original_width)
                gt = torch.tensor([[
                    x1 * geometry.new_width / geometry.original_width + geometry.pad_left,
                    y1 * geometry.new_height / geometry.original_height + geometry.pad_top,
                    x2 * geometry.new_width / geometry.original_width + geometry.pad_left,
                    y2 * geometry.new_height / geometry.original_height + geometry.pad_top,
                ]], device=image.device, dtype=image.dtype)
                hit = predicted_boxes.numel() and bool((box_iou(gt, predicted_boxes) >= iou_threshold).any())
                total += 1
                missed += int(not hit)
    if total == 0:
        raise ValueError("No person GT in EM training-error set")
    return missed / total


def em_cycle(model: torch.nn.Module, clean: torch.Tensor, annotations: Sequence[dict],
             raw_delta: torch.Tensor, surrogate_optimizer: torch.optim.Optimizer,
             schedule: EMSchedule, *, imgsz: int = 640) -> dict:
    """One official-style alternating cycle on one sample (mini sanity unit).

    `raw_delta` persists across cycles; the model and its optimizer also persist.
    A caller handling the full sample set owns the per-sample delta table.
    """
    if schedule.train_steps_per_cycle < 1 or schedule.noise_steps_per_cycle < 1:
        raise ValueError("EM needs both surrogate and noise updates")
    support = support_tensor(clean, annotations)
    train_values = []
    model.train()
    for _ in range(schedule.train_steps_per_cycle):
        surrogate_optimizer.zero_grad(set_to_none=True)
        image = poison_image(clean, raw_delta.detach(), support).detach()
        loss, _ = target_detection_loss(model, image, annotations, imgsz)
        loss.backward()
        surrogate_optimizer.step()
        train_values.append(float(loss.detach()))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Match official EM: frozen surrogate in eval mode for delta search.
    model.eval()
    noise_values = []
    gradient_norms = []
    try:
        for _ in range(schedule.noise_steps_per_cycle):
            raw_delta = raw_delta.detach().requires_grad_(True)
            image = poison_image(clean, raw_delta, support)
            loss, _ = target_detection_loss(model, image, annotations, imgsz)
            grad = torch.autograd.grad(loss, raw_delta)[0]
            gradient_norms.append(float(grad.norm().detach()))
            with torch.no_grad():
                raw_delta = raw_delta - schedule.noise_step_size * grad.sign()
                project_raw_(raw_delta, clean, support)
            noise_values.append(float(loss.detach()))
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return {"raw_delta": raw_delta.detach(), "surrogate_train_losses": train_values,
            "noise_losses": noise_values, "perturbation_gradient_norms": gradient_norms,
            "objective": "min_theta min_delta YOLO_person_GT_loss", "epsilon": EPS}


def em_generate(model: torch.nn.Module, clean: torch.Tensor, annotations: Sequence[dict],
                schedule: EMSchedule, *, stop_error_fn: Callable[[torch.nn.Module], float] | None = None,
                imgsz: int = 640, initial_delta: torch.Tensor | None = None) -> dict:
    """Bounded sample-wise run; formal callers must supply detection stop error.

    The original classifier's training-error criterion cannot be equated with
    detector AP. Until a detection error function is pre-registered, calling
    this with more than one cycle is deliberately blocked.
    """
    if schedule.max_cycles > 1 and stop_error_fn is None:
        raise ValueError("Detection training-error stop function required for multi-cycle EM")
    delta = torch.zeros_like(clean) if initial_delta is None else initial_delta.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule.surrogate_lr)
    history = []
    for cycle in range(schedule.max_cycles):
        result = em_cycle(model, clean, annotations, delta, optimizer, schedule, imgsz=imgsz)
        delta = result.pop("raw_delta")
        error = None if stop_error_fn is None else float(stop_error_fn(model))
        history.append({"cycle": cycle, "training_detection_error": error, **result})
        if error is not None and error <= schedule.stop_error:
            break
    return {"raw_delta": delta, "cycles": history, "stopped_at_error_threshold":
            history[-1]["training_detection_error"] is not None
            and history[-1]["training_detection_error"] <= schedule.stop_error}
