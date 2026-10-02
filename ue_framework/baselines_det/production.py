"""Batched, full-schedule detection adaptations for the four UE baselines.

The public single-image kernels remain the reference for the method equations.
This module batches independent images for throughput while retaining every
method-internal defender, EOT, PGD, and Push iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .core import (
    EPS, TARGET_ID, Letterbox, _xyxy, flip_annotations, letterbox,
    poison_image, project_raw_, support_mask, target_detection_loss,
)


@dataclass
class BatchItem:
    clean: torch.Tensor
    annotations: list[dict]
    support: torch.Tensor
    raw_delta: torch.Tensor
    stem: str


def make_item(clean: torch.Tensor, annotations: Sequence[dict], stem: str) -> BatchItem:
    if clean.ndim != 4 or clean.shape[0] != 1 or clean.shape[1] != 3:
        raise ValueError("Each item must be one RGB NCHW image")
    anns = list(annotations)
    support_np = support_mask(clean.shape[-2], clean.shape[-1], anns)
    if not np.any(support_np):
        raise ValueError(f"No person support for {stem}")
    support = torch.from_numpy(support_np).to(clean.device, clean.dtype)[None, None]
    return BatchItem(clean, anns, support, torch.zeros_like(clean), stem)


def _letterbox_with_geometry(image: torch.Tensor, imgsz: int) -> tuple[torch.Tensor, Letterbox]:
    return letterbox(image, imgsz)


def _batch_dict(images: Sequence[torch.Tensor], annotations: Sequence[Sequence[dict]],
                flips: Sequence[bool] | None = None, imgsz: int = 640) -> dict:
    if not images or len(images) != len(annotations):
        raise ValueError("Expected equally sized non-empty image and annotation batches")
    if flips is None:
        flips = [False] * len(images)
    if len(flips) != len(images):
        raise ValueError("Flip flags must match the image batch")
    padded_images, all_cls, all_boxes, all_batch_idx = [], [], [], []
    for batch_idx, (image, anns, flip) in enumerate(zip(images, annotations, flips)):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Batched helper expects each image to retain its own leading singleton")
        if flip:
            image = image.flip(-1)
            anns = flip_annotations(anns)
        padded, geometry = _letterbox_with_geometry(image, imgsz)
        padded_images.append(padded)
        boxes = []
        for ann in anns:
            if int(ann["cls"]) != TARGET_ID:
                continue
            x1, y1, x2, y2 = _xyxy(ann["bbox"], geometry.original_height, geometry.original_width)
            if x2 <= x1 or y2 <= y1:
                continue
            x1 = x1 * geometry.new_width / geometry.original_width + geometry.pad_left
            x2 = x2 * geometry.new_width / geometry.original_width + geometry.pad_left
            y1 = y1 * geometry.new_height / geometry.original_height + geometry.pad_top
            y2 = y2 * geometry.new_height / geometry.original_height + geometry.pad_top
            boxes.append([(x1 + x2) / (2 * imgsz), (y1 + y2) / (2 * imgsz),
                          (x2 - x1) / imgsz, (y2 - y1) / imgsz])
        if not boxes:
            raise ValueError("Every batch item must contain a valid person box")
        all_cls.append(torch.full((len(boxes), 1), TARGET_ID, dtype=torch.float32,
                                  device=image.device))
        all_boxes.append(torch.tensor(boxes, dtype=torch.float32, device=image.device))
        all_batch_idx.append(torch.full((len(boxes),), batch_idx, dtype=torch.float32,
                                        device=image.device))
    return {"img": torch.cat(padded_images, dim=0), "cls": torch.cat(all_cls),
            "bboxes": torch.cat(all_boxes), "batch_idx": torch.cat(all_batch_idx)}


def _loss_on_batch(model: torch.nn.Module, images: Sequence[torch.Tensor],
                   annotations: Sequence[Sequence[dict]], flips: Sequence[bool] | None,
                   imgsz: int) -> torch.Tensor:
    # Configure exactly the shared person-only BCE used by the single-image
    # kernels, then let Ultralytics' canonical detector criterion assign boxes.
    batch = _batch_dict(images, annotations, flips, imgsz)
    if isinstance(model.args, dict):
        from ultralytics.cfg import get_cfg
        model.args = get_cfg(overrides={**model.args, "box": 7.5, "cls": 0.5, "dfl": 1.5})
    if not getattr(model, "_person_only_criterion_configured", False):
        from .core import PersonOnlyBCE
        model.criterion = model.init_criterion()
        model.criterion.bce = PersonOnlyBCE(TARGET_ID)
        model._person_only_criterion_configured = True
    total, components = model(batch)
    loss = total.sum() / len(images)
    values = components.values() if isinstance(components, dict) else (components,)
    if not torch.isfinite(loss).all() or not all(
        torch.isfinite(value).all() for value in values if isinstance(value, torch.Tensor)
    ):
        raise FloatingPointError("Non-finite batched person-only YOLO loss")
    return loss


def _project_items(deltas: Sequence[torch.Tensor], items: Sequence[BatchItem]) -> list[torch.Tensor]:
    projected = []
    for delta, item in zip(deltas, items):
        value = delta.detach()
        project_raw_(value, item.clean, item.support)
        projected.append(value)
    return projected


def _poisoned(items: Sequence[BatchItem], deltas: Sequence[torch.Tensor] | None = None) -> list[torch.Tensor]:
    if deltas is None:
        deltas = [item.raw_delta for item in items]
    return [poison_image(item.clean, delta, item.support) for item, delta in zip(items, deltas)]


def _random_flips(count: int, generator: torch.Generator, device: torch.device) -> list[bool]:
    return [bool(x) for x in torch.randint(2, (count,), generator=generator, device=device).tolist()]


def _attack_batch(model: torch.nn.Module, bases: Sequence[torch.Tensor],
                  annotations: Sequence[Sequence[dict]], *, radius: float, steps: int,
                  step_size: float, flips: Sequence[bool] | None = None,
                  imgsz: int = 640, counters: dict | None = None) -> list[torch.Tensor]:
    if steps < 1:
        raise ValueError("PGD steps must be positive")
    adversarial = [torch.empty_like(base).uniform_(-radius, radius).detach() for base in bases]
    for _ in range(steps):
        adversarial = [x.detach().requires_grad_(True) for x in adversarial]
        attacked = [(base + adv).clamp(0.0, 1.0) for base, adv in zip(bases, adversarial)]
        loss = _loss_on_batch(model, attacked, annotations, flips, imgsz)
        grads = torch.autograd.grad(loss, tuple(adversarial), only_inputs=True)
        if counters is not None:
            counters["forward_calls"] += 1
            counters["backward_calls"] += 1
            counters["pgd_steps"] += len(bases)
        with torch.no_grad():
            adversarial = [(adv + step_size * grad.sign()).clamp(-radius, radius)
                           for adv, grad in zip(adversarial, grads)]
    return [x.detach() for x in adversarial]


def _surrogate_adversarial_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                items: Sequence[BatchItem], *, radius: float, steps: int,
                                step_size: float, imgsz: int, counters: dict,
                                flips: Sequence[bool] | None = None) -> float:
    bases = [x.detach() for x in _poisoned(items)]
    adv = _attack_batch(model, bases, [x.annotations for x in items], radius=radius,
                        steps=steps, step_size=step_size, flips=flips, imgsz=imgsz,
                        counters=counters)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = _loss_on_batch(model, [(base + delta).clamp(0.0, 1.0)
                                  for base, delta in zip(bases, adv)],
                          [x.annotations for x in items], flips, imgsz)
    loss.backward()
    optimizer.step()
    counters["forward_calls"] += 1
    counters["backward_calls"] += 1
    counters["optimizer_steps"] += 1
    return float(loss.detach())


def em_noise_batch(model: torch.nn.Module, items: Sequence[BatchItem], *, steps: int,
                   step_size: float, imgsz: int, counters: dict,
                   loss_trace: list[float] | None = None) -> list[torch.Tensor]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    deltas = [x.raw_delta.detach().requires_grad_(True) for x in items]
    losses = []
    for _ in range(steps):
        current = _poisoned(items, deltas)
        loss = _loss_on_batch(model, current, [x.annotations for x in items], None, imgsz)
        grads = torch.autograd.grad(loss, tuple(deltas), only_inputs=True)
        counters["forward_calls"] += 1
        counters["backward_calls"] += 1
        counters["noise_steps"] += len(items)
        losses.append(float(loss.detach()))
        if loss_trace is not None:
            loss_trace.append(float(loss.detach()))
        with torch.no_grad():
            deltas = [(d - step_size * g.sign()).detach().requires_grad_(True)
                      for d, g in zip(deltas, grads)]
            deltas = [d.detach() for d in _project_items(deltas, items)]
            deltas = [d.requires_grad_(True) for d in deltas]
    return [x.detach() for x in deltas]


def em_train_batch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   items: Sequence[BatchItem], *, imgsz: int, counters: dict) -> float:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = _loss_on_batch(model, _poisoned(items), [x.annotations for x in items], None, imgsz)
    loss.backward()
    optimizer.step()
    counters["forward_calls"] += 1
    counters["backward_calls"] += 1
    counters["optimizer_steps"] += 1
    return float(loss.detach())


def rem_defender_batch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       items: Sequence[BatchItem], *,
                       defender_steps: int = 10, eot_samples: int = 5,
                       pgd_steps: int = 10, defender_step_size: float = 1.6 / 255.0,
                       adversarial_radius: float = 4.0 / 255.0,
                       adversarial_step_size: float = 0.8 / 255.0,
                       imgsz: int = 640, seed: int = 0, counters: dict,
                       loss_trace: list[float] | None = None) -> list[torch.Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=items[0].clean.device).manual_seed(seed)
    deltas = [x.raw_delta.detach() for x in items]
    for _ in range(defender_steps):
        deltas = [x.detach().requires_grad_(True) for x in deltas]
        gradient_sum = [torch.zeros_like(x) for x in deltas]
        for _ in range(eot_samples):
            # Rebuild the poison graph per EOT draw so only one detector
            # activation graph is resident at a time. The mean of these
            # gradients is exactly the gradient of the sampled mean loss.
            poisoned = _poisoned(items, deltas)
            flips = _random_flips(len(items), generator, items[0].clean.device)
            adv = _attack_batch(model, [x.detach() for x in poisoned],
                                [x.annotations for x in items], radius=adversarial_radius,
                                steps=pgd_steps, step_size=adversarial_step_size,
                                flips=flips, imgsz=imgsz, counters=counters)
            # Danskin: adversarial maximizers are detached, but the robust
            # objective still differentiates through the defender poison.
            attacked = [(base + a).clamp(0.0, 1.0) for base, a in zip(poisoned, adv)]
            robust_sample = _loss_on_batch(model, attacked,
                                           [x.annotations for x in items], flips, imgsz)
            if loss_trace is not None:
                loss_trace.append(float(robust_sample.detach()))
            sample_grads = torch.autograd.grad(robust_sample, tuple(deltas), only_inputs=True)
            gradient_sum = [acc + grad for acc, grad in zip(gradient_sum, sample_grads)]
            counters["forward_calls"] += 1
            counters["backward_calls"] += 1
            counters["eot_samples"] += len(items)
        grads = [grad / eot_samples for grad in gradient_sum]
        with torch.no_grad():
            deltas = [(d - defender_step_size * g.sign()).detach() for d, g in zip(deltas, grads)]
            deltas = _project_items(deltas, items)
    # After the defender minimizes the robust loss, train theta adversarially.
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    defended_items = [BatchItem(x.clean, x.annotations, x.support, d, x.stem)
                      for x, d in zip(items, deltas)]
    surrogate_loss = _surrogate_adversarial_step(
        model, optimizer, defended_items, radius=adversarial_radius, steps=pgd_steps,
        step_size=adversarial_step_size, imgsz=imgsz, counters=counters)
    if loss_trace is not None:
        loss_trace.append(float(surrogate_loss))
    return [x.detach() for x in deltas], surrogate_loss


def sem_defender_batch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       items: Sequence[BatchItem], *,
                       defender_steps: int = 10, stability_samples: int = 5,
                       radius: float = 4.0 / 255.0,
                       defender_step_size: float = 1.6 / 255.0,
                       surrogate_pgd_steps: int = 10,
                       imgsz: int = 640, seed: int = 0, counters: dict,
                       loss_trace: list[float] | None = None) -> list[torch.Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=items[0].clean.device).manual_seed(seed)
    deltas = [x.raw_delta.detach() for x in items]
    for _ in range(defender_steps):
        deltas = [x.detach().requires_grad_(True) for x in deltas]
        gradient_sum = [torch.zeros_like(x) for x in deltas]
        for _ in range(stability_samples):
            poisoned = _poisoned(items, deltas)
            flips = _random_flips(len(items), generator, items[0].clean.device)
            sampled = [(base + torch.empty_like(base).uniform_(-radius, radius,
                        generator=generator)).clamp(0.0, 1.0) for base in poisoned]
            stable_loss = _loss_on_batch(model, sampled, [x.annotations for x in items], flips, imgsz)
            if loss_trace is not None:
                loss_trace.append(float(stable_loss.detach()))
            sample_grads = torch.autograd.grad(stable_loss, tuple(deltas), only_inputs=True)
            gradient_sum = [acc + grad for acc, grad in zip(gradient_sum, sample_grads)]
            counters["forward_calls"] += 1
            counters["backward_calls"] += 1
            counters["stability_samples"] += len(items)
        grads = [grad / stability_samples for grad in gradient_sum]
        with torch.no_grad():
            deltas = [(d - defender_step_size * g.sign()).detach() for d, g in zip(deltas, grads)]
            deltas = _project_items(deltas, items)
    defended_items = [BatchItem(x.clean, x.annotations, x.support, d, x.stem)
                      for x, d in zip(items, deltas)]
    surrogate_loss = _surrogate_adversarial_step(
        model, optimizer, defended_items, radius=radius, steps=surrogate_pgd_steps,
        step_size=0.8 / 255.0, imgsz=imgsz, counters=counters)
    if loss_trace is not None:
        loss_trace.append(float(surrogate_loss))
    return [x.detach() for x in deltas], surrogate_loss


class BatchedPersonROIFeatures:
    """Frozen model. Returns one matrix of person ROI vectors per image."""

    def __init__(self, reference: torch.nn.Module, layer: str = "model.15", imgsz: int = 640):
        modules = dict(reference.named_modules())
        if layer not in modules:
            raise ValueError(f"Feature layer {layer!r} was not found")
        self.reference, self.layer, self.imgsz = reference, layer, imgsz
        self.feature = None
        self.hook = modules[layer].register_forward_hook(self._capture)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    def _capture(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("ROI hook must capture an NCHW feature tensor")
        self.feature = output

    def close(self):
        self.hook.remove()

    def __call__(self, images: Sequence[torch.Tensor],
                 annotations: Sequence[Sequence[dict]]) -> list[torch.Tensor]:
        batch = _batch_dict(images, annotations, None, self.imgsz)
        self.feature = None
        _ = self.reference(batch["img"])
        feature = self.feature
        if feature is None or feature.shape[0] != len(images):
            raise RuntimeError("ROI feature batch mismatch")
        result = []
        for image_idx, (image, anns) in enumerate(zip(images, annotations)):
            # Recover the same letterbox geometry used by _batch_dict.
            _, geometry = letterbox(image, self.imgsz)
            vectors = []
            for ann in anns:
                if int(ann["cls"]) != TARGET_ID:
                    continue
                x1, y1, x2, y2 = _xyxy(ann["bbox"], geometry.original_height,
                                        geometry.original_width)
                if x2 <= x1 or y2 <= y1:
                    continue
                lx1 = round(x1 * geometry.new_width / geometry.original_width) + geometry.pad_left
                lx2 = round(x2 * geometry.new_width / geometry.original_width) + geometry.pad_left
                ly1 = round(y1 * geometry.new_height / geometry.original_height) + geometry.pad_top
                ly2 = round(y2 * geometry.new_height / geometry.original_height) + geometry.pad_top
                mask = feature.new_zeros((1, 1, self.imgsz, self.imgsz))
                mask[:, :, ly1:ly2, lx1:lx2] = 1.0
                mask = F.interpolate(mask, size=feature.shape[-2:], mode="area")
                denom = mask.sum()
                if float(denom.detach()) > 0:
                    vectors.append((feature[image_idx:image_idx + 1] * mask).sum((2, 3))[0] / denom)
            if not vectors:
                raise ValueError("A batch sample had no valid person ROI")
            result.append(torch.stack(vectors))
        return result


def entf_centroid_batch(extractor: BatchedPersonROIFeatures,
                        batches: Iterable[Sequence[BatchItem]]) -> tuple[torch.Tensor, int, int]:
    accumulated = None
    object_count = 0
    batch_count = 0
    with torch.no_grad():
        for items in batches:
            vectors = extractor([x.clean for x in items], [x.annotations for x in items])
            for matrix in vectors:
                val = matrix.sum(0)
                accumulated = val if accumulated is None else accumulated + val
                object_count += matrix.shape[0]
            batch_count += 1
    if accumulated is None or object_count == 0:
        raise ValueError("No person objects for EntF centroid")
    return accumulated.div(object_count).detach(), object_count, batch_count


def entf_push_batch(extractor: BatchedPersonROIFeatures, items: Sequence[BatchItem],
                    centroid: torch.Tensor, *, steps: int = 300,
                    step_size: float = EPS * 0.05, seed: int = 0,
                    counters: dict, loss_trace: list[float] | None = None,
                    initial_deltas: Sequence[torch.Tensor] | None = None) -> list[torch.Tensor]:
    if initial_deltas is None:
        generator = torch.Generator(device=items[0].clean.device).manual_seed(seed)
        deltas = [torch.empty_like(x.clean).uniform_(-EPS, EPS, generator=generator) for x in items]
        deltas = _project_items(deltas, items)
    else:
        if len(initial_deltas) != len(items):
            raise ValueError("EntF continuation requires one delta per item")
        deltas = [value.detach() for value in initial_deltas]
        deltas = _project_items(deltas, items)
    for _ in range(steps):
        deltas = [x.detach().requires_grad_(True) for x in deltas]
        poisoned = _poisoned(items, deltas)
        vectors = extractor(poisoned, [x.annotations for x in items])
        per_image = [-F.mse_loss(v, centroid.detach().expand_as(v)) for v in vectors]
        loss = torch.stack(per_image).mean()
        if loss_trace is not None:
            loss_trace.append(float(loss.detach()))
        grads = torch.autograd.grad(loss, tuple(deltas), only_inputs=True)
        counters["forward_calls"] += 1
        counters["backward_calls"] += 1
        counters["push_steps"] += len(items)
        with torch.no_grad():
            deltas = [(d - step_size * g.sign()).detach() for d, g in zip(deltas, grads)]
            deltas = _project_items(deltas, items)
    return [x.detach() for x in deltas]


def new_counters() -> dict[str, int]:
    return {"forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
            "pgd_steps": 0, "noise_steps": 0, "eot_samples": 0,
            "stability_samples": 0, "push_steps": 0}
