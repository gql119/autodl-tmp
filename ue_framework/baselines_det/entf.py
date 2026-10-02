"""EntF-Push-Det: frozen clean reference -> clean person centroid -> ROI push."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .core import EPS, TARGET_ID, Letterbox, _xyxy, letterbox, poison_image, project_raw_, support_tensor


FEATURE_LAYER = "model.15"


@dataclass(frozen=True)
class EntFSchedule:
    steps: int = 300
    step_size: float = EPS * 0.05  # official tau=0.05 times epsilon
    feature_layer: str = FEATURE_LAYER
    init: str = "uniform"


class PersonROIFeatures:
    """GT-aligned, area-weighted ROI average on a fixed YOLO feature layer."""

    def __init__(self, reference: torch.nn.Module, layer: str = FEATURE_LAYER, imgsz: int = 640):
        named = dict(reference.named_modules())
        if layer not in named:
            raise ValueError(f"Feature layer not found: {layer}")
        self.reference, self.layer, self.imgsz = reference, layer, imgsz
        self._feature = None
        self._hook = named[layer].register_forward_hook(self._capture)
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def _capture(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError(f"{self.layer} must emit one NCHW feature tensor")
        self._feature = output

    def close(self) -> None:
        self._hook.remove()

    @staticmethod
    def _mask_for_box(box: Sequence[float], geometry: Letterbox,
                      feature: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = _xyxy(box, geometry.original_height, geometry.original_width)
        if x2 <= x1 or y2 <= y1:
            return feature.new_zeros((1, 1, *feature.shape[-2:]))
        lx1 = round(x1 * geometry.new_width / geometry.original_width) + geometry.pad_left
        lx2 = round(x2 * geometry.new_width / geometry.original_width) + geometry.pad_left
        ly1 = round(y1 * geometry.new_height / geometry.original_height) + geometry.pad_top
        ly2 = round(y2 * geometry.new_height / geometry.original_height) + geometry.pad_top
        mask = feature.new_zeros((1, 1, geometry.imgsz, geometry.imgsz))
        mask[:, :, ly1:ly2, lx1:lx2] = 1.0
        return F.interpolate(mask, size=feature.shape[-2:], mode="area")

    def __call__(self, image: torch.Tensor, annotations: Sequence[dict]) -> torch.Tensor:
        padded, geometry = letterbox(image, self.imgsz)
        self._feature = None
        _ = self.reference(padded)
        feature = self._feature
        if feature is None:
            raise RuntimeError(f"Forward hook did not observe {self.layer}")
        vectors = []
        for ann in annotations:
            if int(ann["cls"]) != TARGET_ID:
                continue
            mask = self._mask_for_box(ann["bbox"], geometry, feature)
            if float(mask.sum().detach()) <= 0:
                continue
            vectors.append((feature * mask).sum(dim=(2, 3))[0] / mask.sum())
        if not vectors:
            raise ValueError("No valid person ROI at the selected feature resolution")
        return torch.stack(vectors, dim=0)

    def identity(self, image: torch.Tensor, annotations: Sequence[dict]) -> dict:
        features = self(image, annotations)
        return {"module_name": self.layer,
                "feature_resolution": list(self._feature.shape[-2:]),
                "feature_dimension": int(features.shape[-1]),
                "roi_pooling": "area-interpolated GT rectangle mask, spatial weighted mean",
                "object_count": int(features.shape[0])}


def clean_person_centroid(extractor: PersonROIFeatures,
                          clean_samples: Iterable[tuple[torch.Tensor, Sequence[dict]]]) -> tuple[torch.Tensor, int]:
    """One centroid over all *training* person GT objects; never use val."""
    accumulator = None
    count = 0
    with torch.no_grad():
        for image, annotations in clean_samples:
            vectors = extractor(image, annotations)
            summed = vectors.sum(dim=0)
            accumulator = summed if accumulator is None else accumulator + summed
            count += len(vectors)
    if count == 0:
        raise ValueError("No train-split person GT objects for centroid")
    return (accumulator / count).detach(), count


def entf_push(extractor: PersonROIFeatures, clean: torch.Tensor, annotations: Sequence[dict],
              centroid: torch.Tensor, schedule: EntFSchedule, *, seed: int = 0) -> dict:
    """Minimize negative own-centroid MSE, as official EntF Push does."""
    if schedule.steps < 1 or schedule.step_size <= 0:
        raise ValueError("EntF needs positive Push iterations and step size")
    support = support_tensor(clean, annotations)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    delta = torch.empty(clean.shape, device=clean.device, dtype=clean.dtype).uniform_(
        -EPS, EPS, generator=generator)
    project_raw_(delta, clean, support)
    losses, gradient_norms = [], []
    for _ in range(schedule.steps):
        delta = delta.detach().requires_grad_(True)
        features = extractor(poison_image(clean, delta, support), annotations)
        push_loss = -F.mse_loss(features, centroid.detach().expand_as(features))
        grad = torch.autograd.grad(push_loss, delta)[0]
        losses.append(float(push_loss.detach()))
        gradient_norms.append(float(grad.norm().detach()))
        with torch.no_grad():
            delta = delta - schedule.step_size * grad.sign()
            project_raw_(delta, clean, support)
    return {"raw_delta": delta.detach(), "push_losses": losses,
            "perturbation_gradient_norms": gradient_norms,
            "objective": "min_delta -MSE(person_ROI_feature, clean_person_centroid)",
            "reference_frozen": True, "epsilon": EPS}
