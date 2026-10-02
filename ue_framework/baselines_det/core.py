"""Shared *protocol* only: GT geometry, detection loss, and spatial budget.

Method-specific optimization stays in em.py, rem.py, sem.py and entf.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


TARGET_ID = 14
EPS = 8.0 / 255.0
OVERLAP_SCALE = 0.5


class PersonOnlyBCE:
    """Ultralytics loss-compatible BCE with no gradient for other classes."""

    def __init__(self, target_id: int = TARGET_ID):
        self.target_id = target_id

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != 20 or targets.shape != logits.shape:
            raise ValueError("Expected VOC20 matched class logits and targets")
        # Preserve the criterion's [B, anchors, nc] return shape and its
        # person-class normalizer; non-target logits contribute exactly zero.
        person_loss = F.binary_cross_entropy_with_logits(
            logits[..., self.target_id:self.target_id + 1],
            targets[..., self.target_id:self.target_id + 1], reduction="none",
        )
        out = torch.zeros_like(logits)
        out[..., self.target_id:self.target_id + 1] = person_loss
        return out


def _xyxy(box: Sequence[float], height: int, width: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = map(float, box)
    x1 = max(0, min(width, int((cx - bw / 2.0) * width)))
    x2 = max(0, min(width, int(np.ceil((cx + bw / 2.0) * width))))
    y1 = max(0, min(height, int((cy - bh / 2.0) * height)))
    y2 = max(0, min(height, int(np.ceil((cy + bh / 2.0) * height))))
    return x1, y1, x2, y2


def support_mask(height: int, width: int, annotations: Sequence[dict]) -> np.ndarray:
    """A union of person GT rectangles, halved where any other GT overlaps."""
    person = np.zeros((height, width), dtype=bool)
    other = np.zeros_like(person)
    for ann in annotations:
        x1, y1, x2, y2 = _xyxy(ann["bbox"], height, width)
        if int(ann["cls"]) == TARGET_ID:
            person[y1:y2, x1:x2] = True
        else:
            other[y1:y2, x1:x2] = True
    result = person.astype(np.float32)
    result[person & other] = OVERLAP_SCALE
    return result


def support_tensor(image: torch.Tensor, annotations: Sequence[dict]) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected a single RGB image in NCHW")
    support = support_mask(image.shape[-2], image.shape[-1], annotations)
    if not np.any(support):
        raise ValueError("The poison objective requires at least one person GT")
    return torch.from_numpy(support).to(image.device, image.dtype)[None, None]


def project_delta(raw_delta: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project in the unweighted epsilon ball, then apply the shared support."""
    return raw_delta.clamp(-EPS, EPS) * support


def poison_image(clean: torch.Tensor, raw_delta: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return (clean + project_delta(raw_delta, support)).clamp(0.0, 1.0)


def project_raw_(raw_delta: torch.Tensor, clean: torch.Tensor, support: torch.Tensor) -> None:
    """Keep optimizer state feasible without changing the mandatory final rule."""
    with torch.no_grad():
        raw_delta.clamp_(-EPS, EPS)
        raw_delta.mul_((support > 0).to(raw_delta.dtype))
        # This is a pixel-box projection before export, not a new loss term.
        upper = ((1.0 - clean) / support.clamp_min(OVERLAP_SCALE)).clamp_min(0)
        lower = (-clean / support.clamp_min(OVERLAP_SCALE)).clamp_max(0)
        raw_delta.copy_(torch.minimum(torch.maximum(raw_delta, lower), upper))


@dataclass(frozen=True)
class Letterbox:
    original_height: int
    original_width: int
    new_height: int
    new_width: int
    pad_top: int
    pad_left: int
    imgsz: int


def letterbox(image: torch.Tensor, imgsz: int = 640) -> tuple[torch.Tensor, Letterbox]:
    h, w = image.shape[-2:]
    ratio = min(imgsz / h, imgsz / w)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    resized = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)
    pt, pl = (imgsz - nh) // 2, (imgsz - nw) // 2
    padded = F.pad(resized, (pl, imgsz - nw - pl, pt, imgsz - nh - pt), value=114.0 / 255.0)
    return padded, Letterbox(h, w, nh, nw, pt, pl, imgsz)


def person_batch(image: torch.Tensor, annotations: Sequence[dict], imgsz: int = 640) -> dict:
    """Build a one-image Ultralytics training batch with person GT only."""
    padded, transform = letterbox(image, imgsz)
    boxes = []
    for ann in annotations:
        if int(ann["cls"]) != TARGET_ID:
            continue
        x1, y1, x2, y2 = _xyxy(ann["bbox"], transform.original_height, transform.original_width)
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = x1 * transform.new_width / transform.original_width + transform.pad_left
        x2 = x2 * transform.new_width / transform.original_width + transform.pad_left
        y1 = y1 * transform.new_height / transform.original_height + transform.pad_top
        y2 = y2 * transform.new_height / transform.original_height + transform.pad_top
        boxes.append([(x1 + x2) / (2 * imgsz), (y1 + y2) / (2 * imgsz),
                      (x2 - x1) / imgsz, (y2 - y1) / imgsz])
    if not boxes:
        raise ValueError("No valid person GT box")
    n = len(boxes)
    return {
        "img": padded,
        "cls": torch.full((n, 1), TARGET_ID, dtype=torch.float32, device=image.device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=image.device),
        "batch_idx": torch.zeros(n, dtype=torch.float32, device=image.device),
    }


def target_detection_loss(model: torch.nn.Module, image: torch.Tensor,
                          annotations: Sequence[dict], imgsz: int = 640) -> tuple[torch.Tensor, torch.Tensor]:
    # Bare exported VOC checkpoints carry a small dict rather than the full
    # trainer namespace expected by v8DetectionLoss. Populate precisely the
    # locked Ultralytics training gains before its criterion is constructed.
    if isinstance(model.args, dict):
        from ultralytics.cfg import get_cfg
        model.args = get_cfg(overrides={**model.args, "box": 7.5, "cls": 0.5, "dfl": 1.5})
    if not getattr(model, "_person_only_criterion_configured", False):
        model.criterion = model.init_criterion()
        model.criterion.bce = PersonOnlyBCE(TARGET_ID)
        model._person_only_criterion_configured = True
    batch = person_batch(image, annotations, imgsz)
    total, components = model(batch)
    # Ultralytics 8.4.150 returns the already weighted (box, cls, dfl)
    # vector here. Its scalar training objective is their sum.
    total = total.sum()
    component_values = components.values() if isinstance(components, dict) else (components,)
    if not torch.isfinite(total).all() or not all(
        torch.isfinite(value).all() for value in component_values if isinstance(value, torch.Tensor)
    ):
        raise FloatingPointError("Non-finite YOLO target-conditioned detection loss")
    return total, components


def flip_annotations(annotations: Sequence[dict]) -> list[dict]:
    return [{**ann, "bbox": [1.0 - float(ann["bbox"][0]), *ann["bbox"][1:]]} for ann in annotations]


def eot_flip(image: torch.Tensor, annotations: Sequence[dict], flip: bool) -> tuple[torch.Tensor, Sequence[dict]]:
    return (image.flip(-1), flip_annotations(annotations)) if flip else (image, annotations)


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_audit(clean: np.ndarray, poisoned: np.ndarray,
                        annotations: Sequence[dict]) -> dict:
    """Both arrays are uint8 RGB from the original and materialized PNG."""
    if clean.dtype != np.uint8 or poisoned.dtype != np.uint8 or clean.shape != poisoned.shape:
        raise ValueError("Expected same-shaped decoded uint8 images")
    support = support_mask(clean.shape[0], clean.shape[1], annotations)
    diff = np.abs(poisoned.astype(np.int16) - clean.astype(np.int16))
    region_max = {
        "target_only": int(diff[support == 1].max(initial=0)),
        "overlap": int(diff[support == 0.5].max(initial=0)),
        "outside": int(diff[support == 0].max(initial=0)),
    }
    if region_max["target_only"] > 8 or region_max["overlap"] > 4 or region_max["outside"] != 0:
        raise RuntimeError(f"Decoded-pixel spatial budget failed: {region_max}")
    return {"max_abs_delta_byte": region_max,
            "changed_target_pixels": int(np.any(diff[support > 0] > 0, axis=-1).sum()),
            "changed_outside_pixels": int(np.any(diff[support == 0] > 0, axis=-1).sum())}
