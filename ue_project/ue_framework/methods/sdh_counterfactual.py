from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data_utils import label_path_for_image, read_yolo_annotations
from ..ultra.hijacked_loss import HijackedV8Loss
from .sdh_mechanism import SDHBatch
from .semantic_hiding_carrier import SemanticHidingCarrier, render_person_box_carrier


def deterministic_person_audit_subset(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    target_class_id: int,
    per_stratum: int = 32,
) -> Tuple[Tuple[Path, ...], str]:
    """Freeze an equal person-only/person-cooccur subset without model feedback."""

    person_only = []
    person_cooccur = []
    for image_path in sorted((Path(value) for value in image_paths), key=lambda p: p.name):
        annotations = read_yolo_annotations(
            label_path_for_image(str(image_path), str(label_dir))
        )
        classes = {int(item["cls"]) for item in annotations}
        if int(target_class_id) not in classes:
            continue
        destination = person_cooccur if classes.difference({int(target_class_id)}) else person_only
        destination.append(image_path)
    if len(person_only) < per_stratum or len(person_cooccur) < per_stratum:
        raise ValueError("Person audit subset lacks a complete balanced stratum.")
    selected = tuple(person_only[:per_stratum] + person_cooccur[:per_stratum])
    digest = hashlib.sha256()
    digest.update(("target=%d;per_stratum=%d\n" % (target_class_id, per_stratum)).encode("ascii"))
    for path in selected:
        digest.update((path.name + "\n").encode("utf-8"))
    return selected, digest.hexdigest()


def _decoded_predictions(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        decoded = output
    elif isinstance(output, (tuple, list)):
        decoded = next(
            (
                value
                for value in output
                if torch.is_tensor(value) and value.ndim == 3
            ),
            None,
        )
        if decoded is None:
            raise RuntimeError("Victim output has no decoded [B,C,N] tensor.")
    else:
        raise RuntimeError("Victim output has an unsupported type.")
    if decoded.ndim != 3:
        raise RuntimeError("Decoded victim output must be [B,C,N].")
    return decoded


def fixed_person_classification_pair_losses(
    model: torch.nn.Module,
    batch: SDHBatch,
    carrier: SemanticHidingCarrier,
    secret: torch.Tensor,
    *,
    target_class_id: int = 14,
    num_classes: int = 20,
) -> Dict[str, float]:
    """Compare two views under one clean real-TAL person assignment, read-only."""

    hijacked = HijackedV8Loss.from_surrogate(
        model, num_classes=num_classes, target_class_id=target_class_id
    )
    with torch.no_grad():
        clean_decoded = _decoded_predictions(model(batch.images))
        hijacked.last_real_assign = {}
        hijacked.get_assigned_targets_and_loss(clean_decoded, batch.yolo_batch)
        assignment = hijacked.last_real_assign
        required = ("fg_mask", "target_labels", "target_scores")
        if any(not torch.is_tensor(assignment.get(name)) for name in required):
            raise RuntimeError("Learning-preference audit requires complete clean real TAL.")
        rendered = render_person_box_carrier(
            batch.images, batch.boxes_by_image, carrier, secret
        )
        carrier_decoded = _decoded_predictions(model(rendered.poisoned))

    labels = assignment["target_labels"].long()
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    foreground = assignment["fg_mask"].bool()
    if foreground.ndim == 3 and foreground.shape[-1] == 1:
        foreground = foreground[..., 0]
    mask = foreground & labels.eq(int(target_class_id))
    scores = assignment["target_scores"][..., int(target_class_id)].float()
    if not bool(mask.any()):
        raise RuntimeError("Learning-preference audit batch has no real person positives.")
    denominator = scores[mask].sum().clamp_min(1.0e-8)
    class_channel = 4 + int(target_class_id)
    if class_channel >= clean_decoded.shape[1] or class_channel >= carrier_decoded.shape[1]:
        raise RuntimeError("Person class channel is outside decoded victim output.")
    clean_probability = clean_decoded[:, class_channel, :].clamp(1.0e-6, 1.0 - 1.0e-6)
    carrier_probability = carrier_decoded[:, class_channel, :].clamp(1.0e-6, 1.0 - 1.0e-6)
    clean_loss = F.binary_cross_entropy(
        clean_probability[mask], scores[mask], reduction="sum"
    ) / denominator
    carrier_loss = F.binary_cross_entropy(
        carrier_probability[mask], scores[mask], reduction="sum"
    ) / denominator
    if not torch.isfinite(clean_loss) or not torch.isfinite(carrier_loss):
        raise RuntimeError("Learning-preference person classification loss is non-finite.")
    return {
        "clean_counterfactual_loss": float(clean_loss),
        "carrier_loss": float(carrier_loss),
        "person_positive_count": int(mask.sum()),
        "assignment_source": "clean_real_tal_fixed_for_both_views",
    }


def aggregate_person_classification_losses(
    rows: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    if not rows:
        raise ValueError("Learning-preference audit has no batches.")
    counts = np.asarray([float(row["person_positive_count"]) for row in rows])
    clean = np.asarray([float(row["clean_counterfactual_loss"]) for row in rows])
    carrier = np.asarray([float(row["carrier_loss"]) for row in rows])
    if not np.isfinite(counts).all() or not np.isfinite(clean).all() or not np.isfinite(carrier).all():
        raise ValueError("Learning-preference audit contains non-finite batch evidence.")
    if (counts <= 0).any():
        raise ValueError("Every learning-preference batch must have person positives.")
    total = float(counts.sum())
    return {
        "clean_counterfactual_loss": float(np.sum(clean * counts) / total),
        "carrier_loss": float(np.sum(carrier * counts) / total),
        "person_positive_count": int(total),
    }


def build_person_free_transplant_metrics(
    clean_max_person_confidence: Sequence[float],
    carrier_max_person_confidence: Sequence[float],
    *,
    threshold: float = 0.25,
) -> Dict[str, object]:
    clean = np.asarray(clean_max_person_confidence, dtype=np.float64)
    carrier = np.asarray(carrier_max_person_confidence, dtype=np.float64)
    if clean.ndim != 1 or clean.shape != carrier.shape or clean.size == 0:
        raise ValueError("Transplant confidence arrays must be paired non-empty vectors.")
    if not np.isfinite(clean).all() or not np.isfinite(carrier).all():
        raise ValueError("Transplant confidence arrays must be finite.")
    if (clean < 0).any() or (clean > 1).any() or (carrier < 0).any() or (carrier > 1).any():
        raise ValueError("Transplant confidences must lie in [0,1].")
    threshold = float(threshold)
    return {
        "schema": "tausb.sdh-person-free-transplant.v1",
        "image_count": int(clean.size),
        "threshold": threshold,
        "clean_mean_max_person_confidence": float(clean.mean()),
        "carrier_mean_max_person_confidence": float(carrier.mean()),
        "mean_confidence_shift": float((carrier - clean).mean()),
        "clean_person_false_positive_images": int((clean >= threshold).sum()),
        "carrier_person_false_positive_images": int((carrier >= threshold).sum()),
        "false_positive_image_count_shift": int(
            (carrier >= threshold).sum() - (clean >= threshold).sum()
        ),
        "claim_boundary": "descriptive person-free intervention only",
    }
