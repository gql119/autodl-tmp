from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ue_framework.methods.detector_tower_hooks import YOLODetectTowerCapture


TensorTuple = Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class DetectorLFCFeatures:
    """Normalized instance descriptors from YOLO classification towers."""

    classification: TensorTuple

    @property
    def batch_size(self) -> int:
        return int(self.classification[0].shape[0])


@dataclass(frozen=True)
class DetectorLFCResult:
    loss: torch.Tensor
    per_scale_loss: TensorTuple
    per_scale_cosine: TensorTuple

    def detached_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "dlfc_loss": float(self.loss.detach().cpu()),
            "dlfc_num_scales": float(len(self.per_scale_cosine)),
        }
        all_cosine = []
        for index, cosine in enumerate(self.per_scale_cosine):
            values = cosine.detach().float().cpu()
            all_cosine.append(values)
            metrics["dlfc_p%d_cosine_median" % (index + 3)] = float(
                values.median()
            )
            metrics["dlfc_p%d_cosine_q25" % (index + 3)] = float(
                torch.quantile(values, 0.25)
            )
            metrics["dlfc_p%d_coverage" % (index + 3)] = float(
                torch.isfinite(values).float().mean()
            )
        concatenated = torch.cat(all_cosine)
        metrics["dlfc_cosine_median"] = float(concatenated.median())
        metrics["dlfc_cosine_q25"] = float(torch.quantile(concatenated, 0.25))
        return metrics


def _validate_canonical_delta(delta: torch.Tensor) -> None:
    if delta.ndim != 4 or delta.shape[1] != 3:
        raise ValueError("Canonical deltas must have shape [N,3,H,W].")
    if delta.shape[0] < 1:
        raise ValueError("At least one canonical delta is required.")
    if not torch.isfinite(delta).all():
        raise ValueError("Canonical deltas must be finite.")
    flattened = delta.reshape(delta.shape[0], -1)
    if torch.any(flattened.square().mean(dim=1) <= 1.0e-16):
        raise ValueError("Zero canonical delta is invalid for D-LFC.")
    if torch.any(flattened.std(dim=1, unbiased=False) <= 1.0e-8):
        raise ValueError("Constant canonical delta is invalid for D-LFC.")


class DetectorLFCExtractor:
    """Extract D-LFC features from frozen YOLO P3/P4/P5 class towers.

    The detector parameters are frozen, while autograd remains active for the
    canonical perturbation. This makes D-LFC detector-native and prevents the
    ImageNet/ResNet feature-domain mismatch of the earlier draft.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        eps: float = 16.0 / 255.0,
        detect_path: str = "model.22",
        num_scales: int = 3,
        expected_num_classes: int = 20,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.model = model
        self.eps = float(eps)
        self.num_scales = int(num_scales)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.capture = YOLODetectTowerCapture(
            self.model,
            detect_path=detect_path,
            num_scales=self.num_scales,
            expected_num_classes=expected_num_classes,
        )
        self._counter = 0

    def extract(self, canonical_delta: torch.Tensor) -> DetectorLFCFeatures:
        _validate_canonical_delta(canonical_delta)
        delta_visualization = torch.clamp(
            0.5 + canonical_delta / (2.0 * self.eps), 0.0, 1.0
        )
        tag = "dlfc_%d" % self._counter
        self._counter += 1
        with self.capture.record(tag):
            self.model(delta_visualization)
        captured = self.capture.take(tag)
        pooled = tuple(
            F.normalize(feature.mean(dim=(-2, -1)), p=2, dim=1, eps=1.0e-8)
            for feature in captured.classification
        )
        if len(pooled) != self.num_scales:
            raise RuntimeError("D-LFC classification scale count changed.")
        if any(not torch.isfinite(feature).all() for feature in pooled):
            raise RuntimeError("D-LFC produced non-finite normalized features.")
        return DetectorLFCFeatures(classification=pooled)

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "DetectorLFCExtractor":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


class DetectorLFCPrototypeBank:
    """Frozen calibration prototypes for equal-weight P3/P4/P5 D-LFC."""

    def __init__(self, *, num_scales: int = 3) -> None:
        self.num_scales = int(num_scales)
        self._prototypes: Optional[TensorTuple] = None
        self._calibration_count = 0

    @property
    def is_fitted(self) -> bool:
        return self._prototypes is not None

    @property
    def calibration_count(self) -> int:
        return self._calibration_count

    def fit(
        self,
        batches: Iterable[DetectorLFCFeatures],
        *,
        split: str,
    ) -> None:
        if split not in ("calibration", "train_calibration"):
            raise ValueError("D-LFC prototypes may only use the calibration split.")
        if self.is_fitted:
            raise RuntimeError("D-LFC prototype bank is already frozen.")
        collected = [[] for _ in range(self.num_scales)]
        count = 0
        for batch in batches:
            if len(batch.classification) != self.num_scales:
                raise ValueError("D-LFC calibration scale count mismatch.")
            batch_size = batch.batch_size
            count += batch_size
            for index, feature in enumerate(batch.classification):
                if feature.ndim != 2 or feature.shape[0] != batch_size:
                    raise ValueError("D-LFC calibration features must be [N,C].")
                if not torch.isfinite(feature).all():
                    raise ValueError("D-LFC calibration features must be finite.")
                collected[index].append(feature.detach())
        if count < 1:
            raise ValueError("D-LFC calibration split is empty.")
        prototypes = []
        for scale_features in collected:
            combined = torch.cat(scale_features, dim=0)
            prototype = F.normalize(
                combined.mean(dim=0, keepdim=True), p=2, dim=1, eps=1.0e-8
            ).squeeze(0)
            if not torch.isfinite(prototype).all() or prototype.norm() <= 1.0e-8:
                raise ValueError("D-LFC calibration prototype is degenerate.")
            prototypes.append(prototype)
        self._prototypes = tuple(prototypes)
        self._calibration_count = count

    def compute(self, features: DetectorLFCFeatures) -> DetectorLFCResult:
        if self._prototypes is None:
            raise RuntimeError("D-LFC prototype bank is not fitted.")
        if len(features.classification) != self.num_scales:
            raise ValueError("D-LFC feature scale count mismatch.")
        per_scale_cosine = []
        per_scale_loss = []
        for feature, prototype in zip(features.classification, self._prototypes):
            if feature.ndim != 2 or feature.shape[1] != prototype.numel():
                raise ValueError("D-LFC feature channel count mismatch.")
            normalized = F.normalize(feature, p=2, dim=1, eps=1.0e-8)
            reference = prototype.to(device=feature.device, dtype=feature.dtype)
            cosine = (normalized * reference.unsqueeze(0)).sum(dim=1)
            if not torch.isfinite(cosine).all():
                raise RuntimeError("D-LFC cosine is non-finite.")
            per_scale_cosine.append(cosine)
            per_scale_loss.append(1.0 - cosine.mean())
        loss = torch.stack(per_scale_loss).mean()
        return DetectorLFCResult(
            loss=loss,
            per_scale_loss=tuple(per_scale_loss),
            per_scale_cosine=tuple(per_scale_cosine),
        )

    def state_dict(self) -> Dict[str, object]:
        if self._prototypes is None:
            raise RuntimeError("Cannot serialize an unfitted D-LFC prototype bank.")
        return {
            "num_scales": self.num_scales,
            "calibration_count": self._calibration_count,
            "prototypes": tuple(item.detach().cpu() for item in self._prototypes),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if self.is_fitted:
            raise RuntimeError("Cannot overwrite a frozen D-LFC prototype bank.")
        num_scales = int(state["num_scales"])
        prototypes = tuple(state["prototypes"])
        if num_scales != self.num_scales or len(prototypes) != self.num_scales:
            raise ValueError("Serialized D-LFC scale count mismatch.")
        if any(not torch.is_tensor(item) or item.ndim != 1 for item in prototypes):
            raise ValueError("Serialized D-LFC prototypes are invalid.")
        self._prototypes = tuple(item.detach().clone() for item in prototypes)
        self._calibration_count = int(state["calibration_count"])
