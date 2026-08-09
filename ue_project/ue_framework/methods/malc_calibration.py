from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .instance_canonical_carrier import tensor_sha256
from .malc import FrozenMALCPrototypeBank, MALCInstanceResiduals


@dataclass(frozen=True)
class MALCPrototypeCalibration:
    bank: FrozenMALCPrototypeBank
    calibration_hash: str
    split_hash: str
    per_scale_vector_count: tuple[int, ...]


class MALCPrototypeCalibrator:
    def __init__(
        self,
        *,
        num_scales: int,
        split_hash: str,
        energy_floor_multiplier: float = 0.5,
        epsilon: float = 1e-8,
    ) -> None:
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if not split_hash:
            raise ValueError("split_hash must be non-empty.")
        if (
            not math.isfinite(float(energy_floor_multiplier))
            or energy_floor_multiplier <= 0
        ):
            raise ValueError("energy_floor_multiplier must be positive and finite.")
        if not math.isfinite(float(epsilon)) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")
        self.num_scales = int(num_scales)
        self.split_hash = str(split_hash)
        self.energy_floor_multiplier = float(energy_floor_multiplier)
        self.epsilon = float(epsilon)
        self._vectors: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_scales)
        ]
        self._finalized = False

    def update(
        self,
        residuals: MALCInstanceResiduals,
        *,
        split: str,
    ) -> None:
        if split != "calibration":
            raise ValueError("MALC prototypes may update only on split='calibration'.")
        if self._finalized:
            raise RuntimeError("MALC prototype calibration is already finalized.")
        if len(residuals.vectors) != self.num_scales:
            raise ValueError("Residual scale count differs from the calibrator.")
        for scale, (vectors, pooling_valid) in enumerate(
            zip(residuals.vectors, residuals.pooling_valid)
        ):
            if vectors.ndim != 2 or pooling_valid.shape != (vectors.shape[0],):
                raise ValueError(f"Scale {scale} residuals are misaligned.")
            selected = vectors[pooling_valid].detach().to(
                device="cpu", dtype=torch.float64
            )
            if selected.numel() and not torch.isfinite(selected).all():
                raise ValueError(f"Scale {scale} calibration residuals must be finite.")
            if selected.numel():
                self._vectors[scale].append(selected)

    def finalize(self) -> MALCPrototypeCalibration:
        if self._finalized:
            raise RuntimeError("MALC prototype calibration is already finalized.")
        prototypes = []
        medians = []
        floors = []
        counts = []
        digest = hashlib.sha256()
        metadata = {
            "split_hash": self.split_hash,
            "num_scales": self.num_scales,
            "energy_floor_multiplier": self.energy_floor_multiplier,
            "epsilon": self.epsilon,
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        for scale, batches in enumerate(self._vectors):
            if not batches:
                raise RuntimeError(
                    f"Scale {scale} has no valid calibration residuals."
                )
            vectors = torch.cat(batches, dim=0)
            norms = vectors.norm(dim=1)
            direction_valid = norms > self.epsilon
            if not bool(direction_valid.any()):
                raise RuntimeError(
                    f"Scale {scale} has no non-zero calibration residuals."
                )
            directions = vectors[direction_valid] / norms[
                direction_valid
            ].unsqueeze(1)
            prototype = directions.mean(dim=0)
            if float(prototype.norm()) <= self.epsilon:
                raise RuntimeError(
                    f"Scale {scale} calibration directions cancel to zero."
                )
            rms = vectors.square().mean(dim=1).sqrt()
            median = float(torch.quantile(rms, 0.5))
            floor = self.energy_floor_multiplier * float(
                torch.quantile(rms, 0.25)
            )
            if not math.isfinite(median) or median <= self.epsilon:
                raise RuntimeError(f"Scale {scale} median RMS is not usable.")
            if not math.isfinite(floor) or floor <= self.epsilon:
                raise RuntimeError(f"Scale {scale} energy floor is not usable.")
            prototypes.append(prototype.float())
            medians.append(median)
            floors.append(floor)
            counts.append(int(vectors.shape[0]))
            digest.update(tensor_sha256(vectors.float()).encode("ascii"))

        bank = FrozenMALCPrototypeBank(
            direction_prototypes=tuple(prototypes),
            median_rms=tuple(medians),
            energy_floors=tuple(floors),
            epsilon=self.epsilon,
        )
        for tensor in bank.direction_prototypes:
            digest.update(tensor_sha256(tensor).encode("ascii"))
        digest.update(json.dumps({
            "median_rms": medians,
            "energy_floors": floors,
            "counts": counts,
        }, sort_keys=True).encode("utf-8"))
        self._finalized = True
        return MALCPrototypeCalibration(
            bank=bank,
            calibration_hash=digest.hexdigest(),
            split_hash=self.split_hash,
            per_scale_vector_count=tuple(counts),
        )


@dataclass(frozen=True)
class FrozenGradientCalibration:
    weights: Mapping[str, float]
    median_norms: Mapping[str, float]
    clipped_components: tuple[str, ...]
    calibration_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(dict(self.weights)),
        )
        object.__setattr__(
            self,
            "median_norms",
            MappingProxyType(dict(self.median_norms)),
        )


class MALCGradientNormCalibrator:
    def __init__(
        self,
        *,
        component_names: Sequence[str],
        reference_name: str = "easy_cls",
        clip_min: float = 0.1,
        clip_max: float = 10.0,
        max_clipped_fraction: float = 0.5,
        epsilon: float = 1e-12,
    ) -> None:
        names = tuple(str(name) for name in component_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("component_names must be unique and non-empty.")
        if reference_name not in names:
            raise ValueError("reference_name must be included in component_names.")
        if not (
            0 < clip_min < clip_max
            and 0 <= max_clipped_fraction <= 1
            and epsilon > 0
        ):
            raise ValueError("Gradient calibration thresholds are invalid.")
        self.component_names = names
        self.reference_name = str(reference_name)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.max_clipped_fraction = float(max_clipped_fraction)
        self.epsilon = float(epsilon)
        self._norms: dict[str, list[float]] = {name: [] for name in names}
        self._finalized = False

    def update(
        self,
        losses: Mapping[str, torch.Tensor],
        parameters: Sequence[torch.Tensor],
    ) -> None:
        if self._finalized:
            raise RuntimeError("Gradient calibration is already finalized.")
        if set(losses) != set(self.component_names):
            raise ValueError("Gradient calibration losses do not match components.")
        trainable = tuple(parameters)
        if not trainable or any(not value.requires_grad for value in trainable):
            raise ValueError("Calibration parameters must require gradients.")
        for name in self.component_names:
            loss = losses[name]
            if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
                raise ValueError(f"Loss {name} must be a finite scalar.")
            gradients = torch.autograd.grad(
                loss,
                trainable,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            if any(gradient is None for gradient in gradients):
                raise RuntimeError(f"Loss {name} is disconnected from carrier parameters.")
            squared_norm = sum(
                gradient.detach().double().square().sum()
                for gradient in gradients
                if gradient is not None
            )
            norm = float(squared_norm.sqrt())
            if not math.isfinite(norm) or norm <= self.epsilon:
                raise RuntimeError(
                    f"Loss {name} has a zero or non-finite carrier gradient."
                )
            self._norms[name].append(norm)

    def finalize(self) -> FrozenGradientCalibration:
        if self._finalized:
            raise RuntimeError("Gradient calibration is already finalized.")
        if any(not values for values in self._norms.values()):
            raise RuntimeError("Every gradient component requires calibration samples.")
        medians = {
            name: float(
                torch.quantile(
                    torch.tensor(values, dtype=torch.float64),
                    0.5,
                )
            )
            for name, values in self._norms.items()
        }
        reference = medians[self.reference_name]
        weights = {}
        clipped = []
        for name in self.component_names:
            if name == self.reference_name:
                continue
            raw = reference / (medians[name] + self.epsilon)
            value = min(max(raw, self.clip_min), self.clip_max)
            weights[name] = value
            if value != raw:
                clipped.append(name)
        clipped_fraction = len(clipped) / max(len(weights), 1)
        if clipped_fraction > self.max_clipped_fraction:
            raise RuntimeError(
                "Too many MALC gradient weights hit the clipping boundary."
            )
        payload = {
            "component_names": self.component_names,
            "reference_name": self.reference_name,
            "clip": [self.clip_min, self.clip_max],
            "max_clipped_fraction": self.max_clipped_fraction,
            "median_norms": medians,
            "weights": weights,
            "clipped_components": clipped,
        }
        self._finalized = True
        return FrozenGradientCalibration(
            weights=weights,
            median_norms=medians,
            clipped_components=tuple(clipped),
            calibration_hash=hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
