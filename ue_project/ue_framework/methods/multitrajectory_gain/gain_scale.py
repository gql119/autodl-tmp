from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class GainScales:
    protected: float
    authorized: float
    shared: float
    protected_min: float
    authorized_min: float
    shared_min: float
    source_count: int
    quantile: float
    epsilon: float

    def as_tensors(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "protected": torch.tensor(self.protected, device=device, dtype=dtype).detach(),
            "authorized": torch.tensor(self.authorized, device=device, dtype=dtype).detach(),
            "shared": torch.tensor(self.shared, device=device, dtype=dtype).detach(),
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            "protected": float(self.protected),
            "authorized": float(self.authorized),
            "shared": float(self.shared),
            "protected_min": float(self.protected_min),
            "authorized_min": float(self.authorized_min),
            "shared_min": float(self.shared_min),
            "source_count": int(self.source_count),
            "quantile": float(self.quantile),
            "epsilon": float(self.epsilon),
        }


def _finite_abs(values: Iterable[float]) -> np.ndarray:
    arr = np.array([abs(float(v)) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def robust_scale_from_clean_gains(
    gains: Iterable[float],
    quantile: float = 0.50,
    epsilon: float = 1.0e-4,
    minimum: float = 1.0e-4,
) -> float:
    arr = _finite_abs(gains)
    if arr.size == 0:
        return float(max(minimum, epsilon))
    value = float(np.quantile(arr, float(quantile)))
    return float(max(value + float(epsilon), float(minimum)))


def robust_min_from_positive_gains(
    gains: Iterable[float],
    quantile: float = 0.20,
    absolute_floor: float = 1.0e-4,
) -> float:
    arr = np.array([float(v) for v in gains if np.isfinite(float(v)) and float(v) > 0.0], dtype=np.float64)
    if arr.size == 0:
        return float(absolute_floor)
    return float(max(float(absolute_floor), float(np.quantile(arr, float(quantile)))))


def compute_gain_scales_from_rows(
    rows: Iterable[Mapping[str, float]],
    scale_quantile: float = 0.50,
    min_quantile: float = 0.20,
    epsilon: float = 1.0e-4,
    absolute_floor: float = 1.0e-4,
) -> GainScales:
    materialized = list(rows)
    protected = [float(row["G_t_clean"]) for row in materialized]
    authorized = [float(row["G_a_clean"]) for row in materialized]
    shared = [float(row["G_s_clean"]) for row in materialized]
    return GainScales(
        protected=robust_scale_from_clean_gains(protected, scale_quantile, epsilon, absolute_floor),
        authorized=robust_scale_from_clean_gains(authorized, scale_quantile, epsilon, absolute_floor),
        shared=robust_scale_from_clean_gains(shared, scale_quantile, epsilon, absolute_floor),
        protected_min=robust_min_from_positive_gains(protected, min_quantile, absolute_floor),
        authorized_min=robust_min_from_positive_gains(authorized, min_quantile, absolute_floor),
        shared_min=robust_min_from_positive_gains(shared, min_quantile, absolute_floor),
        source_count=len(materialized),
        quantile=float(scale_quantile),
        epsilon=float(epsilon),
    )
