from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassificationResiduals:
    vectors: tuple[torch.Tensor, ...]
    gate_valid: tuple[torch.Tensor, ...]
    gate_mass: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class CICRResult:
    loss: torch.Tensor
    per_scale_cosine: tuple[float, ...]
    per_scale_valid_count: tuple[int, ...]
    per_scale_zero_or_low_energy_count: tuple[int, ...]


def classification_residuals(
    clean_features: Sequence[torch.Tensor],
    adv_features: Sequence[torch.Tensor],
    pag_gates: Sequence[torch.Tensor],
) -> ClassificationResiduals:
    if not (
        len(clean_features) == len(adv_features) == len(pag_gates)
    ):
        raise ValueError("clean, adv, and PAG sequences must have equal length.")
    if not clean_features:
        raise ValueError("At least one feature scale is required.")

    vectors: list[torch.Tensor] = []
    gate_valid: list[torch.Tensor] = []
    gate_mass: list[torch.Tensor] = []
    for scale, (clean, adv, gate) in enumerate(
        zip(clean_features, adv_features, pag_gates)
    ):
        if clean.shape != adv.shape or clean.ndim != 4:
            raise ValueError(
                f"Scale {scale} clean/adv features must share [B,C,H,W]."
            )
        if gate.ndim == 3:
            gate = gate.unsqueeze(1)
        if gate.ndim != 4 or gate.shape[1] != 1:
            raise ValueError(f"Scale {scale} PAG gate must have shape [B,1,H,W].")
        if gate.shape[0] != clean.shape[0] or gate.shape[-2:] != clean.shape[-2:]:
            raise ValueError(
                f"Scale {scale} PAG gate does not align with feature shape."
            )
        gate = gate.to(device=adv.device, dtype=adv.dtype).clamp(min=0)
        mass = gate.sum(dim=(-2, -1)).squeeze(1)
        valid = mass > 0
        residual_map = adv - clean.detach()
        vector = (residual_map * gate).sum(dim=(-2, -1))
        vector = vector / mass.clamp_min(1e-12).unsqueeze(1)
        vector = torch.where(valid.unsqueeze(1), vector, torch.zeros_like(vector))
        vectors.append(vector)
        gate_valid.append(valid)
        gate_mass.append(mass)

    return ClassificationResiduals(
        vectors=tuple(vectors),
        gate_valid=tuple(gate_valid),
        gate_mass=tuple(gate_mass),
    )


class CICRPrototypeBank:
    def __init__(
        self,
        *,
        num_scales: int = 3,
        momentum: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0,1).")
        self.num_scales = int(num_scales)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self._prototypes: list[torch.Tensor | None] = [None] * self.num_scales
        self._energy_floors: list[float | None] = [None] * self.num_scales

    @property
    def energy_floors(self) -> tuple[float | None, ...]:
        return tuple(self._energy_floors)

    def prototype(self, scale: int) -> torch.Tensor | None:
        value = self._prototypes[scale]
        return None if value is None else value.clone()

    def calibrate_energy_floors(
        self,
        warmup_residuals: Sequence[torch.Tensor],
        *,
        multiplier: float = 0.5,
    ) -> tuple[float, ...]:
        if len(warmup_residuals) != self.num_scales:
            raise ValueError("Warmup residual scale count mismatch.")
        if multiplier <= 0:
            raise ValueError("Energy-floor multiplier must be positive.")
        floors: list[float] = []
        for scale, residual in enumerate(warmup_residuals):
            if residual.ndim != 2 or residual.shape[0] == 0:
                raise ValueError(
                    f"Scale {scale} warmup residuals must be non-empty [N,C]."
                )
            norms = residual.detach().norm(dim=1)
            norms = norms[torch.isfinite(norms)]
            if norms.numel() == 0:
                raise ValueError(f"Scale {scale} has no finite warmup norms.")
            floor = float((torch.quantile(norms, 0.25) * multiplier).item())
            if floor <= self.epsilon:
                raise ValueError(
                    f"Scale {scale} energy floor is not above epsilon."
                )
            self._energy_floors[scale] = floor
            floors.append(floor)
        return tuple(floors)

    def _eligible(
        self,
        scale: int,
        residual: torch.Tensor,
        gate_valid: torch.Tensor,
    ) -> torch.Tensor:
        if residual.ndim != 2 or gate_valid.shape != residual.shape[:1]:
            raise ValueError(f"Scale {scale} residual/valid shape mismatch.")
        floor = self._energy_floors[scale]
        threshold = self.epsilon if floor is None else float(floor)
        norm = residual.detach().norm(dim=1)
        return gate_valid & torch.isfinite(norm) & (norm >= threshold)

    def update(
        self,
        residuals: ClassificationResiduals,
        *,
        split: str,
    ) -> None:
        if split != "train":
            raise ValueError("CICR prototypes may only update from split='train'.")
        if len(residuals.vectors) != self.num_scales:
            raise ValueError("Residual scale count mismatch.")

        for scale, (residual, gate_valid) in enumerate(
            zip(residuals.vectors, residuals.gate_valid)
        ):
            eligible = self._eligible(scale, residual, gate_valid)
            if not bool(eligible.any()):
                continue
            normalized = F.normalize(
                residual.detach()[eligible],
                dim=1,
                eps=self.epsilon,
            )
            candidate = F.normalize(
                normalized.mean(dim=0),
                dim=0,
                eps=self.epsilon,
            )
            current = self._prototypes[scale]
            if current is None:
                updated = candidate
            else:
                updated = F.normalize(
                    self.momentum * current.to(candidate)
                    + (1.0 - self.momentum) * candidate,
                    dim=0,
                    eps=self.epsilon,
                )
            self._prototypes[scale] = updated.detach()

    def loss(self, residuals: ClassificationResiduals) -> CICRResult:
        if len(residuals.vectors) != self.num_scales:
            raise ValueError("Residual scale count mismatch.")
        loss_terms: list[torch.Tensor] = []
        cosine_values: list[float] = []
        valid_counts: list[int] = []
        low_energy_counts: list[int] = []

        for scale, (residual, gate_valid) in enumerate(
            zip(residuals.vectors, residuals.gate_valid)
        ):
            prototype = self._prototypes[scale]
            if prototype is None:
                raise RuntimeError(f"CICR prototype for scale {scale} is unset.")
            eligible = self._eligible(scale, residual, gate_valid)
            valid_count = int(eligible.sum().item())
            low_count = int((gate_valid & ~eligible).sum().item())
            valid_counts.append(valid_count)
            low_energy_counts.append(low_count)
            if valid_count == 0:
                cosine_values.append(float("nan"))
                continue

            normalized = F.normalize(
                residual[eligible],
                dim=1,
                eps=self.epsilon,
            )
            target = F.normalize(
                prototype.to(residual).detach(),
                dim=0,
                eps=self.epsilon,
            )
            cosine = (normalized * target.unsqueeze(0)).sum(dim=1)
            loss_terms.append(1.0 - cosine)
            cosine_values.append(float(cosine.detach().median().item()))

        if loss_terms:
            loss = torch.cat(loss_terms).mean()
        else:
            loss = sum(item.sum() * 0.0 for item in residuals.vectors)
        return CICRResult(
            loss=loss,
            per_scale_cosine=tuple(cosine_values),
            per_scale_valid_count=tuple(valid_counts),
            per_scale_zero_or_low_energy_count=tuple(low_energy_counts),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "num_scales": self.num_scales,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "energy_floors": list(self._energy_floors),
            "prototypes": [
                None if item is None else item.detach().cpu()
                for item in self._prototypes
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("num_scales", -1)) != self.num_scales:
            raise ValueError("CICR state num_scales mismatch.")
        floors = state.get("energy_floors")
        prototypes = state.get("prototypes")
        if not isinstance(floors, list) or len(floors) != self.num_scales:
            raise ValueError("CICR state energy_floors mismatch.")
        if not isinstance(prototypes, list) or len(prototypes) != self.num_scales:
            raise ValueError("CICR state prototypes mismatch.")
        loaded: list[torch.Tensor | None] = []
        for scale, prototype in enumerate(prototypes):
            if prototype is not None and not torch.is_tensor(prototype):
                raise ValueError(f"CICR prototype {scale} must be a tensor or None.")
            loaded.append(
                None if prototype is None else prototype.detach().clone()
            )
        self._energy_floors = [
            None if value is None else float(value) for value in floors
        ]
        self._prototypes = loaded
