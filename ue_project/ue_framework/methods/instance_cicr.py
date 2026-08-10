from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cicr import CICRPrototypeBank, ClassificationResiduals


@dataclass(frozen=True)
class InstanceClassificationResiduals:
    vectors: tuple[torch.Tensor, ...]
    gate_valid: tuple[torch.Tensor, ...]
    gate_mass: tuple[torch.Tensor, ...]
    image_indices: torch.Tensor
    gt_indices: torch.Tensor

    def as_classification_residuals(self) -> ClassificationResiduals:
        return ClassificationResiduals(
            vectors=self.vectors,
            gate_valid=self.gate_valid,
            gate_mass=self.gate_mass,
        )


@dataclass(frozen=True)
class InstanceCICRResult:
    loss: torch.Tensor
    per_instance_cosine: torch.Tensor
    per_instance_valid_scale_count: torch.Tensor
    valid_instance_count: int
    total_instance_count: int
    valid_instance_coverage: float
    missing_assignment_ratio: float
    low_energy_ratio: float
    zero_norm_ratio: float
    per_scale_valid_count: tuple[int, ...]


@dataclass(frozen=True)
class FrozenInstanceCICRResult:
    """Detector-instance CICR with a separate residual-energy floor."""

    loss: torch.Tensor
    direction_loss: torch.Tensor
    energy_floor_loss: torch.Tensor
    per_instance_cosine: torch.Tensor
    per_instance_valid_scale_count: torch.Tensor
    valid_instance_count: int
    total_instance_count: int
    valid_instance_coverage: float
    missing_assignment_ratio: float
    low_energy_ratio: float
    zero_norm_ratio: float
    per_scale_valid_count: Tuple[int, ...]


class FrozenInstanceCICRBank:
    """Calibration-only, frozen prototypes and energy thresholds for CICR.

    D-LFC concentrates canonical-delta features. This bank instead operates on
    clean-to-poison residuals at real person TAL/PAG positive locations, so the
    two losses remain experimentally distinct.
    """

    def __init__(
        self,
        *,
        num_scales: int = 3,
        energy_floor_multiplier: float = 0.5,
        epsilon: float = 1.0e-8,
    ) -> None:
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if energy_floor_multiplier <= 0:
            raise ValueError("energy_floor_multiplier must be positive.")
        self.num_scales = int(num_scales)
        self.energy_floor_multiplier = float(energy_floor_multiplier)
        self.epsilon = float(epsilon)
        self._prototypes: Optional[Tuple[torch.Tensor, ...]] = None
        self._energy_floors: Optional[Tuple[float, ...]] = None
        self._calibration_instance_count = 0

    @property
    def is_fitted(self) -> bool:
        return self._prototypes is not None

    @property
    def energy_floors(self) -> Tuple[float, ...]:
        if self._energy_floors is None:
            raise RuntimeError("Frozen instance CICR bank is not fitted.")
        return self._energy_floors

    @property
    def calibration_instance_count(self) -> int:
        return self._calibration_instance_count

    def fit(
        self,
        batches: Iterable[InstanceClassificationResiduals],
        *,
        split: str,
    ) -> None:
        if split not in ("calibration", "train_calibration"):
            raise ValueError("CICR prototypes may only use the calibration split.")
        if self.is_fitted:
            raise RuntimeError("Frozen instance CICR bank is already fitted.")
        collected = [[] for _ in range(self.num_scales)]
        unique_instances = set()
        for residuals in batches:
            if len(residuals.vectors) != self.num_scales:
                raise ValueError("CICR calibration scale count mismatch.")
            for image_index, gt_index in zip(
                residuals.image_indices.detach().cpu().tolist(),
                residuals.gt_indices.detach().cpu().tolist(),
            ):
                unique_instances.add((int(image_index), int(gt_index)))
            for scale, (vectors, gate_valid) in enumerate(
                zip(residuals.vectors, residuals.gate_valid)
            ):
                if vectors.ndim != 2 or gate_valid.shape != vectors.shape[:1]:
                    raise ValueError("CICR calibration residual shape mismatch.")
                valid = gate_valid & torch.isfinite(vectors).all(dim=1)
                if bool(valid.any()):
                    collected[scale].append(vectors.detach()[valid])
        prototypes = []
        floors = []
        for scale, values in enumerate(collected):
            if not values:
                raise ValueError(
                    "CICR calibration scale %d has no assigned instances." % scale
                )
            combined = torch.cat(values, dim=0)
            norms = combined.square().mean(dim=1).sqrt()
            positive = torch.isfinite(norms) & (norms > self.epsilon)
            if not bool(positive.any()):
                raise ValueError(
                    "CICR calibration scale %d has no nonzero residuals." % scale
                )
            active = combined[positive]
            active_norms = norms[positive]
            prototype = F.normalize(
                F.normalize(active, dim=1, eps=self.epsilon).mean(dim=0),
                dim=0,
                eps=self.epsilon,
            )
            floor = float(
                (
                    torch.quantile(active_norms, 0.25)
                    * self.energy_floor_multiplier
                ).item()
            )
            if not torch.isfinite(prototype).all() or floor <= self.epsilon:
                raise ValueError("CICR calibration produced a degenerate state.")
            prototypes.append(prototype.detach())
            floors.append(floor)
        if not unique_instances:
            raise ValueError("CICR calibration contains no person instances.")
        self._prototypes = tuple(prototypes)
        self._energy_floors = tuple(floors)
        self._calibration_instance_count = len(unique_instances)

    def compute(
        self,
        residuals: InstanceClassificationResiduals,
        *,
        energy_weight: float = 1.0,
    ) -> FrozenInstanceCICRResult:
        if self._prototypes is None or self._energy_floors is None:
            raise RuntimeError("Frozen instance CICR bank is not fitted.")
        if energy_weight < 0:
            raise ValueError("energy_weight must be non-negative.")
        if len(residuals.vectors) != self.num_scales:
            raise ValueError("CICR residual scale count mismatch.")
        total = int(residuals.image_indices.numel())
        cosine_by_instance = [[] for _ in range(total)]
        energy_by_instance = [[] for _ in range(total)]
        gate_seen = torch.zeros(
            total, device=residuals.image_indices.device, dtype=torch.bool
        )
        nonzero_seen = torch.zeros_like(gate_seen)
        eligible_seen = torch.zeros_like(gate_seen)
        per_scale_valid_count = []

        for scale, (vectors, gate_valid) in enumerate(
            zip(residuals.vectors, residuals.gate_valid)
        ):
            if vectors.ndim != 2 or gate_valid.shape != vectors.shape[:1]:
                raise ValueError("CICR residual/assignment shape mismatch.")
            floor = self._energy_floors[scale]
            norms = vectors.square().mean(dim=1).sqrt()
            finite = torch.isfinite(norms) & torch.isfinite(vectors).all(dim=1)
            assigned = gate_valid & finite
            eligible = assigned & (norms >= floor)
            gate_seen |= gate_valid
            nonzero_seen |= assigned & (norms > self.epsilon)
            eligible_seen |= eligible
            per_scale_valid_count.append(int(eligible.sum().item()))

            prototype = self._prototypes[scale].to(vectors).detach()
            for instance_index in torch.where(assigned)[0].tolist():
                energy_by_instance[instance_index].append(
                    F.relu(vectors.new_tensor(floor) - norms[instance_index])
                    / floor
                )
            if bool(eligible.any()):
                normalized = F.normalize(
                    vectors[eligible], dim=1, eps=self.epsilon
                )
                target = F.normalize(prototype, dim=0, eps=self.epsilon)
                cosines = (normalized * target.unsqueeze(0)).sum(dim=1)
                for local_index, instance_index in enumerate(
                    torch.where(eligible)[0].tolist()
                ):
                    cosine_by_instance[instance_index].append(cosines[local_index])

        instance_direction_terms = [
            1.0 - torch.stack(values).mean()
            for values in cosine_by_instance
            if values
        ]
        instance_energy_terms = [
            torch.stack(values).mean() for values in energy_by_instance if values
        ]
        zero = sum(vector.sum() * 0.0 for vector in residuals.vectors)
        direction_loss = (
            torch.stack(instance_direction_terms).mean()
            if instance_direction_terms
            else zero
        )
        energy_floor_loss = (
            torch.stack(instance_energy_terms).mean()
            if instance_energy_terms
            else zero
        )
        loss = direction_loss + float(energy_weight) * energy_floor_loss

        detached_cosines = []
        valid_scale_counts = []
        for values in cosine_by_instance:
            valid_scale_counts.append(len(values))
            detached_cosines.append(
                torch.stack(values).median().detach()
                if values
                else residuals.vectors[0].new_tensor(float("nan"))
            )
        per_instance_cosine = (
            torch.stack(detached_cosines)
            if detached_cosines
            else residuals.vectors[0].new_empty((0,))
        )
        per_instance_valid_scale_count = torch.tensor(
            valid_scale_counts,
            device=residuals.image_indices.device,
            dtype=torch.long,
        )
        denominator = max(total, 1)
        valid_count = int(eligible_seen.sum().item())
        assigned_count = int(gate_seen.sum().item())
        low_energy_count = int((gate_seen & ~eligible_seen).sum().item())
        zero_norm_count = int((gate_seen & ~nonzero_seen).sum().item())
        return FrozenInstanceCICRResult(
            loss=loss,
            direction_loss=direction_loss,
            energy_floor_loss=energy_floor_loss,
            per_instance_cosine=per_instance_cosine,
            per_instance_valid_scale_count=per_instance_valid_scale_count,
            valid_instance_count=valid_count,
            total_instance_count=total,
            valid_instance_coverage=valid_count / denominator,
            missing_assignment_ratio=(total - assigned_count) / denominator,
            low_energy_ratio=low_energy_count / denominator,
            zero_norm_ratio=zero_norm_count / denominator,
            per_scale_valid_count=tuple(per_scale_valid_count),
        )

    def state_dict(self) -> Dict[str, object]:
        if self._prototypes is None or self._energy_floors is None:
            raise RuntimeError("Cannot serialize an unfitted CICR bank.")
        return {
            "num_scales": self.num_scales,
            "energy_floor_multiplier": self.energy_floor_multiplier,
            "epsilon": self.epsilon,
            "calibration_instance_count": self._calibration_instance_count,
            "energy_floors": self._energy_floors,
            "prototypes": tuple(item.detach().cpu() for item in self._prototypes),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if self.is_fitted:
            raise RuntimeError("Cannot overwrite a frozen CICR bank.")
        if int(state["num_scales"]) != self.num_scales:
            raise ValueError("Serialized CICR scale count mismatch.")
        prototypes = tuple(state["prototypes"])
        floors = tuple(float(value) for value in state["energy_floors"])
        if len(prototypes) != self.num_scales or len(floors) != self.num_scales:
            raise ValueError("Serialized CICR state is incomplete.")
        if any(not torch.is_tensor(value) or value.ndim != 1 for value in prototypes):
            raise ValueError("Serialized CICR prototypes are invalid.")
        if any(value <= self.epsilon for value in floors):
            raise ValueError("Serialized CICR energy floor is invalid.")
        self._prototypes = tuple(value.detach().clone() for value in prototypes)
        self._energy_floors = floors
        self._calibration_instance_count = int(
            state["calibration_instance_count"]
        )


def target_gt_indices_from_labels(
    gt_labels: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    target_class_id: int,
) -> tuple[tuple[int, ...], ...]:
    if gt_labels.ndim == 3 and gt_labels.shape[-1] == 1:
        gt_labels = gt_labels[..., 0]
    if mask_gt.ndim == 3 and mask_gt.shape[-1] == 1:
        mask_gt = mask_gt[..., 0]
    if gt_labels.ndim != 2 or mask_gt.shape != gt_labels.shape:
        raise ValueError("gt_labels and mask_gt must align as [B,N].")
    return tuple(
        tuple(
            int(index)
            for index in torch.where(
                mask_gt[batch_index].bool()
                & (gt_labels[batch_index].long() == int(target_class_id))
            )[0].tolist()
        )
        for batch_index in range(gt_labels.shape[0])
    )


def instance_classification_residuals(
    clean_features: Sequence[torch.Tensor],
    adv_features: Sequence[torch.Tensor],
    pag_gate: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_gt_indices_by_image: Sequence[Sequence[int]],
    assigned_scores: Optional[torch.Tensor] = None,
) -> InstanceClassificationResiduals:
    if len(clean_features) != len(adv_features) or not clean_features:
        raise ValueError("clean and adv feature sequences must be non-empty and aligned.")
    if pag_gate.ndim == 3 and pag_gate.shape[-1] == 1:
        pag_gate = pag_gate[..., 0]
    if target_gt_idx.ndim == 3 and target_gt_idx.shape[-1] == 1:
        target_gt_idx = target_gt_idx[..., 0]
    if pag_gate.ndim != 2 or target_gt_idx.shape != pag_gate.shape:
        raise ValueError("pag_gate and target_gt_idx must align as [B,A].")
    if assigned_scores is not None:
        if assigned_scores.ndim == 3 and assigned_scores.shape[-1] == 1:
            assigned_scores = assigned_scores[..., 0]
        if assigned_scores.shape != pag_gate.shape:
            raise ValueError("assigned_scores must align with PAG as [B,A].")
    batch_size = clean_features[0].shape[0]
    if pag_gate.shape[0] != batch_size:
        raise ValueError("Assignment batch size does not match features.")
    if len(target_gt_indices_by_image) != batch_size:
        raise ValueError("target_gt_indices_by_image must match batch size.")

    layer_sizes = []
    for scale, (clean, adv) in enumerate(zip(clean_features, adv_features)):
        if clean.shape != adv.shape or clean.ndim != 4:
            raise ValueError(
                f"Scale {scale} clean/adv features must share [B,C,H,W]."
            )
        if clean.shape[0] != batch_size:
            raise ValueError(f"Scale {scale} batch size mismatch.")
        layer_sizes.append(clean.shape[-2] * clean.shape[-1])
    if sum(layer_sizes) != pag_gate.shape[1]:
        raise ValueError("Assignment width does not match feature layer sizes.")

    instance_keys = []
    for image_index, gt_indices in enumerate(target_gt_indices_by_image):
        frozen_indices = tuple(int(value) for value in gt_indices)
        if len(frozen_indices) != len(set(frozen_indices)):
            raise ValueError("Target GT indices must be unique within each image.")
        if any(value < 0 for value in frozen_indices):
            raise ValueError("Target GT indices must be non-negative.")
        instance_keys.extend((image_index, value) for value in frozen_indices)

    device = adv_features[0].device
    image_indices = torch.tensor(
        [key[0] for key in instance_keys],
        device=device,
        dtype=torch.long,
    )
    gt_indices = torch.tensor(
        [key[1] for key in instance_keys],
        device=device,
        dtype=torch.long,
    )
    pag_gate = pag_gate.to(device=device).bool()
    target_gt_idx = target_gt_idx.to(device=device).long()
    score_weights = (
        torch.ones_like(pag_gate, dtype=adv_features[0].dtype)
        if assigned_scores is None
        else assigned_scores.to(device=device, dtype=adv_features[0].dtype).clamp_min(0)
    )

    vectors = []
    valid_flags = []
    masses = []
    offset = 0
    for clean, adv, layer_size in zip(
        clean_features,
        adv_features,
        layer_sizes,
    ):
        residual = (adv - clean.detach()).flatten(2)
        scale_pag = pag_gate[:, offset : offset + layer_size]
        scale_gt_idx = target_gt_idx[:, offset : offset + layer_size]
        scale_scores = score_weights[:, offset : offset + layer_size]
        scale_vectors = []
        scale_valid = []
        scale_mass = []
        for image_index, gt_index in instance_keys:
            gate = scale_pag[image_index] & (
                scale_gt_idx[image_index] == gt_index
            )
            weights = gate.to(dtype=adv.dtype) * scale_scores[image_index]
            mass = weights.sum()
            vector = (
                residual[image_index]
                * weights.unsqueeze(0)
            ).sum(dim=1)
            vector = vector / mass.clamp_min(1.0)
            valid = mass > 0
            scale_vectors.append(
                torch.where(valid, vector, torch.zeros_like(vector))
            )
            scale_valid.append(valid)
            scale_mass.append(mass)
        channels = adv.shape[1]
        vectors.append(
            torch.stack(scale_vectors)
            if scale_vectors
            else adv.new_zeros((0, channels))
        )
        valid_flags.append(
            torch.stack(scale_valid)
            if scale_valid
            else torch.zeros((0,), device=device, dtype=torch.bool)
        )
        masses.append(
            torch.stack(scale_mass)
            if scale_mass
            else adv.new_zeros((0,))
        )
        offset += layer_size

    return InstanceClassificationResiduals(
        vectors=tuple(vectors),
        gate_valid=tuple(valid_flags),
        gate_mass=tuple(masses),
        image_indices=image_indices,
        gt_indices=gt_indices,
    )


def fit_instance_prototype_bank(
    residuals: InstanceClassificationResiduals,
    *,
    momentum: float,
    energy_floor_multiplier: float = 0.5,
) -> CICRPrototypeBank:
    bank = CICRPrototypeBank(
        num_scales=len(residuals.vectors),
        momentum=momentum,
    )
    warmup = [
        vector[valid].detach()
        for vector, valid in zip(residuals.vectors, residuals.gate_valid)
    ]
    bank.calibrate_energy_floors(
        warmup,
        multiplier=energy_floor_multiplier,
    )
    bank.update(residuals.as_classification_residuals(), split="train")
    return bank


def instance_cicr(
    residuals: InstanceClassificationResiduals,
    bank: CICRPrototypeBank,
) -> InstanceCICRResult:
    if len(residuals.vectors) != bank.num_scales:
        raise ValueError("Residual scale count mismatch.")
    total = int(residuals.image_indices.numel())
    cosine_by_instance: list[list[torch.Tensor]] = [[] for _ in range(total)]
    loss_cosines = []
    gate_seen = torch.zeros(
        total,
        device=residuals.image_indices.device,
        dtype=torch.bool,
    )
    nonzero_seen = torch.zeros_like(gate_seen)
    eligible_seen = torch.zeros_like(gate_seen)
    per_scale_valid_count = []

    for scale, (vectors, gate_valid) in enumerate(
        zip(residuals.vectors, residuals.gate_valid)
    ):
        prototype = bank.prototype(scale)
        if prototype is None:
            raise RuntimeError(f"CICR prototype for scale {scale} is unset.")
        floor = float(bank.energy_floors[scale] or bank.epsilon)
        norms = vectors.detach().norm(dim=1)
        finite = torch.isfinite(norms)
        eligible = gate_valid & finite & (norms >= floor)
        gate_seen |= gate_valid
        nonzero_seen |= gate_valid & finite & (norms > bank.epsilon)
        eligible_seen |= eligible
        per_scale_valid_count.append(int(eligible.sum().item()))
        if not bool(eligible.any()):
            continue
        normalized = F.normalize(vectors[eligible], dim=1, eps=bank.epsilon)
        target = F.normalize(
            prototype.to(vectors).detach(),
            dim=0,
            eps=bank.epsilon,
        )
        cosines = (normalized * target.unsqueeze(0)).sum(dim=1)
        eligible_indices = torch.where(eligible)[0]
        for local_index, instance_index in enumerate(eligible_indices.tolist()):
            cosine_by_instance[instance_index].append(cosines[local_index])
        loss_cosines.append(cosines)

    if loss_cosines:
        loss = 1.0 - torch.cat(loss_cosines).mean()
    else:
        loss = sum(vector.sum() * 0.0 for vector in residuals.vectors)

    detached_cosines = []
    valid_scale_counts = []
    for values in cosine_by_instance:
        valid_scale_counts.append(len(values))
        detached_cosines.append(
            torch.stack(values).median().detach()
            if values
            else residuals.vectors[0].new_tensor(float("nan"))
        )
    per_instance_cosine = (
        torch.stack(detached_cosines).to(residuals.image_indices.device)
        if detached_cosines
        else torch.empty(
            (0,),
            device=residuals.image_indices.device,
        )
    )
    per_instance_valid_scale_count = torch.tensor(
        valid_scale_counts,
        device=residuals.image_indices.device,
        dtype=torch.long,
    )
    valid_count = int(eligible_seen.sum().item())
    denominator = max(total, 1)
    assigned_count = int(gate_seen.sum().item())
    low_energy_count = int((gate_seen & ~eligible_seen).sum().item())
    zero_norm_count = int((gate_seen & ~nonzero_seen).sum().item())
    return InstanceCICRResult(
        loss=loss,
        per_instance_cosine=per_instance_cosine,
        per_instance_valid_scale_count=per_instance_valid_scale_count,
        valid_instance_count=valid_count,
        total_instance_count=total,
        valid_instance_coverage=valid_count / denominator,
        missing_assignment_ratio=(total - assigned_count) / denominator,
        low_energy_ratio=low_energy_count / denominator,
        zero_norm_ratio=zero_norm_count / denominator,
        per_scale_valid_count=tuple(per_scale_valid_count),
    )
