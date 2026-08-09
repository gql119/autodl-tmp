from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MALCInstanceResiduals:
    vectors: tuple[torch.Tensor, ...]
    assigned: tuple[torch.Tensor, ...]
    pooling_valid: tuple[torch.Tensor, ...]
    assignment_count: tuple[torch.Tensor, ...]
    score_mass: tuple[torch.Tensor, ...]
    image_indices: torch.Tensor
    gt_indices: torch.Tensor


@dataclass(frozen=True)
class FrozenMALCPrototypeBank:
    direction_prototypes: tuple[torch.Tensor, ...]
    median_rms: tuple[float, ...]
    energy_floors: tuple[float, ...]
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        scale_count = len(self.direction_prototypes)
        if scale_count == 0:
            raise ValueError("MALC prototype bank must contain at least one scale.")
        if not (
            len(self.median_rms) == len(self.energy_floors) == scale_count
        ):
            raise ValueError("MALC prototype bank fields must have equal lengths.")
        if not math.isfinite(float(self.epsilon)) or self.epsilon <= 0:
            raise ValueError("MALC epsilon must be positive and finite.")

        normalized = []
        for scale, prototype in enumerate(self.direction_prototypes):
            if prototype.ndim != 1 or prototype.numel() == 0:
                raise ValueError(
                    f"MALC prototype {scale} must be a non-empty vector."
                )
            if not torch.isfinite(prototype).all():
                raise ValueError(f"MALC prototype {scale} must be finite.")
            norm = prototype.detach().norm()
            if float(norm) <= self.epsilon:
                raise ValueError(f"MALC prototype {scale} has zero norm.")
            normalized.append((prototype.detach() / norm).clone())

        medians = tuple(float(value) for value in self.median_rms)
        floors = tuple(float(value) for value in self.energy_floors)
        if any(not math.isfinite(value) or value <= 0 for value in medians):
            raise ValueError("MALC median RMS values must be positive and finite.")
        if any(not math.isfinite(value) or value <= 0 for value in floors):
            raise ValueError("MALC energy floors must be positive and finite.")
        object.__setattr__(self, "direction_prototypes", tuple(normalized))
        object.__setattr__(self, "median_rms", medians)
        object.__setattr__(self, "energy_floors", floors)


@dataclass(frozen=True)
class MALCResult:
    loss: torch.Tensor
    direction_loss: torch.Tensor
    magnitude_loss: torch.Tensor
    floor_loss: torch.Tensor
    per_scale_loss: tuple[torch.Tensor, ...]
    per_scale_valid_count: tuple[int, ...]
    per_scale_assigned_count: tuple[int, ...]
    scale_contribution_share: tuple[float, ...]
    per_instance_cosine: torch.Tensor
    per_instance_log_energy: torch.Tensor
    valid_instance_count: int
    total_instance_count: int
    valid_instance_coverage: float
    zero_norm_ratio: float
    floor_pass_ratio: float
    valid_scale_count: int


def target_class_assignment_scores(
    target_scores: torch.Tensor,
    *,
    target_class_id: int,
) -> torch.Tensor:
    if target_scores.ndim == 2:
        scores = target_scores
    elif target_scores.ndim == 3:
        if not 0 <= int(target_class_id) < target_scores.shape[-1]:
            raise ValueError("target_class_id is outside target_scores.")
        scores = target_scores[..., int(target_class_id)]
    else:
        raise ValueError("target_scores must have shape [B,A] or [B,A,C].")
    if not torch.isfinite(scores).all():
        raise ValueError("target assignment scores must be finite.")
    if bool((scores < 0).any()):
        raise ValueError("target assignment scores must be non-negative.")
    return scores.detach()


def score_weighted_instance_residuals(
    clean_features: Sequence[torch.Tensor],
    adv_features: Sequence[torch.Tensor],
    *,
    pag_gate: torch.Tensor,
    target_gt_idx: torch.Tensor,
    assigned_scores: torch.Tensor,
    target_gt_indices_by_image: Sequence[Sequence[int]],
    epsilon: float = 1e-8,
) -> MALCInstanceResiduals:
    if len(clean_features) != len(adv_features) or not clean_features:
        raise ValueError("clean and adv feature sequences must be aligned.")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if pag_gate.ndim == 3 and pag_gate.shape[-1] == 1:
        pag_gate = pag_gate[..., 0]
    if target_gt_idx.ndim == 3 and target_gt_idx.shape[-1] == 1:
        target_gt_idx = target_gt_idx[..., 0]
    if assigned_scores.ndim == 3 and assigned_scores.shape[-1] == 1:
        assigned_scores = assigned_scores[..., 0]
    if pag_gate.ndim != 2 or target_gt_idx.shape != pag_gate.shape:
        raise ValueError("pag_gate and target_gt_idx must align as [B,A].")
    if assigned_scores.shape != pag_gate.shape:
        raise ValueError("assigned_scores must align with pag_gate as [B,A].")
    if not torch.isfinite(assigned_scores).all():
        raise ValueError("assigned_scores must be finite.")
    if bool((assigned_scores < 0).any()):
        raise ValueError("assigned_scores must be non-negative.")

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
        if not torch.isfinite(clean).all() or not torch.isfinite(adv).all():
            raise ValueError(f"Scale {scale} features must be finite.")
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
    dtype = adv_features[0].dtype
    image_indices = torch.tensor(
        [key[0] for key in instance_keys], device=device, dtype=torch.long
    )
    gt_indices = torch.tensor(
        [key[1] for key in instance_keys], device=device, dtype=torch.long
    )
    gate_all = pag_gate.to(device=device).bool()
    gt_idx_all = target_gt_idx.to(device=device).long()
    scores_all = assigned_scores.detach().to(device=device, dtype=dtype)

    vectors = []
    assigned_flags = []
    pooling_flags = []
    assignment_counts = []
    score_masses = []
    offset = 0
    for clean, adv, layer_size in zip(
        clean_features, adv_features, layer_sizes
    ):
        residual = (adv - clean.detach()).flatten(2)
        scale_gate = gate_all[:, offset : offset + layer_size]
        scale_gt_idx = gt_idx_all[:, offset : offset + layer_size]
        scale_scores = scores_all[:, offset : offset + layer_size]
        scale_vectors = []
        scale_assigned = []
        scale_pooling = []
        scale_counts = []
        scale_masses = []
        for image_index, gt_index in instance_keys:
            gate = scale_gate[image_index] & (
                scale_gt_idx[image_index] == gt_index
            )
            count = gate.sum()
            weights = scale_scores[image_index] * gate.to(dtype=dtype)
            mass = weights.sum()
            assigned = count > 0
            pooling_valid = assigned & torch.isfinite(mass) & (mass > epsilon)
            normalized = weights / mass.clamp_min(epsilon)
            vector = (residual[image_index] * normalized.unsqueeze(0)).sum(dim=1)
            scale_vectors.append(
                torch.where(pooling_valid, vector, torch.zeros_like(vector))
            )
            scale_assigned.append(assigned)
            scale_pooling.append(pooling_valid)
            scale_counts.append(count.to(dtype=dtype))
            scale_masses.append(mass)

        channels = adv.shape[1]
        vectors.append(
            torch.stack(scale_vectors)
            if scale_vectors
            else adv.new_zeros((0, channels))
        )
        assigned_flags.append(
            torch.stack(scale_assigned)
            if scale_assigned
            else torch.zeros((0,), device=device, dtype=torch.bool)
        )
        pooling_flags.append(
            torch.stack(scale_pooling)
            if scale_pooling
            else torch.zeros((0,), device=device, dtype=torch.bool)
        )
        assignment_counts.append(
            torch.stack(scale_counts)
            if scale_counts
            else adv.new_zeros((0,))
        )
        score_masses.append(
            torch.stack(scale_masses)
            if scale_masses
            else adv.new_zeros((0,))
        )
        offset += layer_size

    return MALCInstanceResiduals(
        vectors=tuple(vectors),
        assigned=tuple(assigned_flags),
        pooling_valid=tuple(pooling_flags),
        assignment_count=tuple(assignment_counts),
        score_mass=tuple(score_masses),
        image_indices=image_indices,
        gt_indices=gt_indices,
    )


def multi_scale_assignment_latent_concentration(
    residuals: MALCInstanceResiduals,
    bank: FrozenMALCPrototypeBank,
) -> MALCResult:
    scale_count = len(residuals.vectors)
    if scale_count != len(bank.direction_prototypes):
        raise ValueError("MALC residual and prototype scale counts differ.")
    if not (
        len(residuals.assigned)
        == len(residuals.pooling_valid)
        == len(residuals.assignment_count)
        == len(residuals.score_mass)
        == scale_count
    ):
        raise ValueError("MALC residual fields have inconsistent scale counts.")

    total_instances = int(residuals.image_indices.numel())
    device = residuals.vectors[0].device
    instance_cosines: list[list[torch.Tensor]] = [
        [] for _ in range(total_instances)
    ]
    instance_log_energies: list[list[torch.Tensor]] = [
        [] for _ in range(total_instances)
    ]
    instance_valid = torch.zeros(total_instances, device=device, dtype=torch.bool)

    scale_losses = []
    direction_losses = []
    magnitude_losses = []
    floor_losses = []
    per_scale_valid_count = []
    per_scale_assigned_count = []
    valid_scale_flags = []
    zero_count = 0
    assigned_entry_count = 0
    floor_pass_count = 0

    for scale, (vectors, assigned, pooling_valid) in enumerate(
        zip(residuals.vectors, residuals.assigned, residuals.pooling_valid)
    ):
        if vectors.ndim != 2 or vectors.shape[0] != total_instances:
            raise ValueError(f"Scale {scale} vectors must be [instances,channels].")
        if assigned.shape != (total_instances,) or pooling_valid.shape != assigned.shape:
            raise ValueError(f"Scale {scale} assignment flags are misaligned.")
        prototype = bank.direction_prototypes[scale].to(vectors)
        if prototype.numel() != vectors.shape[1]:
            raise ValueError(f"Scale {scale} prototype channel count mismatch.")

        finite_vectors = torch.isfinite(vectors).all(dim=1)
        if bool((pooling_valid & ~finite_vectors).any()):
            raise ValueError(f"Scale {scale} contains non-finite pooled residuals.")
        rms = vectors.square().mean(dim=1).add(bank.epsilon**2).sqrt()
        raw_norm = vectors.detach().norm(dim=1)
        direction_valid = pooling_valid & finite_vectors & (raw_norm > bank.epsilon)
        magnitude_valid = pooling_valid & finite_vectors
        floor_valid = assigned & finite_vectors
        valid_scale = bool(assigned.any())
        valid_scale_flags.append(valid_scale)
        per_scale_valid_count.append(int(direction_valid.sum().item()))
        assigned_count = int(assigned.sum().item())
        per_scale_assigned_count.append(assigned_count)
        assigned_entry_count += assigned_count
        zero_count += int((assigned & (raw_norm <= bank.epsilon)).sum().item())
        floor_pass_count += int(
            (floor_valid & (rms.detach() >= bank.energy_floors[scale])).sum().item()
        )

        zero = vectors.sum() * 0.0
        if bool(direction_valid.any()):
            normalized = F.normalize(
                vectors[direction_valid], dim=1, eps=bank.epsilon
            )
            cosines = (normalized * prototype.unsqueeze(0)).sum(dim=1)
            direction_loss = (1.0 - cosines).mean()
            for local, index in enumerate(torch.where(direction_valid)[0].tolist()):
                instance_cosines[index].append(cosines[local].detach())
                instance_valid[index] = True
        else:
            direction_loss = zero

        if bool(magnitude_valid.any()):
            log_ratio = torch.log(
                (rms[magnitude_valid] + bank.epsilon)
                / (bank.median_rms[scale] + bank.epsilon)
            )
            magnitude_loss = F.smooth_l1_loss(
                log_ratio, torch.zeros_like(log_ratio), reduction="mean"
            )
            for local, index in enumerate(torch.where(magnitude_valid)[0].tolist()):
                instance_log_energies[index].append(log_ratio[local].detach())
        else:
            magnitude_loss = zero

        if bool(floor_valid.any()):
            floor = vectors.new_tensor(bank.energy_floors[scale])
            floor_loss = (
                F.relu(floor - rms[floor_valid]) / (floor + bank.epsilon)
            ).mean()
        else:
            floor_loss = zero

        direction_losses.append(direction_loss)
        magnitude_losses.append(magnitude_loss)
        floor_losses.append(floor_loss)
        scale_losses.append(direction_loss + magnitude_loss + floor_loss)

    if not any(valid_scale_flags):
        raise RuntimeError("MALC requires at least one assigned person instance.")
    active_indices = [
        index for index, is_valid in enumerate(valid_scale_flags) if is_valid
    ]
    loss = torch.stack([scale_losses[index] for index in active_indices]).mean()
    direction_loss = torch.stack(
        [direction_losses[index] for index in active_indices]
    ).mean()
    magnitude_loss = torch.stack(
        [magnitude_losses[index] for index in active_indices]
    ).mean()
    floor_loss = torch.stack(
        [floor_losses[index] for index in active_indices]
    ).mean()

    detached_scale_magnitudes = torch.stack(
        [scale_losses[index].detach().abs() for index in active_indices]
    )
    denominator = detached_scale_magnitudes.sum()
    if float(denominator) > bank.epsilon:
        active_shares = detached_scale_magnitudes / denominator
    else:
        active_shares = torch.full_like(
            detached_scale_magnitudes, 1.0 / len(active_indices)
        )
    scale_shares = [0.0] * scale_count
    for local, index in enumerate(active_indices):
        scale_shares[index] = float(active_shares[local])

    def detached_instance_stat(values: list[torch.Tensor]) -> torch.Tensor:
        if values:
            return torch.stack(values).median()
        return residuals.vectors[0].new_tensor(float("nan"))

    per_instance_cosine = torch.stack(
        [detached_instance_stat(values) for values in instance_cosines]
    ) if total_instances else residuals.vectors[0].new_empty((0,))
    per_instance_log_energy = torch.stack(
        [detached_instance_stat(values) for values in instance_log_energies]
    ) if total_instances else residuals.vectors[0].new_empty((0,))
    denominator_instances = max(total_instances, 1)
    denominator_entries = max(assigned_entry_count, 1)
    return MALCResult(
        loss=loss,
        direction_loss=direction_loss,
        magnitude_loss=magnitude_loss,
        floor_loss=floor_loss,
        per_scale_loss=tuple(scale_losses),
        per_scale_valid_count=tuple(per_scale_valid_count),
        per_scale_assigned_count=tuple(per_scale_assigned_count),
        scale_contribution_share=tuple(scale_shares),
        per_instance_cosine=per_instance_cosine,
        per_instance_log_energy=per_instance_log_energy,
        valid_instance_count=int(instance_valid.sum().item()),
        total_instance_count=total_instances,
        valid_instance_coverage=int(instance_valid.sum().item())
        / denominator_instances,
        zero_norm_ratio=zero_count / denominator_entries,
        floor_pass_ratio=floor_pass_count / denominator_entries,
        valid_scale_count=len(active_indices),
    )
