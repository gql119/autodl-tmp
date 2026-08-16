from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NonTargetDistributionDivergence:
    js_per_anchor: torch.Tensor
    clean_to_poison_kl_per_anchor: torch.Tensor


def _bernoulli_kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return probability * torch.log(probability / reference) + (
        1.0 - probability
    ) * torch.log((1.0 - probability) / (1.0 - reference))


def non_target_bernoulli_divergence(
    clean_logits: torch.Tensor,
    poison_logits: torch.Tensor,
    *,
    target_class_id: int,
    temperature: float = 2.0,
    epsilon: float = 1.0e-6,
) -> NonTargetDistributionDivergence:
    """Compare independent non-target class probabilities at aligned anchors.

    YOLOv8 uses sigmoid class outputs, so treating the class vector as a
    single-label softmax distribution would introduce a false exclusivity
    constraint. The clean branch is a detached teacher. The returned tensors
    preserve every leading dimension and average only over non-target classes.
    """

    if clean_logits.shape != poison_logits.shape or clean_logits.ndim < 1:
        raise ValueError("Clean/poison logits must have the same [..., C] shape.")
    class_count = int(clean_logits.shape[-1])
    if class_count < 2:
        raise ValueError("At least two detector classes are required.")
    if not 0 <= int(target_class_id) < class_count:
        raise ValueError("target_class_id is outside the detector class range.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5).")
    if not torch.isfinite(clean_logits).all() or not torch.isfinite(
        poison_logits
    ).all():
        raise ValueError("Distribution-divergence logits must be finite.")

    clean_probability = torch.sigmoid(
        clean_logits.detach() / float(temperature)
    ).clamp(float(epsilon), 1.0 - float(epsilon))
    poison_probability = torch.sigmoid(
        poison_logits / float(temperature)
    ).clamp(float(epsilon), 1.0 - float(epsilon))
    non_target_mask = torch.arange(
        class_count, device=poison_logits.device
    ) != int(target_class_id)
    clean_non_target = clean_probability[..., non_target_mask]
    poison_non_target = poison_probability[..., non_target_mask]
    midpoint = 0.5 * (clean_non_target + poison_non_target)

    clean_to_midpoint = _bernoulli_kl(clean_non_target, midpoint)
    poison_to_midpoint = _bernoulli_kl(poison_non_target, midpoint)
    js_per_anchor = 0.5 * (
        clean_to_midpoint + poison_to_midpoint
    ).mean(dim=-1)
    clean_to_poison = _bernoulli_kl(
        clean_non_target, poison_non_target
    ).mean(dim=-1)
    if not torch.isfinite(js_per_anchor).all() or not torch.isfinite(
        clean_to_poison
    ).all():
        raise ValueError("Non-finite Bernoulli divergence.")
    return NonTargetDistributionDivergence(
        js_per_anchor=js_per_anchor,
        clean_to_poison_kl_per_anchor=clean_to_poison,
    )
