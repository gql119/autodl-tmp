from __future__ import annotations

from dataclasses import dataclass
import time
import tracemalloc
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ue_framework.data_utils import load_image_rgb_float

from .carrier import CarrierConfig, CarrierResult, apply_object_aligned_carrier
from .episodes import DisjointEpisodeSampler, Episode, ImageRecord
from .gains import ClassGainInput, authorized_learning_gain, carrier_query_loss, target_learning_gain
from .model import ObjectCropDetector, class_loss
from .objective import CoreObjectiveConfig, compose_core_objective, delta_metrics, update_delta
from .virtual_update import functional_forward, model_state_unchanged, virtual_update


@dataclass
class SmokeRunResult:
    delta_obj: torch.Tensor
    loss_rows: list[dict]
    class_rows: list[dict]
    carrier_rows: list[dict]
    gradient_rows: list[dict]
    episode_manifests: list[dict]
    runtime_metrics: dict
    summary: dict


def _load_batch(
    records: Sequence[ImageRecord],
    image_size: int,
    image_loader: Callable[[str], np.ndarray],
) -> tuple[torch.Tensor, tuple[tuple[dict, ...], ...]]:
    images = []
    annotations = []
    for record in records:
        image = torch.from_numpy(image_loader(record.image_path)).permute(2, 0, 1).float()
        image = F.interpolate(image.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
        images.append(image)
        annotations.append(record.annotations)
    return torch.stack(images), tuple(annotations)


def _poison_batch(
    clean: torch.Tensor,
    annotations: Sequence[Sequence[dict]],
    delta_obj: torch.Tensor,
    carrier_config: CarrierConfig,
) -> tuple[torch.Tensor, list[CarrierResult]]:
    results = [
        apply_object_aligned_carrier(image, image_annotations, delta_obj, carrier_config)
        for image, image_annotations in zip(clean, annotations)
    ]
    return torch.stack([result.poisoned for result in results]), results


def _target_box_loss(outputs: dict[str, torch.Tensor], target_class_id: int) -> float:
    mask = outputs["labels"] == int(target_class_id)
    if not bool(mask.any()):
        return 0.0
    return float(F.smooth_l1_loss(outputs["boxes"][mask], outputs["target_boxes"][mask]).detach().item())


def _non_target_drifts(
    clean_outputs: dict[str, torch.Tensor],
    poison_outputs: dict[str, torch.Tensor],
    target_class_id: int,
) -> tuple[float, float, float]:
    mask = clean_outputs["labels"] != int(target_class_id)
    if not bool(mask.any()):
        return 0.0, 0.0, 0.0
    logits = float((clean_outputs["logits"][mask] - poison_outputs["logits"][mask]).abs().mean().detach())
    assignment = float(
        (clean_outputs["logits"][mask].argmax(dim=1) != poison_outputs["logits"][mask].argmax(dim=1))
        .float().mean().detach()
    )
    boxes = float((clean_outputs["boxes"][mask] - poison_outputs["boxes"][mask]).abs().mean().detach())
    return logits, assignment, boxes


def run_smoke_chain(
    records: Sequence[ImageRecord],
    config: dict,
    virtual_steps: int,
    outer_steps: int,
    seed: int,
    image_loader: Callable[[str], np.ndarray] = load_image_rgb_float,
) -> SmokeRunResult:
    torch.manual_seed(int(seed))
    model = ObjectCropDetector(num_classes=int(config["num_classes"]), **config["model"])
    model.requires_grad_(False)
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    carrier_options = dict(config["carrier"])
    object_resolution = int(carrier_options.pop("object_resolution"))
    carrier_config = CarrierConfig(target_class_id=int(config["target_class_id"]), **carrier_options)
    generator = torch.Generator().manual_seed(int(seed))
    delta_obj = torch.nn.Parameter(
        torch.empty((3, object_resolution, object_resolution)).uniform_(-0.005, 0.005, generator=generator)
    )
    initial_delta = delta_obj.detach().clone()
    objective_config = CoreObjectiveConfig(**config["objective"])
    optimizer = torch.optim.Adam([delta_obj], lr=float(config["optimization"]["outer_learning_rate"]))
    sampler = DisjointEpisodeSampler(
        records,
        target_class_id=int(config["target_class_id"]),
        num_classes=int(config["num_classes"]),
        support_size=int(config["episode"]["support_size"]),
        query_size=int(config["episode"]["query_size"]),
        minimum_class_samples=int(config["gain"]["minimum_class_samples"]),
        seed=int(seed),
    )
    loss_rows: list[dict] = []
    class_rows: list[dict] = []
    carrier_rows: list[dict] = []
    gradient_rows: list[dict] = []
    manifests: list[dict] = []
    invalid_target_count = 0
    valid_authorized_total = 0
    started = time.perf_counter()
    tracemalloc.start()

    for outer_step in range(int(outer_steps)):
        episode: Episode = sampler.sample(episode_index=outer_step)
        support_clean, support_annotations = _load_batch(
            episode.support_clean, int(config["data"]["image_size"]), image_loader
        )
        query_clean, query_annotations = _load_batch(
            episode.query_clean, int(config["data"]["image_size"]), image_loader
        )
        support_poison, support_carriers = _poison_batch(
            support_clean, support_annotations, delta_obj, carrier_config
        )
        query_poison, query_carriers = _poison_batch(query_clean, query_annotations, delta_obj, carrier_config)

        clean_trajectory = virtual_update(
            model, support_clean, support_annotations, virtual_steps,
            float(config["virtual_update"]["learning_rate"]),
            mode=str(config["virtual_update"]["mode"]),
            selected_modules=config["virtual_update"].get("selected_modules"),
            first_order=bool(config["virtual_update"]["first_order"]),
        )
        poison_trajectory = virtual_update(
            model, support_poison, support_annotations, virtual_steps,
            float(config["virtual_update"]["learning_rate"]),
            mode=str(config["virtual_update"]["mode"]),
            selected_modules=config["virtual_update"].get("selected_modules"),
            first_order=bool(config["virtual_update"]["first_order"]),
        )
        initial_outputs = model(query_clean, query_annotations)
        clean_outputs = functional_forward(
            model, clean_trajectory.parameters, clean_trajectory.buffers, query_clean, query_annotations
        )
        poison_outputs = functional_forward(
            model, poison_trajectory.parameters, poison_trajectory.buffers, query_clean, query_annotations
        )
        poison_query_outputs = functional_forward(
            model, poison_trajectory.parameters, poison_trajectory.buffers, query_poison, query_annotations
        )
        target_id = int(config["target_class_id"])
        initial_target_loss, target_query_count = class_loss(initial_outputs, target_id)
        clean_target_loss, _ = class_loss(clean_outputs, target_id)
        poison_target_loss, _ = class_loss(poison_outputs, target_id)
        target_gain = target_learning_gain(
            initial_target_loss, clean_target_loss, poison_target_loss,
            rho_t=float(config["gain"]["rho_t"]),
            min_valid_clean_gain=float(config["gain"]["min_valid_clean_gain"]),
            eps=float(config["gain"]["eps"]),
        )
        if not target_gain.valid:
            invalid_target_count += 1
        carrier_target_loss, _ = class_loss(poison_query_outputs, target_id)
        carrier_loss = carrier_query_loss(carrier_target_loss)

        class_inputs = {}
        for class_id in range(int(config["num_classes"])):
            if class_id == target_id:
                continue
            initial_class_loss, query_count = class_loss(initial_outputs, class_id)
            clean_class_loss, _ = class_loss(clean_outputs, class_id)
            poison_class_loss, _ = class_loss(poison_outputs, class_id)
            class_inputs[class_id] = ClassGainInput(
                initial_class_loss, clean_class_loss, poison_class_loss,
                episode.class_counts[class_id]["support"], query_count,
            )
        authorized = authorized_learning_gain(
            class_inputs,
            target_class_id=target_id,
            rho_k=float(config["gain"]["rho_k"]),
            min_valid_class_gain=float(config["gain"]["min_valid_class_gain"]),
            minimum_class_samples=int(config["gain"]["minimum_class_samples"]),
            eps=float(config["gain"]["eps"]),
        )
        valid_authorized_total += len(authorized.valid_class_ids)
        objective = compose_core_objective(
            target_gain.protect_loss, carrier_loss, authorized.loss, delta_obj, objective_config
        )
        total_before = float(objective.loss.detach())
        component_values = {name: float(value.detach()) for name, value in objective.components.items()}
        gradient_norm = update_delta(objective, delta_obj, optimizer, objective_config)
        current_delta = delta_metrics(delta_obj, objective_config.eps)

        with torch.no_grad():
            clean_base = model(query_clean, query_annotations)
            poison_base = model(query_poison.detach(), query_annotations)
            nt_logits_drift, nt_assignment_drift, nt_box_drift = _non_target_drifts(
                clean_base, poison_base, target_id
            )
        all_carriers = [("support", result) for result in support_carriers] + [
            ("query", result) for result in query_carriers
        ]
        target_instances = sum(result.metrics["target_instances"] for _, result in all_carriers)
        applied_instances = sum(result.metrics["applied_instances"] for _, result in all_carriers)
        coverage = applied_instances / max(1, target_instances)
        overlap_count = len(set(episode.support_ids) & set(episode.query_ids))
        loss_rows.append({
            "variant_j": int(virtual_steps),
            "outer_step": outer_step,
            "L_core": total_before,
            **component_values,
            "G_t_clean": float(target_gain.clean_gain.detach()),
            "G_t_poison": float(target_gain.poison_gain.detach()),
            "target_gain_ratio": "" if target_gain.ratio is None else float(target_gain.ratio.detach()),
            "target_gain_valid": int(target_gain.valid),
            "target_gain_invalid_reason": target_gain.invalid_reason,
            "target_query_count": target_query_count,
            "valid_authorized_class_count": len(authorized.valid_class_ids),
            "support_query_overlap": overlap_count,
            "target_assignment_coverage_proxy": coverage,
            "target_box_loss_proxy": _target_box_loss(clean_base, target_id),
            "target_dfl_loss": 0.0,
            "target_dfl_available": 0,
            "non_target_logits_drift": nt_logits_drift,
            "non_target_assignment_drift_proxy": nt_assignment_drift,
            "non_target_box_drift_proxy": nt_box_drift,
            "gradient_norm": gradient_norm,
            **current_delta,
        })
        for class_id, result in authorized.classes.items():
            class_rows.append({
                "variant_j": int(virtual_steps),
                "outer_step": outer_step,
                "class_id": class_id,
                "G_k_clean": float(result.clean_gain.detach()),
                "G_k_poison": float(result.poison_gain.detach()),
                "authorized_gain_gap": "" if result.normalized_gap is None else float(result.normalized_gap.detach()),
                "valid": int(result.valid),
                "invalid_reason": result.invalid_reason,
                "support_count": result.support_count,
                "query_count": result.query_count,
            })
        for branch, result in all_carriers:
            carrier_rows.append({
                "variant_j": int(virtual_steps),
                "outer_step": outer_step,
                "branch": branch,
                **result.metrics,
            })
        gradient_rows.append({
            "variant_j": int(virtual_steps),
            "outer_step": outer_step,
            "gradient_norm": gradient_norm,
            "finite": 1,
        })
        manifests.append({
            "variant_j": int(virtual_steps),
            "outer_step": outer_step,
            "support_ids": episode.support_ids,
            "query_ids": episode.query_ids,
            "overlap_count": overlap_count,
            "valid_authorized_class_ids": authorized.valid_class_ids,
        })

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    final_metrics = delta_metrics(delta_obj, objective_config.eps)
    summary = {
        "virtual_steps": int(virtual_steps),
        "outer_steps": int(outer_steps),
        "forward_complete": True,
        "backward_complete": True,
        "delta_updated": bool(not torch.equal(initial_delta, delta_obj.detach())),
        "delta_change_norm": float((delta_obj.detach() - initial_delta).norm().item()),
        "base_model_unchanged": model_state_unchanged(model, base_state),
        "support_query_overlap_max": max(row["support_query_overlap"] for row in loss_rows),
        "target_gain_computable_count": sum(row["target_gain_valid"] for row in loss_rows),
        "valid_authorized_class_total": valid_authorized_total,
        "invalid_target_gain_ratio": invalid_target_count / max(1, int(outer_steps)),
        "finite": bool(all(np.isfinite(float(row["L_core"])) for row in loss_rows) and final_metrics["finite"]),
        **final_metrics,
    }
    runtime = {
        "seconds": elapsed,
        "device": "cpu",
        "python_tracemalloc_current_bytes": current_memory,
        "python_tracemalloc_peak_bytes": peak_memory,
        "cuda_peak_bytes": 0,
    }
    return SmokeRunResult(
        delta_obj.detach().cpu(), loss_rows, class_rows, carrier_rows, gradient_rows, manifests, runtime, summary
    )
