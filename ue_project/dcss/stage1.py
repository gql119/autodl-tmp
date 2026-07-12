import csv
import json
import logging
import math
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from ue_framework.data_utils import (
    copy_label,
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
    save_image_rgb_float,
)
from ue_framework.methods.alce_acgt import project_strict_gate_to_fpn
from ue_framework.methods.tausb_universal import TAUSBMaskGenerator, TAUSBUniversalTrainer

from .losses import dcss_stage1_loss, symmetric_bernoulli_kl
from .stage0_collection import _batch_from_annotations, _gather_vectors, _letterbox_with_annotations
from .subspace_io import load_subspaces
from .unit_partition import partition_tal_units


def _letterbox_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    height, width = mask.shape
    scale = min(image_size / float(height), image_size / float(width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="nearest")
    top = (image_size - new_h) // 2
    bottom = image_size - new_h - top
    left = (image_size - new_w) // 2
    right = image_size - new_w - left
    return F.pad(tensor, (left, right, top, bottom), value=0.0)[0]


def _combine_batch(adjusted_annotations: List[List[dict]], images: torch.Tensor, device: torch.device) -> Dict:
    batch_indices, classes, boxes = [], [], []
    for batch_index, annotations in enumerate(adjusted_annotations):
        for annotation in annotations:
            batch_indices.append(batch_index)
            classes.append([float(annotation["cls"])])
            boxes.append(annotation["bbox"])
    return {
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long, device=device),
        "cls": torch.tensor(classes, dtype=torch.float32, device=device) if classes else torch.zeros((0, 1), device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device) if boxes else torch.zeros((0, 4), device=device),
        "batch_size": images.shape[0],
        "img": images,
    }


def _quantiles(values: torch.Tensor) -> Tuple[float, float, float]:
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    values = values.detach().float().cpu()
    return float(values.mean()), float(torch.quantile(values, 0.1)), float(torch.quantile(values, 0.9))


def _shift_geometry(shifts: torch.Tensor) -> Tuple[float, float]:
    if shifts.shape[0] < 2:
        return 1.0, 1.0
    normalized = F.normalize(shifts, dim=1, eps=1e-8)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    pairwise = float(cosine[mask].mean().detach().item())
    singular = torch.linalg.svdvals(shifts - shifts.mean(dim=0, keepdim=True))
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).detach().item())
    return pairwise, effective_rank


def _load_basis(subspace_path: str, layer: str, source: str, rank: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    payload = load_subspaces(subspace_path)
    basis = payload["layers"][layer]["subspaces"][source][rank]
    return basis.to(device=device, dtype=torch.float32), payload


def train_stage1_poison(
    cfg: Dict,
    experiment_dir: str,
    source: str,
    legacy_method_cfg: Dict,
) -> str:
    os.makedirs(experiment_dir, exist_ok=True)
    logger = logging.getLogger(f"dcss.stage1.{os.path.basename(experiment_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(experiment_dir, "poison_generation.log"), encoding="utf-8"))
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(cfg.get("deterministic", True)), warn_only=True)
    device = torch.device(cfg.get("device", "cuda:0"))
    wrapper = YOLO(cfg["surrogate"]["checkpoint"])
    model = wrapper.model.to(device).eval()
    method_cfg = dict(legacy_method_cfg)
    method_cfg.update(cfg["dcss"].get("carrier_overrides", {}))
    method_cfg["legacy_best_reproduce_mode"] = True
    method_cfg["force_pseudo_mask_fallback"] = True
    method_cfg["universal_epochs"] = int(cfg["dcss"]["poison_epochs"])
    method_cfg["universal_batch_size"] = int(cfg["dcss"]["batch_size"])
    trainer = TAUSBUniversalTrainer(cfg, method_cfg, device, model)
    trainer.hijacked.enable_strict_assign_probe = False
    with torch.no_grad():
        if float(trainer.fourier_coeff.abs().max().item()) == 0.0:
            initialization_generator = torch.Generator(device=device).manual_seed(seed)
            trainer.fourier_coeff.normal_(mean=0.0, std=1e-3, generator=initialization_generator)

    layer = str(cfg["dcss"]["layer"])
    rank = int(cfg["dcss"]["rank"])
    basis, subspace_payload = _load_basis(cfg["dcss"]["subspace_path"], layer, source, rank, device)
    checkpoint_expected = os.path.abspath(cfg["surrogate"]["checkpoint"])
    if os.path.abspath(subspace_payload["checkpoint"]) != checkpoint_expected:
        raise RuntimeError("Stage 0 subspace checkpoint does not match Stage 1 surrogate checkpoint")
    dataset_root = cfg["data"]["dataset_root"]
    image_dir = os.path.join(dataset_root, cfg["data"].get("train_images", "images/train"))
    label_dir = os.path.join(dataset_root, cfg["data"].get("train_labels", "labels/train"))
    target_images = []
    for path in list_images(image_dir):
        annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
        if image_has_target(annotations, int(cfg["experiment"]["target_class_id"])):
            target_images.append(path)
    if not target_images:
        raise RuntimeError("no target images for DCSS Stage 1")
    maximum_target_images = int(cfg["dcss"].get("max_target_images", 0))
    if maximum_target_images > 0:
        target_images = target_images[:maximum_target_images]

    optimizer = torch.optim.Adam([trainer.fourier_coeff], lr=float(cfg["dcss"]["learning_rate"]))
    diagnostics = []
    class_accumulator = defaultdict(lambda: {"leakage_sum": 0.0, "logit_sum": 0.0, "batches": 0})
    global_step = 0
    selected_target_total = 0
    target_positive_total = 0
    for epoch in range(int(cfg["dcss"]["poison_epochs"])):
        generator = random.Random(seed + epoch)
        order = list(range(len(target_images)))
        generator.shuffle(order)
        for start in range(0, len(order), int(cfg["dcss"]["batch_size"])):
            indices = order[start : start + int(cfg["dcss"]["batch_size"])]
            images, inner_masks, ring_masks, adjusted_batch = [], [], [], []
            for index in indices:
                path = target_images[index]
                image_np = load_image_rgb_float(path)
                annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
                inner, ring, support_source = trainer._build_support(
                    image_np.shape, annotations, "mask", trainer.ring_width, path
                )
                if support_source != "forced_pseudo_fallback":
                    raise RuntimeError(f"unexpected carrier support source: {support_source}")
                image, adjusted = _letterbox_with_annotations(image_np, annotations, trainer.imgsz)
                images.append(image[0])
                inner_masks.append(_letterbox_mask(inner, trainer.imgsz))
                ring_masks.append(_letterbox_mask(ring, trainer.imgsz))
                adjusted_batch.append(adjusted)
            clean = torch.stack(images).to(device)
            inner_t = torch.stack(inner_masks).to(device)
            ring_t = torch.stack(ring_masks).to(device)
            batch = _combine_batch(adjusted_batch, clean, device)
            active = trainer.freq_active_idx.long()
            coords = [tuple(trainer.freq_candidate_coords[int(i)]) for i in active.detach().cpu().tolist()]
            coefficients = trainer.fourier_coeff[active]
            raw, perturbation, adv, _, _ = trainer._compose_delta_batched(
                clean, inner_t, ring_t, coords, coefficients, trainer.suppress_small, current_epoch=epoch
            )

            with torch.no_grad():
                trainer._clear_multi_features()
                clean_predictions = trainer._forward_raw(clean)
                clean_features = {name: value.detach() for name, value in trainer.multi_features.items()}
                trainer.hijacked.get_assigned_targets_and_loss(clean_predictions, batch)
                assignment = dict(trainer.hijacked.last_real_assign)
            if not all(torch.is_tensor(assignment.get(name)) for name in ["fg_mask", "target_labels", "target_scores"]):
                raise RuntimeError("real clean TAL assignment unavailable")
            partition = partition_tal_units(
                assignment["fg_mask"], assignment["target_labels"], assignment["target_scores"],
                int(cfg["experiment"]["target_class_id"]), trainer.shape_layers, clean_features,
                trainer.pag_layer_ratios, trainer.pag_min_pos,
            )
            selected_target_total += int(partition.stats["num_selected_target"])
            target_positive_total += int(partition.stats["num_target_positive"])

            trainer._clear_multi_features()
            adv_predictions = trainer._forward_raw(adv)
            adv_features = dict(trainer.multi_features)
            if basis.shape[0] != adv_features[layer].shape[1]:
                raise RuntimeError(
                    f"subspace/feature dimension mismatch: Q={basis.shape[0]}, feature={adv_features[layer].shape[1]}"
                )
            target_map = project_strict_gate_to_fpn(partition.selected_target_gate, trainer.shape_layers, adv_features)[layer]
            target_shift = _gather_vectors(adv_features[layer] - clean_features[layer], target_map)
            if target_shift.shape[0] == 0:
                continue
            non_target_shifts = []
            class_losses = []
            class_batch_metrics = {}
            for class_id, class_gate in partition.non_target_class_gates.items():
                class_map = project_strict_gate_to_fpn(class_gate, trainer.shape_layers, adv_features)[layer]
                class_shift = _gather_vectors(adv_features[layer] - clean_features[layer], class_map)
                if class_shift.shape[0] == 0:
                    continue
                non_target_shifts.append(class_shift)
                projected = (class_shift @ basis).square().sum(dim=1).mean()
                clean_probability = clean_predictions[:, 4 + class_id, :][class_gate].detach().clamp(1e-6, 1 - 1e-6)
                adv_probability = adv_predictions[:, 4 + class_id, :][class_gate].clamp(1e-6, 1 - 1e-6)
                logit_drift = symmetric_bernoulli_kl(torch.logit(clean_probability), torch.logit(adv_probability))
                class_losses.append(logit_drift)
                class_batch_metrics[class_id] = (projected, logit_drift)
            non_target_shift = torch.cat(non_target_shifts, dim=0) if non_target_shifts else target_shift.new_zeros((0, target_shift.shape[1]))
            nt_logit_loss = torch.stack(class_losses).mean() if class_losses else target_shift.new_zeros(())
            budget_regularizer = (raw - raw.clamp(-trainer.eps, trainer.eps)).square().mean()
            total, mechanism = dcss_stage1_loss(
                target_shift, non_target_shift, basis, float(cfg["dcss"]["energy_margin"]),
                regularizer=budget_regularizer,
                lambda_energy=float(cfg["dcss"]["lambda_energy"]),
                lambda_outside=float(cfg["dcss"]["lambda_outside"]),
                lambda_leakage=float(cfg["dcss"]["lambda_leakage"]),
                lambda_logits=0.0,
                lambda_regularizer=float(cfg["dcss"]["lambda_regularizer"]),
            )
            total = total + float(cfg["dcss"]["lambda_logits"]) * nt_logit_loss
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite DCSS Stage 1 loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if trainer.fourier_coeff.grad is None:
                raise RuntimeError(
                    "DCSS carrier gradient missing: "
                    f"total_requires_grad={total.requires_grad}, adv_requires_grad={adv.requires_grad}, "
                    f"perturbation_requires_grad={perturbation.requires_grad}, coefficients_requires_grad={coefficients.requires_grad}"
                )
            if not torch.isfinite(trainer.fourier_coeff.grad).all():
                finite_ratio = float(torch.isfinite(trainer.fourier_coeff.grad).float().mean().item())
                raise RuntimeError(f"DCSS carrier gradient non-finite: finite_ratio={finite_ratio:.6f}")
            optimizer.step()

            projected = (target_shift @ basis).square().sum(dim=1)
            projected_mean, projected_p10, projected_p90 = _quantiles(projected)
            outside = target_shift - (target_shift @ basis) @ basis.T
            outside_energy = outside.square().sum(dim=1)
            total_energy = target_shift.square().sum(dim=1)
            leakage_values = {class_id: float(value[0].detach().item()) for class_id, value in class_batch_metrics.items()}
            logit_values = {class_id: float(value[1].detach().item()) for class_id, value in class_batch_metrics.items()}
            for class_id in leakage_values:
                class_accumulator[class_id]["leakage_sum"] += leakage_values[class_id]
                class_accumulator[class_id]["logit_sum"] += logit_values[class_id]
                class_accumulator[class_id]["batches"] += 1
            leakage_sorted = sorted(leakage_values.values())
            leakage_p95 = leakage_sorted[min(len(leakage_sorted) - 1, int(0.95 * len(leakage_sorted)))] if leakage_sorted else 0.0
            pairwise, effective_rank = _shift_geometry(target_shift)
            diagnostics.append({
                "epoch": epoch,
                "step": global_step,
                "loss": float(total.detach().item()),
                "energy_loss": float(mechanism["energy_loss"].detach().item()),
                "weighted_energy_loss": float(cfg["dcss"]["lambda_energy"]) * float(mechanism["energy_loss"].detach().item()),
                "weighted_outside_loss": float(cfg["dcss"]["lambda_outside"]) * float(mechanism["target_outside_energy"].detach().item()),
                "weighted_leakage_loss": float(cfg["dcss"]["lambda_leakage"]) * float(mechanism["non_target_leakage"].detach().item()),
                "weighted_logit_loss": float(cfg["dcss"]["lambda_logits"]) * float(nt_logit_loss.detach().item()),
                "dcss_layer": layer,
                "dcss_rank": rank,
                "target_projected_energy_mean": projected_mean,
                "target_projected_energy_p10": projected_p10,
                "target_projected_energy_p90": projected_p90,
                "target_outside_energy_mean": float(outside_energy.mean().detach().item()),
                "target_shift_total_energy": float(total_energy.mean().detach().item()),
                "target_in_subspace_ratio": float((projected.sum() / total_energy.sum().clamp_min(1e-12)).detach().item()),
                "non_target_projected_leakage_mean": float(np.mean(leakage_sorted)) if leakage_sorted else 0.0,
                "non_target_projected_leakage_max_class": max(leakage_values, key=leakage_values.get) if leakage_values else -1,
                "non_target_projected_leakage_p95_class": leakage_p95,
                "target_non_target_energy_ratio": projected_mean / (float(np.mean(leakage_sorted)) + 1e-12) if leakage_sorted else float("inf"),
                "target_unit_coverage": partition.stats["target_unit_coverage"],
                "target_unit_coverage_running": selected_target_total / max(1, target_positive_total),
                "target_assignment_overlap": 1.0,
                "target_shift_pairwise_cosine": pairwise,
                "target_shift_effective_rank": effective_rank,
                "perturbation_area_ratio": float((perturbation.detach().abs().amax(dim=1) > (1.0 / 255.0)).float().mean().item()),
                "perturbation_max_amplitude": float(perturbation.detach().abs().max().item()),
                "subspace_source": source,
                "subspace_checkpoint": subspace_payload["checkpoint"],
                **{f"leakage_class_{class_id}": value for class_id, value in leakage_values.items()},
                **{f"logit_drift_class_{class_id}": value for class_id, value in logit_values.items()},
            })
            global_step += 1
        logger.info("epoch=%d steps=%d latest_loss=%.6f", epoch, global_step, diagnostics[-1]["loss"])

    final_coverage = selected_target_total / max(1, target_positive_total)
    if final_coverage < 0.5:
        raise RuntimeError(f"global target unit coverage below 50%: {final_coverage:.4f}")

    active = trainer.freq_active_idx.long()
    save_pack = {
        "coords": [list(trainer.freq_candidate_coords[int(i)]) for i in active.detach().cpu().tolist()],
        "fourier_coeff": trainer.fourier_coeff.detach()[active].cpu(),
        "suppress_small": trainer.suppress_small.detach().cpu(),
        "method": "dcss",
        "target_class_id": trainer.target_class_id,
        "dcss_layer": layer,
        "dcss_rank": rank,
        "subspace_source": source,
        "subspace_checkpoint": subspace_payload["checkpoint"],
    }
    checkpoint_path = os.path.join(experiment_dir, "poison_checkpoint.pt")
    torch.save(save_pack, checkpoint_path)
    fieldnames = sorted({key for row in diagnostics for key in row})
    with open(os.path.join(experiment_dir, "mechanism_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(diagnostics)
    class_rows = []
    for class_id in range(int(cfg["surrogate"]["num_classes"])):
        state = class_accumulator.get(class_id, {"leakage_sum": 0.0, "logit_sum": 0.0, "batches": 0})
        denominator = max(1, state["batches"])
        class_rows.append({"class_id": class_id, "batches": state["batches"], "leakage": state["leakage_sum"] / denominator, "logit_drift": state["logit_sum"] / denominator})
    with open(os.path.join(experiment_dir, "classwise_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(class_rows[0])); writer.writeheader(); writer.writerows(class_rows)
    return checkpoint_path


def materialize_dataset(cfg: Dict, experiment_dir: str, checkpoint_path: str, legacy_method_cfg: Dict) -> str:
    device = torch.device(cfg.get("device", "cuda:0"))
    wrapper = YOLO(cfg["surrogate"]["checkpoint"])
    model = wrapper.model.to(device).eval()
    method_cfg = dict(legacy_method_cfg)
    method_cfg["legacy_best_reproduce_mode"] = True
    method_cfg["force_pseudo_mask_fallback"] = True
    generator = TAUSBMaskGenerator(cfg, method_cfg, device, model, checkpoint_path)
    dataset_root = cfg["data"]["dataset_root"]
    image_dir = os.path.join(dataset_root, cfg["data"].get("train_images", "images/train"))
    label_dir = os.path.join(dataset_root, cfg["data"].get("train_labels", "labels/train"))
    output_root = os.path.join(experiment_dir, "poisoned_dataset")
    output_images = os.path.join(output_root, "images", "train")
    output_labels = os.path.join(output_root, "labels", "train")
    os.makedirs(output_images, exist_ok=True); os.makedirs(output_labels, exist_ok=True)
    rows = []
    for index, path in enumerate(list_images(image_dir)):
        image = load_image_rgb_float(path)
        annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
        has_target = image_has_target(annotations, int(cfg["experiment"]["target_class_id"]))
        stem = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_images, stem + ".png")
        label_path = label_path_for_image(path, label_dir)
        if has_target:
            result = generator.generate(image, annotations, int(cfg["seed"]) + index, int(cfg["dcss"]["poison_epochs"]), float(cfg["experiment"]["eps"]), "mask", path)
            poisoned = result.poisoned_image
            support = result.support_mask
            support_source = result.extras.get("support_source", "unknown")
        else:
            poisoned = image
            support = np.zeros(image.shape[:2], dtype=np.float32)
            support_source = "none"
        save_image_rgb_float(output_path, poisoned)
        copy_label(label_path, os.path.join(output_labels, os.path.basename(label_path)))
        difference = np.max(np.abs(poisoned - image), axis=2)
        mse = float(np.mean(np.square(poisoned - image)))
        linf = float(np.max(np.abs(poisoned - image)))
        rows.append({
            "stem": stem, "image_path": output_path, "is_poisoned": int(has_target), "poisoned": int(has_target),
            "has_target": int(has_target), "support_ratio": float(np.mean(support > 0.5)),
            "perturbed_area_ratio": float(np.mean(difference > 1 / 255)), "linf": linf,
            "psnr": 99.0 if mse <= 1e-12 else 10 * math.log10(1 / mse), "method": "dcss",
            "steps": int(cfg["dcss"]["poison_epochs"]), "seed": int(cfg["seed"]), "support_source": support_source,
        })
    with open(os.path.join(output_root, "manifest.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return output_root
