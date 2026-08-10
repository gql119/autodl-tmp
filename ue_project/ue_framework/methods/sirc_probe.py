from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    read_yolo_annotations,
)
from .background_spectral_basis import (
    build_background_spectral_basis,
    spectrum_energy_ratios,
)
from .bsc_icmo_probe import (
    ICMOBatch,
    ICMOObservation,
    ICMOProbeEngine,
    fit_icmo_bank,
    load_icmo_batch,
    stratified_paired_bootstrap,
    summarize_icmo_observations,
)
from .bsc_rc_gr_probe import (
    _file_sha256,
    _resolve_path,
    _write_probe_json,
    canonical_hash,
    load_background_sources,
    load_required_shared_split,
    make_batches,
)
from .cicr import CICRPrototypeBank
from .instance_canonical_carrier import (
    calibrate_shared_gamma,
    common_initial_coefficients,
    tensor_sha256,
)
from .instance_cicr import instance_cicr
from .semantic_residual_carrier import (
    SemanticCarrierBank,
    VariantMatchedCanonicalCarrier,
    build_semantic_carrier_bank,
    calibrate_variant_shared_gamma,
    center_square_resize,
    construct_phase_amplitude_variants,
    stable_variant_index,
)


PHASE_A_ARMS = {
    "I-C2LM": ("baseline", "fixed"),
    "I-SPC-F": ("control", "fixed"),
    "I-SF": ("semantic", "fixed"),
    "I-SPC-V": ("control", "hashed"),
    "I-SV": ("semantic", "hashed"),
}
PHASE_B_ARMS = {
    "I-SV-E0": 0.0,
    "I-SV-E1": 1.0,
}


def validate_sirc_config(config: Mapping[str, Any]) -> None:
    required = {
        "spec",
        "dataset",
        "model",
        "background",
        "carrier",
        "split",
        "optimization",
        "eot",
        "bootstrap",
        "runtime",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing SIRC config sections: {missing}")
    spec_id = str(config["spec"].get("spec_id", ""))
    exp_id = str(config["spec"].get("exp_id", ""))
    malc_contracts = {
        ("TAUSB-SIRC-MALC-CGR-MAP50-v2", "TAUSB-SIRC-MALC-CGR-MAP50-S0"),
        ("TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1", "TAUSB-MALC-GRAD-GEOMETRY-S0"),
    }
    is_malc_v2 = (spec_id, exp_id) in malc_contracts
    valid_contract = (
        (spec_id == "TAUSB-SIRC-v1" and exp_id == "TAUSB-SIRC-MECH-S0")
        or is_malc_v2
    )
    if not valid_contract:
        raise ValueError("SIRC spec_id/exp_id mismatch.")
    if int(config["spec"].get("seed", -1)) != 0:
        raise ValueError("SIRC experiment seed must remain 0.")
    if int(config["dataset"].get("target_class_id", -1)) != 14:
        raise ValueError("SIRC target class must remain person=14.")
    if int(config["model"].get("num_classes", -1)) != 20:
        raise ValueError("SIRC detector must remain VOC20.")
    if int(config["model"].get("image_size", -1)) != 640:
        raise ValueError("SIRC image_size must remain 640.")

    carrier = config["carrier"]
    if abs(float(carrier.get("epsilon", -1)) - 16.0 / 255.0) > 1e-9:
        raise ValueError("SIRC epsilon must remain 16/255.")
    frozen_carrier = {
        "resolution": 640,
        "num_bases": 16,
        "num_variants": 4,
        "phase_seed": 2101,
        "variant_seed": 2102,
        "initial_seed": 2103,
        "gamma_seed": 2104,
        "gamma_directions": 256,
    }
    for key, expected in frozen_carrier.items():
        if int(carrier.get(key, -1)) != expected:
            raise ValueError(f"SIRC carrier.{key} must remain {expected}.")
    if int(carrier.get("baseline_basis_seed", -1)) != 0:
        raise ValueError("SIRC baseline_basis_seed must remain 0.")
    for key in ("gamma_bisection_iterations", "gamma_chunk_size"):
        if int(carrier.get(key, 0)) <= 0:
            raise ValueError(f"SIRC carrier.{key} must be positive.")
    if tuple(float(value) for value in carrier.get("radial_edges", ())) != (
        2.0,
        5.5,
        10.0,
        16.0,
        24.0,
    ):
        raise ValueError("SIRC radial_edges must remain frozen.")
    if abs(float(carrier.get("coefficient_max_abs", -1)) - 0.25) > 1e-12:
        raise ValueError("SIRC coefficient_max_abs must remain 0.25.")
    if abs(float(carrier.get("target_rms_ratio", -1)) - 0.35) > 1e-12:
        raise ValueError("SIRC target_rms_ratio must remain 0.35.")

    optimization = config["optimization"]
    frozen_optimization = {
        "warmup_steps": 4,
        "optimization_steps": 40,
        "batch_size": 4,
    }
    for key, expected in frozen_optimization.items():
        if int(optimization.get(key, -1)) != expected:
            raise ValueError(f"SIRC optimization.{key} must remain {expected}.")
    if abs(float(optimization.get("learning_rate", -1)) - 0.01) > 1e-12:
        raise ValueError("SIRC learning_rate must remain 0.01.")
    if str(optimization.get("target_route")) != "easy_cls":
        raise ValueError("SIRC target route must remain easy_cls.")
    weight_keys = (
        ("lambda_route", "lambda_rms")
        if is_malc_v2
        else ("lambda_cicr", "lambda_route", "lambda_rms")
    )
    for key in weight_keys:
        if abs(float(optimization.get(key, -1)) - 1.0) > 1e-12:
            raise ValueError(f"SIRC optimization.{key} must remain 1.0.")
    if not 0 <= float(optimization.get("prototype_momentum", -1)) < 1:
        raise ValueError("SIRC prototype_momentum must lie in [0,1).")
    for key in ("box_teacher_weight", "align_alpha", "align_beta"):
        if float(optimization.get(key, 0)) <= 0:
            raise ValueError(f"SIRC optimization.{key} must be positive.")
    if int(optimization.get("assignment_topk", 0)) <= 0:
        raise ValueError("SIRC assignment_topk must be positive.")
    if (
        len(optimization.get("pag_layer_ratios", ())) != 3
        or len(optimization.get("pag_min_pos", ())) != 3
    ):
        raise ValueError("SIRC PAG settings must define P3/P4/P5.")

    eot = config["eot"]
    if is_malc_v2:
        if eot.get("enabled") is not False or int(eot.get("samples", -1)) != 1:
            raise ValueError("MALC v2 requires EOT disabled with samples=1.")
    else:
        if int(eot.get("samples", -1)) != 2:
            raise ValueError("SIRC EOT samples must remain 2.")
    if int(eot.get("seed", -1)) != 2105:
        raise ValueError("SIRC EOT seed must remain 2105.")
    expected_ranges = {
        "scale": (0.90, 1.10),
        "translate": (-0.05, 0.05),
        "blur_sigma": (0.4, 0.8),
    }
    for key, expected in expected_ranges.items():
        actual = tuple(float(value) for value in eot.get(key, ()))
        if actual != expected:
            raise ValueError(f"SIRC eot.{key} must remain {expected}.")
    if abs(float(eot.get("grayscale_probability", -1)) - 0.25) > 1e-12:
        raise ValueError("SIRC grayscale probability must remain 0.25.")

    if int(config["bootstrap"].get("seed", -1)) != 2110:
        raise ValueError("SIRC bootstrap seed must remain 2110.")
    if int(config["bootstrap"].get("iterations", -1)) != 10000:
        raise ValueError("SIRC bootstrap iterations must remain 10000.")
    if (
        str(config["split"].get("required_protocol_prefix", ""))
        != "TAUSB-ALCE-CTX-AUDIT-v1"
    ):
        raise ValueError("SIRC must reuse the frozen ALCE shared split.")
    for section, key in (
        ("dataset", "root"),
        ("dataset", "train_images"),
        ("dataset", "train_labels"),
        ("model", "surrogate_checkpoint"),
        ("background", "source_manifest"),
        ("background", "source_local_map"),
        ("split", "manifest"),
        ("runtime", "artifact_root"),
        ("runtime", "device"),
    ):
        if not str(config[section].get(key, "")).strip():
            raise ValueError(f"SIRC config requires {section}.{key}.")
    for key in (
        "source_manifest_sha256",
        "shared_split_sha256",
        "label_sha256",
        "surrogate_checkpoint_sha256",
        "c2lm_basis_sha256",
        "semantic_bank_sha256",
    ):
        if len(str(config["spec"].get(key, ""))) != 64:
            raise ValueError(f"SIRC spec.{key} must be a SHA256.")
    for key in ("semantic_bank_hash_mode", "c2lm_basis_hash_mode"):
        mode = str(config["spec"].get(key, "tensor-v1"))
        if mode not in {"tensor-v1", "recipe-v1"}:
            raise ValueError(f"SIRC spec.{key} has an unsupported value.")


def _gradient_map(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("image must have shape [H,W].")
    value = image.unsqueeze(0).unsqueeze(0)
    kx = image.new_tensor(
        [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]
    )
    ky = image.new_tensor(
        [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]
    )
    gx = F.conv2d(value, kx, padding=1)
    gy = F.conv2d(value, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)[0, 0]


def normalized_cross_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("NCC inputs must have identical shape.")
    left_flat = left.double().flatten() - left.double().mean()
    right_flat = right.double().flatten() - right.double().mean()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left_flat, right_flat) / denominator)


def semantic_structure_audit(
    bank: SemanticCarrierBank,
    anchor: torch.Tensor,
) -> dict[str, Any]:
    anchor_band = construct_phase_amplitude_variants(
        anchor,
        (anchor,),
        resolution=bank.resolution,
        radial_edges=bank.radial_edges,
    )[0]
    anchor_gradient = _gradient_map(anchor_band)
    semantic_gradients = [_gradient_map(value) for value in bank.variants]
    control_gradients = [_gradient_map(value) for value in bank.controls]
    semantic_pair_ncc = [
        normalized_cross_correlation(semantic_gradients[left], semantic_gradients[right])
        for left in range(len(semantic_gradients))
        for right in range(left + 1, len(semantic_gradients))
    ]
    semantic_anchor_ncc = [
        normalized_cross_correlation(value, anchor_gradient)
        for value in semantic_gradients
    ]
    control_anchor_ncc = [
        normalized_cross_correlation(value, anchor_gradient)
        for value in control_gradients
    ]
    normalized_amplitudes = []
    for value in bank.variants:
        amplitude = torch.fft.fft2(value.double()).abs().flatten()
        normalized_amplitudes.append(amplitude / amplitude.norm().clamp_min(1e-12))
    amplitude_distances = [
        float((normalized_amplitudes[left] - normalized_amplitudes[right]).norm())
        for left in range(len(normalized_amplitudes))
        for right in range(left + 1, len(normalized_amplitudes))
    ]
    return {
        "semantic_pair_gradient_ncc": semantic_pair_ncc,
        "semantic_pair_gradient_ncc_median": float(np.median(semantic_pair_ncc)),
        "semantic_anchor_gradient_ncc": semantic_anchor_ncc,
        "semantic_anchor_gradient_ncc_median": float(
            np.median(semantic_anchor_ncc)
        ),
        "control_anchor_gradient_ncc": control_anchor_ncc,
        "control_anchor_gradient_ncc_median": float(
            np.median(control_anchor_ncc)
        ),
        "pairwise_normalized_amplitude_distance": amplitude_distances,
        "pairwise_normalized_amplitude_distance_median": float(
            np.median(amplitude_distances)
        ),
    }


@dataclass(frozen=True)
class EOTParameters:
    scale: float
    translate_x: float
    translate_y: float
    blur_sigma: float
    grayscale: bool


def deterministic_eot_parameters(
    image_id: str,
    *,
    step: int,
    sample_index: int,
    seed: int,
) -> EOTParameters:
    payload = f"{seed}:{step}:{sample_index}:{image_id}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = random.Random(local_seed)
    return EOTParameters(
        scale=rng.uniform(0.90, 1.10),
        translate_x=rng.uniform(-0.05, 0.05),
        translate_y=rng.uniform(-0.05, 0.05),
        blur_sigma=rng.uniform(0.4, 0.8),
        grayscale=rng.random() < 0.25,
    )


def _gaussian_kernel_3x3(reference: torch.Tensor, sigma: float) -> torch.Tensor:
    coordinates = reference.new_tensor((-1.0, 0.0, 1.0))
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * float(sigma) ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(3, 1, 3, 3)


def _warp_crop(crop: torch.Tensor, parameters: EOTParameters) -> torch.Tensor:
    theta = crop.new_tensor(
        [
            [
                1.0 / parameters.scale,
                0.0,
                -2.0 * parameters.translate_x,
            ],
            [
                0.0,
                1.0 / parameters.scale,
                -2.0 * parameters.translate_y,
            ],
        ]
    ).unsqueeze(0)
    grid = F.affine_grid(
        theta,
        size=(1, 3, *crop.shape[-2:]),
        align_corners=False,
    )
    return F.grid_sample(
        crop.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]


def apply_object_relative_eot(
    image: torch.Tensor,
    boxes: Sequence[tuple[int, int, int, int]],
    parameters: EOTParameters,
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W].")
    height, width = image.shape[-2:]
    weighted = torch.zeros_like(image)
    counts = image.new_zeros((1, height, width))
    for box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("boxes must be clipped inside image.")
        warped = _warp_crop(image[:, y1:y2, x1:x2], parameters)
        weighted[:, y1:y2, x1:x2] += warped
        counts[:, y1:y2, x1:x2] += 1
    transformed = torch.where(counts > 0, weighted / counts.clamp_min(1), image)
    if parameters.grayscale:
        gray = (
            0.299 * transformed[0:1]
            + 0.587 * transformed[1:2]
            + 0.114 * transformed[2:3]
        )
        transformed = gray.expand_as(transformed)
    if parameters.blur_sigma > 0:
        kernel = _gaussian_kernel_3x3(transformed, parameters.blur_sigma)
        transformed = F.conv2d(
            transformed.unsqueeze(0),
            kernel,
            padding=1,
            groups=3,
        )[0]
    return transformed


def paired_object_relative_eot(
    clean: torch.Tensor,
    poisoned: torch.Tensor,
    *,
    boxes_by_image: Sequence[Sequence[tuple[int, int, int, int]]],
    image_ids: Sequence[str],
    step: int,
    sample_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != poisoned.shape or clean.ndim != 4:
        raise ValueError("clean and poisoned must share shape [B,3,H,W].")
    if len(boxes_by_image) != clean.shape[0] or len(image_ids) != clean.shape[0]:
        raise ValueError("Per-image EOT inputs must match batch size.")
    clean_views = []
    poisoned_views = []
    for index, image_id in enumerate(image_ids):
        parameters = deterministic_eot_parameters(
            image_id,
            step=step,
            sample_index=sample_index,
            seed=seed,
        )
        clean_views.append(
            apply_object_relative_eot(clean[index], boxes_by_image[index], parameters)
        )
        poisoned_views.append(
            apply_object_relative_eot(
                poisoned[index],
                boxes_by_image[index],
                parameters,
            )
        )
    return torch.stack(clean_views), torch.stack(poisoned_views)


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("Gradient tensors must have identical shape.")
    left_flat = left.detach().double().flatten()
    right_flat = right.detach().double().flatten()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(left_flat, right_flat) / denominator)


def assert_heldout_bank_immutable(
    bank: CICRPrototypeBank,
    before: Mapping[str, Any],
) -> None:
    def equal(left: Any, right: Any) -> bool:
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, (list, tuple)):
            return isinstance(right, (list, tuple)) and len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, Mapping):
            return isinstance(right, Mapping) and left.keys() == right.keys() and all(
                equal(left[key], right[key]) for key in left
            )
        return left == right

    after = bank.state_dict()
    if not equal(before, after):
        raise RuntimeError("Held-out prototype state changed.")


def evaluate_phase_a(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    structure: Mapping[str, Any],
    semantic_proxy_delta: float,
    semantic_vs_control: Mapping[str, Any],
    fixed_semantic_vs_control: Mapping[str, Any],
    cue_contrast: Mapping[str, Any],
    mechanical_pass: bool,
) -> dict[str, Any]:
    required = set(PHASE_A_ARMS)
    if set(arms) != required:
        raise ValueError(f"Phase A arms must be exactly {sorted(required)}.")
    candidate = arms["I-SV"]
    control = arms["I-SPC-V"]
    fixed_candidate = arms["I-SF"]
    group = candidate["group_cicr_median"]
    identity = float(candidate["heldout_cicr_median"])
    retentions = candidate["robustness_retention"]
    semantic_anchor_ncc = float(
        structure["semantic_anchor_gradient_ncc_median"]
    )
    control_anchor_ncc = float(
        structure["control_anchor_gradient_ncc_median"]
    )
    structure_margin = semantic_anchor_ncc - control_anchor_ncc
    checks = {
        "mechanical": bool(mechanical_pass),
        "structure": float(structure["semantic_pair_gradient_ncc_median"]) >= 0.70
        and semantic_anchor_ncc >= 0.60
        and control_anchor_ncc <= 0.20
        and structure_margin >= 0.30,
        "texture_diversity": float(
            structure["pairwise_normalized_amplitude_distance_median"]
        )
        >= 0.10,
        "candidate_level": identity >= 0.60
        and float(candidate["heldout_cicr_q25"]) >= 0.20
        and float(candidate["valid_instance_coverage"]) >= 0.75,
        "semantic_contrast": float(semantic_vs_control["paired_median_delta"])
        >= 0.05
        and float(semantic_vs_control["ci95"][0]) > 0,
        "fixed_contrast": float(fixed_semantic_vs_control["paired_median_delta"])
        >= 0.05,
        "variant_balance": identity
        >= float(fixed_candidate["heldout_cicr_median"]) - 0.05,
        "robustness": all(float(value) >= 0.75 for value in retentions.values()),
        "route_effect": float(candidate["route_effect"]) >= 0.10,
        "cue": float(candidate["cue_gain_median"]) >= 0.10
        and float(cue_contrast["paired_median_delta"]) >= 0.05
        and float(cue_contrast["ci95"][0]) > 0,
        "preservation": float(candidate["non_target_target_energy_ratio"])
        <= 1.10 * float(control["non_target_target_energy_ratio"])
        and float(candidate["box_residual_energy"])
        <= 1.10 * float(control["box_residual_energy"]),
        "groups": all(float(group[key]) >= 0.50 for key in ("person_only", "person_cooccur"))
        and abs(float(group["person_only"]) - float(group["person_cooccur"]))
        <= 0.20,
        "capacity_frequency": float(candidate["coefficient_saturation_ratio"]) < 0.25
        and float(candidate["active_basis_fraction"]) >= 0.25
        and float(candidate["top1_basis_energy_share"]) < 0.80
        and float(candidate["high_frequency_energy_ratio"]) <= 0.30
        and float(candidate["zero_norm_ratio"]) < 0.10,
    }
    failure_signals = {
        "structure_destroyed": semantic_anchor_ncc < 0.30
        or structure_margin < 0.10,
        "exact_pixel_dependence": float(fixed_semantic_vs_control["paired_median_delta"])
        >= 0.05
        and identity < float(fixed_candidate["heldout_cicr_median"]) - 0.15,
        "calibration_overfit": float(candidate["calibration_cicr_gain"]) >= 0.10
        and float(candidate["heldout_cicr_gain"]) < 0.02,
        "cue_absent": float(candidate["cue_gain_median"]) <= 0
        or float(cue_contrast["paired_median_delta"]) <= 0,
        "collateral_leakage": float(candidate["non_target_target_energy_ratio"])
        > 1.25 * float(control["non_target_target_energy_ratio"])
        or float(candidate["box_residual_energy"])
        > 1.25 * float(control["box_residual_energy"]),
        "high_frequency": float(candidate["high_frequency_energy_ratio"]) > 0.40,
        "insufficient_residual": float(candidate["valid_instance_coverage"]) < 0.60
        or float(candidate["zero_norm_ratio"]) >= 0.25,
        "non_finite": not all(bool(arm["finite"]) for arm in arms.values()),
    }
    return {
        "pass": all(checks.values()) and not any(failure_signals.values()),
        "checks": checks,
        "failure_signals": failure_signals,
        "diagnostics": {
            "semantic_proxy_delta": float(semantic_proxy_delta),
            "structure_ncc_margin": structure_margin,
        },
    }


def evaluate_phase_b(
    e0: Mapping[str, Any],
    e1: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "transformed_gain": float(contrast["paired_median_delta"]) >= 0.05
        and float(contrast["ci95"][0]) > 0,
        "identity_retention": float(e1["heldout_cicr_median"])
        >= float(e0["heldout_cicr_median"]) - 0.03,
        "preservation": float(e1["non_target_target_energy_ratio"])
        <= 1.10 * float(e0["non_target_target_energy_ratio"])
        and float(e1["box_residual_energy"])
        <= 1.10 * float(e0["box_residual_energy"]),
        "finite_coverage": bool(e0["finite"])
        and bool(e1["finite"])
        and float(e1["valid_instance_coverage"]) >= 0.75,
    }
    failure_signals = {
        "identity_route_degradation": float(e1["route_effect"])
        < 0.80 * float(e0["route_effect"]),
        "collateral_leakage": float(e1["non_target_target_energy_ratio"])
        > 1.25 * float(e0["non_target_target_energy_ratio"])
        or float(e1["box_residual_energy"])
        > 1.25 * float(e0["box_residual_energy"]),
        "non_finite": not bool(e0["finite"]) or not bool(e1["finite"]),
    }
    return {
        "pass": all(checks.values()) and not any(failure_signals.values()),
        "checks": checks,
        "failure_signals": failure_signals,
    }


def _finite_median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _residual_energy_loss(observation: ICMOObservation) -> torch.Tensor:
    terms = []
    for residuals in (
        observation.non_target_residuals,
        observation.box_residuals,
    ):
        for vectors, valid in zip(residuals.vectors, residuals.gate_valid):
            if bool(valid.any()):
                terms.append(vectors[valid].square().mean())
    if terms:
        return torch.stack(terms).mean()
    return observation.route.loss * 0.0


def _source_contamination_screen(
    model: torch.nn.Module,
    source_ids: Sequence[str],
    source_images: Sequence[torch.Tensor],
    *,
    resolution: int,
    target_class_id: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    records = []
    with torch.no_grad():
        for source_id, image in zip(source_ids, source_images):
            resized = center_square_resize(image, resolution).unsqueeze(0).to(device)
            output = model(resized)
            prediction = output[0] if isinstance(output, (tuple, list)) else output
            if prediction.ndim != 3:
                raise RuntimeError("Source screen expected rank-3 detector output.")
            if prediction.shape[1] == 4 + num_classes:
                scores = prediction[:, 4:, :]
            elif prediction.shape[2] == 4 + num_classes:
                scores = prediction[:, :, 4:].transpose(1, 2)
            else:
                raise RuntimeError("Source screen detector output has unknown layout.")
            if float(scores.min()) < 0 or float(scores.max()) > 1:
                scores = scores.sigmoid()
            person_confidence = float(scores[:, target_class_id].amax())
            voc_confidence = float(scores.amax())
            records.append(
                {
                    "source_id": source_id,
                    "person_confidence": person_confidence,
                    "voc20_confidence": voc_confidence,
                    "pass": person_confidence < 0.05 and voc_confidence < 0.25,
                }
            )
    return {
        "records": records,
        "pass": all(bool(record["pass"]) for record in records),
        "person_threshold": 0.05,
        "voc20_threshold": 0.25,
    }


def _p5_global_embeddings(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    modules = dict(model.named_modules())
    detect = modules.get("model.22")
    if detect is None:
        raise RuntimeError("Frozen surrogate has no model.22 Detect module.")
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], (list, tuple)):
            raise RuntimeError("Detect pre-hook expected one P3/P4/P5 sequence.")
        pyramid = inputs[0]
        if len(pyramid) != 3 or not torch.is_tensor(pyramid[2]):
            raise RuntimeError("Detect pre-hook did not receive P3/P4/P5.")
        captured.append(pyramid[2])

    handle = detect.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("P5 capture count mismatch.")
    return captured[0].mean(dim=(-2, -1))


def _semantic_proxy_cosine(
    model: torch.nn.Module,
    patterns: torch.Tensor,
    anchor: torch.Tensor,
    *,
    epsilon: float,
    resolution: int,
    device: torch.device,
) -> float:
    visual = (0.5 + 0.5 * patterns / float(epsilon)).clamp(0, 1)
    anchor_image = center_square_resize(anchor, resolution).unsqueeze(0)
    anchor_embedding = _p5_global_embeddings(model, anchor_image.to(device))[0]
    pattern_embeddings = _p5_global_embeddings(model, visual.to(device))
    cosines = F.cosine_similarity(
        pattern_embeddings,
        anchor_embedding.unsqueeze(0),
        dim=1,
    )
    return float(cosines.median())


def _pattern_shape_ncc(
    patterns: torch.Tensor,
    anchor: torch.Tensor,
    *,
    resolution: int,
) -> list[float]:
    anchor_band = construct_phase_amplitude_variants(
        anchor,
        (anchor,),
        resolution=resolution,
    )[0]
    anchor_gradient = _gradient_map(anchor_band.cpu())
    luma = (
        0.299 * patterns[:, 0]
        + 0.587 * patterns[:, 1]
        + 0.114 * patterns[:, 2]
    ).detach().cpu()
    return [
        normalized_cross_correlation(_gradient_map(value), anchor_gradient)
        for value in luma
    ]


class _FixedPatternCarrier(torch.nn.Module):
    def __init__(self, patterns: torch.Tensor, epsilon: float) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("patterns", patterns.detach().clone())

    def forward(self) -> torch.Tensor:
        return self.patterns


def _jpeg50_batch(images: torch.Tensor) -> torch.Tensor:
    results = []
    for image in images.detach().cpu():
        rgb = (
            image.clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 50],
        )
        if not ok:
            raise RuntimeError("JPEG50 encoding failed.")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG50 decoding failed.")
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        results.append(
            torch.from_numpy(decoded).permute(2, 0, 1).float() / 255.0
        )
    return torch.stack(results).to(images.device)


class SIRCProbeWorkflow:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        source_manifest: str | None = None,
        source_local_map: str | None = None,
        device_override: str | None = None,
    ) -> None:
        validate_sirc_config(config)
        self.config = dict(config)
        self.config_path = config_path.resolve()
        self.project_root = self.config_path.parents[2]
        self.seed = int(config["spec"]["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        runtime_device = str(
            device_override
            if device_override is not None
            else config["runtime"]["device"]
        )
        if runtime_device.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA SIRC probe but CUDA is unavailable.")
            self.device = torch.device(
                runtime_device
                if runtime_device.startswith("cuda")
                else f"cuda:{runtime_device}"
            )

        self.artifact_root = _resolve_path(
            self.project_root,
            str(config["runtime"]["artifact_root"]),
        )
        if self.artifact_root.exists():
            raise FileExistsError(
                f"Fresh SIRC probe refuses existing artifact root: {self.artifact_root}"
            )
        dataset_root = _resolve_path(
            self.project_root,
            str(config["dataset"]["root"]),
        )
        self.train_image_dir = _resolve_path(
            dataset_root,
            str(config["dataset"]["train_images"]),
        )
        self.train_label_dir = _resolve_path(
            dataset_root,
            str(config["dataset"]["train_labels"]),
        )
        if not self.train_image_dir.is_dir() or not self.train_label_dir.is_dir():
            raise FileNotFoundError("VOC train image/label directories are missing.")
        target_class_id = int(config["dataset"]["target_class_id"])
        all_train_images = [Path(path) for path in list_images(str(self.train_image_dir))]
        self.target_images = [
            path
            for path in all_train_images
            if image_has_target(
                read_yolo_annotations(
                    label_path_for_image(str(path), str(self.train_label_dir))
                ),
                target_class_id,
            )
        ]
        self.split = load_required_shared_split(
            _resolve_path(self.project_root, str(config["split"]["manifest"])),
            target_images=self.target_images,
            required_protocol_prefix=str(
                config["split"]["required_protocol_prefix"]
            ),
        )
        if self.split["split_hash"] != str(config["spec"]["shared_split_sha256"]):
            raise ValueError("Frozen shared split hash mismatch.")
        if str(self.split["shared_split_manifest"].get("label_hash")) != str(
            config["spec"]["label_sha256"]
        ):
            raise ValueError("Frozen shared split label hash mismatch.")

        manifest_value = source_manifest or str(config["background"]["source_manifest"])
        local_map_value = source_local_map or str(config["background"]["source_local_map"])
        local_map_path = _resolve_path(self.project_root, local_map_value)
        source_images, source_manifest_records, source_manifest_hash = (
            load_background_sources(
                _resolve_path(self.project_root, manifest_value),
                local_map_path,
            )
        )
        if source_manifest_hash != str(config["spec"]["source_manifest_sha256"]):
            raise ValueError("Frozen source manifest hash mismatch.")
        self.source_manifest = source_manifest_records
        self.source_manifest_hash = source_manifest_hash
        self.source_images = source_images
        source_by_id = {
            str(record["source_id"]): image
            for record, image in zip(source_manifest_records, source_images)
        }
        required_ids = (
            "bg-tree-08",
            "bg-waves-01",
            "bg-bubbles-02",
            "bg-beach-03",
            "bg-field-04",
        )
        if any(source_id not in source_by_id for source_id in required_ids):
            raise ValueError("SIRC anchor/donor source IDs are incomplete.")
        required_hashes = {
            str(record["sha256"])
            for record in source_manifest_records
            if str(record["source_id"]) in required_ids
        }
        dataset_images = list(all_train_images)
        val_relative = str(config["dataset"].get("val_images", "")).strip()
        if val_relative:
            val_dir = _resolve_path(dataset_root, val_relative)
            if not val_dir.is_dir():
                raise FileNotFoundError("Configured VOC validation image directory is missing.")
            dataset_images.extend(Path(path) for path in list_images(str(val_dir)))
        local_source_map = json.loads(local_map_path.read_text(encoding="utf-8"))
        required_sizes = {
            Path(local_source_map[source_id]).stat().st_size
            for source_id in required_ids
        }
        self.source_duplicate_matches = [
            str(path)
            for path in dataset_images
            if path.stat().st_size in required_sizes
            and _file_sha256(path) in required_hashes
        ]
        self.anchor = source_by_id[required_ids[0]]
        records_by_id = {
            str(record["source_id"]): record
            for record in source_manifest_records
        }
        semantic_source_provenance = {
            "manifest_sha256": source_manifest_hash,
            "ordered_sources": [
                {
                    "source_id": source_id,
                    "sha256": str(records_by_id[source_id]["sha256"]).lower(),
                }
                for source_id in required_ids
            ],
        }
        semantic_hash_mode = str(
            config["spec"].get("semantic_bank_hash_mode", "tensor-v1")
        )
        self.semantic_bank = build_semantic_carrier_bank(
            self.anchor,
            [source_by_id[source_id] for source_id in required_ids[1:]],
            resolution=int(config["carrier"]["resolution"]),
            phase_seed=int(config["carrier"]["phase_seed"]),
            radial_edges=config["carrier"]["radial_edges"],
            hash_mode=semantic_hash_mode,
            source_provenance=(
                semantic_source_provenance
                if semantic_hash_mode == "recipe-v1"
                else None
            ),
        )
        expected_bank_hash = str(config["spec"].get("semantic_bank_sha256", ""))
        if expected_bank_hash and self.semantic_bank.bank_hash != expected_bank_hash:
            raise ValueError("Frozen semantic carrier bank hash mismatch.")

        baseline_hash_mode = str(
            config["spec"].get("c2lm_basis_hash_mode", "tensor-v1")
        )
        baseline_source_provenance = {
            "manifest_sha256": source_manifest_hash,
            "ordered_sources": [
                {
                    "source_id": str(record["source_id"]),
                    "sha256": str(record["sha256"]).lower(),
                }
                for record in source_manifest_records
            ],
        }
        baseline = build_background_spectral_basis(
            source_images,
            resolution=int(config["carrier"]["resolution"]),
            num_bases=int(config["carrier"]["num_bases"]),
            bands=((2.0, 8.0), (8.0, 24.0)),
            phase_mode="scrambled",
            seed=int(config["carrier"].get("baseline_basis_seed", 0)),
            hash_mode=baseline_hash_mode,
            source_provenance=(
                baseline_source_provenance
                if baseline_hash_mode == "recipe-v1"
                else None
            ),
        )
        expected_baseline_hash = str(config["spec"].get("c2lm_basis_sha256", ""))
        if expected_baseline_hash and baseline.basis_hash != expected_baseline_hash:
            raise ValueError("Frozen C2-LM baseline hash mismatch.")
        self.c2lm_basis_hash = baseline.basis_hash
        self.c2lm_basis_hash_mode = baseline.hash_mode
        baseline_scales = torch.full(
            (baseline.bases.shape[0],),
            1.0 / math.sqrt(baseline.bases.shape[0]),
        )
        self.arm_bases = {
            "I-C2LM": (baseline.bases, baseline_scales),
            "I-SPC-F": (
                self.semantic_bank.control_bases[0],
                self.semantic_bank.control_scales[0],
            ),
            "I-SF": (
                self.semantic_bank.semantic_bases[0],
                self.semantic_bank.semantic_scales[0],
            ),
            "I-SPC-V": (
                self.semantic_bank.control_bases,
                self.semantic_bank.control_scales,
            ),
            "I-SV": (
                self.semantic_bank.semantic_bases,
                self.semantic_bank.semantic_scales,
            ),
        }
        carrier_cfg = config["carrier"]
        self.gamma_calibration = calibrate_variant_shared_gamma(
            {
                "baseline": self.arm_bases["I-C2LM"],
                "semantic": self.arm_bases["I-SV"],
                "control": self.arm_bases["I-SPC-V"],
            },
            epsilon=float(carrier_cfg["epsilon"]),
            seed=int(carrier_cfg["gamma_seed"]),
            num_directions=int(carrier_cfg["gamma_directions"]),
            coefficient_max_abs=float(carrier_cfg["coefficient_max_abs"]),
            target_rms_ratio=float(carrier_cfg["target_rms_ratio"]),
            iterations=int(carrier_cfg["gamma_bisection_iterations"]),
            chunk_size=int(carrier_cfg["gamma_chunk_size"]),
            device=self.device,
        )
        self.initial_coefficients = common_initial_coefficients(
            int(carrier_cfg["num_bases"]),
            seed=int(carrier_cfg["initial_seed"]),
            max_abs=float(carrier_cfg["coefficient_max_abs"]),
        )

        checkpoint_path = _resolve_path(
            self.project_root,
            str(config["model"]["surrogate_checkpoint"]),
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Surrogate checkpoint is missing: {checkpoint_path}")
        checkpoint_hash = _file_sha256(checkpoint_path)
        if checkpoint_hash != str(config["spec"]["surrogate_checkpoint_sha256"]):
            raise ValueError("Frozen surrogate checkpoint hash mismatch.")
        self.surrogate_checkpoint_hash = checkpoint_hash
        self.engine = ICMOProbeEngine(
            config,
            self.device,
            checkpoint_path=checkpoint_path,
        )
        try:
            self.source_screen = _source_contamination_screen(
                self.engine.model,
                required_ids,
                [source_by_id[source_id] for source_id in required_ids],
                resolution=int(config["model"]["image_size"]),
                target_class_id=target_class_id,
                num_classes=int(config["model"]["num_classes"]),
                device=self.device,
            )
            self.structure_audit = semantic_structure_audit(
                self.semantic_bank,
                self.anchor,
            )
            self.artifact_root.mkdir(parents=True)
            (self.artifact_root / "arms").mkdir()
            self._write_initial_artifacts()
        except Exception:
            self.engine.close()
            raise

    def close(self) -> None:
        self.engine.close()

    def _write_initial_artifacts(self) -> None:
        _write_probe_json(self.artifact_root / "config_resolved.json", self.config)
        _write_probe_json(
            self.artifact_root / "source_manifest.json",
            self.source_manifest,
        )
        _write_probe_json(
            self.artifact_root / "input_audit.json",
            {
                "source_contamination": self.source_screen,
                "voc_source_hash_duplicates": self.source_duplicate_matches,
                "structure": self.structure_audit,
                "semantic_bank_hash": self.semantic_bank.bank_hash,
                "semantic_bank_hash_mode": self.semantic_bank.hash_mode,
                "c2lm_basis_hash": self.c2lm_basis_hash,
                "c2lm_basis_hash_mode": self.c2lm_basis_hash_mode,
                "source_manifest_hash": self.source_manifest_hash,
                "split_hash": self.split["split_hash"],
                "surrogate_checkpoint_sha256": self.surrogate_checkpoint_hash,
                "gamma": self.gamma_calibration.gamma,
                "gamma_direction_hash": self.gamma_calibration.direction_hash,
                "claim_boundary": (
                    "surrogate-only semantic residual mechanism probe; forced "
                    "pseudo fallback support; not victim UE evidence"
                ),
            },
        )

    def _carrier(self, arm_id: str) -> VariantMatchedCanonicalCarrier:
        carrier_cfg = self.config["carrier"]
        bases, scales = self.arm_bases[arm_id]
        return VariantMatchedCanonicalCarrier(
            bases,
            scales,
            epsilon=float(carrier_cfg["epsilon"]),
            gamma=self.gamma_calibration.gamma,
            initial_coefficients=self.initial_coefficients,
        ).to(self.device)

    def _load_batch(self, paths: Sequence[Path | str]) -> ICMOBatch:
        return load_icmo_batch(
            [Path(path) for path in paths],
            label_dir=self.train_label_dir,
            image_size=int(self.config["model"]["image_size"]),
            target_class_id=int(self.config["dataset"]["target_class_id"]),
            device=self.device,
        )

    def _variant_indices(self, arm_id: str, batch: ICMOBatch) -> tuple[int, ...]:
        _, mode = PHASE_A_ARMS[arm_id]
        if mode == "fixed":
            return tuple(0 for _ in batch.image_ids)
        seed = int(self.config["carrier"]["variant_seed"])
        return tuple(
            stable_variant_index(image_id, seed=seed)
            for image_id in batch.image_ids
        )

    def _observe(
        self,
        batch: ICMOBatch,
        carrier: VariantMatchedCanonicalCarrier,
        *,
        arm_id: str,
        paired_view_transform=None,
        assignment_reference_images: torch.Tensor | None = None,
    ) -> ICMOObservation:
        return self.engine.observe(
            batch,
            carrier,
            render_mode="instance",
            variant_indices=self._variant_indices(arm_id, batch),
            paired_view_transform=paired_view_transform,
            assignment_reference_images=assignment_reference_images,
        )

    def _collect(
        self,
        paths: Sequence[str],
        carrier: VariantMatchedCanonicalCarrier,
        *,
        arm_id: str,
        gradients: bool,
        limit_batches: int | None = None,
        transform_factory=None,
    ) -> list[ICMOObservation]:
        observations = []
        batches = make_batches(
            paths,
            batch_size=int(self.config["optimization"]["batch_size"]),
        )
        if limit_batches is not None:
            batches = batches[:limit_batches]
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context:
            for batch_index, path_batch in enumerate(batches):
                batch = self._load_batch(path_batch)
                transform = (
                    transform_factory(batch_index)
                    if transform_factory is not None
                    else None
                )
                observations.append(
                    self._observe(
                        batch,
                        carrier,
                        arm_id=arm_id,
                        paired_view_transform=transform,
                    )
                )
        return observations

    def _eot_transform(self, *, step: int, sample_index: int):
        seed = int(self.config["eot"]["seed"])

        def transform(
            clean: torch.Tensor,
            poisoned: torch.Tensor,
            batch: ICMOBatch,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return paired_object_relative_eot(
                clean,
                poisoned,
                boxes_by_image=batch.boxes_by_image,
                image_ids=batch.image_ids,
                step=step,
                sample_index=sample_index,
                seed=seed,
            )

        return transform

    @staticmethod
    def _fixed_transform(parameters: EOTParameters):
        def transform(
            clean: torch.Tensor,
            poisoned: torch.Tensor,
            batch: ICMOBatch,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            clean_views = [
                apply_object_relative_eot(
                    clean[index],
                    batch.boxes_by_image[index],
                    parameters,
                )
                for index in range(clean.shape[0])
            ]
            poisoned_views = [
                apply_object_relative_eot(
                    poisoned[index],
                    batch.boxes_by_image[index],
                    parameters,
                )
                for index in range(poisoned.shape[0])
            ]
            return torch.stack(clean_views), torch.stack(poisoned_views)

        return transform

    @staticmethod
    def _jpeg_transform(
        clean: torch.Tensor,
        poisoned: torch.Tensor,
        _batch: ICMOBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _jpeg50_batch(clean), _jpeg50_batch(poisoned)

    def _robustness_audit(
        self,
        carrier: VariantMatchedCanonicalCarrier,
        bank: CICRPrototypeBank,
        *,
        family_arm: str,
        identity_cicr: float,
        heldout_limit: int | None,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        audit_transforms = {
            "affine": self._fixed_transform(
                EOTParameters(0.90, 0.05, -0.05, 0.0, False)
            ),
            "blur": self._fixed_transform(
                EOTParameters(1.0, 0.0, 0.0, 0.6, False)
            ),
            "grayscale": self._fixed_transform(
                EOTParameters(1.0, 0.0, 0.0, 0.0, True)
            ),
            "jpeg50": self._jpeg_transform,
        }
        summaries = {}
        retentions = {}
        for audit_id, transform in audit_transforms.items():
            observations = self._collect(
                self.split["heldout"],
                carrier,
                arm_id=family_arm,
                gradients=False,
                limit_batches=heldout_limit,
                transform_factory=lambda _batch_index, transform=transform: transform,
            )
            summary = summarize_icmo_observations(observations, bank)
            summaries[audit_id] = summary
            retentions[audit_id] = float(summary["heldout_cicr_median"]) / max(
                float(identity_cicr),
                1e-12,
            )
        return retentions, summaries

    @staticmethod
    def _weaken_person_appearance(batch: ICMOBatch) -> ICMOBatch:
        weakened = []
        coordinates = batch.images.new_tensor((-2.0, -1.0, 0.0, 1.0, 2.0))
        kernel_1d = torch.exp(-coordinates.square() / 2.0)
        kernel_1d /= kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 5, 5)
        for image, supports in zip(batch.images, batch.supports_by_image):
            union = (supports.sum(dim=0) > 0).to(image.dtype)
            gray = (
                0.299 * image[0:1]
                + 0.587 * image[1:2]
                + 0.114 * image[2:3]
            ).expand_as(image)
            blurred = F.conv2d(
                gray.unsqueeze(0),
                kernel,
                padding=2,
                groups=3,
            )[0]
            weakened.append(image * (1.0 - union) + blurred * union)
        return replace(batch, images=torch.stack(weakened))

    def _cue_audit(
        self,
        carrier: VariantMatchedCanonicalCarrier,
        *,
        family_arm: str,
        smoke: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        paths = self.split["heldout"][: (2 if smoke else len(self.split["heldout"]))]
        zero_patterns = torch.zeros_like(carrier())
        zero_carrier = _FixedPatternCarrier(
            zero_patterns,
            epsilon=float(self.config["carrier"]["epsilon"]),
        ).to(self.device)
        records = []
        with torch.no_grad():
            for path in paths:
                original = self._load_batch((path,))
                weakened = self._weaken_person_appearance(original)
                clean_observation = self.engine.observe(
                    weakened,
                    zero_carrier,
                    render_mode="instance",
                    variant_indices=self._variant_indices(family_arm, weakened),
                    assignment_reference_images=original.images,
                )
                carrier_observation = self._observe(
                    weakened,
                    carrier,
                    arm_id=family_arm,
                    assignment_reference_images=original.images,
                )
                clean_loss = float(clean_observation.route.loss.detach())
                carrier_loss = float(carrier_observation.route.loss.detach())
                records.append(
                    {
                        "image_id": original.image_ids[0],
                        "person_cooccur": original.person_cooccur[0],
                        "clean_route_loss": clean_loss,
                        "carrier_route_loss": carrier_loss,
                        "cue_gain": (clean_loss - carrier_loss)
                        / max(abs(clean_loss), 1e-8),
                    }
                )
        return _finite_median([record["cue_gain"] for record in records]), records

    def _run_arm(
        self,
        arm_id: str,
        *,
        family_arm: str | None = None,
        trc_weight: float | None = None,
        smoke: bool = False,
    ) -> dict[str, Any]:
        family_arm = family_arm or arm_id
        carrier = self._carrier(family_arm)
        optimizer = torch.optim.Adam(
            [carrier.coefficients],
            lr=float(self.config["optimization"]["learning_rate"]),
        )
        batch_size = int(self.config["optimization"]["batch_size"])
        calibration_batches = make_batches(
            self.split["calibration"],
            batch_size=batch_size,
        )
        if smoke:
            calibration_batches = calibration_batches[:1]
        heldout_limit = 1 if smoke else None
        initial_heldout_before_warmup = self._collect(
            self.split["heldout"],
            carrier,
            arm_id=family_arm,
            gradients=False,
            limit_batches=heldout_limit,
        )
        initial_route_loss = _finite_median(
            [
                float(observation.route.loss.detach())
                for observation in initial_heldout_before_warmup
            ]
        )
        diagnostics = []
        gradient_norms = []
        warmup_steps = 1 if smoke else int(self.config["optimization"]["warmup_steps"])
        for step in range(warmup_steps):
            batch = self._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self._observe(
                batch,
                carrier,
                arm_id=family_arm,
            )
            optimizer.zero_grad(set_to_none=True)
            observation.route.loss.backward()
            gradient = carrier.coefficients.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm_id} route warmup gradient is invalid.")
            gradient_norm = float(gradient.norm())
            gradient_norms.append(gradient_norm)
            optimizer.step()
            diagnostics.append(
                {
                    "stage": "route_warmup",
                    "step": step,
                    "route_loss": float(observation.route.loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )

        initial_calibration = self._collect(
            self.split["calibration"],
            carrier,
            arm_id=family_arm,
            gradients=False,
            limit_batches=1 if smoke else None,
        )
        bank = fit_icmo_bank(
            initial_calibration,
            momentum=float(self.config["optimization"]["prototype_momentum"]),
        )
        bank_before_heldout = {
            key: (
                [value.clone() if torch.is_tensor(value) else value for value in item]
                if isinstance(item, list)
                else item.clone()
                if torch.is_tensor(item)
                else item
            )
            for key, item in bank.state_dict().items()
        }
        initial_heldout = self._collect(
            self.split["heldout"],
            carrier,
            arm_id=family_arm,
            gradients=False,
            limit_batches=heldout_limit,
        )
        assert_heldout_bank_immutable(bank, bank_before_heldout)
        initial_calibration_summary = summarize_icmo_observations(
            initial_calibration,
            bank,
        )
        initial_heldout_summary = summarize_icmo_observations(
            initial_heldout,
            bank,
        )

        optimization_steps = 1 if smoke else int(
            self.config["optimization"]["optimization_steps"]
        )
        target_rms = float(self.config["carrier"]["epsilon"]) * float(
            self.config["carrier"]["target_rms_ratio"]
        )
        for step in range(optimization_steps):
            batch = self._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self._observe(
                batch,
                carrier,
                arm_id=family_arm,
            )
            cicr_result = instance_cicr(observation.target_residuals, bank)
            pattern = carrier()
            rms_loss = (
                pattern.square().mean().sqrt() / target_rms - 1.0
            ).square()
            trc_losses = []
            if trc_weight is not None:
                for sample_index in range(int(self.config["eot"]["samples"])):
                    transformed = self._observe(
                        batch,
                        carrier,
                        arm_id=family_arm,
                        paired_view_transform=self._eot_transform(
                            step=step,
                            sample_index=sample_index,
                        ),
                    )
                    trc_losses.append(
                        instance_cicr(transformed.target_residuals, bank).loss
                    )
            trc_loss = (
                torch.stack(trc_losses).mean()
                if trc_losses
                else observation.route.loss * 0.0
            )
            total_loss = (
                float(self.config["optimization"]["lambda_cicr"])
                * cicr_result.loss
                + float(self.config["optimization"]["lambda_route"])
                * observation.route.loss
                + float(self.config["optimization"]["lambda_rms"])
                * rms_loss
                + float(trc_weight or 0.0) * trc_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient = carrier.coefficients.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm_id} matched gradient is invalid.")
            gradient_norm = float(gradient.norm())
            gradient_norms.append(gradient_norm)
            optimizer.step()
            diagnostics.append(
                {
                    "stage": "matched",
                    "step": step,
                    "total_loss": float(total_loss.detach()),
                    "cicr_loss": float(cicr_result.loss.detach()),
                    "route_loss": float(observation.route.loss.detach()),
                    "rms_loss": float(rms_loss.detach()),
                    "trc_loss": float(trc_loss.detach()),
                    "trc_weight": float(trc_weight or 0.0),
                    "gradient_norm": gradient_norm,
                    "valid_instance_coverage": cicr_result.valid_instance_coverage,
                }
            )

        final_calibration = self._collect(
            self.split["calibration"],
            carrier,
            arm_id=family_arm,
            gradients=False,
            limit_batches=1 if smoke else None,
        )
        final_heldout = self._collect(
            self.split["heldout"],
            carrier,
            arm_id=family_arm,
            gradients=False,
            limit_batches=heldout_limit,
        )
        assert_heldout_bank_immutable(bank, bank_before_heldout)
        calibration_summary = summarize_icmo_observations(final_calibration, bank)
        summary = summarize_icmo_observations(final_heldout, bank)
        final_pattern = carrier().detach()
        coefficient_activation = torch.tanh(carrier.coefficients.detach())
        basis_energy = coefficient_activation.square().sum(dim=1)
        spectrum = [
            spectrum_energy_ratios(pattern.cpu())
            for pattern in final_pattern
        ]
        shape_ncc = _pattern_shape_ncc(
            final_pattern,
            self.anchor,
            resolution=int(self.config["carrier"]["resolution"]),
        )
        pattern_luma = (
            0.299 * final_pattern[:, 0]
            + 0.587 * final_pattern[:, 1]
            + 0.114 * final_pattern[:, 2]
        ).cpu()
        pair_shape_ncc = [
            normalized_cross_correlation(
                _gradient_map(pattern_luma[left]),
                _gradient_map(pattern_luma[right]),
            )
            for left in range(pattern_luma.shape[0])
            for right in range(left + 1, pattern_luma.shape[0])
        ]
        semantic_proxy = _semantic_proxy_cosine(
            self.engine.model,
            final_pattern,
            self.anchor,
            epsilon=float(self.config["carrier"]["epsilon"]),
            resolution=int(self.config["carrier"]["resolution"]),
            device=self.device,
        )

        if smoke:
            robustness_retention: dict[str, float] = {}
            robustness_summaries: dict[str, dict[str, Any]] = {}
            cue_gain_median = float("nan")
            cue_records: list[dict[str, Any]] = []
        else:
            robustness_retention, robustness_summaries = self._robustness_audit(
                carrier,
                bank,
                family_arm=family_arm,
                identity_cicr=float(summary["heldout_cicr_median"]),
                heldout_limit=heldout_limit,
            )
            cue_gain_median, cue_records = self._cue_audit(
                carrier,
                family_arm=family_arm,
                smoke=False,
            )

        gradient_batch = self._load_batch(calibration_batches[0])
        gradient_observation = self._observe(
            gradient_batch,
            carrier,
            arm_id=family_arm,
        )
        target_gradient = torch.autograd.grad(
            instance_cicr(gradient_observation.target_residuals, bank).loss
            + gradient_observation.route.loss,
            carrier.coefficients,
            retain_graph=True,
        )[0]
        preserve_gradient = torch.autograd.grad(
            _residual_energy_loss(gradient_observation),
            carrier.coefficients,
        )[0]
        gradient_conflict_cosine = gradient_cosine(
            target_gradient,
            preserve_gradient,
        )

        transformed_summary = None
        if trc_weight is not None:
            transformed_observations = []
            for sample_index in range(int(self.config["eot"]["samples"])):
                transformed_observations.extend(
                    self._collect(
                        self.split["heldout"],
                        carrier,
                        arm_id=family_arm,
                        gradients=False,
                        limit_batches=heldout_limit,
                        transform_factory=lambda batch_index, sample_index=sample_index: self._eot_transform(
                            step=10_000 + batch_index,
                            sample_index=sample_index,
                        ),
                    )
                )
            transformed_summary = summarize_icmo_observations(
                transformed_observations,
                bank,
            )

        prototype_state = bank.state_dict()
        prototype_tensors = [
            value
            for value in prototype_state["prototypes"]
            if torch.is_tensor(value)
        ]
        prototype_hash = (
            tensor_sha256(
                torch.cat([value.detach().flatten() for value in prototype_tensors])
            )
            if prototype_tensors
            else ""
        )
        image_ids = sorted(
            {str(record["image_id"]) for record in summary["per_image"]}
        )
        variant_assignments = {
            image_id: (
                0
                if PHASE_A_ARMS[family_arm][1] == "fixed"
                else stable_variant_index(
                    image_id,
                    seed=int(self.config["carrier"]["variant_seed"]),
                )
            )
            for image_id in image_ids
        }

        summary.update(
            {
                "arm_id": arm_id,
                "family_arm": family_arm,
                "initial_route_loss": initial_route_loss,
                "route_effect": (
                    initial_route_loss - float(summary["route_loss"])
                )
                / max(abs(initial_route_loss), 1e-12),
                "initial_calibration_cicr": initial_calibration_summary[
                    "heldout_cicr_median"
                ],
                "initial_heldout_cicr": initial_heldout_summary[
                    "heldout_cicr_median"
                ],
                "calibration_cicr_gain": float(
                    calibration_summary["heldout_cicr_median"]
                    - initial_calibration_summary["heldout_cicr_median"]
                ),
                "heldout_cicr_gain": float(
                    summary["heldout_cicr_median"]
                    - initial_heldout_summary["heldout_cicr_median"]
                ),
                "canonical_rms": float(final_pattern.square().mean().sqrt()),
                "canonical_linf": float(final_pattern.abs().amax()),
                "coefficient_saturation_ratio": float(
                    (coefficient_activation.abs() >= 0.95).float().mean()
                ),
                "active_basis_fraction": float(
                    (coefficient_activation.norm(dim=1) > 1e-3).float().mean()
                ),
                "top1_basis_energy_share": float(
                    basis_energy.max() / basis_energy.sum().clamp_min(1e-12)
                ),
                "high_frequency_energy_ratio": float(
                    np.median([value["high"] for value in spectrum])
                ),
                "canonical_spectrum_energy": spectrum,
                "shape_ncc": shape_ncc,
                "shape_ncc_median": _finite_median(shape_ncc),
                "pair_shape_ncc": pair_shape_ncc,
                "pair_shape_ncc_median": _finite_median(pair_shape_ncc),
                "semantic_proxy_cosine": semantic_proxy,
                "robustness_retention": robustness_retention,
                "robustness_summaries": robustness_summaries,
                "cue_gain_median": cue_gain_median,
                "cue_per_image": cue_records,
                "target_preservation_gradient_cosine": gradient_conflict_cosine,
                "coefficient_hash": tensor_sha256(carrier.coefficients.detach()),
                "prototype_hash": prototype_hash,
                "variant_assignments": variant_assignments,
                "gradient_norms": gradient_norms,
                "diagnostics": diagnostics,
                "transformed": transformed_summary,
            }
        )
        finite_values = [
            summary["heldout_cicr_median"],
            summary["valid_instance_coverage"],
            summary["route_effect"],
            summary["canonical_rms"],
            summary["canonical_linf"],
            summary["shape_ncc_median"],
            *gradient_norms,
        ]
        summary["finite"] = all(
            math.isfinite(float(value)) for value in finite_values
        ) and bool(
            torch.isfinite(carrier.coefficients).all()
            and torch.isfinite(final_pattern).all()
            and all(torch.isfinite(value).all() for value in prototype_tensors)
        )
        torch.save(
            {
                "arm_id": arm_id,
                "family_arm": family_arm,
                "coefficients": carrier.coefficients.detach().cpu(),
                "prototype_bank": prototype_state,
                "gamma": self.gamma_calibration.gamma,
            },
            self.artifact_root / "arms" / f"{arm_id}_state.pt",
        )
        _write_probe_json(
            self.artifact_root / "arms" / f"{arm_id}_metrics.json",
            summary,
        )
        return summary

    def _mechanical_preconditions(self) -> dict[str, Any]:
        spectrum_errors = []
        for semantic, control in zip(
            self.semantic_bank.variants,
            self.semantic_bank.controls,
        ):
            semantic_amplitude = torch.fft.fft2(semantic.double()).abs()
            control_amplitude = torch.fft.fft2(control.double()).abs()
            spectrum_errors.append(
                float(
                    (semantic_amplitude - control_amplitude).abs().amax()
                    / semantic_amplitude.amax().clamp_min(1e-12)
                )
            )
        ranks = torch.cat(
            (
                self.semantic_bank.semantic_ranks,
                self.semantic_bank.control_ranks,
            )
        )
        checks = {
            "source_contamination": bool(self.source_screen["pass"]),
            "voc_hash_separation": not self.source_duplicate_matches,
            "spectrum_match": max(spectrum_errors, default=float("inf")) <= 1e-5,
            "basis_rank": int(ranks.min()) >= 8,
            "gamma_family_match": self.gamma_calibration.family_rms_ratio <= 1.05,
            "finite": bool(
                torch.isfinite(self.semantic_bank.semantic_bases).all()
                and torch.isfinite(self.semantic_bank.control_bases).all()
            ),
        }
        return {
            "pass": all(checks.values()),
            "checks": checks,
            "relative_spectrum_errors": spectrum_errors,
            "minimum_basis_rank": int(ranks.min()),
            "gamma_family_rms_ratio": self.gamma_calibration.family_rms_ratio,
        }

    def _phase_a_mechanical_pass(
        self,
        arms: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        active_rms = [float(arm["active_pixel_rms"]) for arm in arms.values()]
        active_rms_ratio = max(active_rms) / max(min(active_rms), 1e-12)
        checks = {
            "preconditions": bool(self._mechanical_preconditions()["pass"]),
            "active_rms_ratio": active_rms_ratio <= 1.05,
            "linf": all(
                float(arm["active_pixel_linf"])
                <= float(self.config["carrier"]["epsilon"]) + 1e-8
                for arm in arms.values()
            ),
            "outside_support": all(
                float(arm["outside_support_max"]) == 0.0
                for arm in arms.values()
            ),
            "finite": all(bool(arm["finite"]) for arm in arms.values()),
        }
        return all(checks.values()), {
            "checks": checks,
            "active_rms_ratio": active_rms_ratio,
        }

    @staticmethod
    def _cue_records_for_bootstrap(
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "image_id": record["image_id"],
                "person_cooccur": record["person_cooccur"],
                "cicr": record["cue_gain"],
            }
            for record in records
        ]

    def run(self, *, smoke: bool = False) -> dict[str, Any]:
        status = {
            "spec_id": self.config["spec"]["spec_id"],
            "exp_id": self.config["spec"]["exp_id"],
            "state": "running",
            "mode": "local_mechanical_smoke" if smoke else "formal_probe",
            "config_hash": canonical_hash(self.config),
            "semantic_bank_hash": self.semantic_bank.bank_hash,
            "claim_boundary": (
                "surrogate mechanism only; no poisoned dataset, victim training, or mAP"
            ),
        }
        _write_probe_json(self.artifact_root / "status.json", status)
        try:
            preconditions = self._mechanical_preconditions()
            _write_probe_json(
                self.artifact_root / "mechanical_preconditions.json",
                preconditions,
            )
            if not preconditions["pass"]:
                status.update(
                    {
                        "state": "stopped",
                        "stop_reason": "mechanical_precondition_failure",
                        "preconditions": preconditions,
                    }
                )
                _write_probe_json(self.artifact_root / "status.json", status)
                return status

            if smoke:
                smoke_arms = {
                    "I-SPC-V": self._run_arm("I-SPC-V", smoke=True),
                    "I-SV": self._run_arm("I-SV", smoke=True),
                    "I-SV-E1": self._run_arm(
                        "I-SV-E1",
                        family_arm="I-SV",
                        trc_weight=1.0,
                        smoke=True,
                    ),
                }
                status.update(
                    {
                        "state": "completed",
                        "mechanical_smoke": True,
                        "arms": {
                            arm_id: {
                                "finite": result["finite"],
                                "valid_instance_coverage": result[
                                    "valid_instance_coverage"
                                ],
                                "outside_support_max": result[
                                    "outside_support_max"
                                ],
                                "gradient_norms": result["gradient_norms"],
                            }
                            for arm_id, result in smoke_arms.items()
                        },
                        "mechanism_claim": "not_evaluated_by_smoke",
                    }
                )
                _write_probe_json(self.artifact_root / "smoke_metrics.json", smoke_arms)
                _write_probe_json(self.artifact_root / "status.json", status)
                return status

            arms = {
                arm_id: self._run_arm(arm_id)
                for arm_id in PHASE_A_ARMS
            }
            bootstrap_cfg = self.config["bootstrap"]
            semantic_contrast = stratified_paired_bootstrap(
                arms["I-SV"]["per_image"],
                arms["I-SPC-V"]["per_image"],
                seed=int(bootstrap_cfg["seed"]),
                iterations=int(bootstrap_cfg["iterations"]),
            )
            fixed_contrast = stratified_paired_bootstrap(
                arms["I-SF"]["per_image"],
                arms["I-SPC-F"]["per_image"],
                seed=int(bootstrap_cfg["seed"]),
                iterations=int(bootstrap_cfg["iterations"]),
            )
            cue_contrast = stratified_paired_bootstrap(
                self._cue_records_for_bootstrap(arms["I-SV"]["cue_per_image"]),
                self._cue_records_for_bootstrap(
                    arms["I-SPC-V"]["cue_per_image"]
                ),
                seed=int(bootstrap_cfg["seed"]),
                iterations=int(bootstrap_cfg["iterations"]),
            )
            mechanical_pass, active_mechanical = self._phase_a_mechanical_pass(arms)
            final_structure = dict(self.structure_audit)
            final_structure.update(
                {
                    "semantic_pair_gradient_ncc_median": arms["I-SV"][
                        "pair_shape_ncc_median"
                    ],
                    "semantic_anchor_gradient_ncc_median": arms["I-SV"][
                        "shape_ncc_median"
                    ],
                    "control_anchor_gradient_ncc_median": arms["I-SPC-V"][
                        "shape_ncc_median"
                    ],
                }
            )
            phase_a = evaluate_phase_a(
                arms,
                structure=final_structure,
                semantic_proxy_delta=float(arms["I-SV"]["semantic_proxy_cosine"])
                - float(arms["I-SPC-V"]["semantic_proxy_cosine"]),
                semantic_vs_control=semantic_contrast,
                fixed_semantic_vs_control=fixed_contrast,
                cue_contrast=cue_contrast,
                mechanical_pass=mechanical_pass,
            )
            phase_a["active_mechanical"] = active_mechanical
            phase_a["semantic_contrast"] = semantic_contrast
            phase_a["fixed_contrast"] = fixed_contrast
            phase_a["cue_contrast"] = cue_contrast
            _write_probe_json(self.artifact_root / "phase_a.json", phase_a)
            if not phase_a["pass"]:
                status.update(
                    {
                        "state": "stopped",
                        "stop_reason": "phase_a_failure_signal",
                        "phase_a": phase_a,
                        "phase_b": "not_run",
                    }
                )
                _write_probe_json(self.artifact_root / "metrics.json", {"arms": arms, "phase_a": phase_a})
                _write_probe_json(self.artifact_root / "status.json", status)
                return status

            phase_b_arms = {
                arm_id: self._run_arm(
                    arm_id,
                    family_arm="I-SV",
                    trc_weight=weight,
                )
                for arm_id, weight in PHASE_B_ARMS.items()
            }
            transformed_contrast = stratified_paired_bootstrap(
                phase_b_arms["I-SV-E1"]["transformed"]["per_image"],
                phase_b_arms["I-SV-E0"]["transformed"]["per_image"],
                seed=int(bootstrap_cfg["seed"]),
                iterations=int(bootstrap_cfg["iterations"]),
            )
            phase_b = evaluate_phase_b(
                phase_b_arms["I-SV-E0"],
                phase_b_arms["I-SV-E1"],
                transformed_contrast,
            )
            phase_b["transformed_contrast"] = transformed_contrast
            _write_probe_json(self.artifact_root / "phase_b.json", phase_b)
            metrics = {
                "arms": arms,
                "phase_a": phase_a,
                "phase_b_arms": phase_b_arms,
                "phase_b": phase_b,
            }
            _write_probe_json(self.artifact_root / "metrics.json", metrics)
            status.update(
                {
                    "state": "completed" if phase_b["pass"] else "stopped",
                    "stop_reason": None
                    if phase_b["pass"]
                    else "phase_b_failure_signal",
                    "phase_a_pass": True,
                    "phase_b_pass": bool(phase_b["pass"]),
                    "mechanism_pass": bool(phase_b["pass"]),
                }
            )
            _write_probe_json(self.artifact_root / "status.json", status)
            return status
        except Exception as error:
            status.update(
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_probe_json(self.artifact_root / "status.json", status)
            raise
