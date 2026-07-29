from __future__ import annotations

import copy

import torch

from ue_framework.methods.bsc_icmo_probe import (
    AFFINE_AUDITS,
    ARM_DEFINITIONS,
    evaluate_icmo_result,
    renderer_mechanical_audit,
    stratified_paired_bootstrap,
    validate_icmo_config,
)


def _config() -> dict:
    digest = "a" * 64
    return {
        "spec": {
            "spec_id": "TAUSB-BSC-ICMO-v1",
            "exp_id": "TAUSB-BSC-ICMO-MECH-S0",
            "seed": 0,
            "source_manifest_sha256": digest,
            "shared_split_sha256": digest,
            "label_sha256": digest,
            "surrogate_checkpoint_sha256": digest,
            "synthetic_global_params_sha256": digest,
            "c2lm_basis_sha256": digest,
        },
        "dataset": {
            "root": "dataset",
            "train_images": "images/train",
            "train_labels": "labels/train",
            "target_class_id": 14,
        },
        "model": {
            "surrogate_checkpoint": "model.pt",
            "num_classes": 20,
            "image_size": 640,
        },
        "background": {
            "source_manifest": "manifest.json",
            "source_local_map": "local.json",
        },
        "carrier": {
            "epsilon": 16.0 / 255.0,
            "resolution": 640,
            "num_bases": 16,
            "basis_seed": 0,
            "synthetic_global_params_path": "global.pt",
            "gamma_seed": 2032,
            "gamma_directions": 256,
            "gamma_bisection_iterations": 32,
            "gamma_chunk_size": 16,
            "initial_seed": 2033,
            "coefficient_max_abs": 0.25,
            "target_rms_ratio": 0.35,
        },
        "split": {
            "manifest": "split.json",
            "required_protocol_prefix": "TAUSB-ALCE-CTX-AUDIT-v1",
        },
        "optimization": {
            "warmup_steps": 4,
            "optimization_steps": 40,
            "learning_rate": 0.01,
            "batch_size": 4,
            "target_route": "easy_cls",
            "lambda_cicr": 1.0,
            "lambda_route": 1.0,
            "lambda_rms": 1.0,
            "prototype_momentum": 0.90,
            "box_teacher_weight": 1.0,
            "align_alpha": 0.5,
            "align_beta": 6.0,
            "assignment_topk": 100,
            "pag_layer_ratios": [0.7, 0.6, 0.4],
            "pag_min_pos": [8, 6, 4],
        },
        "bootstrap": {"seed": 2040, "iterations": 10000},
        "runtime": {"artifact_root": "artifacts", "device": "cpu"},
    }


def test_config_and_arm_matrix_are_frozen() -> None:
    config = _config()
    validate_icmo_config(config)
    assert ARM_DEFINITIONS == {
        "G-C0": ("C0", "global"),
        "G-C2LM": ("C2-LM", "global"),
        "I-C0": ("C0", "instance"),
        "I-C2LM": ("C2-LM", "instance"),
    }
    assert len(AFFINE_AUDITS) == 6

    changed = copy.deepcopy(config)
    changed["carrier"]["gamma_seed"] = 2033
    try:
        validate_icmo_config(changed)
    except ValueError as error:
        assert "gamma_seed" in str(error)
    else:
        raise AssertionError("Changed calibration seed must fail closed.")


def test_renderer_mechanical_audit_distinguishes_coordinate_systems() -> None:
    axis = torch.linspace(-1, 1, 64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pattern = torch.stack(
        (
            torch.sin(2 * torch.pi * xx),
            torch.cos(2 * torch.pi * yy),
            xx + yy,
        )
    )
    audit = renderer_mechanical_audit(pattern)
    assert audit["ncc_median"] >= 0.98
    assert audit["ncc_q25"] >= 0.95
    assert audit["render_paths_distinct"]


def test_stratified_bootstrap_is_paired_and_deterministic() -> None:
    left = []
    right = []
    for index in range(20):
        record = {
            "image_id": f"image_{index:03d}",
            "person_cooccur": index >= 10,
        }
        left.append({**record, "cicr": 0.6 + index * 0.001})
        right.append({**record, "cicr": 0.5 + index * 0.001})
    first = stratified_paired_bootstrap(
        left,
        right,
        iterations=200,
    )
    second = stratified_paired_bootstrap(
        left,
        right,
        iterations=200,
    )
    assert first == second
    assert first["paired_image_count"] == 20
    assert first["stratum_counts"] == {
        "person_only": 10,
        "person_cooccur": 10,
    }
    assert abs(first["paired_median_delta"] - 0.1) < 1e-12
    assert abs(first["ci95"][0] - 0.1) < 1e-12


def _passing_arm() -> dict:
    return {
        "heldout_cicr_median": 0.70,
        "heldout_cicr_q25": 0.30,
        "valid_image_coverage": 0.95,
        "valid_instance_coverage": 0.95,
        "low_energy_ratio": 0.10,
        "zero_norm_ratio": 0.10,
        "route_effect": 0.20,
        "non_target_target_energy_ratio": 0.20,
        "box_residual_energy": 1.0,
        "group_cicr_median": {
            "person_only": 0.68,
            "person_cooccur": 0.72,
            "small": 0.65,
            "medium": 0.70,
            "large": 0.75,
        },
        "group_non_target_target_energy_ratio": {
            "person_only": 0.20,
            "person_cooccur": 0.22,
        },
        "affine_audit": {key: 0.66 for key in AFFINE_AUDITS},
        "canonical_spectrum_energy": {
            "low": 0.35,
            "mid": 0.45,
            "high": 0.20,
            "dc": 0.0,
        },
        "source_max_abs_correlation": 0.10,
        "active_pixel_linf": 0.06,
        "coefficient_saturation_ratio": 0.10,
        "active_basis_fraction": 0.50,
        "top1_basis_energy_share": 0.30,
        "calibration_cicr_gain": 0.05,
        "heldout_cicr_gain": 0.05,
        "calibration_heldout_gap": 0.05,
        "finite": True,
    }


def test_success_and_independent_failure_gates_are_separate() -> None:
    arms = {arm: _passing_arm() for arm in ARM_DEFINITIONS}
    arms["I-C0"]["non_target_target_energy_ratio"] = 0.25
    arms["I-C0"]["box_residual_energy"] = 1.1
    contrast = {
        "paired_image_count": 90,
        "paired_median_delta": 0.09,
        "arm_median_delta": 0.09,
        "ci95": [0.01, 0.15],
    }
    contrasts = {
        "I-C2LM_vs_G-C2LM": dict(contrast),
        "I-C2LM_vs_I-C0": dict(contrast),
        "I-C0_vs_G-C0": dict(contrast),
    }
    mechanical = {
        "ncc_median": 0.99,
        "ncc_q25": 0.97,
        "render_paths_distinct": True,
    }
    result = evaluate_icmo_result(
        arms,
        contrasts,
        initial_rms_ratio=1.0,
        active_rms_ratio=1.02,
        mechanical=mechanical,
        hashes_complete=True,
    )
    assert result["pass"]
    assert all(result["checks"].values())
    assert not any(result["failure_signals"].values())

    arms["I-C2LM"]["source_max_abs_correlation"] = 0.31
    failed = evaluate_icmo_result(
        arms,
        contrasts,
        initial_rms_ratio=1.0,
        active_rms_ratio=1.02,
        mechanical=mechanical,
        hashes_complete=True,
    )
    assert not failed["pass"]
    assert failed["failure_signals"]["source_semantic_dependence"]
