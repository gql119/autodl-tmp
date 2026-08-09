from __future__ import annotations

from copy import deepcopy

import torch

from ue_framework.methods.cicr import CICRPrototypeBank
from ue_framework.methods.semantic_residual_carrier import build_semantic_carrier_bank
from ue_framework.methods.sirc_probe import (
    EOTParameters,
    apply_object_relative_eot,
    assert_heldout_bank_immutable,
    deterministic_eot_parameters,
    evaluate_phase_a,
    evaluate_phase_b,
    paired_object_relative_eot,
    semantic_structure_audit,
    validate_sirc_config,
)
from ue_framework.tools.probe_tausb_sirc import load_config


def _bank():
    generator = torch.Generator(device="cpu").manual_seed(8)
    anchor = torch.rand((3, 40, 54), generator=generator)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 40),
        torch.linspace(-1, 1, 54),
        indexing="ij",
    )
    anchor = (0.15 * anchor + 0.85 * (torch.exp(-12 * xx.square()) * (yy > -0.3))).clamp(0, 1)
    donors = [
        torch.rand((3, 42 + index, 50 + index), generator=generator)
        for index in range(4)
    ]
    return build_semantic_carrier_bank(anchor, donors, resolution=48), anchor


def test_structure_audit_distinguishes_shared_phase_from_controls() -> None:
    bank, anchor = _bank()
    audit = semantic_structure_audit(bank, anchor)
    assert audit["semantic_pair_gradient_ncc_median"] > 0.5
    assert audit["semantic_anchor_gradient_ncc_median"] > audit[
        "control_anchor_gradient_ncc_median"
    ]
    assert audit["pairwise_normalized_amplitude_distance_median"] > 0


def test_eot_is_deterministic_paired_and_differentiable() -> None:
    first = deterministic_eot_parameters(
        "2008_000001", step=3, sample_index=1, seed=2105
    )
    second = deterministic_eot_parameters(
        "2008_000001", step=3, sample_index=1, seed=2105
    )
    assert first == second
    clean = torch.rand((1, 3, 16, 16))
    residual = torch.full_like(clean, 0.01, requires_grad=True)
    poisoned = clean + residual
    clean_view, poisoned_view = paired_object_relative_eot(
        clean,
        poisoned,
        boxes_by_image=([(2, 2, 14, 14)],),
        image_ids=("2008_000001",),
        step=3,
        sample_index=1,
        seed=2105,
    )
    assert clean_view.shape == clean.shape
    assert poisoned_view.shape == poisoned.shape
    loss = (poisoned_view - clean_view).square().mean()
    loss.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert float(residual.grad.norm()) > 0


def test_object_relative_warp_averages_overlap_and_stays_finite() -> None:
    image = torch.rand((3, 12, 12))
    parameters = EOTParameters(1.05, 0.02, -0.03, 0.6, True)
    transformed = apply_object_relative_eot(
        image,
        [(1, 1, 8, 8), (4, 4, 11, 11)],
        parameters,
    )
    assert transformed.shape == image.shape
    assert torch.isfinite(transformed).all()


def test_heldout_prototype_state_is_checked_exactly() -> None:
    bank = CICRPrototypeBank(num_scales=3, momentum=0.9)
    values = [torch.randn((5, 4)) for _ in range(3)]
    bank.calibrate_energy_floors(values)
    before = deepcopy(bank.state_dict())
    assert_heldout_bank_immutable(bank, before)
    mutated = deepcopy(bank.state_dict())
    mutated["energy_floors"][0] = 99.0
    bank.load_state_dict(mutated)
    try:
        assert_heldout_bank_immutable(bank, before)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("Prototype mutation was not detected.")


def test_phase_b_gate_requires_gain_retention_and_preservation() -> None:
    e0 = {
        "heldout_cicr_median": 0.62,
        "non_target_target_energy_ratio": 0.30,
        "box_residual_energy": 0.20,
        "valid_instance_coverage": 0.80,
        "route_effect": 0.20,
        "finite": True,
    }
    e1 = {
        "heldout_cicr_median": 0.61,
        "non_target_target_energy_ratio": 0.31,
        "box_residual_energy": 0.21,
        "valid_instance_coverage": 0.82,
        "route_effect": 0.19,
        "finite": True,
    }
    result = evaluate_phase_b(
        e0,
        e1,
        {"paired_median_delta": 0.06, "ci95": [0.01, 0.10]},
    )
    assert result["pass"]
    failed = evaluate_phase_b(
        e0,
        {**e1, "non_target_target_energy_ratio": 0.40},
        {"paired_median_delta": 0.06, "ci95": [0.01, 0.10]},
    )
    assert not failed["pass"]
    assert failed["failure_signals"]["collateral_leakage"]


def _passing_phase_a_inputs():
    arm = {
        "heldout_cicr_median": 0.65,
        "heldout_cicr_q25": 0.30,
        "valid_instance_coverage": 0.85,
        "robustness_retention": {
            "affine": 0.80,
            "blur": 0.80,
            "grayscale": 0.80,
            "jpeg50": 0.80,
        },
        "route_effect": 0.20,
        "cue_gain_median": 0.20,
        "non_target_target_energy_ratio": 0.30,
        "box_residual_energy": 0.20,
        "group_cicr_median": {
            "person_only": 0.60,
            "person_cooccur": 0.60,
        },
        "coefficient_saturation_ratio": 0.10,
        "active_basis_fraction": 0.50,
        "top1_basis_energy_share": 0.40,
        "high_frequency_energy_ratio": 0.20,
        "zero_norm_ratio": 0.01,
        "calibration_cicr_gain": 0.05,
        "heldout_cicr_gain": 0.04,
        "finite": True,
    }
    arms = {
        arm_id: deepcopy(arm)
        for arm_id in ("I-C2LM", "I-SPC-F", "I-SF", "I-SPC-V", "I-SV")
    }
    paired = {"paired_median_delta": 0.06, "ci95": [0.01, 0.10]}
    structure = {
        "semantic_pair_gradient_ncc_median": 0.75,
        "semantic_anchor_gradient_ncc_median": 0.65,
        "control_anchor_gradient_ncc_median": 0.10,
        "pairwise_normalized_amplitude_distance_median": 0.20,
    }
    return arms, paired, structure


def test_phase_a_uses_phase_sensitive_ncc_and_keeps_p5_proxy_diagnostic() -> None:
    arms, paired, structure = _passing_phase_a_inputs()
    result = evaluate_phase_a(
        arms,
        structure=structure,
        semantic_proxy_delta=0.01,
        semantic_vs_control=paired,
        fixed_semantic_vs_control=paired,
        cue_contrast=paired,
        mechanical_pass=True,
    )
    assert result["pass"]
    assert result["diagnostics"]["semantic_proxy_delta"] == 0.01
    assert abs(result["diagnostics"]["structure_ncc_margin"] - 0.55) < 1e-12

    failed = evaluate_phase_a(
        arms,
        structure={
            **structure,
            "semantic_anchor_gradient_ncc_median": 0.35,
            "control_anchor_gradient_ncc_median": 0.30,
        },
        semantic_proxy_delta=0.90,
        semantic_vs_control=paired,
        fixed_semantic_vs_control=paired,
        cue_contrast=paired,
        mechanical_pass=True,
    )
    assert not failed["pass"]
    assert failed["failure_signals"]["structure_destroyed"]


def test_formal_config_is_frozen_and_parseable() -> None:
    from pathlib import Path

    path = Path(
        "ue_framework/configs/exp_voc_person_tausb_sirc_probe.yaml"
    )
    config = load_config(path)
    validate_sirc_config(config)
    changed = deepcopy(config)
    changed["carrier"]["variant_seed"] = 999
    try:
        validate_sirc_config(changed)
    except ValueError as error:
        assert "variant_seed" in str(error)
    else:
        raise AssertionError("Changed frozen variant seed was accepted.")
    changed_eot = deepcopy(config)
    changed_eot["eot"]["seed"] = 999
    try:
        validate_sirc_config(changed_eot)
    except ValueError as error:
        assert "EOT seed" in str(error)
    else:
        raise AssertionError("Changed frozen EOT seed was accepted.")
