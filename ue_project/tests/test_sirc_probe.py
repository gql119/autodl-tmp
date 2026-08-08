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
