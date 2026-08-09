from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ue_framework.methods.malc import MALCResult
from ue_framework.methods.malc_mechanism import (
    MALCMechanismBatch,
    aggregate_malc_mechanism_batches,
    assert_matched_mechanism_configs,
    evaluate_malc_mechanism_gate,
    write_malc_mechanism_report,
)


def _result(
    cosine: float,
    log_energy: float,
    *,
    coverage: float = 0.9,
    zero: float = 0.1,
    floor: float = 0.9,
    valid_scales: tuple[int, int, int] = (1, 1, 0),
) -> MALCResult:
    scalar = torch.tensor(0.1)
    return MALCResult(
        loss=scalar,
        direction_loss=scalar,
        magnitude_loss=scalar,
        floor_loss=scalar,
        per_scale_loss=(scalar, scalar, scalar),
        per_scale_valid_count=valid_scales,
        per_scale_assigned_count=(1, 1, 1),
        scale_contribution_share=(0.4, 0.4, 0.2),
        per_instance_cosine=torch.tensor([cosine, cosine + 0.02]),
        per_instance_log_energy=torch.tensor([log_energy, -log_energy]),
        valid_instance_count=2,
        total_instance_count=2,
        valid_instance_coverage=coverage,
        zero_norm_ratio=zero,
        floor_pass_ratio=floor,
        valid_scale_count=sum(value > 0 for value in valid_scales),
    )


def _batch(
    cosine: float,
    energy_spread: float,
    *,
    retention: float = 0.5,
    mode: str = "projected_target",
    leakage: float = 0.4,
    box_energy: float = 0.2,
) -> MALCMechanismBatch:
    return MALCMechanismBatch(
        malc=_result(cosine, energy_spread),
        cgr_attack_retention=retention,
        cgr_max_projected_row_dot=1e-7,
        cgr_selected_mode=mode,
        non_target_target_energy_ratio=leakage,
        box_residual_energy=box_energy,
        size_groups=("small", "large"),
        cooccur_flags=(False, True),
    )


def test_matched_configs_allow_only_the_malc_switch() -> None:
    a0 = {"method": {"enable_malc": False, "enable_cgr": True}, "seed": 0}
    a1 = {"method": {"enable_malc": True, "enable_cgr": True}, "seed": 0}
    assert_matched_mechanism_configs(a0, a1)
    a1["seed"] = 1
    with pytest.raises(ValueError, match="differ only"):
        assert_matched_mechanism_configs(a0, a1)


def test_heldout_aggregation_keeps_scale_and_group_diagnostics() -> None:
    metrics = aggregate_malc_mechanism_batches(
        [_batch(0.5, 0.2), _batch(0.6, 0.1)],
        split="heldout",
    )
    assert metrics["heldout_batch_count"] == 2
    assert metrics["valid_scale_count_at_0_80"] == 2
    assert "size:small" in metrics["groups"]
    assert "context:cooccur" in metrics["groups"]
    with pytest.raises(ValueError, match="heldout"):
        aggregate_malc_mechanism_batches([_batch(0.5, 0.2)], split="calibration")


def test_gate_passes_only_when_all_success_and_leakage_checks_pass() -> None:
    a0 = aggregate_malc_mechanism_batches(
        [_batch(0.40, 0.30, leakage=0.5, box_energy=0.4)] * 2,
        split="heldout",
    )
    a1 = aggregate_malc_mechanism_batches(
        [_batch(0.60, 0.10, leakage=0.55, box_energy=0.45)] * 2,
        split="heldout",
    )
    gate = evaluate_malc_mechanism_gate(a0, a1)
    assert gate["pass"]
    assert gate["allow_fresh_victim"]
    assert all(gate["success_signals"].values())
    assert not any(gate["failure_signals"].values())


def test_gate_rejects_leakage_even_when_concentration_improves() -> None:
    a0 = aggregate_malc_mechanism_batches(
        [_batch(0.40, 0.30, leakage=0.2, box_energy=0.2)] * 2,
        split="heldout",
    )
    a1 = aggregate_malc_mechanism_batches(
        [_batch(0.60, 0.10, leakage=0.4, box_energy=0.4)] * 2,
        split="heldout",
    )
    gate = evaluate_malc_mechanism_gate(a0, a1)
    assert not gate["pass"]
    assert gate["failure_signals"]["non_target_target_energy_leakage"]
    assert gate["failure_signals"]["box_residual_energy_leakage"]


def test_repair_plus_skip_ratio_is_a_strict_gate() -> None:
    a0 = aggregate_malc_mechanism_batches(
        [_batch(0.40, 0.30)] * 2,
        split="heldout",
    )
    a1 = aggregate_malc_mechanism_batches(
        [
            _batch(0.60, 0.10, mode="repair_only"),
            _batch(0.60, 0.10, mode="projected_target"),
        ],
        split="heldout",
    )
    gate = evaluate_malc_mechanism_gate(a0, a1)
    assert not gate["success_signals"]["cgr_repair_skip"]
    assert not gate["pass"]


def test_mechanism_report_is_explicitly_scoped_and_never_overwritten() -> None:
    a0 = aggregate_malc_mechanism_batches(
        [_batch(0.40, 0.30)] * 2,
        split="heldout",
    )
    a1 = aggregate_malc_mechanism_batches(
        [_batch(0.60, 0.10)] * 2,
        split="heldout",
    )
    gate = evaluate_malc_mechanism_gate(a0, a1)
    output = Path("tmp") / "test_malc_mechanism_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    try:
        write_malc_mechanism_report(
            output,
            a0=a0,
            a1=a1,
            gate=gate,
            split_hash="fixed-s0",
        )
        text = output.read_text(encoding="utf-8")
        assert "heldout_mechanism_only_not_fresh_victim_ue" in text
        assert '"split_hash": "fixed-s0"' in text
        with pytest.raises(FileExistsError):
            write_malc_mechanism_report(
                output,
                a0=a0,
                a1=a1,
                gate=gate,
                split_hash="fixed-s0",
            )
    finally:
        if output.exists():
            output.unlink()
