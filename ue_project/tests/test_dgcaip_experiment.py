from __future__ import annotations

import copy
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml

from ue_framework.methods import dgcaip_experiment, sdh_experiment
from ue_framework.methods.dgcaip import DGCAIPInstanceTerm, DGCAIPResult
from ue_framework.methods.dgcaip_experiment import (
    _accepted_target_progress_pass,
    _v4_layered_gate_decision,
    DGCAIP_ARMS,
    R3_DIAGNOSTIC_ARMS,
    _constraint_limits,
    _strict_component_candidate_metrics,
    _p1_replay_report,
    _support_outside_linf,
    _strict_candidate_metrics,
    _validate_d0_report_binding,
)
from ue_framework.methods.sdh_experiment import (
    DGCAIP_DATASET_CGR_PROXY_SPEC_ID,
    DGCAIP_R3_DIAG_SPEC_ID,
    DGCAIP_R4_DIAG_SPEC_ID,
    DGCAIP_P4_E20_SPEC_ID,
    DGCAIP_SPEC_ID,
    validate_sdh_experiment_config,
)


def test_support_outside_linf_accepts_boolean_union_support() -> None:
    observation = types.SimpleNamespace(
        rendered=types.SimpleNamespace(
            perturbation=torch.tensor([[[[0.5, -0.2], [0.1, 0.0]]]]),
            union_support=torch.tensor([[[[True, False], [False, True]]]]),
        )
    )
    assert _support_outside_linf((observation,)) == pytest.approx(0.2)


E2E = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_e2e_v0_mechanism.yaml"
)
R3 = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_r3_diag_v1.yaml"
)
R4 = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_r4_d0_binding_fix_v1.yaml"
)
P4_E20 = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_p4_sparse_e20_v1.yaml"
)


def _config():
    config = yaml.safe_load(E2E.read_text(encoding="utf-8"))
    config["spec"]["spec_id"] = DGCAIP_SPEC_ID
    config["dgcaip"] = {
        "run_mode": "d0",
        "temperature": 2.0,
        "protection_ratio": 0.25,
        "classification_tolerance": 0.005,
        "box_tolerance": 0.02,
        "alignment_tolerance": 0.05,
        "js_backtracking_epsilon": 1.0e-9,
        "minimum_rank_instances": 4,
        "expected_split_sha256": "d" * 64,
        "source_p1_state": "/root/data/p1_state.pt",
        "source_p1_state_sha256": "a" * 64,
    }
    return config


def _dataset_config():
    config = _config()
    config["spec"]["spec_id"] = DGCAIP_DATASET_CGR_PROXY_SPEC_ID
    config["dgcaip"].pop("protection_ratio")
    config["dgcaip"]["run_mode"] = "dataset_risk_scan"
    config["model"]["protection_surrogate_snapshots"] = [
        {"id": "e1", "checkpoint": "/root/data/e1.pt", "sha256": "1" * 64},
        {"id": "e5", "checkpoint": "/root/data/e5.pt", "sha256": "5" * 64},
        {"id": "e20", "checkpoint": "/root/data/e20.pt", "sha256": "2" * 64},
    ]
    config["dataset_ranking"] = {
        "js_weight": 0.7,
        "kl_weight": 0.3,
        "top_fraction": 0.25,
        "minimum_coverage": 0.90,
        "high_risk_replay_fraction": 0.50,
    }
    config["strict_route"] = {
        "repair_floor_fraction": 0.05,
        "max_repair_norm_ratio": 0.25,
        "max_projection_iterations": 64,
    }
    config["proxy_agreement"] = {
        "minimum_spearman": 0.40,
        "minimum_top_overlap": 0.50,
        "minimum_coverage": 0.90,
    }
    return config


def test_dataset_cgr_proxy_config_is_versioned_and_fail_closed() -> None:
    config = _dataset_config()
    validate_sdh_experiment_config(config)
    assert config["dgcaip"]["run_mode"] == "dataset_risk_scan"
    assert [item["id"] for item in config["model"]["protection_surrogate_snapshots"]] == [
        "e1",
        "e5",
        "e20",
    ]
    wrong = copy.deepcopy(config)
    wrong["dataset_ranking"]["js_weight"] = 0.6
    with pytest.raises(ValueError, match="js_weight"):
        validate_sdh_experiment_config(wrong)
    wrong = copy.deepcopy(config)
    wrong["model"]["protection_surrogate_snapshots"][1]["id"] = "other"
    with pytest.raises(ValueError, match="e1/e5/e20"):
        validate_sdh_experiment_config(wrong)

    strict = copy.deepcopy(config)
    strict["dgcaip"]["run_mode"] = "strict_mechanism"
    with pytest.raises(ValueError, match="risk_bank path"):
        validate_sdh_experiment_config(strict)
    strict["dataset_ranking"].update(
        {
            "risk_bank": "/root/data/risk.json",
            "risk_bank_file_sha256": "a" * 64,
            "risk_bank_canonical_sha256": "b" * 64,
            "replay_manifest": "/root/data/replay.json",
            "replay_manifest_file_sha256": "c" * 64,
        }
    )
    validate_sdh_experiment_config(strict)

    short = copy.deepcopy(config)
    short["dgcaip"]["run_mode"] = "short_victim_risk_scan"
    short["dgcaip"].pop("source_p1_state")
    short["dgcaip"].pop("source_p1_state_sha256")
    short["dgcaip"]["source_carrier_state"] = "/root/data/p5.pt"
    short["dgcaip"]["source_carrier_state_sha256"] = "f" * 64
    short["model"]["protection_surrogate_snapshots"] = [
        {"id": "v3", "checkpoint": "/root/data/v3.pt", "sha256": "3" * 64}
    ]
    validate_sdh_experiment_config(short)


def test_dataset_cgr_proxy_routes_to_dedicated_stage(monkeypatch) -> None:
    from ue_framework.methods import dgcaip_dataset_risk_experiment

    config = _dataset_config()
    captured = {}

    def fake_run(bound, *, config_base):
        captured["spec_id"] = bound["spec"]["spec_id"]
        captured["config_base"] = config_base
        return {"schema": "dataset-risk-test"}

    monkeypatch.setattr(
        dgcaip_dataset_risk_experiment,
        "run_dataset_cgr_proxy_stage",
        fake_run,
    )
    result = sdh_experiment.run_mechanism_pilot(
        config, config_base=Path("/tmp/project")
    )
    assert result == {"schema": "dataset-risk-test"}
    assert captured["spec_id"] == DGCAIP_DATASET_CGR_PROXY_SPEC_ID


def test_strict_candidate_missing_metric_is_rejected_with_finite_sentinel() -> None:
    filtered = _strict_candidate_metrics(
        {"e1/7:js": 0.02},
        {"e1/7:js": 0.02, "e5/7:probability": 0.01},
    )
    assert filtered["e1/7:js"] == pytest.approx(0.02)
    assert filtered["e5/7:probability"] == pytest.approx(1.01)


def _dg_result() -> DGCAIPResult:
    term = DGCAIPInstanceTerm(
        batch_index=0,
        gt_index=1,
        class_id=7,
        positive_count=2,
        geometry_risk=1.0,
        divergence_rank=1.0,
        weight=1.0,
        classification_damage=torch.tensor(0.015),
        box_damage=torch.tensor(0.04),
        alignment_damage=torch.tensor(0.08),
        classification_loss=torch.tensor(0.01),
        box_loss=torch.tensor(0.02),
        alignment_loss=torch.tensor(0.03),
        distribution_loss=torch.tensor(0.04),
        clean_to_poison_kl=torch.tensor(0.08),
    )
    return DGCAIPResult(
        loss=torch.tensor(0.10),
        instances=(term,),
        active_classes=(7,),
        per_class_loss={7: torch.tensor(0.10)},
        per_class_instance_count={7: 1},
        eligible_instance_count=1,
        covered_instance_count=1,
        coverage=1.0,
    )


def test_dgcaip_config_and_arm_switches_are_frozen() -> None:
    config = _config()
    validate_sdh_experiment_config(config)
    assert DGCAIP_ARMS == {
        "P1-R": "off",
        "P2-CAIP": "caip",
        "P3-DIST": "dist",
        "P4-DGCAIP": "dgcaip",
    }


def test_mechanism_mode_requires_hashed_passed_d0_artifact() -> None:
    config = _config()
    config["dgcaip"]["run_mode"] = "mechanism"
    with pytest.raises(ValueError, match="passed D0"):
        validate_sdh_experiment_config(config)
    config["dgcaip"]["d0_report"] = "/root/data/d0_locator.json"
    config["dgcaip"]["d0_report_sha256"] = "b" * 64
    config["dgcaip"]["source_p1_metrics"] = "/root/data/mechanism_metrics.json"
    config["dgcaip"]["source_p1_metrics_sha256"] = "c" * 64
    config["dgcaip"]["p1_replay_absolute_tolerance"] = 1.0e-6
    config["dgcaip"]["p1_replay_relative_tolerance"] = 1.0e-4
    validate_sdh_experiment_config(config)


def test_r3_diagnostic_config_is_exact_and_fail_closed() -> None:
    config = yaml.safe_load(R3.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["spec"]["spec_id"] == DGCAIP_R3_DIAG_SPEC_ID
    assert config["dgcaip"]["run_mode"] == "r3_diag"
    assert config["dgcaip"]["r3_diagnostics"]["enabled"] is True
    assert config["mechanism"]["max_seconds"] == 600
    assert config["mechanism"]["optimization_steps"] == 8
    assert R3_DIAGNOSTIC_ARMS == {
        "P1-A": "off",
        "P1-B": "off",
        "P2-CAIP": "caip",
        "P4-DGCAIP": "dgcaip",
    }
    assert "P3-DIST" not in R3_DIAGNOSTIC_ARMS
    assert config["runtime"]["artifact_root"].endswith(
        "TAUSB-SDH-DGCAIP-S0-R3-DIAG"
    )

    disabled = copy.deepcopy(config)
    disabled["dgcaip"]["r3_diagnostics"]["enabled"] = False
    with pytest.raises(ValueError, match="enabled=true"):
        validate_sdh_experiment_config(disabled)

    wrong_cap = copy.deepcopy(config)
    wrong_cap["mechanism"]["max_seconds"] = 601
    with pytest.raises(ValueError, match="600 seconds"):
        validate_sdh_experiment_config(wrong_cap)


def test_r3_config_routes_to_dgcaip_pilot(monkeypatch) -> None:
    config = yaml.safe_load(R3.read_text(encoding="utf-8"))
    captured = {}

    def fake_run(bound, *, config_base):
        captured["spec_id"] = bound["spec"]["spec_id"]
        captured["config_base"] = config_base
        return {"schema": "test"}

    monkeypatch.setattr(dgcaip_experiment, "run_dgcaip_pilot", fake_run)
    result = sdh_experiment.run_mechanism_pilot(
        config, config_base=Path("/tmp/project")
    )
    assert result == {"schema": "test"}
    assert captured == {
        "spec_id": DGCAIP_R3_DIAG_SPEC_ID,
        "config_base": Path("/tmp/project"),
    }


def test_r4_d0_producer_binding_is_explicit_and_fail_closed() -> None:
    config = yaml.safe_load(R4.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["spec"]["spec_id"] == DGCAIP_R4_DIAG_SPEC_ID
    assert config["dgcaip"]["expected_d0_spec_id"] == DGCAIP_SPEC_ID
    assert config["runtime"]["artifact_root"].endswith(
        "TAUSB-SDH-DGCAIP-S0-R4-DIAG"
    )

    report = {
        "decision": {"pass": True},
        "spec_id": DGCAIP_SPEC_ID,
        "split_hash": config["dgcaip"]["expected_split_sha256"],
        "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
    }
    _validate_d0_report_binding(report, config, config["dgcaip"])

    wrong_expected = copy.deepcopy(config)
    wrong_expected["dgcaip"]["expected_d0_spec_id"] = DGCAIP_R3_DIAG_SPEC_ID
    with pytest.raises(ValueError, match="expected_d0_spec_id"):
        validate_sdh_experiment_config(wrong_expected)

    wrong_report = copy.deepcopy(report)
    wrong_report["spec_id"] = DGCAIP_R3_DIAG_SPEC_ID
    with pytest.raises(ValueError, match="SpecID mismatch"):
        _validate_d0_report_binding(wrong_report, config, config["dgcaip"])


def test_p4_e20_config_freezes_strict_production_and_repair_binding() -> None:
    config = yaml.safe_load(P4_E20.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["spec"]["spec_id"] == DGCAIP_P4_E20_SPEC_ID
    assert config["dgcaip"]["run_mode"] == "production_e20"
    assert config["runtime"]["strict_determinism"] is True
    assert config["mechanism"]["optimization_steps"] == 8
    assert config["mechanism"]["max_seconds"] == 1200

    non_strict = copy.deepcopy(config)
    non_strict["runtime"]["strict_determinism"] = False
    with pytest.raises(ValueError, match="strict determinism"):
        validate_sdh_experiment_config(non_strict)

    wrong_repair = copy.deepcopy(config)
    wrong_repair["dgcaip"]["repair_report_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="repair report hash"):
        validate_sdh_experiment_config(wrong_repair)


def test_d0_binding_legacy_fallback_and_existing_gates() -> None:
    config = _config()
    dgcaip = config["dgcaip"]
    report = {
        "decision": {"pass": True},
        "spec_id": DGCAIP_SPEC_ID,
        "split_hash": dgcaip["expected_split_sha256"],
        "source_p1_state_sha256": dgcaip["source_p1_state_sha256"],
    }
    assert "expected_d0_spec_id" not in dgcaip
    _validate_d0_report_binding(report, config, dgcaip)

    mutations = {
        "D0 gate": ("decision", {"pass": False}),
        "split hash": ("split_hash", "e" * 64),
        "source P1 hash": ("source_p1_state_sha256", "f" * 64),
    }
    for message, (key, value) in mutations.items():
        mutated = copy.deepcopy(report)
        mutated[key] = value
        with pytest.raises(ValueError, match=message):
            _validate_d0_report_binding(mutated, config, dgcaip)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 1.0),
        ("protection_ratio", 0.5),
        ("classification_tolerance", 0.0),
        ("minimum_rank_instances", 3),
    ],
)
def test_frozen_dgcaip_config_values_fail_closed(key, value) -> None:
    config = copy.deepcopy(_config())
    config["dgcaip"][key] = value
    with pytest.raises(ValueError, match="DG-CAIP"):
        validate_sdh_experiment_config(config)


def test_backtracking_uses_zero_structural_violation_and_nonincreasing_js() -> None:
    result = _dg_result()
    without_js = _constraint_limits(
        result,
        include_js=False,
        js_epsilon=1.0e-9,
    )
    assert without_js == {
        "7:probability": 0.0,
        "7:iou": 0.0,
        "7:alignment": 0.0,
    }
    with_js = _constraint_limits(
        result,
        include_js=True,
        js_epsilon=1.0e-9,
    )
    assert with_js["7:js"] == pytest.approx(0.040000001)


def _p1_summary(value: float = 0.1):
    return {
        "nla_macro_loss": value,
        "probability_drop_macro": value,
        "route_loss": value,
        "valid_instance_coverage": value,
        "cicr_cosine_median": value,
        "dlfc_cosine_median": value,
        "backtrack_skip_ratio": value,
        "probability_drop_by_class": {"1": value},
        "steps": [
            {
                "route_mode": "projected_target_plus_nla",
                "accepted": True,
                "backtrack_attempts": 1,
                "constraint_rank": 1,
                "null_dimension": 5,
                "attack_retention": value,
                "max_projected_row_dot": value,
                "max_final_row_dot": value,
            }
        ],
    }


def test_p1_replay_report_binds_numeric_and_structural_regression() -> None:
    reference = {"arms": {"P1": _p1_summary()}}
    observed = _p1_summary(0.1000005)
    observed["probability_drop_by_class"] = {1: 0.1000005}
    report = _p1_replay_report(
        observed,
        reference,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-4,
    )
    assert report["pass"] is True

    changed = _p1_summary()
    changed["steps"][0]["accepted"] = False
    report = _p1_replay_report(
        changed,
        reference,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-4,
    )
    assert report["pass"] is False


def test_mechanism_rejects_placeholder_hash_and_unfrozen_replay_tolerance() -> None:
    config = _config()
    config["dgcaip"].update(
        {
            "run_mode": "mechanism",
            "d0_report": "/root/data/d0_locator.json",
            "d0_report_sha256": "0" * 64,
            "source_p1_metrics": "/root/data/mechanism_metrics.json",
            "source_p1_metrics_sha256": "c" * 64,
            "p1_replay_absolute_tolerance": 1.0e-6,
            "p1_replay_relative_tolerance": 1.0e-4,
        }
    )
    with pytest.raises(ValueError, match="d0_report_sha256"):
        validate_sdh_experiment_config(config)
    config["dgcaip"]["d0_report_sha256"] = "b" * 64
    config["dgcaip"]["p1_replay_relative_tolerance"] = 1.0e-3
    with pytest.raises(ValueError, match="p1_replay_relative_tolerance"):
        validate_sdh_experiment_config(config)


def test_load_engine_explicitly_binds_frozen_dgcaip_parameters(monkeypatch) -> None:
    config = _config()
    captured = {}

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeWrapper:
        model = FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        types.SimpleNamespace(YOLO=lambda _path: FakeWrapper()),
    )
    monkeypatch.setattr(
        dgcaip_experiment,
        "SDHObservationEngine",
        lambda _model, **kwargs: captured.update(kwargs) or kwargs,
    )
    dgcaip_experiment._load_engine(config, config_base=Path("."))
    assert captured["dgcaip_temperature"] == 2.0
    assert captured["dgcaip_classification_tolerance"] == 0.005
    assert captured["dgcaip_box_tolerance"] == 0.02
    assert captured["dgcaip_alignment_tolerance"] == 0.05
    assert captured["dgcaip_minimum_rank_instances"] == 4


def test_v3_target_progress_gate_uses_the_postcast_tolerance() -> None:
    accepted = [{"target_progress": 0.5999999642372131}]
    assert _accepted_target_progress_pass(
        accepted,
        minimum=0.60,
        tolerance=1.0e-6,
    )
    assert not _accepted_target_progress_pass(
        accepted,
        minimum=0.60,
        tolerance=0.0,
    )
    assert not _accepted_target_progress_pass(
        [{"target_progress": 0.599998}],
        minimum=0.60,
        tolerance=1.0e-6,
    )


def test_v4_gate_uses_actual_acceptance_and_final_progress_only_for_promotion() -> None:
    decision = _v4_layered_gate_decision(
        {"finite": True, "frozen_modules_unchanged": True},
        {"at_least_one_update": True, "adapter_changed": True},
        accepted_update_ratio=0.50,
        minimum_accepted_update_ratio=0.50,
        target_progress_pass=True,
    )
    assert decision["runtime_pass"] is True
    assert decision["mechanism_valid"] is True
    assert decision["promotion_pass"] is True
    assert decision["pass"] is True
    assert "attack_retention" not in decision["checks"]
    assert "backtrack_skip" not in decision["checks"]

    blocked = _v4_layered_gate_decision(
        {"finite": True},
        {"at_least_one_update": True},
        accepted_update_ratio=0.375,
        minimum_accepted_update_ratio=0.50,
        target_progress_pass=True,
    )
    assert blocked["promotion_pass"] is False
    assert blocked["pass"] is False


def test_v3_candidate_registry_fails_closed_on_missing_or_extra_rows() -> None:
    baselines = {"e1/1:nla": 0.2, "e1/1:probability": 0.3}
    assert _strict_component_candidate_metrics(
        dict(baselines), baselines
    ) == pytest.approx(baselines)
    with pytest.raises(ValueError, match="candidate constraint keys differ"):
        _strict_component_candidate_metrics(
            {"e1/1:nla": 0.2}, baselines
        )
    with pytest.raises(ValueError, match="candidate constraint keys differ"):
        _strict_component_candidate_metrics(
            {**baselines, "e1/1:js": 0.0}, baselines
        )
