from __future__ import annotations

import copy
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml

from ue_framework.methods import dgcaip_experiment
from ue_framework.methods.dgcaip import DGCAIPInstanceTerm, DGCAIPResult
from ue_framework.methods.dgcaip_experiment import (
    DGCAIP_ARMS,
    _constraint_limits,
    _p1_replay_report,
)
from ue_framework.methods.sdh_experiment import (
    DGCAIP_SPEC_ID,
    validate_sdh_experiment_config,
)


E2E = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_e2e_v0_mechanism.yaml"
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


def _dg_result() -> DGCAIPResult:
    term = DGCAIPInstanceTerm(
        batch_index=0,
        gt_index=1,
        class_id=7,
        positive_count=2,
        geometry_risk=1.0,
        divergence_rank=1.0,
        weight=1.0,
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
