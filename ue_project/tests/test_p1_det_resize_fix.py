from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ue_framework.methods.p1_determinism_audit import (
    AUDIT_SPEC_IDS,
    RESIZE_REPAIR_SPEC_ID,
)
from ue_framework.methods.p1_determinism_experiment import (
    _choose_resize_repair_replay_decision,
)
from ue_framework.methods.sdh_experiment import (
    DGCAIP_P1_DET_RESIZE_FIX_SPEC_ID,
    run_mechanism_pilot,
    validate_sdh_experiment_config,
)


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = (
    PROJECT_ROOT
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_p1_det_resize_fix_v1.yaml"
)
PRE_RUN = (
    Path(__file__).parents[2]
    / "research_workspace"
    / "experiments"
    / "TAUSB-SDH-DGCAIP-S0-P1-DET-RESIZE-FIX"
    / "pre_run"
)


def _pair(*, exact: bool) -> dict:
    return {
        "valid": True,
        "pass": exact,
        "bitwise_pass": exact,
    }


def test_resize_repair_config_is_frozen_and_generic_runner_is_blocked():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert RESIZE_REPAIR_SPEC_ID == DGCAIP_P1_DET_RESIZE_FIX_SPEC_ID
    assert RESIZE_REPAIR_SPEC_ID in AUDIT_SPEC_IDS
    assert config["audit"]["normal_lanes"] == ["reset"]
    assert config["audit"]["strict_lanes"] == ["fresh"]
    assert config["audit"]["total_hard_cap_seconds"] == 480
    assert config["mechanism"]["optimization_steps"] == 1
    with pytest.raises(ValueError, match="dedicated zero-update runner"):
        run_mechanism_pilot(config, config_base=Path("/tmp/project"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_commit", "0" * 40),
        ("normal_lanes", ["shared", "reset", "fresh"]),
        ("strict_lanes", ["reset"]),
        ("total_hard_cap_seconds", 481),
    ],
)
def test_resize_repair_config_changes_fail_closed(field, value):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["audit"][field] = value
    with pytest.raises(ValueError):
        validate_sdh_experiment_config(changed)


def test_resize_repair_decision_requires_bitwise_strict_fresh():
    normal = {
        "input_valid": True,
        "state_valid": True,
        "pairs": {"reset": _pair(exact=False)},
    }
    strict = {
        "input_valid": True,
        "state_valid": True,
        "pairs": {"fresh": _pair(exact=True)},
    }
    passed = _choose_resize_repair_replay_decision(normal, strict)
    assert passed["label"] == "strict_replay_pass"
    assert passed["normal_reset_bitwise_pass"] is False
    assert passed["strict_fresh_bitwise_pass"] is True

    strict["pairs"]["fresh"] = _pair(exact=False)
    failed = _choose_resize_repair_replay_decision(normal, strict)
    assert failed["label"] == "strict_replay_mismatch"


def test_resize_repair_decision_preserves_operator_error():
    normal = {
        "input_valid": True,
        "state_valid": True,
        "pairs": {"reset": _pair(exact=False)},
    }
    strict = {
        "input_valid": True,
        "state_valid": True,
        "pairs": {},
        "operator_error": {"error": "unsupported"},
    }
    result = _choose_resize_repair_replay_decision(normal, strict)
    assert result["label"] == "new_cuda_nondeterministic_operator"


def test_single_boot_controller_has_cost_and_shutdown_gates():
    controller = (PRE_RUN / "p1_det_resize_fix_controller_v1.sh").read_text(
        encoding="utf-8"
    )
    launcher = (PRE_RUN / "p1_det_resize_fix_tmux_launch_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "HARD_CAP_SECONDS=480" in controller
    assert "trap shutdown_once EXIT" in controller
    assert "mountpoint -q" in controller
    assert "run_deterministic_resize_probe" in controller
    assert "--mode strict" in controller
    assert "run_p1_resize_repair_writeback" in controller
    assert 'if run_rc == 124:' in controller
    assert 'label = "performance_gate_failed"' in controller
    assert 'writeback.get("status") == "failed_invariant"' in controller
    assert "repair_pass" in controller
    assert "HARD_CAP_SECONDS=480" in launcher
    assert "tmux new-session -d" in launcher
