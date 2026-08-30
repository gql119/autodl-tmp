import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ue_framework.tools.run_tausb_sdh_dgcaip_p4_oneboot import (
    _fresh_output_roots,
    _validate_mechanism_output_binding,
    _write_failure_evidence,
)
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import GuardFailure


def _write_config(path: Path, artifact_root: Path) -> None:
    path.write_text(
        yaml.safe_dump({"runtime": {"artifact_root": str(artifact_root)}}),
        encoding="utf-8",
    )


def test_mechanism_output_binding_accepts_exact_root(tmp_path: Path) -> None:
    mechanism_root = tmp_path / "r3"
    config_path = tmp_path / "mechanism.yaml"
    _write_config(config_path, mechanism_root)
    _validate_mechanism_output_binding(config_path, mechanism_root.resolve())


def test_mechanism_output_binding_rejects_stale_root(tmp_path: Path) -> None:
    mechanism_root = tmp_path / "r3"
    config_path = tmp_path / "mechanism.yaml"
    _write_config(config_path, tmp_path / "r1")
    with pytest.raises(GuardFailure, match="mechanism_artifact_root_mismatch"):
        _validate_mechanism_output_binding(config_path, mechanism_root.resolve())


def test_mechanism_output_binding_rejects_missing_root(tmp_path: Path) -> None:
    config_path = tmp_path / "mechanism.yaml"
    config_path.write_text(yaml.safe_dump({"runtime": {}}), encoding="utf-8")
    with pytest.raises(GuardFailure, match="mechanism_artifact_root_missing"):
        _validate_mechanism_output_binding(config_path, (tmp_path / "r3").resolve())


def test_fresh_output_roots_cover_every_downstream_r3_root(tmp_path: Path) -> None:
    prefix = tmp_path / "r3-VICTIM"
    args = SimpleNamespace(
        binding_root=str(tmp_path / "r3-BINDING"),
        sparse_control_root=str(tmp_path / "r3-SPARSE-CONTROL"),
        comparison_root=str(tmp_path / "r3-COMPARISON"),
        run_root_prefix=str(prefix),
    )
    roots = set(
        _fresh_output_roots(
            args,
            (tmp_path / "r3").resolve(),
            (tmp_path / "r3-CONTROL").resolve(),
            (tmp_path / "r3-LOGS").resolve(),
        )
    )
    assert roots == {
        (tmp_path / "r3").resolve(),
        (tmp_path / "r3-CONTROL").resolve(),
        (tmp_path / "r3-LOGS").resolve(),
        (tmp_path / "r3-BINDING").resolve(),
        (tmp_path / "r3-SPARSE-CONTROL").resolve(),
        (tmp_path / "r3-COMPARISON").resolve(),
        (tmp_path / "r3-VICTIM-E20-C0").resolve(),
        (tmp_path / "r3-VICTIM-E20-M1").resolve(),
    }


def test_failure_evidence_sets_controller_status_terminal(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    control_root.mkdir()
    status_path = control_root / "controller_status.json"
    status_path.write_text(
        json.dumps({"status": "running", "current_stage": "MECHANISM"}),
        encoding="utf-8",
    )
    error = GuardFailure("MECHANISM", "test_failure")
    _write_failure_evidence(control_root, error)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    failure = json.loads(
        (control_root / "controller_failure.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["current_stage"] == ""
    assert status["error_type"] == "GuardFailure"
    assert failure["status"] == "failed"
