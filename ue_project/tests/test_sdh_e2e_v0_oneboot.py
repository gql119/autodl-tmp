from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ue_framework.metrics_utils import VOC20_CLASS_NAMES
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import (
    MECHANISM_CONFIG_SHA256,
    BINDING_ROOT,
    COMPARISON_ROOT,
    CONTROL_ROOT,
    LOG_ROOT,
    MECHANISM_ROOT,
    RUN_ROOTS,
    MAX_PAIRED_E20_SECONDS,
    GuardFailure,
    SHARED_METRIC_HASH_KEYS,
    _canonical_json_sha256,
    _file_sha256,
    _launch_command,
    _run_controller,
    build_smoke_review,
    run_guarded,
    validate_arm,
    validate_pair_identity,
)


ROOT = Path(__file__).parents[1]
WRAPPER = ROOT.parent / (
    "research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/"
    "pre_run/oneboot_controller.sh"
)
MECHANISM_CONFIG = ROOT / "ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml"


def test_frozen_mechanism_config_uses_raw_file_hash_not_canonical_json_hash() -> None:
    import yaml

    parsed = yaml.safe_load(MECHANISM_CONFIG.read_text(encoding="utf-8"))
    assert _file_sha256(MECHANISM_CONFIG) == MECHANISM_CONFIG_SHA256
    assert (
        hashlib.sha256(MECHANISM_CONFIG.read_bytes()).hexdigest()
        == MECHANISM_CONFIG_SHA256
    )
    assert _canonical_json_sha256(parsed) != MECHANISM_CONFIG_SHA256


def test_r2_uses_fresh_roots_and_never_reuses_failed_r1() -> None:
    roots = [MECHANISM_ROOT, BINDING_ROOT, CONTROL_ROOT, LOG_ROOT, COMPARISON_ROOT]
    roots.extend(RUN_ROOTS.values())
    rendered = [str(path) for path in roots]
    assert len(rendered) == len(set(rendered))
    assert all("R2" in path for path in rendered)
    assert all(
        "ONEBOOT-R1" not in path and "BINDING-R1" not in path for path in rendered
    )


def _metrics(arm_id: str, *, pilot_kind: str = "smoke", epochs: int = 1) -> dict:
    payload = {
        "method": "tausb_sdh",
        "steps": 40,
        "seed": 0,
        "run_tag": arm_id,
        "protocol_id": "TAUSB-SDH-E2E-V0-MAP50-v1",
        "pilot_kind": pilot_kind,
        "arm_id": arm_id,
        "victim_epochs": epochs,
        "evidence_scope": "end_to_end_feasibility_not_formal_method",
        "hiding_gate_passed": False,
        "mechanism_gate_passed": False,
        "poisoned_count": 0 if arm_id == "C0" else (40 if epochs == 1 else 6095),
        "actual_linf_max": 0.0 if arm_id == "C0" else 16.0 / 255.0,
        "train_selection_manifest_sha256": "a" * 64 if pilot_kind == "smoke" else "",
        "ap50_by_class": {name: 0.5 for name in VOC20_CLASS_NAMES},
    }
    payload.update({key: "b" * 64 for key in SHARED_METRIC_HASH_KEYS})
    return payload


def _status(tmp_path: Path, arm_id: str, *, epochs: int = 1, seconds: float = 0.1) -> dict:
    checkpoint = tmp_path / (arm_id + "-best.pt")
    checkpoint.write_bytes(b"checkpoint")
    stages = {}
    for index, stage in enumerate(
        ("generate_poisoned_dataset", "train_victim", "evaluate")
    ):
        start = index * 10.0
        record = {
            "status": "completed",
            "start_time": "2026-08-11T00:00:%06.3fZ" % start,
            "end_time": "2026-08-11T00:00:%06.3fZ" % (start + seconds),
        }
        if stage == "train_victim":
            record.update(
                {"latest_epoch": epochs - 1, "best_checkpoint": str(checkpoint)}
            )
        stages[stage] = record
    return {
        "completed_stages": list(stages),
        "stage_state": stages,
    }


def _config(arm_id: str, *, pilot_kind: str = "smoke") -> dict:
    return {
        "experiment": {
            "pilot_kind": pilot_kind,
            "arm_id": arm_id,
            "poisoning_ratio": 0.0 if arm_id == "C0" else 1.0,
            "expected_poisoned_count": 0 if arm_id == "C0" else 40,
            "target_class_id": 14,
        },
        "platform": {"run_root": "/run/" + arm_id, "resume": False},
        "data": {"train_selection_manifest_sha256": "a" * 64},
        "victim": {"epochs": 1, "optimizer": "SGD"},
        "methods": {"tausb_sdh": {"support_type": "bbox"}},
    }


def test_smoke_review_passes_dataflow_and_returns_dynamic_caps(tmp_path: Path) -> None:
    review = build_smoke_review(
        c0_metrics=_metrics("C0"),
        m1_metrics=_metrics("M1"),
        c0_status=_status(tmp_path, "C0"),
        m1_status=_status(tmp_path, "M1"),
        c0_config=_config("C0"),
        m1_config=_config("M1"),
        c0_artifact_bytes=1024,
        m1_artifact_bytes=2048,
        disk_free_bytes=10 * 1024 ** 3,
    )
    assert review["dataflow_gate_passed"] is True
    assert review["cost_gate_passed"] is True
    assert review["decision"] == "continue_e20"
    assert review["paired_e20_estimated_seconds"] < MAX_PAIRED_E20_SECONDS
    assert review["e20_estimates"]["C0"]["hard_cap_seconds"] > 600


def test_smoke_review_stops_on_time_or_disk_cost_without_scientific_gate(
    tmp_path: Path,
) -> None:
    slow = build_smoke_review(
        c0_metrics=_metrics("C0"),
        m1_metrics=_metrics("M1"),
        c0_status=_status(tmp_path, "slow-C0", seconds=20.0),
        m1_status=_status(tmp_path, "slow-M1", seconds=20.0),
        c0_config=_config("C0"),
        m1_config=_config("M1"),
        c0_artifact_bytes=1,
        m1_artifact_bytes=1,
        disk_free_bytes=10 * 1024 ** 3,
    )
    assert slow["decision"] == "cost_gate_stop"
    assert "paired_e20_estimate_exceeds_8_gpu_hours" in slow["stop_reasons"]

    disk = build_smoke_review(
        c0_metrics=_metrics("C0"),
        m1_metrics=_metrics("M1"),
        c0_status=_status(tmp_path, "disk-C0"),
        m1_status=_status(tmp_path, "disk-M1"),
        c0_config=_config("C0"),
        m1_config=_config("M1"),
        c0_artifact_bytes=10 * 1024 ** 3,
        m1_artifact_bytes=10 * 1024 ** 3,
        disk_free_bytes=1024,
    )
    assert disk["decision"] == "cost_gate_stop"
    assert "disk_safety_margin_below_1_5x_projection" in disk["stop_reasons"]


def test_smoke_review_fails_closed_on_metrics_or_pair_drift(tmp_path: Path) -> None:
    broken = _metrics("M1")
    broken["ap50_by_class"].pop("dog")
    with pytest.raises(ValueError, match="VOC20"):
        build_smoke_review(
            c0_metrics=_metrics("C0"),
            m1_metrics=broken,
            c0_status=_status(tmp_path, "bad-C0"),
            m1_status=_status(tmp_path, "bad-M1"),
            c0_config=_config("C0"),
            m1_config=_config("M1"),
            c0_artifact_bytes=1,
            m1_artifact_bytes=1,
            disk_free_bytes=10 * 1024 ** 3,
        )

    m1_config = _config("M1")
    m1_config["victim"]["optimizer"] = "Adam"
    with pytest.raises(ValueError, match="outside the approved arm fields"):
        validate_pair_identity(
            _metrics("C0"), _metrics("M1"), _config("C0"), m1_config
        )


def test_e20_arm_requires_completed_epochs_and_exact_count(tmp_path: Path) -> None:
    metrics = _metrics("M1", pilot_kind="e20", epochs=20)
    status = _status(tmp_path, "e20-M1", epochs=20)
    validate_arm(
        metrics,
        status,
        pilot_kind="e20",
        arm_id="M1",
        expected_epochs=20,
        expected_poisoned_count=6095,
    )
    incomplete = copy.deepcopy(status)
    incomplete["stage_state"]["train_victim"]["latest_epoch"] = 5
    with pytest.raises(ValueError, match="expected epochs"):
        validate_arm(
            metrics,
            incomplete,
            pilot_kind="e20",
            arm_id="M1",
            expected_epochs=20,
            expected_poisoned_count=6095,
        )


def test_launch_command_is_fresh_all_stage_and_wrapper_has_one_shutdown_trap() -> None:
    command = _launch_command(
        Path("/python"), Path("/checkout/ue_project"), "e20", "M1", "0"
    )
    assert command[command.index("--stage") + 1] == "all"
    assert command[command.index("--run_tag") + 1] == "M1"
    assert "--force_resume" not in command
    assert "--poisoned_root_override" not in command

    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "trap shutdown_once EXIT" in wrapper
    assert "trap - EXIT INT TERM" in wrapper
    assert '"${SHUTDOWN_BIN}" -h now' in wrapper
    assert "--expected-commit" in wrapper
    assert "--stage all" not in wrapper


@pytest.mark.parametrize("python_exit", [0, 7])
def test_wrapper_shuts_down_on_success_and_failure(
    tmp_path: Path, python_exit: int
) -> None:
    bash = shutil.which("bash")
    if bash is None and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if bash is None:
        pytest.skip("bash is unavailable")

    repository = tmp_path / "checkout"
    (repository / "ue_project").mkdir(parents=True)
    fake_python = tmp_path / "fake-python.sh"
    fake_shutdown = tmp_path / "fake-shutdown.sh"
    marker = tmp_path / "shutdown-called.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\nexit \"${FAKE_PYTHON_EXIT}\"\n", encoding="utf-8"
    )
    fake_shutdown.write_text(
        "#!/usr/bin/env bash\nprintf called > \"${SHUTDOWN_MARKER}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_shutdown.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "REPOSITORY_ROOT": repository.as_posix(),
            "EXPECTED_COMMIT": "a" * 40,
            "PYTHON_BIN": fake_python.as_posix(),
            "SHUTDOWN_BIN": fake_shutdown.as_posix(),
            "SHUTDOWN_MARKER": marker.as_posix(),
            "FAKE_PYTHON_EXIT": str(python_exit),
        }
    )
    completed = subprocess.run(
        (bash, str(WRAPPER)),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == python_exit
    assert marker.read_text(encoding="utf-8") == "called"


def test_guarded_command_stops_at_wall_timeout(tmp_path: Path) -> None:
    with pytest.raises(GuardFailure, match="wall_timeout"):
        run_guarded(
            stage="TIMEOUT_TEST",
            command=(
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(60)",
            ),
            cwd=tmp_path,
            log_path=tmp_path / "timeout.log",
            wall_seconds=1,
            first_progress_seconds=30,
            idle_seconds=30,
        )


def test_each_smoke_arm_is_validated_before_the_next_arm_starts() -> None:
    source = inspect.getsource(_run_controller)
    smoke_loop = source.index('for arm_id in ("C0", "M1"):')
    review_stage = source.index('stage = "SMOKE_DATAFLOW_REVIEW"', smoke_loop)
    body = source[smoke_loop:review_stage]
    assert body.index("validate_arm(") < body.index("state.complete(stage, result)")
