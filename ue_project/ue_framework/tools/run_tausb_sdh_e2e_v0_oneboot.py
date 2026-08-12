from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from ue_framework.config import load_config
from ue_framework.io_utils import atomic_write_json
from ue_framework.metrics_utils import VOC20_CLASS_NAMES


ONEBOOT_SPEC_ID = "TAUSB-SDH-E2E-V0-ONEBOOT-v2"
PROTOCOL_ID = "TAUSB-SDH-E2E-V0-MAP50-v1"
EXP_ID = "TAUSB-SDH-E2E-V0-S0-E20"
EVIDENCE_SCOPE = "end_to_end_feasibility_not_formal_method"
METHOD = "tausb_sdh"
STEPS = 40
SEED = 0
TARGET_CLASS_ID = 14
EPS_TOLERANCE = 16.0 / 255.0 + 1.0 / 255.0
TRAIN_SCALE = 16551.0 / 200.0
MAX_PAIRED_E20_SECONDS = 8 * 60 * 60
DISK_SAFETY_FACTOR = 1.5
MIN_INITIAL_FREE_BYTES = 10 * 1024 ** 3
MECHANISM_CONFIG_SHA256 = (
    "46f757afa7f0a57944af2bec84cab72549230aa431d41bf99e3ff8a25ab4dc56"
)

MECHANISM_ROOT = Path("/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH")
BINDING_ROOT = Path(
    os.environ.get(
        "TAUSB_SDH_BINDING_ROOT",
        "/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R2",
    )
)
RUN_ROOT_PREFIX = Path(
    os.environ.get(
        "TAUSB_SDH_RUN_ROOT_PREFIX",
        "/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R2",
    )
)
CONTROL_ROOT = Path("/root/tausb-sdh-control/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R2")
LOG_ROOT = Path("/root/tausb-sdh-logs/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R2")
COMPARISON_ROOT = Path(
    "/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-COMPARISON-R2"
)
DATASET_ROOT = Path("/root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready")

RUN_ROOTS = {
    ("smoke", "C0"): Path(str(RUN_ROOT_PREFIX) + "-SMOKE-C0"),
    ("smoke", "M1"): Path(str(RUN_ROOT_PREFIX) + "-SMOKE-M1"),
    ("e20", "C0"): Path(str(RUN_ROOT_PREFIX) + "-E20-C0"),
    ("e20", "M1"): Path(str(RUN_ROOT_PREFIX) + "-E20-M1"),
}

SHARED_METRIC_HASH_KEYS = (
    "clean_val_manifest_sha256",
    "paired_training_protocol_sha256",
    "frozen_sdh_state_sha256",
    "hiding_metrics_sha256",
    "hiding_checkpoint_sha256",
    "hiding_split_sha256",
    "mechanism_metrics_sha256",
    "mechanism_decision_sha256",
    "mechanism_config_sha256",
    "p1_state_sha256",
)


class GuardFailure(RuntimeError):
    def __init__(self, stage: str, reason: str, exit_code: int = 1) -> None:
        super().__init__("%s failed: %s" % (stage, reason))
        self.stage = stage
        self.reason = reason
        self.exit_code = int(exit_code)


class CostGateStop(GuardFailure):
    pass


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved TAUSB-SDH E2E V0 mechanism, paired smoke, "
            "conditional E20, and comparison in one GPU boot."
        )
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-from-binding", action="store_true")
    parser.add_argument("--control-root", default=str(CONTROL_ROOT))
    parser.add_argument("--log-root", default=str(LOG_ROOT))
    parser.add_argument("--comparison-root", default=str(COMPARISON_ROOT))
    return parser.parse_args()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Required JSON is missing: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object: %s" % path)
    return payload


def _directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(str(path)):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def _parse_time(value: object) -> float:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def _stage_seconds(status: Mapping[str, Any], stage: str) -> float:
    stage_state = status.get("stage_state")
    if not isinstance(stage_state, Mapping):
        raise ValueError("Status has no stage_state mapping.")
    record = stage_state.get(stage)
    if not isinstance(record, Mapping) or record.get("status") != "completed":
        raise ValueError("Stage %s is not completed." % stage)
    seconds = _parse_time(record.get("end_time")) - _parse_time(
        record.get("start_time")
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Stage %s has an invalid duration." % stage)
    return seconds


def _artifact_root(run_root: Path, arm_id: str) -> Path:
    return run_root / "artifacts" / METHOD / "steps40" / ("seed0_" + arm_id)


def _metrics_path(pilot_kind: str, arm_id: str) -> Path:
    return _artifact_root(RUN_ROOTS[(pilot_kind, arm_id)], arm_id) / "metrics/metrics.json"


def _status_path(pilot_kind: str, arm_id: str) -> Path:
    return _artifact_root(RUN_ROOTS[(pilot_kind, arm_id)], arm_id) / "status.json"


def _config_path(pilot_kind: str, arm_id: str) -> Path:
    return BINDING_ROOT / ("%s-%s.yaml" % (pilot_kind, arm_id.lower()))


def _strict_ap50(metrics: Mapping[str, Any]) -> Dict[str, float]:
    values = metrics.get("ap50_by_class")
    if not isinstance(values, Mapping) or set(values) != set(VOC20_CLASS_NAMES):
        raise ValueError("Metrics must contain exactly the named VOC20 AP50 values.")
    output = {name: float(values[name]) for name in VOC20_CLASS_NAMES}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in output.values()):
        raise ValueError("VOC20 AP50 contains a non-finite or out-of-range value.")
    return output


def validate_arm(
    metrics: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    pilot_kind: str,
    arm_id: str,
    expected_epochs: int,
    expected_poisoned_count: int,
) -> Dict[str, float]:
    identity = {
        "method": METHOD,
        "steps": STEPS,
        "seed": SEED,
        "protocol_id": PROTOCOL_ID,
        "pilot_kind": pilot_kind,
        "arm_id": arm_id,
        "victim_epochs": expected_epochs,
        "evidence_scope": EVIDENCE_SCOPE,
        "hiding_gate_passed": False,
    }
    for key, expected in identity.items():
        if metrics.get(key) != expected:
            raise ValueError("%s %s does not match %r." % (arm_id, key, expected))
    if metrics.get("run_tag") != arm_id:
        raise ValueError("%s run_tag does not match its arm identity." % arm_id)
    if int(metrics.get("poisoned_count", -1)) != expected_poisoned_count:
        raise ValueError("%s poisoned_count is not %d." % (arm_id, expected_poisoned_count))
    linf = float(metrics.get("actual_linf_max", float("nan")))
    if not math.isfinite(linf) or linf > EPS_TOLERANCE:
        raise ValueError("%s actual_linf_max is invalid." % arm_id)
    if not isinstance(metrics.get("mechanism_gate_passed"), bool):
        raise ValueError("%s mechanism gate provenance is missing." % arm_id)
    for key in SHARED_METRIC_HASH_KEYS:
        if len(str(metrics.get(key, ""))) != 64:
            raise ValueError("%s %s is not a SHA-256 value." % (arm_id, key))
    if pilot_kind == "smoke" and len(
        str(metrics.get("train_selection_manifest_sha256", ""))
    ) != 64:
        raise ValueError("%s smoke selection hash is missing." % arm_id)
    _strict_ap50(metrics)

    completed = set(status.get("completed_stages", []))
    required_stages = {"generate_poisoned_dataset", "train_victim", "evaluate"}
    if not required_stages.issubset(completed):
        raise ValueError("%s status is missing completed stages." % arm_id)
    stage_state = status.get("stage_state")
    if not isinstance(stage_state, Mapping):
        raise ValueError("%s status has no stage_state." % arm_id)
    train_state = stage_state.get("train_victim")
    if not isinstance(train_state, Mapping):
        raise ValueError("%s train state is missing." % arm_id)
    if int(train_state.get("latest_epoch", -1)) < expected_epochs - 1:
        raise ValueError("%s victim did not complete the expected epochs." % arm_id)
    checkpoint = Path(str(train_state.get("best_checkpoint", "")))
    if not checkpoint.is_file():
        raise ValueError("%s victim best checkpoint is missing." % arm_id)
    return {
        stage: _stage_seconds(status, stage)
        for stage in ("generate_poisoned_dataset", "train_victim", "evaluate")
    }


def _normalized_pair_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    experiment = normalized["experiment"]
    platform = normalized["platform"]
    for key in ("arm_id", "poisoning_ratio", "expected_poisoned_count"):
        experiment.pop(key, None)
    platform.pop("run_root", None)
    return normalized


def validate_pair_identity(
    c0_metrics: Mapping[str, Any],
    m1_metrics: Mapping[str, Any],
    c0_config: Mapping[str, Any],
    m1_config: Mapping[str, Any],
) -> None:
    if _normalized_pair_config(c0_config) != _normalized_pair_config(m1_config):
        raise ValueError("C0/M1 configs differ outside the approved arm fields.")
    for key in SHARED_METRIC_HASH_KEYS:
        if c0_metrics.get(key) != m1_metrics.get(key):
            raise ValueError("C0/M1 %s differs." % key)
    if c0_metrics.get("mechanism_gate_passed") != m1_metrics.get(
        "mechanism_gate_passed"
    ):
        raise ValueError("C0/M1 mechanism gate provenance differs.")
    if c0_metrics.get("train_selection_manifest_sha256") != m1_metrics.get(
        "train_selection_manifest_sha256"
    ):
        raise ValueError("C0/M1 train selection differs.")


def build_smoke_review(
    *,
    c0_metrics: Mapping[str, Any],
    m1_metrics: Mapping[str, Any],
    c0_status: Mapping[str, Any],
    m1_status: Mapping[str, Any],
    c0_config: Mapping[str, Any],
    m1_config: Mapping[str, Any],
    c0_artifact_bytes: int,
    m1_artifact_bytes: int,
    disk_free_bytes: int,
) -> Dict[str, Any]:
    c0_times = validate_arm(
        c0_metrics,
        c0_status,
        pilot_kind="smoke",
        arm_id="C0",
        expected_epochs=1,
        expected_poisoned_count=0,
    )
    m1_times = validate_arm(
        m1_metrics,
        m1_status,
        pilot_kind="smoke",
        arm_id="M1",
        expected_epochs=1,
        expected_poisoned_count=40,
    )
    validate_pair_identity(c0_metrics, m1_metrics, c0_config, m1_config)

    estimates: Dict[str, Dict[str, float]] = {}
    for arm_id, values in (("C0", c0_times), ("M1", m1_times)):
        estimate = (
            TRAIN_SCALE * values["generate_poisoned_dataset"]
            + 20.0 * TRAIN_SCALE * values["train_victim"]
            + values["evaluate"]
        )
        estimates[arm_id] = {
            "estimated_seconds": estimate,
            "hard_cap_seconds": float(math.ceil(1.5 * estimate + 600.0)),
        }
    paired_seconds = sum(value["estimated_seconds"] for value in estimates.values())
    projected_bytes = int(
        math.ceil(
            DISK_SAFETY_FACTOR
            * TRAIN_SCALE
            * (int(c0_artifact_bytes) + int(m1_artifact_bytes))
        )
    )
    reasons: List[str] = []
    if paired_seconds > MAX_PAIRED_E20_SECONDS:
        reasons.append("paired_e20_estimate_exceeds_8_gpu_hours")
    if int(disk_free_bytes) < projected_bytes:
        reasons.append("disk_safety_margin_below_1_5x_projection")
    return {
        "schema": "tausb.sdh-e2e-v0-oneboot-smoke-review.v1",
        "spec_id": ONEBOOT_SPEC_ID,
        "exp_id": EXP_ID,
        "dataflow_gate_passed": True,
        "cost_gate_passed": not reasons,
        "decision": "continue_e20" if not reasons else "cost_gate_stop",
        "stop_reasons": reasons,
        "smoke_stage_seconds": {"C0": c0_times, "M1": m1_times},
        "train_scale": TRAIN_SCALE,
        "e20_estimates": estimates,
        "paired_e20_estimated_seconds": paired_seconds,
        "max_paired_e20_seconds": MAX_PAIRED_E20_SECONDS,
        "smoke_artifact_bytes": {
            "C0": int(c0_artifact_bytes),
            "M1": int(m1_artifact_bytes),
        },
        "projected_required_disk_bytes": projected_bytes,
        "disk_free_bytes": int(disk_free_bytes),
        "disk_safety_factor": DISK_SAFETY_FACTOR,
        "claim_boundary": "dataflow and cost gate only; not scientific efficacy",
    }


class ControllerState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "controller_status.json"
        self.payload: Dict[str, Any] = {
            "schema": "tausb.sdh-e2e-v0-oneboot-controller-status.v1",
            "spec_id": ONEBOOT_SPEC_ID,
            "exp_id": EXP_ID,
            "status": "running",
            "current_stage": "",
            "stages": {},
            "started_unix": time.time(),
        }
        root.mkdir(parents=True, exist_ok=False)
        self._write()

    def _write(self) -> None:
        atomic_write_json(str(self.path), self.payload)

    def start(self, stage: str, details: Mapping[str, Any] = None) -> None:
        record: Dict[str, Any] = {"status": "running", "started_unix": time.time()}
        if details:
            record.update(dict(details))
        self.payload["current_stage"] = stage
        self.payload["stages"][stage] = record
        self._write()

    def complete(self, stage: str, details: Mapping[str, Any] = None) -> None:
        record = self.payload["stages"][stage]
        record.update({"status": "completed", "ended_unix": time.time()})
        if details:
            record.update(dict(details))
        self.payload["current_stage"] = ""
        self._write()

    def fail(self, stage: str, error: BaseException) -> None:
        record = self.payload["stages"].setdefault(
            stage, {"started_unix": time.time()}
        )
        record.update(
            {
                "status": "failed",
                "ended_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        self.payload["status"] = (
            "cost_gate_stop" if isinstance(error, CostGateStop) else "failed"
        )
        self.payload["current_stage"] = ""
        self.payload["ended_unix"] = time.time()
        self._write()

    def finish(self) -> None:
        self.payload.update(
            {"status": "completed", "current_stage": "", "ended_unix": time.time()}
        )
        self._write()


def _terminate(process: subprocess.Popen) -> None:
    process.terminate()
    deadline = time.monotonic() + 30.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(1.0)
    if process.poll() is None:
        process.kill()


def run_guarded(
    *,
    stage: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    wall_seconds: int,
    first_progress_seconds: int = 300,
    idle_seconds: int = 1200,
    require_gpu: bool = False,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_growth = started
    first_growth = False
    gpu_process_observed = False
    previous_size = 0
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        reason = ""
        while process.poll() is None:
            time.sleep(5.0)
            now = time.monotonic()
            handle.flush()
            current_size = log_path.stat().st_size
            if current_size > previous_size:
                first_growth = True
                last_growth = now
                previous_size = current_size
            if require_gpu and not gpu_process_observed:
                gpu_probe = subprocess.run(
                    (
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                gpu_process_observed = bool(gpu_probe.stdout.strip())
            if not first_growth and now - started >= first_progress_seconds:
                reason = "no_first_progress"
            elif require_gpu and not gpu_process_observed and now - started >= first_progress_seconds:
                reason = "no_gpu_process"
            elif now - last_growth >= idle_seconds:
                reason = "no_log_progress"
            elif now - started >= wall_seconds:
                reason = "wall_timeout"
            if reason:
                _terminate(process)
                break
        exit_code = process.wait()
    elapsed = time.monotonic() - started
    if reason:
        raise GuardFailure(stage, reason, 124)
    if exit_code != 0:
        raise GuardFailure(stage, "exit_code_%d" % exit_code, exit_code)
    return {
        "command": list(command),
        "log_path": str(log_path),
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "gpu_process_observed": gpu_process_observed,
    }


def _run_output(command: Sequence[str], cwd: Path = None) -> str:
    return subprocess.check_output(
        list(command),
        cwd=str(cwd) if cwd else None,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _precheck(
    repository_root: Path,
    project_root: Path,
    expected_commit: str,
    python_bin: Path,
    *,
    resume_from_binding: bool = False,
    control_root: Path = CONTROL_ROOT,
    log_root: Path = LOG_ROOT,
    comparison_root: Path = COMPARISON_ROOT,
) -> Dict[str, Any]:
    if _run_output(("git", "-C", str(repository_root), "rev-parse", "HEAD")) != expected_commit:
        raise ValueError("Remote checkout HEAD does not match the reviewed commit.")
    if _run_output(
        ("git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("Remote checkout has tracked worktree changes.")
    for executable in (python_bin, Path("/usr/bin/tmux"), Path("/usr/bin/shutdown")):
        if not executable.is_file():
            raise FileNotFoundError("Required executable is missing: %s" % executable)
    if not shutil.which("nvidia-smi"):
        raise FileNotFoundError("nvidia-smi is unavailable.")

    mechanism_config_path = project_root / "ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml"
    base_config_path = project_root / (
        "ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
    )
    if _file_sha256(mechanism_config_path) != MECHANISM_CONFIG_SHA256:
        raise ValueError("Mechanism config hash differs from the reviewed input.")
    mechanism_config = yaml.safe_load(mechanism_config_path.read_text(encoding="utf-8"))
    load_config(str(base_config_path))

    audit_path = repository_root / (
        "research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/remote_input_audit.json"
    )
    audit = _read_json(audit_path)
    if audit.get("pass") is not True or not all(audit.get("checks", {}).values()):
        raise ValueError("No-card remote input audit is not passing.")
    expected_audit = {
        ("dataset", "train_images"): 16551,
        ("dataset", "train_labels"): 16551,
        ("dataset", "train_person_images"): 6095,
        ("dataset", "train_image_path_size_manifest_sha256"): (
            "4954727df8686532a788668fd815092112ac3e3ee1414eba83b616e683708fbd"
        ),
        ("dataset", "train_label_content_manifest_sha256"): (
            "3cd05ad1ab6a546bf2afd5e63cb6c3ff6667064d80af129dd819325625b9d848"
        ),
        ("inputs", "r2_hiding_checkpoint_sha256"): (
            "a765e27a62bb1a1939aaae487ff6e61ec405f457056d2329c1c49f91e02c9f36"
        ),
        ("inputs", "surrogate_checkpoint_sha256"): (
            "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89"
        ),
    }
    for (section, key), expected in expected_audit.items():
        if audit.get(section, {}).get(key) != expected:
            raise ValueError("Remote input audit %s.%s differs." % (section, key))
    for path in (
        DATASET_ROOT / "images/train",
        DATASET_ROOT / "labels/train",
        DATASET_ROOT / "images/val",
        DATASET_ROOT / "labels/val",
    ):
        if not path.is_dir():
            raise FileNotFoundError("Dataset directory is missing: %s" % path)

    if resume_from_binding:
        if not (BINDING_ROOT / "binding_report.json").is_file():
            raise FileNotFoundError("Reviewed binding report is missing.")
        fresh_roots = [control_root, log_root, comparison_root]
    else:
        fresh_roots = [MECHANISM_ROOT, BINDING_ROOT, control_root, log_root, comparison_root]
    fresh_roots.extend(RUN_ROOTS.values())
    existing = [str(path) for path in fresh_roots if path.exists()]
    if existing:
        raise FileExistsError("Fresh one-boot roots already exist: %s" % existing)
    disk_free = shutil.disk_usage("/root").free
    if disk_free < MIN_INITIAL_FREE_BYTES:
        raise ValueError("Initial free disk is below 10 GiB.")
    gpu_rows = _run_output(
        ("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits")
    )
    if not gpu_rows:
        raise ValueError("No GPU is visible.")
    compute_apps = _run_output(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    if compute_apps:
        raise ValueError("Another GPU compute process is already active.")
    return {
        "repository_root": str(repository_root),
        "project_root": str(project_root),
        "commit": expected_commit,
        "mechanism_config_file_sha256": MECHANISM_CONFIG_SHA256,
        "mechanism_config_sha256": _canonical_json_sha256(mechanism_config),
        "disk_free_bytes": disk_free,
        "gpu_rows": gpu_rows.splitlines(),
        "input_audit": str(audit_path),
    }


def _verify_binding() -> Dict[str, Any]:
    report = _read_json(BINDING_ROOT / "binding_report.json")
    selection = _read_json(BINDING_ROOT / "smoke_train_selection.json")
    if report.get("schema") != "tausb.sdh-e2e-v0-binding-report.v1":
        raise ValueError("Binding report schema is invalid.")
    if report.get("protocol_id") != PROTOCOL_ID or len(report.get("configs", [])) != 4:
        raise ValueError("Binding report does not contain the four V0 configs.")
    if selection.get("selected_count") != 200 or selection.get("target_count") != 40:
        raise ValueError("Smoke selection is not 40 person + 160 person-free.")
    if selection.get("person_free_count") != 160:
        raise ValueError("Smoke selection person-free count is invalid.")
    expected_roots = {
        str(RUN_ROOTS[(pilot, arm)])
        for pilot in ("smoke", "e20")
        for arm in ("C0", "M1")
    }
    actual_roots = {str(record.get("run_root")) for record in report["configs"]}
    if actual_roots != expected_roots:
        raise ValueError("Bound run roots differ from the approved roots.")
    for pilot, arm in (("smoke", "C0"), ("smoke", "M1"), ("e20", "C0"), ("e20", "M1")):
        load_config(str(_config_path(pilot, arm)))
    return {
        "binding_report": str(BINDING_ROOT / "binding_report.json"),
        "selection_manifest_sha256": report["selection_manifest_sha256"],
        "config_count": 4,
    }


def _launch_command(
    python_bin: Path, project_root: Path, pilot_kind: str, arm_id: str, device: str
) -> List[str]:
    return [
        str(python_bin),
        "-u",
        str(project_root / "ue_framework/launch_one.py"),
        "--config",
        str(_config_path(pilot_kind, arm_id)),
        "--method",
        METHOD,
        "--steps",
        str(STEPS),
        "--seed",
        str(SEED),
        "--stage",
        "all",
        "--gpu_id",
        str(device),
        "--run_tag",
        arm_id,
    ]


def _run_controller(args: argparse.Namespace) -> int:
    repository_root = Path(args.repository_root).resolve()
    project_root = repository_root / "ue_project"
    python_bin = Path(args.python_bin)
    control_root = Path(args.control_root)
    log_root = Path(args.log_root)
    comparison_root = Path(args.comparison_root)
    try:
        precheck = _precheck(
            repository_root,
            project_root,
            args.expected_commit,
            python_bin,
            resume_from_binding=args.resume_from_binding,
            control_root=control_root,
            log_root=log_root,
            comparison_root=comparison_root,
        )
    except BaseException as error:
        if not control_root.exists():
            control_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(
                str(control_root / "controller_status.json"),
                {
                    "schema": "tausb.sdh-e2e-v0-oneboot-controller-status.v1",
                    "spec_id": ONEBOOT_SPEC_ID,
                    "exp_id": EXP_ID,
                    "status": "failed",
                    "current_stage": "",
                    "stages": {
                        "PRECHECK": {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "ended_unix": time.time(),
                        }
                    },
                    "ended_unix": time.time(),
                },
            )
        raise
    state = ControllerState(control_root)
    log_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(str(control_root / "precheck.json"), precheck)
    state.start("PRECHECK")
    state.complete("PRECHECK", precheck)

    try:
        if args.resume_from_binding:
            stage = "P1_VERIFY_AND_BIND"
            state.start(stage)
            bind_result = {"mode": "reuse_existing_verified_binding"}
            bind_result.update(_verify_binding())
            state.complete(stage, bind_result)
        else:
            stage = "MECHANISM"
            state.start(stage)
            mechanism_result = run_guarded(
                stage=stage,
                command=(
                    str(python_bin),
                    "-u",
                    "-m",
                    "ue_framework.tools.run_tausb_sdh",
                    "--config",
                    "ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml",
                    "--stage",
                    "mechanism",
                ),
                cwd=project_root,
                log_path=log_root / "mechanism.log",
                wall_seconds=1200,
                first_progress_seconds=300,
                idle_seconds=1200,
                require_gpu=True,
            )
            mechanism_status = _read_json(MECHANISM_ROOT / "status_mechanism.json")
            if mechanism_status.get("status") != "completed":
                raise GuardFailure(stage, "mechanism_status_not_completed")
            for path in (
                MECHANISM_ROOT / "mechanism/mechanism_metrics.json",
                MECHANISM_ROOT / "mechanism/p1_state.pt",
                MECHANISM_ROOT / "mechanism/p1_feasibility_sdh_state.pt",
            ):
                if not path.is_file():
                    raise GuardFailure(stage, "missing_%s" % path.name)
            state.complete(stage, mechanism_result)

            stage = "P1_VERIFY_AND_BIND"
            state.start(stage)
            bind_result = run_guarded(
                stage=stage,
                command=(
                    str(python_bin),
                    "-u",
                    "-m",
                    "ue_framework.tools.bind_tausb_sdh_e2e_v0",
                    "--mechanism-root",
                    str(MECHANISM_ROOT),
                    "--mechanism-config",
                    "ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml",
                    "--base-config",
                    "ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml",
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--output-dir",
                    str(BINDING_ROOT),
                    "--run-root-prefix",
                    str(RUN_ROOT_PREFIX),
                ),
                cwd=project_root,
                log_path=log_root / "binding.log",
                wall_seconds=600,
                first_progress_seconds=300,
                idle_seconds=600,
            )
            bind_result.update(_verify_binding())
            state.complete(stage, bind_result)

        smoke_started = time.monotonic()
        for arm_id in ("C0", "M1"):
            stage = "SMOKE_" + arm_id
            state.start(stage)
            remaining = int(7200 - (time.monotonic() - smoke_started))
            if remaining <= 0:
                raise GuardFailure(stage, "paired_smoke_wall_timeout", 124)
            result = run_guarded(
                stage=stage,
                command=_launch_command(python_bin, project_root, "smoke", arm_id, args.device),
                cwd=project_root,
                log_path=log_root / ("smoke_%s.log" % arm_id.lower()),
                wall_seconds=remaining,
                first_progress_seconds=300,
                idle_seconds=1200,
                require_gpu=True,
            )
            validate_arm(
                _read_json(_metrics_path("smoke", arm_id)),
                _read_json(_status_path("smoke", arm_id)),
                pilot_kind="smoke",
                arm_id=arm_id,
                expected_epochs=1,
                expected_poisoned_count=0 if arm_id == "C0" else 40,
            )
            state.complete(stage, result)

        stage = "SMOKE_DATAFLOW_REVIEW"
        state.start(stage)
        c0_metrics = _read_json(_metrics_path("smoke", "C0"))
        m1_metrics = _read_json(_metrics_path("smoke", "M1"))
        c0_status = _read_json(_status_path("smoke", "C0"))
        m1_status = _read_json(_status_path("smoke", "M1"))
        c0_config = yaml.safe_load(_config_path("smoke", "C0").read_text(encoding="utf-8"))
        m1_config = yaml.safe_load(_config_path("smoke", "M1").read_text(encoding="utf-8"))
        smoke_review = build_smoke_review(
            c0_metrics=c0_metrics,
            m1_metrics=m1_metrics,
            c0_status=c0_status,
            m1_status=m1_status,
            c0_config=c0_config,
            m1_config=m1_config,
            c0_artifact_bytes=_directory_size(RUN_ROOTS[("smoke", "C0")]),
            m1_artifact_bytes=_directory_size(RUN_ROOTS[("smoke", "M1")]),
            disk_free_bytes=shutil.disk_usage("/root").free,
        )
        smoke_review_path = control_root / "smoke_review.json"
        atomic_write_json(str(smoke_review_path), smoke_review)
        if smoke_review["decision"] != "continue_e20":
            raise CostGateStop(stage, ",".join(smoke_review["stop_reasons"]), 20)
        state.complete(stage, {"review": str(smoke_review_path), "decision": "continue_e20"})

        for arm_id in ("C0", "M1"):
            stage = "E20_" + arm_id
            state.start(stage)
            cap = int(smoke_review["e20_estimates"][arm_id]["hard_cap_seconds"])
            result = run_guarded(
                stage=stage,
                command=_launch_command(python_bin, project_root, "e20", arm_id, args.device),
                cwd=project_root,
                log_path=log_root / ("e20_%s.log" % arm_id.lower()),
                wall_seconds=cap,
                first_progress_seconds=300,
                idle_seconds=1200,
                require_gpu=True,
            )
            metrics = _read_json(_metrics_path("e20", arm_id))
            status = _read_json(_status_path("e20", arm_id))
            validate_arm(
                metrics,
                status,
                pilot_kind="e20",
                arm_id=arm_id,
                expected_epochs=20,
                expected_poisoned_count=0 if arm_id == "C0" else 6095,
            )
            state.complete(stage, result)

        stage = "COMPARE_AND_FINALIZE"
        state.start(stage)
        comparison_result = run_guarded(
            stage=stage,
            command=(
                str(python_bin),
                "-u",
                "-m",
                "ue_framework.tools.compare_tausb_sdh_e2e_v0",
                "--c0-metrics",
                str(_metrics_path("e20", "C0")),
                "--m1-metrics",
                str(_metrics_path("e20", "M1")),
                "--output-dir",
                str(comparison_root),
            ),
            cwd=project_root,
            log_path=log_root / "comparison.log",
            wall_seconds=300,
            first_progress_seconds=120,
            idle_seconds=300,
        )
        comparison = _read_json(comparison_root / "comparison.json")
        comparison_result["pilot_decision"] = comparison.get("pilot_decision")
        state.complete(stage, comparison_result)
        state.finish()
        return 0
    except BaseException as error:
        state.fail(locals().get("stage", "UNKNOWN"), error)
        if isinstance(error, GuardFailure):
            return error.exit_code
        raise


def main() -> int:
    args = _arguments()
    try:
        return _run_controller(args)
    except BaseException as error:
        print("[OneBoot][Failure] %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return int(getattr(error, "exit_code", 1))


if __name__ == "__main__":
    sys.exit(main())
