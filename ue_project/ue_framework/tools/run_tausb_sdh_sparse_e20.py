from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import cv2
import yaml

from ue_framework.io_utils import atomic_write_json
from ue_framework.metrics_utils import VOC20_CLASS_NAMES
from ue_framework.tools.bind_tausb_sdh_e2e_v0 import _validate_mechanism_binding
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import (
    GuardFailure,
    run_guarded,
    validate_arm,
)


SPEC_ID = "TAUSB-SDH-E2E-V0-SPARSE-E20-v3"
EXP_ID = "TAUSB-SDH-E2E-V0-S0-E20-SPARSE"
RUN_ID = "SPARSE-E20-S0-R1"
OVERALL_WALL_SECONDS = 2 * 60 * 60
MATERIALIZE_WALL_SECONDS = 40 * 60
ARM_TRAIN_EVAL_WALL_SECONDS = 40 * 60
IDLE_SECONDS = 10 * 60
DISK_RESERVE_BYTES = 3 * 1024 ** 3
DISK_CONTINGENCY_BYTES = 1 * 1024 ** 3
CHECKPOINT_BUDGET_BYTES = 512 * 1024 ** 2
FINAL_EVIDENCE_BUDGET_BYTES = 256 * 1024 ** 2
MIN_SYSTEM_DISK_FREE_BYTES = 4 * 1024 ** 3
MAX_SYSTEM_DISK_STAGE_GROWTH_BYTES = 1 * 1024 ** 3
E200_SURROGATE_CHECKPOINT_SHA256 = (
    "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89"
)


@dataclass(frozen=True)
class SparseExperimentContract:
    victim_epochs: int
    spec_id: str
    exp_id: str
    run_id: str
    overall_wall_seconds: int
    materialize_wall_seconds: int
    arm_train_eval_wall_seconds: int
    disk_reserve_bytes: int
    expected_paired_wall_minutes: Sequence[int]


def _experiment_contract(victim_epochs: int) -> SparseExperimentContract:
    if victim_epochs == 20:
        return SparseExperimentContract(
            victim_epochs=20,
            spec_id=SPEC_ID,
            exp_id=EXP_ID,
            run_id=RUN_ID,
            overall_wall_seconds=OVERALL_WALL_SECONDS,
            materialize_wall_seconds=MATERIALIZE_WALL_SECONDS,
            arm_train_eval_wall_seconds=ARM_TRAIN_EVAL_WALL_SECONDS,
            disk_reserve_bytes=DISK_RESERVE_BYTES,
            expected_paired_wall_minutes=(45, 90),
        )
    if victim_epochs == 200:
        return SparseExperimentContract(
            victim_epochs=200,
            spec_id="TAUSB-SDH-E2E-V0-SPARSE-E200-v1",
            exp_id="TAUSB-SDH-E2E-V0-S0-E200-SPARSE",
            run_id="SPARSE-E200-S0-R1",
            overall_wall_seconds=9 * 60 * 60,
            materialize_wall_seconds=20 * 60,
            arm_train_eval_wall_seconds=int(3.5 * 60 * 60),
            disk_reserve_bytes=8 * 1024 ** 3,
            expected_paired_wall_minutes=(300, 540),
        )
    raise ValueError("victim_epochs must be 20 or 200.")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an approved sparse-materialized paired SDH E20/E200 experiment."
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument(
        "--required-storage-root",
        required=True,
        help="Mounted AutoDL data-disk root that must contain the checkout, dataset, and all writable outputs.",
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--device", default="0")
    parser.add_argument("--mechanism-root", required=True)
    parser.add_argument("--mechanism-config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--binding-root", required=True)
    parser.add_argument("--run-root-prefix", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument(
        "--cache-root",
        default="",
        help="Required data-disk cache root for E200; optional for legacy E20.",
    )
    parser.add_argument(
        "--tmp-root",
        default="",
        help="Required data-disk temp root for E200; optional for legacy E20.",
    )
    parser.add_argument(
        "--victim-epochs",
        type=int,
        choices=(20, 200),
        default=20,
        help="Select the frozen E20 compatibility contract or approved E200 contract.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Required JSON is missing: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object: %s" % path)
    return payload


def c0_ap50_is_interpretable(metrics: Mapping[str, Any]) -> bool:
    """Return whether the clean control has at least one non-zero class AP50."""
    values = metrics.get("ap50_by_class")
    if not isinstance(values, Mapping):
        raise ValueError("C0 metrics have no named AP50 mapping.")
    return any(float(value) > 0.0 for value in values.values())


def c0_full_horizon_sanity(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    values = metrics.get("ap50_by_class")
    if not isinstance(values, Mapping) or set(values) != set(VOC20_CLASS_NAMES):
        raise ValueError("C0 metrics must contain exactly 20 named VOC AP50 values.")
    ap50 = {name: float(values[name]) for name in VOC20_CLASS_NAMES}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in ap50.values()):
        raise ValueError("C0 AP50 contains a non-finite or out-of-range value.")
    non_target = [ap50[name] for name in VOC20_CLASS_NAMES if name != "person"]
    non_target_macro = sum(non_target) / len(non_target)
    checks = {
        "person_ap50_ge_0_60": ap50["person"] >= 0.60,
        "non_target_macro_ap50_ge_0_50": non_target_macro >= 0.50,
    }
    return {
        "schema": "tausb.sdh-e200-c0-sanity.v1",
        "person_ap50": ap50["person"],
        "non_target_macro_ap50": non_target_macro,
        "checks": checks,
        "pass": all(checks.values()),
    }


def validate_fresh_init_pair(
    c0: Mapping[str, Any],
    m1: Mapping[str, Any],
    expected_surrogate_sha256: str,
) -> Dict[str, Any]:
    for arm_id, record in (("C0", c0), ("M1", m1)):
        validate_fresh_init_record(record, arm_id, expected_surrogate_sha256)
    if c0["victim_init_tensor_sha256"] != m1["victim_init_tensor_sha256"]:
        raise ValueError("C0/M1 victim init tensor hashes differ.")
    return {
        "schema": "tausb.sdh-e200-fresh-init-pair.v1",
        "victim_init_tensor_sha256": c0["victim_init_tensor_sha256"],
        "surrogate_checkpoint_sha256": expected_surrogate_sha256,
        "matched": True,
    }


def validate_fresh_init_record(
    record: Mapping[str, Any], arm_id: str, expected_surrogate_sha256: str
) -> None:
    if record.get("run_tag") != arm_id:
        raise ValueError("%s fresh init run_tag is mismatched." % arm_id)
    if record.get("resume_enabled") is not False:
        raise ValueError("%s victim is not a fresh no-resume initialization." % arm_id)
    if record.get("surrogate_checkpoint_used_for_victim_init") is not False:
        raise ValueError("%s victim incorrectly used the surrogate checkpoint." % arm_id)
    if record.get("surrogate_checkpoint_sha256") != expected_surrogate_sha256:
        raise ValueError("%s surrogate checkpoint hash is mismatched." % arm_id)
    if len(str(record.get("victim_init_tensor_sha256", ""))) != 64:
        raise ValueError("%s victim init tensor hash is missing." % arm_id)


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repository_root), text=True
    ).strip()


def _git_worktree_status(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repository_root),
        text=True,
    ).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError("No existing ancestor for disk probe: %s" % path)
        candidate = candidate.parent
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_storage_roots(required_root: Path, paths: Mapping[str, Path]) -> Dict[str, str]:
    """Fail closed unless every growing experiment path is on the mounted data disk."""
    root = required_root.resolve()
    if not root.is_dir():
        raise GuardFailure("PRECHECK", "required_storage_root_missing")
    if not os.path.ismount(str(root)):
        raise GuardFailure("PRECHECK", "required_storage_root_not_mountpoint")
    root_device = root.stat().st_dev
    resolved = {}
    for name, path in paths.items():
        candidate = path.resolve()
        if not _is_within(candidate, root):
            raise GuardFailure("PRECHECK", "%s_outside_required_storage_root" % name)
        if _existing_ancestor(candidate).stat().st_dev != root_device:
            raise GuardFailure("PRECHECK", "%s_not_on_required_storage_device" % name)
        resolved[name] = str(candidate)
    return {"required_storage_root": str(root), **resolved}


def validate_cache_environment(cache_root: Path, tmp_root: Path) -> Dict[str, str]:
    expected = {
        "TMPDIR": tmp_root.resolve(),
        "XDG_CACHE_HOME": (cache_root / "xdg").resolve(),
        "TORCH_HOME": (cache_root / "torch").resolve(),
        "YOLO_CONFIG_DIR": (cache_root / "yolo").resolve(),
    }
    resolved = {}
    for name, path in expected.items():
        actual = os.environ.get(name, "")
        if not actual or Path(actual).resolve() != path:
            raise GuardFailure("PRECHECK", "%s_not_bound_to_data_disk" % name)
        resolved[name] = str(path)
    return resolved


class SystemDiskMonitor:
    def __init__(self, root: Path = Path("/")) -> None:
        self.root = root
        self.previous_free_bytes = shutil.disk_usage(str(root)).free
        if self.previous_free_bytes < MIN_SYSTEM_DISK_FREE_BYTES:
            raise GuardFailure("PRECHECK", "system_disk_free_below_4GiB", 20)
        self.initial_free_bytes = self.previous_free_bytes

    def after_stage(self, stage: str) -> Dict[str, Any]:
        current_free = shutil.disk_usage(str(self.root)).free
        growth = max(0, self.previous_free_bytes - current_free)
        record = {
            "root": str(self.root),
            "initial_free_bytes": self.initial_free_bytes,
            "previous_free_bytes": self.previous_free_bytes,
            "current_free_bytes": current_free,
            "stage_growth_bytes": growth,
            "stage_growth_limit_bytes": MAX_SYSTEM_DISK_STAGE_GROWTH_BYTES,
            "pass": (
                current_free >= MIN_SYSTEM_DISK_FREE_BYTES
                and growth <= MAX_SYSTEM_DISK_STAGE_GROWTH_BYTES
            ),
        }
        self.previous_free_bytes = current_free
        if current_free < MIN_SYSTEM_DISK_FREE_BYTES:
            raise GuardFailure(stage, "system_disk_free_below_4GiB", 20)
        if growth > MAX_SYSTEM_DISK_STAGE_GROWTH_BYTES:
            raise GuardFailure(stage, "system_disk_stage_growth_exceeds_1GiB", 20)
        return record


def _config_path(binding_root: Path, arm_id: str, victim_epochs: int = 20) -> Path:
    return binding_root / ("e%d-%s.yaml" % (victim_epochs, arm_id.lower()))


def _run_root(run_root_prefix: str, arm_id: str, victim_epochs: int = 20) -> Path:
    return Path(run_root_prefix.rstrip("-") + "-E%d-" % victim_epochs + arm_id)


def _poisoned_root(run_root: Path) -> Path:
    return run_root / "poisoned_datasets/tausb_sdh/steps40/seed0"


def _artifact_root(run_root: Path, arm_id: str) -> Path:
    return run_root / ("artifacts/tausb_sdh/steps40/seed0_%s" % arm_id)


def _metrics_path(run_root: Path, arm_id: str) -> Path:
    return _artifact_root(run_root, arm_id) / "metrics/metrics.json"


def _status_path(run_root: Path, arm_id: str) -> Path:
    return _artifact_root(run_root, arm_id) / "status.json"


def _fresh_init_path(run_root: Path, arm_id: str) -> Path:
    return _artifact_root(run_root, arm_id) / "logs/fresh_init.json"


def _launch_command(
    python_bin: Path,
    project_root: Path,
    config_path: Path,
    arm_id: str,
    stage: str,
    device: str,
) -> Sequence[str]:
    return (
        str(python_bin),
        "-u",
        str(project_root / "ue_framework/launch_one.py"),
        "--config",
        str(config_path),
        "--method",
        "tausb_sdh",
        "--steps",
        "40",
        "--seed",
        "0",
        "--stage",
        stage,
        "--gpu_id",
        str(device),
        "--run_tag",
        arm_id,
    )


def build_sparse_disk_projection(
    dataset_root: Path,
    sample_count: int = 64,
    disk_reserve_bytes: int = DISK_RESERVE_BYTES,
) -> Dict[str, int]:
    image_dir = dataset_root / "images/train"
    label_dir = dataset_root / "labels/train"
    target_images = []
    target_count = 0
    target_label_bytes = 0
    for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name):
        if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            raise FileNotFoundError("VOC label is missing: %s" % label_path)
        has_target = any(
            line.strip() and int(float(line.split()[0])) == 14
            for line in label_path.read_text(encoding="utf-8").splitlines()
        )
        if has_target:
            target_count += 1
            target_label_bytes += label_path.stat().st_size
            if len(target_images) < sample_count:
                target_images.append(image_path)
    if target_count != 6095:
        raise ValueError(
            "Sparse disk projection requires exactly 6095 target images; got %d."
            % target_count
        )
    if len(target_images) != sample_count:
        raise ValueError("Sparse disk projection requires %d target images." % sample_count)
    encoded_bytes = []
    for image_path in target_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to read disk-projection sample: %s" % image_path)
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        if not ok:
            raise RuntimeError("Failed to encode disk-projection sample: %s" % image_path)
        encoded_bytes.append(int(encoded.nbytes))
    mean_png_bytes = int(sum(encoded_bytes) / len(encoded_bytes))
    poisoned_png_bytes = mean_png_bytes * 6095
    projected_new_bytes = (
        poisoned_png_bytes
        + target_label_bytes
        + CHECKPOINT_BUDGET_BYTES
        + FINAL_EVIDENCE_BUDGET_BYTES
        + DISK_CONTINGENCY_BYTES
    )
    return {
        "sample_count": sample_count,
        "target_count": target_count,
        "mean_sample_png_bytes": mean_png_bytes,
        "projected_poisoned_png_bytes": poisoned_png_bytes,
        "target_label_bytes": target_label_bytes,
        "checkpoint_budget_bytes": CHECKPOINT_BUDGET_BYTES,
        "final_evidence_budget_bytes": FINAL_EVIDENCE_BUDGET_BYTES,
        "contingency_bytes": DISK_CONTINGENCY_BYTES,
        "projected_new_bytes": projected_new_bytes,
        "required_free_bytes_with_reserve": projected_new_bytes + disk_reserve_bytes,
    }


def validate_sparse_pair(c0_report: Mapping[str, Any], m1_report: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("ordered_stems_sha256", "label_content_manifest_sha256", "total_count", "target_count"):
        if c0_report.get(key) != m1_report.get(key):
            raise ValueError("Sparse C0/M1 identity differs for %s." % key)
    expected = {
        "total_count": 16551,
        "target_count": 6095,
        "clean_png_roundtrip_probe_count": 64,
    }
    for key, value in expected.items():
        if c0_report.get(key) != value or m1_report.get(key) != value:
            raise ValueError("Sparse C0/M1 %s does not match the frozen protocol." % key)
    arm_counts = {
        "C0": (c0_report, 0, 0, 16551),
        "M1": (m1_report, 6095, 6095, 10456),
    }
    for arm_id, (report, poisoned, png, jpeg) in arm_counts.items():
        if int(report.get("poisoned_count", -1)) != poisoned:
            raise ValueError("Sparse %s poisoned count mismatch." % arm_id)
        if int(report.get("poisoned_png_count", -1)) != png:
            raise ValueError("Sparse %s PNG count mismatch." % arm_id)
        if int(report.get("original_jpeg_count", -1)) != jpeg:
            raise ValueError("Sparse %s original JPEG count mismatch." % arm_id)
    return {
        "schema": "tausb.sdh-sparse-paired-list-gate.v1",
        "ordered_stems_sha256": c0_report["ordered_stems_sha256"],
        "label_content_manifest_sha256": c0_report["label_content_manifest_sha256"],
        "C0": {"poisoned_png_count": 0, "original_jpeg_count": 16551},
        "M1": {"poisoned_png_count": 6095, "original_jpeg_count": 10456},
        "pass": True,
    }


class SparseControllerState:
    def __init__(
        self,
        root: Path,
        contract: SparseExperimentContract,
        system_disk_monitor: SystemDiskMonitor = None,
    ) -> None:
        self.root = root
        self.path = root / "controller_status.json"
        self.system_disk_monitor = system_disk_monitor
        self.payload = {
            "schema": "tausb.sdh-sparse-e%d-controller-status.v1"
            % contract.victim_epochs,
            "spec_id": contract.spec_id,
            "exp_id": contract.exp_id,
            "run_id": contract.run_id,
            "status": "running",
            "current_stage": "",
            "stages": {},
            "started_unix": time.time(),
        }
        root.mkdir(parents=True, exist_ok=False)
        self._write()

    def _write(self) -> None:
        atomic_write_json(str(self.path), self.payload)

    def start(self, stage: str) -> None:
        self.payload["current_stage"] = stage
        self.payload["stages"][stage] = {
            "status": "running",
            "started_unix": time.time(),
        }
        self._write()

    def complete(self, stage: str, details: Mapping[str, Any] = None) -> None:
        record = {"status": "completed", "ended_unix": time.time(), **dict(details or {})}
        if self.system_disk_monitor is not None:
            record["system_disk"] = self.system_disk_monitor.after_stage(stage)
        self.payload["stages"][stage].update(record)
        self.payload["current_stage"] = ""
        self._write()

    def fail(self, stage: str, error: BaseException) -> None:
        self.payload["stages"].setdefault(stage, {"started_unix": time.time()})
        self.payload["stages"][stage].update(
            {
                "status": "failed",
                "ended_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        self.payload.update(
            {"status": "failed", "current_stage": "", "ended_unix": time.time()}
        )
        self._write()

    def finish(self) -> None:
        self.payload.update(
            {"status": "completed", "current_stage": "", "ended_unix": time.time()}
        )
        self._write()


def _remaining(
    started: float, requested: int, contract: SparseExperimentContract
) -> int:
    remaining = contract.overall_wall_seconds - int(time.monotonic() - started)
    if remaining <= 0:
        raise GuardFailure(
            "OVERALL", "paired_sparse_e%d_wall_timeout" % contract.victim_epochs, 124
        )
    return min(requested, remaining)


def write_terminal_evidence_manifest(
    *,
    control_root: Path,
    binding_root: Path,
    log_root: Path,
    comparison_root: Path,
    run_roots: Mapping[str, Path],
) -> Dict[str, Any]:
    candidates = [
        control_root / "controller_status.json",
        control_root / "precheck.json",
        control_root / "mixed_list_gate.json",
        control_root / "c0_sanity_gate.json",
        control_root / "fresh_init_pair_gate.json",
        binding_root / "binding_report.json",
        comparison_root / "comparison.json",
        comparison_root / "per_class_ap50.csv",
    ]
    candidates.extend(sorted(log_root.glob("*.log")))
    for arm_id, run_root in run_roots.items():
        poisoned_root = _poisoned_root(run_root)
        artifact_root = _artifact_root(run_root, arm_id)
        candidates.extend(
            [
                poisoned_root / "status.json",
                poisoned_root / "manifest.csv",
                poisoned_root / "sparse_materialization.json",
                artifact_root / "status.json",
                artifact_root / "metrics/metrics.json",
                artifact_root / "logs/fresh_init.json",
                artifact_root / "logs/train_stage_summary.json",
                artifact_root / "logs/train_stage_live_summary.json",
            ]
        )
    records = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        records.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    comparison_path = comparison_root / "comparison.json"
    scientific_outcome = "not_evaluable"
    operational_outcome = "failed_or_timeout"
    if comparison_path.is_file():
        scientific_outcome = str(
            _read_json(comparison_path).get("pilot_decision", "unknown")
        )
        operational_outcome = "completed"
    else:
        controller_path = control_root / "controller_status.json"
        controller = _read_json(controller_path) if controller_path.is_file() else {}
        stages = controller.get("stages", {})
        stage_names = set(stages) if isinstance(stages, Mapping) else set()
        if any(
            name.startswith(("C0_TRAIN", "C0_EVALUATE", "C0_VALIDATE", "M1_"))
            for name in stage_names
        ):
            scientific_outcome = "full_horizon_failure_incomplete_or_integrity"
    manifest = {
        "schema": "tausb.sdh-terminal-evidence-manifest.v1",
        "scientific_outcome": scientific_outcome,
        "operational_outcome": operational_outcome,
        "retention_policy": "retain_and_report_all_terminal_outcomes",
        "file_count": len(records),
        "files": records,
    }
    atomic_write_json(str(control_root / "terminal_evidence_manifest.json"), manifest)
    return manifest


def _run_controller(args: argparse.Namespace) -> int:
    started = time.monotonic()
    os.environ.pop("TAUSB_EXPECTED_VICTIM_INIT_TENSOR_SHA256", None)
    contract = _experiment_contract(int(args.victim_epochs))
    repository_root = Path(args.repository_root).resolve()
    required_storage_root = Path(args.required_storage_root).resolve()
    project_root = repository_root / "ue_project"
    python_bin = Path(args.python_bin).resolve()
    mechanism_root = Path(args.mechanism_root).resolve()
    mechanism_config = Path(args.mechanism_config).resolve()
    base_config = Path(args.base_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    binding_root = Path(args.binding_root).resolve()
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    comparison_root = Path(args.comparison_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else None
    tmp_root = Path(args.tmp_root).resolve() if args.tmp_root else None
    run_roots = {
        arm: _run_root(args.run_root_prefix, arm, contract.victim_epochs)
        for arm in ("C0", "M1")
    }
    growing_paths = {
            "repository_root": repository_root,
            "dataset_root": dataset_root,
            "binding_root": binding_root,
            "control_root": control_root,
            "log_root": log_root,
            "comparison_root": comparison_root,
            "run_root_C0": run_roots["C0"],
            "run_root_M1": run_roots["M1"],
    }
    if cache_root is not None and tmp_root is not None:
        growing_paths.update(
            {
                "cache_root": cache_root,
                "tmp_root": tmp_root,
                "xdg_cache_root": cache_root / "xdg",
                "torch_cache_root": cache_root / "torch",
                "yolo_config_root": cache_root / "yolo",
            }
        )
    elif contract.victim_epochs == 200:
        raise GuardFailure("PRECHECK", "E200_requires_cache_root_and_tmp_root")
    storage_paths = validate_storage_roots(required_storage_root, growing_paths)
    fresh_paths = [binding_root, control_root, log_root, comparison_root, *run_roots.values()]
    if _git_head(repository_root) != args.expected_commit:
        raise GuardFailure("PRECHECK", "execution_commit_mismatch")
    if _git_worktree_status(repository_root):
        raise GuardFailure("PRECHECK", "execution_checkout_not_clean")
    if any(path.exists() for path in fresh_paths):
        raise GuardFailure("PRECHECK", "fresh_output_path_already_exists")
    cache_environment = (
        validate_cache_environment(cache_root, tmp_root)
        if cache_root is not None and tmp_root is not None
        else {}
    )
    system_disk_monitor = (
        SystemDiskMonitor() if contract.victim_epochs == 200 else None
    )
    _validate_mechanism_binding(mechanism_root, mechanism_config)
    base_payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    surrogate_checkpoint = Path(base_payload["surrogate"]["ckpt"]).resolve()
    if not surrogate_checkpoint.is_file():
        raise GuardFailure("PRECHECK", "surrogate_checkpoint_missing")
    surrogate_checkpoint_sha256 = _file_sha256(surrogate_checkpoint)
    if (
        contract.victim_epochs == 200
        and surrogate_checkpoint_sha256 != E200_SURROGATE_CHECKPOINT_SHA256
    ):
        raise GuardFailure("PRECHECK", "surrogate_checkpoint_hash_mismatch")
    disk = build_sparse_disk_projection(
        dataset_root, disk_reserve_bytes=contract.disk_reserve_bytes
    )
    disk_probe_path = _existing_ancestor(Path(args.run_root_prefix).resolve().parent)
    disk["disk_probe_path"] = str(disk_probe_path)
    disk["free_bytes"] = shutil.disk_usage(str(disk_probe_path)).free
    if disk["free_bytes"] < disk["required_free_bytes_with_reserve"]:
        raise GuardFailure(
            "PRECHECK",
            "insufficient_disk_for_sparse_e%d" % contract.victim_epochs,
            20,
        )

    state = SparseControllerState(control_root, contract, system_disk_monitor)
    log_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        str(control_root / "precheck.json"),
        {
            "spec_id": contract.spec_id,
            "exp_id": contract.exp_id,
            "run_id": contract.run_id,
            "execution_commit": args.expected_commit,
            "mechanism_root": str(mechanism_root),
            "surrogate_checkpoint": str(surrogate_checkpoint),
            "surrogate_checkpoint_sha256": surrogate_checkpoint_sha256,
            "storage_paths": storage_paths,
            "cache_environment": cache_environment,
            "system_disk_initial_free_bytes": (
                system_disk_monitor.initial_free_bytes
                if system_disk_monitor is not None
                else None
            ),
            "disk_projection": disk,
            "historical_full_voc_e20_seconds_per_arm": 15.96 * 60,
            "expected_paired_wall_minutes": list(contract.expected_paired_wall_minutes),
            "overall_wall_cap_seconds": contract.overall_wall_seconds,
            "victim_epochs": contract.victim_epochs,
        },
    )
    try:
        stage = "BIND_SPARSE_E%d" % contract.victim_epochs
        state.start(stage)
        bind_result = run_guarded(
            stage=stage,
            command=(
                str(python_bin), "-u", "-m", "ue_framework.tools.bind_tausb_sdh_e2e_v0",
                "--mechanism-root", str(mechanism_root),
                "--mechanism-config", str(mechanism_config),
                "--base-config", str(base_config),
                "--dataset-root", str(dataset_root),
                "--output-dir", str(binding_root),
                "--run-root-prefix", args.run_root_prefix,
                "--full-voc-only",
                "--victim-epochs", str(contract.victim_epochs),
            ),
            cwd=project_root,
            log_path=log_root / "binding.log",
            wall_seconds=_remaining(started, 300, contract),
            idle_seconds=300,
        )
        configs = {}
        fresh_init_records = {}
        for arm_id in ("C0", "M1"):
            config_path = _config_path(
                binding_root, arm_id, contract.victim_epochs
            )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config["data"].get("materialization_layout") != "sparse_mixed_list_v1":
                raise GuardFailure(stage, "%s_sparse_layout_missing" % arm_id)
            if Path(config["platform"]["run_root"]) != run_roots[arm_id]:
                raise GuardFailure(stage, "%s_run_root_mismatch" % arm_id)
            configs[arm_id] = config_path
        report = _read_json(binding_root / "binding_report.json")
        expected_binding_scope = "e%d_only" % contract.victim_epochs
        if report.get("binding_scope") != expected_binding_scope or len(report.get("configs", [])) != 2:
            raise GuardFailure(stage, "binding_scope_is_not_%s" % expected_binding_scope)
        if any(binding_root.glob("smoke-*.yaml")) or (
            binding_root / "smoke_train_selection.json"
        ).exists():
            raise GuardFailure(stage, "binding_created_forbidden_smoke_inputs")
        state.complete(stage, bind_result)

        for arm_id in ("C0", "M1"):
            stage = "SPARSE_MATERIALIZE_" + arm_id
            state.start(stage)
            result = run_guarded(
                stage=stage,
                command=_launch_command(
                    python_bin, project_root, configs[arm_id], arm_id,
                    "generate_poisoned_dataset", args.device,
                ),
                cwd=project_root,
                log_path=log_root / ("materialize_%s.log" % arm_id.lower()),
                wall_seconds=_remaining(
                    started,
                    contract.materialize_wall_seconds if arm_id == "M1" else 600,
                    contract,
                ),
                idle_seconds=IDLE_SECONDS,
                require_gpu=arm_id == "M1",
            )
            state.complete(stage, result)

        stage = "MIXED_LIST_GATE"
        state.start(stage)
        reports = {
            arm: _read_json(_poisoned_root(run_roots[arm]) / "sparse_materialization.json")
            for arm in ("C0", "M1")
        }
        paired_gate = validate_sparse_pair(reports["C0"], reports["M1"])
        atomic_write_json(str(control_root / "mixed_list_gate.json"), paired_gate)
        state.complete(stage, paired_gate)

        for arm_id in ("C0", "M1"):
            arm_started = time.monotonic()
            for train_stage in ("train_victim", "evaluate"):
                stage = "%s_%s" % (arm_id, train_stage.upper())
                state.start(stage)
                remaining_arm = contract.arm_train_eval_wall_seconds - int(
                    time.monotonic() - arm_started
                )
                if remaining_arm <= 0:
                    raise GuardFailure(stage, "%s_train_eval_wall_timeout" % arm_id, 124)
                result = run_guarded(
                    stage=stage,
                    command=_launch_command(
                        python_bin, project_root, configs[arm_id], arm_id,
                        train_stage, args.device,
                    ),
                    cwd=project_root,
                    log_path=log_root / ("%s_%s.log" % (arm_id.lower(), train_stage)),
                    wall_seconds=_remaining(started, remaining_arm, contract),
                    idle_seconds=IDLE_SECONDS,
                    require_gpu=True,
                )
                state.complete(stage, result)
            stage = "%s_VALIDATE" % arm_id
            state.start(stage)
            arm_metrics = _read_json(_metrics_path(run_roots[arm_id], arm_id))
            arm_timings = validate_arm(
                arm_metrics,
                _read_json(_status_path(run_roots[arm_id], arm_id)),
                pilot_kind="e%d" % contract.victim_epochs,
                arm_id=arm_id,
                expected_epochs=contract.victim_epochs,
                expected_poisoned_count=0 if arm_id == "C0" else 6095,
            )
            fresh_init_records[arm_id] = _read_json(
                _fresh_init_path(run_roots[arm_id], arm_id)
            )
            validate_fresh_init_record(
                fresh_init_records[arm_id], arm_id, surrogate_checkpoint_sha256
            )
            stage_details = {
                "stage_seconds": arm_timings,
                "fresh_init": fresh_init_records[arm_id],
            }
            if arm_id == "C0":
                if contract.victim_epochs == 200:
                    c0_sanity = c0_full_horizon_sanity(arm_metrics)
                    atomic_write_json(
                        str(control_root / "c0_sanity_gate.json"), c0_sanity
                    )
                    stage_details["c0_sanity"] = c0_sanity
                    if not c0_sanity["pass"]:
                        state.complete(stage, stage_details)
                        raise GuardFailure(
                            stage, "C0_full_horizon_sanity_failed_M1_forbidden", 21
                        )
                elif not c0_ap50_is_interpretable(arm_metrics):
                    raise GuardFailure(
                        stage,
                        "C0_AP50_all_zero_M1_training_forbidden",
                        21,
                    )
            state.complete(stage, stage_details)
            if arm_id == "C0":
                os.environ["TAUSB_EXPECTED_VICTIM_INIT_TENSOR_SHA256"] = str(
                    fresh_init_records["C0"]["victim_init_tensor_sha256"]
                )
            if arm_id == "M1":
                fresh_init_gate = validate_fresh_init_pair(
                    fresh_init_records["C0"],
                    fresh_init_records["M1"],
                    surrogate_checkpoint_sha256,
                )
                atomic_write_json(
                    str(control_root / "fresh_init_pair_gate.json"), fresh_init_gate
                )

        stage = "COMPARE"
        state.start(stage)
        compare_result = run_guarded(
            stage=stage,
            command=(
                str(python_bin), "-u", "-m", "ue_framework.tools.compare_tausb_sdh_e2e_v0",
                "--c0-metrics", str(_metrics_path(run_roots["C0"], "C0")),
                "--m1-metrics", str(_metrics_path(run_roots["M1"], "M1")),
                "--output-dir", str(comparison_root),
            ),
            cwd=project_root,
            log_path=log_root / "comparison.log",
            wall_seconds=_remaining(started, 300, contract),
            idle_seconds=300,
        )
        comparison = _read_json(comparison_root / "comparison.json")
        compare_result["pilot_decision"] = comparison.get("pilot_decision")
        compare_result["final_disk_bytes"] = {
            arm: _directory_size(run_roots[arm]) for arm in ("C0", "M1")
        }
        state.complete(stage, compare_result)
        state.finish()
        write_terminal_evidence_manifest(
            control_root=control_root,
            binding_root=binding_root,
            log_root=log_root,
            comparison_root=comparison_root,
            run_roots=run_roots,
        )
        return 0
    except BaseException as error:
        state.fail(locals().get("stage", "UNKNOWN"), error)
        write_terminal_evidence_manifest(
            control_root=control_root,
            binding_root=binding_root,
            log_root=log_root,
            comparison_root=comparison_root,
            run_roots=run_roots,
        )
        if isinstance(error, GuardFailure):
            return error.exit_code
        raise


def main() -> int:
    args = _arguments()
    try:
        return _run_controller(args)
    except BaseException as error:
        print("[SparseController][Failure] %s: %s" % (type(error).__name__, error), file=sys.stderr)
        control_root = Path(args.control_root).resolve()
        required_root = Path(args.required_storage_root).resolve()
        if (
            not control_root.exists()
            and required_root.is_dir()
            and os.path.ismount(str(required_root))
            and _is_within(control_root, required_root)
        ):
            control_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(
                str(control_root / "controller_status.json"),
                {
                    "schema": "tausb.sdh-sparse-controller-bootstrap-failure.v1",
                    "status": "failed",
                    "current_stage": "",
                    "terminal_stage": "PRECHECK",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "ended_unix": time.time(),
                },
            )
            write_terminal_evidence_manifest(
                control_root=control_root,
                binding_root=Path(args.binding_root).resolve(),
                log_root=Path(args.log_root).resolve(),
                comparison_root=Path(args.comparison_root).resolve(),
                run_roots={
                    arm: _run_root(args.run_root_prefix, arm, int(args.victim_epochs))
                    for arm in ("C0", "M1")
                },
            )
        return int(getattr(error, "exit_code", 1))


if __name__ == "__main__":
    sys.exit(main())
