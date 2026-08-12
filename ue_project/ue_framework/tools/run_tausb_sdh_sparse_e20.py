from __future__ import annotations

import argparse
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the approved sparse-materialized paired SDH E20 experiment."
    )
    parser.add_argument("--repository-root", required=True)
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


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repository_root), text=True
    ).strip()


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


def _config_path(binding_root: Path, arm_id: str) -> Path:
    return binding_root / ("e20-%s.yaml" % arm_id.lower())


def _run_root(run_root_prefix: str, arm_id: str) -> Path:
    return Path(run_root_prefix.rstrip("-") + "-E20-" + arm_id)


def _poisoned_root(run_root: Path) -> Path:
    return run_root / "poisoned_datasets/tausb_sdh/steps40/seed0"


def _artifact_root(run_root: Path, arm_id: str) -> Path:
    return run_root / ("artifacts/tausb_sdh/steps40/seed0_%s" % arm_id)


def _metrics_path(run_root: Path, arm_id: str) -> Path:
    return _artifact_root(run_root, arm_id) / "metrics/metrics.json"


def _status_path(run_root: Path, arm_id: str) -> Path:
    return _artifact_root(run_root, arm_id) / "status.json"


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


def build_sparse_disk_projection(dataset_root: Path, sample_count: int = 64) -> Dict[str, int]:
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
        "required_free_bytes_with_reserve": projected_new_bytes + DISK_RESERVE_BYTES,
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
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "controller_status.json"
        self.payload = {
            "schema": "tausb.sdh-sparse-e20-controller-status.v1",
            "spec_id": SPEC_ID,
            "exp_id": EXP_ID,
            "run_id": RUN_ID,
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
        self.payload["stages"][stage].update(
            {"status": "completed", "ended_unix": time.time(), **dict(details or {})}
        )
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


def _remaining(started: float, requested: int) -> int:
    remaining = OVERALL_WALL_SECONDS - int(time.monotonic() - started)
    if remaining <= 0:
        raise GuardFailure("OVERALL", "paired_sparse_e20_wall_timeout", 124)
    return min(requested, remaining)


def _run_controller(args: argparse.Namespace) -> int:
    started = time.monotonic()
    repository_root = Path(args.repository_root).resolve()
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
    run_roots = {
        arm: _run_root(args.run_root_prefix, arm) for arm in ("C0", "M1")
    }
    fresh_paths = [binding_root, control_root, log_root, comparison_root, *run_roots.values()]
    if _git_head(repository_root) != args.expected_commit:
        raise GuardFailure("PRECHECK", "execution_commit_mismatch")
    if any(path.exists() for path in fresh_paths):
        raise GuardFailure("PRECHECK", "fresh_output_path_already_exists")
    _validate_mechanism_binding(mechanism_root, mechanism_config)
    disk = build_sparse_disk_projection(dataset_root)
    disk_probe_path = _existing_ancestor(Path(args.run_root_prefix).resolve().parent)
    disk["disk_probe_path"] = str(disk_probe_path)
    disk["free_bytes"] = shutil.disk_usage(str(disk_probe_path)).free
    if disk["free_bytes"] < disk["required_free_bytes_with_reserve"]:
        raise GuardFailure("PRECHECK", "insufficient_disk_for_sparse_e20", 20)

    state = SparseControllerState(control_root)
    log_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        str(control_root / "precheck.json"),
        {
            "spec_id": SPEC_ID,
            "exp_id": EXP_ID,
            "run_id": RUN_ID,
            "execution_commit": args.expected_commit,
            "mechanism_root": str(mechanism_root),
            "disk_projection": disk,
            "historical_full_voc_e20_seconds_per_arm": 15.96 * 60,
            "expected_paired_wall_minutes": [45, 90],
            "overall_wall_cap_seconds": OVERALL_WALL_SECONDS,
        },
    )
    try:
        stage = "BIND_SPARSE_E20"
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
                "--e20-only",
            ),
            cwd=project_root,
            log_path=log_root / "binding.log",
            wall_seconds=_remaining(started, 300),
            idle_seconds=300,
        )
        configs = {}
        for arm_id in ("C0", "M1"):
            config_path = _config_path(binding_root, arm_id)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config["data"].get("materialization_layout") != "sparse_mixed_list_v1":
                raise GuardFailure(stage, "%s_sparse_layout_missing" % arm_id)
            if Path(config["platform"]["run_root"]) != run_roots[arm_id]:
                raise GuardFailure(stage, "%s_run_root_mismatch" % arm_id)
            configs[arm_id] = config_path
        report = _read_json(binding_root / "binding_report.json")
        if report.get("binding_scope") != "e20_only" or len(report.get("configs", [])) != 2:
            raise GuardFailure(stage, "binding_scope_is_not_e20_only")
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
                    started, MATERIALIZE_WALL_SECONDS if arm_id == "M1" else 600
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
                remaining_arm = ARM_TRAIN_EVAL_WALL_SECONDS - int(
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
                    wall_seconds=_remaining(started, remaining_arm),
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
                pilot_kind="e20",
                arm_id=arm_id,
                expected_epochs=20,
                expected_poisoned_count=0 if arm_id == "C0" else 6095,
            )
            if arm_id == "C0" and not c0_ap50_is_interpretable(arm_metrics):
                raise GuardFailure(
                    stage,
                    "C0_AP50_all_zero_M1_training_forbidden",
                    21,
                )
            state.complete(stage, {"stage_seconds": arm_timings})

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
            wall_seconds=_remaining(started, 300),
            idle_seconds=300,
        )
        comparison = _read_json(comparison_root / "comparison.json")
        compare_result["pilot_decision"] = comparison.get("pilot_decision")
        compare_result["final_disk_bytes"] = {
            arm: _directory_size(run_roots[arm]) for arm in ("C0", "M1")
        }
        state.complete(stage, compare_result)
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
        print("[SparseE20][Failure] %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return int(getattr(error, "exit_code", 1))


if __name__ == "__main__":
    sys.exit(main())
