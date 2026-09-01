from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Dict, Mapping

import yaml

from ue_framework.io_utils import atomic_write_json
from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config
from ue_framework.tools.run_tausb_dgcaip_c0_snapshots import (
    _file_sha256,
    _git,
    _gpu_precheck,
    _inside,
    _read_json,
    _require_fresh,
    _shutdown,
    validate_tmp_root_path,
)
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import run_guarded


SPEC_ID = "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"
RUN_MODE = "dataset_risk_scan"
WALL_SECONDS = 65 * 60
MIN_FREE_BYTES = 5 * 1024 ** 3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bound DG-CAIP dataset-level KL/JS G0 risk scan once."
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--required-storage-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", default="")
    parser.add_argument("--shutdown-on-exit", action="store_true")
    return parser.parse_args()


def _validate_binding(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    validate_sdh_experiment_config(config)
    if config["spec"].get("spec_id") != SPEC_ID:
        raise ValueError("G0 risk config spec_id mismatch.")
    if config["dgcaip"].get("run_mode") != RUN_MODE:
        raise ValueError("G0 risk config run_mode mismatch.")
    bindings = config.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("G0 risk config has no C0 snapshot binding.")
    manifest_path = Path(str(bindings.get("c0_snapshot_manifest", "")))
    if _file_sha256(manifest_path) != str(
        bindings.get("c0_snapshot_manifest_sha256", "")
    ).lower():
        raise ValueError("C0 snapshot manifest hash mismatch.")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "passed" or manifest.get("spec_id") != SPEC_ID:
        raise ValueError("C0 snapshot manifest is not passed for this Spec.")
    if manifest.get("execution_commit") != bindings.get("c0_execution_commit"):
        raise ValueError("C0 snapshot execution commit mismatch.")

    snapshots = config["model"]["protection_surrogate_snapshots"]
    expected_ids = ("e1", "e5", "e20")
    if tuple(item["id"] for item in snapshots) != expected_ids:
        raise ValueError("G0 snapshot IDs are not e1/e5/e20.")
    snapshot_hashes = {}
    for item in snapshots:
        snapshot_id = str(item["id"])
        checkpoint = Path(str(item["checkpoint"]))
        actual_hash = _file_sha256(checkpoint)
        if actual_hash != str(item["sha256"]).lower():
            raise ValueError("G0 snapshot file hash mismatch: %s" % snapshot_id)
        manifest_record = manifest["snapshots"].get(snapshot_id, {})
        if (
            manifest_record.get("checkpoint") != str(checkpoint)
            or manifest_record.get("checkpoint_sha256") != actual_hash
        ):
            raise ValueError("G0 snapshot differs from C0 manifest: %s" % snapshot_id)
        snapshot_hashes[snapshot_id] = actual_hash
    source_p1 = Path(str(config["dgcaip"]["source_p1_state"]))
    source_p1_hash = _file_sha256(source_p1)
    if source_p1_hash != str(config["dgcaip"]["source_p1_state_sha256"]).lower():
        raise ValueError("G0 source P1 state hash mismatch.")
    return {
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "c0_snapshot_manifest": str(manifest_path),
        "c0_snapshot_manifest_sha256": _file_sha256(manifest_path),
        "snapshot_sha256": snapshot_hashes,
        "source_p1_state_sha256": source_p1_hash,
    }


def _preflight(args: argparse.Namespace, *, require_gpu: bool) -> Dict[str, Any]:
    repository_root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    storage_root = Path(args.required_storage_root).resolve()
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    python_bin = Path(args.python_bin).resolve()
    if _git(repository_root, "rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("Execution checkout does not match expected commit.")
    if _git(
        repository_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Execution checkout has tracked changes.")
    if not python_bin.is_file():
        raise FileNotFoundError("Python executable is missing: %s" % python_bin)
    if not storage_root.is_dir():
        raise FileNotFoundError("Storage root is missing: %s" % storage_root)
    for output in (control_root, log_root, cache_root, tmp_root):
        if not _inside(storage_root, output):
            raise ValueError("Output escapes required storage root: %s" % output)
    validate_tmp_root_path(tmp_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    binding = _validate_binding(config, config_path)
    artifact_root = Path(str(config["runtime"]["artifact_root"])).resolve()
    if not _inside(storage_root, artifact_root):
        raise ValueError("G0 artifact root escapes required storage root.")
    _require_fresh((artifact_root, control_root, log_root, cache_root, tmp_root))
    free_bytes = shutil.disk_usage(str(storage_root)).free
    if free_bytes < MIN_FREE_BYTES:
        raise ValueError("Data disk has less than 5 GiB free.")
    result = {
        "schema": "tausb.dgcaip-g0-risk-preflight.v1",
        "spec_id": SPEC_ID,
        "status": "passed",
        "execution_commit": args.expected_commit,
        "repository_root": str(repository_root),
        "artifact_root": str(artifact_root),
        "storage_root": str(storage_root),
        "storage_free_bytes": free_bytes,
        "binding": binding,
        "gpu_required": bool(require_gpu),
    }
    if require_gpu:
        result.update(_gpu_precheck())
    return result


def _verify_result(config_path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(str(config["runtime"]["artifact_root"])) / RUN_MODE
    manifest_path = output_root / "dgcaip_risk_manifest.json"
    bank_path = output_root / "dgcaip_risk_bank.json"
    replay_path = output_root / "dgcaip_replay_manifest.json"
    raw_path = output_root / "dgcaip_risk_records.jsonl"
    manifest = _read_json(manifest_path)
    for path in (bank_path, replay_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError("G0 risk artifact is missing: %s" % path)
    coverage = float(manifest.get("coverage", float("nan")))
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("G0 coverage is invalid.")
    expected_snapshots = {
        item["id"]: item["sha256"]
        for item in config["model"]["protection_surrogate_snapshots"]
    }
    if manifest.get("snapshot_sha256") != expected_snapshots:
        raise ValueError("G0 output snapshot binding differs.")
    if manifest.get("risk_bank_file_sha256") != _file_sha256(bank_path):
        raise ValueError("G0 risk bank file hash differs.")
    if manifest.get("replay_manifest_sha256") != _file_sha256(replay_path):
        raise ValueError("G0 replay manifest file hash differs.")
    return {
        "risk_manifest": str(manifest_path),
        "risk_manifest_sha256": _file_sha256(manifest_path),
        "risk_bank": str(bank_path),
        "risk_bank_file_sha256": _file_sha256(bank_path),
        "replay_manifest": str(replay_path),
        "replay_manifest_file_sha256": _file_sha256(replay_path),
        "raw_records_sha256": _file_sha256(raw_path),
        "coverage": coverage,
        "expected_instance_count": int(manifest["expected_instance_count"]),
        "covered_instance_count": int(manifest["covered_instance_count"]),
        "person_cooccurrence_image_count": int(
            manifest["person_cooccurrence_image_count"]
        ),
        "gate_pass": bool(manifest.get("decision", {}).get("pass", False)),
    }


def _run(args: argparse.Namespace) -> int:
    preflight = _preflight(args, require_gpu=not args.preflight_only)
    if args.preflight_only:
        if args.preflight_output:
            output = Path(args.preflight_output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(str(output), preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    repository_root = Path(args.repository_root).resolve()
    project_root = repository_root / "ue_project"
    config_path = Path(args.config).resolve()
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    control_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    status_path = control_root / "controller_status.json"
    status: Dict[str, Any] = {
        "schema": "tausb.dgcaip-g0-risk-controller-status.v1",
        "spec_id": SPEC_ID,
        "status": "running",
        "current_stage": "G0_DATASET_RISK_SCAN",
        "execution_commit": args.expected_commit,
        "hard_cap_seconds": WALL_SECONDS,
        "started_unix": time.time(),
    }
    atomic_write_json(str(status_path), status)
    atomic_write_json(str(control_root / "preflight.json"), preflight)
    os.environ.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TMPDIR": str(tmp_root),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TORCH_HOME": str(cache_root / "torch"),
            "YOLO_CONFIG_DIR": str(cache_root / "yolo"),
        }
    )
    try:
        scan = run_guarded(
            stage="G0_DATASET_RISK_SCAN",
            command=(
                str(Path(args.python_bin).resolve()),
                "-u",
                "-m",
                "ue_framework.tools.run_tausb_sdh",
                "--config",
                str(config_path),
                "--stage",
                "mechanism",
            ),
            cwd=project_root,
            log_path=log_root / "g0_risk.log",
            wall_seconds=WALL_SECONDS,
            first_progress_seconds=10 * 60,
            idle_seconds=WALL_SECONDS,
            require_gpu=True,
        )
        result = _verify_result(config_path)
        result_path = control_root / "g0_risk_result.json"
        atomic_write_json(str(result_path), result)
        status.update(
            {
                "status": "completed",
                "current_stage": "",
                "scan": scan,
                "result": result,
                "result_path": str(result_path),
                "ended_unix": time.time(),
            }
        )
        atomic_write_json(str(status_path), status)
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "ended_unix": time.time(),
            }
        )
        atomic_write_json(str(status_path), status)
        raise


def main() -> int:
    args = _arguments()
    should_shutdown = bool(args.shutdown_on_exit and not args.preflight_only)
    try:
        return _run(args)
    except BaseException as error:
        print(
            "[DGCAIP-G0-Risk] %s: %s" % (type(error).__name__, error),
            file=sys.stderr,
        )
        return int(getattr(error, "exit_code", 1))
    finally:
        if should_shutdown:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
