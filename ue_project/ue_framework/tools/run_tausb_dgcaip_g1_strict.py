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
from ue_framework.methods.dgcaip_dataset_risk import load_risk_bank
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
V2_SPEC_ID = "TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2"
SUPPORTED_SPEC_IDS = {SPEC_ID, V2_SPEC_ID}
RUN_MODE = "strict_mechanism"
WALL_SECONDS = 20 * 60
MIN_FREE_BYTES = 5 * 1024 ** 3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bound DG-CAIP G1 strict mechanism once."
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
    current_spec_id = str(config["spec"].get("spec_id", ""))
    if current_spec_id not in SUPPORTED_SPEC_IDS:
        raise ValueError("G1 strict config spec_id mismatch.")
    if config["dgcaip"].get("run_mode") != RUN_MODE:
        raise ValueError("G1 strict config run_mode mismatch.")
    bindings = config.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("G1 strict config has no G0 binding.")
    manifest_path = Path(str(bindings.get("g0_risk_manifest", "")))
    manifest_hash = _file_sha256(manifest_path)
    if manifest_hash != str(bindings.get("g0_risk_manifest_sha256", "")).lower():
        raise ValueError("G0 risk manifest hash mismatch.")
    manifest = _read_json(manifest_path)
    if manifest.get("spec_id") != SPEC_ID:
        raise ValueError("G0 risk manifest Spec mismatch.")
    coverage = float(manifest.get("coverage", float("nan")))
    if (
        not bool(manifest.get("decision", {}).get("pass", False))
        or not math.isfinite(coverage)
        or coverage < float(config["dataset_ranking"]["minimum_coverage"])
    ):
        raise ValueError("G0 risk manifest did not pass coverage.")
    if str(bindings.get("g0_execution_commit", "")) != (
        "cc0f9b42e265100a835985bfc4ab3e95411470dd"
    ):
        raise ValueError("G0 execution commit mismatch.")
    controller_path = Path(str(bindings.get("g0_controller_status", "")))
    controller_hash = _file_sha256(controller_path)
    if controller_hash != str(
        bindings.get("g0_controller_status_sha256", "")
    ).lower():
        raise ValueError("G0 controller-status hash mismatch.")
    controller = _read_json(controller_path)
    controller_result = controller.get("result", {})
    if (
        controller.get("status") != "completed"
        or controller.get("execution_commit")
        != bindings.get("g0_execution_commit")
        or controller_result.get("risk_manifest_sha256") != manifest_hash
        or not bool(controller_result.get("gate_pass", False))
    ):
        raise ValueError("G0 controller terminal binding did not pass.")

    snapshot_hashes = {}
    for item in config["model"]["protection_surrogate_snapshots"]:
        snapshot_id = str(item["id"])
        actual_hash = _file_sha256(Path(str(item["checkpoint"])))
        if actual_hash != str(item["sha256"]).lower():
            raise ValueError("G1 snapshot file hash mismatch: %s" % snapshot_id)
        snapshot_hashes[snapshot_id] = actual_hash
    if manifest.get("snapshot_sha256") != snapshot_hashes:
        raise ValueError("G1 protection snapshots differ from G0.")
    source_p1_path = Path(str(config["dgcaip"]["source_p1_state"]))
    source_p1_hash = _file_sha256(source_p1_path)
    if source_p1_hash != str(config["dgcaip"]["source_p1_state_sha256"]).lower():
        raise ValueError("G1 source P1 file hash mismatch.")
    if manifest.get("source_carrier_state_sha256") != source_p1_hash:
        raise ValueError("G1 source P1 state differs from G0.")

    ranking = config["dataset_ranking"]
    bank_path = Path(str(ranking["risk_bank"]))
    bank_file_hash = _file_sha256(bank_path)
    if bank_file_hash != str(ranking["risk_bank_file_sha256"]).lower():
        raise ValueError("G1 risk-bank file hash mismatch.")
    if manifest.get("risk_bank_file_sha256") != bank_file_hash:
        raise ValueError("G1 risk bank differs from G0 manifest.")
    bank = load_risk_bank(
        bank_path,
        expected_spec_id=SPEC_ID,
        expected_sha256=str(ranking["risk_bank_canonical_sha256"]),
    )
    if manifest.get("risk_bank_canonical_sha256") != bank.canonical_sha256:
        raise ValueError("G1 risk-bank canonical hash differs from G0.")

    replay_path = Path(str(ranking["replay_manifest"]))
    replay_file_hash = _file_sha256(replay_path)
    if replay_file_hash != str(ranking["replay_manifest_file_sha256"]).lower():
        raise ValueError("G1 replay-manifest file hash mismatch.")
    if manifest.get("replay_manifest_sha256") != replay_file_hash:
        raise ValueError("G1 replay manifest differs from G0 manifest.")
    replay = _read_json(replay_path)
    expected_slots = int(config["mechanism"]["optimization_steps"]) * int(
        config["mechanism"]["batch_size"]
    )
    if (
        replay.get("schema") != "tausb.dgcaip-dataset-replay.v1"
        or replay.get("spec_id") != SPEC_ID
        or replay.get("risk_bank_canonical_sha256") != bank.canonical_sha256
        or len(replay.get("image_ids", [])) != expected_slots
    ):
        raise ValueError("G1 replay manifest is not bound to the risk bank.")
    return {
        "spec_id": current_spec_id,
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "g0_risk_manifest": str(manifest_path),
        "g0_risk_manifest_sha256": manifest_hash,
        "g0_controller_status": str(controller_path),
        "g0_controller_status_sha256": controller_hash,
        "risk_bank_file_sha256": bank_file_hash,
        "risk_bank_canonical_sha256": bank.canonical_sha256,
        "replay_manifest_file_sha256": replay_file_hash,
        "replay_slots": expected_slots,
        "coverage": coverage,
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
    if _git(repository_root, "status", "--porcelain", "--untracked-files=no"):
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
        raise ValueError("G1 artifact root escapes required storage root.")
    _require_fresh((artifact_root, control_root, log_root, cache_root, tmp_root))
    free_bytes = shutil.disk_usage(str(storage_root)).free
    if free_bytes < MIN_FREE_BYTES:
        raise ValueError("Data disk has less than 5 GiB free.")
    result = {
        "schema": (
            "tausb.dgcaip-g1-strict-preflight.v2"
            if binding["spec_id"] == V2_SPEC_ID
            else "tausb.dgcaip-g1-strict-preflight.v1"
        ),
        "spec_id": binding["spec_id"],
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
    metrics_path = output_root / "mechanism_metrics.json"
    trace_path = output_root / "backtracking_trace.json"
    state_path = output_root / "p5_dataset_strict_state.pt"
    metrics = _read_json(metrics_path)
    for path in (trace_path, state_path):
        if not path.is_file():
            raise FileNotFoundError("G1 strict artifact is missing: %s" % path)
    ranking = config["dataset_ranking"]
    expected_snapshots = {
        str(item["id"]): str(item["sha256"]).lower()
        for item in config["model"]["protection_surrogate_snapshots"]
    }
    current_spec_id = str(config["spec"].get("spec_id", ""))
    route_mode = str(config.get("strict_route", {}).get("mode", "repair_budget_v1"))
    expected_schema = (
        "tausb.dgcaip-dataset-strict-mechanism.v2"
        if route_mode == "nonworsening_target_progress_v2"
        else "tausb.dgcaip-dataset-strict-mechanism.v1"
    )
    if metrics.get("schema") != expected_schema:
        raise ValueError("G1 metrics schema mismatch.")
    if metrics.get("spec_id") != current_spec_id:
        raise ValueError("G1 metrics Spec mismatch.")
    if current_spec_id == V2_SPEC_ID and metrics.get("strict_route_mode") != route_mode:
        raise ValueError("G1 v2 output route mode differs.")
    if metrics.get("source_p1_state_sha256") != str(
        config["dgcaip"]["source_p1_state_sha256"]
    ).lower():
        raise ValueError("G1 output P1 binding differs.")
    if metrics.get("risk_bank_canonical_sha256") != str(
        ranking["risk_bank_canonical_sha256"]
    ).lower():
        raise ValueError("G1 output risk-bank canonical hash differs.")
    if metrics.get("risk_bank_file_sha256") != str(
        ranking["risk_bank_file_sha256"]
    ).lower():
        raise ValueError("G1 output risk-bank file hash differs.")
    if metrics.get("replay_manifest_file_sha256") != str(
        ranking["replay_manifest_file_sha256"]
    ).lower():
        raise ValueError("G1 output replay hash differs.")
    if metrics.get("protection_snapshot_sha256") != expected_snapshots:
        raise ValueError("G1 output snapshot binding differs.")
    elapsed = float(metrics.get("elapsed_seconds", float("nan")))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("G1 elapsed time is invalid.")
    decision = metrics.get("decision", {})
    checks = decision.get("checks", {})
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("G1 decision checks are missing.")
    return {
        "metrics": str(metrics_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "backtracking_trace": str(trace_path),
        "backtracking_trace_sha256": _file_sha256(trace_path),
        "candidate_state": str(state_path),
        "candidate_state_sha256": _file_sha256(state_path),
        "elapsed_seconds": elapsed,
        "checks": dict(checks),
        "gate_pass": bool(decision.get("pass", False)),
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
        "schema": (
            "tausb.dgcaip-g1-strict-controller-status.v2"
            if preflight["spec_id"] == V2_SPEC_ID
            else "tausb.dgcaip-g1-strict-controller-status.v1"
        ),
        "spec_id": preflight["spec_id"],
        "status": "running",
        "current_stage": "G1_STRICT_MECHANISM",
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
        mechanism = run_guarded(
            stage="G1_STRICT_MECHANISM",
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
            log_path=log_root / "g1_strict.log",
            wall_seconds=WALL_SECONDS,
            first_progress_seconds=10 * 60,
            idle_seconds=WALL_SECONDS,
            require_gpu=True,
        )
        result = _verify_result(config_path)
        result_path = control_root / "g1_strict_result.json"
        atomic_write_json(str(result_path), result)
        status.update(
            {
                "status": "completed",
                "current_stage": "",
                "mechanism": mechanism,
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
            "[DGCAIP-G1-Strict] %s: %s" % (type(error).__name__, error),
            file=sys.stderr,
        )
        return int(getattr(error, "exit_code", 1))
    finally:
        if should_shutdown:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
