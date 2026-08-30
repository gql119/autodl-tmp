from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ue_framework.io_utils import atomic_write_json
from ue_framework.tools.bind_tausb_sdh_dgcaip_p4_e20 import (
    validate_dgcaip_p4_binding,
)
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import (
    GuardFailure,
    run_guarded,
)


OVERALL_WALL_SECONDS = 2 * 60 * 60
MECHANISM_WALL_SECONDS = 20 * 60


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict DG-CAIP P4 production then paired sparse E20 once."
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--required-storage-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--device", default="0")
    parser.add_argument("--mechanism-config", required=True)
    parser.add_argument("--mechanism-root", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--binding-root", required=True)
    parser.add_argument("--run-root-prefix", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--sparse-control-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    return parser.parse_args()


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=str(repository_root), text=True
    ).strip()


def _remaining(started: float) -> int:
    seconds = OVERALL_WALL_SECONDS - int(time.monotonic() - started)
    if seconds <= 0:
        raise GuardFailure("ONEBOOT", "overall_wall_timeout", 124)
    return seconds


def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    repository_root = Path(args.repository_root).resolve()
    project_root = repository_root / "ue_project"
    python_bin = Path(args.python_bin).resolve()
    mechanism_config = Path(args.mechanism_config).resolve()
    mechanism_root = Path(args.mechanism_root).resolve()
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    fresh = [control_root, log_root, mechanism_root]
    if _git(repository_root, "rev-parse", "HEAD") != args.expected_commit:
        raise GuardFailure("PRECHECK", "execution_commit_mismatch")
    if _git(repository_root, "status", "--porcelain"):
        raise GuardFailure("PRECHECK", "execution_checkout_not_clean")
    if any(path.exists() for path in fresh):
        raise GuardFailure("PRECHECK", "fresh_oneboot_path_already_exists")
    if not python_bin.is_file() or not mechanism_config.is_file():
        raise GuardFailure("PRECHECK", "python_or_mechanism_config_missing")
    control_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    status = {
        "schema": "tausb.dgcaip-p4-oneboot-status.v1",
        "status": "running",
        "current_stage": "MECHANISM",
        "execution_commit": args.expected_commit,
        "started_unix": time.time(),
        "overall_wall_cap_seconds": OVERALL_WALL_SECONDS,
        "stages": {},
    }
    atomic_write_json(str(control_root / "controller_status.json"), status)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cache_root = Path(args.cache_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    os.environ.update(
        {
            "TMPDIR": str(tmp_root),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TORCH_HOME": str(cache_root / "torch"),
            "YOLO_CONFIG_DIR": str(cache_root / "yolo"),
        }
    )
    mechanism = run_guarded(
        stage="MECHANISM",
        command=(
            str(python_bin),
            "-u",
            "-m",
            "ue_framework.tools.run_tausb_sdh",
            "--config",
            str(mechanism_config),
            "--stage",
            "mechanism",
        ),
        cwd=project_root,
        log_path=log_root / "mechanism.log",
        wall_seconds=min(MECHANISM_WALL_SECONDS, _remaining(started)),
        first_progress_seconds=300,
        idle_seconds=600,
        require_gpu=True,
    )
    status["stages"]["MECHANISM"] = mechanism
    status["current_stage"] = "P4_BINDING_GATE"
    atomic_write_json(str(control_root / "controller_status.json"), status)
    binding = validate_dgcaip_p4_binding(mechanism_root, mechanism_config)
    status["stages"]["P4_BINDING_GATE"] = {
        "state_integrity_gate_passed": True,
        "mechanism_scientific_gate_passed": bool(
            binding["metrics"]["decision"]["pass"]
        ),
        "candidate_state": str(binding["state_path"]),
    }
    status["current_stage"] = "SPARSE_E20"
    atomic_write_json(str(control_root / "controller_status.json"), status)

    sparse = run_guarded(
        stage="SPARSE_E20",
        command=(
            str(python_bin),
            "-u",
            "-m",
            "ue_framework.tools.run_tausb_sdh_sparse_e20",
            "--repository-root",
            str(repository_root),
            "--required-storage-root",
            args.required_storage_root,
            "--expected-commit",
            args.expected_commit,
            "--python-bin",
            str(python_bin),
            "--device",
            args.device,
            "--mechanism-root",
            str(mechanism_root),
            "--mechanism-config",
            str(mechanism_config),
            "--base-config",
            args.base_config,
            "--dataset-root",
            args.dataset_root,
            "--binding-root",
            args.binding_root,
            "--run-root-prefix",
            args.run_root_prefix,
            "--control-root",
            args.sparse_control_root,
            "--log-root",
            str(log_root / "sparse"),
            "--comparison-root",
            args.comparison_root,
            "--cache-root",
            args.cache_root,
            "--tmp-root",
            args.tmp_root,
            "--victim-epochs",
            "20",
            "--binding-protocol",
            "dgcaip_p4",
        ),
        cwd=project_root,
        log_path=log_root / "sparse_controller.log",
        wall_seconds=_remaining(started),
        first_progress_seconds=1200,
        idle_seconds=OVERALL_WALL_SECONDS,
        require_gpu=False,
    )
    status["stages"]["SPARSE_E20"] = sparse
    status["status"] = "completed"
    status["current_stage"] = ""
    status["ended_unix"] = time.time()
    status["elapsed_seconds"] = time.monotonic() - started
    atomic_write_json(str(control_root / "controller_status.json"), status)
    return 0


def main() -> int:
    args = _arguments()
    control_root = Path(args.control_root).resolve()
    try:
        return _run(args)
    except BaseException as error:
        if control_root.exists():
            atomic_write_json(
                str(control_root / "controller_failure.json"),
                {
                    "schema": "tausb.dgcaip-p4-oneboot-failure.v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "ended_unix": time.time(),
                },
            )
        print("[DGCAIP-P4-OneBoot] %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return int(getattr(error, "exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
