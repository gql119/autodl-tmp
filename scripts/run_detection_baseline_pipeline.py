"""One-method generation -> audited materialization -> matched victim pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928"
)
DATASET_ROOT = Path("/data/guoqinglong/datasets/VOC_0712_Kaggle_Ready")
SURROGATE = Path("/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/checkpoints/voc20_surrogate.pt")
METHODS = ("em_det", "rem_det", "entf_push_det", "sem_det")


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_idle_gpu(poll_seconds: float = 30.0) -> None:
    """Do not start compute while another process uses the selected GPU."""
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != "[Not Found]"]
        if not pids:
            return
        time.sleep(poll_seconds)


def run_stage(command: list[str], cwd: Path, log_path: Path, timeout_seconds: float | None = None) -> None:
    environment = dict(os.environ)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(cwd) if not previous else str(cwd) + os.pathsep + previous
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, timeout=timeout_seconds, env=environment)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-cap-hours", type=float, default=8.0)
    parser.add_argument("--budget-variant", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,95}", args.run_id):
        raise ValueError("RunID must be a simple unique path component")
    if not (0 < args.time_cap_hours <= 8.0) or args.batch_size < 1 or args.max_cycles < 1:
        raise ValueError("Generation must use a positive batch and no more than the authorized eight hours")
    if args.max_cycles > 1 and not args.budget_variant:
        raise ValueError("Multi-cycle generation requires an explicit budget-variant identity")
    run_root = (ALLOWED_ROOT / args.run_id).resolve()
    try:
        run_root.relative_to(ALLOWED_ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("RunID output escaped the authorized root") from exc
    if run_root.exists():
        raise FileExistsError(f"Refusing existing official RunID: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "protocol_lock.json"
    generation = run_root / "generation"
    materialization = run_root / "materialization"
    state_path = run_root / "pipeline_status.json"
    status = {"state": "running", "run_id": args.run_id, "method": args.method,
              "stage": "waiting_for_generation_gpu", "started_at_unix": time.time(),
              "noise_time_cap_hours": args.time_cap_hours, "batch_size": args.batch_size,
              "budget_variant": args.budget_variant, "max_cycles": args.max_cycles,
              "pipeline_code_sha256": {
                  name: sha256_file(ROOT / name) for name in (
                      "scripts/run_detection_baseline_pipeline.py",
                      "scripts/generate_detection_baseline.py",
                      "scripts/materialize_detection_baseline.py",
                      "scripts/run_detection_baseline_victim.py",
                  "ue_framework/baselines_det/locked_data.py",
                  "ue_framework/baselines_det/production.py",
                  ("docs/baseline_budget8_20260930.md" if args.budget_variant
                   else "docs/baseline_schedule_diff.md"),
                  )
              }}
    write_status(state_path, status)
    try:
        wait_for_idle_gpu()
        status.update(stage="generation", stage_started_unix=time.time())
        write_status(state_path, status)
        generation_command = [
            sys.executable, str(ROOT / "scripts" / "generate_detection_baseline.py"),
            "--method", args.method, "--run-id", args.run_id,
            "--dataset-root", str(DATASET_ROOT), "--surrogate", str(SURROGATE),
            "--protocol-lock", str(lock), "--output-root", str(generation),
            "--batch-size", str(args.batch_size), "--time-cap-hours", str(args.time_cap_hours),
            "--seed", str(args.seed), "--code-commit", args.code_commit,
            "--max-cycles", str(args.max_cycles),
        ]
        if args.budget_variant:
            generation_command.append("--budget-variant")
        run_stage(generation_command, ROOT, run_root / "generation.console.log",
                  timeout_seconds=args.time_cap_hours * 3600.0)
        gen_status = json.loads((generation / "status.json").read_text(encoding="utf-8"))
        if (gen_status.get("state") != "complete" or gen_status.get("completed_target_images") != 6095
                or gen_status.get("method_schedule", {}).get("complete_outer_cycles", 0) < 1):
            raise RuntimeError("Generation did not complete a full person-image coverage cycle")
        status.update(stage="materialization", generation_completed_at_unix=time.time(),
                      completed_outer_cycles=gen_status["method_schedule"]["complete_outer_cycles"],
                      generation_stop_reason=gen_status.get("stop_reason"),
                      generation_duration_seconds=gen_status.get("duration_seconds"))
        write_status(state_path, status)
        run_stage([
            sys.executable, str(ROOT / "scripts" / "materialize_detection_baseline.py"),
            "--method", args.method, "--run-id", args.run_id,
            "--dataset-root", str(DATASET_ROOT), "--protocol-lock", str(lock),
            "--noise-run", str(generation), "--output", str(materialization),
        ], ROOT, run_root / "materialization.console.log")
        full_audit = json.loads((materialization / "full_preflight.json").read_text(encoding="utf-8"))
        if full_audit.get("state") != "passed":
            raise RuntimeError("Full materialized-data audit did not pass")
        wait_for_idle_gpu()
        status.update(stage="victim_training", materialization_completed_at_unix=time.time())
        write_status(state_path, status)
        run_stage([
            sys.executable, str(ROOT / "scripts" / "run_detection_baseline_victim.py"),
            "--run-root", str(run_root), "--dataset-root", str(DATASET_ROOT),
            "--protocol-lock", str(lock), "--method", args.method, "--run-id", args.run_id,
            "--code-commit", args.code_commit,
        ], ROOT, run_root / "victim.console.log")
        victim_status = json.loads((run_root / "victim_status.json").read_text(encoding="utf-8"))
        if victim_status.get("state") != "complete" or victim_status.get("epochs_completed") != 200:
            raise RuntimeError("Victim did not complete the locked 200-epoch run")
        status.update(stage="complete", state="complete", victim_completed_at_unix=time.time(),
                      metrics_path=str(run_root / "clean_val_metrics.json"),
                      elapsed_seconds=time.time() - status["started_at_unix"])
        write_status(state_path, status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), failed_stage=status.get("stage"),
                      failed_at_unix=time.time())
        write_status(state_path, status)
        raise
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
