"""Run the four pre-authorized detection baselines sequentially, fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928"
)
ORDER = (
    ("em_det", "em_det_eps8_seed0_20260928"),
    ("rem_det", "rem_det_eps8_seed0_20260928"),
    ("entf_push_det", "entf_push_det_eps8_seed0_20260928"),
    ("sem_det", "sem_det_eps8_seed0_20260928"),
)
BUDGET_ORDER = (
    ("em_det", "em_det_budget8_seed0_20260930"),
    ("rem_det", "rem_det_budget8_seed0_20260930"),
    ("entf_push_det", "entf_push_det_budget8_seed0_20260930"),
    ("sem_det", "sem_det_budget8_seed0_20260930"),
)
BUDGET_MAX_CYCLES = {
    "em_det": 100,
    "rem_det": 4,
    "entf_push_det": 4,
    "sem_det": 32,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-cap-hours", type=float, default=8.0)
    parser.add_argument("--budget-variant", action="store_true")
    args = parser.parse_args()
    if not (0 < args.time_cap_hours <= 8) or args.batch_size < 1:
        raise ValueError("Invalid schedule: cap must be <= 8h and batch positive")
    if not args.sequence_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Sequence ID must be a simple name")
    order = BUDGET_ORDER if args.budget_variant else ORDER
    schedule_document = ("docs/baseline_budget8_20260930.md" if args.budget_variant
                         else "docs/baseline_schedule_diff.md")
    state_path = ALLOWED_ROOT / f"{args.sequence_id}.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to reuse sequence state: {state_path}")
    state = {
        "state": "running", "sequence_id": args.sequence_id,
        "order": [method for method, _ in order],
        "run_ids": [run_id for _, run_id in order],
        "batch_size": args.batch_size, "per_method_noise_cap_hours": args.time_cap_hours,
        "budget_variant": args.budget_variant,
        "code_commit": args.code_commit,
        "orchestrator_sha256": sha256_file(ROOT / "scripts/run_four_detection_baselines.py"),
        "schedule_document_sha256": sha256_file(ROOT / schedule_document),
        "started_at_unix": time.time(), "methods": [],
    }
    write(state_path, state)
    try:
        for method, run_id in order:
            row = {"method": method, "run_id": run_id, "state": "running", "started_at_unix": time.time()}
            state["methods"].append(row)
            state["current_method"] = method
            write(state_path, state)
            command = [
                sys.executable, str(ROOT / "scripts/run_detection_baseline_pipeline.py"),
                "--method", method, "--run-id", run_id,
                "--batch-size", str(args.batch_size),
                "--time-cap-hours", str(args.time_cap_hours),
                "--seed", "0", "--code-commit", args.code_commit,
            ]
            if args.budget_variant:
                command.extend(["--budget-variant", "--max-cycles",
                                str(BUDGET_MAX_CYCLES[method])])
            log_path = ALLOWED_ROOT / f"{run_id}.orchestrator.console.log"
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, text=True)
            row["exit_code"] = result.returncode
            row["completed_at_unix"] = time.time()
            pipeline_status = ALLOWED_ROOT / run_id / "pipeline_status.json"
            if pipeline_status.is_file():
                row["pipeline_status_sha256"] = sha256_file(pipeline_status)
                row["pipeline_state"] = json.loads(pipeline_status.read_text(encoding="utf-8")).get("state")
            if result.returncode != 0 or row.get("pipeline_state") != "complete":
                row["state"] = "failed"
                state.update(state="failed", stopped_after=method,
                             error="Method pipeline failed; later methods were not launched",
                             completed_at_unix=time.time())
                write(state_path, state)
                raise subprocess.CalledProcessError(result.returncode or 1, command)
            metrics = ALLOWED_ROOT / run_id / "clean_val_metrics.json"
            row["metrics_sha256"] = sha256_file(metrics)
            row["state"] = "complete"
            write(state_path, state)
        state.update(state="complete", completed_at_unix=time.time())
        write(state_path, state)
    except BaseException as exc:
        if state.get("state") == "running":
            state.update(state="failed", error=repr(exc), completed_at_unix=time.time())
            write(state_path, state)
        raise
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
