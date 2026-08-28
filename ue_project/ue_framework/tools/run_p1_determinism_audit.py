from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

from ue_framework.methods.p1_determinism_experiment import (
    is_deterministic_operator_error,
    run_p1_determinism_lane,
    summarize_p1_determinism_audit,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one approved bounded P1 determinism audit lane."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("normal", "strict", "summarize"))
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_strict_operator_evidence(config: dict, error: BaseException, stack: str) -> dict:
    return {
        "schema": "tausb.p1-determinism-strict-operator-error.v1",
        "spec_id": config.get("spec", {}).get("spec_id"),
        "mode": "strict",
        "validation_status": "not_completed_due_operator_error",
        "pairs": {},
        "operator_error": {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": stack,
        },
    }


def main() -> int:
    args = _arguments()
    config_path = Path(args.config).resolve()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = Path(str(config["runtime"]["artifact_root"]))
    status_path = artifact_root / ("status_%s.json" % args.mode)
    started = time.time()
    try:
        if args.mode == "summarize":
            result = summarize_p1_determinism_audit(
                config, config_base=project_root
            )
        else:
            result = run_p1_determinism_lane(
                config,
                config_base=project_root,
                mode=args.mode,
            )
        status = {
            "schema": "tausb.p1-determinism-status.v1",
            "mode": args.mode,
            "status": "completed",
            "started_unix": started,
            "ended_unix": time.time(),
            "mechanical_pass": bool(
                result.get("decision", {}).get("mechanical_pass", True)
            ),
        }
        _write_json(status_path, status)
        return 0
    except Exception as error:
        stack = traceback.format_exc()
        if args.mode == "strict" and is_deterministic_operator_error(error):
            evidence = build_strict_operator_evidence(config, error, stack)
            _write_json(artifact_root / "strict_operator_error.json", evidence)
            _write_json(
                status_path,
                {
                    "schema": "tausb.p1-determinism-status.v1",
                    "mode": args.mode,
                    "status": "completed_diagnostic_operator_error",
                    "started_unix": started,
                    "ended_unix": time.time(),
                    "mechanical_pass": True,
                },
            )
            return 0
        _write_json(
            status_path,
            {
                "schema": "tausb.p1-determinism-status.v1",
                "mode": args.mode,
                "status": "failed",
                "started_unix": started,
                "ended_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": stack,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
