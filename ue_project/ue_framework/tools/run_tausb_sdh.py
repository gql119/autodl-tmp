from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

from ue_framework.methods.sdh_experiment import (
    run_hiding_pilot,
    run_mechanism_pilot,
    validate_sdh_experiment_config,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the approved capped TAUSB-SDH hiding or mechanism pilot."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("hiding", "mechanism"))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config_path = Path(args.config).resolve()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    artifact_root = Path(str(config["runtime"]["artifact_root"]))
    status_path = artifact_root / ("status_%s.json" % args.stage)
    started = time.time()
    try:
        if args.stage == "hiding":
            result = run_hiding_pilot(config, config_base=project_root)
        else:
            result = run_mechanism_pilot(config, config_base=project_root)
        status = {
            "stage": args.stage,
            "status": "completed",
            "started_unix": started,
            "ended_unix": time.time(),
            "gate_pass": bool(result.get("gate", result.get("decision", {})).get("pass")),
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return 0
    except Exception as error:
        status = {
            "stage": args.stage,
            "status": "failed",
            "started_unix": started,
            "ended_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
