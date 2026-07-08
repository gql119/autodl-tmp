from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from runners.run_p0_diagnostics import run_diagnostics


NEW_METHODS = {"trajectory_p1", "meta_p2", "trajectory_meta_p2"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poison or run new trajectory diagnostics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    method = str(cfg.get("method", ""))
    if method in NEW_METHODS:
        metrics, output_path = run_diagnostics(args.config)
        print(f"[generate_poison] method={method} diagnostic_output={output_path}")
        print(
            "[generate_poison] VOC materialization for trajectory methods is intentionally not "
            "routed through legacy per-image ALCE code yet."
        )
        print(metrics)
        return

    raise SystemExit(
        "For legacy methods use ue_framework/launch_one.py with --stage generate_poisoned_dataset. "
        f"Unsupported config method for this runner: {method}"
    )


if __name__ == "__main__":
    main()
