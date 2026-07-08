from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper around ue_framework.launch_one evaluate stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--poisoned_root_override", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    method = args.method or str(cfg.get("method", ""))
    steps = args.steps or int(cfg.get("steps", cfg.get("experiment", {}).get("steps", [0])[0]))

    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "ue_framework", "launch_one.py"),
        "--config",
        args.config,
        "--method",
        method,
        "--steps",
        str(steps),
        "--seed",
        str(args.seed),
        "--stage",
        "evaluate",
        "--gpu_id",
        str(args.gpu_id),
        "--force_resume",
    ]
    if args.poisoned_root_override:
        cmd.extend(["--poisoned_root_override", args.poisoned_root_override])
    raise SystemExit(subprocess.call(cmd, cwd=ROOT_DIR))


if __name__ == "__main__":
    main()
