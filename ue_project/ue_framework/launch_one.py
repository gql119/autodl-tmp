import argparse
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ue_framework.config import SUPPORTED_METHODS, SUPPORTED_STAGES, load_config
from ue_framework.env_utils import detect_platform
from ue_framework.paths import build_run_paths, ensure_run_dirs
from ue_framework.runtime import RunContext
from ue_framework.stages import (
    run_aggregate,
    run_evaluate,
    run_generate_poisoned_dataset,
    run_train_victim,
)



def _run_stage(ctx: RunContext, stage: str) -> None:
    if stage == "generate_poisoned_dataset":
        run_generate_poisoned_dataset(ctx)
    elif stage == "train_victim":
        run_train_victim(ctx)
    elif stage == "evaluate":
        run_evaluate(ctx)
    elif stage == "aggregate":
        run_aggregate(ctx)
    else:
        raise ValueError(f"Unsupported stage: {stage}")



def _fresh_reset_if_needed(paths, stage: str, resume: bool):
    if resume:
        return
    if stage not in {"all", "generate_poisoned_dataset"}:
        return

    for p in [paths.poisoned_root, paths.artifact_root]:
        if os.path.isdir(p):
            shutil.rmtree(p)



def parse_args():
    ap = argparse.ArgumentParser(description="Launch one formal experiment run")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--method", type=str, required=True, choices=SUPPORTED_METHODS)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", type=str, default="all", choices=SUPPORTED_STAGES)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--force_resume", action="store_true")
    return ap.parse_args()



def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.steps not in [int(s) for s in cfg["experiment"]["steps"]]:
        raise ValueError(f"steps={args.steps} not in config experiment.steps={cfg['experiment']['steps']}")
    if args.seed not in [int(s) for s in cfg["experiment"]["seeds"]]:
        raise ValueError(f"seed={args.seed} not in config experiment.seeds={cfg['experiment']['seeds']}")

    # Formal experiments default to fresh runs.
    if not args.force_resume:
        cfg["platform"]["resume"] = False

    platform_mode = detect_platform(cfg)
    run_root = cfg["platform"].get("run_root", "./runs_formal")
    os.makedirs(run_root, exist_ok=True)

    paths = build_run_paths(run_root, args.method, args.steps, args.seed)

    _fresh_reset_if_needed(paths, args.stage, bool(cfg["platform"].get("resume", False)))
    ensure_run_dirs(paths)

    ctx = RunContext(
        cfg=cfg,
        method=args.method,
        steps=args.steps,
        seed=args.seed,
        stage=args.stage,
        gpu_id=args.gpu_id,
        platform_mode=platform_mode,
        paths=paths,
    )

    print("[launch_one] method=", args.method)
    print("[launch_one] steps=", args.steps)
    print("[launch_one] seed=", args.seed)
    print("[launch_one] stage=", args.stage)
    print("[launch_one] run_root=", paths.run_root)
    print("[launch_one] artifact_root=", paths.artifact_root)
    print("[launch_one] poisoned_root=", paths.poisoned_root)
    print("[launch_one] resume=", cfg["platform"].get("resume", False))

    if args.stage == "all":
        for s in ["generate_poisoned_dataset", "train_victim", "evaluate"]:
            _run_stage(ctx, s)
    else:
        _run_stage(ctx, args.stage)


if __name__ == "__main__":
    main()
