from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_csdem.dataset import load_config, write_json
from runners.run_mini_csdem_stage1 import run_experiment


ABLATIONS = {
    "A_stage1": {"logits": 0.0, "box": 0.0, "dfl": 0.0, "assignment": 0.0},
    "B_logits": {"logits": 1.0, "box": 0.0, "dfl": 0.0, "assignment": 0.0},
    "C_logits_box_dfl": {"logits": 1.0, "box": 0.5, "dfl": 0.25, "assignment": 0.0},
    "D_all": {"logits": 1.0, "box": 0.5, "dfl": 0.25, "assignment": 0.5},
}


def configure_weights(cfg, weights):
    cfg["preservation"].update(
        {
            "enabled": True,
            "enable_logits": True,
            "enable_box": True,
            "enable_dfl": True,
            "enable_assignment": True,
            "lambda_logits": float(weights["logits"]),
            "lambda_box_keep": float(weights["box"]),
            "lambda_dfl_keep": float(weights["dfl"]),
            "lambda_assign": float(weights["assignment"]),
        }
    )
    cfg["features"]["enable_non_target_preservation"] = True


def run_smoke(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["poison"]["generation_epochs"] = 1
    cfg["victim"]["epochs"] = 1
    cfg["preservation"]["gradient_diagnostics_interval"] = 1
    cfg["paths"]["output_root"] = "outputs/mini_csdem/stage2_smoke"
    return run_experiment(cfg, result_name="stage2_smoke")


def run_ablations(base_cfg):
    results = {}
    for name, weights in ABLATIONS.items():
        cfg = copy.deepcopy(base_cfg)
        configure_weights(cfg, weights)
        cfg["poison"]["generation_epochs"] = 3
        cfg["victim"]["epochs"] = 15
        cfg["paths"]["output_root"] = f"outputs/mini_csdem/stage2_ablations/{name}"
        results[name] = run_experiment(cfg, result_name=f"stage2_ablation_{name}")
    write_json(ROOT / "results/mini_csdem/stage2_ablation_summary.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mini_csdem/stage2.yaml")
    parser.add_argument("--mode", choices=["smoke", "ablation", "formal"], required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
        cfg["victim"]["seed"] = int(args.seed)
        cfg["paths"]["output_root"] = f"outputs/mini_csdem/stage2_seed{args.seed}"
    if args.mode == "smoke":
        result = run_smoke(cfg)
    elif args.mode == "ablation":
        result = run_ablations(cfg)
    else:
        result = run_experiment(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
