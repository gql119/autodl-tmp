import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dcss.resume import apply_relative_overrides, build_resume_run_dir, diagnostic_gate
from dcss.stage1 import train_stage1_poison


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {None, "", "nan", "inf", "-inf"}]
    return float(np.mean(values)) if values else float("nan")


def _metrics(experiment_id, source, margin_multiplier, leakage_multiplier, directory, config):
    with open(os.path.join(directory, "mechanism_metrics.csv"), newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    final_epoch = max(int(row["epoch"]) for row in rows)
    rows = [row for row in rows if int(row["epoch"]) == final_epoch]
    with open(os.path.join(directory, "classwise_metrics.csv"), newline="", encoding="utf-8") as file:
        class_rows = list(csv.DictReader(file))
    observed = [row for row in class_rows if int(row["class_id"]) != 14 and int(row["batches"]) > 0]
    maximum = max(observed, key=lambda row: float(row["leakage"]))
    target = _mean(rows, "target_projected_energy_mean")
    leakage = _mean(rows, "non_target_projected_leakage_mean")
    max_amplitude = max(float(row["perturbation_max_amplitude"]) for row in rows)
    eps = float(config["experiment"]["eps"])
    metrics = {
        "experiment_id": experiment_id,
        "Q_type": source,
        "energy_margin_multiplier": margin_multiplier,
        "leakage_weight_multiplier": leakage_multiplier,
        "target_projected_energy": target,
        "target_in_subspace_ratio": _mean(rows, "target_in_subspace_ratio"),
        "target_outside_energy": _mean(rows, "target_outside_energy_mean"),
        "non_target_leakage": leakage,
        "non_target_leakage_max_class": int(maximum["class_id"]),
        "non_target_leakage_max": float(maximum["leakage"]),
        "non_target_logit_drift": float(np.mean([float(row["logit_drift"]) for row in observed])),
        "R_shift": target / (leakage + 1e-12),
        "target_unit_coverage": float(rows[-1]["target_unit_coverage_running"]),
        "target_assignment_overlap": _mean(rows, "target_assignment_overlap"),
        "perturbation_area_ratio": _mean(rows, "perturbation_area_ratio"),
        "perturbation_max_amplitude": max_amplitude,
        "eps": eps,
        "finite": all(math.isfinite(float(row["loss"])) for row in rows),
        "budget_consistent": max_amplitude <= eps + 1e-6,
    }
    metrics["gate"] = diagnostic_gate(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run one isolated 1-epoch DCSS Stage 1R diagnostic")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source", choices=["dcss", "no_pt"], required=True)
    parser.add_argument("--subspace-path", required=True)
    parser.add_argument("--energy-margin-multiplier", type=float, required=True)
    parser.add_argument("--leakage-weight-multiplier", type=float, required=True)
    args = parser.parse_args()
    directory = build_resume_run_dir(os.path.join(ROOT, "artifacts", "dcss", "resume"), "diagnostic", args.experiment_id)
    os.makedirs(directory, exist_ok=False)
    with open(args.base_config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = apply_relative_overrides(config, args.energy_margin_multiplier, args.leakage_weight_multiplier)
    config["dcss"]["poison_epochs"] = 1
    config["dcss"]["subspace_path"] = args.subspace_path
    with open(os.path.join(directory, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    with open(os.path.join(directory, "command.txt"), "w", encoding="utf-8") as file:
        file.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(directory, "environment.txt"), "w", encoding="utf-8") as file:
        file.write(f"platform={platform.platform()}\npython={platform.python_version()}\ntorch={torch.__version__}\n")
        file.write(f"cuda_available={torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            file.write(f"cuda_device={torch.cuda.get_device_name(0)}\n")
    with open(os.path.join(directory, "git_commit.txt"), "w", encoding="utf-8") as file:
        file.write(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n")
    try:
        with open(config["dcss"]["legacy_config"], encoding="utf-8") as file:
            legacy = yaml.safe_load(file)["methods"]["tausb_mask"]
        checkpoint = train_stage1_poison(config, directory, args.source, legacy)
        protected = os.path.join(directory, "protected_data_checkpoint.pt")
        shutil.copy2(checkpoint, protected)
        shutil.copy2(os.path.join(directory, "poison_generation.log"), os.path.join(directory, "run.log"))
        metrics = _metrics(
            args.experiment_id, args.source, args.energy_margin_multiplier,
            args.leakage_weight_multiplier, directory, config,
        )
        with open(os.path.join(directory, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    except Exception as exc:
        failure = {
            "time": __import__("datetime").datetime.now().astimezone().isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config": os.path.join(directory, "config.yaml"), "command": " ".join(sys.argv),
            "checkpoint": config["surrogate"]["checkpoint"], "data": config["data"]["dataset_root"],
            "failure_stage": "Stage 1R-C2 diagnostic screening", "expected": "finite one-epoch mechanism metrics",
            "actual": repr(exc), "log_evidence": os.path.join(directory, "poison_generation.log"),
            "preliminary_cause": "diagnostic execution failure", "checked_items": [], "fix": "not applied",
            "post_fix_result": "not available", "affects_E0_E4": False,
            "failure_type": "implementation failure", "traceback": traceback.format_exc(),
        }
        with open(os.path.join(directory, "failure.json"), "w", encoding="utf-8") as file:
            json.dump(failure, file, indent=2, ensure_ascii=False)
        raise


if __name__ == "__main__":
    main()
