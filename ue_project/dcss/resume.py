import math
import os
from copy import deepcopy
from typing import Dict


DIAGNOSTIC_THRESHOLDS = {
    "minimum_target_coverage": 0.50,
    "minimum_target_energy": 0.3601,
    "maximum_non_target_leakage": 0.1803,
    "minimum_r_shift": 2.0,
}


def build_resume_run_dir(resume_root: str, prefix: str, run_id: str) -> str:
    if not prefix or not run_id or os.path.basename(prefix) != prefix or os.path.basename(run_id) != run_id:
        raise ValueError("prefix and run_id must be non-empty path components")
    root = os.path.abspath(resume_root)
    path = os.path.abspath(os.path.join(root, f"{prefix}_{run_id}"))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("resume run directory escaped resume root")
    return path


def mean_finite_metric(rows, key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def apply_relative_overrides(config: Dict, energy_margin_multiplier: float, leakage_weight_multiplier: float) -> Dict:
    result = deepcopy(config)
    dcss = result["dcss"]
    dcss["energy_margin"] = float(dcss["energy_margin"]) * float(energy_margin_multiplier)
    dcss["lambda_leakage"] = float(dcss["lambda_leakage"]) * float(leakage_weight_multiplier)
    return result


def diagnostic_gate(metrics: Dict, thresholds: Dict = None) -> Dict:
    limits = dict(DIAGNOSTIC_THRESHOLDS if thresholds is None else thresholds)
    finite_keys = ["target_unit_coverage", "target_projected_energy", "non_target_leakage", "R_shift"]
    finite = all(math.isfinite(float(metrics.get(key, float("nan")))) for key in finite_keys)
    checks = {
        "finite": finite,
        "coverage": finite and float(metrics["target_unit_coverage"]) >= limits["minimum_target_coverage"],
        "target_energy": finite and float(metrics["target_projected_energy"]) >= limits["minimum_target_energy"],
        "non_target_leakage": finite and float(metrics["non_target_leakage"]) <= limits["maximum_non_target_leakage"],
        "R_shift": finite and float(metrics["R_shift"]) >= limits["minimum_r_shift"],
        "budget_consistent": bool(metrics.get("budget_consistent", False)),
    }
    return {"thresholds": limits, "checks": checks, "pass": all(checks.values())}
