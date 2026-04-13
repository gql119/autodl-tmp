#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/root/ue_project/runs}"
OUT_DIR="${RUN_ROOT}/aggregate"
mkdir -p "${OUT_DIR}"

python - <<'PY'
import csv
import glob
import json
import os

import numpy as np

run_root = os.environ.get("RUN_ROOT", "/root/ue_project/runs")
out_dir = os.path.join(run_root, "aggregate")
os.makedirs(out_dir, exist_ok=True)

metrics_files = glob.glob(os.path.join(run_root, "artifacts", "*", "steps*", "seed*", "metrics", "metrics.json"))
rows = []
for p in metrics_files:
    with open(p, "r", encoding="utf-8") as f:
        rows.append(json.load(f))

per_run_csv = os.path.join(out_dir, "per_run_summary.csv")
mean_std_csv = os.path.join(out_dir, "method_summary_mean_std.csv")
mean_std_md = os.path.join(out_dir, "method_summary_mean_std.md")

if not rows:
    for p in [per_run_csv, mean_std_csv]:
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("")
    with open(mean_std_md, "w", encoding="utf-8") as f:
        f.write("No metrics found.\n")
    print("[aggregate_all_4] no metrics found")
    raise SystemExit(0)

all_fields = sorted(set().union(*[set(r.keys()) for r in rows]))
with open(per_run_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

numeric_cols = [
    "mAP50_target",
    "mAP50_non_target",
    "mAP50_all",
    "AP_person_free_non_target",
    "AP_person_cooccur_non_target",
    "PSNR",
    "LPIPS",
    "average_perturbed_area_ratio",
    "target_collapse_score",
    "non_target_retention_score",
]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

by_method = {}
for r in rows:
    m = r.get("method", "unknown")
    by_method.setdefault(m, []).append(r)

summary_rows = []
for method in sorted(by_method.keys()):
    items = by_method[method]
    row = {"method": method, "runs": len(items)}
    for c in numeric_cols:
        vals = [to_float(x.get(c)) for x in items]
        vals = [v for v in vals if not np.isnan(v)]
        row[f"{c}_mean"] = float(np.mean(vals)) if vals else float("nan")
        row[f"{c}_std"] = float(np.std(vals)) if vals else float("nan")
    summary_rows.append(row)

summary_fields = ["method", "runs"] + [f"{c}_mean" for c in numeric_cols] + [f"{c}_std" for c in numeric_cols]
with open(mean_std_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

with open(mean_std_md, "w", encoding="utf-8") as f:
    f.write("# Method Summary (Mean ? Std)\n\n")
    if not summary_rows:
        f.write("No data.\n")
    else:
        headers = ["method", "runs", "mAP50_target_mean", "mAP50_non_target_mean", "target_collapse_score_mean", "non_target_retention_score_mean"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in summary_rows:
            vals = [str(r.get(h, "")) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

print("[aggregate_all_4] wrote:", per_run_csv)
print("[aggregate_all_4] wrote:", mean_std_csv)
print("[aggregate_all_4] wrote:", mean_std_md)
PY

echo "[aggregate_all_4] done"

