import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..runtime import RunContext
from ..status import load_or_init_status, mark_stage_completed, mark_stage_running, stage_completed


NUMERIC_FIELDS = [
    "mAP50_target",
    "mAP50_non_target",
    "mAP50_all",
    "AP_person_free_non_target",
    "AP_person_cooccur_non_target",
    "PSNR",
    "LPIPS",
    "average_perturbed_area_ratio",
    "average_support_area_ratio",
    "poisoned_count",
    "target_collapse_score",
    "non_target_retention_score",
]



def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")



def _pareto_rank(rows: List[Dict]) -> List[int]:
    # maximize both target_collapse_score and non_target_retention_score
    vals = [
        (
            _safe_float(r.get("target_collapse_score")),
            _safe_float(r.get("non_target_retention_score")),
        )
        for r in rows
    ]
    ranks = [999] * len(rows)
    alive = set(range(len(rows)))
    cur_rank = 1

    while alive:
        front = []
        for i in list(alive):
            dominated = False
            ai, bi = vals[i]
            for j in alive:
                if i == j:
                    continue
                aj, bj = vals[j]
                if np.isnan(ai) or np.isnan(bi) or np.isnan(aj) or np.isnan(bj):
                    continue
                if (aj >= ai and bj >= bi) and (aj > ai or bj > bi):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        if not front:
            for i in alive:
                ranks[i] = cur_rank
            break

        for i in front:
            ranks[i] = cur_rank
            alive.remove(i)
        cur_rank += 1

    return ranks



def aggregate_root(run_root: str) -> Dict[str, str]:
    metrics_files = glob.glob(
        os.path.join(run_root, "artifacts", "*", "steps*", "seed*", "metrics", "metrics.json")
    )

    rows = []
    for m in metrics_files:
        with open(m, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(data)

    out_summary = os.path.join(run_root, "summary.csv")
    out_grouped = os.path.join(run_root, "summary_grouped.csv")
    out_pareto = os.path.join(run_root, "pareto_summary.csv")

    if not rows:
        for p in [out_summary, out_grouped, out_pareto]:
            with open(p, "w", encoding="utf-8") as f:
                f.write("")
        return {
            "summary_csv": out_summary,
            "summary_grouped_csv": out_grouped,
            "pareto_summary_csv": out_pareto,
            "count": 0,
        }

    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(out_summary, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("method"), r.get("steps"))
        grouped[key].append(r)

    grouped_rows = []
    for (method, steps), items in grouped.items():
        row = {"method": method, "steps": steps, "runs": len(items)}
        for nf in NUMERIC_FIELDS:
            vals = [_safe_float(i.get(nf)) for i in items]
            vals = [v for v in vals if not np.isnan(v)]
            row[nf] = float(np.mean(vals)) if vals else float("nan")
        grouped_rows.append(row)

    g_fields = ["method", "steps", "runs"] + NUMERIC_FIELDS
    with open(out_grouped, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=g_fields)
        w.writeheader()
        for r in grouped_rows:
            w.writerow(r)

    ranks = _pareto_rank(grouped_rows)
    for i, r in enumerate(grouped_rows):
        r["pareto_rank"] = ranks[i]

    p_fields = g_fields + ["pareto_rank"]
    with open(out_pareto, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=p_fields)
        w.writeheader()
        for r in sorted(grouped_rows, key=lambda x: (_safe_float(x.get("pareto_rank")), x.get("method", ""))):
            w.writerow(r)

    return {
        "summary_csv": out_summary,
        "summary_grouped_csv": out_grouped,
        "pareto_summary_csv": out_pareto,
        "count": len(rows),
    }



def run_aggregate(ctx: RunContext) -> None:
    status = load_or_init_status(ctx.paths.artifact_status_json, ctx.method, ctx.steps, ctx.seed)
    if ctx.cfg["platform"].get("resume", True) and stage_completed(status, "aggregate"):
        print("[aggregate] already completed, skipping.")
        return

    status = mark_stage_running(ctx.paths.artifact_status_json, status, "aggregate")
    outputs = aggregate_root(ctx.paths.run_root)
    mark_stage_completed(ctx.paths.artifact_status_json, status, "aggregate", outputs)
    print(
        "[aggregate] done: "
        f"summary={outputs['summary_csv']}, pareto={outputs['pareto_summary_csv']}, count={outputs['count']}"
    )

