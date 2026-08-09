import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..runtime import RunContext
from ..metrics_utils import VOC20_CLASS_NAMES
from ..io_utils import atomic_write_csv, atomic_write_json
from ..status import load_or_init_status, mark_stage_completed, mark_stage_running, stage_completed


NUMERIC_FIELDS = [
    "mAP50_target",
    "mAP50_non_target",
    "mAP50_non_target_macro",
    "mAP50_all",
    "AP_person_free_non_target",
    "AP_person_cooccur_non_target",
    "PSNR",
    "LPIPS",
    "average_perturbed_area_ratio",
    "average_support_area_ratio",
    "poisoned_count",
    "actual_linf_max",
    "actual_linf_mean",
    "target_collapse_score",
    "non_target_retention_score",
]


def build_c0_m1_comparison(
    rows: List[Dict],
    *,
    method: str,
    steps: int,
    seed: int,
) -> Dict:
    selected = [
        row
        for row in rows
        if str(row.get("method")) == str(method)
        and int(row.get("steps")) == int(steps)
        and int(row.get("seed")) == int(seed)
        and str(row.get("run_tag", "")) in {"C0", "M1"}
    ]
    by_tag = defaultdict(list)
    for row in selected:
        by_tag[str(row["run_tag"])].append(row)
    if set(by_tag) != {"C0", "M1"} or any(len(values) != 1 for values in by_tag.values()):
        raise ValueError("C0/M1 comparison requires exactly one metric row per arm.")
    c0 = by_tag["C0"][0]
    m1 = by_tag["M1"][0]
    if list(c0.get("voc20_class_names", [])) != list(VOC20_CLASS_NAMES):
        raise ValueError("C0 VOC20 class order is missing or incorrect.")
    if list(m1.get("voc20_class_names", [])) != list(VOC20_CLASS_NAMES):
        raise ValueError("M1 VOC20 class order is missing or incorrect.")
    c0_ap = c0.get("ap50_by_class")
    m1_ap = m1.get("ap50_by_class")
    if not isinstance(c0_ap, dict) or not isinstance(m1_ap, dict):
        raise ValueError("C0/M1 metrics require named ap50_by_class mappings.")
    if set(c0_ap) != set(VOC20_CLASS_NAMES) or set(m1_ap) != set(VOC20_CLASS_NAMES):
        raise ValueError("C0/M1 named AP50 mappings must contain all VOC20 classes.")

    per_class = []
    for class_id, name in enumerate(VOC20_CLASS_NAMES):
        clean = float(c0_ap[name])
        poison = float(m1_ap[name])
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in (clean, poison)):
            raise ValueError(f"Non-finite or out-of-range AP50 for class {name}.")
        if clean <= 0:
            raise ValueError(f"C0 AP50 for class {name} is zero; retention is undefined.")
        per_class.append(
            {
                "class_id": class_id,
                "class_name": name,
                "C0_AP50": clean,
                "M1_AP50": poison,
                "delta_M1_minus_C0": poison - clean,
                "drop_C0_minus_M1": clean - poison,
                "retention_M1_over_C0": poison / clean,
                "is_target": name == "person",
            }
        )
    target = next(row for row in per_class if row["is_target"])
    non_target = [row for row in per_class if not row["is_target"]]
    non_target_macro_c0 = float(np.mean([row["C0_AP50"] for row in non_target]))
    non_target_macro_m1 = float(np.mean([row["M1_AP50"] for row in non_target]))
    non_target_macro_drop = non_target_macro_c0 - non_target_macro_m1
    if int(c0.get("poisoned_count", -1)) != 0:
        raise ValueError("C0 must have poisoned_count=0.")
    if int(m1.get("poisoned_count", -1)) != 6095:
        raise ValueError("M1 must have poisoned_count=6095.")
    actual_linf_max = float(m1["actual_linf_max"])
    if not np.isfinite(actual_linf_max) or actual_linf_max > 16.0 / 255.0 + 1.0 / 255.0:
        raise ValueError("M1 actual_linf_max is non-finite or exceeds the approved budget.")
    summary = {
        "schema_version": 1,
        "method": method,
        "steps": int(steps),
        "seed": int(seed),
        "claim_status": "tentative_single_seed",
        "AP50_person_C0": target["C0_AP50"],
        "AP50_person_M1": target["M1_AP50"],
        "AP50_person_drop": target["drop_C0_minus_M1"],
        "mAP50_non_target_macro_C0": non_target_macro_c0,
        "mAP50_non_target_macro_M1": non_target_macro_m1,
        "mAP50_non_target_macro_drop": non_target_macro_drop,
        "non_target_classes_drop_le_0_10": sum(
            row["drop_C0_minus_M1"] <= 0.10 for row in non_target
        ),
        "non_target_classes_drop_gt_0_15": sum(
            row["drop_C0_minus_M1"] > 0.15 for row in non_target
        ),
        "target_success": target["drop_C0_minus_M1"] >= 0.30,
        "non_target_macro_success": non_target_macro_drop <= 0.05,
        "non_target_count_success": sum(
            row["drop_C0_minus_M1"] <= 0.10 for row in non_target
        ) >= 16,
        "poisoned_count": 6095,
        "actual_linf_max": actual_linf_max,
        "quality_validation_gaps": list(m1.get("quality_validation_gaps", [])),
    }
    summary["fresh_victim_success"] = bool(
        summary["target_success"]
        and summary["non_target_macro_success"]
        and summary["non_target_count_success"]
    )
    return {"summary": summary, "per_class": per_class}



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
    out_comparison_json = os.path.join(run_root, "c0_m1_comparison.json")
    out_comparison_csv = os.path.join(run_root, "c0_m1_per_class.csv")

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
        key = (r.get("method"), r.get("steps"), r.get("run_tag", ""))
        grouped[key].append(r)

    grouped_rows = []
    for (method, steps, run_tag), items in grouped.items():
        row = {"method": method, "steps": steps, "run_tag": run_tag, "runs": len(items)}
        for nf in NUMERIC_FIELDS:
            vals = [_safe_float(i.get(nf)) for i in items]
            vals = [v for v in vals if not np.isnan(v)]
            row[nf] = float(np.mean(vals)) if vals else float("nan")
        grouped_rows.append(row)

    g_fields = ["method", "steps", "run_tag", "runs"] + NUMERIC_FIELDS
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

    comparison_paths = {}
    formal_rows = [
        row
        for row in rows
        if row.get("method") == "sirc_malc_cgr"
        and str(row.get("run_tag", "")) in {"C0", "M1"}
    ]
    if formal_rows:
        comparison = build_c0_m1_comparison(
            rows,
            method="sirc_malc_cgr",
            steps=40,
            seed=0,
        )
        atomic_write_json(out_comparison_json, comparison)
        atomic_write_csv(
            out_comparison_csv,
            comparison["per_class"],
            [
                "class_id",
                "class_name",
                "C0_AP50",
                "M1_AP50",
                "delta_M1_minus_C0",
                "drop_C0_minus_M1",
                "retention_M1_over_C0",
                "is_target",
            ],
        )
        comparison_paths = {
            "c0_m1_comparison_json": out_comparison_json,
            "c0_m1_per_class_csv": out_comparison_csv,
        }

    return {
        "summary_csv": out_summary,
        "summary_grouped_csv": out_grouped,
        "pareto_summary_csv": out_pareto,
        "count": len(rows),
        **comparison_paths,
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

