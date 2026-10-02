"""Compare existing noise diagnostics without treating them as causal proof."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FIELDS = {
    "non_target_feature_drift": "non_target_feature_drift",
    "target_feature_drift": "target_feature_drift",
    "L_orth_worst": "L_orth_worst",
    "BG_alignment": "cos_target_bg_sel",
    "attack_orth_gradient_cosine": "cos_grad_attack_orth",
}


def summarize(path, safe=False):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty diagnostics CSV: {path}")
    out = {name: statistics.mean(float(row[column]) for row in rows)
           for name, column in FIELDS.items()}
    if safe:
        for column in ("safe_direction_retention", "removed_nt_component_ratio",
                       "collapsed_safe_direction_ratio", "cos_poison_safe_bg",
                       "cos_poison_raw_bg"):
            values = [float(row[column]) for row in rows]
            if not all(math.isfinite(x) for x in values):
                raise RuntimeError(f"Nonfinite mechanism diagnostic {column}")
            out[column + "_mean"] = statistics.mean(values)
            if column == "safe_direction_retention":
                out[column + "_median_of_step_medians"] = statistics.median(values)
    return {"steps": len(rows), "summary": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-b2-to-csv", required=True)
    parser.add_argument("--safe-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"formal_B2_TO": summarize(args.formal_b2_to_csv),
              "B2_TO_Safe": summarize(args.safe_csv, safe=True),
              "interpretation": "Descriptive matched-run diagnostics only; these summaries do not establish causation.",
              "orthogonality_lock": {"lambda_orth": 0.0005, "orth_worst_k": 3,
                                     "objective": "angular; unchanged from formal B2-TO"}}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
