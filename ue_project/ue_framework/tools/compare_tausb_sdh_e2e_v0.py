from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ue_framework.methods.sdh_evaluation import build_sdh_e2e_v0_comparison


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit paired C0/M1 VOC20 metrics for E2E V0."
    )
    parser.add_argument("--c0-metrics", required=True)
    parser.add_argument("--m1-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    c0_path = Path(args.c0_metrics).resolve()
    m1_path = Path(args.m1_metrics).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Comparison output directory already exists: %s" % output_dir)
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    comparison = build_sdh_e2e_v0_comparison(c0, m1)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "per_class_ap50.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        rows = comparison["per_class"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(comparison["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
