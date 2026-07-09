from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-config", default=str(ROOT / "configs/mtepi/voc_yolov8n_stage2.yaml"))
    parser.add_argument("--stage2-output-dir", default=str(ROOT / "outputs/mtepi_stage2"))
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "runners/run_mtepi_stage2.py"),
        "--config",
        args.stage2_config,
        "--output-dir",
        args.stage2_output_dir,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    summary_path = Path(args.stage2_output_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["stage2_gate"]["gate"] != "PASS":
        print(json.dumps({"pipeline": "stopped", "reason": "STAGE_2_GATE failed", "stage2_summary": str(summary_path)}, indent=2))
        return
    raise SystemExit("Stage 3 runner is intentionally not implemented until Stage 2 passes in a legal checkpoint setting.")


if __name__ == "__main__":
    main()
