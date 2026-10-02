"""Evaluate both locked B2-TO-Safe victim checkpoints on the copied clean VOC val."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.metrics_utils import compute_non_target_map


NAMES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def summarize(box):
    ids = np.asarray(box.ap_class_index, dtype=int).reshape(-1).tolist()
    ap = np.asarray(box.ap50, dtype=float).reshape(-1)
    recall = np.asarray(box.r, dtype=float).reshape(-1)
    if ids != list(range(20)) or len(ap) != 20 or len(recall) != 20:
        raise RuntimeError(f"Expected ordered VOC20 metrics; got ids={ids}")
    if not np.isfinite(ap).all() or not np.isfinite(recall).all():
        raise RuntimeError("Nonfinite checkpoint metrics")
    return {"mAP50_all": float(box.map50), "mAP50_target": float(ap[14]),
            "mAP50_non_target": float(compute_non_target_map(ap.tolist(), 14)),
            "AP50_per_class": ap.tolist(), "Recall_target": float(recall[14]),
            "Recall_per_class": recall.tolist(), "class_names": list(NAMES)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--victim", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    victim = Path(args.victim).resolve()
    root = Path(__file__).resolve().parents[1]
    if not victim.is_relative_to(root):
        raise ValueError("Victim output must remain in the isolated project")
    status = json.loads((victim / "status.json").read_text(encoding="utf-8"))
    spec = json.loads((victim / "run_spec.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or int(spec["train_args"]["epochs"]) != 200:
        raise RuntimeError("Victim must finish the locked 200-epoch training first")
    results_csv = victim / "victim" / "results.csv"
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200:
        raise RuntimeError(f"Victim results.csv has {len(rows)} rows, expected 200")
    weights = victim / "victim" / "weights"
    data_yaml = victim / "train_data.yaml"
    output = Path(args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Checkpoint evaluation output must remain in the isolated project")
    output.mkdir(parents=True, exist_ok=True)
    eval_args = dict(data=str(data_yaml), imgsz=int(cfg["victim"]["imgsz"]),
                     batch=int(cfg["victim"]["batch"]), workers=int(cfg["victim"]["workers"]),
                     device="0", verbose=False, plots=False, seed=0)
    metrics = {}
    for role, filename in (("primary_last", "last.pt"), ("secondary_best", "best.pt")):
        checkpoint = weights / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = YOLO(str(checkpoint)).val(**eval_args, project=str(output), name=role, exist_ok=False)
        summary = summarize(result.box)
        summary["checkpoint"] = str(checkpoint)
        summary["checkpoint_role"] = role
        summary["validation_split"] = "clean VOC validation copied byte-identically by run_isolated_victim"
        metrics[role] = summary
        (output / f"{role}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "checkpoint_metrics.json").write_text(json.dumps({
        "primary": metrics["primary_last"], "secondary": metrics["secondary_best"],
        "primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"last": metrics["primary_last"], "best": metrics["secondary_best"]}, indent=2))


if __name__ == "__main__":
    main()
