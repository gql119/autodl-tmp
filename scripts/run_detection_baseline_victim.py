"""Train and evaluate one locked, random-init victim on an audited dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ue_framework.baselines_det.locked_data import locked_source_inventory, sha256_file
from scripts.run_locked_clean_baseline import (
    check_installed_protocol, locked_train_args, summarize_metrics,
)


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928"
)


def check_materialized(lock: dict, source_summary: dict, run_root: Path,
                       method: str, run_id: str, protocol_lock_path: Path) -> tuple[Path, dict]:
    preflight_path = run_root / "materialization" / "full_preflight.json"
    manifest_path = run_root / "materialization" / "materialization_manifest.jsonl"
    if not preflight_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Full materialization audit evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("state") != "passed":
        raise RuntimeError("Full materialization audit did not pass")
    if preflight.get("method") != method or preflight.get("run_id") != run_id:
        raise RuntimeError("Materialization audit belongs to another method or RunID")
    if preflight.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Materialization manifest changed after its integrity audit")
    if preflight.get("protocol_lock_sha256") != sha256_file(protocol_lock_path):
        raise RuntimeError("Materialization used a different protocol lock")
    if preflight.get("dataset_fingerprint") != source_summary["dataset_fingerprint"]:
        raise RuntimeError("Materialization data identity differs from the source lock")
    expected_counts = {
        "train_images": int(lock["dataset"]["train_images"]),
        "val_images": int(lock["dataset"]["val_images"]),
        "target_images": int(lock["dataset"]["target_images"]),
        "target_objects": int(lock["dataset"]["target_objects"]),
    }
    for key, expected in expected_counts.items():
        if preflight.get(key) != expected:
            raise RuntimeError(f"Materialization audit count mismatch for {key}")
    if preflight.get("changed_person_gt_boxes") != expected_counts["target_objects"]:
        raise RuntimeError("Not every target person GT box received a decoded-pixel change")
    for flag in ("label_bytes_identical", "non_target_train_bytes_identical", "clean_val_bytes_identical"):
        if preflight.get(flag) is not True:
            raise RuntimeError(f"Materialization integrity flag is false: {flag}")
    if preflight.get("changed_outside_pixels") != 0:
        raise RuntimeError("Materialized data changed pixels outside target support")
    budgets = preflight.get("max_abs_delta_byte", {})
    if set(budgets) != {"person_only", "person_overlap"} or budgets["person_only"] > 8 or budgets["person_overlap"] > 4:
        raise RuntimeError("Materialized decoded-pixel epsilon audit exceeds the lock")
    data = Path(preflight["materialized_root"]).resolve()
    expected_data = (run_root / "materialization" / "poisoned_data").resolve()
    if data != expected_data:
        raise RuntimeError("Audited dataset root is not inside this unique RunID")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(lock["dataset"]["train_images"]) + int(lock["dataset"]["val_images"]):
        raise RuntimeError("Materialization manifest row count is incomplete")
    target_rows = [row for row in rows if row.get("split") == "train" and row.get("target") is True]
    if len(target_rows) != expected_counts["target_images"]:
        raise RuntimeError("Materialization manifest target image count differs from the lock")
    if sum(int(row.get("changed_person_gt_boxes", 0)) for row in target_rows) != expected_counts["target_objects"]:
        raise RuntimeError("Materialization manifest lacks a change for every person GT box")
    for row in rows:
        source_image = Path(row["source_image"])
        output_image = Path(row["output_image"])
        output_label = Path(row["output_label"])
        expected_image_root = data / "images" / row["split"]
        expected_label_root = data / "labels" / row["split"]
        if output_image.parent.resolve() != expected_image_root.resolve() or output_label.parent.resolve() != expected_label_root.resolve():
            raise RuntimeError("Manifest points outside the audited materialized split")
        if not output_image.is_file() or not output_label.is_file():
            raise RuntimeError(f"Materialized file is missing: {row['stem']}")
        if sha256_file(output_image) != row["output_image_sha256"] or sha256_file(output_label) != row["label_sha256"]:
            raise RuntimeError(f"Materialized bytes changed after audit: {row['stem']}")
        if sha256_file(source_image) != row["source_image_sha256"]:
            raise RuntimeError(f"Locked source image changed after audit: {row['stem']}")
        if row["split"] == "train" and row.get("target", False) and int(row.get("changed_target_pixels", 0)) < 1:
            raise RuntimeError(f"Target image had no changed decoded pixel: {row['stem']}")
        if row["split"] == "val" or not row.get("target", False):
            if row["source_image_sha256"] != row["output_image_sha256"]:
                raise RuntimeError(f"A clean/non-target image is not byte-identical: {row['stem']}")
    for split in ("train", "val"):
        images = list((data / "images" / split).glob("*"))
        labels = list((data / "labels" / split).glob("*.txt"))
        if len(images) != expected_counts["train_images" if split == "train" else "val_images"]:
            raise RuntimeError(f"Materialized {split} image count changed after audit")
        if len(labels) != len(images) or {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"Materialized {split} image/label stems changed after audit")
    yaml_path = Path(preflight["data_yaml"]).resolve()
    if not yaml_path.is_file():
        raise RuntimeError("Materialized Ultralytics data YAML is missing")
    return yaml_path, preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    try:
        run_root.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Victim output must remain under {ALLOWED_ROOT}") from exc
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    _, source_summary = locked_source_inventory(lock, args.dataset_root)
    if ((run_root / "victim_status.json").exists() or (run_root / "victim").exists() or
            (run_root / "clean_val").exists() or (run_root / "clean_val_metrics.json").exists()):
        raise FileExistsError("Victim stage already exists; refusing a duplicate run")
    yaml_path, preflight = check_materialized(lock, source_summary, run_root,
                                              args.method, args.run_id, args.protocol_lock.resolve())
    check_installed_protocol(lock)
    victim = lock["victim"]
    init_yaml = Path(victim["init_yaml"])
    if sha256_file(init_yaml) != victim["init_yaml_sha256"]:
        raise RuntimeError("Victim initialization YAML hash differs from the lock")
    train_args = locked_train_args(lock, run_root, yaml_path, smoke=False)
    victim_spec = {
        "run_id": args.run_id, "method": args.method,
        "candidate_code_commit": args.code_commit,
        "initialization": "fresh random YAML; pretrained=false",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "dataset_fingerprint": source_summary["dataset_fingerprint"],
        "materialization_preflight_sha256": sha256_file(run_root / "materialization" / "full_preflight.json"),
        "materialization_manifest_sha256": preflight["manifest_sha256"],
        "noise_run_spec_sha256": preflight["noise_run_spec_sha256"],
        "noise_source_file_sha256": preflight["noise_source_file_sha256"],
        "victim_runner_sha256": sha256_file(Path(__file__)),
        "locked_training_helper_sha256": sha256_file(ROOT / "scripts" / "run_locked_clean_baseline.py"),
        "init_yaml": str(init_yaml), "init_yaml_sha256": sha256_file(init_yaml),
        "train_args": train_args, "started_at_unix": time.time(),
    }
    victim_spec_path = run_root / "victim_run_spec.json"
    victim_status_path = run_root / "victim_status.json"
    victim_spec_path.write_text(json.dumps(victim_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {"state": "running", "stage": "victim_training", "started_at_unix": time.time(),
              "run_id": args.run_id, "method": args.method}
    victim_status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["YOLO_CONFIG_DIR"] = str(run_root / ".ultralytics")
    os.environ["WANDB_MODE"] = "disabled"
    try:
        seed = int(victim["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from ultralytics import YOLO
        model = YOLO(str(init_yaml))
        model.train(**train_args)
        rows_path = run_root / "victim" / "results.csv"
        with rows_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != int(victim["epochs"]) or int(float(rows[-1]["epoch"])) != int(victim["epochs"]):
            raise RuntimeError(f"Victim incomplete: {len(rows)}/{victim['epochs']} epochs")
        best = run_root / "victim" / "weights" / "best.pt"
        last = run_root / "victim" / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Victim checkpoint is missing")
        metrics = YOLO(str(best)).val(
            data=str(yaml_path), imgsz=int(victim["imgsz"]), batch=int(victim["batch"]),
            workers=int(victim["workers"]), device=victim["device"], verbose=False,
            project=str(run_root), name="clean_val", exist_ok=False,
        )
        result = summarize_metrics(metrics.box)
        result.update({"run_id": args.run_id, "method": args.method,
                       "candidate_code_commit": args.code_commit,
                       "checkpoint": str(best), "checkpoint_sha256": sha256_file(best),
                       "validation_split": "clean VOC20 val copied byte-identically from locked source",
                       "protocol_lock_sha256": sha256_file(args.protocol_lock),
                       "dataset_fingerprint": source_summary["dataset_fingerprint"],
                       "materialization_preflight_sha256": sha256_file(run_root / "materialization" / "full_preflight.json"),
                       "victim_epochs": len(rows)})
        metrics_path = run_root / "clean_val_metrics.json"
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        status.update({"state": "complete", "epochs_completed": len(rows),
                       "clean_val_metrics": str(metrics_path), "completed_at_unix": time.time()})
    except BaseException as exc:
        status.update({"state": "failed", "error": repr(exc), "completed_at_unix": time.time()})
        raise
    finally:
        victim_status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
