"""Strict pixel/label audit for the isolated B2-TO-Safe materialization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ue_framework.config import load_config
from ue_framework.data_utils import list_images, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle
from ue_framework.paths import build_run_paths


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(args.output).resolve()
    project = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(project):
        raise ValueError("Audit output must remain inside the isolated project")
    paths = build_run_paths(cfg["platform"]["run_root"], "legacy_kproto_ret",
                            int(cfg["methods"]["legacy_kproto_ret"]["universal_epochs"]), 0)
    poisoned = Path(paths.poisoned_root)
    dataset = Path(cfg["data"]["dataset_root"])
    clean_images = {Path(p).stem: Path(p) for p in list_images(str(dataset / cfg["data"]["train_images"]))}
    if len(clean_images) != 16551:
        raise RuntimeError(f"Expected 16551 source train images, found {len(clean_images)}")
    manifest_path = Path(paths.manifest_csv)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(clean_images) or {r["stem"] for r in rows} != set(clean_images):
        raise RuntimeError("Materialized train stems do not exactly match clean VOC train")
    global_cap = int(round(float(cfg["experiment"]["eps"]) * 255))
    # bbox_half_overlap uses exactly half of the target support strength.
    target_cap = global_cap
    overlap_cap = int(round(global_cap * 0.5))
    maxima = {"global": 0, "target_only": 0, "overlap": 0, "outside": 0}
    target_boxes = 0
    for index, row in enumerate(rows, 1):
        stem = row["stem"]
        source = clean_images[stem]
        label = dataset / cfg["data"]["train_labels"] / f"{stem}.txt"
        out_image = Path(row["image_path"])
        out_label = poisoned / "labels" / "train" / f"{stem}.txt"
        if out_image.resolve().parent != (poisoned / "images" / "train").resolve():
            raise RuntimeError(f"Materialized image escaped train output: {stem}")
        if not label.is_file() or not out_label.is_file() or sha256(label) != sha256(out_label):
            raise RuntimeError(f"Label bytes changed: {stem}")
        annotations = read_yolo_annotations(str(label))
        clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
        noisy = cv2.imread(str(out_image), cv2.IMREAD_COLOR)
        if clean is None or noisy is None or clean.shape != noisy.shape:
            raise RuntimeError(f"Image read/shape failure: {stem}")
        difference = np.abs(clean.astype(np.int16) - noisy.astype(np.int16)).max(axis=2)
        maxima["global"] = max(maxima["global"], int(difference.max(initial=0)))
        if int(difference.max(initial=0)) > global_cap:
            raise RuntimeError(f"Global decoded-pixel budget exceeded: {stem}")
        target = np.zeros(difference.shape, dtype=bool)
        non_target = np.zeros(difference.shape, dtype=bool)
        for ann in annotations:
            box = _rectangle(difference.shape, ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann["cls"]) == 14:
                target |= box
                if np.any(difference[box]):
                    target_boxes += 1
                else:
                    raise RuntimeError(f"Person box has no changed pixels: {stem}")
            else:
                non_target |= box
        if np.any(difference[~target]):
            raise RuntimeError(f"Decoded pixels changed outside person boxes: {stem}")
        target_only = target & ~non_target
        overlap = target & non_target
        target_max = int(difference[target_only].max(initial=0))
        overlap_max = int(difference[overlap].max(initial=0))
        maxima["target_only"] = max(maxima["target_only"], target_max)
        maxima["overlap"] = max(maxima["overlap"], overlap_max)
        if target_max > target_cap:
            raise RuntimeError(f"Person-only 8/255 budget exceeded: {stem} ({target_max})")
        if overlap_max > overlap_cap:
            raise RuntimeError(f"Person/non-target overlap 4/255 budget exceeded: {stem} ({overlap_max})")
        if not np.array_equal(difference[~target], np.zeros_like(difference[~target])):
            raise RuntimeError(f"Outside-target pixel difference is nonzero: {stem}")
    if len(list_images(str(poisoned / "images" / "train"))) != 16551:
        raise RuntimeError("Materialized image count changed during audit")
    receipt = {
        "state": "passed", "train_images": len(rows), "labels_byte_identical": True,
        "person_boxes_with_changed_pixels": target_boxes, "pixel_maxima": maxima,
        "global_cap": global_cap, "person_only_cap": target_cap, "overlap_cap": overlap_cap,
        "outside_person_max": 0, "noise_status_sha256": sha256(Path(args.noise) / "status.json"),
        "manifest_sha256": sha256(manifest_path),
    }
    output = root / "materialization_audit.json"
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
