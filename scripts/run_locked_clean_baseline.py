"""Train the matched, random-init clean VOC victim from protocol_lock.json.

All generated files, including Ultralytics label caches, stay below this
isolated project's output directory. The original dataset is read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_fingerprint(paths: list[Path]) -> str:
    # Exact lock algorithm: SHA256 of sorted absolute-path sha256sum lines.
    lines = [f"{sha256_file(path)}  {path.resolve()}\n" for path in sorted(paths)]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def source_inventory(lock: dict) -> tuple[dict, dict]:
    source = Path(lock["dataset"]["root"])
    result = {}
    for split in ("train", "val"):
        images = sorted(path for path in (source / "images" / split).iterdir() if path.is_file())
        labels = sorted(path for path in (source / "labels" / split).iterdir() if path.is_file())
        expected = int(lock["dataset"][f"{split}_images"])
        if len(images) != expected or len(labels) != int(lock["dataset"][f"{split}_labels"]):
            raise RuntimeError(f"Unexpected {split} source counts")
        if len({path.stem for path in images}) != expected or {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError(f"Unexpected {split} source stems")
        fingerprint = split_fingerprint(images + labels)
        if fingerprint != lock["dataset"][f"{split}_split_fingerprint"]:
            raise RuntimeError(f"{split} source fingerprint differs from protocol lock: {fingerprint}")
        result[split] = {"images": images, "labels": labels, "fingerprint": fingerprint}
    combined = hashlib.sha256((result["train"]["fingerprint"] + "\n" +
                               result["val"]["fingerprint"]).encode()).hexdigest()
    if combined != lock["dataset"]["fingerprint"]:
        raise RuntimeError("Combined dataset fingerprint differs from protocol lock")
    target_images = 0
    target_objects = 0
    for label in result["train"]["labels"]:
        objects = sum(line.split(maxsplit=1)[0] == "14" for line in label.read_text().splitlines() if line.strip())
        target_images += int(objects > 0)
        target_objects += objects
    if (target_images != lock["dataset"]["target_images"] or
            target_objects != lock["dataset"]["target_objects"]):
        raise RuntimeError("Person source counts differ from protocol lock")
    return result, {"train_images": len(result["train"]["images"]),
                    "val_images": len(result["val"]["images"]),
                    "target_images": target_images, "target_objects": target_objects,
                    "train_fingerprint": result["train"]["fingerprint"],
                    "val_fingerprint": result["val"]["fingerprint"],
                    "dataset_fingerprint": combined}


def stage_copy(index: dict, output: Path, *, smoke: bool) -> tuple[Path, int, int]:
    data = output / "clean_data"
    for split in ("train", "val"):
        images = index[split]["images"][:16 if split == "train" else 8] if smoke else index[split]["images"]
        labels = {path.stem: path for path in index[split]["labels"]}
        for kind in ("images", "labels"):
            (data / kind / split).mkdir(parents=True, exist_ok=False)
        for image in images:
            source_label = labels[image.stem]
            for source_file, kind in ((image, "images"), (source_label, "labels")):
                dest = data / kind / split / source_file.name
                shutil.copy2(source_file, dest)
                if sha256_file(source_file) != sha256_file(dest):
                    raise RuntimeError(f"Clean staged copy differs: {source_file}")
    return data, 16 if smoke else len(index["train"]["images"]), 8 if smoke else len(index["val"]["images"])


def data_yaml(data: Path, output: Path) -> Path:
    path = output / "clean_data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    path.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return path


def locked_train_args(lock: dict, output: Path, yaml: Path, *, smoke: bool) -> dict:
    victim = lock["victim"]
    args = dict(
        data=str(yaml), epochs=1 if smoke else int(victim["epochs"]),
        imgsz=int(victim["imgsz"]), batch=4 if smoke else int(victim["batch"]),
        workers=0 if smoke else int(victim["workers"]),
        project=str(output), name="victim", exist_ok=True,
        optimizer=victim["optimizer"], cos_lr=bool(victim["cos_lr"]),
        close_mosaic=0 if smoke else int(victim["close_mosaic"]),
        cache=False if smoke else bool(victim["cache"]),
        amp=bool(victim["amp"]), save_period=-1 if smoke else int(victim["save_period"]),
        device=victim["device"], lr0=float(victim["lr0"]),
        lrf=float(victim["lrf"]), momentum=float(victim["momentum"]),
        weight_decay=float(victim["weight_decay"]), seed=int(victim["seed"]),
        pretrained=bool(victim["pretrained"]),
        val=False if smoke else bool(victim["val_during_training"]),
    )
    reference = {"epochs": 200, "imgsz": 640, "batch": 36, "workers": 16,
                 "optimizer": "SGD", "cos_lr": True, "close_mosaic": 100,
                 "cache": True, "amp": True, "save_period": 10, "device": "0",
                 "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
                 "weight_decay": 0.0005, "seed": 0, "pretrained": False,
                 "val": True}
    if not smoke and any(args[key] != value for key, value in reference.items()):
        raise RuntimeError("Victim arguments differ from the verified B2-T run_spec")
    return args


def summarize_metrics(box) -> dict:
    classes = np.asarray(box.ap_class_index, dtype=int).reshape(-1).tolist()
    ap = np.asarray(box.ap50, dtype=float).reshape(-1)
    recall = np.asarray(box.r, dtype=float).reshape(-1)
    if (classes != list(range(20)) or len(ap) != 20 or len(recall) != 20 or
            not np.isfinite(ap).all() or not np.isfinite(recall).all() or
            np.any((recall < 0) | (recall > 1))):
        raise RuntimeError("Clean validation did not return all VOC20 classes")
    return {"mAP50_all": float(box.map50), "mAP50_target": float(ap[14]),
            "mAP50_non_target": float(np.mean(np.delete(ap, 14))),
            "AP50_per_class": ap.tolist(), "Recall_target": float(recall[14]),
            "Recall_per_class": recall.tolist()}


def check_installed_protocol(lock: dict) -> str:
    from ultralytics import __version__
    from ultralytics.cfg import DEFAULT_CFG_DICT
    if __version__ != lock["evaluation"]["ultralytics_version"]:
        raise RuntimeError(f"Ultralytics version differs from lock: {__version__}")
    for key, locked in lock["victim"]["augmentation"].items():
        if DEFAULT_CFG_DICT.get(key) != locked:
            raise RuntimeError(f"Default augmentation differs from protocol lock: {key}")
    for key in ("warmup_epochs", "warmup_momentum", "warmup_bias_lr", "deterministic"):
        locked = lock["victim"][key]
        if DEFAULT_CFG_DICT.get(key) != locked:
            raise RuntimeError(f"Default training setting differs from protocol lock: {key}")
    return __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / "protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    init_yaml = Path(lock["victim"]["init_yaml"])
    if sha256_file(init_yaml) != lock["victim"]["init_yaml_sha256"]:
        raise RuntimeError("Victim architecture YAML hash differs from protocol lock")
    index, inventory = source_inventory(lock)
    version = check_installed_protocol(lock)
    if args.inventory_only:
        print(json.dumps({"lock_sha256": sha256_file(lock_path),
                          "ultralytics_version": version, **inventory}, indent=2))
        return
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError(f"Output must be new and inside isolated baseline project: {output}")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
    os.environ["WANDB_MODE"] = "disabled"
    status_path = output / "status.json"
    status = {"state": "running", "smoke": args.smoke, "started_at_unix": time.time()}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        data, train_count, val_count = stage_copy(index, output, smoke=args.smoke)
        yaml = data_yaml(data, output)
        train_args = locked_train_args(lock, output, yaml, smoke=args.smoke)
        (output / "run_spec.json").write_text(json.dumps({
            "protocol_lock_sha256": sha256_file(lock_path),
            "dataset_fingerprint": inventory["dataset_fingerprint"],
            "train_count": train_count, "val_count": val_count,
            "init_yaml": str(init_yaml), "init_yaml_sha256": sha256_file(init_yaml),
            "train_args": train_args,
        }, indent=2), encoding="utf-8")
        random.seed(int(lock["victim"]["seed"]))
        np.random.seed(int(lock["victim"]["seed"]))
        torch.manual_seed(int(lock["victim"]["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(lock["victim"]["seed"]))
        from ultralytics import YOLO
        model = YOLO(str(init_yaml))
        model.train(**train_args)
        if args.smoke:
            status.update(state="complete", official_metrics_disabled=True,
                          artifact_ingest_disabled=True, completed_at_unix=time.time())
            return
        rows_path = output / "victim" / "results.csv"
        with rows_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 200 or int(float(rows[-1]["epoch"])) != 200:
            raise RuntimeError(f"Clean victim did not complete 200 epochs: {len(rows)}")
        best = output / "victim" / "weights" / "best.pt"
        last = output / "victim" / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Clean victim checkpoint missing")
        metrics = YOLO(str(best)).val(
            data=str(yaml), imgsz=int(lock["victim"]["imgsz"]),
            batch=int(lock["victim"]["batch"]), workers=int(lock["victim"]["workers"]),
            device="0", verbose=False, project=str(output), name="clean_val", exist_ok=True,
        )
        result = summarize_metrics(metrics.box)
        result.update(checkpoint=str(best), checkpoint_sha256=sha256_file(best),
                      validation_split="copied clean VOC val, verified byte-identical")
        (output / "clean_val_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        status.update(state="complete", epochs_completed=200,
                      clean_val_metrics=str(output / "clean_val_metrics.json"),
                      completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
