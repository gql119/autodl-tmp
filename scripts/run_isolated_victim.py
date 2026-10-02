"""Train a fresh VOC20 victim using only an isolated dataset and output root."""

import argparse
import csv
import filecmp
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import cv2

from ue_framework.config import load_config
from ue_framework.methods.semantic_coordinates import semantic_identity
from ue_framework.data_utils import image_has_target, list_images, read_yolo_annotations
from ue_framework.metrics_utils import compute_non_target_map
from ue_framework.methods.legacy_kproto_ret import _rectangle
from ue_framework.paths import build_run_paths


ROOT = Path(__file__).resolve().parent.parent
VOC_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def _summarize_clean_box_metrics(box, target_id=14):
    class_ids = np.asarray(box.ap_class_index, dtype=int).reshape(-1).tolist()
    ap = np.asarray(box.ap50, dtype=float).reshape(-1)
    recall = np.asarray(box.r, dtype=float).reshape(-1)
    if (class_ids != list(range(len(VOC_NAMES))) or len(ap) != len(VOC_NAMES)
            or len(recall) != len(VOC_NAMES) or not np.isfinite(ap).all()
            or not np.isfinite(recall).all() or np.any((recall < 0) | (recall > 1))):
        raise RuntimeError(f"Incomplete or unordered per-class AP50/Recall: class_ids={class_ids}")
    return {"mAP50_all": float(box.map50), "mAP50_target": float(ap[target_id]),
            "mAP50_non_target": float(compute_non_target_map(ap.tolist(), target_id)),
            "AP50_per_class": ap.tolist(), "Recall_target": float(recall[target_id]),
            "Recall_per_class": recall.tolist()}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within_root(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing output outside isolated project: {resolved}")
    return resolved


def _copy_split(source_images, source_labels, dest, limit=None):
    images = list_images(str(source_images))
    if limit is not None:
        images = images[:limit]
    image_dest = dest / "images" / "val"
    label_dest = dest / "labels" / "val"
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)
    for image in images:
        src = Path(image)
        label = source_labels / (src.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        dst_image = image_dest / src.name
        dst_label = label_dest / label.name
        if not dst_image.exists():
            shutil.copy2(src, dst_image)
        if not dst_label.exists():
            shutil.copy2(label, dst_label)
        if not filecmp.cmp(src, dst_image, shallow=False) or not filecmp.cmp(label, dst_label, shallow=False):
            raise RuntimeError(f"Clean validation copy differs from original: {src.stem}")
    return len(images)


def _smoke_data(cfg, output):
    dataset = Path(cfg["data"]["dataset_root"])
    root = output / "smoke_data"
    train_images = list_images(str(dataset / cfg["data"]["train_images"]))[:16]
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for image in train_images:
        src = Path(image)
        shutil.copy2(src, root / "images" / "train" / src.name)
        shutil.copy2(dataset / cfg["data"]["train_labels"] / (src.stem + ".txt"),
                     root / "labels" / "train" / (src.stem + ".txt"))
    val_count = _copy_split(dataset / cfg["data"]["val_images"],
                            dataset / cfg["data"]["val_labels"], root, limit=8)
    return root, len(train_images), val_count


def _bbox_masks(shape, annotations):
    height, width = shape[:2]
    target = np.zeros((height, width), dtype=bool)
    other = np.zeros((height, width), dtype=bool)
    for ann in annotations:
        if ann.get("bbox") is None:
            continue
        box = _rectangle((height, width), ann["bbox"])[0, 0].numpy().astype(bool)
        if int(ann["cls"]) == 14:
            target |= box
        else:
            other |= box
    return target, other


def _budget_image_stats(clean_pixels, poison_pixels, annotations, eps, overlap_scale):
    clean_i = clean_pixels.astype(np.int16)
    poison_i = poison_pixels.astype(np.int16)
    delta = np.abs(clean_i - poison_i)
    per_pixel = delta.max(axis=2)
    target, other = _bbox_masks(clean_pixels.shape, annotations)
    regions = {"target_non_overlap": target & ~other, "overlap": target & other,
               "outside_target": ~target}
    target_limit = int(round(float(eps) * 255))
    overlap_limit = int(round(float(eps) * float(overlap_scale) * 255))
    limits = {"target_non_overlap": target_limit, "overlap": overlap_limit}
    maxima = {"global": int(per_pixel.max(initial=0))}
    for name, mask in regions.items():
        maxima[name] = int(per_pixel[mask].max(initial=0))
    if maxima["global"] > target_limit:
        raise RuntimeError(f"Global decoded-pixel budget exceeded: {maxima['global']} > {target_limit}")
    for name, limit in limits.items():
        if maxima[name] > limit:
            raise RuntimeError(f"{name} decoded-pixel budget exceeded: {maxima[name]} > {limit}")
    if maxima["outside_target"] != 0:
        raise RuntimeError(f"Decoded perturbation outside target boxes: {maxima['outside_target']}")
    if np.any(per_pixel[target] == 0):
        raise RuntimeError("Unchanged pixel inside a target rectangle")
    return delta, per_pixel, regions, limits, maxima


def _new_budget_accumulator():
    return {"max_abs_delta_byte": {"global": 0, "target_non_overlap": 0,
                                   "overlap": 0, "outside_target": 0},
            "sum_abs_delta_byte": {"global": 0, "target_non_overlap": 0, "overlap": 0},
            "channel_values": {"global": 0, "target_non_overlap": 0, "overlap": 0},
            "saturated_pixels": {"target_non_overlap": 0, "overlap": 0},
            "region_pixels": {"target_non_overlap": 0, "overlap": 0},
            "perturbed_pixels": 0, "all_pixels": 0, "l2_squared_byte": 0}


def _accumulate_budget(acc, delta, per_pixel, regions, limits):
    acc["all_pixels"] += int(per_pixel.size)
    acc["perturbed_pixels"] += int(np.count_nonzero(per_pixel))
    acc["l2_squared_byte"] += float(np.square(delta.astype(np.float64)).sum())
    for name in ("global", "target_non_overlap", "overlap"):
        values = delta if name == "global" else delta[regions[name]]
        if values.size:
            acc["max_abs_delta_byte"][name] = max(
                acc["max_abs_delta_byte"][name], int(values.max()))
            acc["sum_abs_delta_byte"][name] += int(values.astype(np.int64).sum())
            acc["channel_values"][name] += int(values.size)
    acc["max_abs_delta_byte"]["outside_target"] = max(
        acc["max_abs_delta_byte"]["outside_target"],
        int(per_pixel[regions["outside_target"]].max(initial=0)))
    for name in ("target_non_overlap", "overlap"):
        mask = regions[name]
        acc["region_pixels"][name] += int(mask.sum())
        acc["saturated_pixels"][name] += int(np.count_nonzero(per_pixel[mask] >= limits[name]))


def _finish_budget_accumulator(acc):
    result = {"max_abs_delta_byte": acc["max_abs_delta_byte"],
              "max_abs_delta": {k: v / 255.0 for k, v in acc["max_abs_delta_byte"].items()},
              "mean_abs_delta": {}, "saturation_ratio": {},
              "perturbed_pixel_ratio": (acc["perturbed_pixels"] / acc["all_pixels"]
                                         if acc["all_pixels"] else 0.0),
              "perturbed_pixels": acc["perturbed_pixels"], "all_pixels": acc["all_pixels"],
              "l2_norm_byte": float(np.sqrt(acc["l2_squared_byte"]))}
    for name in ("global", "target_non_overlap", "overlap"):
        count = acc["channel_values"][name]
        result["mean_abs_delta"][name] = (
            acc["sum_abs_delta_byte"][name] / count / 255.0 if count else None)
    for name in ("target_non_overlap", "overlap"):
        count = acc["region_pixels"][name]
        result["saturation_ratio"][name] = (
            acc["saturated_pixels"][name] / count if count else None)
    result["region_pixels"] = acc["region_pixels"]
    return result


def _full_data(cfg, seed, noise_dir, config_path, budget_reference_root=None):
    if seed != 0:
        raise ValueError("Only the completed seed-0 noise run is available")
    dataset = Path(cfg["data"]["dataset_root"])
    epochs = int(cfg["methods"]["legacy_kproto_ret"]["universal_epochs"])
    if cfg["experiment"]["steps"] != [epochs]:
        raise ValueError("B2 steps and universal_epochs must match")
    paths = build_run_paths(cfg["platform"]["run_root"], "legacy_kproto_ret", epochs, seed)
    root = _within_root(paths.poisoned_root)
    noise_status = noise_dir / "status.json"
    with open(noise_status, encoding="utf-8") as handle:
        noise = json.load(handle)
    if noise["state"] != "complete" or noise.get("smoke"):
        raise RuntimeError("Official noise optimization is incomplete")
    if _sha256(config_path) != noise["config_sha256"]:
        raise RuntimeError("Victim config differs from the optimized-noise config")
    if semantic_identity(cfg, ROOT) != noise.get("semantic_identity"):
        raise RuntimeError("Victim semantic code/artifact identity differs from noise")
    adaptive_enabled = bool(cfg["methods"]["legacy_kproto_ret"].get("adaptive", {}).get("enabled", False))
    if adaptive_enabled != bool(noise.get("adaptive_enabled", False)):
        raise RuntimeError("Victim adaptive method differs from optimized noise")
    if adaptive_enabled:
        for name, key in (("adaptive_learner.py", "adaptive_code_sha256"),
                          ("b2_adaptive.py", "adaptive_trainer_sha256")):
            if noise.get(key) != _sha256(ROOT / "ue_framework" / "methods" / name):
                raise RuntimeError(f"Victim adaptive code differs: {name}")
    with open(root / "materialization_identity.json", encoding="utf-8") as handle:
        identity = json.load(handle)
    noise_params = noise_dir / "global_params.pt"
    staged_params = Path(paths.noise_dir) / "global_params.pt"
    if (identity["noise_run"] != str(noise_status.parent)
            or identity["noise_global_params_sha256"] != _sha256(noise_params)
            or identity["noise_global_params_sha256"] != _sha256(staged_params)
            or identity["noise_config_sha256"] != noise["config_sha256"]
            or identity["noise_method_sha256"] != noise["code_sha256"]
            or identity["noise_support_parent_sha256"] != noise["support_parent_sha256"]
            or identity["noise_support_parent_sha256"] != _sha256(ROOT / "ue_framework" / "methods" / "tausb_universal.py")
            or identity.get("noise_adaptive_code_sha256") != noise.get("adaptive_code_sha256")
            or identity.get("noise_adaptive_trainer_sha256") != noise.get("adaptive_trainer_sha256")
            or identity.get("semantic_identity") != noise.get("semantic_identity")
            or identity["seed"] != seed
            or identity["generator_stage_sha256"] != _sha256(ROOT / "ue_framework" / "stages" / "generate.py")):
        raise RuntimeError("Materialized dataset does not match this noise run and generation code")
    with open(paths.poisoned_status_json, encoding="utf-8") as handle:
        generated = json.load(handle)
    if generated["stage_state"]["generate_poisoned_dataset"]["status"] != "completed":
        raise RuntimeError("Poisoned dataset materialization is incomplete")
    with open(paths.manifest_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    originals = list_images(str(dataset / cfg["data"]["train_images"]))
    source_by_stem = {Path(image).stem: Path(image) for image in originals}
    expected = len(originals)
    if len(source_by_stem) != expected:
        raise RuntimeError("Original training image stems are not unique")
    manifest_stems = {r["stem"] for r in rows}
    poisoned_images = list_images(str(root / "images" / "train"))
    poisoned_stems = {Path(image).stem for image in poisoned_images}
    poisoned_by_stem = {Path(image).stem: Path(image) for image in poisoned_images}
    source_stems_sha256 = hashlib.sha256("\n".join(sorted(source_by_stem)).encode("utf-8")).hexdigest()
    if (len(rows) != expected or manifest_stems != set(source_by_stem)
            or len(poisoned_images) != expected or poisoned_stems != set(source_by_stem)
            or identity["source_stems_sha256"] != source_stems_sha256
            or identity["source_image_count"] != expected):
        raise RuntimeError(f"Poisoned manifest incomplete: {len(rows)} rows, expected {expected}")
    eps = float(cfg["experiment"]["eps"])
    method_cfg = cfg["methods"]["legacy_kproto_ret"]
    overlap_scale = float(method_cfg.get("overlap_scale", 0.5))
    if not 0.0 <= overlap_scale <= 1.0:
        raise ValueError(f"Invalid overlap_scale: {overlap_scale}")
    target_byte_limit = int(round(eps * 255))
    overlap_byte_limit = int(round(eps * overlap_scale * 255))
    if abs(eps * 255 - target_byte_limit) > 1e-4 or abs(
            eps * overlap_scale * 255 - overlap_byte_limit) > 1e-4:
        raise ValueError("This decoded-PNG audit requires integer-valued channel budgets")
    target_stems = []
    for row in rows:
        image = Path(row["image_path"])
        if (image.resolve().parent != (root / "images" / "train").resolve()
                or image.stem != row["stem"]):
            raise RuntimeError(f"Manifest image path is outside poisoned train split: {row['stem']}")
        label = root / "labels" / "train" / (row["stem"] + ".txt")
        source_image = source_by_stem.get(row["stem"])
        if source_image is None:
            raise RuntimeError(f"Original image identity is missing: {row['stem']}")
        source_label = dataset / cfg["data"]["train_labels"] / (row["stem"] + ".txt")
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"Missing poisoned image or label: {row['stem']}")
        if not source_label.is_file() or not filecmp.cmp(source_label, label, shallow=False):
            raise RuntimeError(f"Training label changed or is missing: {row['stem']}")
        annotations = read_yolo_annotations(str(source_label))
        has_target = image_has_target(annotations, 14)
        if row["has_target"] != str(int(has_target)):
            raise RuntimeError(f"Manifest target label disagrees with source: {row['stem']}")
        clean_pixels = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        poison_pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if clean_pixels is None or poison_pixels is None or clean_pixels.shape != poison_pixels.shape:
            raise RuntimeError(f"Image cannot be compared: {row['stem']}")
        delta, per_pixel, regions, limits, maxima = _budget_image_stats(
            clean_pixels, poison_pixels, annotations, eps, overlap_scale)
        pixel_linf = maxima["global"]
        if row["has_target"] == "0" and pixel_linf != 0:
            raise RuntimeError(f"Clean non-target-only image changed: {row['stem']}, {pixel_linf}")
        if cfg["methods"]["legacy_kproto_ret"].get("support_mode") == "bbox_half_overlap":
            if has_target and row["is_poisoned"] != "1":
                raise RuntimeError(f"Target image not marked poisoned: {row['stem']}")
        if float(row["linf"]) > eps + 1e-6:
            raise RuntimeError(f"Perturbation exceeds epsilon: {row['stem']}")
        if row["has_target"] == "0" and row["is_poisoned"] != "0":
            raise RuntimeError(f"Non-target-only image was poisoned: {row['stem']}")
        if has_target:
            target_stems.append(row["stem"])

    # Deterministic random audit over decoded, materialized PNGs. This happens
    # before the caller can construct or train the victim model.
    sample_count = 256
    if len(target_stems) < sample_count:
        raise RuntimeError(f"Need at least {sample_count} person images for the pixel audit")
    sample_stems = sorted(random.Random(seed).sample(sorted(target_stems), sample_count))
    new_acc = _new_budget_accumulator()
    old_acc = _new_budget_accumulator() if budget_reference_root else None
    reference_root = Path(budget_reference_root).resolve() if budget_reference_root else None
    if reference_root and not reference_root.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Budget reference root outside isolated project: {reference_root}")
    reference_by_stem = ({Path(path).stem: Path(path) for path in
                          list_images(str(reference_root / "images" / "train"))}
                         if reference_root else {})
    for stem in sample_stems:
        source_image = source_by_stem[stem]
        source_label = dataset / cfg["data"]["train_labels"] / (stem + ".txt")
        annotations = read_yolo_annotations(str(source_label))
        clean_pixels = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        current_image = poisoned_by_stem.get(stem)
        if current_image is None:
            raise RuntimeError(f"Materialized image missing from stem index: {stem}")
        poison_pixels = cv2.imread(str(current_image), cv2.IMREAD_COLOR)
        if clean_pixels is None or poison_pixels is None:
            raise RuntimeError(f"Cannot decode sampled budget audit image: {stem}")
        delta, per_pixel, regions, limits, maxima = _budget_image_stats(
            clean_pixels, poison_pixels, annotations, eps, overlap_scale)
        _accumulate_budget(new_acc, delta, per_pixel, regions, limits)
        if reference_root:
            old_image = reference_by_stem.get(stem)
            old_label = reference_root / "labels" / "train" / (stem + ".txt")
            if (old_image is None or not old_image.is_file() or not old_label.is_file()
                    or not filecmp.cmp(source_label, old_label, shallow=False)):
                raise RuntimeError(f"B2-TO paired audit reference missing/different labels: {stem}")
            old_pixels = cv2.imread(str(old_image), cv2.IMREAD_COLOR)
            if old_pixels is None or old_pixels.shape != clean_pixels.shape:
                raise RuntimeError(f"Cannot decode paired B2-TO audit image: {stem}")
            old_delta, old_per_pixel, old_regions, old_limits, _ = _budget_image_stats(
                clean_pixels, old_pixels, annotations, 16 / 255, 0.5)
            _accumulate_budget(old_acc, old_delta, old_per_pixel, old_regions, old_limits)
    new_stats = _finish_budget_accumulator(new_acc)
    audit = {"sample_count": sample_count, "sample_seed": seed,
             "sample_stems_sha256": hashlib.sha256("\n".join(sample_stems).encode()).hexdigest(),
             "sample_stems": sample_stems, "eps_target": eps,
             "eps_overlap": eps * overlap_scale, "overlap_scale": overlap_scale,
             "budget_units": "decoded uint8 channel differences; normalized values divide by 255",
             "saturation_definition": "fraction of spatial region pixels whose max absolute RGB-channel delta reaches that region's integer budget",
             "perturbed_pixel_ratio_definition": "changed spatial pixels / all spatial pixels in the sampled images",
             "max_abs_delta_global": new_stats["max_abs_delta"]["global"],
             "max_abs_delta_target_non_overlap": new_stats["max_abs_delta"]["target_non_overlap"],
             "max_abs_delta_overlap": new_stats["max_abs_delta"]["overlap"],
             "max_abs_delta_outside_target": new_stats["max_abs_delta"]["outside_target"],
             "mean_abs_delta_target_non_overlap": new_stats["mean_abs_delta"]["target_non_overlap"],
             "mean_abs_delta_overlap": new_stats["mean_abs_delta"]["overlap"],
             "saturation_ratio_target": new_stats["saturation_ratio"]["target_non_overlap"],
             "saturation_ratio_overlap": new_stats["saturation_ratio"]["overlap"],
             **new_stats}
    if old_acc:
        old_stats = _finish_budget_accumulator(old_acc)
        comparison = {"original_B2_TO_eps_target": 16 / 255,
                      "new_eps_target": eps,
                      "original_B2_TO_eps_overlap": 8 / 255,
                      "new_eps_overlap": eps * overlap_scale,
                      "sample_count": sample_count,
                      "sample_seed": seed,
                      "sample_stems_sha256": audit["sample_stems_sha256"],
                      "B2_TO": old_stats, "B2_TO_E8": new_stats,
                      "mean_abs_delta_ratio_E8_over_B2_TO": {
                          key: (new_stats["mean_abs_delta"][key] / old_stats["mean_abs_delta"][key]
                                if old_stats["mean_abs_delta"][key] else None)
                          for key in ("global", "target_non_overlap", "overlap")},
                      "L2_perturbation_ratio_E8_over_B2_TO": (
                          new_stats["l2_norm_byte"] / old_stats["l2_norm_byte"]
                          if old_stats["l2_norm_byte"] else None),
                      "saturation_ratio_E8_over_B2_TO": {
                          key: (new_stats["saturation_ratio"][key] / old_stats["saturation_ratio"][key]
                                if old_stats["saturation_ratio"][key] else None)
                          for key in ("target_non_overlap", "overlap")},
                      "perturbed_area_ratio_E8_over_B2_TO": (
                          new_stats["perturbed_pixel_ratio"] / old_stats["perturbed_pixel_ratio"]
                          if old_stats["perturbed_pixel_ratio"] else None)}
        (root / "perturbation_budget_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8")
    (root / "perturbation_budget_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    val_count = _copy_split(dataset / cfg["data"]["val_images"],
                            dataset / cfg["data"]["val_labels"], root)
    return root, expected, val_count


def _write_data_yaml(root, output):
    path = output / "train_data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    path.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
                    encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise", help="Completed official noise-run directory; required outside smoke mode")
    parser.add_argument("--budget-reference-poison-root",
                        help="Optional matched B2-TO materialized train root for paired E8 pixel statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.noise:
        parser.error("--noise is required for official victim training")
    noise_dir = _within_root(args.noise) if args.noise else None
    output = _within_root(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
    os.environ["WANDB_MODE"] = "disabled"
    cfg = load_config(args.config)
    status_path = output / "status.json"
    status = {"state": "running", "smoke": args.smoke, "seed": args.seed,
              "started_at_unix": time.time()}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        data_root, train_count, val_count = (
            _smoke_data(cfg, output) if args.smoke else _full_data(
                cfg, args.seed, noise_dir, args.config, args.budget_reference_poison_root)
        )
        data_yaml = _write_data_yaml(data_root, output)
        init_yaml = ROOT / cfg["victim"]["init"]
        if not init_yaml.is_file() or init_yaml.suffix != ".yaml":
            raise RuntimeError(f"Random-init model YAML is missing: {init_yaml}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        from ultralytics import YOLO
        victim = cfg["victim"]
        train_args = dict(
            data=str(data_yaml), epochs=1 if args.smoke else int(victim["epochs"]),
            imgsz=int(victim["imgsz"]), batch=4 if args.smoke else int(victim["batch"]),
            workers=0 if args.smoke else int(victim["workers"]),
            project=str(output), name="victim", exist_ok=True,
            optimizer=victim["optimizer"], cos_lr=bool(victim["cos_lr"]),
            close_mosaic=0 if args.smoke else int(victim["close_mosaic"]),
            cache=False if args.smoke else bool(victim["cache"]),
            amp=bool(victim["amp"]), save_period=-1 if args.smoke else int(victim["save_period"]),
            device="0", lr0=float(victim["lr0"]), lrf=float(victim["lrf"]),
            momentum=float(victim["momentum"]), weight_decay=float(victim["weight_decay"]),
            seed=args.seed, pretrained=False, val=not args.smoke,
        )
        spec = {"train_count": train_count, "val_count": val_count,
                "data_yaml": str(data_yaml), "init_yaml": str(init_yaml),
                "init_yaml_sha256": _sha256(init_yaml), "train_args": train_args,
                "noise_status": str(noise_dir / "status.json") if noise_dir else ""}
        (output / "run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        model = YOLO(str(init_yaml))
        model.train(**train_args)
        best = output / "victim" / "weights" / "best.pt"
        last = output / "victim" / "weights" / "last.pt"
        status.update(best_checkpoint=str(best), last_checkpoint=str(last))
        if not last.is_file():
            raise RuntimeError("Victim training finished without last.pt")
        if not args.smoke:
            checkpoint = best if best.is_file() else last
            metrics = YOLO(str(checkpoint)).val(data=str(data_yaml), imgsz=int(victim["imgsz"]),
                                                      batch=int(victim["batch"]), workers=int(victim["workers"]),
                                                      device="0", verbose=False,
                                                      project=str(output), name="clean_val", exist_ok=True)
            result = _summarize_clean_box_metrics(metrics.box)
            result.update(checkpoint=str(checkpoint),
                          validation_split="original clean VOC validation copied byte-for-byte")
            (output / "clean_val_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            status["clean_val_metrics"] = str(output / "clean_val_metrics.json")
        status.update(state="complete", completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
