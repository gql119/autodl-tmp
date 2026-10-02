"""Five-image GPU sanity for one detection-adapted published baseline.

This never creates a full poison dataset or a victim. Each output path must
be new. Formal generation remains blocked by protocol_lock.json and the
user's section-33 review gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ue_framework.baselines_det.core import (
    EPS, decoded_pixel_audit, file_sha256, model_state_sha256,
    poison_image, support_tensor,
)
from ue_framework.baselines_det.em import EMSchedule, detection_training_error, em_cycle
from ue_framework.baselines_det.rem import REMSchedule, rem_update
from ue_framework.baselines_det.sem import SEMSchedule, sem_update
from ue_framework.baselines_det.entf import (
    EntFSchedule, PersonROIFeatures, clean_person_centroid, entf_push,
)
from ue_framework.data_utils import label_path_for_image, read_yolo_annotations


METHODS = ("em_det", "rem_det", "entf_det", "sem_det")


def _load(image_path: Path, label_dir: Path, device: torch.device):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {image_path}")
    clean_bytes = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    label_path = Path(label_path_for_image(str(image_path), str(label_dir)))
    label_bytes = label_path.read_bytes()
    anns = read_yolo_annotations(str(label_path))
    image = torch.from_numpy(clean_bytes.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0
    return image, clean_bytes, anns, label_path, label_bytes


def _choose_five(image_dir: Path, label_dir: Path) -> list[Path]:
    target_only, overlap = [], []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        annotations = read_yolo_annotations(label_path_for_image(str(path), str(label_dir)))
        if not any(int(a["cls"]) == 14 for a in annotations):
            continue
        if any(int(a["cls"]) != 14 for a in annotations):
            overlap.append(path)
        else:
            target_only.append(path)
        if len(target_only) >= 4 and len(overlap) >= 4:
            break
    # One image with another class is required, but actual box overlap is
    # checked in the report rather than assumed from class co-occurrence.
    selected = (target_only[:2] + overlap[:3])[:5]
    if len(selected) != 5:
        selected = (target_only + overlap)[:5]
    if len(selected) != 5:
        raise RuntimeError("Fewer than five target-containing train images")
    return selected


def _scalar_summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "raw_delta"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite mini sanity: {args.output}")
    if args.max_images != 5:
        raise ValueError("This pre-registered sanity uses exactly five train images")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if lock["target"]["class_id"] != 14 or lock["poison"]["epsilon"] != "8/255":
        raise RuntimeError("Protocol lock differs from mini sanity constants")
    if file_sha256(args.surrogate) != lock["surrogate"]["checkpoint_sha256"]:
        raise RuntimeError("Surrogate checkpoint hash differs from lock")
    image_dir = args.dataset_root / "images" / "train"
    label_dir = args.dataset_root / "labels" / "train"
    paths = _choose_five(image_dir, label_dir)
    args.output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Production-reaching mini sanity requires CUDA")
    torch.manual_seed(0)
    np.random.seed(0)
    model = YOLO(str(args.surrogate)).model.to(device)
    if int(model.nc) != 20:
        raise RuntimeError("Surrogate is not VOC20")
    # The common exported checkpoint was saved after a frozen-reference run.
    # EM/REM/SEM must explicitly re-enable trainable surrogate parameters.
    if args.method != "entf_det":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    start = time.monotonic()
    state_before = model_state_sha256(model)
    max_memory = 0
    report = {"method": args.method, "official_run": False,
              "source": "five training images only", "surrogate_sha256": file_sha256(args.surrogate),
              "labels_unchanged": True, "images": []}
    extractor = None
    centroid = None
    if args.method == "entf_det":
        extractor = PersonROIFeatures(model, imgsz=args.imgsz)
        samples = [(_load(path, label_dir, device)[0], _load(path, label_dir, device)[2]) for path in paths]
        centroid, count = clean_person_centroid(extractor, samples)
        report["centroid_train_person_object_count"] = count
        report["feature_identity"] = extractor.identity(*samples[0])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    try:
        for index, path in enumerate(paths):
            clean, clean_bytes, annotations, label_path, label_bytes = _load(path, label_dir, device)
            raw_delta = torch.zeros_like(clean)
            if args.method == "em_det":
                result = em_cycle(model, clean, annotations, raw_delta, optimizer,
                                  EMSchedule(train_steps_per_cycle=1, noise_steps_per_cycle=2),
                                  imgsz=args.imgsz)
                if index == 0:
                    report["training_detection_error_probe"] = detection_training_error(
                        model, [(poison_image(clean, result["raw_delta"],
                                              support_tensor(clean, annotations)).detach(), annotations)],
                        imgsz=args.imgsz,
                    )
            elif args.method == "rem_det":
                result = rem_update(model, clean, annotations, raw_delta, optimizer,
                                    REMSchedule(defender_steps=2, eot_samples=2,
                                                adversarial_steps=2),
                                    imgsz=args.imgsz, seed=index,
                                    diagnose_fixed=True)
            elif args.method == "sem_det":
                result = sem_update(model, clean, annotations, raw_delta, optimizer,
                                    SEMSchedule(defender_steps=2, stability_samples=2,
                                                surrogate_adversarial_steps=2),
                                    imgsz=args.imgsz, seed=index,
                                    diagnose_fixed=True)
            else:
                result = entf_push(extractor, clean, annotations, centroid,
                                   EntFSchedule(steps=3), seed=index)
            gradients = result["perturbation_gradient_norms"]
            if not gradients or not all(np.isfinite(g) and g > 0 for g in gradients):
                raise RuntimeError(f"Missing or invalid perturbation gradient: {gradients}")
            support = support_tensor(clean, annotations)
            poisoned = poison_image(clean, result["raw_delta"], support)
            png = np.rint(poisoned.detach().cpu()[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            out = args.output / f"{path.stem}.png"
            Image.fromarray(png).save(out)
            decoded = np.asarray(Image.open(out).convert("RGB"))
            audit = decoded_pixel_audit(clean_bytes, decoded, annotations)
            if audit["changed_target_pixels"] <= 0:
                raise RuntimeError("The target support did not change")
            if label_path.read_bytes() != label_bytes:
                raise RuntimeError("Source label bytes changed")
            report["images"].append({"stem": path.stem,
                                     "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                                     "decoded_pixel_audit": audit,
                                     "core": _scalar_summary(result)})
            max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))
    finally:
        if extractor is not None:
            extractor.close()
    state_after = model_state_sha256(model)
    report.update({"state": "passed", "model_state_before": state_before,
                   "model_state_after": state_after,
                   "surrogate_updated": state_before != state_after,
                   "surrogate_update_expected": args.method != "entf_det",
                   "duration_seconds": time.monotonic() - start,
                   "peak_gpu_memory_bytes": max_memory,
                   "epsilon": EPS, "overlap_epsilon": EPS / 2,
                   "outside_target_epsilon": 0})
    if report["surrogate_updated"] != report["surrogate_update_expected"]:
        raise RuntimeError("Surrogate dynamics differ from the method's original mechanism")
    out_json = args.output / f"sanity_{args.method}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": "passed", "method": args.method,
                      "images": len(report["images"]),
                      "output": str(out_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
