"""Materialize and fully audit one generated detection-baseline dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ue_framework.baselines_det.core import TARGET_ID, _xyxy, decoded_pixel_audit
from ue_framework.baselines_det.locked_data import locked_source_inventory, sha256_file
from ue_framework.data_utils import read_yolo_annotations


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928"
)
NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def data_yaml(data: Path) -> Path:
    path = data.parent / "poisoned_data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    path.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return path


def _source_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode source image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def materialize(lock: dict, source: dict, noise_run: Path, output: Path,
                method: str, run_id: str, protocol_lock_path: Path) -> dict:
    noise_dir = noise_run / "noise"
    noise_status_path = noise_run / "status.json"
    noise_spec_path = noise_run / "run_spec.json"
    noise_status = json.loads(noise_status_path.read_text(encoding="utf-8"))
    noise_spec = json.loads(noise_spec_path.read_text(encoding="utf-8"))
    if noise_status.get("state") != "complete" or noise_status.get("completed_target_images") != int(lock["dataset"]["target_images"]):
        raise RuntimeError("Noise generation is not a complete, verified target-coverage run")
    if noise_spec.get("state") != "complete" or noise_spec.get("method") != method or noise_spec.get("run_id") != run_id:
        raise RuntimeError("Noise run_spec is missing, incomplete, or identifies a different run")
    if noise_status.get("dataset_fingerprint") != lock["dataset"]["fingerprint"]:
        raise RuntimeError("Noise run dataset fingerprint differs from protocol lock")
    if noise_status.get("protocol_lock_sha256") != sha256_file(protocol_lock_path):
        raise RuntimeError("Noise run was generated against a different protocol lock")
    if noise_status.get("method") != method or noise_status.get("run_id") != run_id:
        raise RuntimeError("Noise run identity does not match the requested method/RunID")

    data = output / "poisoned_data"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (data / kind / split).mkdir(parents=True, exist_ok=False)

    labels_by_split = {
        split: {path.stem: path for path in source[split]["labels"]}
        for split in ("train", "val")
    }
    target_stems: set[str] = set()
    target_objects = 0
    target_noise_stems = {path.stem for path in noise_dir.glob("*.png")}
    expected_noise_stems: set[str] = set()
    changed_images = changed_boxes = 0
    max_target = max_overlap = outside_changed = 0
    manifest_path = output / "materialization_manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as manifest:
        for image_path in source["train"]["images"]:
            label_path = labels_by_split["train"][image_path.stem]
            annotations = read_yolo_annotations(str(label_path))
            person_boxes = [ann for ann in annotations if int(ann["cls"]) == TARGET_ID]
            label_out = data / "labels" / "train" / label_path.name
            shutil.copy2(label_path, label_out)
            if sha256_file(label_path) != sha256_file(label_out):
                raise RuntimeError(f"Train label changed while copying: {image_path.stem}")
            image_out_dir = data / "images" / "train"
            if person_boxes:
                target_stems.add(image_path.stem)
                expected_noise_stems.add(image_path.stem)
                target_objects += len(person_boxes)
                poison_path = noise_dir / f"{image_path.stem}.png"
                if not poison_path.is_file():
                    raise RuntimeError(f"Missing generated target image noise: {image_path.stem}")
                clean_rgb = _source_rgb(image_path)
                poison_rgb = np.asarray(Image.open(poison_path).convert("RGB"), dtype=np.uint8)
                if poison_rgb.shape != clean_rgb.shape:
                    raise RuntimeError(f"Target poison shape differs from source: {image_path.stem}")
                audit = decoded_pixel_audit(clean_rgb, poison_rgb, annotations)
                if audit["changed_target_pixels"] < 1:
                    raise RuntimeError(f"Target image is unchanged: {image_path.stem}")
                diff = np.abs(poison_rgb.astype(np.int16) - clean_rgb.astype(np.int16))
                for ann in person_boxes:
                    x1, y1, x2, y2 = _xyxy(ann["bbox"], clean_rgb.shape[0], clean_rgb.shape[1])
                    if x2 <= x1 or y2 <= y1 or not np.any(diff[y1:y2, x1:x2] > 0):
                        raise RuntimeError(f"Person GT box has no changed decoded pixels: {image_path.stem}")
                    changed_boxes += 1
                changed_images += 1
                maxima = audit["max_abs_delta_byte"]
                max_target = max(max_target, maxima["target_only"])
                max_overlap = max(max_overlap, maxima["overlap"])
                outside_changed += audit["changed_outside_pixels"]
                out_path = image_out_dir / f"{image_path.stem}.png"
                Image.fromarray(poison_rgb, mode="RGB").save(out_path, format="PNG", compress_level=1)
                output_hash = sha256_file(out_path)
                if not np.array_equal(np.asarray(Image.open(out_path).convert("RGB")), poison_rgb):
                    raise RuntimeError(f"Materialized target PNG does not decode identically: {image_path.stem}")
                manifest.write(json.dumps({
                    "split": "train", "stem": image_path.stem, "target": True,
                    "source_image": str(image_path), "source_image_sha256": sha256_file(image_path),
                    "output_image": str(out_path), "output_image_sha256": output_hash,
                    "output_label": str(label_out), "label_sha256": sha256_file(label_path),
                    "person_objects": len(person_boxes), "changed_person_gt_boxes": len(person_boxes),
                    "changed_target_pixels": audit["changed_target_pixels"],
                    "max_abs_delta_byte": maxima,
                }, ensure_ascii=False) + "\n")
            else:
                out_path = image_out_dir / image_path.name
                shutil.copy2(image_path, out_path)
                if sha256_file(image_path) != sha256_file(out_path):
                    raise RuntimeError(f"Non-target-only train image changed: {image_path.stem}")
                manifest.write(json.dumps({
                    "split": "train", "stem": image_path.stem, "target": False,
                    "source_image": str(image_path), "output_image": str(out_path),
                    "output_label": str(label_out),
                    "source_image_sha256": sha256_file(image_path),
                    "output_image_sha256": sha256_file(out_path),
                    "label_sha256": sha256_file(label_path),
                }, ensure_ascii=False) + "\n")

        if target_stems != expected_noise_stems or target_noise_stems != expected_noise_stems:
            raise RuntimeError("Noise PNG stem set does not exactly match target train images")
        if changed_images != int(lock["dataset"]["target_images"]):
            raise RuntimeError("Not every locked person train image changed")
        if target_objects != int(lock["dataset"]["target_objects"]):
            raise RuntimeError("Materialized target object count differs from lock")
        for image_path in source["val"]["images"]:
            label_path = labels_by_split["val"][image_path.stem]
            for source_file, kind in ((image_path, "images"), (label_path, "labels")):
                dest = data / kind / "val" / source_file.name
                shutil.copy2(source_file, dest)
                if sha256_file(source_file) != sha256_file(dest):
                    raise RuntimeError(f"Clean validation bytes changed: {source_file.name}")
            manifest.write(json.dumps({
                "split": "val", "stem": image_path.stem,
                "source_image": str(image_path),
                "output_image": str(data / "images" / "val" / image_path.name),
                "output_label": str(data / "labels" / "val" / label_path.name),
                "source_image_sha256": sha256_file(image_path),
                "output_image_sha256": sha256_file(data / "images" / "val" / image_path.name),
                "label_sha256": sha256_file(label_path),
            }, ensure_ascii=False) + "\n")

    summary = {
        "state": "passed", "run_id": run_id, "method": method,
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "dataset_fingerprint": lock["dataset"]["fingerprint"],
        "noise_run": str(noise_run), "noise_status_sha256": sha256_file(noise_status_path),
        "noise_run_spec_sha256": sha256_file(noise_spec_path),
        "noise_source_file_sha256": noise_spec.get("source_file_sha256"),
        "train_images": int(lock["dataset"]["train_images"]),
        "val_images": int(lock["dataset"]["val_images"]),
        "target_images": changed_images, "target_objects": target_objects,
        "changed_person_gt_boxes": changed_boxes,
        "max_abs_delta_byte": {"person_only": max_target, "person_overlap": max_overlap},
        "changed_outside_pixels": outside_changed,
        "label_bytes_identical": True, "non_target_train_bytes_identical": True,
        "clean_val_bytes_identical": True,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "materializer_sha256": sha256_file(Path(__file__)),
        "materialized_root": str(data), "completed_at_unix": time.time(),
    }
    summary["data_yaml"] = str(data_yaml(data))
    (output / "full_preflight.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("em_det", "rem_det", "entf_push_det", "sem_det"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--noise-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Materialization output must remain under {ALLOWED_ROOT}") from exc
    if output.exists():
        raise FileExistsError(f"Refusing materialization output collision: {output}")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    source, _ = locked_source_inventory(lock, args.dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = materialize(lock, source, args.noise_run.resolve(), output,
                             args.method, args.run_id, args.protocol_lock.resolve())
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as exc:
        (output / "materialization_failure.json").write_text(json.dumps({
            "state": "failed", "run_id": args.run_id, "method": args.method,
            "error": repr(exc), "failed_at_unix": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
