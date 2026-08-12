from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import atomic_write_text


SPARSE_MIXED_LIST_LAYOUT = "sparse_mixed_list_v1"
SPARSE_TRAIN_LIST_NAME = "train-images.txt"
SPARSE_REPORT_NAME = "sparse_materialization.json"


def is_sparse_mixed_list(cfg: Mapping[str, object]) -> bool:
    data = cfg.get("data", {})
    return isinstance(data, Mapping) and data.get("materialization_layout") == (
        SPARSE_MIXED_LIST_LAYOUT
    )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lines_sha256(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_train_path_list(path: str, image_paths: Sequence[str]) -> str:
    normalized = [os.path.abspath(value) for value in image_paths]
    atomic_write_text(path, "".join(value + "\n" for value in normalized))
    return canonical_lines_sha256(normalized)


def _rgb_float_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def png_roundtrip_exact(image: np.ndarray) -> bool:
    expected = _rgb_float_to_uint8(image)
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(expected, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not success:
        raise RuntimeError("Failed to encode the clean PNG round-trip probe.")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("Failed to decode the clean PNG round-trip probe.")
    decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return bool(np.array_equal(expected, decoded))


def save_png_and_reload(path: str, image: np.ndarray) -> Tuple[np.ndarray, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    expected = _rgb_float_to_uint8(image)
    ok = cv2.imwrite(
        path,
        cv2.cvtColor(expected, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise RuntimeError("Failed to save sparse poisoned PNG: %s" % path)
    decoded_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("Failed to reload sparse poisoned PNG: %s" % path)
    decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    if not np.array_equal(expected, decoded):
        raise RuntimeError("Saved sparse PNG does not round-trip losslessly: %s" % path)
    return decoded.astype(np.float32) / 255.0, file_sha256(path)


def yolo_label_path_for_image(image_path: str) -> str:
    normalized = os.path.normpath(os.path.abspath(image_path))
    marker = os.path.sep + "images" + os.path.sep
    if marker not in normalized:
        raise ValueError(
            "Sparse training image path lacks an /images/ component: %s" % image_path
        )
    prefix, suffix = normalized.rsplit(marker, 1)
    label_suffix = os.path.splitext(suffix)[0] + ".txt"
    return prefix + os.path.sep + "labels" + os.path.sep + label_suffix


def _validated_label_classes(label_path: str, num_classes: int) -> List[int]:
    if not os.path.isfile(label_path):
        raise FileNotFoundError("Sparse training label is missing: %s" % label_path)
    class_ids: List[int] = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(
                    "Invalid YOLO label field count at %s:%d." % (label_path, line_number)
                )
            values = np.asarray([float(value) for value in parts], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("Non-finite YOLO label at %s:%d." % (label_path, line_number))
            class_id = int(values[0])
            if values[0] != class_id or class_id < 0 or class_id >= num_classes:
                raise ValueError("YOLO class id is out of range at %s:%d." % (label_path, line_number))
            if np.any(values[1:] < 0.0) or np.any(values[1:] > 1.0):
                raise ValueError("YOLO box is outside [0,1] at %s:%d." % (label_path, line_number))
            class_ids.append(class_id)
    return class_ids


def audit_sparse_training_list(
    train_list_path: str,
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    expected_total: int,
    expected_poisoned: int,
    expected_target: int,
    target_class_id: int,
    num_classes: int,
    verify_hashes: bool = True,
) -> Dict[str, object]:
    if not os.path.isfile(train_list_path):
        raise FileNotFoundError("Sparse train path list is missing: %s" % train_list_path)
    with open(train_list_path, "r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle if line.strip()]
    if len(image_paths) != expected_total:
        raise ValueError(
            "Sparse train list count mismatch: actual=%d expected=%d."
            % (len(image_paths), expected_total)
        )
    if any(not os.path.isabs(path) for path in image_paths):
        raise ValueError("Sparse train list requires absolute image paths.")
    stems = [Path(path).stem for path in image_paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Sparse train list contains duplicate stems.")
    row_by_stem = {str(row.get("stem", "")): row for row in manifest_rows}
    if len(row_by_stem) != len(manifest_rows) or set(row_by_stem) != set(stems):
        raise ValueError("Sparse manifest stems do not match the train path list.")

    poisoned_count = 0
    target_count = 0
    original_jpeg_count = 0
    poisoned_png_count = 0
    label_hash_lines: List[str] = []
    verified_source_hashes: Dict[str, str] = {}
    verified_label_hashes: Dict[str, str] = {}
    verified_saved_hashes: Dict[str, str] = {}

    def cached_hash(cache: Dict[str, str], path: str) -> str:
        if path not in cache:
            cache[path] = file_sha256(path)
        return cache[path]

    for image_path, stem in zip(image_paths, stems):
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Sparse training image is missing: %s" % image_path)
        row = row_by_stem[stem]
        if os.path.abspath(str(row.get("image_path", ""))) != os.path.abspath(image_path):
            raise ValueError("Sparse manifest effective path mismatch for %s." % stem)
        inferred_label = yolo_label_path_for_image(image_path)
        recorded_label = os.path.abspath(str(row.get("label_path", "")))
        if recorded_label != os.path.abspath(inferred_label):
            raise ValueError("Sparse label inference mismatch for %s." % stem)
        class_ids = _validated_label_classes(inferred_label, num_classes)
        has_target = target_class_id in class_ids
        target_count += int(has_target)
        if bool(int(str(row.get("has_target", "0")))) is not has_target:
            raise ValueError("Sparse manifest target flag mismatch for %s." % stem)
        is_poisoned = bool(int(str(row.get("is_poisoned", "0"))))
        poisoned_count += int(is_poisoned)
        source_path = os.path.abspath(str(row.get("source_image_path", "")))
        if not os.path.isfile(source_path):
            raise FileNotFoundError("Sparse source image is missing for %s." % stem)
        recorded_source_hash = str(row.get("source_image_sha256", ""))
        source_hash = (
            cached_hash(verified_source_hashes, source_path)
            if verify_hashes
            else recorded_source_hash
        )
        if len(recorded_source_hash) != 64 or recorded_source_hash != source_hash:
            raise ValueError("Sparse source image hash mismatch for %s." % stem)
        recorded_label_hash = str(row.get("label_sha256", ""))
        label_sha256 = (
            cached_hash(verified_label_hashes, inferred_label)
            if verify_hashes
            else recorded_label_hash
        )
        if len(recorded_label_hash) != 64 or recorded_label_hash != label_sha256:
            raise ValueError("Sparse label hash mismatch for %s." % stem)
        if is_poisoned:
            if Path(image_path).suffix.lower() != ".png" or image_path == source_path:
                raise ValueError("Sparse poisoned row is not a distinct PNG for %s." % stem)
            recorded_saved_hash = str(row.get("saved_image_sha256", ""))
            saved_hash = (
                cached_hash(verified_saved_hashes, image_path)
                if verify_hashes
                else recorded_saved_hash
            )
            if len(recorded_saved_hash) != 64 or recorded_saved_hash != saved_hash:
                raise ValueError("Sparse poisoned PNG hash mismatch for %s." % stem)
            poisoned_png_count += 1
            outside = float(row.get("support_outside_linf", "nan"))
            if not np.isfinite(outside) or outside != 0.0:
                raise ValueError("Sparse perturbation escaped support for %s." % stem)
        else:
            if os.path.abspath(image_path) != source_path:
                raise ValueError("Sparse clean row does not reference its source for %s." % stem)
            if str(row.get("saved_image_sha256", "")):
                raise ValueError("Sparse clean row unexpectedly records a saved PNG for %s." % stem)
            if Path(image_path).suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError("Sparse clean row must reference an original JPEG for %s." % stem)
            original_jpeg_count += 1
        label_hash_lines.append(stem + ":" + label_sha256)

    if poisoned_count != expected_poisoned:
        raise ValueError("Sparse poisoned count mismatch.")
    if target_count != expected_target:
        raise ValueError("Sparse target-image count mismatch.")
    if poisoned_png_count != expected_poisoned:
        raise ValueError("Sparse poisoned PNG count mismatch.")
    if original_jpeg_count != expected_total - expected_poisoned:
        raise ValueError("Sparse original JPEG count mismatch.")
    return {
        "schema": "tausb.sdh-sparse-train-audit.v1",
        "train_list_path": os.path.abspath(train_list_path),
        "train_list_sha256": canonical_lines_sha256(image_paths),
        "ordered_stems_sha256": canonical_lines_sha256(stems),
        "label_content_manifest_sha256": canonical_lines_sha256(label_hash_lines),
        "total_count": len(image_paths),
        "target_count": target_count,
        "poisoned_count": poisoned_count,
        "poisoned_png_count": poisoned_png_count,
        "original_jpeg_count": original_jpeg_count,
    }
