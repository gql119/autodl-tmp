from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ue_framework.data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    read_yolo_annotations,
    stem_of,
)
from ue_framework.io_utils import atomic_write_csv, atomic_write_json
from ue_framework.sparse_dataset import (
    audit_sparse_training_list,
    file_sha256,
    write_train_path_list,
)


FIELDS = (
    "stem",
    "image_path",
    "source_image_path",
    "label_path",
    "source_image_sha256",
    "label_sha256",
    "saved_image_sha256",
    "is_poisoned",
    "has_target",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and audit a real full-VOC sparse C0 path list without GPU/model loading."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError("Local sparse audit output already exists: %s" % output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    image_dir = dataset_root / "images/train"
    label_dir = dataset_root / "labels/train"
    images = list_images(str(image_dir))
    rows = []
    target_count = 0
    for image_path in images:
        label_path = label_path_for_image(image_path, str(label_dir))
        if not os.path.isfile(label_path):
            raise FileNotFoundError("VOC label is missing: %s" % label_path)
        has_target = image_has_target(read_yolo_annotations(label_path), 14)
        target_count += int(has_target)
        rows.append(
            {
                "stem": stem_of(image_path),
                "image_path": os.path.abspath(image_path),
                "source_image_path": os.path.abspath(image_path),
                "label_path": os.path.abspath(label_path),
                "source_image_sha256": file_sha256(image_path),
                "label_sha256": file_sha256(label_path),
                "saved_image_sha256": "",
                "is_poisoned": "0",
                "has_target": "1" if has_target else "0",
            }
        )
    if len(rows) != 16551 or target_count != 6095:
        raise ValueError("Full VOC C0 count gate failed.")
    manifest = output_root / "manifest.csv"
    path_list = output_root / "train-images.txt"
    atomic_write_csv(str(manifest), rows, list(FIELDS))
    write_train_path_list(str(path_list), [row["image_path"] for row in rows])
    report = audit_sparse_training_list(
        str(path_list),
        rows,
        expected_total=16551,
        expected_poisoned=0,
        expected_target=6095,
        target_class_id=14,
        num_classes=20,
    )
    atomic_write_json(str(output_root / "audit.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
