"""Read-only source-dataset audit of the production rectangle support."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ue_framework.config import load_config
from ue_framework.data_utils import image_has_target, list_images, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = Path(cfg["data"]["dataset_root"])
    image_dir = dataset / cfg["data"]["train_images"]
    label_dir = dataset / cfg["data"]["train_labels"]
    method = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
    method.support_mode = "bbox_half_overlap"
    method.target_class_id = int(cfg["experiment"]["target_class_id"])
    count = 0
    target_pixels = overlap_pixels = total_pixels = 0
    images_with_overlap = 0
    min_target_pixels = None
    sources = {}
    for image_path in list_images(str(image_dir)):
        image = Path(image_path)
        annotations = read_yolo_annotations(str(label_dir / (image.stem + ".txt")))
        if not image_has_target(annotations, method.target_class_id):
            continue
        with Image.open(image) as source:
            width, height = source.size
        support, ring, name = method._build_support((height, width, 3), annotations, image_path=image_path)
        area = int(np.count_nonzero(support))
        overlap = int(np.count_nonzero(support == 0.5))
        if area == 0 or name != "bbox_half_overlap" or np.any(ring):
            raise RuntimeError(f"Invalid rectangle support for {image.stem}")
        count += 1
        target_pixels += area
        overlap_pixels += overlap
        total_pixels += height * width
        images_with_overlap += bool(overlap)
        min_target_pixels = area if min_target_pixels is None else min(min_target_pixels, area)
        sources[name] = sources.get(name, 0) + 1
    report = {
        "target_images": count,
        "target_pixels": target_pixels,
        "overlap_pixels": overlap_pixels,
        "images_with_overlap": images_with_overlap,
        "total_pixels_in_target_images": total_pixels,
        "min_target_pixels": min_target_pixels,
        "support_sources": sources,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
