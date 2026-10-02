"""Check every target training image with the production mask-support path."""

import argparse
import json
import os

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    root = data["dataset_root"]
    image_dir = os.path.join(root, data["train_images"])
    label_dir = os.path.join(root, data["train_labels"])
    method = cfg["methods"]["legacy_kproto_ret"]
    if not method["strict_instance_mask"] or data["allow_pseudo_mask_fallback"]:
        raise ValueError("Preflight requires strict pre-baked masks with fallback disabled")
    trainer = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
    trainer.target_class_id = int(cfg["experiment"]["target_class_id"])
    trainer.instance_mask_dir = data["instance_mask_dir"]
    trainer.use_prebaked_instance_mask = True
    trainer.strict_instance_mask = True
    paths = trainer._collect_target_images(image_dir, label_dir)
    counts = {"prebaked": 0, "prebaked_excluded": 0}
    minimum_support = None
    for path in paths:
        image = load_image_rgb_float(path)
        annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
        support, _, source = trainer._build_support(image.shape, annotations, image_path=path)
        if source not in counts:
            raise RuntimeError(f"Unexpected support source {source}: {path}")
        counts[source] += 1
        if source == "prebaked":
            area = int(support.sum())
            minimum_support = area if minimum_support is None else min(minimum_support, area)
    print(json.dumps({"target_images": len(paths), "support_sources": counts,
                      "minimum_usable_support_pixels": minimum_support,
                      "mask_dir": trainer.instance_mask_dir}))


if __name__ == "__main__":
    main()
