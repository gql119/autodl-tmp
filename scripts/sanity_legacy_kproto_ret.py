"""Tiny generation-only probe. Never enters victim training or full materialization."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator, LegacyKProtoTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    method_cfg = cfg["methods"]["legacy_kproto_ret"]
    if not method_cfg["kproto"].get("max_images") or not method_cfg["kproto"].get("max_steps"):
        raise ValueError("Sanity config must bound max_images and max_steps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = LegacyKProtoTrainer(cfg, method_cfg, device, model)
    data_root = cfg["data"]["dataset_root"]
    image_dir = os.path.join(data_root, cfg["data"]["train_images"])
    label_dir = os.path.join(data_root, cfg["data"]["train_labels"])
    os.makedirs(args.output, exist_ok=True)
    params_path = os.path.join(args.output, "global_params.pt")
    csv_path = os.path.join(args.output, "diagnostics.csv")
    json_path = os.path.join(args.output, "diagnostics.json")
    images = trainer._collect_target_images(image_dir, label_dir)[:method_cfg["kproto"]["max_images"]]
    bank = trainer._build_bank(images, label_dir, os.path.join(args.output, "background_prototypes.pt"),
                               trainer._cache_key(image_dir, label_dir))
    first = images[0]
    image = load_image_rgb_float(first)
    annotations = read_yolo_annotations(label_path_for_image(first, label_dir))
    support, _, _ = trainer._build_support(image.shape, annotations, image_path=first)
    trainer._epoch = 0
    trainer._active_idx = torch.arange(trainer.shortcut_num_bases, device=device)
    trainer._active_coords = trainer.coords
    random.seed(17); torch.manual_seed(17)
    baseline = trainer._one_image_loss(image, annotations, support, bank)
    if baseline is None:
        raise RuntimeError("First sanity image lacks a valid target/context pair")
    before = {key: float(value.detach()) if torch.is_tensor(value) else value
              for key, value in baseline[1].items()}
    trainer.train_universal(image_dir, label_dir, params_path, csv_path, json_path, seed=0)
    trainer._epoch = 0
    trainer._active_idx = torch.arange(trainer.shortcut_num_bases, device=device)
    trainer._active_coords = trainer.coords
    random.seed(17); torch.manual_seed(17)
    updated = trainer._one_image_loss(image, annotations, support, bank)
    after = {key: float(value.detach()) if torch.is_tensor(value) else value
             for key, value in updated[1].items()}
    generator = LegacyKProtoGenerator(cfg, method_cfg, device, model, params_path)
    saved = []
    from PIL import Image
    for path in images:
        clean = load_image_rgb_float(path)
        anns = read_yolo_annotations(label_path_for_image(path, label_dir))
        result = generator.generate(clean, anns, 0, 2, trainer.eps, "mask", image_path=path)
        delta = result.perturbation
        output = os.path.join(args.output, os.path.basename(path).rsplit(".", 1)[0] + ".png")
        Image.fromarray(np.rint(np.clip(result.poisoned_image, 0, 1) * 255).astype(np.uint8)).save(output)
        original_label = label_path_for_image(path, label_dir)
        copied_label = os.path.join(args.output, os.path.basename(original_label))
        shutil.copyfile(original_label, copied_label)
        saved.append({"image": output, "linf": float(np.abs(delta).max()),
                      "outside_support_max": float(np.abs(delta * (result.support_mask == 0)[:, :, None]).max()),
                      "nonzero": bool(np.abs(delta).max() > 0),
                      "labels_unchanged": Path(original_label).read_bytes() == Path(copied_label).read_bytes()})
    report = {"before": before, "after": after, "samples": saved,
              "checkpoint": cfg["surrogate"]["ckpt"], "eps": trainer.eps}
    with open(os.path.join(args.output, "sanity_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
