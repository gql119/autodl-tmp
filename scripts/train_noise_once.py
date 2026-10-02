"""Train the universal perturbation once, without victim training or full materialization."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator, LegacyKProtoTrainer
from ue_framework.methods.b2_adaptive import B2AdaptiveTrainer
from ue_framework.methods.b2_semantic import B2SemanticTrainer
from ue_framework.methods.semantic_coordinates import semantic_identity


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-max-images", type=int, default=0)
    parser.add_argument("--smoke-max-steps", type=int, default=0)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output}")
    cfg = load_config(args.config)
    method_cfg = cfg["methods"]["legacy_kproto_ret"]
    adaptive_enabled = bool(method_cfg.get("adaptive", {}).get("enabled", False))
    is_smoke = bool(args.smoke_max_images or args.smoke_max_steps)
    if is_smoke:
        if args.smoke_max_images < 1 or args.smoke_max_steps < 1:
            raise ValueError("Both smoke limits must be positive")
        method_cfg["kproto"]["max_images"] = args.smoke_max_images
        method_cfg["kproto"]["max_steps"] = args.smoke_max_steps
    elif method_cfg["kproto"].get("max_images") or method_cfg["kproto"].get("max_steps"):
        raise ValueError("Official noise training requires the unbounded method config")
    if args.seed not in cfg["experiment"]["seeds"]:
        raise ValueError("Seed is not in experiment.seeds")
    os.makedirs(args.output)
    shutil.copyfile(args.config, os.path.join(args.output, "source_config.yaml"))
    _write_json(os.path.join(args.output, "resolved_config.json"), cfg)
    status_path = os.path.join(args.output, "status.json")
    model_path = cfg["surrogate"]["ckpt"]
    data_root = cfg["data"]["dataset_root"]
    image_dir = os.path.join(data_root, cfg["data"]["train_images"])
    label_dir = os.path.join(data_root, cfg["data"]["train_labels"])
    code_path = os.path.join(os.path.dirname(__file__), "..", "ue_framework", "methods", "legacy_kproto_ret.py")
    support_parent_path = os.path.join(os.path.dirname(__file__), "..", "ue_framework", "methods", "tausb_universal.py")
    adaptive_path = os.path.join(os.path.dirname(__file__), "..", "ue_framework", "methods", "adaptive_learner.py")
    adaptive_trainer_path = os.path.join(os.path.dirname(__file__), "..", "ue_framework", "methods", "b2_adaptive.py")
    started = time.time()
    status = {
        "state": "running", "started_at_unix": started, "seed": args.seed,
        "smoke": is_smoke, "official_metrics_disabled": True,
        "artifact_ingest_disabled": True, "checkpoint_cleanup_completed": False,
        "deleted_paths": [], "code_sha256": _sha256(code_path),
        "support_parent_sha256": _sha256(support_parent_path),
        "adaptive_enabled": adaptive_enabled,
        "adaptive_code_sha256": _sha256(adaptive_path) if adaptive_enabled else None,
        "adaptive_trainer_sha256": _sha256(adaptive_trainer_path) if adaptive_enabled else None,
        "config_sha256": _sha256(args.config), "surrogate_sha256": _sha256(model_path),
        "output": args.output,
        "worst_class_logging": "provisional while running; recomputed from EOT-mean class conflicts on completion",
        "semantic_identity": semantic_identity(cfg, os.path.dirname(os.path.dirname(__file__))),
    }
    _write_json(status_path, status)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(model_path).model.to(device)
        if int(model.nc) != int(cfg["surrogate"]["num_classes"]):
            raise ValueError("Surrogate class count differs from VOC20 config")
        frozen_before = _model_state_sha256(model)
        if adaptive_enabled:
            adaptive_model = YOLO(model_path).model.to(device)
            trainer_class = B2SemanticTrainer if method_cfg.get("semantic", {}).get("enabled", False) else B2AdaptiveTrainer
            trainer = trainer_class(cfg, method_cfg, device, model, adaptive_model)
            adaptive_before = _model_state_sha256(adaptive_model)
        else:
            trainer = LegacyKProtoTrainer(cfg, method_cfg, device, model)
        params_path = os.path.join(args.output, "global_params.pt")
        csv_path = os.path.join(args.output, "diagnostics.csv")
        json_path = os.path.join(args.output, "diagnostics.json")
        initial_coeff = trainer.fourier_coeff.detach().clone() if is_smoke else None
        initial_suppress = trainer.suppress_small.detach().clone() if is_smoke else None
        trainer.train_universal(image_dir, label_dir, params_path, csv_path, json_path, args.seed)
        if is_smoke:
            from semantic_sanity_probe import fixed_pair
            fixed_pair(trainer, image_dir, label_dir, initial_coeff, initial_suppress, params_path, args.seed)
        frozen_after = _model_state_sha256(model)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen reference parameters or buffers changed")
        status.update(frozen_state_sha256_before=frozen_before, frozen_state_sha256_after=frozen_after)
        if adaptive_enabled:
            adaptive_after = _model_state_sha256(adaptive_model)
            if adaptive_after != adaptive_before:
                raise RuntimeError("Adaptive base model changed outside the virtual inner update")
            status.update(adaptive_state_sha256_before=adaptive_before,
                          adaptive_state_sha256_after=adaptive_after,
                          adaptive_batch_size=trainer.adaptive_batch_size)
        generator = LegacyKProtoGenerator(cfg, method_cfg, device, model, params_path)
        images = trainer._collect_target_images(image_dir, label_dir)[:8]
        preview_dir = os.path.join(args.output, "preview")
        os.makedirs(preview_dir)
        previews = []
        for path in images:
            clean = load_image_rgb_float(path)
            annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
            result = generator.generate(
                clean, annotations, args.seed, trainer.universal_epochs,
                trainer.eps, "mask", image_path=path,
            )
            delta = result.perturbation
            out_path = os.path.join(preview_dir, os.path.basename(path).rsplit(".", 1)[0] + ".png")
            Image.fromarray(np.rint(np.clip(result.poisoned_image, 0, 1) * 255).astype(np.uint8)).save(out_path)
            previews.append({"path": out_path, "linf": float(np.abs(delta).max()),
                             "outside_support_max": float(np.abs(delta * (result.support_mask == 0)[:, :, None]).max())})
        _write_json(os.path.join(args.output, "preview_manifest.json"), previews)
        with open(json_path, "r", encoding="utf-8") as handle:
            latest = json.load(handle)["latest"]
        status.update(state="complete", completed_at_unix=time.time(),
                      duration_seconds=time.time() - started, steps=latest["step"] + 1,
                      epochs_completed=latest["epoch"], latest=latest,
                      checkpoint_paths_remaining=[params_path,
                          os.path.join(args.output, "background_prototypes.pt")])
        _write_json(status_path, status)
        print(json.dumps({"state": "complete", "steps": status["steps"],
                          "epochs": status["epochs_completed"], "latest": latest}, ensure_ascii=False))
    except BaseException as error:
        status.update(state="failed", completed_at_unix=time.time(),
                      error=repr(error), traceback=traceback.format_exc())
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
