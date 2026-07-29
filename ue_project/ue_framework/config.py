import copy
import os
from typing import Any, Dict

import yaml

SUPPORTED_METHODS = ["em_bbox", "em_mask", "rem_mask", "tausb_mask", "ours_mask"]
SUPPORTED_STAGES = [
    "generate_poisoned_dataset",
    "train_victim",
    "evaluate",
    "aggregate",
    "all",
]



def _default_config() -> Dict[str, Any]:
    return {
        "experiment": {
            "dataset": "voc",
            "target_class_name": "person",
            "target_class_id": 14,
            "poisoning_ratio": 1.0,
            "eps": 16 / 255,
            "steps": [10, 20, 40],
            "seeds": [0, 1, 2],
            "num_classes": 20,
            "reference_target_map50": 0.81,
            "reference_non_target_map50": 0.78,
        },
        "data": {
            "dataset_root": "/kaggle/input/voc-0712-kaggle-ready/VOC_0712_Kaggle_Ready",
            "train_images": "images/train",
            "train_labels": "labels/train",
            "val_images": "images/val",
            "val_labels": "labels/val",
            "instance_mask_dir": "",
            "allow_pseudo_mask_fallback": True,
        },
        "platform": {
            "mode": "auto",
            "run_root": "/kaggle/working/ue_runs",
            "resume": True,
            "save_every_n_images": 50,
            "save_every_n_epochs": 1,
            "pack_every_n_epochs": 1,
            "zip_after_stage": True,
            "debug": False,
        },
        "surrogate": {
            "model": "yolov8n",
            "ckpt": "yolov8n.pt",
            "num_classes": 20,
            "imgsz": 640,
            "eot_samples": 2,
        },
        "victim": {
            "model": "yolov8n",
            "init": "yolov8n.yaml",
            "cache": False,
            "amp": True,
            "epochs": 250,
            "imgsz": 640,
            "batch": 8,
            "workers": 2,
            "optimizer": "SGD",
            "cos_lr": True,
            "close_mosaic": 10,
            "save_period": 1,
            "device": "0",
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
        },
        "methods": {
            "em_bbox": {
                "support_type": "bbox",
                "step_size": 2 / 255,
                "noise_scale": 1.0,
            },
            "em_mask": {
                "support_type": "mask",
                "step_size": 2 / 255,
                "noise_scale": 1.0,
            },
            "rem_mask": {
                "support_type": "mask",
                "step_size": 2 / 255,
                "noise_scale": 1.0,
                "eot_samples": 4,
            },
            "tausb_mask": {
                "support_type": "mask",
                "ring_width": 4,
                "shortcut_num_bases": 2,
                "carrier_basis_mode": "synthetic_fourier",
                "background_basis_path": "",
                "suppress_small_size": 32,
                "align_alpha": 0.5,
                "align_beta": 6.0,
                "assignment_topk": 100,
                "gamma_start": 0.05,
                "gamma_end": 0.30,
                "gamma_schedule": "linear",
                "lambda_induce": 4.0,
                "lambda_shape": 0.5,
                "lambda_preserve": 1.0,
                "lambda_cls_aux": 0.5,
                "lambda_tv": 0.001,
                "lambda_budget": 10.0,
                "universal_epochs": 5,
                "universal_batch_size": 8,
                "universal_lr_fourier": 0.03,
                "universal_lr_suppress": 0.01,
                "eot_samples": 4,
                "jnd_floor": 0.2,
                "jnd_ceiling": 1.0,
            },
            "ours_mask": {
                "support_type": "mask",
                "ring_width": 4,
                "enable_conditional_sps": True,
                "enable_midfreq_search": True,
                "enable_jnd_gain": True,
                "enable_assignment_shortcut": True,
                "enable_shape_entanglement": True,
                "shortcut_frequency_mode": "midfreq_sparse",
                "shortcut_num_bases": 2,
                "shortcut_ring_weight": 1.0,
                "suppress_inner_weight": 1.0,
                "suppress_ring_weight": 0.5,
                "suppress_step_size": 1 / 255,
                "assignment_topk": 100,
                "align_alpha": 1.0,
                "align_beta": 6.0,
                "assign_margin": 0.4,
                "lambda_align": 4.0,
                "lambda_rank": 2.0,
                "lambda_shape": 1.0,
                "lambda_preserve": 1.0,
                "lambda_suppress": 0.5,
                "lambda_jnd": 0.5,
                "lambda_tv": 0.001,
                "lambda_budget": 10.0,
                "jnd_floor": 0.2,
                "jnd_ceiling": 1.0,
                "eot_samples": 4,
            },
        },
    }



def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base



def _ensure_method_defaults(cfg: Dict[str, Any]) -> None:
    methods = cfg.get("methods", {})

    ours = methods.get("ours_mask", {})
    if isinstance(ours, dict):
        ours_required = {
            "lambda_align": 4.0,
            "lambda_rank": 2.0,
            "lambda_shape": 1.0,
            "lambda_preserve": 1.0,
            "lambda_suppress": 0.5,
            "align_alpha": 1.0,
            "align_beta": 6.0,
            "assign_margin": 0.4,
        }
        for k, v in ours_required.items():
            if k not in ours:
                ours[k] = v

    tausb = methods.get("tausb_mask", {})
    if isinstance(tausb, dict):
        tausb_required = {
            "support_type": "mask",
            "shortcut_num_bases": 2,
            "carrier_basis_mode": "synthetic_fourier",
            "background_basis_path": "",
            "suppress_small_size": 32,
            "align_alpha": 0.5,
            "align_beta": 6.0,
            "assignment_topk": 100,
            "gamma_start": 0.05,
            "gamma_end": 0.30,
            "gamma_schedule": "linear",
            "lambda_induce": 4.0,
            "lambda_shape": 0.5,
            "lambda_preserve": 1.0,
            "lambda_cls_aux": 0.5,
            "lambda_tv": 0.001,
            "lambda_budget": 10.0,
            "universal_epochs": 5,
            "universal_batch_size": 8,
            "universal_lr_fourier": 0.03,
            "universal_lr_suppress": 0.01,
            "eot_samples": 4,
            "jnd_floor": 0.2,
            "jnd_ceiling": 1.0,
        }
        for k, v in tausb_required.items():
            if k not in tausb:
                tausb[k] = v



def load_config(config_path: str) -> Dict[str, Any]:
    defaults = _default_config()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_update(copy.deepcopy(defaults), user_cfg)
    _ensure_method_defaults(cfg)

    target_id = cfg["experiment"]["target_class_id"]
    num_classes = cfg["surrogate"]["num_classes"]
    if target_id < 0 or target_id >= num_classes:
        raise ValueError(
            "Target class id is out of surrogate class range. "
            f"target_class_id={target_id}, surrogate.num_classes={num_classes}"
        )

    return cfg
