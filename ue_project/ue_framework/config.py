import copy
import os
from typing import Any, Dict

import yaml

SUPPORTED_METHODS = [
    "em_bbox",
    "em_mask",
    "rem_mask",
    "tausb_mask",
    "ours_mask",
    "sirc_malc_cgr",
    "tausb_sdh",
]
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
            "sirc_malc_cgr": {
                "support_type": "mask",
                "enable_sirc_carrier": True,
                "enable_malc": True,
                "enable_cgr": True,
                "require_mechanism_pass": True,
                "frozen_carrier_state": "",
                "semantic_bank_hash": "",
                "source_manifest_hash": "",
                "split_hash": "",
                "variant_seed": 2102,
                "jnd_floor": 0.5,
            },
            "tausb_sdh": {
                "protocol_id": "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3",
                "materialization_mode": "formal_p1_state",
                "allow_failed_scientific_gates": False,
                "binding_status": "bound",
                "evidence_scope": "formal_method",
                "support_type": "bbox",
                "enable_deep_hiding": True,
                "enable_dlfc": True,
                "enable_cicr": True,
                "enable_cgr": True,
                "enable_nla_loss": True,
                "require_hiding_gate_pass": True,
                "require_mechanism_gate_pass": True,
                "frozen_sdh_state": "",
                "secret_source_sha256": "",
                "secret_tensor_sha256": "",
                "source_manifest_sha256": "",
                "train_split_sha256": "",
                "frozen_sdh_state_sha256": "",
                "hiding_metrics_sha256": "",
                "hiding_checkpoint_sha256": "",
                "hiding_split_sha256": "",
                "mechanism_metrics_sha256": "",
                "mechanism_decision_sha256": "",
                "mechanism_config_sha256": "",
                "p1_state_sha256": "",
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


def _is_sha256(value: object) -> bool:
    digest = str(value).strip().lower()
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)



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

    sirc_malc = methods.get("sirc_malc_cgr", {})
    if isinstance(sirc_malc, dict):
        required = {
            "support_type": "mask",
            "enable_sirc_carrier": True,
            "enable_malc": True,
            "enable_cgr": True,
            "require_mechanism_pass": True,
            "frozen_carrier_state": "",
            "semantic_bank_hash": "",
            "source_manifest_hash": "",
            "split_hash": "",
            "variant_seed": 2102,
            "jnd_floor": 0.5,
        }
        for key, value in required.items():
            if key not in sirc_malc:
                sirc_malc[key] = value

    sdh = methods.get("tausb_sdh", {})
    if isinstance(sdh, dict):
        required = {
            "protocol_id": "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3",
            "materialization_mode": "formal_p1_state",
            "allow_failed_scientific_gates": False,
            "binding_status": "bound",
            "evidence_scope": "formal_method",
            "support_type": "bbox",
            "enable_deep_hiding": True,
            "enable_dlfc": True,
            "enable_cicr": True,
            "enable_cgr": True,
            "enable_nla_loss": True,
            "require_hiding_gate_pass": True,
            "require_mechanism_gate_pass": True,
            "frozen_sdh_state": "",
            "secret_source_sha256": "",
            "secret_tensor_sha256": "",
            "source_manifest_sha256": "",
            "train_split_sha256": "",
            "frozen_sdh_state_sha256": "",
            "hiding_metrics_sha256": "",
            "hiding_checkpoint_sha256": "",
            "hiding_split_sha256": "",
            "mechanism_metrics_sha256": "",
            "mechanism_decision_sha256": "",
            "mechanism_config_sha256": "",
            "p1_state_sha256": "",
        }
        for key, value in required.items():
            if key not in sdh:
                sdh[key] = value



def load_config(config_path: str) -> Dict[str, Any]:
    defaults = _default_config()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    sirc_malc_requested = "sirc_malc_cgr" in user_cfg.get("methods", {})
    sdh_requested = "tausb_sdh" in user_cfg.get("methods", {})

    cfg = _deep_update(copy.deepcopy(defaults), user_cfg)
    _ensure_method_defaults(cfg)

    target_id = cfg["experiment"]["target_class_id"]
    num_classes = cfg["surrogate"]["num_classes"]
    if target_id < 0 or target_id >= num_classes:
        raise ValueError(
            "Target class id is out of surrogate class range. "
            f"target_class_id={target_id}, surrogate.num_classes={num_classes}"
        )

    method_cfg = cfg.get("methods", {}).get("sirc_malc_cgr", {})
    switches = tuple(
        bool(method_cfg.get(key, True))
        for key in ("enable_sirc_carrier", "enable_malc", "enable_cgr")
    )
    if sirc_malc_requested and any(switches):
        if target_id != 14 or int(cfg["surrogate"]["num_classes"]) != 20:
            raise ValueError("SIRC-MALC-CGR requires VOC20 person class id 14.")
        if abs(float(cfg["experiment"]["eps"]) - 16.0 / 255.0) > 1e-9:
            raise ValueError("SIRC-MALC-CGR epsilon must remain 16/255.")
        if int(cfg["surrogate"].get("imgsz", -1)) != 640:
            raise ValueError("SIRC-MALC-CGR surrogate imgsz must remain 640.")
        if int(cfg["surrogate"].get("eot_samples", -1)) != 1:
            raise ValueError("SIRC-MALC-CGR formal config forbids EOT.")
        victim = cfg["victim"]
        frozen_victim = {
            "epochs": 200,
            "imgsz": 640,
            "batch": 36,
            "optimizer": "SGD",
        }
        for key, expected in frozen_victim.items():
            if victim.get(key) != expected:
                raise ValueError(
                    f"SIRC-MALC-CGR victim.{key} must remain {expected}."
                )
        if [int(value) for value in cfg["experiment"].get("seeds", [])] != [0]:
            raise ValueError("SIRC-MALC-CGR first experiment must remain seed 0.")
        if float(cfg["experiment"].get("poisoning_ratio", -1)) != 1.0:
            raise ValueError("SIRC-MALC-CGR M1 poisoning_ratio must remain 1.0.")
        if int(cfg["experiment"].get("expected_poisoned_count", -1)) != 6095:
            raise ValueError("SIRC-MALC-CGR expected_poisoned_count must remain 6095.")
        for key in (
            "frozen_carrier_state",
            "semantic_bank_hash",
            "source_manifest_hash",
            "split_hash",
        ):
            if not str(method_cfg.get(key, "")).strip():
                raise ValueError(f"SIRC-MALC-CGR methods config requires {key}.")

    if sdh_requested:
        method_cfg = cfg.get("methods", {}).get("tausb_sdh", {})
        feature_switches = (
            "enable_deep_hiding",
            "enable_dlfc",
            "enable_cicr",
            "enable_cgr",
            "enable_nla_loss",
        )
        if not all(method_cfg.get(key) is True for key in feature_switches):
            raise ValueError("TAUSB-SDH formal config requires all method gates enabled.")
        if method_cfg.get("support_type") != "bbox":
            raise ValueError("TAUSB-SDH formal support must be person GT bbox.")
        if target_id != 14 or int(cfg["surrogate"]["num_classes"]) != 20:
            raise ValueError("TAUSB-SDH requires VOC20 person class id 14.")
        if abs(float(cfg["experiment"]["eps"]) - 16.0 / 255.0) > 1e-9:
            raise ValueError("TAUSB-SDH epsilon must remain 16/255.")
        if int(cfg["surrogate"].get("imgsz", -1)) != 640:
            raise ValueError("TAUSB-SDH surrogate imgsz must remain 640.")
        if int(cfg["surrogate"].get("eot_samples", -1)) != 1:
            raise ValueError("TAUSB-SDH first round forbids EOT.")
        if [int(value) for value in cfg["experiment"].get("seeds", [])] != [0]:
            raise ValueError("TAUSB-SDH first experiment must remain seed 0.")
        victim = cfg["victim"]
        protocol_id = str(method_cfg.get("protocol_id", ""))
        feasibility_protocol = protocol_id == "TAUSB-SDH-E2E-V0-MAP50-v1"
        frozen_victim = {"imgsz": 640, "batch": 36, "optimizer": "SGD"}
        frozen_victim["epochs"] = (
            int(victim.get("epochs", -1)) if feasibility_protocol else 200
        )
        for key, expected in frozen_victim.items():
            if victim.get(key) != expected:
                raise ValueError("TAUSB-SDH victim.%s must remain %s." % (key, expected))
        for key in (
            "frozen_sdh_state",
            "secret_source_sha256",
            "secret_tensor_sha256",
            "source_manifest_sha256",
            "train_split_sha256",
        ):
            if not str(method_cfg.get(key, "")).strip():
                raise ValueError("TAUSB-SDH methods config requires %s." % key)
        if feasibility_protocol:
            if method_cfg.get("materialization_mode") != "p1_feasibility_state":
                raise ValueError("E2E V0 requires p1_feasibility_state materialization.")
            if method_cfg.get("allow_failed_scientific_gates") is not True:
                raise ValueError("E2E V0 requires allow_failed_scientific_gates=true.")
            if method_cfg.get("binding_status") != "bound":
                raise ValueError("E2E V0 config is not bound to mechanism artifacts.")
            if method_cfg.get("evidence_scope") != (
                "end_to_end_feasibility_not_formal_method"
            ):
                raise ValueError("E2E V0 evidence scope mismatch.")
            if method_cfg.get("require_hiding_gate_pass") is not False:
                raise ValueError("E2E V0 must preserve hiding_gate_passed=false.")
            if method_cfg.get("require_mechanism_gate_pass") is not False:
                raise ValueError("E2E V0 mechanism gate must remain diagnostic.")
            if bool(cfg["data"].get("allow_pseudo_mask_fallback", True)):
                raise ValueError("E2E V0 requires explicit person GT bbox support.")
            pilot_kind = str(cfg["experiment"].get("pilot_kind", ""))
            arm_id = str(cfg["experiment"].get("arm_id", ""))
            if pilot_kind not in {"smoke", "e20", "e200"} or arm_id not in {"C0", "M1"}:
                raise ValueError("E2E V0 requires pilot_kind smoke/e20/e200 and arm C0/M1.")
            expected_epochs = {"smoke": 1, "e20": 20, "e200": 200}[pilot_kind]
            if int(victim.get("epochs", -1)) != expected_epochs:
                raise ValueError("E2E V0 victim epochs do not match the pilot kind.")
            expected_train_images = 200 if pilot_kind == "smoke" else 16551
            if int(cfg["experiment"].get("expected_train_images", -1)) != (
                expected_train_images
            ):
                raise ValueError("E2E V0 expected_train_images mismatch.")
            expected_target_images = 40 if pilot_kind == "smoke" else 6095
            if int(cfg["experiment"].get("expected_target_images", -1)) != (
                expected_target_images
            ):
                raise ValueError("E2E V0 expected_target_images mismatch.")
            expected_poisoned = 0 if arm_id == "C0" else (40 if pilot_kind == "smoke" else 6095)
            expected_ratio = 0.0 if arm_id == "C0" else 1.0
            if int(cfg["experiment"].get("expected_poisoned_count", -1)) != expected_poisoned:
                raise ValueError("E2E V0 expected_poisoned_count mismatch.")
            if float(cfg["experiment"].get("poisoning_ratio", -1)) != expected_ratio:
                raise ValueError("E2E V0 poisoning_ratio does not match arm identity.")
            selection_path = str(cfg["data"].get("train_selection_manifest", "")).strip()
            selection_hash = str(
                cfg["data"].get("train_selection_manifest_sha256", "")
            ).strip().lower()
            if pilot_kind == "smoke":
                if not selection_path or not _is_sha256(selection_hash):
                    raise ValueError("E2E V0 smoke requires a hash-bound selection manifest.")
            elif selection_path or selection_hash:
                raise ValueError("E2E V0 full-VOC pilot forbids a train selection manifest.")
            layout = str(cfg["data"].get("materialization_layout", "full_png_v1"))
            if pilot_kind in {"e20", "e200"} and layout not in {
                "full_png_v1",
                "sparse_mixed_list_v1",
            }:
                raise ValueError("E2E V0 E20 materialization layout is unsupported.")
            if pilot_kind == "smoke" and layout != "full_png_v1":
                raise ValueError("Legacy E2E V0 smoke must preserve full_png_v1 materialization.")
            for key in (
                "frozen_sdh_state_sha256",
                "hiding_metrics_sha256",
                "hiding_checkpoint_sha256",
                "hiding_split_sha256",
                "mechanism_metrics_sha256",
                "mechanism_decision_sha256",
                "mechanism_config_sha256",
                "p1_state_sha256",
            ):
                if not _is_sha256(method_cfg.get(key, "")):
                    raise ValueError("E2E V0 methods config requires SHA-256 %s." % key)
        else:
            if protocol_id != "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3":
                raise ValueError("TAUSB-SDH formal protocol_id mismatch.")
            if method_cfg.get("materialization_mode") != "formal_p1_state":
                raise ValueError("TAUSB-SDH formal materialization mode mismatch.")
            if method_cfg.get("allow_failed_scientific_gates") is not False:
                raise ValueError("TAUSB-SDH formal path forbids failed scientific gates.")
            if not all(
                method_cfg.get(key) is True
                for key in ("require_hiding_gate_pass", "require_mechanism_gate_pass")
            ):
                raise ValueError("TAUSB-SDH formal config requires all method gates enabled.")
            if int(cfg["experiment"].get("expected_poisoned_count", -1)) != 6095:
                raise ValueError("TAUSB-SDH expected_poisoned_count must remain 6095.")
            if float(cfg["experiment"].get("poisoning_ratio", -1)) != 1.0:
                raise ValueError("TAUSB-SDH poisoning_ratio must remain 1.0.")
            if int(victim.get("epochs", -1)) != 200:
                raise ValueError("TAUSB-SDH victim.epochs must remain 200.")

    return cfg
