"""Validate the preregistered five-step B2-TO-Safe correctness gate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, list_images, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle
from ue_framework.methods.semantic_coordinates import semantic_identity


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name):
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Nonfinite sanity value: {name}")
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    noise = Path(args.noise)
    out = Path(args.output)
    status = json.loads((noise / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or not status.get("smoke") or status.get("steps") != 5:
        raise RuntimeError("Sanity run must complete exactly five outer steps")
    if status.get("epochs_completed") != 1 or status.get("seed") != 0:
        raise RuntimeError("Sanity run seed/epoch differs from locked protocol")
    if status.get("semantic_identity") != semantic_identity(cfg, ROOT):
        raise RuntimeError("Sanity semantic identity differs from generated Strict TSFL/Safe-BG")
    if status.get("frozen_state_sha256_before") != status.get("frozen_state_sha256_after"):
        raise RuntimeError("Frozen reference changed during sanity")
    if status.get("adaptive_state_sha256_before") != status.get("adaptive_state_sha256_after"):
        raise RuntimeError("Adaptive base changed during sanity")
    artifact = Path(cfg["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"])
    strict = json.loads((artifact / "strict_tsfl_summary.json").read_text(encoding="utf-8"))
    orth_channels = torch.load(artifact / "orth_target_specific_channels.pt",
                               map_location="cpu", weights_only=True)
    reference_channels = torch.load(
        Path(cfg["methods"]["legacy_kproto_ret"]["semantic"]["safe_bg"]["reference_artifact_dir"])
        / "target_specific_channels.pt", map_location="cpu", weights_only=True)
    if not torch.equal(orth_channels["indices"], reference_channels["indices"]):
        raise RuntimeError("L_orth channels differ from formal B2-TO reference")
    if strict["final_selected_channels"] < 4 or not strict["selected_channel_indices"]:
        raise RuntimeError("Strict TSFL selected fewer than four channels")
    if strict["mean_S_t_selected"] <= strict["mean_S_t_unselected"]:
        raise RuntimeError("Selected channels do not have higher target sensitivity")
    if strict["mean_S_nt_worst_selected"] >= strict["mean_S_nt_worst_unselected"]:
        raise RuntimeError("Selected channels do not have lower worst-NT sensitivity")
    fixed = json.loads((noise / "fixed_pair.json").read_text(encoding="utf-8"))
    after = fixed["after_5"]
    retention = finite(after["safe_direction_retention"], "safe_direction_retention")
    if retention < 0.10:
        raise RuntimeError(f"Median Safe-BG direction retention below 0.10: {retention}")
    if finite(after["safe_direction_norm"], "safe_direction_norm") <= 0:
        raise RuntimeError("Safe-BG direction collapsed to zero")
    if abs(finite(after["cos_d_safe_projected_nt"], "projector orthogonality")) > 1e-3:
        raise RuntimeError("Safe direction is not orthogonal to the active NT projector")
    for key in ("L_BG", "L_rel", "L_orth_worst", "L_total"):
        finite(after[key], key)
    for key in ("grad_BG_safe_norm", "grad_rel_safe_norm"):
        if finite(after[key], key) <= 0:
            raise RuntimeError(f"Safe loss has no effective perturbation gradient: {key}")
    if abs(finite(after["L_orth_worst"], "L_orth_worst") - finite(after["L_orth_worst_tsfl_channels"], "L_orth_worst_tsfl_channels")) > 1e-8:
        raise RuntimeError("L_orth diagnostic identity mismatch")
    with (noise / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise RuntimeError(f"Expected five diagnostic steps, found {len(rows)}")
    eps = float(cfg["experiment"]["eps"])
    for row in rows:
        if finite(row["max_abs_delta"], "max_abs_delta") > eps + 1e-6:
            raise RuntimeError("Global perturbation budget exceeded in sanity")
        if finite(row["outside_support_max"], "outside_support_max") > 1e-8:
            raise RuntimeError("Perturbation changed pixels outside person support")
        if finite(row["L_BG"], "L_BG") < 0 or finite(row["L_rel"], "L_rel") < 0:
            raise RuntimeError("BG/relative loss became invalid")
        if finite(row["safe_direction_retention"], "safe_direction_retention") < 0:
            raise RuntimeError("Negative Safe-BG retention")
    data = Path(cfg["data"]["dataset_root"])
    preview = json.loads((noise / "preview_manifest.json").read_text(encoding="utf-8"))
    if len(preview) < 1:
        raise RuntimeError("No decoded-pixel preview was produced")
    train_dir = Path(cfg["data"]["train_images"])
    if not train_dir.is_absolute():
        train_dir = data / train_dir
    clean_by_stem = {Path(p).stem: Path(p) for p in list_images(str(train_dir))}
    pixel_cap = int(round(eps * 255))
    overlap_cap = int(round(pixel_cap * 0.5))
    pixel_maxima = {"global": 0, "target": 0, "overlap": 0, "outside": 0}
    for item in preview:
        image_path = Path(item["path"])
        clean_path = clean_by_stem.get(image_path.stem)
        if clean_path is None:
            raise RuntimeError(f"Preview stem is absent from clean training split: {image_path.stem}")
        label_path = label_path_for_image(str(clean_path), str(data / cfg["data"]["train_labels"]))
        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        poisoned = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if clean is None or poisoned is None or clean.shape != poisoned.shape:
            raise RuntimeError(f"Could not compare decoded smoke preview {image_path.stem}")
        annotations = read_yolo_annotations(label_path)
        difference = np.abs(clean.astype(np.int16) - poisoned.astype(np.int16)).max(axis=2)
        target = np.zeros(difference.shape, dtype=bool)
        non_target = np.zeros(difference.shape, dtype=bool)
        for ann in annotations:
            box = _rectangle(difference.shape, ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann["cls"]) == 14:
                target |= box
            else:
                non_target |= box
        outside = difference[~target]
        target_values = difference[target]
        overlap_values = difference[target & non_target]
        pixel_maxima["global"] = max(pixel_maxima["global"], int(difference.max(initial=0)))
        pixel_maxima["outside"] = max(pixel_maxima["outside"], int(outside.max(initial=0)))
        pixel_maxima["target"] = max(pixel_maxima["target"], int(target_values.max(initial=0)))
        pixel_maxima["overlap"] = max(pixel_maxima["overlap"], int(overlap_values.max(initial=0)))
        if pixel_maxima["global"] > pixel_cap or pixel_maxima["target"] > pixel_cap:
            raise RuntimeError("Decoded smoke preview exceeds the locked pixel epsilon")
        if pixel_maxima["overlap"] > overlap_cap:
            raise RuntimeError("Decoded smoke preview exceeds the half-strength overlap budget")
        if pixel_maxima["outside"] != 0:
            raise RuntimeError("Decoded smoke preview changed pixels outside person boxes")
    orth_cfg = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    if float(orth_cfg["lambda_orth"]) != 0.0005 or int(orth_cfg["orth_worst_k"]) != 3:
        raise RuntimeError("Locked B2-TO orthogonality parameters changed")
    receipt = {
        "state": "passed", "gate": "B2-TO-Safe five-step correctness sanity",
        "config_sha256": sha256(args.config), "noise_status_sha256": sha256(noise / "status.json"),
        "steps": 5, "seed": 0, "lambda_orth": float(orth_cfg["lambda_orth"]),
        "orth_worst_k": int(orth_cfg["orth_worst_k"]),
        "orth_channel_count": int(orth_channels["indices"].numel()),
        "strict_tsfl_channel_count": strict["final_selected_channels"],
        "selected_channel_count": strict["final_selected_channels"],
        "selected_channel_indices": strict["selected_channel_indices"],
        "mean_target_sensitivity_selected": strict["mean_target_sensitivity_selected_raw"],
        "mean_worst_nt_sensitivity_selected": strict["mean_S_nt_worst_selected"],
        "median_safe_direction_retention_fixed_probe": retention,
        "removed_nt_component_ratio_fixed_probe": finite(after["removed_nt_component_ratio"], "removed_nt_component_ratio"),
        "orthogonality_cosine_fixed_probe": finite(after["cos_d_safe_projected_nt"], "cos_d_safe_projected_nt"),
        "grad_BG_safe_norm": finite(after["grad_BG_safe_norm"], "grad_BG_safe_norm"),
        "grad_rel_safe_norm": finite(after["grad_rel_safe_norm"], "grad_rel_safe_norm"),
        "grad_attack_vs_orth_cosine": finite(after["cos_grad_attack_orth"], "cos_grad_attack_orth"),
        "global_epsilon": eps, "adaptive_base_frozen": True, "frozen_reference_unchanged": True,
        "decoded_preview_pixel_maxima": pixel_maxima,
        "final_validation_used_for_selection": False,
    }
    (out / "sanity_gate.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
