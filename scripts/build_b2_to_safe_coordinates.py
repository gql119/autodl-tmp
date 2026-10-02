"""Build B2-TO-Safe coordinates from clean training data and frozen reference only."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import list_images, label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer, _feature_mask, _rectangle
from ue_framework.methods.safebg import select_strict_tsfl
from ue_framework.methods.semantic_coordinates import CLASSES, class_subspace, gt_evidence, sha256


def state_hash(model):
    d = hashlib.sha256()
    for key, value in model.state_dict().items():
        d.update(key.encode())
        d.update(value.detach().cpu().numpy().tobytes())
    return d.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root):
        raise ValueError("Strict TSFL artifacts must stay in the isolated project")
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    settings = method["semantic"]
    strict = settings["strict_tsfl"]
    safe_bg = settings["safe_bg"]
    if not strict.get("enabled") or not safe_bg.get("enabled"):
        raise ValueError("B2-TO-Safe config must enable both Strict TSFL and Safe-BG")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = YOLO(cfg["surrogate"]["ckpt"]).model.cuda().eval()
    trainer = LegacyKProtoTrainer(cfg, method, torch.device("cuda"), model)
    frozen_hash = state_hash(model)
    reference_dir = Path(args.reference_artifact_dir)
    reference_channels = torch.load(reference_dir / "target_specific_channels.pt",
                                    map_location="cpu", weights_only=True)
    reference_subspaces = torch.load(reference_dir / "non_target_subspaces.pt",
                                     map_location="cpu", weights_only=True)
    data = Path(cfg["data"]["dataset_root"])
    label_dir = data / cfg["data"]["train_labels"]
    images = sorted(list_images(str(data / cfg["data"]["train_images"])))
    class_sums = {c: None for c in range(cfg["experiment"]["num_classes"]) if c != trainer.target_class_id}
    class_counts = {c: 0 for c in class_sums}
    pooled = {c: [] for c in class_sums}
    target_sum = None
    target_count = 0
    manifest = hashlib.sha256()
    started = time.time()
    write_json(output / "status.json", {"state": "running", "images_total": len(images), "started_unix": started})
    for image_index, image_path in enumerate(images):
        image = load_image_rgb_float(image_path)
        label_path = label_path_for_image(image_path, str(label_dir))
        annotations = read_yolo_annotations(label_path)
        manifest.update((Path(image_path).stem + ":" + sha256(image_path) + ":" + sha256(label_path) + "\n").encode())
        if annotations:
            tensor = trainer._to_tensor(image).requires_grad_(True)
            trainer.multi_features.clear()
            decoded = trainer._forward_raw(tensor)
            feature = trainer.multi_features[trainer.feature_layer]
            for ann in annotations:
                cls = int(ann["cls"])
                roi = _feature_mask(_rectangle(image.shape[:2], ann["bbox"]), feature,
                                    image.shape[:2], trainer.imgsz)
                pooled_values = trainer._pooled(feature.detach(), [roi])
                if cls == trainer.target_class_id:
                    target_count += 1
                elif cls in pooled and pooled_values:
                    pooled[cls].append(pooled_values[0].cpu())
            for cls, evidence in gt_evidence(decoded, annotations, image.shape[:2], trainer.imgsz):
                gradient = torch.autograd.grad(evidence, feature, retain_graph=True)[0]
                sensitivity = gradient.detach().abs().mean(dim=(0, 2, 3)).double().cpu()
                if not torch.isfinite(sensitivity).all():
                    raise RuntimeError("Nonfinite GT-aligned class sensitivity")
                if cls == trainer.target_class_id:
                    target_sum = sensitivity.clone() if target_sum is None else target_sum + sensitivity
                elif cls in class_sums:
                    class_sums[cls] = sensitivity.clone() if class_sums[cls] is None else class_sums[cls] + sensitivity
                    class_counts[cls] += 1
            del feature, decoded, tensor
        if (image_index + 1) % 100 == 0:
            write_json(output / "status.json", {"state": "running", "images_done": image_index + 1,
                "images_total": len(images), "target_objects": target_count,
                "non_target_objects": sum(class_counts.values()), "elapsed_seconds": time.time() - started})
            print(f"{image_index + 1}/{len(images)} elapsed={time.time() - started:.1f}s", flush=True)
    if state_hash(model) != frozen_hash:
        raise RuntimeError("Frozen reference changed during Strict TSFL coordinate build")
    if target_count == 0 or target_sum is None:
        raise RuntimeError("No valid person sensitivity was observed")
    missing = [c for c in class_sums if not class_counts[c] or class_sums[c] is None or len(pooled[c]) < 2]
    if missing:
        raise RuntimeError(f"Clean training split has insufficient valid features for classes {missing}")
    s_target = target_sum / target_count
    s_nt_by_class = {c: class_sums[c] / class_counts[c] for c in class_sums}
    strict_result = select_strict_tsfl(
        s_target, s_nt_by_class,
        candidate_fraction=float(strict["target_candidate_fraction"]),
        safety_quantile=float(strict["nt_safety_quantile"]),
        worst_k=int(strict["sensitivity_worst_k"]),
        selected_fraction=float(strict["target_channel_fraction"]),
        min_selected=int(strict["min_selected_channels"]),
        eps=float(settings["orth_eps"]),
    )
    indices = strict_result["indices"].cpu()
    old_channels = reference_channels["indices"].long()
    s_nt_mean = torch.stack(list(s_nt_by_class.values())).mean(0)
    identity = {
        "feature_layer": trainer.feature_layer,
        "channels": int(s_target.numel()),
        "seed": int(args.seed),
        "checkpoint_sha256": sha256(cfg["surrogate"]["ckpt"]),
        "frozen_state_sha256": frozen_hash,
        "clean_train_manifest_sha256": manifest.hexdigest(),
        "images": len(images),
        "data_root": str(data),
        "train_images": cfg["data"]["train_images"],
        "train_labels": cfg["data"]["train_labels"],
        "validation_used": False,
        "builder_sha256": sha256(__file__),
        "coordinate_code_sha256": sha256(root / "ue_framework/methods/semantic_coordinates.py"),
        "safe_bg_code_sha256": sha256(root / "ue_framework/methods/safebg.py"),
        "reference_target_channels_sha256": sha256(reference_dir / "target_specific_channels.pt"),
        "reference_non_target_subspaces_sha256": sha256(reference_dir / "non_target_subspaces.pt"),
        "settings": {k: settings[k] for k in ("target_channel_fraction", "lambda_specificity",
            "subspace_variance", "subspace_min_rank", "subspace_max_rank_fraction")},
        "strict_tsfl_settings": strict,
        "safe_bg_settings": safe_bg,
    }
    ref_identity = reference_subspaces["identity"]
    for key in ("feature_layer", "checkpoint_sha256", "frozen_state_sha256", "clean_train_manifest_sha256",
                "data_root", "train_images", "train_labels", "validation_used", "settings"):
        if identity[key] != ref_identity[key]:
            raise RuntimeError(f"Formal B2-TO reference artifact mismatch in {key}")
    if sorted(int(c) for c in reference_subspaces["classes"]) != sorted(pooled):
        raise RuntimeError("Formal B2-TO reference is missing one or more NT class bases")
    tsfl = {
        "indices": indices,
        "S_target": s_target,
        "S_nt": s_nt_mean,
        "S_target_normalized": strict_result["S_target_normalized"],
        "S_nt_normalized": strict_result["S_nt_worst_normalized"],
        "S_nt_by_class": s_nt_by_class,
        "S_nt_class_normalized": strict_result["S_nt_class_normalized"],
        "S_nt_worst_normalized": strict_result["S_nt_worst_normalized"],
        "specificity": strict_result["ratio"],
        "identity": identity,
    }
    torch.save(tsfl, output / "target_specific_channels.pt")
    # Keep the exact formal B2-TO channel coordinates for its unchanged L_orth.
    torch.save(reference_channels, output / "orth_target_specific_channels.pt")
    torch.save({"indices": indices, "summary": strict_result["summary"],
                "identity": identity}, output / "strict_tsfl_channels.pt")
    # Preserve the exact full-space bases used by B2-TO's existing L_orth.
    full_subspaces = reference_subspaces["classes"]
    torch.save({"classes": full_subspaces, "identity": identity,
                "source_identity": ref_identity}, output / "non_target_subspaces.pt")
    selected_subspaces = {}
    for cls, vectors in pooled.items():
        selected_vectors = [v.index_select(0, indices) for v in vectors]
        selected_subspaces[cls] = class_subspace(selected_vectors,
            float(settings["subspace_variance"]), int(settings["subspace_min_rank"]),
            float(settings["subspace_max_rank_fraction"]))
    torch.save({"classes": selected_subspaces, "identity": identity}, output / "safe_bg_selected_subspaces.pt")
    summary = dict(strict_result["summary"])
    summary["selected_channel_top_conflicting_classes"] = {
        str(channel): [CLASSES[c] for c in class_ids]
        for channel, class_ids in summary["selected_channel_top_conflicting_classes"].items()
    }
    selected_mask = torch.zeros_like(s_target, dtype=torch.bool); selected_mask[indices] = True
    nt_worst = strict_result["S_nt_worst_normalized"]
    summary.update({
        "selected_channel_indices": indices.tolist(),
        "mean_target_sensitivity_selected_raw": float(s_target[selected_mask].mean()),
        "mean_target_sensitivity_unselected_raw": float(s_target[~selected_mask].mean()),
        "mean_S_nt_worst_selected": float(nt_worst[selected_mask].mean()),
        "mean_S_nt_worst_unselected": float(nt_worst[~selected_mask].mean()),
        "mean_S_t_selected": float(strict_result["S_target_normalized"][selected_mask].mean()),
        "mean_S_t_unselected": float(strict_result["S_target_normalized"][~selected_mask].mean()),
        "mean_ratio_selected": float(strict_result["ratio"][selected_mask].mean()),
        "mean_ratio_unselected": float(strict_result["ratio"][~selected_mask].mean()) if (~selected_mask).any() else None,
        "mean_S_nt_selected": float(s_nt_mean[selected_mask].mean()),
        "mean_S_nt_unselected": float(s_nt_mean[~selected_mask].mean()),
        "channel_selection_overlap_with_previous_run": float(len(set(indices.tolist()) & set(old_channels.tolist())) / max(1, len(indices))),
        "target_objects": target_count,
        "non_target_objects_by_class": class_counts,
        "images": len(images),
        "aggregation": "mean spatial absolute GT-evidence gradient per object, then equal-object mean",
        "calibration_split": "clean training only; final validation never opened",
        "identity": identity,
    })
    write_json(output / "tsfl_sensitivity_stats.json", {
        "classes": {str(c): {"class_name": CLASSES[c], "object_count": class_counts[c],
            "mean_sensitivity": s_nt_by_class[c].tolist(),
            "mean_normalized_sensitivity": strict_result["S_nt_class_normalized"][i].tolist()}
            for i, c in enumerate(sorted(s_nt_by_class))},
        "target_mean_sensitivity": s_target.tolist(),
        "worst3_nt_sensitivity_normalized": nt_worst.tolist(),
        "identity": identity,
    })
    write_json(output / "strict_tsfl_summary.json", summary)
    write_json(output / "tsfl_summary.json", {
        "num_channels_total": summary["total_channels"],
        "num_channels_selected": summary["final_selected_channels"],
        "target_channel_fraction": summary["final_selected_fraction"],
        "indices": indices.tolist(),
        "mean_S_target_selected": summary["mean_S_t_selected"],
        "mean_S_nt_selected": summary["mean_S_nt_selected"],
        "mean_S_target_unselected": summary["mean_S_t_unselected"],
        "mean_S_nt_unselected": summary["mean_S_nt_unselected"],
        "specificity_selected_mean": summary["mean_ratio_selected"],
        "channel_selection_overlap_with_previous_run": summary["channel_selection_overlap_with_previous_run"],
        "num_channels_selected": summary["final_selected_channels"],
        "mean_target_sensitivity_selected": summary["mean_target_sensitivity_selected_raw"],
        "mean_non_target_sensitivity_selected": summary["mean_S_nt_selected"],
        "mean_target_sensitivity_unselected": summary["mean_target_sensitivity_unselected_raw"],
        "mean_non_target_sensitivity_unselected": summary["mean_S_nt_unselected"],
        "determinism_test": "fixed seed and sorted clean training manifest; selection repeated by smoke",
        "aggregation": summary["aggregation"],
        "images": len(images), "target_objects": target_count,
        "non_target_objects": sum(class_counts.values()),
    })
    write_json(output / "subspace_summary.json", {
        CLASSES[c]: {k: v for k, v in item.items() if k not in ("basis", "mean")} |
        {"gt_count": class_counts[c], "invalid_pool_count": class_counts[c] - item["count"]}
        for c, item in full_subspaces.items()
    })
    write_json(output / "coordinates_identity.json", identity)
    write_json(output / "status.json", {"state": "complete", "images_done": len(images),
        "duration_seconds": time.time() - started, "frozen_hash_unchanged": True,
        "target_objects": target_count, "non_target_objects": sum(class_counts.values()),
        "manifest_sha256": manifest.hexdigest(), "selected_channels": indices.tolist()})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
