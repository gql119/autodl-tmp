import csv
import os
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from ..data_utils import (
    load_image_rgb_float,
    split_val_image_lists,
    write_image_list_txt,
)
from ..env_utils import resolve_workers
from ..io_utils import atomic_write_json, read_csv_rows
from ..metrics_utils import (
    build_pareto_scores,
    compute_lpips_batch,
    compute_non_target_map,
    extract_map50_per_class,
)
from ..runtime import RunContext
from ..status import (
    load_or_init_status,
    mark_stage_completed,
    mark_stage_running,
    stage_completed,
)


TRUE_LIKE = {"1", "1.0", "true", "yes", "y", "t"}


def _write_eval_yaml(ctx: RunContext, yaml_path: str, val_spec: str) -> str:
    content = f"""path: {ctx.paths.poisoned_root}
train: images/train
val: {val_spec}
names:
"""
    for i in range(int(ctx.cfg["experiment"]["num_classes"])):
        if i == int(ctx.cfg["experiment"]["target_class_id"]):
            cls_name = ctx.cfg["experiment"]["target_class_name"]
        else:
            cls_name = f"class_{i}"
        content += f"  {i}: {cls_name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def _extract_metrics_dict(metrics_obj, num_classes: int, target_id: int) -> Dict:
    box = getattr(metrics_obj, "box", None)
    map50_all = float(getattr(box, "map50", float("nan"))) if box is not None else float("nan")
    ap50_cls = extract_map50_per_class(metrics_obj, num_classes)

    m_target = float(ap50_cls[target_id]) if target_id < len(ap50_cls) else float("nan")
    m_non = compute_non_target_map(ap50_cls, target_id)

    return {
        "mAP50_all": map50_all,
        "mAP50_target": m_target,
        "mAP50_non_target": m_non,
        "ap50_per_class": ap50_cls,
    }


def _is_true_like(v) -> bool:
    s = str(v).strip().lower()
    return s in TRUE_LIKE


def _compute_psnr(clean: np.ndarray, poisoned: np.ndarray) -> float:
    mse = float(np.mean((clean - poisoned) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _compute_area_ratio(clean: np.ndarray, poisoned: np.ndarray) -> float:
    diff = np.max(np.abs(clean - poisoned), axis=2)
    return float(np.mean(diff > 1e-6))


def _compute_image_quality(ctx: RunContext, manifest_rows: List[Dict]) -> Dict:
    poisoned_rows = [
        r
        for r in manifest_rows
        if _is_true_like(r.get("is_poisoned", r.get("poisoned", "0")))
    ]
    # Fallback: if manifest poisoning flags are missing/inconsistent, infer from actual pixel diff.
    if not poisoned_rows:
        inferred = []
        for r in manifest_rows:
            p_path = str(r.get("image_path", "")).strip()
            if not p_path or not os.path.isfile(p_path):
                continue
            c_path = os.path.join(ctx.train_img_dir, os.path.basename(p_path))
            if not os.path.isfile(c_path):
                continue
            try:
                clean = load_image_rgb_float(c_path)
                poison = load_image_rgb_float(p_path)
            except Exception:
                continue
            if _compute_area_ratio(clean, poison) > 1e-8:
                inferred.append(r)
        poisoned_rows = inferred

    if not poisoned_rows:
        return {
            "PSNR": float("nan"),
            "LPIPS": float("nan"),
            "average_perturbed_area_ratio": 0.0,
            "poisoned_count": 0,
        }

    psnr_values: List[float] = []
    area_values: List[float] = []
    clean_batch = []
    poison_batch = []

    for row in poisoned_rows:
        p_path = str(row.get("image_path", "")).strip()
        if not p_path or not os.path.isfile(p_path):
            continue

        base = os.path.basename(p_path)
        c_path = os.path.join(ctx.train_img_dir, base)
        if not os.path.isfile(c_path):
            continue

        try:
            clean = load_image_rgb_float(c_path)
            poison = load_image_rgb_float(p_path)
        except Exception:
            continue

        # 强制按像素差重算，避免 manifest 残留值污染质量指标。
        psnr_val = _compute_psnr(clean, poison)
        area_val = _compute_area_ratio(clean, poison)

        psnr_values.append(psnr_val)
        area_values.append(area_val)

        if len(clean_batch) < 64:
            clean_batch.append(clean)
            poison_batch.append(poison)

    psnr_mean = float(np.nanmean(psnr_values)) if psnr_values else float("nan")
    area_mean = float(np.nanmean(area_values)) if area_values else 0.0

    lpips_val = float("nan")
    if clean_batch and poison_batch:
        maybe_lpips = compute_lpips_batch(clean_batch, poison_batch)
        if maybe_lpips is not None:
            lpips_val = float(maybe_lpips)

    return {
        "PSNR": psnr_mean,
        "LPIPS": lpips_val,
        "average_perturbed_area_ratio": area_mean,
        "poisoned_count": len(psnr_values),
    }


def _nan_subset_metrics() -> Dict:
    return {
        "mAP50_all": float("nan"),
        "mAP50_target": float("nan"),
        "mAP50_non_target": float("nan"),
        "ap50_per_class": [],
    }


def run_evaluate(ctx: RunContext) -> None:
    cfg = ctx.cfg
    exp_cfg = cfg["experiment"]

    status = load_or_init_status(ctx.paths.artifact_status_json, ctx.method, ctx.steps, ctx.seed)
    if cfg["platform"].get("resume", True) and stage_completed(status, "evaluate"):
        print("[evaluate] already completed, skipping.")
        return
    status = mark_stage_running(ctx.paths.artifact_status_json, status, "evaluate")

    best_ckpt = os.path.join(ctx.paths.checkpoints_dir, "best.pt")
    latest_ckpt = os.path.join(ctx.paths.checkpoints_dir, "latest.pt")
    ckpt = best_ckpt if os.path.isfile(best_ckpt) else latest_ckpt
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "Checkpoint not found for evaluation. Run train_victim first. "
            f"checked: {best_ckpt}, {latest_ckpt}"
        )

    workers = resolve_workers(ctx.platform_mode, cfg)
    model = YOLO(ckpt)

    full_yaml = _write_eval_yaml(
        ctx,
        os.path.join(ctx.paths.eval_dir, "eval_full.yaml"),
        ctx.val_img_dir,
    )
    full_metrics_obj = model.val(
        data=full_yaml,
        imgsz=int(cfg["victim"]["imgsz"]),
        batch=int(cfg["victim"]["batch"]),
        device=str(ctx.gpu_id),
        workers=int(workers),
        verbose=False,
    )
    full_metrics = _extract_metrics_dict(
        full_metrics_obj,
        int(exp_cfg["num_classes"]),
        int(exp_cfg["target_class_id"]),
    )

    person_free, person_cooccur = split_val_image_lists(
        ctx.val_img_dir,
        ctx.val_label_dir,
        int(exp_cfg["target_class_id"]),
    )

    person_free_txt = os.path.join(ctx.paths.eval_dir, "val_person_free.txt")
    person_cooccur_txt = os.path.join(ctx.paths.eval_dir, "val_person_cooccur.txt")
    write_image_list_txt(person_free_txt, person_free)
    write_image_list_txt(person_cooccur_txt, person_cooccur)

    if person_free:
        free_yaml = _write_eval_yaml(ctx, os.path.join(ctx.paths.eval_dir, "eval_person_free.yaml"), person_free_txt)
        free_obj = model.val(
            data=free_yaml,
            imgsz=int(cfg["victim"]["imgsz"]),
            batch=int(cfg["victim"]["batch"]),
            device=str(ctx.gpu_id),
            workers=int(workers),
            verbose=False,
        )
        free_metrics = _extract_metrics_dict(free_obj, int(exp_cfg["num_classes"]), int(exp_cfg["target_class_id"]))
    else:
        free_metrics = _nan_subset_metrics()

    if person_cooccur:
        co_yaml = _write_eval_yaml(ctx, os.path.join(ctx.paths.eval_dir, "eval_person_cooccur.yaml"), person_cooccur_txt)
        co_obj = model.val(
            data=co_yaml,
            imgsz=int(cfg["victim"]["imgsz"]),
            batch=int(cfg["victim"]["batch"]),
            device=str(ctx.gpu_id),
            workers=int(workers),
            verbose=False,
        )
        co_metrics = _extract_metrics_dict(co_obj, int(exp_cfg["num_classes"]), int(exp_cfg["target_class_id"]))
    else:
        co_metrics = _nan_subset_metrics()

    manifest_rows = read_csv_rows(ctx.paths.manifest_csv)
    quality_metrics = _compute_image_quality(ctx, manifest_rows)

    final_metrics = {
        "method": ctx.method,
        "steps": ctx.steps,
        "seed": ctx.seed,
        "mAP50_target": full_metrics["mAP50_target"],
        "mAP50_non_target": full_metrics["mAP50_non_target"],
        "mAP50_all": full_metrics["mAP50_all"],
        "AP_person_free_non_target": free_metrics["mAP50_non_target"],
        "AP_person_cooccur_non_target": co_metrics["mAP50_non_target"],
        "PSNR": quality_metrics["PSNR"],
        "LPIPS": quality_metrics["LPIPS"],
        "average_perturbed_area_ratio": quality_metrics["average_perturbed_area_ratio"],
    }

    final_metrics.update(
        build_pareto_scores(
            final_metrics,
            ref_target=float(exp_cfg.get("reference_target_map50", 1.0)),
            ref_non_target=float(exp_cfg.get("reference_non_target_map50", 1.0)),
        )
    )

    os.makedirs(ctx.paths.metrics_dir, exist_ok=True)
    metrics_json = os.path.join(ctx.paths.metrics_dir, "metrics.json")
    metrics_csv = os.path.join(ctx.paths.metrics_dir, "metrics.csv")
    eval_log = os.path.join(ctx.paths.logs_dir, "eval_log.txt")

    atomic_write_json(metrics_json, final_metrics)

    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_metrics.keys()))
        writer.writeheader()
        writer.writerow(final_metrics)

    with open(eval_log, "w", encoding="utf-8") as f:
        f.write("Unified evaluation complete.\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"poisoned_count: {quality_metrics['poisoned_count']}\n")

    mark_stage_completed(
        ctx.paths.artifact_status_json,
        status,
        "evaluate",
        {
            "metrics_json": metrics_json,
            "metrics_csv": metrics_csv,
            "eval_log": eval_log,
        },
    )

    print(
        "[evaluate] done: "
        f"mAP50_target={final_metrics['mAP50_target']:.4f}, "
        f"mAP50_non_target={final_metrics['mAP50_non_target']:.4f}, "
        f"PSNR={final_metrics['PSNR']:.4f}, area={final_metrics['average_perturbed_area_ratio']:.6f}"
    )
