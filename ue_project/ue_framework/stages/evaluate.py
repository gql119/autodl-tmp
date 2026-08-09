import csv
import os
import json
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from ..data_utils import (
    list_images,
    load_image_rgb_float,
    split_val_image_lists,
    stem_of,
    write_image_list_txt,
)
from ..env_utils import resolve_workers
from ..io_utils import atomic_write_json, read_csv_rows
from ..metrics_utils import (
    build_pareto_scores,
    compute_lpips_batch,
    compute_non_target_map,
    extract_map50_per_class,
    VOC20_CLASS_NAMES,
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
    num_classes = int(ctx.cfg["experiment"]["num_classes"])
    if num_classes != len(VOC20_CLASS_NAMES):
        raise ValueError("The approved evaluator requires the full VOC20 class space.")
    for i, cls_name in enumerate(VOC20_CLASS_NAMES):
        content += f"  {i}: {cls_name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def _extract_metrics_dict(
    metrics_obj,
    num_classes: int,
    target_id: int,
    *,
    strict: bool,
) -> Dict:
    box = getattr(metrics_obj, "box", None)
    reported_map50 = float(getattr(box, "map50", float("nan"))) if box is not None else float("nan")
    ap50_cls = extract_map50_per_class(metrics_obj, num_classes, strict=strict)
    finite = [value for value in ap50_cls if np.isfinite(value)]
    map50_all = float(np.mean(finite)) if finite else float("nan")
    if strict:
        if len(finite) != num_classes:
            raise ValueError("Strict VOC20 evaluation requires 20 finite AP50 values.")
        if not np.isfinite(reported_map50):
            raise ValueError("Ultralytics reported map50 is non-finite.")
        if abs(reported_map50 - map50_all) > 1e-5:
            raise ValueError(
                "Reported map50 disagrees with the mapped 20-class AP50 mean."
            )

    m_target = float(ap50_cls[target_id]) if target_id < len(ap50_cls) else float("nan")
    m_non = compute_non_target_map(ap50_cls, target_id)

    return {
        "mAP50_all": map50_all,
        "mAP50_target": m_target,
        "mAP50_non_target": m_non,
        "ap50_per_class": ap50_cls,
        "ap50_by_class": {
            name: float(ap50_cls[index])
            for index, name in enumerate(VOC20_CLASS_NAMES)
            if np.isfinite(ap50_cls[index])
        },
    }


def _is_true_like(v) -> bool:
    s = str(v).strip().lower()
    return s in TRUE_LIKE


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _compute_psnr(clean: np.ndarray, poisoned: np.ndarray) -> float:
    mse = float(np.mean((clean - poisoned) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _compute_area_ratio(clean: np.ndarray, poisoned: np.ndarray) -> float:
    diff = np.max(np.abs(clean - poisoned), axis=2)
    return float(np.mean(diff > 1e-6))


def _read_poison_stats_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def _compute_quality_from_poison_stats(poison_rows: List[Dict]) -> Dict:
    if not poison_rows:
        return {
            "PSNR": float("nan"),
            "LPIPS": float("nan"),
            "average_perturbed_area_ratio": 0.0,
            "average_support_area_ratio": 0.0,
            "poisoned_count": 0,
        }

    poisoned_rows = []
    for r in poison_rows:
        v = r.get("poisoned", 0)
        try:
            is_p = int(v) == 1
        except Exception:
            is_p = _is_true_like(v)
        if is_p:
            poisoned_rows.append(r)

    poisoned_count = len(poisoned_rows)
    if poisoned_count <= 0:
        print("[WARN][PoisonStats] poisoned_count=0. Check generation.")
        return {
            "PSNR": float("nan"),
            "LPIPS": float("nan"),
            "average_perturbed_area_ratio": 0.0,
            "average_support_area_ratio": 0.0,
            "poisoned_count": 0,
        }

    psnr_vals = []
    lpips_vals = []
    changed_vals = []
    support_vals = []
    for r in poisoned_rows:
        p = _safe_float(r.get("psnr"))
        if np.isfinite(p):
            psnr_vals.append(p)

        l = r.get("lpips", None)
        l = _safe_float(l) if l is not None else float("nan")
        if np.isfinite(l):
            lpips_vals.append(l)

        c = _safe_float(r.get("changed_pixel_ratio"))
        if np.isfinite(c):
            changed_vals.append(c)

        s = _safe_float(r.get("support_area_ratio"))
        if np.isfinite(s) and s > 0:
            support_vals.append(s)

    psnr_mean = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
    lpips_mean = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    changed_mean = float(np.mean(changed_vals)) if changed_vals else 0.0
    support_mean = float(np.mean(support_vals)) if support_vals else 0.0
    return {
        "PSNR": psnr_mean,
        "LPIPS": lpips_mean,
        "average_perturbed_area_ratio": changed_mean,
        "average_support_area_ratio": support_mean,
        "poisoned_count": poisoned_count,
    }


def _compute_image_quality_legacy(ctx: RunContext, manifest_rows: List[Dict]) -> Dict:
    poison_stats_jsonl = os.path.join(ctx.paths.artifact_root, "poison_stats.jsonl")
    poison_rows = _read_poison_stats_jsonl(poison_stats_jsonl)
    if poison_rows:
        return _compute_quality_from_poison_stats(poison_rows)

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
            "average_support_area_ratio": 0.0,
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
        "average_support_area_ratio": 0.0,
        "poisoned_count": len(psnr_values),
    }


def _compute_image_quality(ctx: RunContext, manifest_rows: List[Dict]) -> Dict:
    poisoned_rows = [
        row
        for row in manifest_rows
        if _is_true_like(row.get("is_poisoned", row.get("poisoned", "0")))
    ]
    if not poisoned_rows:
        return {
            "PSNR": None,
            "LPIPS": None,
            "average_perturbed_area_ratio": 0.0,
            "average_support_area_ratio": 0.0,
            "poisoned_count": 0,
            "actual_linf_max": 0.0,
            "actual_linf_mean": 0.0,
            "validation_gaps": ["no_poisoned_manifest_rows"],
        }

    linf_values = [_safe_float(row.get("linf")) for row in poisoned_rows]
    if not all(np.isfinite(value) and value >= 0 for value in linf_values):
        raise ValueError("Every poisoned manifest row requires finite non-negative linf.")
    psnr_values = [
        value
        for value in (_safe_float(row.get("psnr")) for row in poisoned_rows)
        if np.isfinite(value)
    ]
    area_values = [
        value
        for value in (
            _safe_float(row.get("perturbed_area_ratio"))
            for row in poisoned_rows
        )
        if np.isfinite(value)
    ]
    support_values = [
        value
        for value in (_safe_float(row.get("support_ratio")) for row in poisoned_rows)
        if np.isfinite(value)
    ]
    gaps = []
    if len(psnr_values) != len(poisoned_rows):
        gaps.append("manifest_psnr_incomplete")
    if len(area_values) != len(poisoned_rows):
        gaps.append("manifest_perturbed_area_incomplete")
    if len(support_values) != len(poisoned_rows):
        gaps.append("manifest_support_ratio_incomplete")

    clean_batch = []
    poison_batch = []
    clean_by_stem = {stem_of(path): path for path in list_images(ctx.train_img_dir)}
    for row in poisoned_rows[:64]:
        poisoned_path = str(row.get("image_path", "")).strip()
        if not poisoned_path or not os.path.isfile(poisoned_path):
            gaps.append("lpips_poison_image_missing")
            continue
        clean_path = clean_by_stem.get(stem_of(poisoned_path))
        if not clean_path:
            gaps.append("lpips_clean_image_missing")
            continue
        try:
            clean_batch.append(load_image_rgb_float(clean_path))
            poison_batch.append(load_image_rgb_float(poisoned_path))
        except Exception:
            gaps.append("lpips_image_read_failure")

    lpips_value = None
    if clean_batch and poison_batch:
        computed = compute_lpips_batch(clean_batch, poison_batch)
        if computed is None:
            gaps.append("LPIPS_dependency_unavailable")
        else:
            lpips_value = float(computed)
    else:
        gaps.append("LPIPS_no_readable_pairs")

    return {
        "PSNR": float(np.mean(psnr_values)) if psnr_values else None,
        "LPIPS": lpips_value,
        "average_perturbed_area_ratio": float(np.mean(area_values)) if area_values else 0.0,
        "average_support_area_ratio": float(np.mean(support_values)) if support_values else 0.0,
        "poisoned_count": len(poisoned_rows),
        "actual_linf_max": float(np.max(linf_values)),
        "actual_linf_mean": float(np.mean(linf_values)),
        "validation_gaps": sorted(set(gaps)),
    }


def _nan_subset_metrics() -> Dict:
    return {
        "mAP50_all": float("nan"),
        "mAP50_target": float("nan"),
        "mAP50_non_target": float("nan"),
        "ap50_per_class": [],
    }


def _sirc_manifest_provenance(ctx: RunContext, manifest_rows: List[Dict]) -> Dict:
    if ctx.method != "sirc_malc_cgr":
        return {}
    poisoned = [
        row
        for row in manifest_rows
        if _is_true_like(row.get("is_poisoned", row.get("poisoned", "0")))
    ]
    if not poisoned:
        return {"materializer_provenance": "not_applicable_clean_arm"}
    expected = ctx.cfg["methods"]["sirc_malc_cgr"]
    output = {}
    for key in (
        "state_content_hash",
        "semantic_bank_hash",
        "source_manifest_hash",
        "split_hash",
    ):
        values = {str(row.get(key, "")).strip() for row in poisoned}
        if len(values) != 1 or "" in values:
            raise ValueError(f"Poisoned manifest has inconsistent {key} values.")
        value = next(iter(values))
        if len(value) != 64:
            raise ValueError(f"Poisoned manifest {key} is not a SHA-256 digest.")
        output[key] = value
    for key in ("semantic_bank_hash", "source_manifest_hash", "split_hash"):
        if output[key] != str(expected[key]):
            raise ValueError(f"Poisoned manifest {key} differs from formal config.")
    variants = {int(row["variant_index"]) for row in poisoned}
    if not variants or any(value < 0 or value >= 4 for value in variants):
        raise ValueError("Poisoned manifest variant indices are invalid.")
    support_sources = {str(row.get("support_source", "")) for row in poisoned}
    if support_sources != {"forced_pseudo_fallback"}:
        raise ValueError("Formal M1 must use only forced_pseudo_fallback support.")
    output["variant_indices_used"] = sorted(variants)
    output["support_source"] = "forced_pseudo_fallback"
    return output


def _sirc_mechanism_evidence(ctx: RunContext) -> Dict:
    """Load the immutable held-out mechanism report for the formal M1 arm."""

    if ctx.method != "sirc_malc_cgr" or ctx.run_tag != "M1":
        return {"mechanism_diagnostics": "not_applicable_clean_or_legacy_arm"}
    method_cfg = ctx.cfg["methods"]["sirc_malc_cgr"]
    frozen_state_value = str(method_cfg.get("frozen_carrier_state", "")).strip()
    if not frozen_state_value:
        raise ValueError("Formal M1 requires frozen_carrier_state.")
    frozen_state = os.path.abspath(frozen_state_value)
    report_path = os.path.join(os.path.dirname(frozen_state), "mechanism_report.json")
    if not os.path.isfile(report_path):
        raise FileNotFoundError(
            f"Formal M1 mechanism report is missing: {report_path}"
        )
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if int(report.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported MALC mechanism report schema.")
    if report.get("evidence_scope") != "heldout_mechanism_only_not_fresh_victim_ue":
        raise ValueError("MALC mechanism report has an invalid evidence scope.")
    if str(report.get("split_hash", "")) != str(method_cfg.get("split_hash", "")):
        raise ValueError("MALC mechanism report split hash differs from formal config.")
    gate = report.get("gate")
    if not isinstance(gate, dict) or gate.get("pass") is not True:
        raise ValueError("Formal M1 requires a passing MALC mechanism gate report.")
    if gate.get("allow_fresh_victim") is not True:
        raise ValueError("MALC mechanism report forbids fresh-victim execution.")
    for arm in ("A0", "A1"):
        if not isinstance(report.get(arm), dict):
            raise ValueError(f"MALC mechanism report is missing {arm} diagnostics.")
    return {
        "mechanism_diagnostics": {
            "report_path": report_path,
            "evidence_scope": report["evidence_scope"],
            "split_hash": report["split_hash"],
            "A0": report["A0"],
            "A1": report["A1"],
            "gate": gate,
        }
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
        strict=True,
    )

    person_free, person_cooccur = split_val_image_lists(
        ctx.val_img_dir,
        ctx.val_label_dir,
        int(exp_cfg["target_class_id"]),
    )
    if not person_free or not person_cooccur:
        raise ValueError("VOC validation split must contain both person-free and person-cooccur images.")

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
        free_metrics = _extract_metrics_dict(
            free_obj,
            int(exp_cfg["num_classes"]),
            int(exp_cfg["target_class_id"]),
            strict=False,
        )
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
        co_metrics = _extract_metrics_dict(
            co_obj,
            int(exp_cfg["num_classes"]),
            int(exp_cfg["target_class_id"]),
            strict=False,
        )
    else:
        co_metrics = _nan_subset_metrics()

    manifest_rows = read_csv_rows(ctx.paths.manifest_csv)
    quality_metrics = _compute_image_quality(ctx, manifest_rows)
    provenance = _sirc_manifest_provenance(ctx, manifest_rows)
    mechanism_evidence = _sirc_mechanism_evidence(ctx)
    if not all(
        np.isfinite(value)
        for value in (
            full_metrics["mAP50_target"],
            full_metrics["mAP50_non_target"],
            full_metrics["mAP50_all"],
            free_metrics["mAP50_non_target"],
            co_metrics["mAP50_non_target"],
        )
    ):
        raise ValueError("Evaluation produced a non-finite detection metric.")
    expected_poisoned = exp_cfg.get("expected_poisoned_count")
    if ctx.run_tag == "M1" and expected_poisoned is not None:
        if quality_metrics["poisoned_count"] != int(expected_poisoned):
            raise ValueError("M1 poisoned_count differs from the frozen protocol.")
        if quality_metrics["actual_linf_max"] > float(exp_cfg["eps"]) + 1.0 / 255.0:
            raise ValueError("M1 actual Linf exceeds the approved tolerance.")
    final_metrics = {
        "method": ctx.method,
        "steps": ctx.steps,
        "seed": ctx.seed,
        "run_tag": ctx.run_tag,
        "mAP50_target": full_metrics["mAP50_target"],
        "mAP50_non_target": full_metrics["mAP50_non_target"],
        "mAP50_non_target_macro": full_metrics["mAP50_non_target"],
        "mAP50_all": full_metrics["mAP50_all"],
        "ap50_by_class": full_metrics["ap50_by_class"],
        "voc20_class_names": list(VOC20_CLASS_NAMES),
        "AP_person_free_non_target": free_metrics["mAP50_non_target"],
        "AP_person_cooccur_non_target": co_metrics["mAP50_non_target"],
        "PSNR": quality_metrics["PSNR"],
        "LPIPS": quality_metrics["LPIPS"],
        "average_perturbed_area_ratio": quality_metrics["average_perturbed_area_ratio"],
        "average_support_area_ratio": quality_metrics.get("average_support_area_ratio", 0.0),
        "poisoned_count": int(quality_metrics.get("poisoned_count", 0)),
        "actual_linf_max": quality_metrics["actual_linf_max"],
        "actual_linf_mean": quality_metrics["actual_linf_mean"],
        "quality_validation_gaps": quality_metrics["validation_gaps"],
    }
    final_metrics.update(provenance)
    final_metrics.update(mechanism_evidence)

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

    psnr_display = (
        f"{final_metrics['PSNR']:.4f}"
        if final_metrics["PSNR"] is not None
        else "validation_gap"
    )
    print(
        "[evaluate] done: "
        f"mAP50_target={final_metrics['mAP50_target']:.4f}, "
        f"mAP50_non_target={final_metrics['mAP50_non_target']:.4f}, "
        f"PSNR={psnr_display}, area={final_metrics['average_perturbed_area_ratio']:.6f}, "
        f"poisoned_count={int(quality_metrics.get('poisoned_count', 0))}"
    )
