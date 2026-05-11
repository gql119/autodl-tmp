import os
import random
import shutil
import json
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ..utils.robust_transforms import apply_poison_post_transform
from ..data_utils import (
    copy_label,
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
    save_image_rgb_float,
    stem_of,
)
from ..env_utils import collect_runtime_info, select_device, set_global_seed
from ..io_utils import atomic_write_csv, atomic_write_json, atomic_write_text, read_csv_rows
from ..methods import build_generator
from ..methods.tausb_universal import TAUSBMaskGenerator, TAUSBUniversalTrainer
from ..paths import ensure_run_dirs
from ..runtime import RunContext
from ..status import (
    load_or_init_status,
    mark_stage_completed,
    mark_stage_running,
    save_status,
    stage_completed,
)
from ..utils.poison_stats import LPIPSComputer, compute_image_poison_stats


MANIFEST_FIELDS = [
    "stem",
    "image_path",
    "is_poisoned",
    "poisoned",  # 🚑 修复：去掉了错误的 out_img_path
    "has_target",
    "support_ratio",
    "perturbed_area_ratio",
    "linf",
    "psnr",
    "lpips",
    "method",
    "steps",
    "seed",
    "support_source",
    "objective",
    "det_loss_mode",
    "steps_eff",
]


def _save_quad_viz(path: str, clean: np.ndarray, support: np.ndarray, perturb: np.ndarray, poisoned: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_u8 = (np.clip(clean, 0, 1) * 255).astype(np.uint8)
    poisoned_u8 = (np.clip(poisoned, 0, 1) * 255).astype(np.uint8)

    support_rgb = np.repeat((np.clip(support, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    pert_vis = np.clip((perturb + 0.5) * 255.0, 0, 255).astype(np.uint8)

    panel = np.concatenate([clean_u8, support_rgb, pert_vis, poisoned_u8], axis=1)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, panel)


def _save_extra_viz(viz_dir: str, stem: str, extras: Dict, support: np.ndarray, ring: np.ndarray):
    if not extras:
        return
    if "spectrum" in extras:
        spec = extras["spectrum"]
        spec_u8 = (np.clip(spec, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(viz_dir, f"{stem}_spectrum.png"), spec_u8)
    if "jnd_gain" in extras:
        jnd = extras["jnd_gain"]
        jnd_u8 = (np.clip(jnd, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(viz_dir, f"{stem}_jnd.png"), jnd_u8)

    support_u8 = (np.clip(support, 0, 1) * 255).astype(np.uint8)
    ring_u8 = (np.clip(ring, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_inner_mask.png"), support_u8)
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_ring_mask.png"), ring_u8)


def _assert_surrogate_alignment(yolo_wrapper: YOLO, cfg: Dict):
    expected = int(cfg["surrogate"]["num_classes"])
    names = getattr(yolo_wrapper, "names", None)
    if isinstance(names, dict):
        got = len(names)
    elif isinstance(names, list):
        got = len(names)
    else:
        got = expected

    if got != expected:
        raise RuntimeError(
            "Surrogate class space mismatch. "
            f"expected num_classes={expected}, but model has {got}. "
            "This check prevents VOC/COCO index mismatch."
        )


def _calc_psnr(clean: np.ndarray, poisoned: np.ndarray) -> float:
    mse = float(np.mean((clean - poisoned) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))

def _save_png_rgb_float(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _calc_area_ratio(clean: np.ndarray, poisoned: np.ndarray) -> float:
    diff = np.max(np.abs(clean - poisoned), axis=2)
    return float(np.mean(diff > (1.0 / 255.0)))


def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict] = []
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


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text = text + "\n"
    atomic_write_text(path, text)


def _build_tausb_generator(
    ctx: RunContext,
    cfg: Dict,
    method_cfg: Dict,
    device: torch.device,
    surrogate,
    poison_status: Dict,
    artifact_status: Dict,
):
    global_params_path = os.path.join(ctx.paths.noise_dir, "global_params.pt")
    diagnostics_csv = os.path.join(ctx.paths.logs_dir, "tausb_universal_diagnostics.csv")
    diagnostics_json = os.path.join(ctx.paths.logs_dir, "tausb_universal_diagnostics.json")

    if not os.path.isfile(global_params_path):
        trainer = TAUSBUniversalTrainer(cfg, method_cfg, device, surrogate)
        print("[TAUSB] training global universal params ...")
        trainer.train_universal(
            train_img_dir=ctx.train_img_dir,
            train_label_dir=ctx.train_label_dir,
            global_params_path=global_params_path,
            diagnostics_csv_path=diagnostics_csv,
            diagnostics_json_path=diagnostics_json,
            seed=ctx.seed,
        )
        poison_status["stage_state"]["generate_poisoned_dataset"]["tausb_global_params"] = global_params_path
        artifact_status["stage_state"]["generate_poisoned_dataset"]["tausb_global_params"] = global_params_path
        save_status(ctx.paths.poisoned_status_json, poison_status)
        save_status(ctx.paths.artifact_status_json, artifact_status)

    return TAUSBMaskGenerator(cfg, method_cfg, device, surrogate, global_params_path=global_params_path)


def run_generate_poisoned_dataset(ctx: RunContext) -> None:
    cfg = ctx.cfg
    paths = ctx.paths
    ensure_run_dirs(paths)

    methods_cfg = cfg.get("methods", {})
    method_cfg = methods_cfg.get(ctx.method, cfg.get(ctx.method, {}))
    if not isinstance(method_cfg, dict):
        raise ValueError(f"Invalid method config for {ctx.method}. Expected dict, got {type(method_cfg)}")
    support_type = method_cfg.get("support_type", "mask")

    poison_status = load_or_init_status(paths.poisoned_status_json, ctx.method, ctx.steps, ctx.seed)
    artifact_status = load_or_init_status(paths.artifact_status_json, ctx.method, ctx.steps, ctx.seed)

    if cfg["platform"].get("resume", True) and stage_completed(poison_status, "generate_poisoned_dataset"):
        print("[generate_poisoned_dataset] already completed, skipping.")
        return

    poison_status = mark_stage_running(paths.poisoned_status_json, poison_status, "generate_poisoned_dataset")
    artifact_status = mark_stage_running(paths.artifact_status_json, artifact_status, "generate_poisoned_dataset")

    runtime_info = collect_runtime_info(ctx.seed, ctx.method, ctx.steps)
    atomic_write_json(os.path.join(paths.logs_dir, "runtime_info.json"), runtime_info)

    set_global_seed(ctx.seed)
    device = select_device(ctx.gpu_id)
    lpips_computer = LPIPSComputer(device)

    surrogate_ckpt = cfg["surrogate"]["ckpt"]
    yolo_wrapper = YOLO(surrogate_ckpt)
    _assert_surrogate_alignment(yolo_wrapper, cfg)
    surrogate = yolo_wrapper.model.to(device)
    surrogate.eval()

    if ctx.method == "tausb_mask":
        generator = _build_tausb_generator(ctx, cfg, method_cfg, device, surrogate, poison_status, artifact_status)
    else:
        generator = build_generator(ctx.method, cfg, method_cfg, device, surrogate)

    train_img_dir = ctx.train_img_dir
    train_label_dir = ctx.train_label_dir
    all_images = list_images(train_img_dir)
    total_images = len(all_images)

    existing_rows = read_csv_rows(paths.manifest_csv)
    done_stems = set(r["stem"] for r in existing_rows)
    rows: List[Dict] = existing_rows[:]
    poison_stats_jsonl = os.path.join(paths.artifact_root, "poison_stats.jsonl")
    poison_stats_rows: List[Dict] = (
        _read_jsonl(poison_stats_jsonl) if cfg["platform"].get("resume", True) else []
    )
    if len(existing_rows) == 0:
        poison_stats_rows = []

    save_every = int(cfg["platform"].get("save_every_n_images", 50))
    poisoning_ratio = float(cfg["experiment"].get("poisoning_ratio", 1.0))

    viz_saved = len([n for n in os.listdir(paths.viz_dir) if n.endswith("_quad.png")])
    noise_meta_rows = []
    poison_batch_size = int(method_cfg.get("poison_batch_size", 1))
    has_batch_api = callable(getattr(generator, "generate_batch", None))
    use_batch_generation = bool(has_batch_api and poison_batch_size > 1)

    processed_since_last_flush = 0
    pending_poison_items: List[Dict] = []

    def _flush_partial_state():
        nonlocal processed_since_last_flush
        if processed_since_last_flush < save_every:
            return

        atomic_write_csv(paths.manifest_csv, rows, MANIFEST_FIELDS)
        atomic_write_json(os.path.join(paths.noise_dir, "noise_meta_last_flush.json"), {"items": noise_meta_rows})
        _write_jsonl(poison_stats_jsonl, poison_stats_rows)
        poison_status["stage_state"]["generate_poisoned_dataset"].update(
            {
                "processed": len(done_stems),
                "total": total_images,
            }
        )
        artifact_status["stage_state"]["generate_poisoned_dataset"].update(
            {
                "processed": len(done_stems),
                "total": total_images,
            }
        )
        save_status(paths.poisoned_status_json, poison_status)
        save_status(paths.artifact_status_json, artifact_status)
        processed_since_last_flush = 0

    def _record_poisoned_item(item: Dict, result):
        nonlocal viz_saved, processed_since_last_flush, poison_stats_rows
        clean = item["clean"]
        stem = item["stem"]
        label_path = item["label_path"]
        out_img_path = item["out_img_path"]
        out_label_path = item["out_label_path"]
        has_target = item["has_target"]

        poisoned = result.poisoned_image
        perturb = result.perturbation
        support = result.support_mask
        support_source = str(result.extras.get("support_source", "unknown"))

        if bool(cfg["platform"].get("debug", False)) and viz_saved < 2:
            print(
                f"[DEBUG][generate] stem={stem} "
                f"clean={clean.shape} support={support.shape} perturb={perturb.shape}"
            )

        _save_png_rgb_float(out_img_path, poisoned)
        copy_label(label_path, out_label_path)

        stats = compute_image_poison_stats(
            clean_img=clean,
            poison_img=poisoned,
            support_mask=support,
            threshold=(1.0 / 255.0),
        )
        lpips_val = lpips_computer(clean, poisoned)
        stats["lpips"] = lpips_val

        # Force unified stats as source of truth for all methods.
        if result.extras is None:
            result.extras = {}
        result.extras.update(stats)

        linf = float(stats["delta_linf"])
        psnr = float(stats["psnr"])
        lpips = stats.get("lpips", None)
        support_ratio = float(stats["support_area_ratio"])
        perturbed_area_ratio = float(stats["changed_pixel_ratio"])
        actual_poisoned = bool(stats["is_poisoned"])

        if viz_saved < 16:
            _save_quad_viz(
                os.path.join(paths.viz_dir, f"{stem}_quad.png"),
                clean,
                support,
                perturb,
                poisoned,
            )
            if ctx.method in {"ours_mask", "tausb_mask"}:
                _save_extra_viz(paths.viz_dir, stem, result.extras, result.support_mask, result.ring_mask)
            viz_saved += 1

        noise_meta_rows.append(
            {
                "stem": stem,
                "losses": result.losses,
                "linf": linf,
                "is_poisoned": int(actual_poisoned),
            }
        )

        poison_stats_rows.append(
            {
                "image_path": out_img_path,
                "method": ctx.method,
                "seed": int(ctx.seed),
                "steps": int(ctx.steps),
                "poisoned": int(stats["poisoned"]),
                "is_poisoned": bool(stats["is_poisoned"]),
                "delta_linf": float(stats["delta_linf"]),
                "mse": float(stats["mse"]),
                "psnr": float(stats["psnr"]),
                "lpips": (None if lpips is None else float(lpips)),
                "changed_pixel_ratio": float(stats["changed_pixel_ratio"]),
                "support_area_ratio": float(stats["support_area_ratio"]),
                "objective": str(result.extras.get("objective", "")),
                "det_loss_mode": str(result.extras.get("det_loss_mode", "")),
                "steps_eff": int(result.extras.get("steps_eff", ctx.steps)),
            }
        )

        row = {
            "stem": stem,
            "image_path": out_img_path,
            "is_poisoned": "1" if actual_poisoned else "0",
            "poisoned": "1" if actual_poisoned else "0",
            "has_target": "1" if has_target else "0",
            "support_ratio": f"{support_ratio:.8f}",
            "perturbed_area_ratio": f"{perturbed_area_ratio:.8f}",
            "linf": f"{linf:.8f}",
            "psnr": f"{psnr:.6f}",
            "lpips": "" if lpips is None else f"{float(lpips):.6f}",
            "method": ctx.method,
            "steps": str(ctx.steps),
            "seed": str(ctx.seed),
            "support_source": support_source,
            "objective": str(result.extras.get("objective", "")),
            "det_loss_mode": str(result.extras.get("det_loss_mode", "")),
            "steps_eff": str(int(result.extras.get("steps_eff", ctx.steps))),
        }
        rows.append(row)
        done_stems.add(stem)
        processed_since_last_flush += 1
        _flush_partial_state()

    def _flush_pending_poison_items():
        nonlocal pending_poison_items
        if len(pending_poison_items) == 0:
            return

        if use_batch_generation:
            batch_results = generator.generate_batch(
                images=[x["clean"] for x in pending_poison_items],
                annotations_list=[x["anns"] for x in pending_poison_items],
                image_paths=[x["img_path"] for x in pending_poison_items],
                seed=ctx.seed + int(pending_poison_items[0]["idx"]),
                steps=ctx.steps,
                eps=cfg["experiment"]["eps"],
                support_type=support_type,
            )
            if len(batch_results) != len(pending_poison_items):
                raise RuntimeError(
                    f"generate_batch returned {len(batch_results)} results "
                    f"for {len(pending_poison_items)} inputs."
                )
            for item, result in zip(pending_poison_items, batch_results):
                _record_poisoned_item(item, result)
        else:
            for item in pending_poison_items:
                result = generator.generate(
                    image=item["clean"],
                    annotations=item["anns"],
                    seed=ctx.seed + int(item["idx"]),
                    steps=ctx.steps,
                    eps=cfg["experiment"]["eps"],
                    support_type=support_type,
                    image_path=item["img_path"],
                )

                has_poison_support = False
                try:
                    has_poison_support = (
                        result.support_mask is not None
                        and float(np.asarray(result.support_mask).sum()) > 0.0
                    )
                except Exception:
                    has_poison_support = False

                result.poisoned_image, post_transform_info = apply_poison_post_transform(
                    result.poisoned_image,
                    ctx.cfg,
                    has_poison_support=has_poison_support,
                )

                # 如果后处理返回 uint8，这里统一转回 float32 [0,1]
                if isinstance(result.poisoned_image, np.ndarray) and result.poisoned_image.dtype == np.uint8:
                    result.poisoned_image = result.poisoned_image.astype(np.float32) / 255.0

                if not hasattr(result, "extras") or result.extras is None:
                    result.extras = {}

                result.extras.update(post_transform_info)

                _record_poisoned_item(item, result)

        pending_poison_items = []

    for idx, img_path in enumerate(all_images):
        stem = stem_of(img_path)
        if stem in done_stems:
            continue

        clean = load_image_rgb_float(img_path)
        label_path = label_path_for_image(img_path, train_label_dir)
        anns = read_yolo_annotations(label_path)
        has_target = image_has_target(anns, cfg["experiment"]["target_class_id"])

        # ---------------------------------------------------------
        # 🚀 强制改为 .png 格式并确定输出路径
        # ---------------------------------------------------------
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(paths.poisoned_images, stem + ".png")
        out_label_path = os.path.join(paths.poisoned_labels, os.path.basename(label_path)) # 🚑 修复：存入 poisoned_labels

        # 🚑 修复：补回漏掉的 should_poison 定义
        should_poison = has_target and (random.random() <= poisoning_ratio)

        if should_poison:
            pending_poison_items.append(
                {
                    "idx": idx,
                    "img_path": img_path,
                    "stem": stem,
                    "clean": clean,
                    "anns": anns,
                    "label_path": label_path,
                    "out_img_path": out_img_path,
                    "out_label_path": out_label_path,
                    "has_target": has_target,
                }
            )
            if len(pending_poison_items) >= max(1, poison_batch_size):
                _flush_pending_poison_items()

        else:
            _flush_pending_poison_items()
            _save_png_rgb_float(out_img_path, clean)
            copy_label(label_path, out_label_path)

            stats = compute_image_poison_stats(
                clean_img=clean,
                poison_img=clean,
                support_mask=None,
                threshold=(1.0 / 255.0),
            )
            lpips_val = lpips_computer(clean, clean)
            support_ratio = float(stats["support_area_ratio"])
            perturbed_area_ratio = float(stats["changed_pixel_ratio"])
            linf = float(stats["delta_linf"])
            psnr = float(stats["psnr"])
            support_source = "none"
            actual_poisoned = bool(stats["is_poisoned"])
            poison_stats_rows.append(
                {
                    "image_path": out_img_path,
                    "method": ctx.method,
                    "seed": int(ctx.seed),
                    "steps": int(ctx.steps),
                    "poisoned": int(stats["poisoned"]),
                    "is_poisoned": bool(stats["is_poisoned"]),
                    "delta_linf": float(stats["delta_linf"]),
                    "mse": float(stats["mse"]),
                    "psnr": float(stats["psnr"]),
                    "lpips": (None if lpips_val is None else float(lpips_val)),
                    "changed_pixel_ratio": float(stats["changed_pixel_ratio"]),
                    "support_area_ratio": float(stats["support_area_ratio"]),
                    "objective": "none",
                    "det_loss_mode": "none",
                    "steps_eff": int(ctx.steps),
                }
            )
            row = {
                "stem": stem,
                "image_path": out_img_path,
                "is_poisoned": "1" if actual_poisoned else "0",
                "poisoned": "1" if actual_poisoned else "0",
                "has_target": "1" if has_target else "0",
                "support_ratio": f"{support_ratio:.8f}",
                "perturbed_area_ratio": f"{perturbed_area_ratio:.8f}",
                "linf": f"{linf:.8f}",
                "psnr": f"{psnr:.6f}",
                "lpips": "" if lpips_val is None else f"{float(lpips_val):.6f}",
                "method": ctx.method,
                "steps": str(ctx.steps),
                "seed": str(ctx.seed),
                "support_source": support_source,
                "objective": "none",
                "det_loss_mode": "none",
                "steps_eff": str(int(ctx.steps)),
            }
            rows.append(row)
            done_stems.add(stem)
            processed_since_last_flush += 1
            _flush_partial_state()

    _flush_pending_poison_items()

    atomic_write_csv(paths.manifest_csv, rows, MANIFEST_FIELDS)
    atomic_write_json(os.path.join(paths.noise_dir, "noise_meta.json"), {"items": noise_meta_rows})
    _write_jsonl(poison_stats_jsonl, poison_stats_rows)

    mark_stage_completed(
        paths.poisoned_status_json,
        poison_status,
        "generate_poisoned_dataset",
        {
            "processed": len(done_stems),
            "total": total_images,
            "manifest_csv": paths.manifest_csv,
            "poison_stats_jsonl": poison_stats_jsonl,
        },
    )
    mark_stage_completed(
        paths.artifact_status_json,
        artifact_status,
        "generate_poisoned_dataset",
        {
            "processed": len(done_stems),
            "total": total_images,
            "manifest_csv": paths.manifest_csv,
            "poison_stats_jsonl": poison_stats_jsonl,
        },
    )

    print(
        "[generate_poisoned_dataset] done: "
        f"processed={len(done_stems)}/{total_images}, method={ctx.method}, steps={ctx.steps}, seed={ctx.seed}, "
        f"poison_stats_jsonl={poison_stats_jsonl}"
    )
