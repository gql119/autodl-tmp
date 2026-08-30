import hashlib
import json
import os
import random
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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
from ..io_utils import atomic_write_csv, atomic_write_json, read_csv_rows
from ..methods import build_generator
from ..methods.tausb_universal import TAUSBMaskGenerator, TAUSBUniversalTrainer
from ..methods.sirc_malc_cgr import resolve_sirc_malc_effective_method
from ..paths import ensure_run_dirs
from ..runtime import RunContext
from ..sparse_dataset import (
    SPARSE_REPORT_NAME,
    SPARSE_TRAIN_LIST_NAME,
    audit_sparse_training_list,
    file_sha256,
    is_sparse_mixed_list,
    png_roundtrip_exact,
    save_png_and_reload,
    write_train_path_list,
)
from ..status import (
    load_or_init_status,
    mark_stage_completed,
    mark_stage_running,
    save_status,
    stage_completed,
)


MANIFEST_FIELDS = [
    "stem",
    "image_path",
    "source_image_path",
    "label_path",
    "source_image_sha256",
    "label_sha256",
    "saved_image_sha256",
    "is_poisoned",
    "poisoned",  # 🚑 修复：去掉了错误的 out_img_path
    "has_target",
    "support_ratio",
    "support_outside_linf",
    "perturbed_area_ratio",
    "linf",
    "psnr",
    "method",
    "steps",
    "seed",
    "support_source",
    "state_content_hash",
    "semantic_bank_hash",
    "source_manifest_hash",
    "split_hash",
    "variant_index",
    "protocol_id",
    "evidence_scope",
    "hiding_gate_passed",
    "mechanism_gate_passed",
    "frozen_sdh_state_sha256",
    "hiding_metrics_sha256",
    "hiding_checkpoint_sha256",
    "hiding_split_sha256",
    "mechanism_metrics_sha256",
    "mechanism_decision_sha256",
    "mechanism_config_sha256",
    "p1_state_sha256",
    "source_arm_id",
    "state_integrity_gate_passed",
    "mechanism_scientific_gate_passed",
    "mechanism_scientific_decision_sha256",
    "state_integrity_decision_sha256",
    "p4_state_sha256",
    "source_p1_state_sha256",
    "source_p1_metrics_sha256",
    "d0_report_sha256",
    "repair_report_sha256",
    "secret_source_sha256",
]


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_train_image_selection(
    all_images: List[str],
    *,
    train_label_dir: str,
    selection_manifest_path: str,
    expected_manifest_sha256: str,
    target_class_id: int,
) -> Tuple[List[str], str]:
    if not selection_manifest_path:
        if expected_manifest_sha256:
            raise ValueError("Selection hash was set without a selection manifest.")
        return all_images, ""
    if not os.path.isabs(selection_manifest_path):
        raise ValueError("Train selection manifest must use an absolute path.")
    if not os.path.isfile(selection_manifest_path):
        raise FileNotFoundError(
            "Train selection manifest not found: %s" % selection_manifest_path
        )
    with open(selection_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual_manifest_sha256 = _canonical_json_sha256(manifest)
    if actual_manifest_sha256 != str(expected_manifest_sha256).lower():
        raise ValueError("Train selection manifest hash mismatch.")
    if manifest.get("schema") != "tausb.sdh-e2e-v0-train-selection.v1":
        raise ValueError("Unsupported train selection manifest schema.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Train selection manifest records are missing.")
    by_stem = {stem_of(path): path for path in all_images}
    if len(by_stem) != len(all_images):
        raise ValueError("Train image stems must be unique for selection.")
    selected = []
    seen = set()
    for record in records:
        stem = str(record.get("stem", ""))
        if not stem or stem in seen or stem not in by_stem:
            raise ValueError("Train selection contains a duplicate or unknown stem.")
        label_path = label_path_for_image(by_stem[stem], train_label_dir)
        if _file_sha256(label_path) != str(record.get("label_sha256", "")).lower():
            raise ValueError("Train selection label hash mismatch for %s." % stem)
        annotations = read_yolo_annotations(label_path)
        has_target = image_has_target(annotations, target_class_id)
        if bool(record.get("has_target")) is not has_target:
            raise ValueError("Train selection target label mismatch for %s." % stem)
        selected.append(by_stem[stem])
        seen.add(stem)
    return selected, actual_manifest_sha256


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


def resolve_effective_generation_method(
    requested_method: str,
    cfg: Dict,
) -> Tuple[str, Dict]:
    method_cfg = cfg["methods"][requested_method]
    if requested_method != "sirc_malc_cgr":
        return requested_method, method_cfg
    effective = resolve_sirc_malc_effective_method(method_cfg)
    if effective == "tausb_mask":
        return effective, cfg["methods"]["tausb_mask"]
    return effective, method_cfg


def run_generate_poisoned_dataset(ctx: RunContext) -> None:
    cfg = ctx.cfg
    paths = ctx.paths
    ensure_run_dirs(paths)

    method_cfg = cfg["methods"][ctx.method]
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
    sparse_mode = is_sparse_mixed_list(cfg)
    expected_poisoned_count = cfg["experiment"].get("expected_poisoned_count")
    needs_generator = not sparse_mode or int(expected_poisoned_count or 0) > 0
    generator = None
    if needs_generator:
        device = select_device(ctx.gpu_id)
        surrogate_ckpt = cfg["surrogate"]["ckpt"]
        yolo_wrapper = YOLO(surrogate_ckpt)
        _assert_surrogate_alignment(yolo_wrapper, cfg)
        surrogate = yolo_wrapper.model.to(device)
        surrogate.eval()

        effective_method, effective_method_cfg = resolve_effective_generation_method(
            ctx.method,
            cfg,
        )

        if effective_method == "tausb_mask":
            generator = _build_tausb_generator(
                ctx,
                cfg,
                effective_method_cfg,
                device,
                surrogate,
                poison_status,
                artifact_status,
            )
        else:
            generator = build_generator(
                effective_method, cfg, effective_method_cfg, device, surrogate
            )

    train_img_dir = ctx.train_img_dir
    train_label_dir = ctx.train_label_dir
    all_images = list_images(train_img_dir)
    all_images, selection_manifest_sha256 = resolve_train_image_selection(
        all_images,
        train_label_dir=train_label_dir,
        selection_manifest_path=str(cfg["data"].get("train_selection_manifest", "")),
        expected_manifest_sha256=str(
            cfg["data"].get("train_selection_manifest_sha256", "")
        ),
        target_class_id=int(cfg["experiment"]["target_class_id"]),
    )
    total_images = len(all_images)
    expected_train_images = cfg["experiment"].get("expected_train_images")
    if expected_train_images is not None and total_images != int(expected_train_images):
        raise RuntimeError(
            "Selected train image count differs from the frozen protocol: "
            f"actual={total_images}, expected={expected_train_images}."
        )

    existing_rows = read_csv_rows(paths.manifest_csv)
    if sparse_mode and existing_rows:
        raise RuntimeError(
            "Sparse E2E V0 generation forbids partial resume; use a fresh run root."
        )
    done_stems = set(r["stem"] for r in existing_rows)
    rows: List[Dict] = existing_rows[:]

    save_every = int(cfg["platform"].get("save_every_n_images", 50))
    poisoning_ratio = float(cfg["experiment"].get("poisoning_ratio", 1.0))
    if sparse_mode and poisoning_ratio not in {0.0, 1.0}:
        raise ValueError("Sparse E2E V0 supports only the frozen C0/M1 poison ratios.")
    sparse_train_list = os.path.join(paths.poisoned_root, SPARSE_TRAIN_LIST_NAME)
    clean_roundtrip_probes = 0
    source_image_hashes: Dict[str, str] = {}
    source_label_hashes: Dict[str, str] = {}

    viz_saved = len([n for n in os.listdir(paths.viz_dir) if n.endswith("_quad.png")])
    noise_meta_rows = []

    processed_since_last_flush = 0
    target_image_count = 0
    for idx, img_path in enumerate(all_images):
        stem = stem_of(img_path)
        if stem in done_stems:
            continue

        label_path = label_path_for_image(img_path, train_label_dir)
        if sparse_mode and not os.path.isfile(label_path):
            raise FileNotFoundError("Sparse E2E V0 label is missing: %s" % label_path)
        anns = read_yolo_annotations(label_path)
        has_target = image_has_target(anns, cfg["experiment"]["target_class_id"])
        target_image_count += int(has_target)

        # ---------------------------------------------------------
        # 🚀 强制改为 .png 格式并确定输出路径
        # ---------------------------------------------------------
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if sparse_mode:
            out_img_path = os.path.join(
                paths.poisoned_root, "images", "poisoned", stem + ".png"
            )
            out_label_path = os.path.join(
                paths.poisoned_root, "labels", "poisoned", stem + ".txt"
            )
        else:
            out_img_path = os.path.join(paths.poisoned_images, stem + ".png")
            out_label_path = os.path.join(
                paths.poisoned_labels, os.path.basename(label_path)
            )

        support_source = "none"
        state_content_hash = ""
        semantic_bank_hash = ""
        source_manifest_hash = ""
        split_hash = ""
        variant_index = ""
        sdh_provenance = {
            key: ""
            for key in (
                "protocol_id",
                "evidence_scope",
                "hiding_gate_passed",
                "mechanism_gate_passed",
                "frozen_sdh_state_sha256",
                "hiding_metrics_sha256",
                "hiding_checkpoint_sha256",
                "hiding_split_sha256",
                "mechanism_metrics_sha256",
                "mechanism_decision_sha256",
                "mechanism_config_sha256",
                "p1_state_sha256",
                "source_arm_id",
                "state_integrity_gate_passed",
                "mechanism_scientific_gate_passed",
                "mechanism_scientific_decision_sha256",
                "state_integrity_decision_sha256",
                "p4_state_sha256",
                "source_p1_state_sha256",
                "source_p1_metrics_sha256",
                "d0_report_sha256",
                "repair_report_sha256",
                "secret_source_sha256",
            )
        }
        
        # 🚑 修复：补回漏掉的 should_poison 定义
        should_poison = has_target and (
            poisoning_ratio == 1.0
            if sparse_mode
            else random.random() <= poisoning_ratio
        )

        if should_poison:
            if generator is None:
                raise RuntimeError("Sparse poison arm did not initialize its generator.")
            clean = load_image_rgb_float(img_path)
            result = generator.generate(
                image=clean,
                annotations=anns,
                seed=ctx.seed + idx,
                steps=ctx.steps,
                eps=cfg["experiment"]["eps"],
                support_type=support_type,
                image_path=img_path,
            )
            poisoned = result.poisoned_image
            perturb = result.perturbation
            support = result.support_mask

            support_source = str(result.extras.get("support_source", "unknown"))
            state_content_hash = str(result.extras.get("state_content_hash", ""))
            semantic_bank_hash = str(result.extras.get("semantic_bank_hash", ""))
            source_manifest_hash = str(result.extras.get("source_manifest_hash", ""))
            split_hash = str(result.extras.get("split_hash", ""))
            variant_index = str(result.extras.get("variant_index", ""))
            for key in sdh_provenance:
                sdh_provenance[key] = str(result.extras.get(key, ""))

            if bool(cfg["platform"].get("debug", False)) and viz_saved < 2:
                print(
                    f"[DEBUG][generate] stem={stem} "
                    f"clean={clean.shape} support={support.shape} perturb={perturb.shape}"
                )

            if sparse_mode:
                saved_poisoned, saved_image_sha256 = save_png_and_reload(
                    out_img_path, poisoned
                )
            else:
                _save_png_rgb_float(out_img_path, poisoned)
                saved_poisoned = poisoned
                saved_image_sha256 = ""
            copy_label(label_path, out_label_path)

            if sparse_mode:
                poisoned = saved_poisoned
                perturb = poisoned - clean
            linf = float(np.max(np.abs(perturb)))
            psnr = _calc_psnr(clean, poisoned)
            support_ratio = float(np.mean(support > 0.5))
            support_outside_linf = float(
                np.max(np.abs(perturb[support <= 0.5]))
            ) if np.any(support <= 0.5) else 0.0
            if sparse_mode and support_outside_linf != 0.0:
                raise RuntimeError(
                    "Saved sparse perturbation escaped person GT support for %s."
                    % stem
                )
            perturbed_area_ratio = _calc_area_ratio(clean, poisoned)
            actual_poisoned = bool(
                result.extras.get("is_poisoned", result.extras.get("poisoned", linf > (1.0 / 255.0)))
            )

            # 🚑 修复：加回可视化存储代码，方便我们在 viz_dir 中查看效果
            if viz_saved < 16:
                _save_quad_viz(
                    os.path.join(paths.viz_dir, f"{stem}_quad.png"),
                    clean,
                    support,
                    perturb,
                    poisoned,
                )
                if ctx.method in {"ours_mask", "tausb_mask", "sirc_malc_cgr"}:
                    _save_extra_viz(paths.viz_dir, stem, result.extras, result.support_mask, result.ring_mask)
                viz_saved += 1

            noise_meta = {
                "stem": stem,
                "losses": result.losses,
                "linf": linf,
                "is_poisoned": int(actual_poisoned),
            }
            noise_meta_rows.append(noise_meta)

        else:
            if sparse_mode:
                out_img_path = os.path.abspath(img_path)
                out_label_path = os.path.abspath(label_path)
                saved_image_sha256 = ""
                if clean_roundtrip_probes < 64:
                    clean = load_image_rgb_float(img_path)
                    if not png_roundtrip_exact(clean):
                        raise RuntimeError(
                            "Clean JPEG decode-to-PNG round-trip changed pixels for %s."
                            % stem
                        )
                    clean_roundtrip_probes += 1
                perturb = None
            else:
                clean = load_image_rgb_float(img_path)
                _save_png_rgb_float(out_img_path, clean)
                copy_label(label_path, out_label_path)
                saved_image_sha256 = _file_sha256(out_img_path)
                perturb = np.zeros_like(clean)
            support_ratio = 0.0
            support_outside_linf = 0.0
            perturbed_area_ratio = 0.0
            linf = 0.0
            psnr = 99.0
            support_source = "none"
            actual_poisoned = False

        if sparse_mode:
            source_image_hashes[stem] = file_sha256(img_path)
            source_label_hashes[stem] = file_sha256(label_path)
        row = {
            "stem": stem,
            "image_path": out_img_path,
            "source_image_path": os.path.abspath(img_path),
            "label_path": os.path.abspath(out_label_path),
            "source_image_sha256": source_image_hashes.get(stem, ""),
            "label_sha256": source_label_hashes.get(stem, ""),
            "saved_image_sha256": saved_image_sha256,
            "is_poisoned": "1" if actual_poisoned else "0",
            "poisoned": "1" if actual_poisoned else "0",
            "has_target": "1" if has_target else "0",
            "support_ratio": f"{support_ratio:.8f}",
            "support_outside_linf": f"{support_outside_linf:.8f}",
            "perturbed_area_ratio": f"{perturbed_area_ratio:.8f}",
            "linf": f"{linf:.8f}",
            "psnr": f"{psnr:.6f}",
            "method": ctx.method,
            "steps": str(ctx.steps),
            "seed": str(ctx.seed),
            "support_source": support_source,
            "state_content_hash": state_content_hash,
            "semantic_bank_hash": semantic_bank_hash,
            "source_manifest_hash": source_manifest_hash,
            "split_hash": split_hash,
            "variant_index": variant_index,
            **sdh_provenance,
        }
        rows.append(row)
        done_stems.add(stem)
        processed_since_last_flush += 1

        if processed_since_last_flush >= save_every:
            atomic_write_csv(paths.manifest_csv, rows, MANIFEST_FIELDS)
            atomic_write_json(os.path.join(paths.noise_dir, "noise_meta_last_flush.json"), {"items": noise_meta_rows})
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
            print(
                "[generate_poisoned_dataset] progress: "
                f"processed={len(done_stems)}/{total_images}, "
                f"poisoned={sum(int(str(row.get('poisoned', '0')) == '1') for row in rows)}"
            )
            processed_since_last_flush = 0

    atomic_write_csv(paths.manifest_csv, rows, MANIFEST_FIELDS)
    atomic_write_json(os.path.join(paths.noise_dir, "noise_meta.json"), {"items": noise_meta_rows})

    sparse_audit = None
    if sparse_mode:
        if clean_roundtrip_probes != 64:
            raise RuntimeError(
                "Sparse clean format gate requires exactly 64 probes; got %d."
                % clean_roundtrip_probes
            )
        ordered_effective_paths = [str(row["image_path"]) for row in rows]
        train_list_sha256 = write_train_path_list(
            sparse_train_list, ordered_effective_paths
        )
        sparse_audit = audit_sparse_training_list(
            sparse_train_list,
            rows,
            expected_total=total_images,
            expected_poisoned=int(expected_poisoned_count or 0),
            expected_target=int(cfg["experiment"].get("expected_target_images", 0)),
            target_class_id=int(cfg["experiment"]["target_class_id"]),
            num_classes=int(cfg["experiment"]["num_classes"]),
            verify_hashes=False,
        )
        if sparse_audit["train_list_sha256"] != train_list_sha256:
            raise RuntimeError("Sparse train list changed during its audit.")
        sparse_audit.update(
            {
                "materialization_layout": cfg["data"]["materialization_layout"],
                "manifest_csv": os.path.abspath(paths.manifest_csv),
                "clean_png_roundtrip_probe_count": clean_roundtrip_probes,
            }
        )
        atomic_write_json(
            os.path.join(paths.poisoned_root, SPARSE_REPORT_NAME), sparse_audit
        )

    actual_poisoned_count = sum(
        int(str(row.get("poisoned", "0")) == "1") for row in rows
    )
    if expected_poisoned_count is not None and actual_poisoned_count != int(expected_poisoned_count):
        raise RuntimeError(
            "Poisoned image count differs from the frozen protocol: "
            f"actual={actual_poisoned_count}, expected={expected_poisoned_count}."
        )
    expected_target_images = cfg["experiment"].get("expected_target_images")
    if expected_target_images is not None and target_image_count != int(expected_target_images):
        raise RuntimeError(
            "Selected target-image count differs from the frozen protocol: "
            f"actual={target_image_count}, expected={expected_target_images}."
        )

    mark_stage_completed(
        paths.poisoned_status_json,
        poison_status,
        "generate_poisoned_dataset",
        {
            "processed": len(done_stems),
            "total": total_images,
            "poisoned_count": actual_poisoned_count,
            "target_image_count": target_image_count,
            "selection_manifest_sha256": selection_manifest_sha256,
            "manifest_csv": paths.manifest_csv,
            "sparse_train_list": sparse_train_list if sparse_mode else "",
            "sparse_audit": sparse_audit or {},
        },
    )
    mark_stage_completed(
        paths.artifact_status_json,
        artifact_status,
        "generate_poisoned_dataset",
        {
            "processed": len(done_stems),
            "total": total_images,
            "poisoned_count": actual_poisoned_count,
            "target_image_count": target_image_count,
            "selection_manifest_sha256": selection_manifest_sha256,
            "manifest_csv": paths.manifest_csv,
            "sparse_train_list": sparse_train_list if sparse_mode else "",
            "sparse_audit": sparse_audit or {},
        },
    )

    print(
        "[generate_poisoned_dataset] done: "
        f"processed={len(done_stems)}/{total_images}, method={ctx.method}, steps={ctx.steps}, seed={ctx.seed}"
    )
