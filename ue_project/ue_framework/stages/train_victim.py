import csv
import json
import os
import shutil

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset

from ..env_utils import resolve_workers, set_global_seed
from ..io_utils import atomic_write_json, read_csv_rows
from ..metrics_utils import VOC20_CLASS_NAMES
from ..pack import pack_run_artifacts
from ..runtime import RunContext
from ..sparse_dataset import (
    SPARSE_REPORT_NAME,
    SPARSE_TRAIN_LIST_NAME,
    audit_sparse_training_list,
    is_sparse_mixed_list,
)
from ..status import (
    load_or_init_status,
    mark_stage_completed,
    mark_stage_running,
    stage_completed,
)
from ..train_utils import copy_if_exists, remove_yolo_cache_files



def _sparse_train_audit(ctx: RunContext) -> dict:
    train_list = os.path.join(ctx.paths.poisoned_root, SPARSE_TRAIN_LIST_NAME)
    audit = audit_sparse_training_list(
        train_list,
        read_csv_rows(ctx.paths.manifest_csv),
        expected_total=int(ctx.cfg["experiment"]["expected_train_images"]),
        expected_poisoned=int(ctx.cfg["experiment"]["expected_poisoned_count"]),
        expected_target=int(ctx.cfg["experiment"]["expected_target_images"]),
        target_class_id=int(ctx.cfg["experiment"]["target_class_id"]),
        num_classes=int(ctx.cfg["experiment"]["num_classes"]),
    )
    generation_report_path = os.path.join(
        ctx.paths.poisoned_root, SPARSE_REPORT_NAME
    )
    if not os.path.isfile(generation_report_path):
        raise FileNotFoundError(
            "Sparse generation report is missing: %s" % generation_report_path
        )
    with open(generation_report_path, "r", encoding="utf-8") as handle:
        generation_report = json.load(handle)
    for key in (
        "train_list_sha256",
        "ordered_stems_sha256",
        "label_content_manifest_sha256",
        "total_count",
        "target_count",
        "poisoned_count",
        "poisoned_png_count",
        "original_jpeg_count",
    ):
        if audit[key] != generation_report.get(key):
            raise ValueError("Sparse generation/train audit differs for %s." % key)
    return audit


def _write_train_yaml(ctx: RunContext, sparse_audit: dict = None) -> str:
    yaml_path = os.path.join(ctx.paths.artifact_root, "train_data.yaml")
    if sparse_audit is None:
        data_root = ctx.paths.poisoned_root
        train_source = "images/train"
    else:
        data_root = ctx.dataset_root
        train_source = str(sparse_audit["train_list_path"])
    content = f"""path: {json.dumps(os.path.abspath(data_root))}
train: {json.dumps(train_source)}
val: {json.dumps(os.path.abspath(ctx.val_img_dir))}
names:
"""
    num_classes = int(ctx.cfg["experiment"]["num_classes"])
    if num_classes != len(VOC20_CLASS_NAMES):
        raise ValueError("The approved victim trainer requires the full VOC20 class space.")
    for i, cls_name in enumerate(VOC20_CLASS_NAMES):
        content += f"  {i}: {cls_name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def _probe_sparse_dataloader(ctx: RunContext, sparse_audit: dict) -> dict:
    train_list = str(sparse_audit["train_list_path"])
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "mode": "train",
            "imgsz": int(ctx.cfg["victim"]["imgsz"]),
            "cache": False,
            "workers": 0,
            "rect": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "close_mosaic": 0,
        }
    )
    names = {index: name for index, name in enumerate(VOC20_CLASS_NAMES)}
    dataset = build_yolo_dataset(
        cfg,
        train_list,
        batch=1,
        data={"names": names, "nc": len(names)},
        mode="train",
        rect=True,
        stride=32,
    )
    if len(dataset) != int(sparse_audit["total_count"]):
        raise ValueError("Ultralytics sparse dataset silently changed the image count.")
    loader = build_dataloader(
        dataset,
        batch=1,
        workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    batch = next(iter(loader))
    images = batch.get("img")
    classes = batch.get("cls")
    if images is None or int(images.shape[0]) != 1:
        raise ValueError("Ultralytics sparse dataloader probe did not yield one image.")
    if classes is None:
        raise ValueError("Ultralytics sparse dataloader probe omitted class labels.")
    if classes.numel() and (
        int(classes.min()) < 0 or int(classes.max()) >= len(VOC20_CLASS_NAMES)
    ):
        raise ValueError("Ultralytics sparse dataloader yielded an invalid class id.")
    return {
        "dataset_count": len(dataset),
        "batch_image_shape": [int(value) for value in images.shape],
        "batch_label_count": int(classes.numel()),
    }



def _count_latest_epoch(results_csv: str) -> int:
    if not os.path.isfile(results_csv):
        return 0
    with open(results_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return 0
    return len(rows) - 2



def _snapshot_train_state(ctx: RunContext, run_dir: str, latest_ckpt_ultra: str, best_ckpt_ultra: str) -> int:
    os.makedirs(ctx.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(ctx.paths.logs_dir, exist_ok=True)

    latest_ckpt = os.path.join(ctx.paths.checkpoints_dir, "latest.pt")
    best_ckpt = os.path.join(ctx.paths.checkpoints_dir, "best.pt")
    copy_if_exists(latest_ckpt_ultra, latest_ckpt)
    copy_if_exists(best_ckpt_ultra, best_ckpt)

    results_csv = os.path.join(run_dir, "results.csv")
    latest_epoch = _count_latest_epoch(results_csv)
    if os.path.isfile(results_csv):
        shutil.copy2(results_csv, os.path.join(ctx.paths.logs_dir, "train_log.csv"))

    live_summary = {
        "method": ctx.method,
        "steps": ctx.steps,
        "seed": ctx.seed,
        "run_tag": ctx.run_tag,
        "latest_epoch": latest_epoch,
        "latest_checkpoint": latest_ckpt if os.path.isfile(latest_ckpt) else "",
        "best_checkpoint": best_ckpt if os.path.isfile(best_ckpt) else "",
        "train_run_dir": run_dir,
    }
    atomic_write_json(os.path.join(ctx.paths.logs_dir, "train_stage_live_summary.json"), live_summary)
    return latest_epoch



def _prepare_train_run_dir(ctx: RunContext, run_dir: str, resume_enabled: bool) -> None:
    if resume_enabled:
        os.makedirs(run_dir, exist_ok=True)
        return

    abs_run_dir = os.path.abspath(run_dir)
    abs_artifact_root = os.path.abspath(ctx.paths.artifact_root)
    if not abs_run_dir.startswith(abs_artifact_root):
        raise RuntimeError(
            "Refuse to clean train run dir outside artifact root. "
            f"run_dir={abs_run_dir}, artifact_root={abs_artifact_root}"
        )

    if os.path.exists(abs_run_dir):
        raise FileExistsError(
            "Fresh victim training refuses an existing run directory: "
            f"{abs_run_dir}"
        )
    os.makedirs(abs_run_dir, exist_ok=True)



def run_train_victim(ctx: RunContext) -> None:
    cfg = ctx.cfg
    victim_cfg = cfg["victim"]
    resume_enabled = bool(cfg["platform"].get("resume", True))

    # C0 and M1 are independent fresh victims under one matched seed. Seed
    # before constructing YOLO so model initialization is matched as well as
    # the subsequent Ultralytics training loop.
    set_global_seed(ctx.seed)

    status = load_or_init_status(ctx.paths.artifact_status_json, ctx.method, ctx.steps, ctx.seed)
    if resume_enabled and stage_completed(status, "train_victim"):
        print("[train_victim] already completed, skipping.")
        return

    status = mark_stage_running(ctx.paths.artifact_status_json, status, "train_victim")

    sparse_audit = _sparse_train_audit(ctx) if is_sparse_mixed_list(cfg) else None
    sparse_dataloader_probe = (
        _probe_sparse_dataloader(ctx, sparse_audit) if sparse_audit else None
    )
    yaml_path = _write_train_yaml(ctx, sparse_audit=sparse_audit)
    deleted_cache = 0 if sparse_audit else remove_yolo_cache_files(
        [
            ctx.paths.poisoned_root,
            ctx.dataset_root,
            ctx.train_img_dir,
            ctx.val_img_dir,
            ctx.train_label_dir,
            ctx.val_label_dir,
        ]
    )

    train_project = ctx.paths.train_project_dir
    run_name = "victim"
    run_dir = os.path.join(train_project, run_name)
    _prepare_train_run_dir(ctx, run_dir, resume_enabled=resume_enabled)

    latest_ckpt_ultra = os.path.join(run_dir, "weights", "last.pt")
    best_ckpt_ultra = os.path.join(run_dir, "weights", "best.pt")

    workers = resolve_workers(ctx.platform_mode, cfg)

    save_every = int(cfg["platform"].get("save_every_n_epochs", victim_cfg.get("save_period", 5)))
    pack_every = int(cfg["platform"].get("pack_every_n_epochs", save_every))

    train_args = dict(
        data=yaml_path,
        epochs=int(victim_cfg["epochs"]),
        imgsz=int(victim_cfg["imgsz"]),
        batch=int(victim_cfg["batch"]),
        workers=int(workers),
        project=train_project,
        name=run_name,
        exist_ok=True,
        optimizer=victim_cfg["optimizer"],
        cos_lr=bool(victim_cfg["cos_lr"]),
        close_mosaic=int(victim_cfg["close_mosaic"]),
        cache=bool(victim_cfg["cache"]),
        amp=bool(victim_cfg["amp"]),
        save_period=save_every,
        device=str(ctx.gpu_id),
        lr0=float(victim_cfg.get("lr0", 0.01)),
        lrf=float(victim_cfg.get("lrf", 0.01)),
        momentum=float(victim_cfg.get("momentum", 0.937)),
        weight_decay=float(victim_cfg.get("weight_decay", 0.0005)),
        seed=int(ctx.seed),
    )

    def on_fit_epoch_end(trainer):
        epoch_num = int(getattr(trainer, "epoch", -1)) + 1
        if epoch_num <= 0:
            return

        if epoch_num % save_every == 0:
            latest_epoch = _snapshot_train_state(ctx, run_dir, latest_ckpt_ultra, best_ckpt_ultra)
            print(f"[train_victim] snapshot saved at epoch={latest_epoch}")

        if (
            bool(cfg["platform"].get("zip_after_stage", True))
            and pack_every > 0
            and epoch_num % pack_every == 0
        ):
            bundle_path = pack_run_artifacts(ctx.paths)
            print(f"[train_victim] packed bundle at epoch={epoch_num}: {bundle_path}")

    if resume_enabled and os.path.isfile(latest_ckpt_ultra):
        model = YOLO(latest_ckpt_ultra)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(resume=True)
    else:
        model = YOLO(victim_cfg["init"])
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(**train_args)

    latest_epoch = _snapshot_train_state(ctx, run_dir, latest_ckpt_ultra, best_ckpt_ultra)
    bundle_path = (
        pack_run_artifacts(ctx.paths)
        if bool(cfg["platform"].get("zip_after_stage", True))
        else ""
    )

    latest_ckpt = os.path.join(ctx.paths.checkpoints_dir, "latest.pt")
    best_ckpt = os.path.join(ctx.paths.checkpoints_dir, "best.pt")

    stage_extra = {
        "run_tag": ctx.run_tag,
        "deleted_cache_files": deleted_cache,
        "train_data_yaml": yaml_path,
        "latest_checkpoint": latest_ckpt if os.path.isfile(latest_ckpt) else "",
        "best_checkpoint": best_ckpt if os.path.isfile(best_ckpt) else "",
        "train_run_dir": run_dir,
        "latest_epoch": latest_epoch,
        "bundle_path": bundle_path,
        "save_every_n_epochs": save_every,
        "pack_every_n_epochs": pack_every,
        "resume_enabled": resume_enabled,
        "sparse_train_audit": sparse_audit or {},
        "sparse_dataloader_probe": sparse_dataloader_probe or {},
    }
    mark_stage_completed(ctx.paths.artifact_status_json, status, "train_victim", stage_extra)
    atomic_write_json(os.path.join(ctx.paths.logs_dir, "train_stage_summary.json"), stage_extra)

    print(
        "[train_victim] done: "
        f"method={ctx.method}, steps={ctx.steps}, seed={ctx.seed}, "
        f"latest_epoch={latest_epoch}, deleted_cache={deleted_cache}, bundle={bundle_path}"
    )

