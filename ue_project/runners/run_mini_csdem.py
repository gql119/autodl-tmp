from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_csdem.dataset import (
    build_split,
    file_sha256,
    load_config,
    materialize_clean_dataset,
    parse_voc,
    resolve_target_class,
    write_json,
)
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def feature_hook_count(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) + len(module._backward_hooks) for module in model.modules())


def prepare(config_path: Path) -> Dict:
    cfg = load_config(config_path)
    names = list(cfg["data"]["names"])
    target_id = resolve_target_class(names, cfg["experiment"]["target_class_name"])
    voc_root = (ROOT / cfg["data"]["voc_root"]).resolve()
    catalog = parse_voc(voc_root, names, cfg["data"].get("split", "trainval"))
    train, val, manifest = build_split(
        catalog,
        int(cfg["experiment"]["train_size"]),
        int(cfg["experiment"]["val_size"]),
        target_id,
        int(cfg["experiment"]["seed"]),
        len(names),
    )
    manifest.update({"names": names, "target_class_name": cfg["experiment"]["target_class_name"]})
    split_path = ROOT / cfg["paths"]["split_manifest"]
    write_json(split_path, manifest)
    output_root = ROOT / cfg["paths"]["output_root"]
    dataset_info = materialize_clean_dataset(train, val, output_root / "clean_dataset", names, target_id)
    prepared = {
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "split_manifest": str(split_path.resolve()),
        "split_file_sha256": file_sha256(split_path),
        "target_class_id": target_id,
        "target_class_name": names[target_id],
        "catalog_size": len(catalog),
        "dataset": dataset_info,
    }
    write_json(output_root / "prepare.json", prepared)
    return prepared


def _metric_arrays(metrics, num_classes: int) -> np.ndarray:
    ap50 = np.full(num_classes, np.nan, dtype=np.float64)
    for index, class_id in enumerate(metrics.box.ap_class_index):
        ap50[int(class_id)] = float(metrics.box.ap50[index])
    return ap50


def _val(model: YOLO, data_yaml: Path, val_list: Path, cfg: Dict, run_name: str) -> Dict:
    import yaml

    base = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base["val"] = str(val_list.resolve())
    temp_yaml = data_yaml.parent / f"{run_name}.yaml"
    temp_yaml.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    result = model.val(
        data=str(temp_yaml),
        imgsz=int(cfg["training"]["imgsz"]),
        batch=int(cfg["training"]["batch"]),
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
    )
    return {"map50": float(result.box.map50), "ap50": _metric_arrays(result, len(cfg["data"]["names"]))}


def _diagnostic_batch(
    split_manifest: Dict, dataset_root: Path, device: torch.device, imgsz: int, limit: int = 32
):
    ids = split_manifest["val_ids"][:limit]
    images, cls, bboxes, batch_idx = [], [], [], []
    for index, image_id in enumerate(ids):
        image = Image.open(dataset_root / "images" / "val" / f"{image_id}.jpg").convert("RGB").resize((imgsz, imgsz))
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
        label_path = dataset_root / "labels" / "val" / f"{image_id}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            cls.append(float(parts[0]))
            bboxes.append([float(value) for value in parts[1:5]])
            batch_idx.append(float(index))
    image_tensor = torch.stack(images).to(device)
    batch = {
        "img": image_tensor,
        "cls": torch.tensor(cls, device=device).view(-1, 1),
        "bboxes": torch.tensor(bboxes, device=device),
        "batch_idx": torch.tensor(batch_idx, device=device),
        "batch_size": len(ids),
    }
    return image_tensor, batch


def class_conditioned_losses(checkpoint: Path, cfg: Dict, split_manifest: Dict, dataset_root: Path) -> Dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = YOLO(str(checkpoint))
    model = wrapper.model.to(device).train()
    adapter = YOLOv8TALAdapter(model, len(cfg["data"]["names"]), split_manifest["target_class_id"])
    images, batch = _diagnostic_batch(split_manifest, dataset_root, device, int(cfg["training"]["imgsz"]))
    predictions = adapter.forward(images)
    target = adapter.compute_detection_loss(
        predictions, batch, class_filter=[split_manifest["target_class_id"]], return_components=True
    )
    non_target_ids = [index for index in range(len(cfg["data"]["names"])) if index != split_manifest["target_class_id"]]
    non_target = adapter.compute_detection_loss(predictions, batch, class_filter=non_target_ids, return_components=True)
    assignment = adapter.get_task_aligned_assignments(predictions, batch)
    target_positive = assignment.fg_mask & (assignment.target_labels == split_manifest["target_class_id"])
    return {
        "target_cls_loss": float(target["cls_loss"].detach()),
        "target_box_loss": float(target["box_loss"].detach()),
        "target_dfl_loss": float(target["dfl_loss"].detach()),
        "non_target_cls_loss": float(non_target["cls_loss"].detach()),
        "non_target_box_loss": float(non_target["box_loss"].detach()),
        "non_target_dfl_loss": float(non_target["dfl_loss"].detach()),
        "target_positive_count": int(target_positive.sum()),
        "non_target_positive_count": int((assignment.fg_mask & ~target_positive).sum()),
    }


def run(mode: str, config_path: Path) -> Dict:
    cfg = load_config(config_path)
    prepared = prepare(config_path)
    output_root = ROOT / cfg["paths"]["output_root"]
    dataset_root = output_root / "clean_dataset"
    data_yaml = Path(prepared["dataset"]["dataset_yaml"])
    split_manifest = json.loads((ROOT / cfg["paths"]["split_manifest"]).read_text(encoding="utf-8"))
    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    init_path = (ROOT / cfg["model"]["init"]).resolve()
    model = YOLO(str(init_path))
    initial_hash = state_sha256(model.model)
    hooks_before = feature_hook_count(model.model)
    sample_path = dataset_root / "images" / "train" / f"{split_manifest['train_ids'][0]}.jpg"
    sample_hash_before = file_sha256(sample_path)
    epochs = 1 if mode == "smoke" else int(cfg["training"]["epochs"])
    name = "clean_smoke" if mode == "smoke" else "clean_baseline_seed0"
    started = time.time()
    train_result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=int(cfg["training"]["imgsz"]),
        batch=int(cfg["training"]["batch"]),
        workers=int(cfg["training"]["workers"]),
        cache=False,
        amp=bool(cfg["training"]["amp"]),
        pretrained=False,
        resume=False,
        optimizer=str(cfg["training"]["optimizer"]),
        seed=seed,
        deterministic=True,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(output_root / "train_runs"),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
        save=mode != "smoke",
    )
    run_dir = Path(train_result.save_dir)
    results_csv = run_dir / "results.csv"
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    numeric = []
    for row in rows:
        numeric.extend(float(value) for value in row.values() if value not in ("", None))
    finite_losses = all(math.isfinite(value) for value in numeric)
    full_val = _val(model, data_yaml, Path(prepared["dataset"]["val"]), cfg, f"{name}_full")
    smoke = {
        "mode": mode,
        "epochs": epochs,
        "elapsed_seconds": time.time() - started,
        "model_init": str(init_path),
        "pretrained": False,
        "resume": False,
        "initial_state_sha256": initial_hash,
        "legacy_hook_count_before_training": hooks_before,
        "clean_image_unchanged": sample_hash_before == file_sha256(sample_path),
        "finite_training_values": finite_losses,
        "validation_map50": full_val["map50"],
        "results_csv": str(results_csv.resolve()),
    }
    if mode == "smoke":
        write_json(output_root / "smoke_test.json", smoke)
        return smoke

    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "weights" / "last.pt"
    person_free = _val(model, data_yaml, Path(prepared["dataset"]["val_person_free"]), cfg, f"{name}_free")
    cooccur = _val(model, data_yaml, Path(prepared["dataset"]["val_person_cooccur"]), cfg, f"{name}_cooccur")
    target_id = int(prepared["target_class_id"])
    non_target_ids = [index for index in range(len(cfg["data"]["names"])) if index != target_id]
    per_class = full_val["ap50"]
    loss_diag = class_conditioned_losses(checkpoint, cfg, split_manifest, dataset_root)
    metrics = {
        **smoke,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "mAP50_target": float(per_class[target_id]),
        "mAP50_non_target": float(np.nanmean(per_class[non_target_ids])),
        "mAP50_all": float(full_val["map50"]),
        "worst_non_target_AP": float(np.nanmin(per_class[non_target_ids])),
        "AP_person_free_non_target": float(np.nanmean(person_free["ap50"][non_target_ids])),
        "AP_person_cooccur_non_target": float(np.nanmean(cooccur["ap50"][non_target_ids])),
        "per_class_ap50": {cfg["data"]["names"][index]: float(value) if math.isfinite(value) else None for index, value in enumerate(per_class)},
        **loss_diag,
    }
    results_root = ROOT / cfg["paths"]["results_root"]
    write_json(results_root / "clean_baseline_metrics.json", metrics)
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "clean_baseline_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "is_target", "ap50"])
        for index, value in enumerate(per_class):
            writer.writerow([index, cfg["data"]["names"][index], index == target_id, "" if not math.isfinite(value) else value])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mini_csdem/clean_baseline.yaml")
    parser.add_argument("--mode", choices=["prepare", "smoke", "baseline"], required=True)
    args = parser.parse_args()
    result = prepare(args.config) if args.mode == "prepare" else run(args.mode, args.config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
