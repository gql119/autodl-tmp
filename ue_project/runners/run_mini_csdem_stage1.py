from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_csdem.dataset import file_sha256, load_config, resolve_target_class, write_json
from mini_csdem.gt_conditioned_partition import build_target_partition
from mini_csdem.non_target_preservation import compute_non_target_preservation
from mini_csdem.object_aligned_perturbation import ObjectAlignedPerturbation
from mini_csdem.selective_detection_loss import target_only_detection_loss
from runners.run_mini_csdem import _val, feature_hook_count, state_sha256
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter


class MiniDetectionDataset(Dataset):
    def __init__(self, root: Path, image_ids: Sequence[str], split: str, imgsz: int) -> None:
        self.root = root
        self.image_ids = list(image_ids)
        self.split = split
        self.imgsz = int(imgsz)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self.root / "images" / self.split / f"{image_id}.jpg").convert("RGB")
        image = image.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        classes, boxes = [], []
        for line in (self.root / "labels" / self.split / f"{image_id}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            classes.append(float(parts[0]))
            boxes.append([float(value) for value in parts[1:5]])
        return image_id, image_tensor, torch.tensor(classes), torch.tensor(boxes)


def collate(rows):
    image_ids, images, class_rows, box_rows = zip(*rows)
    classes, boxes, batch_indices = [], [], []
    for index, (row_classes, row_boxes) in enumerate(zip(class_rows, box_rows)):
        classes.append(row_classes)
        boxes.append(row_boxes)
        batch_indices.append(torch.full((len(row_classes),), float(index)))
    image_tensor = torch.stack(images)
    return list(image_ids), image_tensor, {
        "img": image_tensor,
        "cls": torch.cat(classes).view(-1, 1),
        "bboxes": torch.cat(boxes).view(-1, 4),
        "batch_idx": torch.cat(batch_indices),
        "batch_size": len(images),
    }


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def set_batch_norm_eval(model: torch.nn.Module) -> List[torch.nn.Module]:
    changed = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            module.eval()
            changed.append(module)
    return changed


def generate_delta(cfg: Dict, manifest: Dict, names: Sequence[str], target_id: int, output_root: Path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_wrapper = YOLO(str((ROOT / cfg["model"]["init"]).resolve()))
    model = model_wrapper.model.to(device).train()
    initial_hash = state_sha256(model)
    adapter = YOLOv8TALAdapter(model, len(names), target_id)
    perturbation = ObjectAlignedPerturbation(
        int(cfg["poison"]["object_size"]), float(cfg["poison"]["epsilon"]), seed=seed
    ).to(device)
    initial_delta = perturbation.delta_object.detach().clone()
    surrogate_optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["poison"]["surrogate_lr"]))
    noise_optimizer = torch.optim.Adam([perturbation.delta_object], lr=float(cfg["poison"]["noise_lr"]))
    clean_root = (ROOT / cfg["data"]["clean_dataset_root"]).resolve()
    loader = DataLoader(
        MiniDetectionDataset(clean_root, manifest["train_ids"], "train", int(cfg["victim"]["imgsz"])),
        batch_size=int(cfg["poison"]["batch"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    logs = []
    preserve_enabled = bool(cfg.get("preservation", {}).get("enabled", False)) and bool(
        cfg.get("features", {}).get("enable_non_target_preservation", False)
    )
    preserve_cfg = cfg.get("preservation", {})
    preserve_weights = {
        "logits": float(preserve_cfg.get("lambda_logits", 0.0)) if preserve_cfg.get("enable_logits", False) else 0.0,
        "box": float(preserve_cfg.get("lambda_box_keep", 0.0)) if preserve_cfg.get("enable_box", False) else 0.0,
        "dfl": float(preserve_cfg.get("lambda_dfl_keep", 0.0)) if preserve_cfg.get("enable_dfl", False) else 0.0,
        "assignment": float(preserve_cfg.get("lambda_assign", 0.0)) if preserve_cfg.get("enable_assignment", False) else 0.0,
    }
    diagnostic_interval = int(preserve_cfg.get("gradient_diagnostics_interval", 20))
    preserve_ema: Dict[str, float] = {}
    ema_decay = float(preserve_cfg.get("ema_decay", 0.9))
    normalization_floor = float(preserve_cfg.get("normalization_floor", 1.0e-8))
    for epoch in range(int(cfg["poison"]["generation_epochs"])):
        epoch_values: Dict[str, List[float]] = {
            key: []
            for key in [
                "surrogate_loss",
                "target_loss",
                "target_cls_loss",
                "target_box_loss",
                "target_dfl_loss",
                "target_positive_count",
                "non_target_positive_count",
                "target_instance_count",
                "fallback_ratio",
                "gradient_norm_target",
                "raw_area",
                "effective_area",
                "non_target_cls_drift",
                "non_target_raw_logits_drift",
                "non_target_box_drift",
                "non_target_dfl_drift",
                "non_target_assignment_drift",
                "preserve_total_loss",
                "clean_non_target_positive_count",
                "poison_non_target_positive_count",
                "aligned_non_target_count",
                "gt_index_mismatch_count",
                "alignment_coverage",
                "gradient_norm_preserve",
                "gradient_norm_logits",
                "gradient_norm_box_keep",
                "gradient_norm_dfl_keep",
                "gradient_norm_assignment_keep",
                "target_preserve_gradient_cosine",
                "clean_soft_mean",
                "clean_soft_std",
                "poison_soft_mean",
                "poison_soft_std",
            ]
        }
        classwise_epoch: Dict[int, List[float]] = {}
        class_alignment_epoch: Dict[int, int] = {}
        for batch_number, (_image_ids, clean_images, batch) in enumerate(loader):
            clean_images = clean_images.to(device)
            batch = to_device(batch, device)

            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.train()
            surrogate_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                poison_for_model = perturbation(
                    clean_images,
                    batch,
                    target_id,
                    bool(cfg["poison"]["exclude_non_target_overlap"]),
                    int(cfg["poison"]["non_target_dilation"]),
                ).images
            predictions = adapter.forward(poison_for_model)
            surrogate = adapter.compute_detection_loss(predictions, batch, return_components=True)
            surrogate["total_loss"].backward()
            surrogate_optimizer.step()

            surrogate_optimizer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            changed_bn = set_batch_norm_eval(model)
            noise_optimizer.zero_grad(set_to_none=True)
            clean_state = None
            if preserve_enabled:
                with torch.no_grad():
                    clean_predictions = adapter.forward(clean_images)
                    clean_state = adapter._build_assignment_state(clean_predictions, batch)
            application = perturbation(
                clean_images,
                batch,
                target_id,
                bool(cfg["poison"]["exclude_non_target_overlap"]),
                int(cfg["poison"]["non_target_dilation"]),
            )
            predictions = adapter.forward(application.images)
            state = adapter._build_assignment_state(predictions, batch)
            partition = build_target_partition(
                adapter,
                state,
                batch,
                target_id,
                int(cfg["poison"]["minimum_target_positives"]),
                int(cfg["poison"]["fallback_min_per_level"]),
            )
            losses = target_only_detection_loss(adapter, state, partition)
            preservation = (
                compute_non_target_preservation(clean_state, state, target_id, preserve_weights)
                if preserve_enabled
                else None
            )
            normalized_components = {}
            if preservation is not None:
                for key, component in {
                    "logits": preservation.logits_loss,
                    "box": preservation.box_loss,
                    "dfl": preservation.dfl_loss,
                    "assignment": preservation.assignment_loss,
                }.items():
                    current = float(component.detach())
                    previous = preserve_ema.get(key, current)
                    preserve_ema[key] = ema_decay * previous + (1.0 - ema_decay) * current
                    denominator = max(preserve_ema[key], normalization_floor)
                    normalized_components[key] = component / denominator
                preserve_objective = sum(
                    preserve_weights[key] * normalized_components[key]
                    for key in ["logits", "box", "dfl", "assignment"]
                )
            else:
                preserve_objective = losses["total_loss"] * 0.0
            total_delta_loss = losses["total_loss"] + (
                preserve_objective
            )
            gradient_norm = float("nan") if preserve_enabled else 0.0
            gradient_norm_preserve = 0.0
            component_gradient_norms = {"logits": 0.0, "box": 0.0, "dfl": 0.0, "assignment": 0.0}
            cosine = float("nan")
            run_gradient_diagnostics = preserve_enabled and batch_number % max(diagnostic_interval, 1) == 0
            if run_gradient_diagnostics and total_delta_loss.requires_grad:
                target_gradient = torch.autograd.grad(
                    losses["total_loss"], perturbation.delta_object, retain_graph=True, allow_unused=True
                )[0]
                preserve_gradient = torch.autograd.grad(
                    preserve_objective, perturbation.delta_object, retain_graph=True, allow_unused=True
                )[0]
                if target_gradient is not None:
                    gradient_norm = float(target_gradient.norm())
                if preserve_gradient is not None:
                    gradient_norm_preserve = float(preserve_gradient.norm())
                if target_gradient is not None and preserve_gradient is not None:
                    denominator = target_gradient.norm() * preserve_gradient.norm()
                    if float(denominator) > 0:
                        cosine = float((target_gradient * preserve_gradient).sum() / denominator)
                for key, component in {
                    "logits": preservation.logits_loss,
                    "box": preservation.box_loss,
                    "dfl": preservation.dfl_loss,
                    "assignment": preservation.assignment_loss,
                }.items():
                    component_gradient = torch.autograd.grad(
                        normalized_components[key], perturbation.delta_object, retain_graph=True, allow_unused=True
                    )[0]
                    if component_gradient is not None:
                        component_gradient_norms[key] = float(component_gradient.norm())
            if application.target_instances > 0 and total_delta_loss.requires_grad:
                total_delta_loss.backward()
                if not preserve_enabled and perturbation.delta_object.grad is not None:
                    gradient_norm = float(perturbation.delta_object.grad.norm())
                noise_optimizer.step()
                perturbation.project_()
            else:
                gradient_norm = float("nan") if preserve_enabled else 0.0
            for module in changed_bn:
                module.train()

            epoch_values["surrogate_loss"].append(float(surrogate["total_loss"].detach()))
            epoch_values["target_loss"].append(float(losses["total_loss"].detach()))
            epoch_values["target_cls_loss"].append(float(losses["cls_loss"].detach()))
            epoch_values["target_box_loss"].append(float(losses["box_loss"].detach()))
            epoch_values["target_dfl_loss"].append(float(losses["dfl_loss"].detach()))
            epoch_values["target_positive_count"].append(float(partition.unit_mask.sum()))
            epoch_values["non_target_positive_count"].append(
                float((state["fg_mask"].bool() & (state["target_labels"].long() != target_id)).sum())
            )
            epoch_values["target_instance_count"].append(float(application.target_instances))
            epoch_values["fallback_ratio"].append(partition.fallback_ratio)
            epoch_values["gradient_norm_target"].append(gradient_norm)
            epoch_values["raw_area"].append(float(application.raw_target_pixels))
            epoch_values["effective_area"].append(float(application.effective_target_pixels))
            if preservation is None:
                for key in [
                    "non_target_cls_drift",
                    "non_target_raw_logits_drift",
                    "non_target_box_drift",
                    "non_target_dfl_drift",
                    "non_target_assignment_drift",
                    "preserve_total_loss",
                    "clean_non_target_positive_count",
                    "poison_non_target_positive_count",
                    "aligned_non_target_count",
                    "gt_index_mismatch_count",
                    "alignment_coverage",
                    "gradient_norm_preserve",
                    "gradient_norm_logits",
                    "gradient_norm_box_keep",
                    "gradient_norm_dfl_keep",
                    "gradient_norm_assignment_keep",
                    "clean_soft_mean",
                    "clean_soft_std",
                    "poison_soft_mean",
                    "poison_soft_std",
                ]:
                    epoch_values[key].append(0.0)
                epoch_values["target_preserve_gradient_cosine"].append(float("nan"))
            else:
                alignment = preservation.alignment
                epoch_values["non_target_cls_drift"].append(float(preservation.normalized_logits_drift.detach()))
                epoch_values["non_target_raw_logits_drift"].append(float(preservation.raw_logits_drift.detach()))
                epoch_values["non_target_box_drift"].append(float(preservation.box_loss.detach()))
                epoch_values["non_target_dfl_drift"].append(float(preservation.dfl_loss.detach()))
                epoch_values["non_target_assignment_drift"].append(float(preservation.assignment_loss.detach()))
                epoch_values["preserve_total_loss"].append(float(preserve_objective.detach()))
                epoch_values["clean_non_target_positive_count"].append(float(alignment.clean_non_target_count))
                epoch_values["poison_non_target_positive_count"].append(float(alignment.poison_non_target_count))
                epoch_values["aligned_non_target_count"].append(float(alignment.matched_count))
                epoch_values["gt_index_mismatch_count"].append(float(alignment.gt_index_mismatch_count))
                epoch_values["alignment_coverage"].append(alignment.coverage)
                epoch_values["gradient_norm_preserve"].append(gradient_norm_preserve)
                epoch_values["gradient_norm_logits"].append(component_gradient_norms["logits"])
                epoch_values["gradient_norm_box_keep"].append(component_gradient_norms["box"])
                epoch_values["gradient_norm_dfl_keep"].append(component_gradient_norms["dfl"])
                epoch_values["gradient_norm_assignment_keep"].append(component_gradient_norms["assignment"])
                epoch_values["target_preserve_gradient_cosine"].append(cosine)
                epoch_values["clean_soft_mean"].append(float(preservation.clean_soft_mean.detach()))
                epoch_values["clean_soft_std"].append(float(preservation.clean_soft_std.detach()))
                epoch_values["poison_soft_mean"].append(float(preservation.poison_soft_mean.detach()))
                epoch_values["poison_soft_std"].append(float(preservation.poison_soft_std.detach()))
                for class_id, drift in preservation.classwise_drift.items():
                    classwise_epoch.setdefault(class_id, []).append(float(drift.detach()))
                for class_id, count in alignment.class_alignment_counts.items():
                    class_alignment_epoch[class_id] = class_alignment_epoch.get(class_id, 0) + count

        row = {"epoch": epoch + 1, "seed": seed, "target_class_id": target_id}
        row.update({key: float(np.nanmean(values)) if not all(np.isnan(values)) else None for key, values in epoch_values.items()})
        row.update(perturbation.statistics())
        row["delta_change_from_initial_linf"] = float(
            (perturbation.delta_object.detach() - initial_delta).abs().max()
        )
        row["clean_teacher_requires_grad"] = False
        row.update(
            {
                "classwise_drift": {
                    str(class_id): float(np.mean(values)) for class_id, values in sorted(classwise_epoch.items())
                },
                "class_alignment_counts": {str(key): value for key, value in sorted(class_alignment_epoch.items())},
                "classwise_dual_weights": {},
            }
        )
        row["perturbed_area_ratio"] = row["effective_area"] / max(
            int(cfg["poison"]["batch"]) * int(cfg["victim"]["imgsz"]) ** 2, 1
        )
        logs.append(row)
        print(json.dumps(row))

    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta_object": perturbation.delta_object.detach().cpu(),
            "epsilon": perturbation.epsilon,
            "target_class_id": target_id,
            "surrogate_initial_hash": initial_hash,
            "surrogate_final_hash": state_sha256(model),
        },
        output_root / "delta_object.pt",
    )
    with (output_root / "generation_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    write_json(output_root / "generation_log.json", logs)
    return perturbation, logs, {"initial_hash": initial_hash, "final_hash": state_sha256(model)}


def materialize_poison(
    perturbation: ObjectAlignedPerturbation,
    cfg: Dict,
    manifest: Dict,
    target_id: int,
    names: Sequence[str],
    output_root: Path,
) -> Dict:
    clean_root = (ROOT / cfg["data"]["clean_dataset_root"]).resolve()
    poison_root = output_root / "poisoned_dataset"
    device = perturbation.delta_object.device
    rows, train_paths = [], []
    for image_id in manifest["train_ids"]:
        source = clean_root / "images" / "train" / f"{image_id}.jpg"
        label_source = clean_root / "labels" / "train" / f"{image_id}.txt"
        annotations = [line.split() for line in label_source.read_text(encoding="utf-8").splitlines()]
        classes = torch.tensor([float(row[0]) for row in annotations], device=device).view(-1, 1)
        boxes = torch.tensor([[float(value) for value in row[1:5]] for row in annotations], device=device)
        has_target = bool((classes.reshape(-1).long() == target_id).any())
        label_path = poison_root / "labels" / "train" / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_source, label_path)
        if not has_target:
            image_path = poison_root / "images" / "train" / source.name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_path)
            rows.append({"image_id": image_id, "has_target": False, "linf": 0.0, "mean_abs": 0.0, "area": 0.0})
        else:
            clean_np = np.asarray(Image.open(source).convert("RGB")).copy()
            clean = torch.from_numpy(clean_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            batch = {
                "cls": classes,
                "bboxes": boxes,
                "batch_idx": torch.zeros(len(classes), device=device),
                "batch_size": 1,
            }
            with torch.no_grad():
                application = perturbation(
                    clean,
                    batch,
                    target_id,
                    bool(cfg["poison"]["exclude_non_target_overlap"]),
                    int(cfg["poison"]["non_target_dilation"]),
                )
            poison = application.images[0].mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            image_path = poison_root / "images" / "train" / f"{image_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(poison).save(image_path)
            difference = np.abs(poison.astype(np.float32) - clean_np.astype(np.float32)) / 255.0
            rows.append(
                {
                    "image_id": image_id,
                    "has_target": True,
                    "linf": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "area": float((difference.max(axis=2) > 0).mean()),
                    "raw_area": application.raw_target_pixels,
                    "effective_area": application.effective_target_pixels,
                }
            )
        train_paths.append(str(image_path.resolve()))

    train_list = poison_root / "train.txt"
    train_list.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    data_yaml = poison_root / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(poison_root.resolve()),
                "train": str(train_list.resolve()),
                "val": str((clean_root / "val.txt").resolve()),
                "names": {index: name for index, name in enumerate(names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(poison_root / "manifest.json", rows)
    target_rows = [row for row in rows if row["has_target"]]
    stats = {
        "image_count": len(rows),
        "target_image_count": len(target_rows),
        "person_free_byte_identical": all(
            file_sha256(poison_root / "images" / "train" / f"{row['image_id']}.jpg")
            == file_sha256(clean_root / "images" / "train" / f"{row['image_id']}.jpg")
            for row in rows
            if not row["has_target"]
        ),
        "perturbation_linf": max(row["linf"] for row in rows),
        "perturbation_mean_abs": float(np.mean([row["mean_abs"] for row in rows])),
        "perturbed_area_ratio": float(np.mean([row["area"] for row in rows])),
        "dataset_yaml": str(data_yaml.resolve()),
        "train_list": str(train_list.resolve()),
    }
    write_json(poison_root / "checks.json", stats)
    return stats


def train_and_evaluate(cfg: Dict, manifest: Dict, names: Sequence[str], target_id: int, poison: Dict, output_root: Path):
    seed = int(cfg["victim"]["seed"])
    stage = str(cfg["experiment"].get("stage", "stage1"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_path = (ROOT / cfg["model"]["init"]).resolve()
    victim = YOLO(str(init_path))
    victim_initial_hash = state_sha256(victim.model)
    started = time.time()
    train_result = victim.train(
        data=poison["dataset_yaml"],
        epochs=int(cfg["victim"]["epochs"]),
        imgsz=int(cfg["victim"]["imgsz"]),
        batch=int(cfg["victim"]["batch"]),
        workers=int(cfg["victim"]["workers"]),
        cache=False,
        amp=bool(cfg["victim"]["amp"]),
        pretrained=False,
        resume=False,
        seed=seed,
        deterministic=True,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(output_root / "victim_runs"),
        name=f"{stage}_seed{seed}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    run_dir = Path(train_result.save_dir)
    checkpoint = run_dir / "weights" / "best.pt"
    eval_cfg = {"training": cfg["victim"], "data": {"names": list(names)}}
    clean_root = (ROOT / cfg["data"]["clean_dataset_root"]).resolve()
    data_yaml = Path(poison["dataset_yaml"])
    full = _val(victim, data_yaml, clean_root / "val.txt", eval_cfg, f"{stage}_seed{seed}_clean_val")
    free = _val(victim, data_yaml, clean_root / "val_person_free.txt", eval_cfg, f"{stage}_seed{seed}_person_free")
    cooccur = _val(victim, data_yaml, clean_root / "val_person_cooccur.txt", eval_cfg, f"{stage}_seed{seed}_cooccur")
    poison_train = _val(victim, data_yaml, Path(poison["train_list"]), eval_cfg, f"{stage}_seed{seed}_poison_train")
    non_target_ids = [index for index in range(len(names)) if index != target_id]
    return {
        "victim_initial_hash": victim_initial_hash,
        "victim_checkpoint": str(checkpoint.resolve()),
        "victim_checkpoint_sha256": file_sha256(checkpoint),
        "victim_training_seconds": time.time() - started,
        "legacy_hook_count_before_victim_training": feature_hook_count(YOLO(str(init_path)).model),
        "clean_val_target_mAP50": float(full["ap50"][target_id]),
        "clean_val_non_target_mAP50": float(np.nanmean(full["ap50"][non_target_ids])),
        "clean_val_all_mAP50": float(full["map50"]),
        "person_free_non_target_mAP50": float(np.nanmean(free["ap50"][non_target_ids])),
        "person_cooccur_non_target_mAP50": float(np.nanmean(cooccur["ap50"][non_target_ids])),
        "poisoned_train_target_mAP50": float(poison_train["ap50"][target_id]),
        "per_class_ap50": {names[index]: float(value) if math.isfinite(value) else None for index, value in enumerate(full["ap50"])},
    }


def run_experiment(cfg: Dict, result_name: str | None = None) -> Dict:
    stage = str(cfg["experiment"].get("stage", "stage1"))
    manifest = json.loads((ROOT / cfg["data"]["split_manifest"]).read_text(encoding="utf-8"))
    names = list(manifest["names"])
    target_id = resolve_target_class(names, cfg["experiment"]["target_class_name"])
    output_root = (ROOT / cfg["paths"]["output_root"]).resolve()
    perturbation, logs, surrogate_hashes = generate_delta(cfg, manifest, names, target_id, output_root)
    poison = materialize_poison(perturbation, cfg, manifest, target_id, names, output_root)
    victim = train_and_evaluate(cfg, manifest, names, target_id, poison, output_root)
    baseline = json.loads((ROOT / "results/mini_csdem/clean_baseline_metrics.json").read_text(encoding="utf-8"))
    target_drop = baseline["mAP50_target"] - victim["clean_val_target_mAP50"]
    relative_drop = target_drop / max(baseline["mAP50_target"], 1.0e-12)
    mechanics = (
        logs[-1]["perturbation_linf"] <= float(cfg["poison"]["epsilon"]) + 1.0e-6
        and logs[-1]["perturbation_mean_abs"] > 0
        and all(math.isfinite(row["target_loss"]) for row in logs)
        and poison["person_free_byte_identical"]
        and victim["victim_initial_hash"] == surrogate_hashes["initial_hash"]
        and victim["victim_initial_hash"] != surrogate_hashes["final_hash"]
    )
    if stage == "stage1":
        effective = (
            relative_drop >= 0.25 or target_drop >= 0.10
        ) and victim["poisoned_train_target_mAP50"] - victim["clean_val_target_mAP50"] >= 0.10
    else:
        stage1 = json.loads(
            (ROOT / f"results/mini_csdem/stage1_seed{int(cfg['experiment']['seed'])}_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        stage1_drop = baseline["mAP50_non_target"] - stage1["clean_val_non_target_mAP50"]
        current_drop = baseline["mAP50_non_target"] - victim["clean_val_non_target_mAP50"]
        non_target_improved = (
            victim["clean_val_non_target_mAP50"] - stage1["clean_val_non_target_mAP50"] >= 0.015
            or current_drop <= 0.8 * max(stage1_drop, 0.0)
        )
        target_retained = victim["clean_val_target_mAP50"] - stage1["clean_val_target_mAP50"] <= 0.05
        cooccur_retained = (
            victim["person_cooccur_non_target_mAP50"] >= stage1["person_cooccur_non_target_mAP50"]
        )
        alignment_ok = logs[-1].get("alignment_coverage", 0.0) >= 0.5
        delta_ok = poison["perturbation_mean_abs"] >= 0.2 * stage1["poison"]["perturbation_mean_abs"]
        effective = non_target_improved and target_retained and cooccur_retained and alignment_ok and delta_ok
    status = "PASS" if mechanics and effective else "PARTIAL" if mechanics and target_drop > 0 else "FAIL"
    result = {
        "stage": stage,
        "seed": int(cfg["experiment"]["seed"]),
        "target_class_id": target_id,
        "surrogate": surrogate_hashes,
        "poison": poison,
        "generation_first_epoch": logs[0],
        "generation_last_epoch": logs[-1],
        **victim,
        "target_absolute_drop": target_drop,
        "target_relative_drop": relative_drop,
        "poison_train_clean_val_gap": victim["poisoned_train_target_mAP50"] - victim["clean_val_target_mAP50"],
        "mechanical_correctness": mechanics,
        "effectiveness_criteria_met": effective,
        "status": status,
    }
    results_root = ROOT / cfg["paths"]["results_root"]
    seed = int(cfg["experiment"]["seed"])
    result_name = result_name or f"{stage}_seed{seed}"
    write_json(results_root / f"{result_name}_metrics.json", result)
    with (results_root / f"{result_name}_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "is_target", "clean_ap50", f"{stage}_ap50", "drop"])
        for index, name in enumerate(names):
            clean_ap = baseline["per_class_ap50"][name]
            stage_ap = result["per_class_ap50"][name]
            writer.writerow([index, name, index == target_id, clean_ap, stage_ap, clean_ap - stage_ap])
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mini_csdem/stage1.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage = str(cfg["experiment"].get("stage", "stage1"))
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
        cfg["victim"]["seed"] = int(args.seed)
        configured_root = Path(cfg["paths"]["output_root"])
        cfg["paths"]["output_root"] = str(configured_root.parent / f"{stage}_seed{args.seed}")
    run_experiment(cfg)


if __name__ == "__main__":
    main()
