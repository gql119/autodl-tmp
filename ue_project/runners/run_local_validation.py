from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ue_framework.core import ClassConditionedRouter
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter
from ue_framework.methods.learning_trajectory import LearningTrajectoryMethod
from ue_framework.methods.learning_trajectory.class_conditioned_loss import compute_class_conditioned_detection_loss
from ue_framework.methods.learning_trajectory.virtual_update import (
    make_virtual_parameters,
    parameter_leak_max_abs_diff,
    snapshot_parameters,
)


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}
PROTECTED_CLASS_ID = 14


def run_command(args: Sequence[str], cwd: Path = ROOT_DIR, timeout: int = 30) -> Dict[str, object]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "args": list(args),
            "returncode": int(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception:
        return {"args": list(args), "exception": traceback.format_exc()}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def box_iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1.0e-8, None)


def xywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x, y, w, h = boxes.T
    return np.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], axis=1)


def parse_voc_annotation(voc_root: Path, image_id: str) -> Dict:
    xml_path = voc_root / "Annotations" / f"{image_id}.xml"
    root = ET.parse(xml_path).getroot()
    size_node = root.find("size")
    width = float(size_node.findtext("width"))
    height = float(size_node.findtext("height"))
    labels: List[int] = []
    boxes_xywh: List[List[float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in VOC_CLASS_TO_ID:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
        x1 = max(0.0, min(width, xmin))
        y1 = max(0.0, min(height, ymin))
        x2 = max(0.0, min(width, xmax))
        y2 = max(0.0, min(height, ymax))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append(VOC_CLASS_TO_ID[name])
        boxes_xywh.append(
            [
                ((x1 + x2) * 0.5) / width,
                ((y1 + y2) * 0.5) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
        )
    labels_np = np.asarray(labels, dtype=np.int64)
    boxes_np = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    person_boxes = xywh_to_xyxy_np(boxes_np[labels_np == PROTECTED_CLASS_ID])
    authorized_boxes = xywh_to_xyxy_np(boxes_np[labels_np != PROTECTED_CLASS_ID])
    overlap = 0.0
    if person_boxes.size and authorized_boxes.size:
        overlap = float(box_iou_xyxy_np(person_boxes, authorized_boxes).max())
    return {
        "image_id": image_id,
        "image_path": str(voc_root / "JPEGImages" / f"{image_id}.jpg"),
        "labels": labels_np,
        "boxes_xywh": boxes_np,
        "classes": sorted(set(int(x) for x in labels_np.tolist())),
        "has_person": bool(np.any(labels_np == PROTECTED_CLASS_ID)),
        "has_authorized": bool(np.any(labels_np != PROTECTED_CLASS_ID)),
        "max_person_authorized_iou": overlap,
    }


def load_voc_catalog(voc_root: Path, split: str = "trainval") -> Tuple[List[Dict], Dict[str, object]]:
    ids_path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog = [parse_voc_annotation(voc_root, image_id) for image_id in image_ids]
    person = [x for x in catalog if x["has_person"]]
    authorized_only = [x for x in catalog if (not x["has_person"]) and x["has_authorized"]]
    cooccur = [x for x in catalog if x["has_person"] and x["has_authorized"]]
    stats = {
        "voc_root": str(voc_root),
        "split": split,
        "total_images": len(catalog),
        "person_images": len(person),
        "authorized_only_images": len(authorized_only),
        "person_authorized_cooccur_images": len(cooccur),
        "class_counts": {
            str(cls_id): int(sum(np.sum(sample["labels"] == cls_id) for sample in catalog))
            for cls_id in range(len(VOC_CLASSES))
        },
    }
    return catalog, stats


def choose_validation_samples(catalog: List[Dict]) -> Dict[str, Dict]:
    person_only = [x for x in catalog if x["has_person"] and not x["has_authorized"]]
    person_any = [x for x in catalog if x["has_person"]]
    authorized_only = [x for x in catalog if (not x["has_person"]) and x["has_authorized"]]
    cooccur = [x for x in catalog if x["has_person"] and x["has_authorized"]]
    if not person_any:
        raise RuntimeError("VOC catalog contains no person images.")
    if not authorized_only:
        raise RuntimeError("VOC catalog contains no authorized-only images.")
    mixed_low = sorted(cooccur, key=lambda x: x["max_person_authorized_iou"])[0] if cooccur else person_any[0]
    mixed_high = sorted(cooccur, key=lambda x: x["max_person_authorized_iou"], reverse=True)[0] if cooccur else person_any[0]
    return {
        "protected_only": person_only[0] if person_only else person_any[0],
        "authorized_only": authorized_only[0],
        "mixed_low_overlap": mixed_low,
        "mixed_high_overlap": mixed_high,
    }


def load_sample_tensor(sample: Dict, imgsz: int, device: torch.device) -> torch.Tensor:
    image = Image.open(sample["image_path"]).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    return tensor


def collate_samples(samples: Sequence[Dict], imgsz: int, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images = torch.stack([load_sample_tensor(sample, imgsz, device) for sample in samples], dim=0)
    cls_parts: List[torch.Tensor] = []
    box_parts: List[torch.Tensor] = []
    batch_idx_parts: List[torch.Tensor] = []
    for batch_idx, sample in enumerate(samples):
        labels = torch.as_tensor(sample["labels"], device=device, dtype=torch.float32).reshape(-1)
        boxes = torch.as_tensor(sample["boxes_xywh"], device=device, dtype=torch.float32).reshape(-1, 4)
        if labels.numel() == 0:
            continue
        cls_parts.append(labels)
        box_parts.append(boxes)
        batch_idx_parts.append(torch.full((labels.numel(),), batch_idx, device=device, dtype=torch.float32))
    if cls_parts:
        cls = torch.cat(cls_parts, dim=0)
        bboxes = torch.cat(box_parts, dim=0)
        batch_idx = torch.cat(batch_idx_parts, dim=0)
    else:
        cls = torch.empty(0, device=device, dtype=torch.float32)
        bboxes = torch.empty(0, 4, device=device, dtype=torch.float32)
        batch_idx = torch.empty(0, device=device, dtype=torch.float32)
    return images, {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "batch_size": len(samples),
    }


def make_artificial_batch(device: torch.device, imgsz: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    images = torch.rand((2, 3, imgsz, imgsz), generator=generator, device=device)
    cls = torch.tensor([14, 1, 14, 1], dtype=torch.float32, device=device)
    bboxes = torch.tensor(
        [
            [0.30, 0.30, 0.22, 0.26],
            [0.72, 0.70, 0.20, 0.20],
            [0.34, 0.34, 0.20, 0.22],
            [0.70, 0.32, 0.20, 0.24],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=device)
    return images, {"img": images, "cls": cls, "bboxes": bboxes, "batch_idx": batch_idx, "batch_size": 2}


def load_yolo_model(ckpt: Path, device: torch.device) -> Tuple[YOLO, torch.nn.Module]:
    wrapper = YOLO(str(ckpt))
    model = wrapper.model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return wrapper, model


def base_method_config(use_p1_regularizer: bool = False) -> Dict:
    return {
        "protected_class_id": PROTECTED_CLASS_ID,
        "authorized_class_ids": "auto",
        "num_classes": len(VOC_CLASSES),
        "trajectory": {
            "parameter_scope": "head",
            "normalize_per_parameter": True,
            "use_protected": True,
            "use_authorized": True,
            "lambda_protected": 1.0,
            "lambda_authorized": 1.0,
            "eps": 1.0e-8,
        },
        "class_routing": {
            "exclude_ambiguous": True,
            "include_background_negatives": False,
        },
        "virtual_update": {
            "parameter_scope": "head",
            "steps": 1,
            "lr": 0.001,
        },
        "meta": {
            "use_p1_regularizer": bool(use_p1_regularizer),
            "lambda_meta": 1.0,
            "lambda_p1": 0.2,
            "lambda_protected_query": 1.0,
            "enable_clean_counterfactual": True,
        },
    }


def scalar_logs(logs: Dict[str, object]) -> Dict[str, float]:
    return {k: float(v) for k, v in logs.items() if isinstance(v, (float, int))}


def cuda_memory(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }


def make_initial_delta(
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    seed: int,
    scale: float = 0.25,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    delta = torch.empty(tuple(shape), device=device, dtype=dtype)
    delta.uniform_(-float(eps) * float(scale), float(eps) * float(scale), generator=generator)
    delta.requires_grad_(True)
    return delta


def audit_new_method_imports() -> Dict[str, object]:
    forbidden = ["alce", "rlcp", "context prototype", "des-r", "fdacb", "weighted ring", "tausb_universal"]
    roots = [ROOT_DIR / "ue_framework" / "methods" / "learning_trajectory", ROOT_DIR / "ue_framework" / "core" / "yolov8_tal_adapter.py"]
    hits: Dict[str, List[str]] = {token: [] for token in forbidden}
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for token in forbidden:
                if token in text:
                    hits[token].append(str(path.relative_to(ROOT_DIR)))
    return {"forbidden_tokens": forbidden, "hits": hits, "has_hits": any(bool(v) for v in hits.values())}


def run_artificial_isolation(adapter: YOLOv8TALAdapter, device: torch.device, imgsz: int) -> Dict[str, object]:
    images, batch = make_artificial_batch(device, imgsz)
    predictions = adapter.forward(images)
    router = ClassConditionedRouter(PROTECTED_CLASS_ID, "auto", len(VOC_CLASSES), exclude_ambiguous=True)
    assignment = adapter.get_task_aligned_assignments(predictions, batch)
    routing = router.route(assignment)
    losses = compute_class_conditioned_detection_loss(adapter, predictions, batch, router, assignment, routing)
    protected_grad = torch.autograd.grad(
        losses["protected_total_loss"], predictions["scores"], retain_graph=True, allow_unused=False
    )[0]
    authorized_grad = torch.autograd.grad(
        losses["authorized_total_loss"], predictions["scores"], retain_graph=False, allow_unused=False
    )[0]

    def assigned_class_grad_max(grad: torch.Tensor, mask: torch.Tensor) -> float:
        if not mask.any():
            return 0.0
        pos = torch.nonzero(mask, as_tuple=False)
        b_idx = pos[:, 0]
        anchor_idx = pos[:, 1]
        labels = assignment.target_labels[b_idx, anchor_idx].long().clamp(0, len(VOC_CLASSES) - 1)
        return float(grad[b_idx, labels, anchor_idx].detach().abs().max().item())

    stats = {
        "protected_class_id": PROTECTED_CLASS_ID,
        "authorized_class_ids": [i for i in range(len(VOC_CLASSES)) if i != PROTECTED_CLASS_ID],
        "protected_positive_count": int(routing.stats["protected_positive_count"]),
        "authorized_positive_count": int(routing.stats["authorized_positive_count"]),
        "ambiguous_positive_count": int(routing.stats["ambiguous_positive_count"]),
        "protected_loss": float(losses["protected_total_loss"].detach().item()),
        "authorized_loss": float(losses["authorized_total_loss"].detach().item()),
        "protected_loss_grad_on_protected_assigned_class_max": assigned_class_grad_max(
            protected_grad, routing.protected_mask
        ),
        "protected_loss_grad_on_authorized_assigned_class_max": assigned_class_grad_max(
            protected_grad, routing.authorized_mask
        ),
        "authorized_loss_grad_on_protected_assigned_class_max": assigned_class_grad_max(
            authorized_grad, routing.protected_mask
        ),
        "authorized_loss_grad_on_authorized_assigned_class_max": assigned_class_grad_max(
            authorized_grad, routing.authorized_mask
        ),
    }
    return stats


def run_p1_once(
    method: LearningTrajectoryMethod,
    adapter: YOLOv8TALAdapter,
    images: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    eps: float,
) -> Dict[str, object]:
    snapshot = snapshot_parameters(adapter.model)
    delta = make_initial_delta(
        (1, 3, images.shape[-2], images.shape[-1]),
        images.device,
        images.dtype,
        eps,
        seed=1001,
    )
    if images.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(images.device)
    result = method.compute_p1_step(images, delta, batch)
    grad = torch.autograd.grad(result["loss"], delta, retain_graph=False, allow_unused=False)[0]
    logs = scalar_logs(result["logs"])
    logs.update(
        {
            "p1_loss": float(result["loss"].detach().item()),
            "delta_grad_norm": float(grad.detach().norm().item()),
            "surrogate_parameter_max_abs_diff": float(parameter_leak_max_abs_diff(adapter.model, snapshot)),
            "eps": float(eps),
        }
    )
    logs.update(cuda_memory(images.device))
    return logs


def run_p1_optimization(
    method: LearningTrajectoryMethod,
    images: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    eps: float,
    steps: int,
    lr: float,
) -> Dict[str, object]:
    delta = torch.zeros((1, 3, images.shape[-2], images.shape[-1]), device=images.device, dtype=images.dtype)
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    sequence: List[Dict[str, float]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = method.compute_p1_step(images, delta, batch)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-float(eps), float(eps))
        entry = scalar_logs(result["logs"])
        entry["step"] = float(step + 1)
        entry["p1_loss"] = float(loss.detach().item())
        entry["delta_linf"] = float(delta.detach().abs().max().item())
        sequence.append(entry)
        del result, loss
    return {
        "steps": steps,
        "lr": lr,
        "sequence": sequence,
        "final_delta_linf": float(delta.detach().abs().max().item()),
        "final_delta_mean_abs": float(delta.detach().abs().mean().item()),
    }


def run_p2_once(
    method: LearningTrajectoryMethod,
    adapter: YOLOv8TALAdapter,
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    support_batch: Dict[str, torch.Tensor],
    query_batch: Dict[str, torch.Tensor],
    eps: float,
    seed: int = 2001,
) -> Dict[str, object]:
    snapshot = snapshot_parameters(adapter.model)
    delta = make_initial_delta(
        (1, 3, support_images.shape[-2], support_images.shape[-1]),
        support_images.device,
        support_images.dtype,
        eps,
        seed=seed,
    )
    if support_images.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(support_images.device)
    result = method.compute_p2_step(support_images, query_images, support_batch, query_batch, delta)
    grad = torch.autograd.grad(result["loss"], delta, retain_graph=False, allow_unused=True)[0]
    logs = scalar_logs(result["logs"])
    logs["p2_loss"] = float(result["loss"].detach().item())
    logs["p2_autograd_gradient_norm_to_delta"] = 0.0 if grad is None else float(grad.detach().norm().item())
    logs["parameter_leak_max_abs_diff_final"] = float(parameter_leak_max_abs_diff(adapter.model, snapshot))
    logs["query_total_loss_before_update"] = logs.get("query_protected_loss_before_update", 0.0) + logs.get(
        "query_authorized_loss_before_update", 0.0
    )
    logs["query_total_loss_after_poisoned_update"] = logs.get("query_protected_loss_after_update", 0.0) + logs.get(
        "query_authorized_loss_after_update", 0.0
    )
    logs["query_total_loss_after_clean_update"] = logs.get("query_protected_loss_clean_update", 0.0) + logs.get(
        "query_authorized_loss_clean_update", 0.0
    )
    logs.update(cuda_memory(support_images.device))
    return logs


def run_functional_forward_check(
    method: LearningTrajectoryMethod,
    adapter: YOLOv8TALAdapter,
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    support_batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    selected = adapter.get_named_trainable_parameters("head")
    snapshot = snapshot_parameters(adapter.model)
    delta = torch.zeros((1, 3, support_images.shape[-2], support_images.shape[-1]), device=support_images.device)
    delta.requires_grad_(True)
    support_predictions = adapter.forward((support_images + delta).clamp(0.0, 1.0))
    support_loss = adapter.compute_detection_loss(support_predictions, support_batch)
    virtual = make_virtual_parameters(adapter.model, selected, support_loss, lr=0.001, create_graph=True)
    base_predictions = adapter.forward(query_images)
    virtual_predictions = adapter.forward_with_parameters(query_images, virtual.updated_parameters)
    score_diff = (virtual_predictions["scores"] - base_predictions["scores"]).detach().abs().max()
    box_diff = (virtual_predictions["boxes"] - base_predictions["boxes"]).detach().abs().max()
    return {
        "selected_parameter_count": float(len(selected)),
        "virtual_parameter_update_norm": float(virtual.update_norm.detach().item()),
        "functional_scores_max_abs_diff": float(score_diff.item()),
        "functional_boxes_max_abs_diff": float(box_diff.item()),
        "parameter_leak_max_abs_diff": float(parameter_leak_max_abs_diff(adapter.model, snapshot)),
    }


def run_p2_memory_stability(
    method: LearningTrajectoryMethod,
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    support_batch: Dict[str, torch.Tensor],
    query_batch: Dict[str, torch.Tensor],
    iters: int,
    eps: float,
) -> Dict[str, object]:
    device = support_images.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records: List[Dict[str, float]] = []
    for idx in range(iters):
        delta = make_initial_delta(
            (1, 3, support_images.shape[-2], support_images.shape[-1]),
            device,
            support_images.dtype,
            eps,
            seed=3001,
        )
        result = method.compute_p2_step(support_images, query_images, support_batch, query_batch, delta)
        grad = torch.autograd.grad(result["loss"], delta, retain_graph=False, allow_unused=True)[0]
        record = {"iteration": float(idx + 1), "loss": float(result["loss"].detach().item())}
        record["delta_grad_norm"] = 0.0 if grad is None else float(grad.detach().norm().item())
        record.update(cuda_memory(device))
        records.append(record)
        del result, grad, delta
    allocated = [row.get("allocated_mb", 0.0) for row in records]
    slope = 0.0
    if len(allocated) >= 2:
        x = np.arange(len(allocated), dtype=np.float64)
        slope = float(np.polyfit(x, np.asarray(allocated, dtype=np.float64), deg=1)[0])
    return {
        "iterations": iters,
        "records": records,
        "allocated_mb_slope_per_iter": slope,
        "allocated_mb_first": allocated[0] if allocated else 0.0,
        "allocated_mb_last": allocated[-1] if allocated else 0.0,
        "allocated_mb_max": max(allocated) if allocated else 0.0,
    }


def run_mode_smokes(
    p1_method: LearningTrajectoryMethod,
    meta_method: LearningTrajectoryMethod,
    combo_method: LearningTrajectoryMethod,
    adapter: YOLOv8TALAdapter,
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    support_batch: Dict[str, torch.Tensor],
    query_batch: Dict[str, torch.Tensor],
    eps: float,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    results["p1_only"] = run_p1_once(p1_method, adapter, support_images, support_batch, eps)
    results["meta_only"] = run_p2_once(meta_method, adapter, support_images, query_images, support_batch, query_batch, eps)
    results["p1_plus_meta"] = run_p2_once(
        combo_method,
        adapter,
        support_images,
        query_images,
        support_batch,
        query_batch,
        eps,
    )
    return results


def run_legacy_best_smoke(
    voc_sample: Dict,
    ckpt: Path,
    global_params_path: Path,
    config_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    try:
        from ue_framework.methods.tausb_universal import TAUSBMaskGenerator

        cfg = read_yaml(config_path)
        cfg["surrogate"]["ckpt"] = str(ckpt)
        cfg["surrogate"]["imgsz"] = 640
        cfg["data"]["instance_mask_dir"] = ""
        cfg["platform"]["run_root"] = str(ROOT_DIR / "runs")
        method_cfg = deepcopy(cfg["methods"]["tausb_mask"])
        wrapper = YOLO(str(ckpt))
        surrogate = wrapper.model.to(device)
        surrogate.eval()
        generator = TAUSBMaskGenerator(cfg, method_cfg, device, surrogate, str(global_params_path))
        image = np.asarray(Image.open(voc_sample["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        annotations = [
            {"cls": int(cls_id), "bbox": [float(v) for v in box]}
            for cls_id, box in zip(voc_sample["labels"].tolist(), voc_sample["boxes_xywh"].tolist())
        ]
        result = generator.generate(
            image=image,
            annotations=annotations,
            seed=0,
            steps=40,
            eps=float(cfg["experiment"]["eps"]),
            support_type="mask",
            image_path=voc_sample["image_path"],
        )
        perturb = np.asarray(result.perturbation)
        return {
            "ok": True,
            "sample_id": voc_sample["image_id"],
            "support_source": str(result.extras.get("support_source", "")),
            "is_poisoned": bool(result.extras.get("is_poisoned", False)),
            "support_ratio": float(np.mean(np.asarray(result.support_mask) > 0.5)),
            "perturbed_area_ratio": float(np.mean(np.max(np.abs(perturb), axis=2) > (1.0 / 255.0))),
            "linf": float(np.max(np.abs(perturb))),
            "losses": {k: float(v) for k, v in result.losses.items()},
            "note": str(result.extras.get("note", "")),
        }
    except Exception:
        return {"ok": False, "exception": traceback.format_exc()}


def collect_environment(device: torch.device) -> Dict[str, object]:
    env: Dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "git_head": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "git_status_short": run_command(["git", "status", "--short"], timeout=60),
        "git_diff_legacy_best_names": run_command(["git", "diff", "--name-status", "legacy-best...HEAD"], timeout=60),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            timeout=10,
        ),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(device)
        env["cuda_device_properties"] = str(torch.cuda.get_device_properties(device))
    return env


def write_markdown_report(report_path: Path, results: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env = results.get("environment", {})
    p1_once = results.get("p1_once", {})
    p2_once = results.get("p2_once", {})
    memory = results.get("p2_memory_stability", {})
    legacy = results.get("legacy_best_smoke", {})
    tests = results.get("unit_tests", {})
    lines = [
        "# Local P1/P2 Validation Report",
        "",
        "## Git And Environment",
        "",
        f"- HEAD: `{env.get('git_head', {}).get('stdout', '')}`",
        f"- Branch: `{env.get('git_branch', {}).get('stdout', '')}`",
        f"- Python: `{env.get('python_executable', '')}`",
        f"- Torch: `{env.get('torch_version', '')}`, CUDA: `{env.get('torch_cuda_version', '')}`",
        f"- GPU: `{env.get('cuda_device_name', 'cpu')}`",
        "",
        "## Architecture Check",
        "",
        f"- New-method forbidden import hits: `{results.get('new_method_import_audit', {}).get('hits', {})}`",
        f"- Functional scores max abs diff: `{results.get('functional_forward_check', {}).get('functional_scores_max_abs_diff', 'n/a')}`",
        f"- Functional parameter leak max abs diff: `{results.get('functional_forward_check', {}).get('parameter_leak_max_abs_diff', 'n/a')}`",
        "",
        "## Category Loss Isolation",
        "",
        "```json",
        json.dumps(results.get("artificial_isolation", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## P1 Gradient Validation",
        "",
        f"- cos_protected_clean_poison: `{p1_once.get('cos_protected_clean_poison', 'n/a')}`",
        f"- cos_authorized_clean_poison: `{p1_once.get('cos_authorized_clean_poison', 'n/a')}`",
        f"- p1_loss: `{p1_once.get('p1_loss', 'n/a')}`",
        f"- delta_grad_norm: `{p1_once.get('delta_grad_norm', 'n/a')}`",
        f"- surrogate_parameter_max_abs_diff: `{p1_once.get('surrogate_parameter_max_abs_diff', 'n/a')}`",
        f"- 20-step sequence saved in `{results.get('outputs', {}).get('p1_sequence_json', '')}`",
        "",
        "## P2 Virtual Update Validation",
        "",
        f"- virtual_parameter_update_norm: `{p2_once.get('virtual_parameter_update_norm', 'n/a')}`",
        f"- parameter_leak_max_abs_diff: `{p2_once.get('parameter_leak_max_abs_diff', 'n/a')}`",
        f"- query loss before update: `{p2_once.get('query_total_loss_before_update', 'n/a')}`",
        f"- query loss after clean update: `{p2_once.get('query_total_loss_after_clean_update', 'n/a')}`",
        f"- query loss after poisoned update: `{p2_once.get('query_total_loss_after_poisoned_update', 'n/a')}`",
        f"- protected_learning_gap: `{p2_once.get('protected_learning_gap', 'n/a')}`",
        f"- authorized_learning_gap: `{p2_once.get('authorized_learning_gap', 'n/a')}`",
        f"- meta_gradient_norm_to_delta: `{p2_once.get('meta_gradient_norm_to_delta', 'n/a')}`",
        "",
        "## Memory Stability",
        "",
        f"- iterations: `{memory.get('iterations', 'n/a')}`",
        f"- allocated first/last/max MB: `{memory.get('allocated_mb_first', 'n/a')}` / `{memory.get('allocated_mb_last', 'n/a')}` / `{memory.get('allocated_mb_max', 'n/a')}`",
        f"- allocated slope MB/iter: `{memory.get('allocated_mb_slope_per_iter', 'n/a')}`",
        "",
        "## Legacy-Best Compatibility",
        "",
        "```json",
        json.dumps(legacy, indent=2, sort_keys=True),
        "```",
        "",
        "## Unit Tests",
        "",
        f"- tests/run_tests.py returncode: `{tests.get('run_tests', {}).get('returncode', 'n/a')}`",
        f"- pytest returncode: `{tests.get('pytest', {}).get('returncode', 'n/a')}`",
        "",
        "## Unresolved Issues",
        "",
    ]
    issues = results.get("unresolved_issues", [])
    if issues:
        lines.extend([f"- {issue}" for issue in issues])
    else:
        lines.append("- None from this local validation run.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            str(results.get("recommendation", "")),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    results: Dict[str, object] = {"outputs": {}, "exceptions": {}, "unresolved_issues": []}
    results["environment"] = collect_environment(device)
    results["new_method_import_audit"] = audit_new_method_imports()
    results["unit_tests"] = {
        "run_tests": run_command([sys.executable, "tests/run_tests.py"], timeout=180),
        "pytest": run_command([sys.executable, "-m", "pytest", "tests", "-q"], timeout=180),
    }

    voc_root = Path(args.voc_root).resolve()
    ckpt = Path(args.ckpt).resolve()
    global_params_path = Path(args.global_params).resolve()
    catalog, voc_stats = load_voc_catalog(voc_root)
    selected = choose_validation_samples(catalog)
    results["voc_stats"] = voc_stats
    results["selected_samples"] = {
        key: {
            "image_id": sample["image_id"],
            "classes": sample["classes"],
            "has_person": sample["has_person"],
            "has_authorized": sample["has_authorized"],
            "max_person_authorized_iou": sample["max_person_authorized_iou"],
        }
        for key, sample in selected.items()
    }

    wrapper, model = load_yolo_model(ckpt, device)
    names = getattr(wrapper, "names", {})
    results["surrogate_alignment"] = {
        "num_names": len(names) if hasattr(names, "__len__") else None,
        "class_14_name": names.get(14) if isinstance(names, dict) else names[14],
        "expected_class_14_name": "person",
        "ok": (names.get(14) if isinstance(names, dict) else names[14]) == "person",
    }
    adapter = YOLOv8TALAdapter(model, num_classes=len(VOC_CLASSES), protected_class_id=PROTECTED_CLASS_ID, tal_topk=10)
    p1_method = LearningTrajectoryMethod(adapter, base_method_config(use_p1_regularizer=False))
    meta_method = LearningTrajectoryMethod(adapter, base_method_config(use_p1_regularizer=False))
    combo_method = LearningTrajectoryMethod(adapter, base_method_config(use_p1_regularizer=True))

    eps = float(args.eps)
    support_samples = [selected["protected_only"], selected["authorized_only"]]
    query_samples = [selected["mixed_low_overlap"], selected["mixed_high_overlap"]]
    support_images, support_batch = collate_samples(support_samples, args.imgsz, device)
    query_images, query_batch = collate_samples(query_samples, args.imgsz, device)

    try:
        results["artificial_isolation"] = run_artificial_isolation(adapter, device, args.imgsz)
    except Exception:
        results["exceptions"]["artificial_isolation"] = traceback.format_exc()
        results["unresolved_issues"].append("Artificial class isolation check failed.")

    try:
        results["p1_once"] = run_p1_once(p1_method, adapter, support_images, support_batch, eps)
    except Exception:
        results["exceptions"]["p1_once"] = traceback.format_exc()
        results["unresolved_issues"].append("P1 one-batch check failed.")

    try:
        p1_sequence = run_p1_optimization(p1_method, support_images, support_batch, eps, args.p1_steps, args.p1_lr)
        results["p1_optimization"] = p1_sequence
        p1_sequence_path = output_dir / "p1_20_step_sequence.json"
        write_json(p1_sequence_path, p1_sequence)
        results["outputs"]["p1_sequence_json"] = str(p1_sequence_path)
    except Exception:
        results["exceptions"]["p1_optimization"] = traceback.format_exc()
        results["unresolved_issues"].append("P1 20-step optimization failed.")

    try:
        results["functional_forward_check"] = run_functional_forward_check(
            meta_method, adapter, support_images, query_images, support_batch
        )
    except Exception:
        results["exceptions"]["functional_forward_check"] = traceback.format_exc()
        results["unresolved_issues"].append("Functional forward virtual-parameter check failed.")

    try:
        results["p2_once"] = run_p2_once(
            meta_method,
            adapter,
            support_images,
            query_images,
            support_batch,
            query_batch,
            eps,
        )
    except Exception:
        results["exceptions"]["p2_once"] = traceback.format_exc()
        results["unresolved_issues"].append("P2 one-batch check failed.")

    try:
        results["mode_smokes"] = run_mode_smokes(
            p1_method,
            meta_method,
            combo_method,
            adapter,
            support_images,
            query_images,
            support_batch,
            query_batch,
            eps,
        )
    except Exception:
        results["exceptions"]["mode_smokes"] = traceback.format_exc()
        results["unresolved_issues"].append("P1 only / Meta only / P1+Meta smoke failed.")

    try:
        results["p2_memory_stability"] = run_p2_memory_stability(
            meta_method,
            support_images,
            query_images,
            support_batch,
            query_batch,
            args.memory_iters,
            eps,
        )
    except Exception:
        results["exceptions"]["p2_memory_stability"] = traceback.format_exc()
        results["unresolved_issues"].append("P2 memory stability loop failed.")

    try:
        results["legacy_best_smoke"] = run_legacy_best_smoke(
            selected["protected_only"],
            ckpt,
            global_params_path,
            Path(args.legacy_config).resolve(),
            device,
        )
        if not results["legacy_best_smoke"].get("ok", False):
            results["unresolved_issues"].append("Legacy-best smoke failed.")
    except Exception:
        results["legacy_best_smoke"] = {"ok": False, "exception": traceback.format_exc()}
        results["unresolved_issues"].append("Legacy-best smoke failed.")

    p1 = results.get("p1_once", {})
    p2 = results.get("p2_once", {})
    memory = results.get("p2_memory_stability", {})
    mechanisms_pass = (
        not results["unresolved_issues"]
        and float(p1.get("delta_grad_norm", 0.0)) > 0.0
        and float(p2.get("meta_gradient_norm_to_delta", 0.0)) > 0.0
        and float(p2.get("parameter_leak_max_abs_diff_final", 1.0)) == 0.0
        and abs(float(memory.get("allocated_mb_slope_per_iter", 0.0))) < 1.0
    )
    results["recommendation"] = (
        "Mechanism checks pass locally; it is reasonable to proceed to a small victim retraining smoke."
        if mechanisms_pass
        else "Do not start victim retraining yet; resolve the unresolved validation issues first."
    )
    results["mechanisms_pass"] = mechanisms_pass

    results_path = output_dir / "local_validation_results.json"
    write_json(results_path, results)
    results["outputs"]["results_json"] = str(results_path)
    report_path = ROOT_DIR / "docs" / "local_validation_report.md"
    write_markdown_report(report_path, results)
    results["outputs"]["markdown_report"] = str(report_path)
    write_json(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local real YOLOv8/VOC validation for P1/P2.")
    parser.add_argument("--voc-root", default=str(ROOT_DIR / "outputs/local_validation/voc_raw/VOCdevkit/VOC2007"))
    parser.add_argument("--ckpt", default=str(ROOT_DIR / "checkpoints/voc20_surrogate.pt"))
    parser.add_argument("--global-params", default=str(ROOT_DIR / "runs/artifacts/tausb_mask/steps40/seed0/noise/global_params.pt"))
    parser.add_argument("--legacy-config", default=str(ROOT_DIR / "ue_framework/configs/exp_voc_person_tausb_fhml2_cooccur_hinge_full.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs/local_validation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--eps", type=float, default=16.0 / 255.0)
    parser.add_argument("--p1-steps", type=int, default=20)
    parser.add_argument("--p1-lr", type=float, default=0.01)
    parser.add_argument("--memory-iters", type=int, default=50)
    args = parser.parse_args()
    started = time.time()
    results = run_validation(args)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[local_validation] elapsed_sec={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
