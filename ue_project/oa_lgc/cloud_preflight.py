from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from ue_framework.data_utils import (
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)
from ue_framework.ultra.hijacked_loss import HijackedV8Loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    return (result.stdout + result.stderr).strip()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _manifest_entries(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _role(name: str, detect_prefix: str) -> str:
    if name.startswith(f"{detect_prefix}.cv3."):
        return "classification_branch"
    if name.startswith(f"{detect_prefix}.cv2."):
        return "box_dfl_distribution_branch"
    if name.startswith(f"{detect_prefix}.dfl."):
        return "dfl_integral_module"
    if name.startswith(f"{detect_prefix}."):
        return "detect_head_other"
    return "backbone_or_neck"


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    log: list[str] = []

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    train_images = dataset_root / "images" / "train"
    train_labels = dataset_root / "labels" / "train"
    val_images = dataset_root / "images" / "val"
    val_labels = dataset_root / "labels" / "val"
    for required in (train_images, train_labels, val_images, val_labels):
        if not required.is_dir():
            raise FileNotFoundError(required)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wrapper = YOLO(str(checkpoint))
    model = wrapper.model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    checkpoint_args = dict(model.args) if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=checkpoint_args)
    detect_index = len(model.model) - 1
    detect_prefix = f"model.{detect_index}"
    detect = model.model[detect_index]

    parameter_lines = ["name\tshape\tnumel\trequires_grad\trole"]
    for name, parameter in model.named_parameters():
        parameter_lines.append(
            f"{name}\t{tuple(parameter.shape)}\t{parameter.numel()}\t{parameter.requires_grad}\t{_role(name, detect_prefix)}"
        )
    (output / "model_parameter_manifest.txt").write_text("\n".join(parameter_lines) + "\n", encoding="utf-8")

    buffer_lines = ["name\tshape\tnumel\tdtype"]
    for name, buffer in model.named_buffers():
        buffer_lines.append(f"{name}\t{tuple(buffer.shape)}\t{buffer.numel()}\t{buffer.dtype}")
    (output / "buffer_manifest.txt").write_text("\n".join(buffer_lines) + "\n", encoding="utf-8")

    train_paths = list_images(str(train_images))
    val_paths = list_images(str(val_images))
    train_ids = {Path(path).stem for path in train_paths}
    val_ids = {Path(path).stem for path in val_paths}
    train_manifest = _manifest_entries(dataset_root / "train.txt")
    val_manifest = _manifest_entries(dataset_root / "val.txt")
    dataset_summary = {
        "dataset_root": str(dataset_root),
        "train_image_count": len(train_paths),
        "train_label_count": len(list(train_labels.glob("*.txt"))),
        "val_image_count": len(val_paths),
        "val_label_count": len(list(val_labels.glob("*.txt"))),
        "train_manifest_path": str(dataset_root / "train.txt"),
        "train_manifest_entries": len(train_manifest),
        "val_manifest_path": str(dataset_root / "val.txt"),
        "val_manifest_entries": len(val_manifest),
        "test_manifest_path": None,
        "test_manifest_entries": None,
        "train_val_stem_overlap_count": len(train_ids & val_ids),
        "dataset_yaml": str(dataset_root / "dataset.yaml"),
    }
    (output / "dataset_manifest_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ckpt = wrapper.ckpt if isinstance(wrapper.ckpt, dict) else {}
    checkpoint_metadata = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "keys": sorted(str(key) for key in ckpt),
        "epoch": _json_value(ckpt.get("epoch")),
        "best_fitness": _json_value(ckpt.get("best_fitness")),
        "date": _json_value(ckpt.get("date")),
        "version": _json_value(ckpt.get("version")),
        "train_args": _json_value(ckpt.get("train_args")),
        "loaded_model_type": f"{type(model).__module__}.{type(model).__name__}",
        "class_count": len(wrapper.names),
        "class_names": _json_value(wrapper.names),
        "checkpoint_args_before_compatibility_merge": _json_value(checkpoint_args),
        "effective_loss_gains": {
            "box": float(model.args.box),
            "cls": float(model.args.cls),
            "dfl": float(model.args.dfl),
        },
    }
    (output / "checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sample_path = None
    sample_annotations = None
    for image_path in train_paths:
        annotations = read_yolo_annotations(label_path_for_image(image_path, str(train_labels)))
        if any(int(annotation["cls"]) == int(args.target_class_id) for annotation in annotations):
            sample_path = image_path
            sample_annotations = annotations
            break
    if sample_path is None or sample_annotations is None:
        raise RuntimeError("no target-class sample found in mini VOC train split")
    image_np = load_image_rgb_float(sample_path)
    image, adjusted = _letterbox_with_annotations(image_np, sample_annotations, args.imgsz)
    image = image.to(device)
    batch = _batch_from_annotations(adjusted, image, device)

    model.eval()
    with torch.no_grad():
        forward_output = model(image)
    forward_tensor = forward_output[0] if isinstance(forward_output, (tuple, list)) else forward_output
    if not torch.is_tensor(forward_tensor) or not torch.isfinite(forward_tensor).all():
        raise RuntimeError("real YOLO forward did not return a finite tensor")
    hijacked = HijackedV8Loss.from_surrogate(
        model, num_classes=len(wrapper.names), target_class_id=args.target_class_id
    )
    hijacked.enable_strict_assign_probe = False
    hijacked.get_assigned_targets_and_loss(forward_tensor, batch)
    assignment = hijacked.last_real_assign
    assignment_fields = ("target_labels", "target_scores", "fg_mask", "target_gt_idx")
    if not hijacked._super_ready or not all(torch.is_tensor(assignment.get(key)) for key in assignment_fields):
        raise RuntimeError("real TAL diagnostics are unavailable")

    model.train()
    model.zero_grad(set_to_none=True)
    loss_components, detached_components = model.loss(batch)
    total_loss = loss_components.sum()
    if loss_components.numel() != 3 or not torch.isfinite(total_loss):
        raise RuntimeError(f"unexpected detection loss: shape={tuple(loss_components.shape)}")
    total_loss.backward()
    gradient_square = sum(
        (parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.grad is not None),
        torch.zeros((), device=device),
    )

    disk_free_bytes = shutil.disk_usage(output).free
    gpu = {}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu = {
            "gpu_name": properties.name,
            "gpu_total_bytes": properties.total_memory,
            "gpu_free_bytes_after_smoke": free_bytes,
            "gpu_runtime_total_bytes": total_bytes,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    environment = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        **gpu,
        "disk_free_bytes": disk_free_bytes,
        "device": str(device),
        "model_parameter_tensors": sum(1 for _ in model.parameters()),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_buffer_tensors": sum(1 for _ in model.buffers()),
        "model_buffer_count": sum(buffer.numel() for buffer in model.buffers()),
        "batchnorm_module_count": sum(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
        ),
        "detect_head_index": detect_index,
        "detect_head_type": f"{type(detect).__module__}.{type(detect).__name__}",
        "classification_parameter_tensors": sum(
            name.startswith(f"{detect_prefix}.cv3.") for name, _ in model.named_parameters()
        ),
        "box_distribution_parameter_tensors": sum(
            name.startswith(f"{detect_prefix}.cv2.") for name, _ in model.named_parameters()
        ),
        "dfl_module_parameter_tensors": sum(
            name.startswith(f"{detect_prefix}.dfl.") for name, _ in model.named_parameters()
        ),
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8"
    )
    (output / "git_status.txt").write_text(
        "branch:\n"
        + _git("branch", "--show-current")
        + "\n\nhead:\n"
        + _git("rev-parse", "HEAD")
        + "\n\nstatus:\n"
        + _git("status", "--short", "--branch")
        + "\n",
        encoding="utf-8",
    )

    log.extend(
        [
            "C0 preflight completed",
            f"sample={sample_path}",
            f"sample_gt_count={len(adjusted)}",
            f"forward_shape={tuple(forward_tensor.shape)}",
            f"loss_components_box_cls_dfl={[float(value) for value in detached_components.flatten()]}",
            f"loss_total={float(total_loss.detach())}",
            f"gradient_norm={float(torch.sqrt(gradient_square))}",
            "tal_interface=ue_framework.ultra.hijacked_loss.HijackedV8Loss.last_real_assign",
            "tal_interface_role=diagnostics_only; native model.loss supplies training TAL/box/DFL",
            f"tal_foreground_count={int(assignment['fg_mask'].sum())}",
            f"tal_target_score_mass={float(assignment['target_scores'].sum())}",
            f"runtime_seconds={time.time() - started:.3f}",
        ]
    )
    (output / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the OA-LGC real-YOLO C0 environment.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--target-class-id", type=int, default=14)
    return parser.parse_args()


if __name__ == "__main__":
    destination = run(parse_args())
    print(destination)
