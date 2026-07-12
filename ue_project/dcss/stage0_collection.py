import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

from ue_framework.data_utils import label_path_for_image, list_images, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.alce_acgt import project_strict_gate_to_fpn
from ue_framework.ultra.hijacked_loss import HijackedV8Loss

from .feature_hooks import FeatureHookBank
from .statistics import RunningCovariance
from .unit_partition import partition_tal_units


@dataclass
class VectorSummary:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = torch.zeros(self.dimension, dtype=torch.float64)
        self.energy_sum = 0.0

    def update(self, values: torch.Tensor) -> None:
        x = values.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.numel() == 0:
            return
        self.count += int(x.shape[0])
        self.sum += x.sum(dim=0)
        self.energy_sum += float(x.square().sum().item())

    def state_dict(self) -> Dict:
        return {"count": self.count, "sum": self.sum, "energy_sum": self.energy_sum}


def _git_text(arguments: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *arguments], text=True, encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _letterbox_with_annotations(
    image: np.ndarray,
    annotations: List[dict],
    image_size: int,
) -> Tuple[torch.Tensor, List[dict]]:
    height, width = image.shape[:2]
    scale = min(image_size / float(height), image_size / float(width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_top = (image_size - new_h) // 2
    pad_bottom = image_size - new_h - pad_top
    pad_left = (image_size - new_w) // 2
    pad_right = image_size - new_w - pad_left
    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.447)

    adjusted = []
    for annotation in annotations:
        box = annotation.get("bbox")
        if box is None or len(box) != 4:
            continue
        cx, cy, bw, bh = [float(value) for value in box]
        cx = (cx * width * scale + pad_left) / image_size
        cy = (cy * height * scale + pad_top) / image_size
        bw = bw * width * scale / image_size
        bh = bh * height * scale / image_size
        adjusted.append({"cls": int(annotation["cls"]), "bbox": [cx, cy, bw, bh]})
    return tensor, adjusted


def _batch_from_annotations(annotations: List[dict], image: torch.Tensor, device: torch.device) -> Dict:
    classes = [[float(annotation["cls"])] for annotation in annotations]
    boxes = [annotation["bbox"] for annotation in annotations]
    return {
        "batch_idx": torch.zeros(len(annotations), dtype=torch.long, device=device),
        "cls": torch.tensor(classes, dtype=torch.float32, device=device) if classes else torch.zeros((0, 1), device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device) if boxes else torch.zeros((0, 4), device=device),
        "batch_size": 1,
        "img": image,
    }


def _gather_vectors(tensor: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
    values = tensor.permute(0, 2, 3, 1)
    mask = spatial_mask[:, 0].bool()
    return values[mask]


def _scope_state(dimension: int) -> Dict[str, RunningCovariance]:
    return {
        "target_gradient": RunningCovariance(dimension),
        "non_target_gradient": RunningCovariance(dimension),
        "target_feature": RunningCovariance(dimension),
    }


def _state_dict(scopes: Dict[str, Dict[str, RunningCovariance]]) -> Dict:
    return {
        scope: {name: statistic.state_dict() for name, statistic in values.items()}
        for scope, values in scopes.items()
    }


def collect_stage0(config: Dict, output_dir: str, max_images: int = 0) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"dcss.stage0.{os.path.basename(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8"))

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(config.get("deterministic", True))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    dataset_root = os.path.abspath(os.path.expandvars(config["data"]["dataset_root"]))
    image_dir = os.path.join(dataset_root, config["data"].get("train_images", "images/train"))
    label_dir = os.path.join(dataset_root, config["data"].get("train_labels", "labels/train"))
    checkpoint = os.path.abspath(os.path.expandvars(config["surrogate"]["checkpoint"]))
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(f"DCSS dataset directories missing: {image_dir}, {label_dir}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"DCSS surrogate checkpoint missing: {checkpoint}")

    device = torch.device(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    wrapper = YOLO(checkpoint)
    if len(wrapper.names) != int(config["surrogate"].get("num_classes", 20)):
        raise RuntimeError(f"surrogate class mismatch: expected 20, got {len(wrapper.names)}")
    model = wrapper.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layer_names = list(config["stage0"].get("layers", ["model.15", "model.18", "model.21"]))
    hooks = FeatureHookBank(model, layer_names)
    hijacked = HijackedV8Loss.from_surrogate(
        model,
        num_classes=int(config["surrogate"].get("num_classes", 20)),
        target_class_id=int(config["experiment"].get("target_class_id", 14)),
    )
    hijacked.enable_strict_assign_probe = False

    images = list_images(image_dir)
    if max_images > 0:
        images = images[: int(max_images)]
    if not images:
        raise RuntimeError("no Stage 0 images found")

    scopes_by_layer: Dict[str, Dict[str, Dict[str, RunningCovariance]]] = {}
    class_summaries: Dict[str, Dict[int, VectorSummary]] = {}
    count_summaries: Dict[str, Dict[str, int]] = {}
    processed = 0
    skipped_no_fg = 0
    image_size = int(config["surrogate"].get("imgsz", 640))
    target_class_id = int(config["experiment"].get("target_class_id", 14))
    pag_ratios = config["stage0"].get("pag_layer_ratios", [0.7, 0.6, 0.4])
    pag_minimums = config["stage0"].get("pag_min_pos", [8, 6, 4])

    try:
        for image_index, image_path in enumerate(images):
            annotations = read_yolo_annotations(label_path_for_image(image_path, label_dir))
            if not annotations:
                continue
            image_np = load_image_rgb_float(image_path)
            image, adjusted = _letterbox_with_annotations(image_np, annotations, image_size)
            image = image.to(device).requires_grad_(True)
            batch = _batch_from_annotations(adjusted, image, device)
            hooks.clear()
            output = model(image)
            predictions = output[0] if isinstance(output, (tuple, list)) else output
            features = {name: hooks.outputs[name] for name in layer_names}
            hijacked.get_assigned_targets_and_loss(predictions, batch)
            assignment = hijacked.last_real_assign
            if not all(torch.is_tensor(assignment.get(name)) for name in ["fg_mask", "target_labels", "target_scores"]):
                raise RuntimeError(f"real TAL assignment unavailable for {image_path}")
            partition = partition_tal_units(
                assignment["fg_mask"],
                assignment["target_labels"],
                assignment["target_scores"],
                target_class_id,
                layer_names,
                features,
                pag_ratios,
                pag_minimums,
            )
            if partition.stats["num_fg"] == 0:
                skipped_no_fg += 1
                continue

            for layer_name, feature in features.items():
                dimension = int(feature.shape[1])
                if layer_name not in scopes_by_layer:
                    scopes_by_layer[layer_name] = {
                        "full": _scope_state(dimension),
                        "split_a": _scope_state(dimension),
                        "split_b": _scope_state(dimension),
                    }
                    class_summaries[layer_name] = {}
                    count_summaries[layer_name] = {
                        "num_fg": 0,
                        "num_target_positive": 0,
                        "num_selected_target": 0,
                        "num_non_target_positive": 0,
                    }

            losses: List[Tuple[str, int, torch.Tensor, torch.Tensor]] = []
            target_gate = partition.selected_target_gate
            if target_gate.any():
                probability = predictions[:, 4 + target_class_id, :].clamp(1e-8, 1.0 - 1e-8)
                losses.append(("target", target_class_id, -probability[target_gate].log().mean(), target_gate))
            for class_id, class_gate in sorted(partition.non_target_class_gates.items()):
                if class_gate.any():
                    probability = predictions[:, 4 + class_id, :].clamp(1e-8, 1.0 - 1e-8)
                    losses.append(("non_target", class_id, -probability[class_gate].log().mean(), class_gate))

            feature_tuple = tuple(features[name] for name in layer_names)
            split_name = "split_a" if image_index % 2 == 0 else "split_b"
            for loss_index, (kind, class_id, loss, unit_gate) in enumerate(losses):
                gradients = torch.autograd.grad(
                    loss,
                    feature_tuple,
                    retain_graph=loss_index < len(losses) - 1,
                    allow_unused=True,
                )
                unit_maps = project_strict_gate_to_fpn(unit_gate, layer_names, features)
                for layer_name, gradient in zip(layer_names, gradients):
                    if gradient is None:
                        raise RuntimeError(f"classification loss is disconnected from {layer_name}")
                    vectors = _gather_vectors(gradient, unit_maps[layer_name])
                    if kind == "target":
                        feature_vectors = _gather_vectors(features[layer_name], unit_maps[layer_name])
                        for scope in ["full", split_name]:
                            scopes_by_layer[layer_name][scope]["target_gradient"].update(vectors)
                            scopes_by_layer[layer_name][scope]["target_feature"].update(feature_vectors)
                    else:
                        for scope in ["full", split_name]:
                            scopes_by_layer[layer_name][scope]["non_target_gradient"].update(vectors)
                        summary = class_summaries[layer_name].setdefault(class_id, VectorSummary(vectors.shape[1]))
                        summary.update(vectors)

            offset = 0
            for layer_name in layer_names:
                size = features[layer_name].shape[-2] * features[layer_name].shape[-1]
                layer_slice = slice(offset, offset + size)
                count_summaries[layer_name]["num_fg"] += int(partition.target_gate[:, layer_slice].sum().item() + partition.non_target_gate[:, layer_slice].sum().item())
                count_summaries[layer_name]["num_target_positive"] += int(partition.target_gate[:, layer_slice].sum().item())
                count_summaries[layer_name]["num_selected_target"] += int(partition.selected_target_gate[:, layer_slice].sum().item())
                count_summaries[layer_name]["num_non_target_positive"] += int(partition.non_target_gate[:, layer_slice].sum().item())
                offset += size
            processed += 1
            if processed % int(config["stage0"].get("log_every", 10)) == 0:
                logger.info("processed=%d/%d skipped_no_fg=%d", processed, len(images), skipped_no_fg)
            del image, predictions, output

        payload = {
            "schema_version": 1,
            "seed": seed,
            "checkpoint": checkpoint,
            "dataset_root": dataset_root,
            "layers": {},
            "processed_images": processed,
            "requested_images": len(images),
            "skipped_no_fg": skipped_no_fg,
            "gradient_definition": "classification-only negative log assigned-class probability gradient with respect to FPN feature",
            "statistics_mode": "full-channel streaming covariance/second-moment",
        }
        for layer_name in layer_names:
            payload["layers"][layer_name] = {
                "scopes": _state_dict(scopes_by_layer[layer_name]),
                "class_summaries": {class_id: value.state_dict() for class_id, value in class_summaries[layer_name].items()},
                "counts": count_summaries[layer_name],
                "dimension": scopes_by_layer[layer_name]["full"]["target_gradient"].dimension,
            }
        torch.save(payload, os.path.join(output_dir, "raw_statistics.pt"))
        logger.info("collection complete: processed=%d requested=%d", processed, len(images))
        return payload
    finally:
        hooks.close()


def write_provenance(config: Dict, output_dir: str, command: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    with open(os.path.join(output_dir, "git_commit.txt"), "w", encoding="utf-8") as file:
        file.write(_git_text(["rev-parse", "HEAD"]) + "\n")
        file.write(_git_text(["status", "--short"]) + "\n")
    with open(os.path.join(output_dir, "environment.txt"), "w", encoding="utf-8") as file:
        file.write(f"platform={platform.platform()}\npython={platform.python_version()}\n")
        file.write(f"torch={torch.__version__}\ncuda_available={torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            file.write(f"cuda_device={torch.cuda.get_device_name(0)}\n")
    with open(os.path.join(output_dir, "command.txt"), "w", encoding="utf-8") as file:
        file.write(command + "\n")
