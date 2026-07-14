from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
import ultralytics
import yaml

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier
from oa_lgc.gains import ClassGainInput, authorized_learning_gain, carrier_query_loss, target_learning_gain
from oa_lgc.objective import CoreObjectiveConfig, compose_core_objective
from oa_lgc.yolo_adapter import YOLOFunctionalAdapter
from ue_framework.data_utils import (
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gradient_norm(loss: torch.Tensor, delta: torch.Tensor, retain_graph: bool = True) -> float:
    gradient = torch.autograd.grad(loss, delta, retain_graph=retain_graph, allow_unused=True)[0]
    if gradient is None:
        return 0.0
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite delta gradient")
    return float(gradient.detach().norm())


def _class_count(annotations: list[dict], class_id: int) -> int:
    return sum(int(annotation["cls"]) == int(class_id) for annotation in annotations)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(config.get("deterministic", True)), warn_only=True)
    device = torch.device(config["device"])

    adapter = YOLOFunctionalAdapter.from_checkpoint(
        config["surrogate"]["checkpoint"],
        device=device,
        num_classes=int(config["surrogate"]["num_classes"]),
        target_class_id=int(config["experiment"]["target_class_id"]),
    )
    base_hash_before = adapter.hash_base_state()
    dataset_root = Path(config["data"]["dataset_root"])
    image_dir = dataset_root / config["data"]["train_images"]
    label_dir = dataset_root / config["data"]["train_labels"]
    samples = []
    for path in list_images(str(image_dir)):
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        if any(int(annotation["cls"]) == int(config["experiment"]["target_class_id"]) for annotation in annotations):
            image, adjusted = _letterbox_with_annotations(
                load_image_rgb_float(path), annotations, int(config["surrogate"]["imgsz"])
            )
            samples.append((Path(path).stem, image.to(device), adjusted))
        if len(samples) == 2:
            break
    if len(samples) != 2 or samples[0][0] == samples[1][0]:
        raise RuntimeError("failed to build one disjoint real-YOLO support/query episode")
    support_id, support_image, support_annotations = samples[0]
    query_id, query_image, query_annotations = samples[1]
    support_batch = _batch_from_annotations(support_annotations, support_image, device)
    query_batch = _batch_from_annotations(query_annotations, query_image, device)

    carrier_cfg = CarrierConfig(
        target_class_id=int(config["experiment"]["target_class_id"]),
        eps=float(config["carrier"]["eps"]),
        non_target_dilation=int(config["carrier"]["non_target_dilation"]),
        soft_mask=bool(config["carrier"]["soft_mask"]),
        soft_edge_pixels=float(config["carrier"]["soft_edge_pixels"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    delta = torch.nn.Parameter(
        torch.randn(
            (3, int(config["carrier"]["object_resolution"]), int(config["carrier"]["object_resolution"])),
            generator=generator,
            device=device,
        )
        * 1e-3
    )
    poison_support = apply_object_aligned_carrier(
        support_image[0], support_annotations, delta, carrier_cfg
    )
    poison_support_batch = _batch_from_annotations(
        support_annotations, poison_support.poisoned.unsqueeze(0), device
    )
    poison_query = apply_object_aligned_carrier(query_image[0], query_annotations, delta, carrier_cfg)
    poison_query_batch = _batch_from_annotations(
        query_annotations, poison_query.poisoned.unsqueeze(0), device
    )

    steps = int(config["virtual_update"]["steps"])
    learning_rate = float(config["virtual_update"]["learning_rate"])
    clean_trajectory = adapter.virtual_update(
        support_batch, steps, learning_rate, "classification_head_only", create_graph=True
    )
    poison_trajectory = adapter.virtual_update(
        poison_support_batch, steps, learning_rate, "classification_head_only", create_graph=True
    )
    detection_trajectory = adapter.virtual_update(
        poison_support_batch, steps, learning_rate, "detection_head", create_graph=False
    )
    neck_trajectory = adapter.virtual_update(
        poison_support_batch, steps, learning_rate / 10.0, "selected_neck_and_head", create_graph=False
    )
    full_manifest = adapter.parameter_manifest("full_model")

    reference = adapter.reference_assignment(query_image, query_batch)
    initial_query = adapter.compute_classwise_query_loss(
        query_image,
        query_batch,
        adapter.base_parameters(),
        adapter.clone_buffers(),
        reference,
        "fixed_clean_reference",
    )
    clean_query = adapter.compute_classwise_query_loss(
        query_image,
        query_batch,
        clean_trajectory.parameters,
        clean_trajectory.buffers,
        reference,
        "fixed_clean_reference",
    )
    poison_query_updated = adapter.compute_classwise_query_loss(
        query_image,
        query_batch,
        poison_trajectory.parameters,
        poison_trajectory.buffers,
        reference,
        "fixed_clean_reference",
    )
    target_class = int(config["experiment"]["target_class_id"])
    if not all(result.valid[target_class] for result in (initial_query, clean_query, poison_query_updated)):
        raise RuntimeError("target class has no fixed-reference positive query loss")
    target_gain = target_learning_gain(
        initial_query.losses[target_class],
        clean_query.losses[target_class],
        poison_query_updated.losses[target_class],
        rho_t=float(config["objective"]["rho_t"]),
        min_valid_clean_gain=-1e9,
    )
    if not target_gain.valid:
        raise RuntimeError(f"target gain invalid: {target_gain.invalid_reason}")

    class_inputs = {}
    for class_id in range(int(config["surrogate"]["num_classes"])):
        if class_id == target_class:
            continue
        if all(result.valid[class_id] for result in (initial_query, clean_query, poison_query_updated)):
            class_inputs[class_id] = ClassGainInput(
                initial_query.losses[class_id],
                clean_query.losses[class_id],
                poison_query_updated.losses[class_id],
                _class_count(support_annotations, class_id),
                _class_count(query_annotations, class_id),
            )
    authorized = authorized_learning_gain(
        class_inputs,
        target_class_id=target_class,
        rho_k=float(config["objective"]["rho_k"]),
        min_valid_class_gain=0.0,
        minimum_class_samples=1,
    )
    authorized_loss = authorized.loss.to(device)
    if not authorized_loss.requires_grad:
        authorized_loss = authorized_loss + delta.sum() * 0.0

    carrier_query_result = adapter.compute_classwise_query_loss(
        poison_query.poisoned.unsqueeze(0),
        poison_query_batch,
        adapter.base_parameters(),
        adapter.clone_buffers(),
        reference,
        "fixed_clean_reference",
    )
    carrier_loss = carrier_query_loss(carrier_query_result.losses[target_class])
    regularizer = delta.square().mean()
    objective = compose_core_objective(
        target_gain.protect_loss,
        carrier_loss,
        authorized_loss,
        delta,
        CoreObjectiveConfig(
            lambda_carrier=float(config["objective"]["lambda_carrier"]),
            lambda_auth=float(config["objective"]["lambda_auth"]),
            lambda_reg=float(config["objective"]["lambda_reg"]),
            eps=float(config["carrier"]["eps"]),
        ),
    )
    gradient_decomposition = {
        "protect_only_grad_norm": _gradient_norm(target_gain.protect_loss, delta),
        "carrier_only_grad_norm": _gradient_norm(carrier_loss, delta),
        "auth_only_grad_norm": _gradient_norm(authorized_loss, delta),
        "regularizer_only_grad_norm": _gradient_norm(regularizer, delta),
        "total_grad_norm": _gradient_norm(objective.loss, delta, retain_graph=False),
        "protect_query_uses_poison_pixels": False,
        "protect_path": "clean query -> poison fast parameters -> poison support carrier -> delta_obj",
    }
    if gradient_decomposition["protect_only_grad_norm"] <= 0:
        raise RuntimeError("protect-only mixed derivative is zero")

    reproducible_first = adapter.virtual_update(
        support_batch, 1, learning_rate, "classification_head_only", create_graph=False
    )
    reproducible_second = adapter.virtual_update(
        support_batch, 1, learning_rate, "classification_head_only", create_graph=False
    )
    reproducibility_max_abs = max(
        float(
            (
                reproducible_first.parameters[name] - reproducible_second.parameters[name]
            ).detach().abs().max()
        )
        for name in reproducible_first.manifest.selected_names
    )
    base_hash_after = adapter.hash_base_state()
    state_hashes = {
        "base_before": base_hash_before,
        "base_after": base_hash_after,
        "base_unchanged": base_hash_before == base_hash_after,
        "clean_poison_states_independent": adapter.clean_poison_states_independent(
            clean_trajectory, poison_trajectory
        ),
        "classification_selected_hash": clean_trajectory.manifest.selected_hash,
        "detection_selected_hash": detection_trajectory.manifest.selected_hash,
        "selected_neck_head_hash": neck_trajectory.manifest.selected_hash,
        "full_model_interface_hash": full_manifest.selected_hash,
        "reproducibility_max_abs_difference": reproducibility_max_abs,
        "reproducibility_tolerance": 1e-7,
        "reproducible_within_tolerance": reproducibility_max_abs <= 1e-7,
    }
    if not state_hashes["base_unchanged"] or not state_hashes["clean_poison_states_independent"]:
        raise RuntimeError("functional state isolation gate failed")

    modes = {
        "classification_head_only": clean_trajectory.manifest,
        "detection_head": detection_trajectory.manifest,
        "selected_neck_and_head": neck_trajectory.manifest,
        "full_model": full_manifest,
    }
    parameter_rows = []
    all_parameters = dict(adapter.model.named_parameters())
    for mode, manifest in modes.items():
        selected = set(manifest.selected_names)
        for name, value in all_parameters.items():
            parameter_rows.append(
                {
                    "mode": mode,
                    "name": name,
                    "shape": str(tuple(value.shape)),
                    "numel": value.numel(),
                    "selected": int(name in selected),
                    "selected_hash": manifest.selected_hash,
                }
            )
    _write_csv(
        output / "parameter_manifest.csv",
        ["mode", "name", "shape", "numel", "selected", "selected_hash"],
        parameter_rows,
    )
    buffer_rows = [
        {"name": name, "shape": str(tuple(value.shape)), "numel": value.numel(), "dtype": str(value.dtype)}
        for name, value in adapter.model.named_buffers()
    ]
    _write_csv(output / "buffer_manifest.csv", ["name", "shape", "numel", "dtype"], buffer_rows)
    inner_rows = []
    for trajectory_name, trajectory in (
        ("clean_classification", clean_trajectory),
        ("poison_classification", poison_trajectory),
        ("poison_detection", detection_trajectory),
        ("poison_selected_neck_head", neck_trajectory),
    ):
        for step_index, values in enumerate(trajectory.step_losses, start=1):
            inner_rows.append(
                {
                    "trajectory": trajectory_name,
                    "step": step_index,
                    **values,
                    "parameter_delta_norm": trajectory.parameter_delta_norms[step_index - 1],
                    "runtime_seconds": trajectory.step_times_seconds[step_index - 1],
                }
            )
    _write_csv(
        output / "inner_step_metrics.csv",
        [
            "trajectory",
            "step",
            "total",
            "box",
            "classification",
            "dfl",
            "parameter_delta_norm",
            "runtime_seconds",
        ],
        inner_rows,
    )
    (output / "gradient_decomposition.json").write_text(
        json.dumps(gradient_decomposition, indent=2) + "\n", encoding="utf-8"
    )
    (output / "state_hashes.json").write_text(json.dumps(state_hashes, indent=2) + "\n", encoding="utf-8")
    assignment_payload = {
        "query_assignment_mode": config["experiment"]["query_assignment_mode"],
        "reference_assignment_detached": True,
        "tal_topk_gradient": False,
        "scores_outside_assignment_retain_gradient": True,
        "target_reference_positive_count": initial_query.positive_count[target_class],
        "target_score_mass": initial_query.target_score_mass[target_class],
        "foreground_count": int(reference.fg_mask.sum()),
        "support_id": support_id,
        "query_id": query_id,
        "support_query_overlap": int(support_id == query_id),
    }
    (output / "assignment_mode.json").write_text(
        json.dumps(assignment_payload, indent=2) + "\n", encoding="utf-8"
    )
    memory = {
        "device": str(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
        "gpu_total_bytes": torch.cuda.get_device_properties(device.index or 0).total_memory if device.type == "cuda" else 0,
    }
    (output / "memory_profile.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "total_seconds": time.perf_counter() - started,
        "classification_head_j1_seconds": clean_trajectory.step_times_seconds[0],
        "detection_head_j1_seconds": detection_trajectory.step_times_seconds[0],
        "selected_neck_head_j1_seconds": neck_trajectory.step_times_seconds[0],
    }
    (output / "runtime_metrics.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    environment = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else None,
        "backend": adapter.backend,
        "functional_call_backend": adapter.functional_call_backend,
        "virtual_buffer_mode": adapter.virtual_buffer_mode,
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "pass",
        "backend": adapter.backend,
        "classification_head_j1": True,
        "detection_head_j1": True,
        "selected_neck_head_j1": True,
        "full_model_interface": True,
        "native_detection_loss": True,
        "protect_only_mixed_gradient": gradient_decomposition["protect_only_grad_norm"],
        "base_state_unchanged": state_hashes["base_unchanged"],
        "reproducible_within_tolerance": state_hashes["reproducible_within_tolerance"],
        "target_clean_gain": float(target_gain.clean_gain.detach()),
        "target_poison_gain": float(target_gain.poison_gain.detach()),
        "target_gain_ratio": float(target_gain.ratio.detach()),
        "authorized_valid_class_ids": list(authorized.valid_class_ids),
    }
    (output / "run.log").write_text(
        "C1 real-YOLO adapter smoke pass\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OA-LGC C1 real-YOLO adapter gate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.config, arguments.output)
    print(json.dumps(result, indent=2))
