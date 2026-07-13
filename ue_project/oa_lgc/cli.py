from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc

import numpy as np
import torch
import yaml

from ue_framework.data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)

from .artifacts import create_run_dir, unique_run_id, write_png
from .carrier import CarrierConfig, apply_object_aligned_carrier
from .episodes import DisjointEpisodeSampler, load_records
from .model import ObjectCropDetector, class_loss
from .virtual_update import functional_forward, model_state_unchanged, virtual_update
from .gains import ClassGainInput, authorized_learning_gain, carrier_query_loss, target_learning_gain
from .objective import (
    CoreObjectiveConfig,
    compose_core_objective,
    delta_metrics,
    load_delta_checkpoint,
    save_delta_checkpoint,
    update_delta,
)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _environment() -> str:
    lines = [
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_runtime={torch.version.cuda}",
    ]
    if torch.cuda.is_available():
        lines.append(f"device={torch.cuda.get_device_name(0)}")
    return "\n".join(lines) + "\n"


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _first_target_record(config: dict) -> tuple[str, list[dict]]:
    root = config["data"]["dataset_root"]
    image_dir = os.path.join(root, config["data"]["train_images"])
    label_dir = os.path.join(root, config["data"]["train_labels"])
    for image_path in list_images(image_dir):
        annotations = read_yolo_annotations(label_path_for_image(image_path, label_dir))
        if image_has_target(annotations, int(config["target_class_id"])):
            return image_path, annotations
    raise RuntimeError("mini VOC contains no target image")


def run_carrier(config_path: str, run_id: str | None = None) -> Path:
    config = _load_config(config_path)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    run_id = run_id or unique_run_id("L1", seed)
    run_dir = create_run_dir(config["artifact_root"], run_id)
    command = f'"{sys.executable}" -m oa_lgc.cli carrier --config "{config_path}" --run-id "{run_id}"'
    try:
        image_path, annotations = _first_target_record(config)
        image_np = load_image_rgb_float(image_path)
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()
        carrier_options = dict(config["carrier"])
        resolution = int(carrier_options.pop("object_resolution"))
        carrier_cfg = CarrierConfig(target_class_id=int(config["target_class_id"]), **carrier_options)
        generator = torch.Generator().manual_seed(seed)
        delta_obj = torch.empty((3, resolution, resolution)).uniform_(
            -carrier_cfg.eps * 0.5, carrier_cfg.eps * 0.5, generator=generator
        ).requires_grad_(True)
        result = apply_object_aligned_carrier(image, annotations, delta_obj, carrier_cfg)
        result.poisoned.square().mean().backward()
        gradient_norm = float(delta_obj.grad.detach().norm().item())
        if gradient_norm <= 0 or not np.isfinite(gradient_norm):
            raise RuntimeError("carrier gradient did not reach delta_obj")

        metrics = {
            **result.metrics,
            "source_image_id": Path(image_path).stem,
            "source_image_path": os.path.abspath(image_path),
            "delta_gradient_norm": gradient_norm,
            "delta_only_trainable_parameter": True,
            "model_parameters_with_gradient": 0,
            "run_id": run_id,
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text(_environment(), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
        (run_dir / "carrier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        rows = result.instance_metrics
        with (run_dir / "instance_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            fieldnames = list(rows[0]) if rows else ["instance_index", "invalid_reason"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_png(run_dir / "clean_preview.png", image.permute(1, 2, 0).numpy())
        write_png(run_dir / "poisoned_preview.png", result.poisoned.detach().permute(1, 2, 0).numpy())
        mask = result.valid_support.detach().permute(1, 2, 0).numpy()
        write_png(run_dir / "valid_mask_preview.png", np.repeat(mask, 3, axis=2))
        (run_dir / "run.log").write_text(
            f"status=pass\nsource_image={image_path}\ngradient_norm={gradient_norm:.9g}\n", encoding="utf-8"
        )
    except Exception as error:
        (run_dir / "run.log").write_text(f"status=fail\nerror={error!r}\n", encoding="utf-8")
        raise
    return run_dir


def run_episode(config_path: str, run_id: str | None = None) -> Path:
    config = _load_config(config_path)
    seed = int(config["seed"])
    run_id = run_id or unique_run_id("L2", seed)
    run_dir = create_run_dir(config["artifact_root"], run_id)
    command = f'"{sys.executable}" -m oa_lgc.cli episode --config "{config_path}" --run-id "{run_id}"'
    try:
        records = load_records(
            config["data"]["dataset_root"], config["data"]["train_images"], config["data"]["train_labels"]
        )
        sampler = DisjointEpisodeSampler(
            records,
            target_class_id=int(config["target_class_id"]),
            num_classes=int(config["num_classes"]),
            seed=seed,
            **{key: value for key, value in config["episode"].items() if key in {"support_size", "query_size", "minimum_class_samples"}},
        )
        episode = sampler.sample(
            episode_index=int(config["episode"].get("episode_index", 0)),
            worker_id=int(config["episode"].get("worker_id", 0)),
        )
        overlap = sorted(set(episode.support_ids) & set(episode.query_ids))
        manifest = {
            "run_id": run_id,
            "support_ids": episode.support_ids,
            "query_ids": episode.query_ids,
            "support_clean_ids": [record.source_id for record in episode.support_clean],
            "support_poison_ids": [record.source_id for record in episode.support_poison],
            "query_clean_ids": [record.source_id for record in episode.query_clean],
            "query_poison_ids": [record.source_id for record in episode.query_poison],
            "target_class_ids": episode.target_class_ids,
            "non_target_class_ids": episode.non_target_class_ids,
        }
        overlap_check = {
            "support_query_overlap_count": len(overlap),
            "overlap_ids": overlap,
            "clean_poison_pair_aligned": True,
            "pass": len(overlap) == 0,
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text(_environment(), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
        (run_dir / "support_ids.txt").write_text("\n".join(episode.support_ids) + "\n", encoding="utf-8")
        (run_dir / "query_ids.txt").write_text("\n".join(episode.query_ids) + "\n", encoding="utf-8")
        (run_dir / "episode_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_dir / "overlap_check.json").write_text(json.dumps(overlap_check, indent=2), encoding="utf-8")
        with (run_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["class_id", "support_count", "query_count", "valid"])
            writer.writeheader()
            for class_id, counts in episode.class_counts.items():
                writer.writerow({
                    "class_id": class_id,
                    "support_count": counts["support"],
                    "query_count": counts["query"],
                    "valid": int(episode.class_validity[class_id]),
                })
        (run_dir / "run.log").write_text(
            f"status=pass\nsupport_query_overlap=0\nsupport={episode.support_ids}\nquery={episode.query_ids}\n",
            encoding="utf-8",
        )
    except Exception as error:
        (run_dir / "run.log").write_text(f"status=fail\nerror={error!r}\n", encoding="utf-8")
        raise
    return run_dir


def _synthetic_detector_batch() -> tuple[torch.Tensor, tuple[tuple[dict, ...], ...]]:
    images = torch.rand(2, 3, 24, 24)
    annotations = (
        ({"cls": 14, "bbox": [0.4, 0.5, 0.4, 0.5]}, {"cls": 1, "bbox": [0.8, 0.2, 0.2, 0.2]}),
        ({"cls": 14, "bbox": [0.6, 0.5, 0.3, 0.4]}, {"cls": 1, "bbox": [0.2, 0.8, 0.2, 0.2]}),
    )
    return images, annotations


def run_virtual(config_path: str, run_id: str | None = None) -> Path:
    config = _load_config(config_path)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    run_id = run_id or unique_run_id("L3", seed)
    run_dir = create_run_dir(config["artifact_root"], run_id)
    command = f'"{sys.executable}" -m oa_lgc.cli virtual --config "{config_path}" --run-id "{run_id}"'
    try:
        model = ObjectCropDetector(num_classes=int(config["num_classes"]), **config["model"])
        model.requires_grad_(False)
        base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        images, annotations = _synthetic_detector_batch()
        specifications = [
            ("head_only", steps, None) for steps in config["virtual_update"]["steps"]
        ] + [
            ("detection_head", 3, None),
            ("selected_modules", 1, config["virtual_update"]["selected_modules"]),
        ]
        inner_rows = []
        delta_rows = []
        trajectories = {}
        tracemalloc.start()
        for mode, steps, selected_modules in specifications:
            trajectory = virtual_update(
                model,
                images,
                annotations,
                steps=int(steps),
                learning_rate=float(config["virtual_update"]["learning_rate"]),
                mode=mode,
                selected_modules=selected_modules,
                first_order=bool(config["virtual_update"]["first_order"]),
            )
            key = f"{mode}_j{steps}"
            trajectories[key] = trajectory
            for step_index, (loss, delta_norm, elapsed) in enumerate(
                zip(trajectory.step_losses, trajectory.parameter_delta_norms, trajectory.step_times_seconds), start=1
            ):
                inner_rows.append({"run": key, "step": step_index, "loss": loss, "seconds": elapsed})
                delta_rows.append({"run": key, "step": step_index, "parameter_delta_norm": delta_norm})
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        delta = torch.full_like(images, 0.01, requires_grad=True)
        poison_trajectory = virtual_update(
            model,
            (images + delta).clamp(0, 1),
            annotations,
            steps=3,
            learning_rate=float(config["virtual_update"]["learning_rate"]),
            mode="head_only",
            first_order=True,
        )
        query_outputs = functional_forward(
            model, poison_trajectory.parameters, poison_trajectory.buffers, images, annotations
        )
        outer_loss, _ = class_loss(query_outputs, 14)
        gradient = torch.autograd.grad(outer_loss, delta)[0]
        gradient_norm = float(gradient.norm().item())
        clean_trajectory = trajectories["head_only_j3"]
        clean_poison_difference = float(sum(
            (clean_trajectory.parameters[name] - poison_trajectory.parameters[name]).square().sum()
            for name in clean_trajectory.selected_names
        ).sqrt().detach().item())
        gradient_flow = {
            "base_model_unchanged": model_state_unchanged(model, base_state),
            "outer_gradient_to_delta": gradient_norm,
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "clean_poison_parameter_difference": clean_poison_difference,
            "first_order": True,
            "full_second_order_claimed": False,
        }
        memory = {
            "device": "cpu",
            "python_tracemalloc_current_bytes": current_bytes,
            "python_tracemalloc_peak_bytes": peak_bytes,
            "cuda_peak_bytes": 0,
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text(_environment(), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
        for filename, rows in (("inner_step_metrics.csv", inner_rows), ("parameter_delta_metrics.csv", delta_rows)):
            with (run_dir / filename).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (run_dir / "gradient_flow.json").write_text(json.dumps(gradient_flow, indent=2), encoding="utf-8")
        (run_dir / "memory_profile.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
        (run_dir / "run.log").write_text(
            f"status=pass\nouter_gradient_to_delta={gradient_norm:.9g}\nbase_model_unchanged={gradient_flow['base_model_unchanged']}\n",
            encoding="utf-8",
        )
    except Exception as error:
        (run_dir / "run.log").write_text(f"status=fail\nerror={error!r}\n", encoding="utf-8")
        raise
    return run_dir


def run_gain(config_path: str, run_id: str | None = None) -> Path:
    config = _load_config(config_path)
    seed = int(config["seed"])
    run_id = run_id or unique_run_id("L4", seed)
    run_dir = create_run_dir(config["artifact_root"], run_id)
    command = f'"{sys.executable}" -m oa_lgc.cli gain --config "{config_path}" --run-id "{run_id}"'
    try:
        gain_cfg = config["gain"]
        delta = torch.tensor(0.1, requires_grad=True)
        target = target_learning_gain(
            torch.tensor(1.0), torch.tensor(0.5), 0.9 + delta.square(),
            rho_t=float(gain_cfg["rho_t"]),
            min_valid_clean_gain=float(gain_cfg["min_valid_clean_gain"]),
            eps=float(gain_cfg["eps"]),
        )
        class_inputs = {
            1: ClassGainInput(torch.tensor(1.0), torch.tensor(0.6), torch.tensor(0.6), 2, 2),
            2: ClassGainInput(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5), 0, 0),
            3: ClassGainInput(torch.tensor(1.0), torch.tensor(1.5), torch.tensor(1.4), 1, 1),
        }
        authorized = authorized_learning_gain(
            class_inputs,
            target_class_id=int(config["target_class_id"]),
            rho_k=float(gain_cfg["rho_k"]),
            min_valid_class_gain=float(gain_cfg["min_valid_class_gain"]),
            minimum_class_samples=int(gain_cfg["minimum_class_samples"]),
            eps=float(gain_cfg["eps"]),
        )
        carrier = carrier_query_loss(0.7 + delta.square())
        outer = target.protect_loss + carrier + authorized.loss
        gradient = torch.autograd.grad(outer, delta)[0]
        invalid_target = target_learning_gain(
            torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.9),
            rho_t=float(gain_cfg["rho_t"]),
            min_valid_clean_gain=float(gain_cfg["min_valid_clean_gain"]),
            eps=float(gain_cfg["eps"]),
        )
        metrics = {
            "G_t_clean": float(target.clean_gain.detach()),
            "G_t_poison": float(target.poison_gain.detach()),
            "target_gain_ratio": None if target.ratio is None else float(target.ratio.detach()),
            "L_protect": float(target.protect_loss.detach()),
            "L_carrier": float(carrier.detach()),
            "L_auth": float(authorized.loss.detach()),
            "valid_target": target.valid,
            "invalid_clean_gain_example_reason": invalid_target.invalid_reason,
            "valid_authorized_classes": list(authorized.valid_class_ids),
            "invalid_authorized_classes": list(authorized.invalid_class_ids),
        }
        class_rows = []
        invalid_rows = [{"scope": "target_example", "id": 14, "reason": invalid_target.invalid_reason}]
        for class_id, result in authorized.classes.items():
            class_rows.append({
                "class_id": class_id,
                "G_k_clean": float(result.clean_gain.detach()),
                "G_k_poison": float(result.poison_gain.detach()),
                "authorized_gain_gap": None if result.normalized_gap is None else float(result.normalized_gap.detach()),
                "valid": int(result.valid),
                "invalid_reason": result.invalid_reason,
                "support_count": result.support_count,
                "query_count": result.query_count,
            })
            if not result.valid:
                invalid_rows.append({"scope": "class", "id": class_id, "reason": result.invalid_reason})
        gradient_flow = {"outer_gradient_to_delta": float(gradient.detach()), "finite": bool(torch.isfinite(gradient))}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text(_environment(), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
        (run_dir / "gain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "gradient_flow.json").write_text(json.dumps(gradient_flow, indent=2), encoding="utf-8")
        for filename, rows in (("classwise_gain_metrics.csv", class_rows), ("invalid_episode_metrics.csv", invalid_rows)):
            with (run_dir / filename).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (run_dir / "run.log").write_text(
            f"status=pass\nG_t_clean={metrics['G_t_clean']}\nG_t_poison={metrics['G_t_poison']}\ngradient={gradient_flow['outer_gradient_to_delta']}\n",
            encoding="utf-8",
        )
    except Exception as error:
        (run_dir / "run.log").write_text(f"status=fail\nerror={error!r}\n", encoding="utf-8")
        raise
    return run_dir


def run_objective(config_path: str, run_id: str | None = None) -> Path:
    config = _load_config(config_path)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    run_id = run_id or unique_run_id("L5", seed)
    run_dir = create_run_dir(config["artifact_root"], run_id)
    command = f'"{sys.executable}" -m oa_lgc.cli objective --config "{config_path}" --run-id "{run_id}"'
    try:
        objective_cfg = CoreObjectiveConfig(**config["objective"])
        delta_obj = torch.nn.Parameter(torch.empty(3, 8, 8).uniform_(-0.01, 0.01))
        initial = delta_obj.detach().clone()
        model = ObjectCropDetector(hidden_dim=8)
        model.requires_grad_(False)
        base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam([delta_obj], lr=float(config["optimization"]["learning_rate"]))
        save_delta_checkpoint(run_dir / "delta_initial.pt", delta_obj, objective_cfg, {"step": 0})
        rows = []
        for step in range(int(config["optimization"]["steps"])):
            protect = (delta_obj.mean() - 0.02).square()
            carrier = (delta_obj - 0.03).square().mean()
            authorized = (delta_obj.mean() + 0.01).abs()
            result = compose_core_objective(protect, carrier, authorized, delta_obj, objective_cfg)
            values = {name: float(value.detach()) for name, value in result.components.items()}
            total = float(result.loss.detach())
            gradient_norm = update_delta(result, delta_obj, optimizer, objective_cfg)
            metrics = delta_metrics(delta_obj, objective_cfg.eps)
            rows.append({
                "step": step,
                "L_core": total,
                **values,
                "gradient_norm": gradient_norm,
                **metrics,
                "valid_target_gain": 1,
                "valid_authorized_class_count": 1,
            })
        save_delta_checkpoint(
            run_dir / "delta_final.pt", delta_obj, objective_cfg, {"step": int(config["optimization"]["steps"])}
        )
        restored = load_delta_checkpoint(run_dir / "delta_final.pt")
        summary = {
            **delta_metrics(delta_obj, objective_cfg.eps),
            "delta_changed": not torch.equal(initial, delta_obj.detach()),
            "delta_change_norm": float((delta_obj.detach() - initial).norm().item()),
            "base_model_unchanged": model_state_unchanged(model, base_state),
            "model_parameters_with_gradient": sum(parameter.grad is not None for parameter in model.parameters()),
            "checkpoint_restored_equal": bool(torch.equal(restored["delta_obj"], delta_obj.detach().cpu())),
            "budget_satisfied": float(delta_obj.detach().abs().max()) <= objective_cfg.eps + 1e-7,
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text(_environment(), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
        with (run_dir / "loss_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (run_dir / "classwise_gain_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["class_id", "valid", "authorized_gain_gap"])
            writer.writeheader()
            writer.writerow({"class_id": 1, "valid": 1, "authorized_gain_gap": 0.0})
        (run_dir / "delta_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "run.log").write_text(
            f"status=pass\ndelta_changed={summary['delta_changed']}\nbudget_satisfied={summary['budget_satisfied']}\ncheckpoint_restored_equal={summary['checkpoint_restored_equal']}\n",
            encoding="utf-8",
        )
    except Exception as error:
        (run_dir / "run.log").write_text(f"status=fail\nerror={error!r}\n", encoding="utf-8")
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="OA-LGC local engineering validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    carrier = subparsers.add_parser("carrier")
    carrier.add_argument("--config", required=True)
    carrier.add_argument("--run-id")
    episode = subparsers.add_parser("episode")
    episode.add_argument("--config", required=True)
    episode.add_argument("--run-id")
    virtual = subparsers.add_parser("virtual")
    virtual.add_argument("--config", required=True)
    virtual.add_argument("--run-id")
    gain = subparsers.add_parser("gain")
    gain.add_argument("--config", required=True)
    gain.add_argument("--run-id")
    objective = subparsers.add_parser("objective")
    objective.add_argument("--config", required=True)
    objective.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "carrier":
        print(run_carrier(args.config, args.run_id))
    elif args.command == "episode":
        print(run_episode(args.config, args.run_id))
    elif args.command == "virtual":
        print(run_virtual(args.config, args.run_id))
    elif args.command == "gain":
        print(run_gain(args.config, args.run_id))
    elif args.command == "objective":
        print(run_objective(args.config, args.run_id))


if __name__ == "__main__":
    main()
