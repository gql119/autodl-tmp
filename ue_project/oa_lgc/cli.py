from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(description="OA-LGC local engineering validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    carrier = subparsers.add_parser("carrier")
    carrier.add_argument("--config", required=True)
    carrier.add_argument("--run-id")
    episode = subparsers.add_parser("episode")
    episode.add_argument("--config", required=True)
    episode.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "carrier":
        print(run_carrier(args.config, args.run_id))
    elif args.command == "episode":
        print(run_episode(args.config, args.run_id))


if __name__ == "__main__":
    main()
