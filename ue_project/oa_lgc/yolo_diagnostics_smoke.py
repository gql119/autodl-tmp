from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import yaml

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier
from oa_lgc.yolo_adapter import YOLOFunctionalAdapter
from oa_lgc.yolo_diagnostics import (
    CLASSWISE_DIAGNOSTIC_FIELDS,
    TARGET_DIAGNOSTIC_FIELDS,
    build_episode_diagnostics,
)
from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations


VOC_NAMES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow", 10: "diningtable",
    11: "dog", 12: "horse", 13: "motorbike", 14: "person", 15: "pottedplant",
    16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_sample(dataset_root: Path, config: dict, source_id: str, device: torch.device):
    image_path = dataset_root / config["data"]["train_images"] / f"{source_id}.jpg"
    label_path = dataset_root / config["data"]["train_labels"] / f"{source_id}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"episode source is unavailable: {source_id}")
    annotations = read_yolo_annotations(str(label_path))
    image, adjusted = _letterbox_with_annotations(
        load_image_rgb_float(str(image_path)), annotations, int(config["surrogate"]["imgsz"])
    )
    return image.to(device), adjusted


def run(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    adapter = YOLOFunctionalAdapter.from_checkpoint(
        config["surrogate"]["checkpoint"],
        device=device,
        num_classes=int(config["surrogate"]["num_classes"]),
        target_class_id=int(config["experiment"]["target_class_id"]),
    )
    dataset_root = Path(config["data"]["dataset_root"])
    carrier_config = CarrierConfig(
        target_class_id=int(config["experiment"]["target_class_id"]),
        eps=float(config["carrier"]["eps"]),
        non_target_dilation=int(config["carrier"]["non_target_dilation"]),
        soft_mask=bool(config["carrier"]["soft_mask"]),
        soft_edge_pixels=float(config["carrier"]["soft_edge_pixels"]),
    )
    episode_rows = []
    class_rows = []
    invalid_rows = []
    valid_non_target_ids = set()
    base_hash = adapter.hash_base_state()
    for episode_index, episode in enumerate(config["experiment"]["episodes"]):
        support_id = str(episode["support_id"])
        query_id = str(episode["query_id"])
        if support_id == query_id:
            raise RuntimeError(f"support/query overlap in episode {episode_index}")
        support_image, support_annotations = _load_sample(dataset_root, config, support_id, device)
        query_image, query_annotations = _load_sample(dataset_root, config, query_id, device)
        clean_batch = _batch_from_annotations(support_annotations, support_image, device)
        generator = torch.Generator(device=device).manual_seed(seed + episode_index)
        delta = torch.nn.Parameter(
            torch.randn(
                (3, int(config["carrier"]["object_resolution"]), int(config["carrier"]["object_resolution"])),
                generator=generator,
                device=device,
            )
            * 1e-3
        )
        carrier = apply_object_aligned_carrier(
            support_image[0], support_annotations, delta, carrier_config
        )
        poison_batch = _batch_from_annotations(
            support_annotations, carrier.poisoned.unsqueeze(0), device
        )
        clean = adapter.virtual_update(
            clean_batch,
            int(config["virtual_update"]["steps"]),
            float(config["virtual_update"]["learning_rate"]),
            config["virtual_update"]["mode"],
            create_graph=False,
        )
        poison = adapter.virtual_update(
            poison_batch,
            int(config["virtual_update"]["steps"]),
            float(config["virtual_update"]["learning_rate"]),
            config["virtual_update"]["mode"],
            create_graph=False,
        )
        query_batch = _batch_from_annotations(query_annotations, query_image, device)
        diagnostics = build_episode_diagnostics(
            adapter,
            query_image,
            query_batch,
            clean,
            poison,
            class_names=VOC_NAMES,
            minimum_target_coverage=float(config["experiment"]["minimum_target_coverage"]),
        )
        target_row = {
            "episode": episode_index,
            "support_id": support_id,
            "query_id": query_id,
            **diagnostics.target_dict(),
        }
        episode_rows.append(target_row)
        if not diagnostics.target.valid:
            invalid_rows.append(
                {
                    "episode": episode_index,
                    "support_id": support_id,
                    "query_id": query_id,
                    "reason": diagnostics.target.target_valid_reason,
                }
            )
        for row in diagnostics.class_rows():
            full_row = {
                "episode": episode_index,
                "support_id": support_id,
                "query_id": query_id,
                **row,
            }
            class_rows.append(full_row)
            if row["class_id"] != int(config["experiment"]["target_class_id"]) and row["valid"]:
                valid_non_target_ids.add(int(row["class_id"]))

    _write_csv(
        output / "episode_diagnostics.csv",
        ["episode", "support_id", "query_id", *TARGET_DIAGNOSTIC_FIELDS],
        episode_rows,
    )
    _write_csv(
        output / "classwise_diagnostics.csv",
        ["episode", "support_id", "query_id", *CLASSWISE_DIAGNOSTIC_FIELDS],
        class_rows,
    )
    coverage_values = [float(row["target_positive_coverage"]) for row in episode_rows]
    overlap_values = [float(row["target_assignment_overlap"]) for row in episode_rows]
    low_coverage = [value < float(config["experiment"]["minimum_target_coverage"]) for value in coverage_values]
    tal_summary = {
        "episode_count": len(episode_rows),
        "target_coverage_definition": "poison target positive units / max(reference target positive units, 1)",
        "target_coverage_median": statistics.median(coverage_values),
        "target_coverage_min": min(coverage_values),
        "target_assignment_overlap_definition": "Jaccard(reference target units, poison target units)",
        "target_assignment_overlap_median": statistics.median(overlap_values),
        "low_coverage_episode_count": sum(low_coverage),
        "low_coverage_episode_ratio": sum(low_coverage) / len(low_coverage),
        "real_tal": True,
        "proxy_fallback": False,
    }
    box_values = [float(row["target_box_loss"]) for row in episode_rows]
    dfl_values = [float(row["target_dfl_loss"]) for row in episode_rows]
    box_dfl_summary = {
        "target_box_loss_available": all(torch.isfinite(torch.tensor(box_values))) and all(value > 0 for value in box_values),
        "target_dfl_loss_available": all(torch.isfinite(torch.tensor(dfl_values))) and all(value > 0 for value in dfl_values),
        "target_box_loss_median": statistics.median(box_values),
        "target_dfl_loss_median": statistics.median(dfl_values),
        "valid_non_target_class_ids": sorted(valid_non_target_ids),
        "valid_non_target_class_count": len(valid_non_target_ids),
        "fixed_reference_box_dfl": True,
    }
    invalid_summary = {
        "invalid_episode_count": len(invalid_rows),
        "invalid_episode_ratio": len(invalid_rows) / len(episode_rows),
        "episodes": invalid_rows,
    }
    (output / "tal_summary.json").write_text(json.dumps(tal_summary, indent=2) + "\n", encoding="utf-8")
    (output / "box_dfl_summary.json").write_text(
        json.dumps(box_dfl_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "invalid_episode_summary.json").write_text(
        json.dumps(invalid_summary, indent=2) + "\n", encoding="utf-8"
    )
    gate = {
        "target_coverage_median_ge_0_50": tal_summary["target_coverage_median"] >= 0.5,
        "low_coverage_episode_ratio_le_0_50": tal_summary["low_coverage_episode_ratio"] <= 0.5,
        "target_box_loss_available": box_dfl_summary["target_box_loss_available"],
        "target_dfl_loss_available": box_dfl_summary["target_dfl_loss_available"],
        "non_target_valid_class_available": box_dfl_summary["valid_non_target_class_count"] >= 1,
        "diagnostic_schema_complete": all(set(row) == {"episode", "support_id", "query_id", *TARGET_DIAGNOSTIC_FIELDS} for row in episode_rows)
        and len(class_rows) == 20 * len(episode_rows),
        "base_hash_unchanged": adapter.hash_base_state() == base_hash,
    }
    status = "pass" if all(gate.values()) else "fail"
    log = {
        "status": status,
        "gate": gate,
        "tal_summary": tal_summary,
        "box_dfl_summary": box_dfl_summary,
        "runtime_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
    }
    (output / "run.log").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"C2 gate failed: {gate}")
    return log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OA-LGC C2 real TAL/box/DFL diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.config, arguments.output), indent=2))
