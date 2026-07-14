from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
import ultralytics
import yaml

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier
from oa_lgc.gains import ClassGainInput, authorized_learning_gain, carrier_query_loss, target_learning_gain
from oa_lgc.objective import CoreObjectiveConfig, compose_core_objective, delta_metrics, update_delta
from oa_lgc.yolo_adapter import YOLOFunctionalAdapter
from oa_lgc.yolo_diagnostics import build_episode_diagnostics
from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations


VOC_NAMES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow", 10: "diningtable",
    11: "dog", 12: "horse", 13: "motorbike", 14: "person", 15: "pottedplant",
    16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()


def _load_sample(root: Path, config: dict, source_id: str, device: torch.device):
    image_path = root / config["data"]["train_images"] / f"{source_id}.jpg"
    label_path = root / config["data"]["train_labels"] / f"{source_id}.txt"
    annotations = read_yolo_annotations(str(label_path))
    image, adjusted = _letterbox_with_annotations(
        load_image_rgb_float(str(image_path)), annotations, int(config["surrogate"]["imgsz"])
    )
    return image.to(device), adjusted


def _class_count(annotations: list[dict], class_id: int) -> int:
    return sum(int(annotation["cls"]) == int(class_id) for annotation in annotations)


def _gradient_norm(loss: torch.Tensor, delta: torch.Tensor, retain_graph: bool = True) -> float:
    gradient = torch.autograd.grad(loss, delta, retain_graph=retain_graph, allow_unused=True)[0]
    if gradient is None:
        return 0.0
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite smoke gradient")
    return float(gradient.detach().norm())


def _finite_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def _run_variant(
    adapter: YOLOFunctionalAdapter,
    config: dict,
    run_config: dict,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset_root = Path(config["data"]["dataset_root"])
    target_id = int(config["experiment"]["target_class_id"])
    generator = torch.Generator(device=device).manual_seed(seed)
    delta = torch.nn.Parameter(
        torch.empty(
            (3, int(config["carrier"]["object_resolution"]), int(config["carrier"]["object_resolution"])),
            device=device,
        ).uniform_(-0.001, 0.001, generator=generator)
    )
    initial_delta = delta.detach().clone()
    optimizer = torch.optim.Adam([delta], lr=float(config["optimization"]["outer_learning_rate"]))
    carrier_config = CarrierConfig(
        target_class_id=target_id,
        eps=float(config["carrier"]["eps"]),
        non_target_dilation=int(config["carrier"]["non_target_dilation"]),
        min_valid_fraction=float(config["carrier"]["min_valid_fraction"]),
        interpolation=str(config["carrier"]["interpolation"]),
        soft_mask=bool(config["carrier"]["soft_mask"]),
        soft_edge_pixels=float(config["carrier"]["soft_edge_pixels"]),
    )
    objective_config = CoreObjectiveConfig(**config["objective"])
    base_hash_before = adapter.hash_base_state()
    started = time.perf_counter()
    gain_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    tal_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    episode_pairs = config["experiment"]["episode_pairs"][: int(run_config["episodes"])]
    if len(episode_pairs) != int(run_config["outer_steps"]):
        raise RuntimeError("C3 requires one unique episode per outer step")

    for outer_step, (support_id, query_id) in enumerate(episode_pairs):
        support_id, query_id = str(support_id), str(query_id)
        if support_id == query_id:
            raise RuntimeError("support/query source overlap")
        support_image, support_annotations = _load_sample(dataset_root, config, support_id, device)
        query_image, query_annotations = _load_sample(dataset_root, config, query_id, device)
        clean_support_batch = _batch_from_annotations(support_annotations, support_image, device)
        poison_support_carrier = apply_object_aligned_carrier(
            support_image[0], support_annotations, delta, carrier_config
        )
        poison_query_carrier = apply_object_aligned_carrier(
            query_image[0], query_annotations, delta, carrier_config
        )
        poison_support_batch = _batch_from_annotations(
            support_annotations, poison_support_carrier.poisoned.unsqueeze(0), device
        )
        poison_query_batch = _batch_from_annotations(
            query_annotations, poison_query_carrier.poisoned.unsqueeze(0), device
        )
        virtual_steps = int(run_config["virtual_steps"])
        virtual_mode = str(run_config["virtual_mode"])
        clean_trajectory = adapter.virtual_update(
            clean_support_batch,
            virtual_steps,
            float(config["virtual_update"]["learning_rate"]),
            virtual_mode,
            create_graph=True,
        )
        poison_trajectory = adapter.virtual_update(
            poison_support_batch,
            virtual_steps,
            float(config["virtual_update"]["learning_rate"]),
            virtual_mode,
            create_graph=True,
        )
        for branch, trajectory in (("clean", clean_trajectory), ("poison", poison_trajectory)):
            for inner_step, losses in enumerate(trajectory.step_losses, start=1):
                inner_rows.append(
                    {
                        "run": run_config["name"],
                        "outer_step": outer_step,
                        "branch": branch,
                        "inner_step": inner_step,
                        **losses,
                        "parameter_delta_norm": trajectory.parameter_delta_norms[inner_step - 1],
                        "runtime_seconds": trajectory.step_times_seconds[inner_step - 1],
                    }
                )
        query_batch = _batch_from_annotations(query_annotations, query_image, device)
        reference = adapter.reference_assignment(query_image, query_batch)
        initial_query = adapter.compute_classwise_query_loss(
            query_image,
            query_batch,
            adapter.base_parameters(),
            adapter.clone_buffers(),
            reference,
        )
        clean_query = adapter.compute_classwise_query_loss(
            query_image,
            query_batch,
            clean_trajectory.parameters,
            clean_trajectory.buffers,
            reference,
        )
        poison_query = adapter.compute_classwise_query_loss(
            query_image,
            query_batch,
            poison_trajectory.parameters,
            poison_trajectory.buffers,
            reference,
        )
        if not all(result.valid[target_id] for result in (initial_query, clean_query, poison_query)):
            raise RuntimeError("target fixed-reference query loss is unavailable")
        target_gain = target_learning_gain(
            initial_query.losses[target_id],
            clean_query.losses[target_id],
            poison_query.losses[target_id],
            rho_t=float(config["gain"]["rho_t"]),
            min_valid_clean_gain=float(config["gain"]["min_valid_clean_gain"]),
            eps=float(config["gain"]["eps"]),
        )
        if not target_gain.valid or target_gain.ratio is None:
            raise RuntimeError(f"target gain is invalid: {target_gain.invalid_reason}")
        class_inputs = {}
        for class_id in range(int(config["surrogate"]["num_classes"])):
            if class_id == target_id:
                continue
            if all(result.valid[class_id] for result in (initial_query, clean_query, poison_query)):
                class_inputs[class_id] = ClassGainInput(
                    initial_query.losses[class_id],
                    clean_query.losses[class_id],
                    poison_query.losses[class_id],
                    _class_count(support_annotations, class_id),
                    _class_count(query_annotations, class_id),
                )
        authorized = authorized_learning_gain(
            class_inputs,
            target_class_id=target_id,
            rho_k=float(config["gain"]["rho_k"]),
            min_valid_class_gain=float(config["gain"]["min_valid_class_gain"]),
            minimum_class_samples=int(config["gain"]["minimum_class_samples"]),
            eps=float(config["gain"]["eps"]),
        )
        authorized_loss = authorized.loss.to(device)
        if not authorized_loss.requires_grad:
            authorized_loss = authorized_loss + delta.sum() * 0.0
        carrier_query = adapter.compute_classwise_query_loss(
            poison_query_carrier.poisoned.unsqueeze(0),
            poison_query_batch,
            poison_trajectory.parameters,
            poison_trajectory.buffers,
            reference,
        )
        carrier_loss = carrier_query_loss(carrier_query.losses[target_id])
        objective = compose_core_objective(
            target_gain.protect_loss, carrier_loss, authorized_loss, delta, objective_config
        )
        protect_gradient = _gradient_norm(target_gain.protect_loss, delta, retain_graph=True)
        total_gradient = _gradient_norm(objective.loss, delta, retain_graph=True)

        diagnostics = build_episode_diagnostics(
            adapter,
            query_image,
            query_batch,
            clean_trajectory,
            poison_trajectory,
            class_names=VOC_NAMES,
            minimum_target_coverage=float(config["experiment"]["minimum_target_coverage"]),
        )
        valid_non_target = [
            row for class_id, row in diagnostics.classes.items() if class_id != target_id and row.valid
        ]
        non_target_assignment_drift = statistics.mean(
            row.assignment_drift for row in valid_non_target
        ) if valid_non_target else 0.0
        non_target_box_drift = statistics.mean(
            float(row.box_drift) for row in valid_non_target if row.box_drift is not None
        ) if valid_non_target else 0.0
        non_target_dfl_drift = statistics.mean(
            float(row.dfl_drift) for row in valid_non_target if row.dfl_drift is not None
        ) if valid_non_target else 0.0
        loss_before = float(objective.loss.detach())
        total_gradient_backward = update_delta(objective, delta, optimizer, objective_config)
        current_delta = delta_metrics(delta, objective_config.eps)
        gain_rows.append(
            {
                "run": run_config["name"],
                "outer_step": outer_step,
                "virtual_steps": virtual_steps,
                "virtual_mode": virtual_mode,
                "target_clean_gain": float(target_gain.clean_gain.detach()),
                "target_poison_gain": float(target_gain.poison_gain.detach()),
                "target_gain_gap": float((target_gain.clean_gain - target_gain.poison_gain).detach()),
                "target_gain_ratio": float(target_gain.ratio.detach()),
                "carrier_query_loss": float(carrier_loss.detach()),
                "classwise_authorized_gain": float(authorized_loss.detach()),
                "valid_authorized_classes": len(authorized.valid_class_ids),
                "invalid_authorized_classes": len(authorized.invalid_class_ids),
                "L_core": loss_before,
                "mean_abs_delta": current_delta["mean_abs_delta"],
                "max_abs_delta": current_delta["max_abs_delta"],
                "saturation_ratio": current_delta["saturation_ratio"],
                "perturbed_area": statistics.mean(
                    [poison_support_carrier.metrics["perturbed_area"], poison_query_carrier.metrics["perturbed_area"]]
                ),
                "non_target_overlap": statistics.mean(
                    [
                        poison_support_carrier.metrics["non_target_overlap_ratio"],
                        poison_query_carrier.metrics["non_target_overlap_ratio"],
                    ]
                ),
            }
        )
        for class_id, result in authorized.classes.items():
            class_rows.append(
                {
                    "run": run_config["name"],
                    "outer_step": outer_step,
                    "class_id": class_id,
                    "class_name": VOC_NAMES[class_id],
                    "clean_gain": float(result.clean_gain.detach()),
                    "poison_gain": float(result.poison_gain.detach()),
                    "authorized_gain_gap": "" if result.normalized_gap is None else float(result.normalized_gap.detach()),
                    "valid": int(result.valid),
                    "invalid_reason": result.invalid_reason,
                    "support_count": result.support_count,
                    "query_count": result.query_count,
                }
            )
        tal_rows.append(
            {
                "run": run_config["name"],
                "outer_step": outer_step,
                "target_positive_coverage": diagnostics.target.target_positive_coverage,
                "target_assignment_overlap": diagnostics.target.target_assignment_overlap,
                "target_reference_positive_count": diagnostics.target.target_reference_positive_count,
                "target_poison_positive_count": diagnostics.target.target_poison_positive_count,
                "target_localization_recall": diagnostics.target.target_localization_recall,
                "target_valid": int(diagnostics.target.valid),
                "target_valid_reason": diagnostics.target.target_valid_reason,
                "non_target_assignment_drift": non_target_assignment_drift,
            }
        )
        box_rows.append(
            {
                "run": run_config["name"],
                "outer_step": outer_step,
                "target_box_loss": diagnostics.target.target_box_loss,
                "target_dfl_loss": diagnostics.target.target_dfl_loss,
                "non_target_box_drift": non_target_box_drift,
                "non_target_dfl_drift": non_target_dfl_drift,
            }
        )
        gradient_rows.append(
            {
                "run": run_config["name"],
                "outer_step": outer_step,
                "protect_only_grad_norm": protect_gradient,
                "total_grad_norm": total_gradient,
                "backward_total_grad_norm": total_gradient_backward,
                "finite": 1,
            }
        )
        manifests.append(
            {
                "run": run_config["name"],
                "outer_step": outer_step,
                "support_ids": [support_id],
                "query_ids": [query_id],
                "overlap_count": 0,
                "reference_target_positive_count": diagnostics.target.target_reference_positive_count,
                "valid_authorized_class_ids": list(authorized.valid_class_ids),
            }
        )

    runtime = time.perf_counter() - started
    base_hash_after = adapter.hash_base_state()
    summary = {
        "run": run_config["name"],
        "virtual_steps": int(run_config["virtual_steps"]),
        "virtual_mode": str(run_config["virtual_mode"]),
        "outer_steps": int(run_config["outer_steps"]),
        "episodes": int(run_config["episodes"]),
        "status": "pass",
        "delta_updated": not torch.equal(initial_delta, delta.detach()),
        "delta_change_norm": float((delta.detach() - initial_delta).norm()),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_unchanged": base_hash_before == base_hash_after,
        "support_query_overlap_max": max(row["overlap_count"] for row in manifests),
        "target_coverage_median": statistics.median(row["target_positive_coverage"] for row in tal_rows),
        "target_gain_computable": all(math.isfinite(row["target_gain_ratio"]) for row in gain_rows),
        "valid_authorized_class_total": sum(row["valid_authorized_classes"] for row in gain_rows),
        "box_dfl_available": all(row["target_box_loss"] > 0 and row["target_dfl_loss"] > 0 for row in box_rows),
        "protect_only_nonzero_count": sum(row["protect_only_grad_norm"] > 0 for row in gradient_rows),
        "finite": _finite_rows(gain_rows + tal_rows + box_rows + gradient_rows),
        "runtime_seconds": runtime,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
    }
    if not all(
        [
            summary["delta_updated"],
            summary["base_unchanged"],
            summary["support_query_overlap_max"] == 0,
            summary["target_coverage_median"] >= 0.5,
            summary["target_gain_computable"],
            summary["valid_authorized_class_total"] >= 1,
            summary["box_dfl_available"],
            summary["protect_only_nonzero_count"] >= 1,
            summary["finite"],
        ]
    ):
        summary["status"] = "fail"
    return {
        "initial_delta": initial_delta.cpu(),
        "final_delta": delta.detach().cpu(),
        "gain_rows": gain_rows,
        "class_rows": class_rows,
        "tal_rows": tal_rows,
        "box_rows": box_rows,
        "gradient_rows": gradient_rows,
        "inner_rows": inner_rows,
        "manifests": manifests,
        "summary": summary,
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "git_commit.txt").write_text(_git_head() + "\n", encoding="utf-8")
    environment = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8"
    )
    torch.use_deterministic_algorithms(bool(config.get("deterministic", True)), warn_only=True)
    device = torch.device(config["device"])
    adapter = YOLOFunctionalAdapter.from_checkpoint(
        config["surrogate"]["checkpoint"],
        device=device,
        num_classes=int(config["surrogate"]["num_classes"]),
        target_class_id=int(config["experiment"]["target_class_id"]),
    )
    results = []
    for run_config in config["runs"]:
        result = _run_variant(adapter, config, run_config, device)
        results.append(result)
        if result["summary"]["status"] != "pass":
            raise RuntimeError(f"C3 run failed: {result['summary']}")

    repeated_name = str(config["reproducibility"]["repeat_run"])
    repeated_config = next(run_config for run_config in config["runs"] if run_config["name"] == repeated_name)
    repeated = _run_variant(adapter, config, repeated_config, device)
    original = next(result for result in results if result["summary"]["run"] == repeated_name)
    same_manifest = original["manifests"] == repeated["manifests"]
    gain_fields = ("target_clean_gain", "target_poison_gain", "target_gain_gap", "target_gain_ratio")
    gain_max_difference = max(
        abs(float(first[field]) - float(second[field]))
        for first, second in zip(original["gain_rows"], repeated["gain_rows"])
        for field in gain_fields
    )
    delta_max_difference = float((original["final_delta"] - repeated["final_delta"]).abs().max())
    reproducibility = {
        "repeat_run": repeated_name,
        "support_query_ids_identical": same_manifest,
        "reference_assignments_identical": all(
            first["reference_target_positive_count"] == second["reference_target_positive_count"]
            for first, second in zip(original["manifests"], repeated["manifests"])
        ),
        "gain_max_abs_difference": gain_max_difference,
        "gain_tolerance": float(config["reproducibility"]["gain_tolerance"]),
        "delta_max_abs_difference": delta_max_difference,
        "delta_tolerance": float(config["reproducibility"]["delta_tolerance"]),
    }
    reproducibility["pass"] = all(
        [
            reproducibility["support_query_ids_identical"],
            reproducibility["reference_assignments_identical"],
            gain_max_difference <= reproducibility["gain_tolerance"],
            delta_max_difference <= reproducibility["delta_tolerance"],
        ]
    )

    all_results = results + [
        {
            **repeated,
            "summary": {**repeated["summary"], "run": f"{repeated_name}_repeat"},
            "gain_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["gain_rows"]],
            "class_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["class_rows"]],
            "tal_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["tal_rows"]],
            "box_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["box_rows"]],
            "gradient_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["gradient_rows"]],
            "inner_rows": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["inner_rows"]],
            "manifests": [{**row, "run": f"{repeated_name}_repeat"} for row in repeated["manifests"]],
        }
    ]
    _write_csv(output / "inner_step_metrics.csv", [row for result in all_results for row in result["inner_rows"]])
    _write_csv(output / "gain_metrics.csv", [row for result in all_results for row in result["gain_rows"]])
    _write_csv(output / "classwise_gain_metrics.csv", [row for result in all_results for row in result["class_rows"]])
    _write_csv(output / "tal_metrics.csv", [row for result in all_results for row in result["tal_rows"]])
    _write_csv(output / "box_dfl_metrics.csv", [row for result in all_results for row in result["box_rows"]])
    _write_csv(output / "gradient_metrics.csv", [row for result in all_results for row in result["gradient_rows"]])
    (output / "episode_manifest.json").write_text(
        json.dumps([row for result in all_results for row in result["manifests"]], indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(results[0]["initial_delta"], output / "delta_initial.pt")
    torch.save(
        {result["summary"]["run"]: result["final_delta"] for result in results}, output / "delta_final.pt"
    )
    state_hashes = {
        result["summary"]["run"]: {
            "before": result["summary"]["base_hash_before"],
            "after": result["summary"]["base_hash_after"],
            "unchanged": result["summary"]["base_unchanged"],
        }
        for result in results
    }
    (output / "state_hashes.json").write_text(json.dumps(state_hashes, indent=2) + "\n", encoding="utf-8")
    memory = {
        result["summary"]["run"]: {"peak_memory_bytes": result["summary"]["peak_memory_bytes"]}
        for result in results
    }
    (output / "memory_profile.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "total_seconds": time.perf_counter() - started,
        "runs": {result["summary"]["run"]: result["summary"]["runtime_seconds"] for result in results},
    }
    (output / "runtime_metrics.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    coverages = [row["target_positive_coverage"] for result in results for row in result["tal_rows"]]
    gate = {
        "real_yolo_forward": True,
        "native_detection_loss": True,
        "j1_classification_head": results[0]["summary"]["status"] == "pass",
        "j1_detection_head": results[1]["summary"]["status"] == "pass",
        "j3_classification_head": results[2]["summary"]["status"] == "pass",
        "j3_detection_head": results[3]["summary"]["status"] == "pass",
        "j5_classification_head": results[4]["summary"]["status"] == "pass",
        "protect_only_mixed_gradient": all(result["summary"]["protect_only_nonzero_count"] >= 1 for result in results),
        "delta_updated": all(result["summary"]["delta_updated"] for result in results),
        "base_model_unchanged": all(result["summary"]["base_unchanged"] for result in results),
        "support_query_overlap_zero": all(result["summary"]["support_query_overlap_max"] == 0 for result in results),
        "target_coverage_median_ge_0_50": statistics.median(coverages) >= 0.5,
        "target_gain_computable": all(result["summary"]["target_gain_computable"] for result in results),
        "non_target_gain_computable": all(result["summary"]["valid_authorized_class_total"] >= 1 for result in results),
        "box_dfl_available": all(result["summary"]["box_dfl_available"] for result in results),
        "finite": all(result["summary"]["finite"] for result in results),
        "artifact_complete": True,
        "reproducible": reproducibility["pass"],
        "no_proxy_fallback": adapter.backend == "real_ultralytics_yolo",
    }
    summary = {
        "status": "real-detector engineering chain pass" if all(gate.values()) else "real-detector engineering chain fail",
        "gate": gate,
        "run_summaries": [result["summary"] for result in results],
        "reproducibility": reproducibility,
        "target_coverage_median": statistics.median(coverages),
        "optional_j5_detection_head": "not_run_optional",
    }
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "run.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not all(gate.values()):
        raise RuntimeError(f"C3 gate failed: {gate}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OA-LGC C3 real-YOLO end-to-end smoke matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))
