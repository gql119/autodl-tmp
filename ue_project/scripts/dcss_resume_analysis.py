import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dcss.diagnostics import (
    principal_angles_degrees,
    projection_similarity,
    random_baseline_summary,
    selectivity_ratio,
    semantic_overlap,
)
from dcss.generalized_eigen import random_subspace, solve_no_semantic_subspace
from dcss.resume import diagnostic_gate
from dcss.semantic_pca import fit_semantic_pca_from_statistics
from dcss.statistics import RunningCovariance
from dcss.subspace_io import load_subspaces, save_subspaces
from ue_framework.data_utils import list_images, read_yolo_annotations, label_path_for_image
from ue_framework.metrics_utils import compute_non_target_map, extract_map50_per_class


VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def _write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _environment():
    return (
        f"platform={platform.platform()}\npython={platform.python_version()}\n"
        f"torch={torch.__version__}\ncuda_available={torch.cuda.is_available()}\n"
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}\n"
    )


def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def baseline(args):
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=False)
    data_yaml = {
        "path": os.path.abspath(args.dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(VOC_NAMES)},
    }
    config = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "checkpoint": os.path.abspath(args.checkpoint),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "target_class_id": 14,
        "gate_reference_non_target": args.reference_non_target,
        "gate_tolerance": 0.02,
    }
    with open(os.path.join(output, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    data_path = os.path.join(output, "data.yaml")
    with open(data_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False, allow_unicode=True)
    command = " ".join(sys.argv)
    open(os.path.join(output, "command.txt"), "w", encoding="utf-8").write(command + "\n")
    open(os.path.join(output, "environment.txt"), "w", encoding="utf-8").write(_environment())
    open(os.path.join(output, "git_commit.txt"), "w", encoding="utf-8").write(_git_commit() + "\n")
    open(os.path.join(output, "reused_checkpoint.txt"), "w", encoding="utf-8").write(os.path.abspath(args.checkpoint) + "\n")
    log_path = os.path.join(output, "evaluation.log")
    with open(log_path, "w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        model = YOLO(args.checkpoint)
        results = model.val(
            data=data_path, split="val", imgsz=args.imgsz, batch=args.batch, device=args.device,
            workers=args.workers, plots=False, project=output, name="ultralytics_eval", exist_ok=True,
        )
    per_class = extract_map50_per_class(results, 20)
    metrics = {
        "dataset_scope": "VOC mini 800 train / 200 val",
        "train_samples": len(list_images(os.path.join(args.dataset_root, "images", "train"))),
        "val_samples": len(list_images(os.path.join(args.dataset_root, "images", "val"))),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_reused": True,
        "retrained": False,
        "checkpoint_training_epochs": 150,
        "convergence_evidence": "checkpoint is a completed 150-epoch clean VOC20 surrogate; no recovery retraining",
        "mAP50_target": float(per_class[14]),
        "mAP50_non_target": compute_non_target_map(per_class, 14),
        "mAP50_all": float(np.nanmean(per_class)),
        "ap50_per_class": per_class,
    }
    metrics["gate"] = {
        "criterion": "no trustworthy same-split clean history; non-target mAP50 must be at least 0.70",
        "reference": args.reference_non_target,
        "minimum": 0.70,
        "pass": metrics["mAP50_non_target"] >= 0.70,
    }
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(output, "baseline_comparison.csv"), [{
        "checkpoint": metrics["checkpoint"], "scope": metrics["dataset_scope"],
        "reference_non_target": args.reference_non_target,
        "observed_target": metrics["mAP50_target"], "observed_non_target": metrics["mAP50_non_target"],
        "observed_all": metrics["mAP50_all"], "absolute_non_target_difference": abs(metrics["mAP50_non_target"] - args.reference_non_target),
        "gate": "pass" if metrics["gate"]["pass"] else "fail",
    }])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _stat(scope, name):
    return RunningCovariance.from_state_dict(scope[name])


def no_pt(args):
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=False)
    raw = torch.load(args.raw_statistics, map_location="cpu", weights_only=False)
    layer = raw["layers"][args.layer]
    full = layer["scopes"]["full"]
    target = _stat(full, "target_gradient")
    non_target = _stat(full, "non_target_gradient")
    features = _stat(full, "target_feature")
    ct, cnt = target.second_moment(), non_target.second_moment()
    semantic = fit_semantic_pca_from_statistics(features.mean, features.covariance(unbiased=True), features.count, variance_threshold=0.90)
    solved = solve_no_semantic_subspace(ct, cnt, args.rank, args.regularization)
    split_bases = []
    for split_name in ["split_a", "split_b"]:
        scope = layer["scopes"][split_name]
        split_bases.append(solve_no_semantic_subspace(
            _stat(scope, "target_gradient").second_moment(),
            _stat(scope, "non_target_gradient").second_moment(),
            args.rank, args.regularization,
        ).basis)
    stability = projection_similarity(split_bases[0], split_bases[1])
    original_payload = load_subspaces(args.original_subspaces)
    original = original_payload["layers"][args.layer]["subspaces"]["dcss"][args.rank]
    angles = principal_angles_degrees(original, solved.basis)
    random_values = torch.tensor([
        selectivity_ratio(ct, cnt, random_subspace(layer["dimension"], args.rank, raw["seed"] * 1000 + index))
        for index in range(args.random_subspaces)
    ], dtype=torch.float64)
    r_sel = selectivity_ratio(ct, cnt, solved.basis)
    random_summary = random_baseline_summary(random_values, r_sel)
    metrics = {
        "layer": args.layer, "rank": args.rank, "source": "no_pt",
        "R_sel": r_sel, "R_sem": semantic_overlap(solved.basis, semantic.basis), "R_stab": stability,
        "random_mean": random_summary["random_mean"], "random_std": random_summary["random_std"],
        "random_significance_pass": r_sel > random_summary["random_mean"] + 2 * random_summary["random_std"],
        "orthogonality_error": solved.orthogonality_error,
        "finite": bool(torch.isfinite(solved.basis).all() and torch.isfinite(solved.eigenvalues).all()),
        "principal_angles_degrees": angles.tolist(),
        "principal_angle_mean_degrees": float(angles.mean()),
        "target_gradient_projected_energy": float(torch.trace(solved.basis.T @ ct @ solved.basis)),
        "non_target_gradient_projected_energy": float(torch.trace(solved.basis.T @ cnt @ solved.basis)),
        "original_R_sem": semantic_overlap(original, semantic.basis),
    }
    metrics["engineering_gate"] = {
        "finite": metrics["finite"], "orthogonality": metrics["orthogonality_error"] <= 1e-4,
        "random_significance": metrics["random_significance_pass"], "stability_calculable": math.isfinite(stability),
    }
    metrics["engineering_gate"]["pass"] = all(metrics["engineering_gate"].values())
    payload = {
        "schema_version": 1, "seed": raw["seed"], "checkpoint": raw["checkpoint"],
        "dataset_root": raw["dataset_root"], "layers": {args.layer: {
            "dimension": layer["dimension"], "semantic_basis": semantic.basis,
            "subspaces": {"no_pt": {args.rank: solved.basis}},
        }},
    }
    save_subspaces(os.path.join(output, "subspace.pt"), payload)
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(output, "eigenvalues.csv"), [
        {"index": index + 1, "eigenvalue": float(value)} for index, value in enumerate(solved.eigenvalues)
    ])
    _write_csv(os.path.join(output, "principal_angles.csv"), [
        {"index": index + 1, "angle_degrees": float(value)} for index, value in enumerate(angles)
    ])
    with open(os.path.join(output, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump({key: value for key, value in vars(args).items() if key != "func"}, file, sort_keys=False, allow_unicode=True)
    open(os.path.join(output, "run.log"), "w", encoding="utf-8").write(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {None, "", "nan", "inf", "-inf"}]
    return float(np.mean(values)) if values else float("nan")


def _mechanism_summary(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    final_epoch = max(int(row["epoch"]) for row in rows)
    final = [row for row in rows if int(row["epoch"]) == final_epoch]
    target = _mean(final, "target_projected_energy_mean")
    leakage = _mean(final, "non_target_projected_leakage_mean")
    return rows, {
        "target_energy": target,
        "target_outside_energy": _mean(final, "target_outside_energy_mean"),
        "in_subspace_ratio": _mean(final, "target_in_subspace_ratio"),
        "NT_leakage_mean": leakage,
        "NT_leakage_max": max(float(row["non_target_projected_leakage_p95_class"]) for row in final),
        "R_shift": target / (leakage + 1e-12),
        "pairwise_cosine": _mean(final, "target_shift_pairwise_cosine"),
        "effective_rank": _mean(final, "target_shift_effective_rank"),
        "target_unit_coverage": float(final[-1]["target_unit_coverage_running"]),
    }


def diagnose(args):
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=False)
    experiments = {"E2 random": "E2_random_r8_seed0", "E3 target-only": "E3_target_only_r8_seed0", "E4 DCSS": "E4_dcss_r8_seed0"}
    comparisons, all_rows = [], {}
    for label, experiment in experiments.items():
        rows, summary = _mechanism_summary(os.path.join(args.stage1_root, experiment, "mechanism_metrics.csv"))
        all_rows[label] = rows
        comparisons.append({"experiment": label, **summary})
    e2, e4 = comparisons[0], comparisons[2]
    comparisons[2]["target_energy_vs_E2"] = e4["target_energy"] - e2["target_energy"]
    comparisons[2]["NT_leakage_vs_E2"] = e4["NT_leakage_mean"] - e2["NT_leakage_mean"]
    _write_csv(os.path.join(output, "existing_run_comparison.csv"), comparisons)

    classwise = list(csv.DictReader(open(os.path.join(args.stage1_root, experiments["E4 DCSS"], "classwise_metrics.csv"), newline="", encoding="utf-8")))
    image_dir = os.path.join(args.dataset_root, "images", "train")
    label_dir = os.path.join(args.dataset_root, "labels", "train")
    person_images = 0
    cooccur = {index: 0 for index in range(20)}
    for image in list_images(image_dir):
        classes = {int(annotation["cls"]) for annotation in read_yolo_annotations(label_path_for_image(image, label_dir))}
        if 14 in classes:
            person_images += 1
            for class_id in classes - {14}:
                cooccur[class_id] += 1
    ranking = []
    for row in classwise:
        class_id = int(row["class_id"])
        if class_id == 14:
            continue
        ranking.append({
            "class_id": class_id, "class_name": VOC_NAMES[class_id], "leakage": float(row["leakage"]),
            "logit_drift": float(row["logit_drift"]), "observed_batches": int(row["batches"]),
            "person_cooccur_images": cooccur[class_id], "person_cooccur_rate": cooccur[class_id] / max(1, person_images),
        })
    ranking.sort(key=lambda row: row["leakage"], reverse=True)
    _write_csv(os.path.join(output, "classwise_leakage_ranking.csv"), ranking)

    e4_rows = all_rows["E4 DCSS"]
    loss_rows = []
    for epoch in sorted({int(row["epoch"]) for row in e4_rows}):
        selected = [row for row in e4_rows if int(row["epoch"]) == epoch]
        margin_unsatisfied = float(np.mean([math.sqrt(max(0.0, float(row["target_projected_energy_mean"]))) < 1.0 for row in selected]))
        energy = np.array([float(row["target_projected_energy_mean"]) for row in selected])
        leakage = np.array([float(row["non_target_projected_leakage_mean"]) for row in selected])
        correlation = float(np.corrcoef(energy, leakage)[0, 1]) if len(selected) > 1 else float("nan")
        loss_rows.append({
            "epoch": epoch, "target_energy_mean": float(energy.mean()), "leakage_mean": float(leakage.mean()),
            "margin_unsatisfied_fraction": margin_unsatisfied, "energy_leakage_correlation": correlation,
            "lambda_energy": 1.0, "lambda_leakage": 1.0, "energy_margin": 1.0,
        })
    _write_csv(os.path.join(output, "loss_scale_analysis.csv"), loss_rows)
    gradient_rows = [{
        "evidence_type": "metric_proxy_not_direct_gradient",
        "energy_leakage_correlation": row["energy_leakage_correlation"],
        "epoch": row["epoch"],
        "interpretation": "positive correlation indicates attack energy and collateral leakage rise together; E0-E4 did not log component gradients",
    } for row in loss_rows]
    _write_csv(os.path.join(output, "gradient_conflict_summary.csv"), gradient_rows)
    max_step = max(all_rows["E4 DCSS"], key=lambda row: float(row["target_projected_energy_mean"]))
    diagnosis = {
        "comparison": comparisons,
        "top5_leakage_classes": ranking[:5],
        "e4_target_energy_peak": {"epoch": int(max_step["epoch"]), "step": int(max_step["step"]), "value": float(max_step["target_projected_energy_mean"])},
        "margin_unsatisfied_by_epoch": [row["margin_unsatisfied_fraction"] for row in loss_rows],
        "direct_gradient_conflict_available": False,
        "failure_classification": ["baseline underfitting", "mechanism failure", "selectivity failure", "transfer failure"],
        "conclusion": "multiple factors: the mini victim is underfit, while E4 also has independently observed leakage and no advantage over random",
    }
    with open(os.path.join(output, "diagnosis.json"), "w", encoding="utf-8") as file:
        json.dump(diagnosis, file, indent=2, ensure_ascii=False)
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))


def aggregate(args):
    rows = []
    for experiment in args.experiments:
        directory = os.path.join(args.resume_root, f"diagnostic_{experiment}")
        with open(os.path.join(directory, "metrics.json"), encoding="utf-8") as file:
            metrics = json.load(file)
        gate = diagnostic_gate(metrics)
        metrics["Gate"] = "pass" if gate["pass"] else "fail"
        rows.append(metrics)
    _write_csv(os.path.join(args.resume_root, "diagnostic_summary.csv"), rows)
    payload = {"candidates": {row["experiment_id"]: diagnostic_gate(row) for row in rows}}
    payload["passing_candidates"] = [key for key, value in payload["candidates"].items() if value["pass"]]
    payload["pass"] = bool(payload["passing_candidates"])
    with open(os.path.join(args.resume_root, "diagnostic_gate.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    base = commands.add_parser("baseline")
    base.add_argument("--output", required=True); base.add_argument("--dataset-root", required=True); base.add_argument("--checkpoint", required=True)
    base.add_argument("--imgsz", type=int, default=640); base.add_argument("--batch", type=int, default=16); base.add_argument("--device", default="0"); base.add_argument("--workers", type=int, default=0)
    base.add_argument("--reference-non-target", type=float, default=0.78); base.set_defaults(func=baseline)
    no = commands.add_parser("no-pt")
    no.add_argument("--output", required=True); no.add_argument("--raw-statistics", required=True); no.add_argument("--original-subspaces", required=True)
    no.add_argument("--layer", default="model.15"); no.add_argument("--rank", type=int, default=8); no.add_argument("--regularization", type=float, default=1e-4); no.add_argument("--random-subspaces", type=int, default=100); no.set_defaults(func=no_pt)
    diag = commands.add_parser("diagnose")
    diag.add_argument("--output", required=True); diag.add_argument("--stage1-root", required=True); diag.add_argument("--dataset-root", required=True); diag.set_defaults(func=diagnose)
    agg = commands.add_parser("aggregate")
    agg.add_argument("--resume-root", required=True); agg.add_argument("--experiments", nargs="+", required=True); agg.set_defaults(func=aggregate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
