from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_j3_learning_gain_validation import batch_from_sample, build_engine, row_from_output, scalar, write_csv, write_json
from runners.run_supervision_decomposition_p0 import read_voc_samples, select_groups
from ue_framework.methods.multitrajectory_gain import TrajectoryBatchSequence
from ue_framework.methods.multitrajectory_gain.feasibility import (
    DualState,
    constrained_objective,
    natural_variation_thresholds,
    raw_counterfactual_gap,
    summarize_clean_gains,
)
from ue_framework.methods.multitrajectory_gain.gradient_diagnostics import gradient_conflict_diagnostics
from ue_framework.methods.multitrajectory_gain.objective import project_delta_linf
from ue_framework.methods.multitrajectory_gain.online_sampler import OnlineTrajectorySampler


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_samples(cfg: Dict, limit: int) -> List[Dict]:
    samples = read_voc_samples(ROOT / cfg["paths"]["voc_root"])
    groups = select_groups(samples, int(cfg["protected_class_id"]), float(cfg["support"].get("ambiguous_iou_threshold", 0.5)), max(limit, 120))
    return groups["person_only"] + groups["low_overlap"] + groups["high_overlap"]


def build_protected_sequences(cfg: Dict, count: int, steps: int, offset: int, imgsz: int, device: torch.device, seed: int) -> List[TrajectoryBatchSequence]:
    pool = protected_samples(cfg, max((count + offset + 1) * (steps + 1), 160))
    sequences = []
    for i in range(count):
        support = []
        base = (offset + i) * (steps + 1)
        used = set()
        for j in range(steps):
            sample = pool[(base + j) % len(pool)]
            used.add(sample["image_id"])
            support.append(batch_from_sample(sample, imgsz, device, seed + i * 10 + j))
        q = pool[(base + steps) % len(pool)]
        if q["image_id"] in used:
            q = pool[(base + steps + 1) % len(pool)]
        query = batch_from_sample(q, imgsz, device, seed + i * 10 + 9)
        sequences.append(TrajectoryBatchSequence(support, query, [b.image_ids for b in support], query.image_ids, seed + i, seed + i))
    return sequences


def build_mixed_sequences(cfg: Dict, count: int, steps: int, offset: int, imgsz: int, device: torch.device, seed: int) -> List[TrajectoryBatchSequence]:
    samples = read_voc_samples(ROOT / cfg["paths"]["voc_root"])
    groups = select_groups(samples, int(cfg["protected_class_id"]), float(cfg["support"].get("ambiguous_iou_threshold", 0.5)), max(count + offset + 80, 120))
    group_names = ["person_only", "low_overlap", "high_overlap", "authorized_only"]
    pointers = {name: offset for name in group_names}

    def take(group: str, used: set[str]) -> Dict:
        pool = groups[group]
        for _ in range(len(pool)):
            sample = pool[pointers[group] % len(pool)]
            pointers[group] += 1
            if sample["image_id"] not in used:
                used.add(sample["image_id"])
                return sample
        raise RuntimeError(f"Could not sample unique image from group {group}.")

    sequences = []
    for i in range(count):
        used: set[str] = set()
        query_group = group_names[(i + seed) % len(group_names)]
        support_groups = [group_names[(i + j) % len(group_names)] for j in range(steps)]
        if query_group in support_groups:
            support_groups = ["high_overlap" if g == query_group and g != "high_overlap" else "low_overlap" if g == query_group else g for g in support_groups]
        support_samples = [take(group, used) for group in support_groups]
        query_sample = take(query_group, used)
        support = [batch_from_sample(sample, imgsz, device, seed + i * 10 + j) for j, sample in enumerate(support_samples)]
        query = batch_from_sample(query_sample, imgsz, device, seed + i * 10 + 9)
        sequences.append(TrajectoryBatchSequence(support, query, [b.image_ids for b in support], query.image_ids, seed + i, seed + i))
    return sequences


def screen_checkpoint(cfg: Dict, checkpoint: Dict, steps: int, device: torch.device, out_dir: Path) -> tuple[List[Dict], Dict]:
    local_cfg = dict(cfg)
    local_cfg["paths"] = dict(cfg["paths"])
    local_cfg["paths"]["checkpoint"] = checkpoint["path"]
    imgsz = int(cfg["optimization"].get("imgsz", 160))
    count = int(cfg["checkpoint_screening"].get("trajectories", 30))
    sequences = build_protected_sequences(local_cfg, count, steps, 0, imgsz, device, seed=steps * 100)
    engine = build_engine(local_cfg, steps, device)
    delta = torch.zeros((1, 3, imgsz, imgsz), device=device, requires_grad=True)
    rows = []
    for idx, seq in enumerate(sequences):
        out = engine.run(seq, delta, create_graph=False)
        stats = out.metrics.statistics
        rows.append(
            {
                "checkpoint_name": checkpoint["name"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_hash": sha256_file(ROOT / checkpoint["path"]),
                "epoch": checkpoint.get("epoch", "unknown"),
                "role": checkpoint.get("role", "unknown"),
                "J": steps,
                "trajectory_index": idx,
                "support_ids": "|".join(sum(seq.support_image_ids, [])),
                "query_ids": "|".join(seq.query_image_ids),
                "protected_loss_before": stats["L_t_before"],
                "protected_loss_after_clean": stats["L_t_after_clean"],
                "protected_raw_gain": stats["G_t_clean"],
                "authorized_loss_before": stats["L_a_before"],
                "authorized_loss_after_clean": stats["L_a_after_clean"],
                "authorized_raw_gain": stats["G_a_clean"],
                "shared_loss_before": stats["L_s_before"],
                "shared_loss_after_clean": stats["L_s_after_clean"],
                "shared_raw_gain": stats["G_s_clean"],
                "protected_positive_count": stats["protected_valid"] and out.logs.get("protected_support_batches", 0.0),
                "authorized_positive_count": out.logs.get("authorized_support_batches", 0.0),
                "shared_supervision_count": out.logs.get("shared_support_batches", 0.0),
                "protected_valid": stats["protected_valid"],
                "parameter_leak": out.logs["surrogate_parameter_max_abs_diff"],
            }
        )
    summary = summarize_clean_gains(rows)
    summary.update(
        {
            "checkpoint_name": checkpoint["name"],
            "checkpoint_path": checkpoint["path"],
            "checkpoint_hash": sha256_file(ROOT / checkpoint["path"]),
            "epoch": checkpoint.get("epoch", "unknown"),
            "role": checkpoint.get("role", "unknown"),
            "J": steps,
            "protected_ap": float("nan"),
            "authorized_ap": float("nan"),
            "full_map": float("nan"),
            "training_loss": float("nan"),
        }
    )
    return rows, summary


def select_screening_winner(summaries: List[Dict], required: float) -> Dict | None:
    passed = [s for s in summaries if float(s["protected_valid_ratio"]) >= required and float(s["protected_gain_median"]) > 0.0]
    if not passed:
        return None
    passed.sort(key=lambda s: (int(s["J"]), -float(s["protected_valid_ratio"]), float(s["protected_gain_std"])))
    return passed[0]


def clean_clean_calibration(cfg: Dict, checkpoint: Dict, steps: int, device: torch.device) -> tuple[List[Dict], Dict]:
    imgsz = int(cfg["optimization"].get("imgsz", 160))
    pairs = int(cfg["clean_clean"].get("pairs", 50))
    sequences_a = build_mixed_sequences(cfg, pairs, steps, 20, imgsz, device, seed=3000)
    sequences_b = build_mixed_sequences(cfg, pairs, steps, 80, imgsz, device, seed=4000)
    engine = build_engine(cfg, steps, device)
    delta = torch.zeros((1, 3, imgsz, imgsz), device=device, requires_grad=True)
    rows = []
    for idx, (a, b) in enumerate(zip(sequences_a, sequences_b)):
        b = TrajectoryBatchSequence(b.support_batches, a.query_batch, b.support_image_ids, a.query_image_ids, b.augmentation_seed, b.batch_order_seed)
        out_a = engine.run(a, delta, create_graph=False)
        out_b = engine.run(b, delta, create_graph=False)
        rows.append(
            {
                "pair_index": idx,
                "query_ids": "|".join(a.query_image_ids),
                "N_t": abs(scalar(out_a.clean_query_losses["protected"]) - scalar(out_b.clean_query_losses["protected"])),
                "N_a": abs(scalar(out_a.clean_query_losses["authorized"]) - scalar(out_b.clean_query_losses["authorized"])),
                "N_s": abs(scalar(out_a.clean_query_losses["shared"]) - scalar(out_b.clean_query_losses["shared"])),
            }
        )
    thresholds = natural_variation_thresholds(rows, float(cfg["clean_clean"].get("threshold_quantile", 90)), float(cfg["clean_clean"].get("kappa_t", 2.0)))
    return rows, thresholds


def build_train_batches(cfg: Dict, heldout_ids: set[str], count: int, imgsz: int, device: torch.device):
    pool = protected_samples(cfg, max(count + len(heldout_ids), 240))
    selected = [sample for sample in pool if sample["image_id"] not in heldout_ids][:count]
    return [batch_from_sample(sample, imgsz, device, 5000 + idx) for idx, sample in enumerate(selected)]


def evaluate_raw_gap(engine, sequences, delta, thresholds: Dict, split: str, step: int) -> tuple[List[Dict], Dict]:
    rows = []
    for idx, seq in enumerate(sequences):
        out = engine.run(seq, delta.detach(), create_graph=False)
        gaps = raw_counterfactual_gap(out.clean_query_losses, out.poison_query_losses)
        vt = max(0.0, thresholds["protected_margin"] - scalar(gaps["Delta_t"]))
        va = max(0.0, abs(scalar(gaps["Delta_a"])) - thresholds["tau_a"])
        vs = max(0.0, abs(scalar(gaps["Delta_s"])) - thresholds["tau_s"])
        rows.append(
            {
                "split": split,
                "outer_step": step,
                "trajectory_index": idx,
                "Delta_t": scalar(gaps["Delta_t"]),
                "Delta_a": scalar(gaps["Delta_a"]),
                "Delta_s": scalar(gaps["Delta_s"]),
                "v_t": vt,
                "v_a": va,
                "v_s": vs,
                "composite": scalar(gaps["Delta_t"]) - va - vs,
                "support_ratio_mean": out.logs["support_ratio_mean"],
                "outside_support_max_abs_delta": out.logs["outside_support_max_abs_delta"],
                "parameter_leak": out.logs["surrogate_parameter_max_abs_diff"],
            }
        )
    summary = {k: float(np.mean([row[k] for row in rows])) for k in ["Delta_t", "Delta_a", "Delta_s", "v_t", "v_a", "v_s", "composite", "parameter_leak"]}
    summary["feasible_ratio"] = float(np.mean([row["v_a"] == 0.0 and row["v_s"] == 0.0 and row["Delta_t"] > 0.0 for row in rows]))
    return rows, summary


def finite_difference_check(engine, sequence, thresholds: Dict, imgsz: int, device: torch.device) -> Dict:
    torch.manual_seed(123)
    base = (torch.randn((1, 3, imgsz, imgsz), device=device) * 1.0e-3).detach()
    direction = torch.randn_like(base)
    direction = direction / direction.norm().clamp_min(1.0e-12)
    h = 1.0e-4

    def objective_value(delta_value: torch.Tensor, create_graph: bool) -> torch.Tensor:
        out = engine.run(sequence, delta_value, create_graph=create_graph)
        gaps = raw_counterfactual_gap(out.clean_query_losses, out.poison_query_losses)
        obj = constrained_objective(gaps, thresholds, 1.0, 1.0, delta_value.pow(2).mean(), 1.0)
        return obj["total"]

    probe = base.clone().detach().requires_grad_(True)
    objective_value(probe, create_graph=True).backward()
    autograd_dir = float((probe.grad * direction).sum().detach().cpu().item())
    plus = objective_value((base + h * direction).detach(), create_graph=False)
    minus = objective_value((base - h * direction).detach(), create_graph=False)
    finite_diff_dir = float(((plus - minus) / (2.0 * h)).detach().cpu().item())
    abs_error = abs(autograd_dir - finite_diff_dir)
    rel_error = abs_error / max(1.0e-12, abs(finite_diff_dir))
    status = "pass" if rel_error <= 0.25 else "mismatch"
    return {
        "status": status,
        "note": "Finite differences are a diagnostic only; dynamic assignment and nonsmooth ReLU constraints can make this mismatch even when autograd is connected.",
        "h": h,
        "autograd_directional_derivative": autograd_dir,
        "finite_difference_directional_derivative": finite_diff_dir,
        "abs_error": abs_error,
        "relative_error": rel_error,
    }


def optimize_raw_constraints(cfg: Dict, checkpoint: Dict, steps: int, thresholds: Dict, heldout_sequences, device: torch.device, out_dir: Path):
    imgsz = int(cfg["optimization"].get("imgsz", 160))
    engine = build_engine(cfg, steps, device)
    heldout_ids = {image_id for seq in heldout_sequences for ids in seq.support_image_ids for image_id in ids} | {image_id for seq in heldout_sequences for image_id in seq.query_image_ids}
    train_batches = build_train_batches(cfg, heldout_ids, int(cfg["trajectory_sampling"].get("train_pool_batches", 180)), imgsz, device)
    sampler = OnlineTrajectorySampler(train_batches, [seq.query_batch for seq in heldout_sequences], support_steps=steps, seed=9000, recent_image_exclusion_window=int(cfg["trajectory_sampling"].get("recent_image_exclusion_window", 20)), max_attempts=int(cfg["trajectory_sampling"].get("max_sampling_attempts", 30)))
    delta = torch.zeros((1, 3, imgsz, imgsz), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(cfg["optimization"].get("delta_learning_rate", 0.01)))
    dual = DualState(float(cfg["constraints"].get("mu_authorized_init", 1.0)), float(cfg["constraints"].get("mu_shared_init", 1.0)))
    eps = float(cfg["optimization"].get("eps", 16.0 / 255.0))
    eval_interval = int(cfg["optimization"].get("evaluation_interval", 5))
    max_steps = int(cfg["optimization"].get("outer_steps", 30))
    candidate_dir = out_dir / "delta_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    curve = []
    heldout_curve = []
    dual_rows = []
    gradient_rows = []
    memory = []
    failures = []
    best = {"protected_gap": None, "feasible": None, "composite": None}
    fd_result = finite_difference_check(engine, heldout_sequences[0], thresholds, imgsz, device) if heldout_sequences else {"status": "not_run", "reason": "no held-out sequences"}

    def save_candidate(name: str, step: int, heldout_summary: Dict) -> None:
        path = candidate_dir / f"{name}.pt"
        torch.save(delta.detach().cpu(), path)
        meta = {"name": name, "step": step, "path": str(path), **heldout_summary}
        (candidate_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    rows, summary = evaluate_raw_gap(engine, heldout_sequences, delta, thresholds, "heldout", 0)
    heldout_curve.append({"outer_step": 0, **summary})
    save_candidate("delta_step_0", 0, summary)
    for step in range(1, max_steps + 1):
        sample = sampler.sample(step)
        if sample.sequence is None:
            failures.append({"outer_step": step, "reason": sample.rejection_reason})
            continue
        probe = engine.run(sample.sequence, delta.detach(), create_graph=False)
        if float(probe.metrics.statistics["G_t_clean"]) <= 0.0:
            failures.append({"outer_step": step, "reason": "nonpositive_clean_gain"})
            continue
        optimizer.zero_grad(set_to_none=True)
        out = engine.run(sample.sequence, delta, create_graph=True)
        gaps = raw_counterfactual_gap(out.clean_query_losses, out.poison_query_losses)
        obj = constrained_objective(gaps, thresholds, dual.mu_authorized, dual.mu_shared, delta.pow(2).mean(), float(cfg["constraints"].get("lambda_regularization", 1.0)))
        diag = gradient_conflict_diagnostics({"protected": obj["v_t"], "authorized": obj["v_a"], "shared": obj["v_s"]}, delta)
        obj["total"].backward()
        optimizer.step()
        project_delta_linf(delta, eps)
        dual = dual.update(obj["v_a"], obj["v_s"], float(cfg["constraints"].get("dual_learning_rate", 0.1)), float(cfg["constraints"].get("mu_max", 100.0)))
        curve.append({"outer_step": step, "loss": scalar(obj["total"]), "Delta_t": scalar(gaps["Delta_t"]), "Delta_a": scalar(gaps["Delta_a"]), "Delta_s": scalar(gaps["Delta_s"]), "v_t": scalar(obj["v_t"]), "v_a": scalar(obj["v_a"]), "v_s": scalar(obj["v_s"]), "mu_a": dual.mu_authorized, "mu_s": dual.mu_shared, "delta_linf": float(delta.detach().abs().max().item())})
        dual_rows.append({"outer_step": step, "mu_a": dual.mu_authorized, "mu_s": dual.mu_shared, "mu_a_at_max": dual.mu_authorized >= float(cfg["constraints"].get("mu_max", 100.0)), "mu_s_at_max": dual.mu_shared >= float(cfg["constraints"].get("mu_max", 100.0))})
        for row in diag["rows"]:
            gradient_rows.append({"outer_step": step, **row})
        if torch.cuda.is_available():
            memory.append({"outer_step": step, "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(), "max_allocated": torch.cuda.max_memory_allocated()})
        if step % eval_interval == 0 or step == max_steps:
            rows, summary = evaluate_raw_gap(engine, heldout_sequences, delta, thresholds, "heldout", step)
            heldout_curve.append({"outer_step": step, **summary})
            save_candidate(f"delta_step_{step}", step, summary)
            if best["protected_gap"] is None or summary["Delta_t"] > best["protected_gap"]["Delta_t"]:
                best["protected_gap"] = {"step": step, **summary}
                save_candidate("delta_best_protected_gap", step, summary)
            if summary["v_a"] == 0.0 and summary["v_s"] == 0.0 and summary["Delta_t"] > 0.0 and (best["feasible"] is None or summary["Delta_t"] > best["feasible"]["Delta_t"]):
                best["feasible"] = {"step": step, **summary}
                save_candidate("delta_best_constraint_feasible", step, summary)
            if best["composite"] is None or summary["composite"] > best["composite"]["composite"]:
                best["composite"] = {"step": step, **summary}
                save_candidate("delta_best_composite", step, summary)
    return {"curve": curve, "heldout_curve": heldout_curve, "dual": dual_rows, "gradient": gradient_rows, "memory": memory, "failures": failures, "best": best, "finite_difference": fd_result}


def write_report(path: Path, summary: Dict) -> None:
    winner = summary.get("winner")
    if winner is None:
        decision = "D. Current YOLOv8 head-only short-horizon rollout cannot form stable protected learning signal."
    elif not summary.get("constraints_feasible", False):
        decision = "D. Constraints cannot be satisfied simultaneously; class-selective multi-step proxy is not feasible."
    elif summary.get("proxy_ap_status") == "not_run":
        decision = "E. Proxy/AP correlation was not established; stop the multi-step trajectory route."
    else:
        decision = "G. Implementation or data issue needs repair before a route decision."
    lines = [
        "# Short-Horizon Learnability Proxy Feasibility Report",
        "",
        "## Checkpoint Learnability",
        f"- checkpoint candidates found: `{summary['checkpoint_candidates']}`",
        f"- checkpoint screening rows: `{summary.get('checkpoint_screening')}`",
        f"- selected setting: `{winner}`",
        f"- any protected-valid ratio >= 0.80: `{winner is not None}`",
        "",
        "## Natural Variation",
        f"- thresholds: `{summary.get('thresholds')}`",
        "",
        "## Raw Counterfactual Gap",
        f"- final held-out summary: `{summary.get('final_heldout')}`",
        f"- best candidates: `{summary.get('best_candidates')}`",
        f"- constraints feasible: `{summary.get('constraints_feasible')}`",
        "",
        "## Finite Difference",
        f"- result: `{summary.get('finite_difference')}`",
        "",
        "## Proxy/AP Correlation",
        f"- status: `{summary.get('proxy_ap_status')}`",
        f"- reason: `{summary.get('proxy_ap_reason')}`",
        "",
        "## Final Decision",
        f"- `{decision}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/multitrajectory_gain/short_horizon_feasibility.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/short_horizon_feasibility"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()
    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    distributions: Dict[str, List[Dict]] = {}
    for checkpoint in cfg["checkpoint_screening"]["checkpoint_candidates"]:
        if checkpoint.get("role") == "skipped_non_voc20":
            summaries.append({"checkpoint_name": checkpoint["name"], "checkpoint_path": checkpoint["path"], "role": checkpoint["role"], "J": 0, "protected_valid_ratio": 0.0, "skip_reason": "unknown_or_non_voc20_class_space"})
            continue
        for steps in cfg["checkpoint_screening"]["rollout_steps"]:
            rows, summary = screen_checkpoint(cfg, checkpoint, int(steps), device, out_dir)
            all_rows.extend(rows)
            summaries.append(summary)
            distributions[f"{checkpoint['name']}_J{steps}"] = rows
    required = float(cfg["checkpoint_screening"].get("valid_ratio_required", 0.8))
    winner = select_screening_winner(summaries, required)
    write_csv(out_dir / "checkpoint_screening.csv", summaries)
    write_json(out_dir / "checkpoint_gain_distributions.json", distributions)
    write_csv(out_dir / "clean_clean_calibration.csv", [])
    write_json(out_dir / "natural_variation_thresholds.json", {})
    write_csv(out_dir / "optimization_curve.csv", [])
    write_csv(out_dir / "heldout_curve.csv", [])
    write_csv(out_dir / "dual_multiplier_curve.csv", [])
    write_csv(out_dir / "gradient_scale_diagnostics.csv", [])
    write_csv(out_dir / "candidate_proxy_metrics.csv", [])
    write_csv(out_dir / "candidate_victim_metrics.csv", [])
    write_json(out_dir / "proxy_ap_correlations.json", {"status": "not_run"})
    write_csv(out_dir / "j1_vs_j3_correlation.csv", [])
    write_csv(out_dir / "memory_profile.csv", [])
    write_json(out_dir / "finite_difference_results.json", {"status": "pending"})
    failure_cases = []
    summary = {
        "checkpoint_candidates": [c["name"] for c in cfg["checkpoint_screening"]["checkpoint_candidates"]],
        "checkpoint_screening": summaries,
        "winner": winner,
        "elapsed_sec": time.time() - started,
    }
    if winner is not None:
        selected = next(c for c in cfg["checkpoint_screening"]["checkpoint_candidates"] if c["name"] == winner["checkpoint_name"])
        cfg["paths"]["checkpoint"] = selected["path"]
        steps = int(winner["J"])
        rows, thresholds = clean_clean_calibration(cfg, selected, steps, device)
        write_csv(out_dir / "clean_clean_calibration.csv", rows)
        write_json(out_dir / "natural_variation_thresholds.json", thresholds)
        heldout = build_mixed_sequences(cfg, int(cfg["trajectory_sampling"].get("heldout_trajectories", 30)), steps, 160, int(cfg["optimization"].get("imgsz", 160)), device, seed=7000)
        opt = optimize_raw_constraints(cfg, selected, steps, thresholds, heldout, device, out_dir)
        write_csv(out_dir / "optimization_curve.csv", opt["curve"])
        write_csv(out_dir / "heldout_curve.csv", opt["heldout_curve"])
        write_csv(out_dir / "candidate_proxy_metrics.csv", opt["heldout_curve"])
        write_csv(out_dir / "dual_multiplier_curve.csv", opt["dual"])
        write_csv(out_dir / "gradient_scale_diagnostics.csv", opt["gradient"])
        write_csv(out_dir / "memory_profile.csv", opt["memory"])
        write_json(out_dir / "finite_difference_results.json", opt["finite_difference"])
        failure_cases = opt["failures"]
        final_heldout = opt["heldout_curve"][-1] if opt["heldout_curve"] else {}
        constraints_feasible = bool(final_heldout.get("Delta_t", 0.0) > 0.0 and final_heldout.get("v_a", 1.0) == 0.0 and final_heldout.get("v_s", 1.0) == 0.0)
        summary.update({"thresholds": thresholds, "final_heldout": final_heldout, "best_candidates": opt["best"], "constraints_feasible": constraints_feasible})
        summary.update({"finite_difference": opt["finite_difference"]})
        if constraints_feasible:
            summary.update({"proxy_ap_status": "not_run", "proxy_ap_reason": "victim smoke not implemented in this stopped feasibility runner"})
        else:
            summary.update({"proxy_ap_status": "not_run", "proxy_ap_reason": "raw counterfactual constraints failed hard-stop criteria"})
    else:
        failure_cases.append({"stage": "checkpoint_screening", "reason": "no checkpoint/J reached protected-valid ratio >= 0.80"})
        summary.update({"proxy_ap_status": "not_run", "proxy_ap_reason": "checkpoint screening failed"})
    write_json(out_dir / "failure_cases.json", failure_cases)
    write_json(out_dir / "summary.json", summary)
    write_report(ROOT / "docs/short_horizon_feasibility_report.md", summary)
    write_report(ROOT / "docs/checkpoint_screening_report.md", summary)
    if summary.get("thresholds"):
        (ROOT / "docs/clean_clean_calibration_report.md").write_text("# Clean-Clean Calibration Report\n\n" + json.dumps(summary["thresholds"], indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "report": str(ROOT / "docs/short_horizon_feasibility_report.md")}, indent=2))


if __name__ == "__main__":
    main()
