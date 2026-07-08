from __future__ import annotations

import argparse
import json
import math
import statistics
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

from runners.run_j3_learning_gain_validation import batch_from_sample, build_engine, build_sequences, row_from_output, scalar, summarize, write_csv, write_json
from runners.run_supervision_decomposition_p0 import read_voc_samples, select_groups
from ue_framework.methods.multitrajectory_gain import HeldoutEarlyStopping
from ue_framework.methods.multitrajectory_gain.gain_scale import compute_gain_scales_from_rows
from ue_framework.methods.multitrajectory_gain.gradient_diagnostics import gradient_conflict_diagnostics
from ue_framework.methods.multitrajectory_gain.objective import project_delta_linf
from ue_framework.methods.multitrajectory_gain.online_sampler import OnlineTrajectorySampler


def build_engine_v2(cfg: Dict, robust_scales: Dict[str, float], device: torch.device):
    engine = build_engine(cfg, int(cfg["rollout"].get("steps", 3)), device)
    obj = cfg["gain_objective_v2"]
    engine.objective_version = "v2"
    engine.robust_scales = {k: float(v) for k, v in robust_scales.items()}
    engine.protected_margin = float(obj.get("protected_margin", 0.10))
    engine.authorized_tolerance = float(obj.get("authorized_tolerance", 0.10))
    engine.shared_tolerance = float(obj.get("shared_tolerance", 0.10))
    engine.lambda_protected = float(obj.get("lambda_protected", 1.0))
    engine.lambda_authorized = float(obj.get("lambda_authorized", 2.0))
    engine.lambda_shared = float(obj.get("lambda_shared", 2.0))
    engine.lambda_regularization = float(obj.get("lambda_regularization", 1.0))
    engine.protected_support_min_batches = int(obj.get("protected_support_min_batches", 2))
    return engine


def evaluate(engine, sequences, delta: torch.Tensor, split: str, step: int):
    rows = []
    per_step = []
    for idx, sequence in enumerate(sequences):
        out = engine.run(sequence, delta.detach(), create_graph=False)
        row = row_from_output(split, step, idx, out, sequence)
        row["rejection_reason"] = rejection_reason(row)
        rows.append(row)
        per_step.append({"split": split, "outer_step": step, "trajectory_index": idx, "steps": out.per_step})
    return rows, summarize(rows), per_step


def rejection_reason(row: Dict) -> str:
    if float(row.get("protected_valid", 0.0)) <= 0.0:
        if float(row.get("protected_positive_count", 0.0)) <= 0.0:
            return "no_protected_query"
        if float(row.get("protected_support_batches", 0.0)) < 2.0:
            return "no_protected_support"
        if float(row.get("G_t_clean", 0.0)) <= 0.0:
            return "nonpositive_clean_gain"
        return "gain_below_threshold"
    if float(row.get("authorized_valid", 0.0)) <= 0.0 and float(row.get("authorized_positive_count", 0.0)) <= 0.0:
        return "no_authorized_query"
    if float(row.get("shared_valid", 0.0)) <= 0.0:
        return "shared_gain_too_small"
    return ""


def distribution_stats(rows: List[Dict], prefix: str) -> Dict[str, float]:
    values = np.array([float(row[prefix]) for row in rows if math.isfinite(float(row[prefix]))], dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "positive_gain_ratio": float((values > 0).mean()),
        "negative_gain_ratio": float((values < 0).mean()),
        "near_zero_gain_ratio": float((np.abs(values) < 1.0e-4).mean()),
    }


def gain_scale_audit(train_rows: List[Dict], heldout_rows: List[Dict], cfg: Dict) -> Dict:
    lg = cfg["learning_gain"]
    scales = compute_gain_scales_from_rows(
        train_rows,
        scale_quantile=float(lg.get("scale_quantile", 0.50)),
        min_quantile=float(lg.get("min_quantile", 0.20)),
        epsilon=float(lg.get("scale_epsilon", 1.0e-4)),
        absolute_floor=float(lg.get("absolute_floor", 1.0e-4)),
    )
    audit = {"robust_scales": scales.to_dict(), "train": {}, "heldout": {}}
    for name in ["G_t_clean", "G_t_clean_minus_poison", "G_a_clean", "G_a_poison_minus_clean", "G_s_clean", "G_s_poison_minus_clean", "denominator_t", "denominator_a", "denominator_s", "d_protected", "e_authorized", "e_shared"]:
        audit["train"][name] = distribution_stats(train_rows, name)
        audit["heldout"][name] = distribution_stats(heldout_rows, name)
    audit["answers"] = {
        "heldout_negative_D_mainly_small_denominator": bool(abs(audit["heldout"]["G_t_clean_minus_poison"]["mean"]) < 0.1 * abs(audit["heldout"]["d_protected"]["mean"]) if audit["heldout"].get("d_protected") else False),
        "raw_protected_gap_reversed": bool(audit["heldout"]["G_t_clean_minus_poison"]["mean"] < 0.0),
        "protected_valid_ratio_train": float(np.mean([float(row["protected_valid"]) for row in train_rows])),
        "protected_valid_ratio_heldout": float(np.mean([float(row["protected_valid"]) for row in heldout_rows])),
    }
    return audit


def optimize_fixed(engine, train_sequences, heldout_sequences, cfg, out_dir: Path, label: str):
    opt = cfg["optimization"]
    early = cfg["early_stopping"]
    steps = int(opt.get("outer_steps", 50))
    eval_interval = int(early.get("evaluation_interval", 5))
    eps = float(opt.get("eps", 16.0 / 255.0))
    imgsz = int(opt.get("imgsz", 160))
    delta = torch.zeros((1, 3, imgsz, imgsz), device=train_sequences[0].support_batches[0].images.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(opt.get("delta_learning_rate", 0.01)))
    stopper = HeldoutEarlyStopping(int(early.get("patience", 3)))
    curve, heldout_curve, train_rows_all, heldout_rows_all, grad_rows, matrices, memory = [], [], [], [], [], [], []
    fd = finite_difference_first_valid(engine, train_sequences, delta)
    for step in range(0, steps + 1):
        if step % eval_interval == 0:
            train_rows, train_summary, _ = evaluate(engine, train_sequences, delta, f"{label}_train", step)
            held_rows, held_summary, _ = evaluate(engine, heldout_sequences, f"{label}_heldout" and delta, f"{label}_heldout", step)
            score = held_summary["s_gain_mean"]
            stop = stopper.update(step, score, delta)
            curve.append({"experiment": label, "outer_step": step, **{f"train_{k}": v for k, v in train_summary.items()}, **{f"heldout_{k}": v for k, v in held_summary.items()}, "delta_linf": float(delta.detach().abs().max().item()), "best_step": stop.best_step, "best_score": stop.best_score})
            heldout_curve.append({"experiment": label, "outer_step": step, **held_summary, "best_step": stop.best_step, "best_score": stop.best_score})
            train_rows_all.extend(train_rows)
            heldout_rows_all.extend(held_rows)
            if step > 0 and stop.should_stop:
                break
        if step == steps:
            break
        sequence = train_sequences[step % len(train_sequences)]
        optimizer.zero_grad(set_to_none=True)
        out = engine.run(sequence, delta, create_graph=True)
        diag = gradient_conflict_diagnostics({"protected": out.metrics.protected_loss, "authorized": out.metrics.authorized_loss, "shared": out.metrics.shared_loss}, delta)
        for row in diag["rows"]:
            grad_rows.append({"experiment": label, "outer_step": step + 1, **row})
        matrices.append({"experiment": label, "outer_step": step + 1, **diag})
        out.loss.backward()
        optimizer.step()
        project_delta_linf(delta, eps)
        memory.append(memory_row(label, step + 1))
    if bool(early.get("restore_best", True)):
        stopper.restore_best(delta)
    torch.save(delta.detach().cpu(), out_dir / f"{label}_last_delta.pt")
    torch.save(stopper.best_delta.detach().cpu() if stopper.best_delta is not None else delta.detach().cpu(), out_dir / f"{label}_best_delta_by_heldout.pt")
    return {"delta": delta.detach(), "curve": curve, "heldout_curve": heldout_curve, "train_rows": train_rows_all, "heldout_rows": heldout_rows_all, "gradient_rows": grad_rows, "matrices": matrices, "memory": memory, "finite_difference": fd, "best": {"best_step": stopper.best_step, "best_score": stopper.best_score}}


def optimize_online(engine, train_batches, heldout_sequences, cfg, out_dir: Path, label: str):
    opt = cfg["optimization"]
    sampling = cfg["trajectory_sampling"]
    early = cfg["early_stopping"]
    steps = int(opt.get("outer_steps", 50))
    eval_interval = int(early.get("evaluation_interval", 5))
    eps = float(opt.get("eps", 16.0 / 255.0))
    imgsz = int(opt.get("imgsz", 160))
    heldout_batches = [seq.query_batch for seq in heldout_sequences]
    sampler = OnlineTrajectorySampler(
        train_batches,
        heldout_batches,
        support_steps=int(sampling.get("support_steps", 3)),
        seed=200,
        recent_image_exclusion_window=int(sampling.get("recent_image_exclusion_window", 20)),
        max_attempts=int(sampling.get("max_sampling_attempts", 30)),
    )
    delta = torch.zeros((1, 3, imgsz, imgsz), device=train_batches[0].images.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(opt.get("delta_learning_rate", 0.01)))
    stopper = HeldoutEarlyStopping(int(early.get("patience", 3)))
    curve, heldout_curve, train_rows_all, heldout_rows_all, grad_rows, matrices, memory, sampling_log, failures = [], [], [], [], [], [], [], [], []
    fd = None
    for step in range(0, steps + 1):
        if step % eval_interval == 0:
            held_rows, held_summary, _ = evaluate(engine, heldout_sequences, delta, f"{label}_heldout", step)
            stop = stopper.update(step, held_summary["s_gain_mean"], delta)
            heldout_curve.append({"experiment": label, "outer_step": step, **held_summary, "best_step": stop.best_step, "best_score": stop.best_score})
            curve.append({"experiment": label, "outer_step": step, **{f"heldout_{k}": v for k, v in held_summary.items()}, "delta_linf": float(delta.detach().abs().max().item()), "best_step": stop.best_step, "best_score": stop.best_score})
            heldout_rows_all.extend(held_rows)
            if step > 0 and stop.should_stop:
                break
        if step == steps:
            break
        accepted = None
        for attempt in range(int(sampling.get("max_sampling_attempts", 30))):
            sample = sampler.sample(step * 100 + attempt)
            if sample.sequence is None:
                sampling_log.append({"experiment": label, "outer_step": step + 1, "accepted": 0, "reason": sample.rejection_reason, "attempts": sample.attempts})
                continue
            probe = engine.run(sample.sequence, delta.detach(), create_graph=False)
            row = row_from_output(f"{label}_train", step + 1, 0, probe, sample.sequence)
            reason = rejection_reason(row)
            if reason:
                sampling_log.append({"experiment": label, "outer_step": step + 1, "accepted": 0, "reason": reason, "attempts": attempt + 1, "image_ids": "|".join(sum(sample.sequence.support_image_ids, []) + sample.sequence.query_image_ids)})
                continue
            accepted = sample.sequence
            sampling_log.append({"experiment": label, "outer_step": step + 1, "accepted": 1, "reason": "", "attempts": attempt + 1, "image_ids": "|".join(sum(sample.sequence.support_image_ids, []) + sample.sequence.query_image_ids)})
            break
        if accepted is None:
            failures.append({"outer_step": step + 1, "reason": "no_valid_online_trajectory"})
            continue
        if fd is None:
            fd = finite_difference(engine, accepted, delta)
        optimizer.zero_grad(set_to_none=True)
        out = engine.run(accepted, delta, create_graph=True)
        train_rows_all.append(row_from_output(f"{label}_train", step + 1, 0, out, accepted))
        diag = gradient_conflict_diagnostics({"protected": out.metrics.protected_loss, "authorized": out.metrics.authorized_loss, "shared": out.metrics.shared_loss}, delta)
        for row in diag["rows"]:
            grad_rows.append({"experiment": label, "outer_step": step + 1, **row})
        matrices.append({"experiment": label, "outer_step": step + 1, **diag})
        out.loss.backward()
        optimizer.step()
        project_delta_linf(delta, eps)
        memory.append(memory_row(label, step + 1))
    if bool(early.get("restore_best", True)):
        stopper.restore_best(delta)
    torch.save(delta.detach().cpu(), out_dir / "last_delta.pt")
    torch.save(stopper.best_delta.detach().cpu() if stopper.best_delta is not None else delta.detach().cpu(), out_dir / "best_delta_by_heldout.pt")
    return {"delta": delta.detach(), "curve": curve, "heldout_curve": heldout_curve, "train_rows": train_rows_all, "heldout_rows": heldout_rows_all, "gradient_rows": grad_rows, "matrices": matrices, "memory": memory, "sampling_log": sampling_log, "finite_difference": fd or {}, "failure_cases": failures, "best": {"best_step": stopper.best_step, "best_score": stopper.best_score}}


def finite_difference(engine, sequence, delta):
    base = delta.detach().clone().requires_grad_(True)
    out = engine.run(sequence, base, create_graph=True)
    grad = torch.autograd.grad(out.loss, base, allow_unused=False)[0]
    before = scalar(out.loss)
    best = {"loss_after": float("inf"), "step_size": 0.0, "ok": False}
    grad_norm = grad.detach().norm()
    directions = [("raw", grad)]
    if float(grad_norm.item()) > 0.0:
        directions.append(("unit", grad / grad_norm.clamp_min(1.0e-12)))
    for direction_name, direction in directions:
        for step_size in [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6]:
            nxt = (base - step_size * direction).detach().requires_grad_(True)
            out_next = engine.run(sequence, nxt, create_graph=True)
            after = scalar(out_next.loss)
            if after < best["loss_after"]:
                best = {
                    "loss_after": after,
                    "step_size": step_size,
                    "direction": direction_name,
                    "ok": bool(after < before and float(grad_norm.item()) > 0.0),
                }
            if best["ok"]:
                break
        if best["ok"]:
            break
    return {"loss_before": before, "loss_after": best["loss_after"], "step_size": best["step_size"], "direction": best.get("direction", ""), "gradient_norm": float(grad_norm.item()), "ok": best["ok"]}


def finite_difference_first_valid(engine, sequences, delta):
    best = None
    for sequence in sequences:
        result = finite_difference(engine, sequence, delta)
        if result["gradient_norm"] > 0.0:
            return result
        best = result
    return best or {}


def memory_row(label: str, step: int) -> Dict:
    if not torch.cuda.is_available():
        return {"experiment": label, "outer_step": step, "allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    return {"experiment": label, "outer_step": step, "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(), "max_allocated": torch.cuda.max_memory_allocated(), "max_reserved": torch.cuda.max_memory_reserved()}


def build_train_batches(cfg: Dict, heldout_ids: set[str], count: int, imgsz: int, device: torch.device) -> List:
    samples = read_voc_samples(ROOT / cfg["paths"]["voc_root"])
    groups = select_groups(samples, int(cfg["protected_class_id"]), float(cfg["support"].get("ambiguous_iou_threshold", 0.5)), max(count, 80))
    ordered = []
    for idx in range(max(len(v) for v in groups.values())):
        for name in ["person_only", "low_overlap", "high_overlap", "authorized_only"]:
            if idx < len(groups[name]) and groups[name][idx]["image_id"] not in heldout_ids:
                ordered.append(groups[name][idx])
            if len(ordered) >= count:
                break
        if len(ordered) >= count:
            break
    return [batch_from_sample(sample, imgsz, device, seed=300 + i) for i, sample in enumerate(ordered)]


def experiment_comparison_rows(v1_summary_path: Path, fixed_result: Dict, online_result: Dict) -> List[Dict]:
    rows = []
    if v1_summary_path.exists():
        v1 = json.loads(v1_summary_path.read_text(encoding="utf-8"))
        for key, label in [("j1", "A_J1_fixed_old"), ("j3", "B_J3_fixed_old")]:
            end = v1[key]["curve_rows"][-1]
            rows.append({"experiment": label, "heldout_delta_t": end["validation_d_protected_mean"], "heldout_abs_delta_a": end["validation_e_authorized_mean"], "heldout_abs_delta_s": end["validation_e_shared_mean"], "heldout_s_gain_v2": end["validation_s_gain_mean"]})
    for label, result in [("C_J3_fixed_v2", fixed_result), ("D_J3_online_v2", online_result)]:
        end = result["heldout_curve"][-1]
        rows.append({"experiment": label, "heldout_delta_t": end["d_protected_mean"], "heldout_abs_delta_a": end["e_authorized_mean"], "heldout_abs_delta_s": end["e_shared_mean"], "heldout_s_gain_v2": end["s_gain_mean"]})
    return rows


def write_report(path: Path, summary: Dict) -> None:
    c = summary["comparison"][-2]
    d = summary["comparison"][-1]
    audit = summary["gain_scale_audit"]["answers"]
    if d["heldout_s_gain_v2"] > c["heldout_s_gain_v2"] and d["heldout_delta_t"] > 0:
        conclusion = "A. online J3-v2 passes held-out proxy checks; proceed only to short AP-correlation proxy experiments."
    elif c["heldout_delta_t"] > 0 and d["heldout_delta_t"] <= 0:
        conclusion = "B. robust normalization helps numerics, but trajectory overfitting remains."
    elif d["heldout_delta_t"] > 0 and (d["heldout_abs_delta_a"] > 0.5 * d["heldout_delta_t"] or d["heldout_abs_delta_s"] > 0.5 * d["heldout_delta_t"]):
        conclusion = "C. online resampling helps protected gain, but authorized/shared gradient conflict remains severe."
    elif summary["online_valid_ratio"] < 0.5:
        conclusion = "D. protected clean gain is not stable enough for this surrogate/checkpoint."
    elif len(summary["comparison"]) >= 2 and d["heldout_s_gain_v2"] <= summary["comparison"][0]["heldout_s_gain_v2"]:
        conclusion = "E. J3 still does not beat J1; stop the multi-step route."
    else:
        conclusion = "F. implementation or data leakage still needs investigation."
    lines = [
        "# J3-v2 Trajectory Generalization and Gain-Scale Diagnosis",
        "",
        "## Root Cause Audit",
        f"- held-out negative D mainly small denominator: `{audit['heldout_negative_D_mainly_small_denominator']}`",
        f"- raw protected gain gap reversed: `{audit['raw_protected_gap_reversed']}`",
        f"- train protected-valid ratio: `{audit['protected_valid_ratio_train']}`",
        f"- held-out protected-valid ratio: `{audit['protected_valid_ratio_heldout']}`",
        "",
        "## Objective-v2",
        f"- robust scales: `{summary['robust_scales']}`",
        f"- fixed v2 held-out Delta_t/abs(Delta_a)/abs(Delta_s)/S: `{c['heldout_delta_t']}` / `{c['heldout_abs_delta_a']}` / `{c['heldout_abs_delta_s']}` / `{c['heldout_s_gain_v2']}`",
        f"- online v2 held-out Delta_t/abs(Delta_a)/abs(Delta_s)/S: `{d['heldout_delta_t']}` / `{d['heldout_abs_delta_a']}` / `{d['heldout_abs_delta_s']}` / `{d['heldout_s_gain_v2']}`",
        "",
        "## Online Resampling",
        f"- online accepted valid ratio: `{summary['online_valid_ratio']}`",
        f"- trajectory repetition rate: `{summary['trajectory_repetition_rate']}`",
        f"- online best held-out step/score: `{summary['online_best']}`",
        "",
        "## Engineering Checks",
        f"- finite difference: `{summary.get('finite_difference', {})}`",
        f"- parameter leak max: `{summary.get('parameter_leak_max', 0.0)}`",
        f"- memory allocated first/last: `{summary.get('memory_allocated_first_last', [])}`",
        "",
        "## Gradient Conflict",
        f"- mean protected-authorized cosine: `{summary['gradient_conflict']['protected_authorized_mean']}`",
        f"- mean protected-shared cosine: `{summary['gradient_conflict']['protected_shared_mean']}`",
        f"- mean authorized-shared cosine: `{summary['gradient_conflict']['authorized_shared_mean']}`",
        f"- mean gradient norms: `{summary['gradient_conflict']['mean_norms']}`",
        "",
        "## Decision",
        f"- `{conclusion}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def gradient_summary(matrices: List[Dict]) -> Dict:
    if not matrices:
        return {"protected_authorized_mean": 0.0, "protected_shared_mean": 0.0, "authorized_shared_mean": 0.0, "mean_norms": {}}
    pa = [m["matrix"][0][1] for m in matrices]
    ps = [m["matrix"][0][2] for m in matrices]
    a_s = [m["matrix"][1][2] for m in matrices]
    norms: Dict[str, List[float]] = {}
    for m in matrices:
        for row in m["rows"]:
            norms.setdefault(row["component"], []).append(row["gradient_norm"])
    return {"protected_authorized_mean": float(np.mean(pa)), "protected_shared_mean": float(np.mean(ps)), "authorized_shared_mean": float(np.mean(a_s)), "mean_norms": {k: float(np.mean(v)) for k, v in norms.items()}}


def repetition_rate(sampling_rows: List[Dict]) -> float:
    seen = set()
    repeated = 0
    total = 0
    for row in sampling_rows:
        if int(row.get("accepted", 0)) != 1:
            continue
        for image_id in str(row.get("image_ids", "")).split("|"):
            if not image_id:
                continue
            total += 1
            if image_id in seen:
                repeated += 1
            seen.add(image_id)
    return float(repeated / total) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/multitrajectory_gain/voc_yolov8n_j3_v2.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/j3_gain_v2"))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--train-trajectories", type=int, default=0)
    parser.add_argument("--heldout-trajectories", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if args.steps:
        cfg["optimization"]["outer_steps"] = int(args.steps)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = int(cfg["optimization"].get("imgsz", 160))
    train_n = int(args.train_trajectories or cfg["trajectory_sampling"].get("train_trajectories", 30))
    held_n = int(args.heldout_trajectories or cfg["trajectory_sampling"].get("heldout_trajectories", 30))

    started = time.time()
    train_sequences = build_sequences(cfg, train_n, 0, imgsz, device, seed=10)
    heldout_sequences = build_sequences(cfg, held_n, train_n + 25, imgsz, device, seed=110)

    audit_engine = build_engine(cfg, int(cfg["rollout"].get("steps", 3)), device)
    zero_delta = torch.zeros((1, 3, imgsz, imgsz), device=device, requires_grad=True)
    gain_train_rows, _, _ = evaluate(audit_engine, train_sequences, zero_delta, "gain_train", 0)
    gain_held_rows, _, _ = evaluate(audit_engine, heldout_sequences, zero_delta, "gain_heldout", 0)
    audit = gain_scale_audit(gain_train_rows, gain_held_rows, cfg)
    robust_scales = {k: audit["robust_scales"][k] for k in ["protected", "authorized", "shared"]}

    write_csv(out_dir / "gain_distribution_train.csv", gain_train_rows)
    write_csv(out_dir / "gain_distribution_heldout.csv", gain_held_rows)
    write_json(out_dir / "gain_scale_audit.json", audit)
    write_json(out_dir / "robust_scales.json", audit["robust_scales"])

    fixed_engine = build_engine_v2(cfg, robust_scales, device)
    fixed = optimize_fixed(fixed_engine, train_sequences, heldout_sequences, cfg, out_dir, "C_fixed_v2")

    heldout_ids = {image_id for seq in heldout_sequences for ids in seq.support_image_ids for image_id in ids} | {image_id for seq in heldout_sequences for image_id in seq.query_image_ids}
    train_batches = build_train_batches(cfg, heldout_ids, int(cfg["trajectory_sampling"].get("train_pool_batches", 160)), imgsz, device)
    online_engine = build_engine_v2(cfg, robust_scales, device)
    online = optimize_online(online_engine, train_batches, heldout_sequences, cfg, out_dir, "D_online_v2")

    curve = fixed["curve"] + online["curve"]
    heldout_curve = fixed["heldout_curve"] + online["heldout_curve"]
    train_rows = fixed["train_rows"] + online["train_rows"]
    heldout_rows = fixed["heldout_rows"] + online["heldout_rows"]
    gradient_rows = fixed["gradient_rows"] + online["gradient_rows"]
    matrices = fixed["matrices"] + online["matrices"]
    memory = fixed["memory"] + online["memory"]
    comparison = experiment_comparison_rows(ROOT / "outputs/j3_learning_gain_v1/summary.json", fixed, online)

    write_csv(out_dir / "optimization_curve.csv", curve)
    write_csv(out_dir / "heldout_curve.csv", heldout_curve)
    write_csv(out_dir / "trajectory_sampling_log.csv", online["sampling_log"])
    write_csv(out_dir / "outer_gradient_conflicts.csv", gradient_rows)
    write_json(out_dir / "outer_gradient_matrices.json", matrices)
    write_csv(out_dir / "experiment_comparison.csv", comparison)
    write_json(out_dir / "best_delta_metadata.json", {"fixed": fixed["best"], "online": online["best"]})
    write_csv(out_dir / "memory_profile.csv", memory)
    write_json(out_dir / "finite_difference_results.json", {"fixed": fixed["finite_difference"], "online": online["finite_difference"]})
    write_json(out_dir / "failure_cases.json", online["failure_cases"])

    accepted = [row for row in online["sampling_log"] if int(row.get("accepted", 0)) == 1]
    attempted_steps = {int(row["outer_step"]) for row in online["sampling_log"]}
    summary = {
        "elapsed_sec": time.time() - started,
        "robust_scales": audit["robust_scales"],
        "gain_scale_audit": audit,
        "comparison": comparison,
        "online_valid_ratio": float(len(accepted) / max(len(attempted_steps), 1)),
        "trajectory_repetition_rate": repetition_rate(online["sampling_log"]),
        "online_best": online["best"],
        "finite_difference": {"fixed": fixed["finite_difference"], "online": online["finite_difference"]},
        "gradient_conflict": gradient_summary(matrices),
        "parameter_leak_max": max([float(row.get("surrogate_parameter_max_abs_diff", 0.0)) for row in train_rows + heldout_rows], default=0.0),
        "memory_allocated_first_last": [memory[0]["allocated"] if memory else 0, memory[-1]["allocated"] if memory else 0],
    }
    write_json(out_dir / "summary.json", summary)
    write_report(ROOT / "docs/j3_gain_generalization_v2_report.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "report": str(ROOT / "docs/j3_gain_generalization_v2_report.md")}, indent=2))


if __name__ == "__main__":
    main()
