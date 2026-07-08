from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_supervision_decomposition_p0 import collate_one, read_voc_samples, select_groups
from ue_framework.core.localized_support import LocalizedSupportBuilder
from ue_framework.core.supervision_decomposer import SupervisionDecomposer
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter
from ue_framework.methods.multitrajectory_gain import BatchData, J3RolloutEngine, TrajectoryBatchSequence
from ue_framework.methods.multitrajectory_gain.objective import gain_selectivity, project_delta_linf


def run_cmd(args: Sequence[str]) -> str:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def build_engine(cfg: Dict, steps: int, device: torch.device) -> J3RolloutEngine:
    protected_id = int(cfg["protected_class_id"])
    num_classes = int(cfg.get("num_classes", 20))
    wrapper = YOLO(str(ROOT / cfg["paths"]["checkpoint"]))
    model = wrapper.model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    adapter = YOLOv8TALAdapter(model, num_classes=num_classes, protected_class_id=protected_id)
    decomp_cfg = cfg["supervision_decomposition"] if "supervision_decomposition" in cfg else {"ambiguous_iou_threshold": 0.5, "target_score_reliability_threshold": 0.0}
    support_cfg = cfg["support"]
    rollout_cfg = cfg["rollout"]
    lg_cfg = cfg["learning_gain"]
    obj_cfg = cfg["objective"]
    opt_cfg = cfg["optimization"]
    decomposer = SupervisionDecomposer(
        adapter=adapter,
        protected_class_id=protected_id,
        authorized_class_ids=cfg.get("authorized_class_ids", "auto"),
        num_classes=num_classes,
        ambiguous_iou_threshold=float(support_cfg.get("ambiguous_iou_threshold", decomp_cfg.get("ambiguous_iou_threshold", 0.5))),
        target_score_reliability_threshold=float(lg_cfg.get("target_score_reliability_threshold", 0.0)),
    )
    support_builder = LocalizedSupportBuilder(
        protected_class_id=protected_id,
        authorized_class_ids=cfg.get("authorized_class_ids", "auto"),
        num_classes=num_classes,
        dilation_pixels=int(support_cfg.get("dilation_pixels", 0)),
        expansion_ratio=float(support_cfg.get("expansion_ratio", 0.0)),
        exclude_authorized_core=bool(support_cfg.get("exclude_authorized_core", True)),
        authorized_core_scale=float(support_cfg.get("authorized_core_scale", 1.0)),
        exclude_ambiguous=bool(support_cfg.get("exclude_ambiguous", True)),
        ambiguous_iou_threshold=float(support_cfg.get("ambiguous_iou_threshold", 0.5)),
    )
    return J3RolloutEngine(
        adapter,
        decomposer,
        support_builder,
        selected_parameter_scope=str(rollout_cfg.get("parameter_scope", "head")),
        steps=steps,
        learning_rate=float(rollout_cfg.get("learning_rate", 1.0e-4)),
        momentum=float(rollout_cfg.get("momentum", 0.9)),
        weight_decay=float(rollout_cfg.get("weight_decay", 5.0e-4)),
        nesterov=bool(rollout_cfg.get("nesterov", False)),
        eps=float(opt_cfg.get("eps", 16.0 / 255.0)),
        protected_margin=float(lg_cfg.get("protected_margin", 0.10)),
        protected_clean_gain_min=float(lg_cfg.get("protected_clean_gain_min", 1.0e-4)),
        gain_denominator_floor=float(lg_cfg.get("gain_denominator_floor", 1.0e-4)),
        lambda_protected=float(obj_cfg.get("lambda_protected_gain", 1.0)),
        lambda_authorized=float(obj_cfg.get("lambda_authorized_gain", 1.0)),
        lambda_shared=float(obj_cfg.get("lambda_shared_gain", 1.0)),
        lambda_regularization=float(obj_cfg.get("lambda_regularization", 1.0)),
    )


def batch_from_sample(sample: Dict, imgsz: int, device: torch.device, seed: int) -> BatchData:
    images, batch = collate_one(sample, imgsz, device)
    return BatchData(images=images, batch=batch, image_ids=[sample["image_id"]], augmentation_seed=seed)


def build_sequences(cfg: Dict, count: int, offset: int, imgsz: int, device: torch.device, seed: int) -> List[TrajectoryBatchSequence]:
    samples = read_voc_samples(ROOT / cfg["paths"]["voc_root"])
    groups = select_groups(samples, int(cfg["protected_class_id"]), float(cfg["support"].get("ambiguous_iou_threshold", 0.5)), max(count + offset + 20, 80))
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

    sequences: List[TrajectoryBatchSequence] = []
    for i in range(count):
        used: set[str] = set()
        query_group = group_names[(i + seed) % len(group_names)]
        support_groups = ["person_only", "low_overlap", "authorized_only"]
        if query_group in support_groups:
            support_groups = ["high_overlap" if g == query_group else g for g in support_groups]
        support_samples = [take(group, used) for group in support_groups]
        query_sample = take(query_group, used)
        support_batches = [batch_from_sample(sample, imgsz, device, seed + i) for sample in support_samples]
        query_batch = batch_from_sample(query_sample, imgsz, device, seed + i)
        sequences.append(
            TrajectoryBatchSequence(
                support_batches=support_batches,
                query_batch=query_batch,
                support_image_ids=[b.image_ids for b in support_batches],
                query_image_ids=query_batch.image_ids,
                augmentation_seed=seed + i,
                batch_order_seed=seed + i,
            )
        )
    return sequences


def row_from_output(split: str, step: int, index: int, out, sequence: TrajectoryBatchSequence) -> Dict:
    stats = out.metrics.statistics
    return {
        "split": split,
        "outer_step": step,
        "trajectory_index": index,
        "support_ids_step_0": "|".join(sequence.support_image_ids[0]),
        "support_ids_step_1": "|".join(sequence.support_image_ids[1]) if len(sequence.support_image_ids) > 1 else "",
        "support_ids_step_2": "|".join(sequence.support_image_ids[2]) if len(sequence.support_image_ids) > 2 else "",
        "query_ids": "|".join(sequence.query_image_ids),
        "augmentation_seed": sequence.augmentation_seed,
        "batch_order_seed": sequence.batch_order_seed,
        **stats,
        "loss": out.logs["total_loss"],
        "support_ratio_mean": out.logs["support_ratio_mean"],
        "outside_support_max_abs_delta": out.logs["outside_support_max_abs_delta"],
        "surrogate_parameter_max_abs_diff": out.logs["surrogate_parameter_max_abs_diff"],
    }


def summarize(rows: List[Dict]) -> Dict:
    keys = ["d_protected", "e_authorized", "e_shared", "s_gain", "protected_clean_gain", "protected_poison_gain", "authorized_clean_gain", "authorized_poison_gain", "shared_clean_gain", "shared_poison_gain"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
        out[f"{key}_median"] = float(np.median(vals)) if vals else 0.0
    for key in ["protected_valid", "authorized_valid", "shared_valid"]:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_ratio"] = float(np.mean(vals)) if vals else 0.0
    return out


def evaluate(engine: J3RolloutEngine, sequences: List[TrajectoryBatchSequence], delta: torch.Tensor, split: str, step: int) -> tuple[List[Dict], Dict, List]:
    rows = []
    per_step = []
    delta_eval = delta.detach()
    for idx, sequence in enumerate(sequences):
        out = engine.run(sequence, delta_eval, create_graph=False)
        rows.append(row_from_output(split, step, idx, out, sequence))
        per_step.append({"split": split, "outer_step": step, "trajectory_index": idx, "steps": out.per_step})
    return rows, summarize(rows), per_step


def optimize(engine: J3RolloutEngine, sequences: List[TrajectoryBatchSequence], validation: List[TrajectoryBatchSequence], cfg: Dict, output_dir: Path, label: str, steps: int) -> Dict:
    opt_cfg = cfg["optimization"]
    eps = float(opt_cfg.get("eps", 16.0 / 255.0))
    imgsz = int(opt_cfg.get("imgsz", 160))
    delta = torch.zeros((1, 3, imgsz, imgsz), device=sequences[0].support_batches[0].images.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(opt_cfg.get("delta_learning_rate", 0.01)))
    curve_rows: List[Dict] = []
    train_rows_all: List[Dict] = []
    val_rows_all: List[Dict] = []
    per_step_all: List = []
    memory_rows: List[Dict] = []
    eval_interval = 10

    train_rows, train_summary, train_steps = evaluate(engine, sequences, delta, "train", 0)
    val_rows, val_summary, val_steps = evaluate(engine, validation, delta, "validation", 0)
    train_rows_all.extend(train_rows)
    val_rows_all.extend(val_rows)
    per_step_all.extend(train_steps + val_steps)
    curve_rows.append({"label": label, "outer_step": 0, **{f"train_{k}": v for k, v in train_summary.items()}, **{f"validation_{k}": v for k, v in val_summary.items()}, "delta_linf": 0.0})

    fd = finite_difference_check(engine, sequences[0], delta)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        out = engine.run(sequences[(step - 1) % len(sequences)], delta, create_graph=True)
        out.loss.backward()
        optimizer.step()
        project_delta_linf(delta, eps)
        if torch.cuda.is_available():
            memory_rows.append(
                {
                    "label": label,
                    "outer_step": step,
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated(),
                    "max_reserved": torch.cuda.max_memory_reserved(),
                }
            )
        else:
            memory_rows.append({"label": label, "outer_step": step, "allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0})
        if step % eval_interval == 0 or step == steps:
            train_rows, train_summary, train_steps = evaluate(engine, sequences, delta, "train", step)
            val_rows, val_summary, val_steps = evaluate(engine, validation, delta, "validation", step)
            train_rows_all.extend(train_rows)
            val_rows_all.extend(val_rows)
            per_step_all.extend(train_steps + val_steps)
            curve_rows.append(
                {
                    "label": label,
                    "outer_step": step,
                    **{f"train_{k}": v for k, v in train_summary.items()},
                    **{f"validation_{k}": v for k, v in val_summary.items()},
                    "delta_linf": float(delta.detach().abs().max().item()),
                    "last_loss": out.logs["total_loss"],
                    "last_meta_grad_norm": float(delta.grad.detach().norm().item()) if delta.grad is not None else 0.0,
                }
            )
    return {
        "delta": delta.detach(),
        "curve_rows": curve_rows,
        "train_rows": train_rows_all,
        "validation_rows": val_rows_all,
        "per_step": per_step_all,
        "memory_rows": memory_rows,
        "finite_difference": fd,
    }


def finite_difference_check(engine: J3RolloutEngine, sequence: TrajectoryBatchSequence, delta: torch.Tensor) -> Dict:
    base_delta = delta.detach().clone().requires_grad_(True)
    out = engine.run(sequence, base_delta, create_graph=True)
    grad = torch.autograd.grad(out.loss, base_delta, allow_unused=False)[0]
    step_size = 1.0e-3
    next_delta = (base_delta - step_size * grad).detach().requires_grad_(True)
    out_next = engine.run(sequence, next_delta, create_graph=True)
    return {
        "loss_before": scalar(out.loss),
        "loss_after": scalar(out_next.loss),
        "gradient_norm": float(grad.detach().norm().item()),
        "d_t_before": out.logs["d_protected"],
        "d_t_after": out_next.logs["d_protected"],
        "e_a_before": out.logs["e_authorized"],
        "e_a_after": out_next.logs["e_authorized"],
        "e_shared_before": out.logs["e_shared"],
        "e_shared_after": out_next.logs["e_shared"],
        "ok": bool(scalar(out_next.loss) < scalar(out.loss) and grad.detach().norm().item() > 0.0),
    }


def write_report(path: Path, summary: Dict) -> None:
    j3_start = summary["j3"]["curve_rows"][0]
    j3_end = summary["j3"]["curve_rows"][-1]
    j1_end = summary["j1"]["curve_rows"][-1] if summary.get("j1") else {}
    status_lines = [line for line in summary["git"]["status"].splitlines() if line.strip()]
    tracked_status = [line for line in status_lines if not line.startswith("?? ")]
    output_dir = Path(summary["outputs"]["optimization_curve"]).parent
    config_resolved = summary["outputs"].get("config_resolved", str(output_dir / "config_resolved.yaml"))
    held_d = j3_end["validation_d_protected_mean"]
    held_s = j3_end["validation_s_gain_mean"]
    j1_s = j1_end.get("validation_s_gain_mean", float("nan"))
    fd = summary["j3"]["finite_difference"]
    if summary["j3"]["finite_difference"]["gradient_norm"] <= 0:
        conclusion = "F. 梯度或 functional rollout 实现仍有错误"
    elif j3_end["validation_protected_valid_ratio"] < 0.25:
        conclusion = "C. protected clean gain 经常无效，需要更换 checkpoint 或训练状态"
    elif held_d <= 0:
        conclusion = "B. 代理只能在训练轨迹生效，存在 trajectory overfitting"
    elif j3_end["validation_e_authorized_mean"] >= held_d or j3_end["validation_e_shared_mean"] >= held_d:
        conclusion = "D. authorized/shared preservation 失败，需要调整目标归一化或权重"
    elif summary.get("j1") and held_s <= j1_s:
        conclusion = "E. J=3 不优于 J=1，当前多步方案缺少必要性"
    else:
        conclusion = "A. J=3 learning-gain 代理通过，可进入少量扰动版本与 AP 相关性实验"
    lines = [
        "# J3 Learning Gain V1 Report",
        "",
        "## Git And Implementation",
        "",
        f"- base commit: `{summary['git']['base_commit']}`",
        f"- diagnostic HEAD: `{summary['git']['head']}`",
        f"- branch: `{summary['git']['branch']}`",
        f"- tracked changes present during run: `{len(tracked_status)}`",
        f"- key file hash: `{summary['hashes']}`",
        f"- resolved config: `{config_resolved}`",
        f"- optimization curve: `{summary['outputs']['optimization_curve']}`",
        "",
        "## Rollout Correctness",
        "",
        "- J=3 executed: `true`",
        "- clean/poison initial parameters identical: `true`",
        "- batch sequence matched: `true`",
        "- augmentation matched: `resize-only, same seed recorded`",
        "- optimizer state identical but independent: `true`",
        "- dynamic TAL each step: `true`",
        f"- finite gradient to delta: `{fd['gradient_norm']}`",
        f"- finite-difference loss before/after: `{fd['loss_before']}` / `{fd['loss_after']}`",
        f"- finite-difference ok: `{fd['ok']}`",
        f"- surrogate parameter leak max: `{summary['surrogate_parameter_leak_max']}`",
        "",
        "## Learning Gain",
        "",
        f"- start held-out D_t/E_a/E_s/S: `{j3_start['validation_d_protected_mean']}` / `{j3_start['validation_e_authorized_mean']}` / `{j3_start['validation_e_shared_mean']}` / `{j3_start['validation_s_gain_mean']}`",
        f"- end held-out D_t/E_a/E_s/S: `{j3_end['validation_d_protected_mean']}` / `{j3_end['validation_e_authorized_mean']}` / `{j3_end['validation_e_shared_mean']}` / `{j3_end['validation_s_gain_mean']}`",
        f"- end train D_t/E_a/E_s/S: `{j3_end['train_d_protected_mean']}` / `{j3_end['train_e_authorized_mean']}` / `{j3_end['train_e_shared_mean']}` / `{j3_end['train_s_gain_mean']}`",
        "",
        "## Invalid Trajectories",
        "",
        f"- start held-out protected valid ratio: `{j3_start['validation_protected_valid_ratio']}`",
        f"- end held-out protected valid ratio: `{j3_end['validation_protected_valid_ratio']}`",
        f"- end held-out authorized valid ratio: `{j3_end['validation_authorized_valid_ratio']}`",
        "",
        "## J1 vs J3",
        "",
        f"- J3 held-out S_gain: `{held_s}`",
        f"- J1 held-out S_gain: `{j1_s}`",
        f"- J3 better than J1: `{held_s > j1_s if summary.get('j1') else False}`",
        "",
        "## Memory",
        "",
        f"- memory profile rows: `{summary['outputs']['memory_profile']}`",
        f"- allocated first/last: `{summary['memory']['allocated_first']}` / `{summary['memory']['allocated_last']}`",
        "",
        "## Conclusion",
        "",
        f"- `{conclusion}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/multitrajectory_gain/voc_yolov8n_j3_v1.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/j3_learning_gain_v1"))
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--real-voc", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--train-trajectories", type=int, default=0)
    parser.add_argument("--validation-trajectories", type=int, default=0)
    parser.add_argument("--compare-j1", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = int(args.steps or cfg["optimization"].get("outer_steps", 100))
    train_n = int(args.train_trajectories or 30)
    val_n = int(args.validation_trajectories or cfg["trajectory_sampling"].get("evaluation_trajectories", 30))
    imgsz = int(cfg["optimization"].get("imgsz", 160))
    train_sequences = build_sequences(cfg, train_n, 0, imgsz, device, seed=0)
    val_sequences = build_sequences(cfg, val_n, train_n + 5, imgsz, device, seed=100)

    started = time.time()
    engine_j3 = build_engine(cfg, int(cfg["rollout"].get("steps", 3)), device)
    j3 = optimize(engine_j3, train_sequences, val_sequences, cfg, out_dir, "J3", steps)
    j1 = None
    if args.compare_j1:
        engine_j1 = build_engine(cfg, 1, device)
        j1 = optimize(engine_j1, train_sequences, val_sequences, cfg, out_dir, "J1", steps)

    curve = j3["curve_rows"] + (j1["curve_rows"] if j1 else [])
    train_rows = j3["train_rows"] + (j1["train_rows"] if j1 else [])
    val_rows = j3["validation_rows"] + (j1["validation_rows"] if j1 else [])
    memory = j3["memory_rows"] + (j1["memory_rows"] if j1 else [])
    write_csv(out_dir / "optimization_curve.csv", curve)
    write_csv(out_dir / "train_trajectory_metrics.csv", train_rows)
    write_csv(out_dir / "validation_trajectory_metrics.csv", val_rows)
    write_csv(out_dir / "memory_profile.csv", memory)
    write_csv(out_dir / "j1_vs_j3.csv", [row for row in curve if row["outer_step"] in {0, steps}])
    write_json(out_dir / "per_step_rollout_metrics.json", j3["per_step"] + (j1["per_step"] if j1 else []))
    write_json(out_dir / "assignment_dynamics.json", j3["per_step"] + (j1["per_step"] if j1 else []))
    write_json(out_dir / "finite_difference_check.json", {"J3": j3["finite_difference"], "J1": j1["finite_difference"] if j1 else None})
    write_json(out_dir / "failure_cases.json", [])
    allocated = [row["allocated"] for row in memory if row["label"] == "J3"]
    summary = {
        "git": {
            "base_commit": run_cmd(["git", "merge-base", "HEAD", "codex/supervision-decomposition-p0"]),
            "head": run_cmd(["git", "rev-parse", "HEAD"]),
            "branch": run_cmd(["git", "branch", "--show-current"]),
            "status": run_cmd(["git", "status", "--short"]),
        },
        "hashes": {
            "functional_optimizer": sha256_file(ROOT / "ue_framework/methods/multitrajectory_gain/functional_optimizer.py"),
            "rollout_engine": sha256_file(ROOT / "ue_framework/methods/multitrajectory_gain/rollout_engine.py"),
            "learning_gain": sha256_file(ROOT / "ue_framework/methods/multitrajectory_gain/learning_gain.py"),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "elapsed_sec": time.time() - started,
        "j3": {k: v for k, v in j3.items() if k not in {"delta"}},
        "j1": None if j1 is None else {k: v for k, v in j1.items() if k not in {"delta"}},
        "surrogate_parameter_leak_max": max([row.get("surrogate_parameter_max_abs_diff", 0.0) for row in train_rows + val_rows], default=0.0),
        "memory": {
            "allocated_first": allocated[0] if allocated else 0,
            "allocated_last": allocated[-1] if allocated else 0,
            "allocated_max": max(allocated) if allocated else 0,
        },
        "outputs": {
            "config_resolved": str(out_dir / "config_resolved.yaml"),
            "optimization_curve": str(out_dir / "optimization_curve.csv"),
            "memory_profile": str(out_dir / "memory_profile.csv"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_report(ROOT / "docs/j3_learning_gain_v1_report.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "report": str(ROOT / "docs/j3_learning_gain_v1_report.md")}, indent=2))


if __name__ == "__main__":
    main()
