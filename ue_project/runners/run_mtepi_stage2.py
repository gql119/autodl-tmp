from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_supervision_decomposition_p0 import read_voc_samples, select_groups
from ue_framework.methods.mtepi import ChannelScoreThresholds, build_checkpoint_manifest, stage2_gate


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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_pathway_manifests(cfg: Dict, output_dir: Path) -> Dict:
    voc_root = ROOT / cfg["paths"]["voc_root"]
    samples = read_voc_samples(voc_root)
    data_cfg = cfg["data_manifest"]
    protected = int(cfg["protected_class_id"])
    groups = select_groups(
        samples,
        protected,
        float(data_cfg.get("ambiguous_iou_threshold", 0.5)),
        max(int(data_cfg.get("calibration_count", 240)), int(data_cfg.get("validation_count", 240))),
    )
    ordered = groups["person_only"] + groups["low_overlap"] + groups["high_overlap"] + groups["authorized_only"]
    seen = set()
    unique = []
    for sample in ordered:
        if sample["image_id"] not in seen:
            seen.add(sample["image_id"])
            unique.append(sample)
    cal_n = int(data_cfg.get("calibration_count", 240))
    val_n = int(data_cfg.get("validation_count", 240))
    calibration = unique[:cal_n]
    validation = unique[cal_n : cal_n + val_n]

    cfg_dir = ROOT / "configs/pathway_analysis"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "voc_pathway_calibration.txt").write_text(
        "\n".join(s["image_id"] for s in calibration) + ("\n" if calibration else ""),
        encoding="utf-8",
    )
    (cfg_dir / "voc_pathway_validation.txt").write_text(
        "\n".join(s["image_id"] for s in validation) + ("\n" if validation else ""),
        encoding="utf-8",
    )
    metadata = {
        "voc_root": str(voc_root),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "overlap_count": len({s["image_id"] for s in calibration} & {s["image_id"] for s in validation}),
        "group_counts": {k: len(v) for k, v in groups.items()},
    }
    write_json(cfg_dir / "voc_pathway_metadata.json", metadata)
    write_json(output_dir / "pathway_data_manifest.json", metadata)
    return metadata


def register_candidate_layers(cfg: Dict, checkpoint_manifest: Dict, output_dir: Path) -> List[Dict]:
    valid = [c for c in checkpoint_manifest["checkpoints"] if c["exists"] and c["class_space"] == "VOC20"]
    if not valid:
        write_json(output_dir / "layer_registry.json", [])
        return []
    try:
        from ultralytics import YOLO
    except Exception as exc:
        write_json(output_dir / "layer_registry.json", {"status": "not_run", "reason": f"ultralytics import failed: {exc}"})
        return []

    checkpoint_path = ROOT / valid[0]["path"]
    wrapper = YOLO(str(checkpoint_path))
    model = wrapper.model.eval()
    imgsz = int(cfg["layer_registry"].get("imgsz", 160))
    captured: Dict[str, Dict] = {}
    handles = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name not in {"Conv", "C2f", "SPPF", "Detect"}:
            continue

        def _hook(_module, _inputs, output, key=name, cls_name=class_name):
            if torch.is_tensor(output) and output.ndim == 4:
                captured[key] = {"module_path": key, "module_class": cls_name, "shape": list(output.shape)}

        handles.append(module.register_forward_hook(_hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    for handle in handles:
        handle.remove()

    rows = []
    by_stride = {}
    for item in captured.values():
        shape = item["shape"]
        _, channels, h, w = shape
        if h < 2 or w < 2:
            continue
        stride = max(1, imgsz // max(h, w))
        by_stride.setdefault(stride, item)
    role_targets = [("P3", 8), ("P4", 16), ("P5", 32)]
    for role, target_stride in role_targets[: int(cfg["layer_registry"].get("max_candidate_layers", 3))]:
        if target_stride not in by_stride:
            continue
        item = by_stride[target_stride]
        _, channels, h, w = item["shape"]
        rows.append(
            {
                "name": role,
                "module_path": item["module_path"],
                "module_class": item["module_class"],
                "output_shape": item["shape"],
                "channel_count": int(channels),
                "stride": int(target_stride),
                "hook_position": "module_output_after_bn_activation",
                "after_bn": True,
                "after_activation": True,
                "shared_downstream": True,
            }
        )
    write_json(output_dir / "layer_registry.json", rows)
    return rows


def write_stage2_report(path: Path, summary: Dict) -> None:
    layer_rows = summary["layer_registry"] if isinstance(summary["layer_registry"], list) else []
    lines = [
        "# MTEPI Stage 2 Report",
        "",
        "## Executive Decision",
        "- Person-selective functional channels were not established in this run.",
        "- The blocking issue is checkpoint legality: only a late VOC20 surrogate checkpoint is available, with no same-run early/middle checkpoints or shared initialization/training-manifest metadata.",
        "- Stage 3 was not implemented or run because Stage 2 failed the hard gate.",
        "",
        "## Gate",
        f"- STAGE_2_GATE: `{summary['stage2_gate']['gate']}`",
        f"- reasons: `{summary['stage2_gate']['reasons']}`",
        "",
        "## Checkpoints",
        f"- legal same-trajectory checkpoints: `{summary['checkpoint_manifest']['legal_same_trajectory']}`",
        f"- valid checkpoint count: `{summary['checkpoint_manifest']['valid_checkpoint_count']}`",
        f"- roles present: `{summary['checkpoint_manifest']['roles_present']}`",
        "",
        "## Layer Registry",
        f"- candidate layer count: `{len(layer_rows)}`",
        f"- registered layers: `{[(r.get('name'), r.get('module_path'), r.get('output_shape'), r.get('stride')) for r in layer_rows]}`",
        "",
        "## Required Stage 2 Answers",
        "1. person-selective functional channels: `not established`.",
        "2. local ROI ablation vs global Top-k ablation: `not evaluated`; checkpoint hard gate failed before functional scoring.",
        "3. most selective layer: `not determined`.",
        "4. Top 1/5/10/20% AP effect: `not run`.",
        "5. random/activation/gradient baseline comparison: `not run`.",
        "6. same-trajectory checkpoint index stability: `not valid to compute`.",
        "7. cross-checkpoint functional transfer: `not valid to compute`.",
        "8. bootstrap confidence interval support: `not available`.",
        "9. legal consensus pathway: `not formed`.",
        f"10. STAGE_2_GATE: `{summary['stage2_gate']['gate']}`.",
        "",
        "## Stage 3",
        "- Not run. Stage 2 failed the legal checkpoint hard gate, so no perturbation store, delta, poisoned dataset, ARPS loss, or victim training was created.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/mtepi/voc_yolov8n_stage2.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/mtepi_stage2"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    checkpoint_manifest = build_checkpoint_manifest(cfg["paths"]["checkpoint_candidates"], ROOT)
    write_json(out_dir / "checkpoint_manifest.json", checkpoint_manifest)
    data_manifest = build_pathway_manifests(cfg, out_dir)
    layer_registry = register_candidate_layers(cfg, checkpoint_manifest, out_dir)

    thresholds = ChannelScoreThresholds(**cfg["channel_scoring"])
    ranked_rows: List[Dict] = []
    topk_ap_rows: List[Dict] = []
    transfer_rows: List[Dict] = []
    bootstrap_rows: List[Dict] = []
    consensus: List[Dict] = []
    write_csv(out_dir / "localized_channel_scores.csv", ranked_rows)
    write_csv(out_dir / "constraint_first_ranking.csv", ranked_rows)
    write_csv(out_dir / "baseline_channel_scores.csv", [])
    write_csv(out_dir / "topk_ablation_ap_curve.csv", topk_ap_rows)
    write_csv(out_dir / "checkpoint_pathway_overlap.csv", transfer_rows)
    write_csv(out_dir / "cross_checkpoint_transfer.csv", transfer_rows)
    write_csv(out_dir / "bootstrap_confidence_intervals.csv", bootstrap_rows)
    write_json(out_dir / "consensus_pathways.json", consensus)
    write_json(out_dir / "frozen_consensus_pathways.json", {"status": "not_created", "reason": "STAGE_2_GATE failed"})

    gate_cfg = cfg["stage2_gate"]
    gate = stage2_gate(
        checkpoint_manifest,
        ranked_rows,
        topk_ap_rows,
        transfer_rows,
        bootstrap_rows,
        consensus,
        min_target_selective_channels=int(gate_cfg.get("min_target_selective_channels", 1)),
        min_authorized_retention=float(cfg["topk_ablation"].get("min_authorized_retention", 0.9)),
        min_topk_protected_drop=float(cfg["topk_ablation"].get("min_protected_ap_drop", 0.1)),
        min_transfer_jaccard=float(cfg["transfer"].get("min_jaccard", 0.5)),
    )
    summary = {
        "stage2_gate": gate,
        "checkpoint_manifest": checkpoint_manifest,
        "data_manifest": data_manifest,
        "layer_registry": layer_registry,
        "thresholds": cfg["channel_scoring"],
        "arps_stage3": {"status": "not_run", "reason": "STAGE_2_GATE failed"},
        "perturbation_outputs_created": False,
    }
    write_json(out_dir / "stage2_gate.json", gate)
    write_json(out_dir / "summary.json", summary)
    write_stage2_report(ROOT / "docs/mtepi_stage2_report.md", summary)
    print(json.dumps({"stage2_gate": gate["gate"], "summary": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
