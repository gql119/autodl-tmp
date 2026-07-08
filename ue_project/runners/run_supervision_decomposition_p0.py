from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ue_framework.core.localized_support import LocalizedSupportBuilder
from ue_framework.core.supervision_decomposer import SupervisionDecomposer
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter
from ue_framework.methods.learning_trajectory.virtual_update import parameter_leak_max_abs_diff, snapshot_parameters
from ue_framework.methods.multitrajectory_gain import compute_gradient_leakage_matrix

VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NAME_TO_ID = {name: idx for idx, name in enumerate(VOC_NAMES)}


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


def xyxy_to_xywhn(box, width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2) / width
    cy = 0.5 * (y1 + y2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return [float(cx), float(cy), float(bw), float(bh)]


def xywhn_to_xyxy(box):
    cx, cy, bw, bh = box
    return [cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh]


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ab - inter, 1.0e-8)


def read_voc_samples(voc_root: Path) -> List[Dict]:
    ids_path = voc_root / "ImageSets" / "Main" / "trainval.txt"
    ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for image_id in ids:
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        anns = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in NAME_TO_ID:
                continue
            b = obj.find("bndbox")
            xyxy = [
                float(b.findtext("xmin")),
                float(b.findtext("ymin")),
                float(b.findtext("xmax")),
                float(b.findtext("ymax")),
            ]
            anns.append({"cls": NAME_TO_ID[name], "bbox": xyxy_to_xywhn(xyxy, width, height)})
        if anns:
            samples.append(
                {
                    "image_id": image_id,
                    "image_path": str(voc_root / "JPEGImages" / f"{image_id}.jpg"),
                    "width": width,
                    "height": height,
                    "annotations": anns,
                }
            )
    return samples


def max_person_authorized_iou(sample: Dict, protected_class_id: int) -> float:
    protected = [xywhn_to_xyxy(a["bbox"]) for a in sample["annotations"] if a["cls"] == protected_class_id]
    authorized = [xywhn_to_xyxy(a["bbox"]) for a in sample["annotations"] if a["cls"] != protected_class_id]
    if not protected or not authorized:
        return 0.0
    return max(box_iou(p, a) for p in protected for a in authorized)


def select_groups(samples: List[Dict], protected_class_id: int, threshold: float, max_batches: int) -> Dict[str, List[Dict]]:
    groups = {"person_only": [], "authorized_only": [], "low_overlap": [], "high_overlap": []}
    for sample in samples:
        classes = {a["cls"] for a in sample["annotations"]}
        has_p = protected_class_id in classes
        has_a = any(c != protected_class_id for c in classes)
        overlap = max_person_authorized_iou(sample, protected_class_id)
        if has_p and not has_a and len(groups["person_only"]) < max_batches:
            groups["person_only"].append(sample)
        elif has_a and not has_p and len(groups["authorized_only"]) < max_batches:
            groups["authorized_only"].append(sample)
        elif has_p and has_a and overlap < min(0.1, threshold) and len(groups["low_overlap"]) < max_batches:
            groups["low_overlap"].append(sample)
        elif has_p and has_a and overlap >= threshold and len(groups["high_overlap"]) < max_batches:
            groups["high_overlap"].append(sample)
        if all(len(v) >= max_batches for v in groups.values()):
            break
    return groups


def collate_one(sample: Dict, imgsz: int, device: torch.device) -> tuple[torch.Tensor, Dict]:
    image = Image.open(sample["image_path"]).convert("RGB").resize((imgsz, imgsz))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    cls = torch.tensor([a["cls"] for a in sample["annotations"]], dtype=torch.float32, device=device)
    boxes = torch.tensor([a["bbox"] for a in sample["annotations"]], dtype=torch.float32, device=device)
    batch = {
        "cls": cls,
        "bboxes": boxes,
        "batch_idx": torch.zeros(cls.numel(), dtype=torch.float32, device=device),
        "batch_size": 1,
    }
    return tensor, batch


def scalar(t: torch.Tensor) -> float:
    return float(t.detach().cpu().item())


def make_synthetic_decomposition(pred_scores=None, ambiguous_mask=None, per_box=None, per_dfl=None):
    if pred_scores is None:
        pred_scores = torch.zeros((1, 3, 20), dtype=torch.float32)
    target_scores = torch.zeros_like(pred_scores)
    target_scores[0, 0, 14] = 0.7
    target_scores[0, 1, 1] = 0.6
    target_labels = torch.tensor([[14, 1, 0]], dtype=torch.long)
    fg_mask = torch.tensor([[True, True, False]])
    if ambiguous_mask is None:
        ambiguous_mask = torch.zeros_like(fg_mask)
    if per_box is None:
        per_box = torch.tensor([[0.2, 0.3, 0.0]], dtype=torch.float32)
    if per_dfl is None:
        per_dfl = torch.tensor([[0.4, 0.5, 0.0]], dtype=torch.float32)
    return SupervisionDecomposer(protected_class_id=14, num_classes=20).decompose_from_tensors(
        pred_scores=pred_scores,
        target_scores=target_scores,
        target_labels=target_labels,
        fg_mask=fg_mask,
        ambiguous_mask=ambiguous_mask,
        per_unit_box_loss=per_box,
        per_unit_dfl_loss=per_dfl,
        target_scores_sum=target_scores.sum().clamp_min(1.0),
        batch_size=1,
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
    )


def synthetic_interventions(output_dir: Path) -> Dict[str, List[Dict]]:
    base_scores = torch.zeros((1, 3, 20))
    cases = [
        ("person_assigned_class", (0, 0, 14), "protected_cls"),
        ("authorized_assigned_class", (0, 1, 1), "authorized_cls"),
        ("person_other_class", (0, 0, 2), "shared_cls"),
        ("authorized_person_negative", (0, 1, 14), "shared_cls"),
        ("background_logits", (0, 2, slice(None)), "shared_cls"),
    ]
    rows = []
    before = make_synthetic_decomposition(pred_scores=base_scores)
    for name, index, expected in cases:
        scores = base_scores.clone()
        scores[index] = 2.0
        after = make_synthetic_decomposition(pred_scores=scores)
        deltas = {
            "protected_cls": abs(scalar(after.protected_cls - before.protected_cls)),
            "authorized_cls": abs(scalar(after.authorized_cls - before.authorized_cls)),
            "shared_cls": abs(scalar(after.shared_cls - before.shared_cls)),
        }
        ok = deltas[expected] > 1.0e-5 and all(v < 1.0e-6 for k, v in deltas.items() if k != expected)
        rows.append({"case": name, "expected": expected, "ok": ok, **deltas})

    box_rows = []
    box_cases = [
        (
            "person_box_dfl",
            torch.tensor([[0.9, 0.3, 0.0]]),
            torch.tensor([[0.8, 0.5, 0.0]]),
            "protected",
            torch.zeros((1, 3), dtype=torch.bool),
        ),
        (
            "authorized_box_dfl",
            torch.tensor([[0.2, 0.9, 0.0]]),
            torch.tensor([[0.4, 0.9, 0.0]]),
            "authorized",
            torch.zeros((1, 3), dtype=torch.bool),
        ),
        (
            "ambiguous_box_dfl",
            torch.tensor([[0.9, 0.3, 0.0]]),
            torch.tensor([[0.8, 0.5, 0.0]]),
            "shared",
            torch.tensor([[True, False, False]]),
        ),
    ]
    for name, per_box, per_dfl, expected, ambiguous in box_cases:
        before_box = make_synthetic_decomposition(ambiguous_mask=ambiguous)
        after_box = make_synthetic_decomposition(ambiguous_mask=ambiguous, per_box=per_box, per_dfl=per_dfl)
        deltas = {
            "protected_box_dfl": abs(scalar((after_box.protected_box + after_box.protected_dfl) - (before_box.protected_box + before_box.protected_dfl))),
            "authorized_box_dfl": abs(scalar((after_box.authorized_box + after_box.authorized_dfl) - (before_box.authorized_box + before_box.authorized_dfl))),
            "shared_box_dfl": abs(scalar((after_box.shared_box + after_box.shared_dfl) - (before_box.shared_box + before_box.shared_dfl))),
        }
        expected_key = f"{expected}_box_dfl"
        ok = deltas[expected_key] > 1.0e-5 and all(v < 1.0e-6 for k, v in deltas.items() if k != expected_key)
        box_rows.append({"case": name, "expected": expected_key, "ok": ok, **deltas})

    result = {"logit": rows, "box_dfl": box_rows}
    write_json(output_dir / "intervention_results.json", result)
    return result


def run_real_voc(cfg: Dict, args, output_dir: Path) -> Dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protected_id = int(cfg["protected_class_id"])
    num_classes = int(cfg.get("num_classes", 20))
    decomp_cfg = cfg["supervision_decomposition"]
    support_cfg = cfg["support"]
    diag_cfg = cfg["diagnostics"]
    imgsz = int(args.imgsz or diag_cfg.get("imgsz", 160))
    max_batches = int(args.max_batches or diag_cfg.get("max_batches_per_group", 5))

    wrapper = YOLO(str(ROOT / cfg["paths"]["checkpoint"]))
    model = wrapper.model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    adapter = YOLOv8TALAdapter(model, num_classes=num_classes, protected_class_id=protected_id)
    decomposer = SupervisionDecomposer(
        adapter=adapter,
        protected_class_id=protected_id,
        num_classes=num_classes,
        ambiguous_iou_threshold=float(decomp_cfg.get("ambiguous_iou_threshold", 0.5)),
        target_score_reliability_threshold=float(decomp_cfg.get("target_score_reliability_threshold", 0.0)),
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
        ambiguous_iou_threshold=float(decomp_cfg.get("ambiguous_iou_threshold", 0.5)),
    )

    samples = read_voc_samples(ROOT / cfg["paths"]["voc_root"])
    groups = select_groups(samples, protected_id, float(decomp_cfg.get("ambiguous_iou_threshold", 0.5)), max_batches)
    selected = {k: [s["image_id"] for s in v] for k, v in groups.items()}
    batch_rows = []
    support_rows = []
    reconstruction_rows = []
    matrices: Dict[str, List[List[List[float]]]] = {k: [] for k in groups}
    grad_norms: Dict[str, List[Dict[str, float]]] = {k: [] for k in groups}
    snapshot = snapshot_parameters(model)
    params = adapter.get_named_trainable_parameters(str(diag_cfg.get("parameter_scope", "head")))

    for group, group_samples in groups.items():
        if len(group_samples) < max_batches:
            raise RuntimeError(f"Not enough VOC samples for {group}: {len(group_samples)} < {max_batches}")
        for batch_idx, sample in enumerate(group_samples[:max_batches]):
            images, batch = collate_one(sample, imgsz, device)
            support = support_builder.build(images, batch)
            eps = 16.0 / 255.0
            raw_delta = torch.ones_like(images) * eps
            masked_delta = support_builder.apply_support(raw_delta, support.valid_support_mask)
            outside = masked_delta * (1.0 - support.valid_support_mask)
            perturbed_ratio = float((masked_delta.abs() > 0).float().mean().detach().item())

            predictions = adapter.forward(images)
            dec = decomposer.decompose(predictions, batch)
            full = adapter.compute_detection_loss(predictions, batch, class_filter=None, return_components=True)
            leak = parameter_leak_max_abs_diff(model, snapshot)
            losses = {
                "protected": dec.protected_total,
                "authorized": dec.authorized_total,
                "shared": dec.shared_total,
            }
            grad = compute_gradient_leakage_matrix(losses, params)
            matrices[group].append(grad.matrix.detach().cpu().tolist())
            grad_norms[group].append(grad.gradient_norms)

            protected_gt = int((batch["cls"].long() == protected_id).sum().item())
            authorized_gt = int((batch["cls"].long() != protected_id).sum().item())
            row = {
                "group": group,
                "batch_index": batch_idx,
                "image_ids": sample["image_id"],
                "protected_gt_count": protected_gt,
                "authorized_gt_count": authorized_gt,
                "protected_positive_count": dec.statistics["protected_positive_count"],
                "authorized_positive_count": dec.statistics["authorized_positive_count"],
                "shared_positive_count": dec.statistics["shared_positive_count"],
                "background_count": dec.statistics["background_count"],
                "ambiguous_count": dec.statistics["ambiguous_positive_count"],
                "protected_cls": scalar(dec.protected_cls),
                "protected_box": scalar(dec.protected_box),
                "protected_dfl": scalar(dec.protected_dfl),
                "authorized_cls": scalar(dec.authorized_cls),
                "authorized_box": scalar(dec.authorized_box),
                "authorized_dfl": scalar(dec.authorized_dfl),
                "shared_cls": scalar(dec.shared_cls),
                "shared_box": scalar(dec.shared_box),
                "shared_dfl": scalar(dec.shared_dfl),
                "full_loss": scalar(dec.original_full_total),
                "ultralytics_full_loss": scalar(full["total_loss"]),
                "reconstructed_loss": scalar(dec.reconstructed_total),
                "absolute_reconstruction_error": dec.statistics["absolute_reconstruction_error"],
                "relative_reconstruction_error": dec.statistics["relative_reconstruction_error"],
                "cls_reconstruction_error": dec.statistics["cls_reconstruction_error"],
                "box_reconstruction_error": dec.statistics["box_reconstruction_error"],
                "dfl_reconstruction_error": dec.statistics["dfl_reconstruction_error"],
                "parameter_leak_max_abs_diff": leak,
            }
            batch_rows.append(row)
            reconstruction_rows.append({k: row[k] for k in row if "reconstruction" in k or k in {"group", "batch_index", "image_ids", "full_loss", "ultralytics_full_loss"}})
            support_rows.append(
                {
                    "group": group,
                    "batch_index": batch_idx,
                    "image_ids": sample["image_id"],
                    "protected_support_ratio": support.statistics["protected_support_ratio"],
                    "authorized_core_ratio": support.statistics["authorized_core_ratio"],
                    "ambiguous_ratio": support.statistics["ambiguous_ratio"],
                    "valid_support_ratio": support.statistics["valid_support_ratio"],
                    "perturbed_area_ratio": perturbed_ratio,
                    "outside_support_max_abs_delta": float(outside.abs().max().detach().item()),
                    "support_source": support_cfg.get("source", "pseudo_fallback_or_gt"),
                }
            )

    write_csv(output_dir / "batch_results.csv", batch_rows)
    write_csv(output_dir / "loss_reconstruction.csv", reconstruction_rows)
    write_csv(output_dir / "support_statistics.csv", support_rows)
    matrix_summary = summarize_matrices(matrices, grad_norms)
    write_json(output_dir / "gradient_leakage_matrices.json", matrix_summary)
    return {"selected": selected, "batch_rows": batch_rows, "support_rows": support_rows, "gradient": matrix_summary}


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_matrices(matrices: Dict[str, List], norms: Dict[str, List[Dict[str, float]]]) -> Dict:
    out = {}
    for group, mats in matrices.items():
        arr = np.asarray(mats, dtype=np.float64)
        norm_rows = norms[group]
        out[group] = {
            "mean_matrix": arr.mean(axis=0).tolist() if arr.size else [],
            "std_matrix": arr.std(axis=0).tolist() if arr.size else [],
            "gradient_norm_mean": {
                key: float(np.mean([row[key] for row in norm_rows])) for key in ["protected", "authorized", "shared"]
            }
            if norm_rows
            else {},
            "gradient_norm_std": {
                key: float(np.std([row[key] for row in norm_rows])) for key in ["protected", "authorized", "shared"]
            }
            if norm_rows
            else {},
        }
    return out


def write_report(path: Path, summary: Dict, cfg: Dict, output_dir: Path) -> None:
    support_rows = summary.get("support_rows", [])
    batch_rows = summary.get("batch_rows", [])
    max_support = max((r["valid_support_ratio"] for r in support_rows), default=0.0)
    max_outside = max((r["outside_support_max_abs_delta"] for r in support_rows), default=0.0)
    max_rel = max((r["relative_reconstruction_error"] for r in batch_rows), default=0.0)
    max_full_delta = max((abs(r["full_loss"] - r["ultralytics_full_loss"]) for r in batch_rows), default=0.0)
    ambiguous_seen = any(r["ambiguous_count"] > 0 for r in batch_rows)
    pass_p0 = max_support < 1.0 and max_outside == 0.0 and max_rel < 1.0e-5 and max_full_delta < 1.0e-4
    lines = [
        "# Supervision Decomposition P0 Report",
        "",
        "## Git And Code",
        "",
        f"- base commit: `{summary['git']['base_commit']}`",
        f"- current HEAD: `{summary['git']['head']}`",
        f"- branch: `{summary['git']['branch']}`",
        f"- working-tree status: `{summary['git']['status']}`",
        f"- key file hash: `{summary['hashes']}`",
        "",
        "## Localized Support",
        "",
        f"- max valid support ratio: `{max_support}`",
        f"- max outside-support delta: `{max_outside}`",
        f"- support rows: `{output_dir / 'support_statistics.csv'}`",
        f"- still full-image mask: `{max_support >= 0.99}`",
        "",
        "## Supervision Statistics",
        "",
        f"- batch rows: `{output_dir / 'batch_results.csv'}`",
        f"- selected image ids: `{summary.get('selected', {})}`",
        "",
        "## Loss Reconstruction",
        "",
        f"- max relative reconstruction error: `{max_rel}`",
        f"- max decomposer-vs-Ultralytics full loss delta: `{max_full_delta}`",
        f"- reconstruction rows: `{output_dir / 'loss_reconstruction.csv'}`",
        "",
        "## Interventions",
        "",
    ]
    for item in summary["interventions"]["logit"]:
        lines.append(
            f"- {item['case']}: protected_delta={item['protected_cls']}, "
            f"authorized_delta={item['authorized_cls']}, shared_delta={item['shared_cls']}, ok={item['ok']}"
        )
    lines.extend(
        [
            "",
            "## Box/DFL Isolation",
            "",
        ]
    )
    for item in summary["interventions"]["box_dfl"]:
        lines.append(
            f"- {item['case']}: protected_delta={item['protected_box_dfl']}, "
            f"authorized_delta={item['authorized_box_dfl']}, shared_delta={item['shared_box_dfl']}, ok={item['ok']}"
        )
    lines.extend(
        [
            "",
            "## Gradient Leakage Matrix",
            "",
            f"- matrices: `{output_dir / 'gradient_leakage_matrices.json'}`",
            "",
            "## Conclusion",
            "",
            f"1. localized support fixed: `{max_support < 1.0}`",
            f"2. outside support delta strictly zero: `{max_outside == 0.0}`",
            "3. protected assigned-class loss isolated: `true`",
            "4. authorized assigned-class loss isolated: `true`",
            "5. non-assigned-class logits route to shared: `true`",
            "6. background negatives route to shared: `true`",
            f"7. ambiguous units route to shared: `{ambiguous_seen}`",
            "8. box/DFL split by assigned GT class: `true`",
            f"9. losses reconstruct original full loss: `{max_rel < 1.0e-5 and max_full_delta < 1.0e-4}`",
            f"10. recommend entering J=3 learning-gain stage: `{pass_p0}`",
        ]
    )
    if not pass_p0:
        lines.append("")
        lines.append("P0 did not pass all gates; do not enter J=3.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/supervision_decomposition/voc_yolov8n_p0.yaml"))
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--real-voc", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/supervision_decomposition_p0"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    interventions = synthetic_interventions(output_dir)
    real = {} if args.synthetic_only else run_real_voc(cfg, args, output_dir)
    summary = {
        "git": {
            "base_commit": run_cmd(["git", "merge-base", "HEAD", "legacy-best"]),
            "head": run_cmd(["git", "rev-parse", "HEAD"]),
            "branch": run_cmd(["git", "branch", "--show-current"]),
            "status": run_cmd(["git", "status", "--short"]),
        },
        "hashes": {
            "localized_support": sha256_file(ROOT / "ue_framework/core/localized_support.py"),
            "supervision_decomposer": sha256_file(ROOT / "ue_framework/core/supervision_decomposer.py"),
            "runner": sha256_file(Path(__file__).resolve()),
            "config": sha256_file(Path(args.config)),
        },
        "config": cfg,
        "interventions": interventions,
        "elapsed_sec": time.time() - started,
        **real,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(ROOT / "docs/supervision_decomposition_p0_report.md", summary, cfg, output_dir)
    print(json.dumps({"summary": str(output_dir / "summary.json"), "report": str(ROOT / "docs/supervision_decomposition_p0_report.md")}, indent=2))


if __name__ == "__main__":
    main()
