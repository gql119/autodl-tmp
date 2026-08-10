from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import (
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
    save_image_rgb_float,
)
from ue_framework.metrics_utils import VOC20_CLASS_NAMES
from ue_framework.methods.sdh_counterfactual import (
    aggregate_person_classification_losses,
    build_person_free_transplant_metrics,
    deterministic_person_audit_subset,
    fixed_person_classification_pair_losses,
)
from ue_framework.methods.sdh_evaluation import (
    build_learning_preference_audit,
    build_sdh_counterfactual_metrics,
)
from ue_framework.methods.sdh_materializer import load_frozen_sdh_state
from ue_framework.methods.sdh_mechanism import load_sdh_batch
from ue_framework.methods.semantic_hiding_carrier import render_person_box_carrier
from ue_framework.stages.evaluate import _extract_metrics_dict
from ue_framework.support import _bbox_to_pixels


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the read-only TAUSB-SDH carrier counterfactual and learning audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--victim-checkpoint", required=True)
    parser.add_argument("--clean-metrics", required=True)
    parser.add_argument(
        "--epoch-checkpoint",
        action="append",
        required=True,
        help="Repeat exactly as epoch=checkpoint for epochs 1,5,10,20.",
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-batch-size", type=int, default=4)
    parser.add_argument("--audit-per-stratum", type=int, default=32)
    parser.add_argument("--transplant-limit", type=int, default=256)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _parse_epoch_checkpoints(values: Sequence[str]) -> Dict[int, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("epoch checkpoint must use epoch=path.")
        epoch_text, path_text = value.split("=", 1)
        epoch = int(epoch_text)
        path = Path(path_text).resolve()
        if epoch in output:
            raise ValueError("Duplicate epoch checkpoint: %d" % epoch)
        if not path.is_file():
            raise FileNotFoundError("Epoch checkpoint not found: %s" % path)
        output[epoch] = path
    if set(output) != {1, 5, 10, 20}:
        raise ValueError("Epoch checkpoints must be exactly 1/5/10/20.")
    return output


def _person_boxes(
    annotations: Sequence[dict], height: int, width: int, *, target_class_id: int
) -> List[List[float]]:
    return [
        [float(value) for value in _bbox_to_pixels(item["bbox"], width, height)]
        for item in annotations
        if int(item["cls"]) == int(target_class_id)
    ]


def _write_eval_yaml(path: Path, dataset_root: Path) -> None:
    payload = {
        "path": str(dataset_root),
        "train": "images",
        "val": "images",
        "names": {index: name for index, name in enumerate(VOC20_CLASS_NAMES)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _materialize_target_carrier_val(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    output_root: Path,
    state,
    device: torch.device,
) -> Dict[str, object]:
    image_output = output_root / "images"
    label_output = output_root / "labels"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)
    rows = []
    linf_values = []
    person_images = 0
    for image_path in image_paths:
        label_path = Path(label_path_for_image(str(image_path), str(label_dir)))
        if not label_path.is_file():
            raise FileNotFoundError("Validation label missing: %s" % label_path)
        annotations = read_yolo_annotations(str(label_path))
        image = load_image_rgb_float(str(image_path))
        height, width = image.shape[:2]
        boxes = _person_boxes(
            annotations, height, width, target_class_id=state.target_class_id
        )
        output_path = image_output / (image_path.stem + ".png")
        linf = 0.0
        if boxes:
            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            with torch.no_grad():
                rendered = render_person_box_carrier(
                    image_tensor, (box_tensor,), state.carrier, state.secret
                )
            outside = float(
                (
                    rendered.perturbation.abs()
                    * (1.0 - rendered.union_support)
                ).max()
            )
            if outside != 0.0:
                raise RuntimeError("Counterfactual carrier leaked outside person GT boxes.")
            linf = float(rendered.perturbation.abs().max())
            if linf > state.epsilon + 1.0 / 255.0:
                raise RuntimeError("Counterfactual carrier exceeds the approved Linf budget.")
            output_image = rendered.poisoned[0].permute(1, 2, 0).cpu().numpy()
            person_images += 1
        else:
            output_image = image
        save_image_rgb_float(str(output_path), output_image)
        shutil.copy2(str(label_path), str(label_output / (image_path.stem + ".txt")))
        rows.append(
            {
                "image_id": image_path.stem,
                "person_boxes": len(boxes),
                "linf": linf,
                "support_source": "person_gt_bbox" if boxes else "none",
            }
        )
        linf_values.append(linf)
    _write_json(output_root / "manifest.json", rows)
    return {
        "image_count": len(rows),
        "person_image_count": person_images,
        "linf_max": float(max(linf_values, default=0.0)),
        "state_content_hash": state.state_content_hash,
        "secret_tensor_sha256": state.secret_tensor_sha256,
        "support_source": "person_gt_bbox",
    }


def _evaluate_target_carrier(
    model: YOLO,
    *,
    dataset_root: Path,
    artifact_root: Path,
    device: str,
    batch: int,
) -> Dict[str, object]:
    data_yaml = artifact_root / "target_carrier_val.yaml"
    _write_eval_yaml(data_yaml, dataset_root)
    result = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=int(batch),
        device=device,
        workers=4,
        verbose=False,
        project=str(artifact_root / "yolo_eval"),
        name="target_carrier_val",
        exist_ok=False,
    )
    return _extract_metrics_dict(result, 20, 14, strict=True)


def _run_learning_preference(
    checkpoints: Dict[int, Path],
    *,
    subset: Sequence[Path],
    subset_hash: str,
    label_dir: Path,
    state,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    epoch_losses = {}
    checkpoint_hashes = {}
    for epoch in (1, 5, 10, 20):
        checkpoint = checkpoints[epoch]
        wrapper = YOLO(str(checkpoint))
        model = wrapper.model.to(device).eval()
        rows = []
        for start in range(0, len(subset), batch_size):
            batch = load_sdh_batch(
                subset[start : start + batch_size],
                label_dir=label_dir,
                image_size=640,
                target_class_id=14,
                device=device,
            )
            rows.append(
                fixed_person_classification_pair_losses(
                    model, batch, state.carrier, state.secret
                )
            )
        epoch_losses[epoch] = aggregate_person_classification_losses(rows)
        checkpoint_hashes[str(epoch)] = _sha256(checkpoint)
        del wrapper, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    audit = build_learning_preference_audit(epoch_losses)
    audit["audit_subset_hash"] = subset_hash
    audit["audit_image_ids"] = [path.stem for path in subset]
    audit["checkpoint_sha256"] = checkpoint_hashes
    audit["assignment_source"] = "clean_real_tal_fixed_for_both_views"
    return audit


def _max_person_confidences(
    model: YOLO, paths: Sequence[Path], *, device: str, batch: int
) -> List[float]:
    results = model.predict(
        source=[str(path) for path in paths],
        imgsz=640,
        batch=int(batch),
        device=device,
        verbose=False,
        save=False,
    )
    output = []
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.cls.numel() == 0:
            output.append(0.0)
            continue
        mask = boxes.cls.long().eq(14)
        output.append(float(boxes.conf[mask].max()) if bool(mask.any()) else 0.0)
    return output


def _materialize_person_free_transplant(
    image_paths: Sequence[Path],
    *,
    label_dir: Path,
    output_root: Path,
    state,
    device: torch.device,
    limit: int,
) -> Tuple[List[Path], List[Path]]:
    person_shapes = []
    candidates = []
    for image_path in image_paths:
        annotations = read_yolo_annotations(
            label_path_for_image(str(image_path), str(label_dir))
        )
        person = [item for item in annotations if int(item["cls"]) == 14]
        for item in person:
            _, _, width, height = [float(value) for value in item["bbox"]]
            person_shapes.append((width * height, width / max(height, 1.0e-8)))
        if not person and annotations:
            candidates.append((image_path, annotations))
    if not person_shapes:
        raise ValueError("Cannot define transplant matching without person boxes.")
    shape_array = np.asarray(person_shapes, dtype=np.float64)
    area_low, area_high = np.quantile(shape_array[:, 0], [0.1, 0.9])
    ratio_low, ratio_high = np.quantile(shape_array[:, 1], [0.1, 0.9])
    area_mid = float(np.median(shape_array[:, 0]))
    ratio_mid = float(np.median(shape_array[:, 1]))
    output_root.mkdir(parents=True)
    clean_paths = []
    carrier_paths = []
    for image_path, annotations in candidates:
        eligible = []
        for item in annotations:
            _, _, width, height = [float(value) for value in item["bbox"]]
            area = width * height
            ratio = width / max(height, 1.0e-8)
            if area_low <= area <= area_high and ratio_low <= ratio <= ratio_high:
                distance = abs(np.log(area / area_mid)) + abs(np.log(ratio / ratio_mid))
                eligible.append((float(distance), item))
        if not eligible:
            continue
        selected = min(eligible, key=lambda value: value[0])[1]
        image = load_image_rgb_float(str(image_path))
        height, width = image.shape[:2]
        box = _bbox_to_pixels(selected["bbox"], width, height)
        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        )
        box_tensor = torch.tensor([box], dtype=torch.float32, device=device)
        with torch.no_grad():
            rendered = render_person_box_carrier(
                image_tensor, (box_tensor,), state.carrier, state.secret
            )
        destination = output_root / (image_path.stem + ".png")
        save_image_rgb_float(
            str(destination), rendered.poisoned[0].permute(1, 2, 0).cpu().numpy()
        )
        clean_paths.append(image_path)
        carrier_paths.append(destination)
        if len(clean_paths) >= int(limit):
            break
    if not clean_paths:
        raise ValueError("No scale/aspect-matched person-free transplant boxes were found.")
    return clean_paths, carrier_paths


def run(args: argparse.Namespace) -> Dict[str, object]:
    config = load_config(str(Path(args.config).resolve()))
    method_cfg = config["methods"]["tausb_sdh"]
    artifact_root = Path(args.artifact_root).resolve()
    if artifact_root.exists():
        raise FileExistsError("Counterfactual artifact root must be fresh: %s" % artifact_root)
    artifact_root.mkdir(parents=True)
    device = torch.device(str(args.device))
    dataset_root = Path(config["data"]["dataset_root"]).resolve()
    val_image_dir = dataset_root / config["data"]["val_images"]
    val_label_dir = dataset_root / config["data"]["val_labels"]
    image_paths = tuple(Path(path) for path in list_images(str(val_image_dir)))
    victim_checkpoint = Path(args.victim_checkpoint).resolve()
    clean_metrics_path = Path(args.clean_metrics).resolve()
    if not victim_checkpoint.is_file() or not clean_metrics_path.is_file():
        raise FileNotFoundError("Victim checkpoint and clean metrics must both exist.")
    epoch_checkpoints = _parse_epoch_checkpoints(args.epoch_checkpoint)
    state = load_frozen_sdh_state(
        str(Path(method_cfg["frozen_sdh_state"]).resolve()),
        device=device,
        expected_target_class_id=14,
        expected_epsilon=16.0 / 255.0,
        expected_hashes={
            name: method_cfg[name]
            for name in (
                "secret_source_sha256",
                "secret_tensor_sha256",
                "source_manifest_sha256",
                "train_split_sha256",
            )
        },
    )
    materialization = _materialize_target_carrier_val(
        image_paths,
        label_dir=val_label_dir,
        output_root=artifact_root / "target_carrier_val",
        state=state,
        device=device,
    )
    victim = YOLO(str(victim_checkpoint))
    target_carrier_metrics = _evaluate_target_carrier(
        victim,
        dataset_root=artifact_root / "target_carrier_val",
        artifact_root=artifact_root,
        device=str(args.device),
        batch=int(config["victim"]["batch"]),
    )
    clean_metrics = json.loads(clean_metrics_path.read_text(encoding="utf-8"))
    counterfactual = build_sdh_counterfactual_metrics(
        clean_metrics, target_carrier_metrics
    )
    subset, subset_hash = deterministic_person_audit_subset(
        image_paths,
        label_dir=val_label_dir,
        target_class_id=14,
        per_stratum=int(args.audit_per_stratum),
    )
    dynamics = _run_learning_preference(
        epoch_checkpoints,
        subset=subset,
        subset_hash=subset_hash,
        label_dir=val_label_dir,
        state=state,
        device=device,
        batch_size=int(args.audit_batch_size),
    )
    clean_transplant_paths, carrier_transplant_paths = _materialize_person_free_transplant(
        image_paths,
        label_dir=val_label_dir,
        output_root=artifact_root / "person_free_transplant",
        state=state,
        device=device,
        limit=int(args.transplant_limit),
    )
    clean_confidence = _max_person_confidences(
        victim,
        clean_transplant_paths,
        device=str(args.device),
        batch=int(args.audit_batch_size),
    )
    carrier_confidence = _max_person_confidences(
        victim,
        carrier_transplant_paths,
        device=str(args.device),
        batch=int(args.audit_batch_size),
    )
    transplant = build_person_free_transplant_metrics(
        clean_confidence, carrier_confidence
    )
    result = {
        "schema": "tausb.sdh-counterfactual-run.v1",
        "read_only": True,
        "used_for_checkpoint_selection": False,
        "primary_metric_remains": "P1-V clean-val AP50",
        "victim_checkpoint_sha256": _sha256(victim_checkpoint),
        "clean_metrics_sha256": _sha256(clean_metrics_path),
        "materialization": materialization,
        "target_carrier_val": target_carrier_metrics,
        "counterfactual": counterfactual,
        "learning_preference": dynamics,
        "person_free_transplant": transplant,
    }
    _write_json(artifact_root / "counterfactual_metrics.json", result)
    return result


def main() -> int:
    args = _arguments()
    started = time.time()
    artifact_root = Path(args.artifact_root).resolve()
    try:
        result = run(args)
        _write_json(
            artifact_root / "status.json",
            {
                "status": "completed",
                "started_unix": started,
                "ended_unix": time.time(),
                "schema": result["schema"],
            },
        )
        return 0
    except Exception as error:
        artifact_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_root / "status.json",
            {
                "status": "failed",
                "started_unix": started,
                "ended_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
