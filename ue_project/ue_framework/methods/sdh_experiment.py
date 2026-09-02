from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..data_utils import (
    image_has_target,
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)
from ..support import _bbox_to_pixels
from .constraint_gradient_router import (
    backtrack_multi_parameter_update,
    route_multi_parameter_gradients,
)
from .detector_lfc import DetectorLFCPrototypeBank
from .detector_lfc import DetectorLFCExtractor
from .instance_cicr import FrozenInstanceCICRBank
from .non_target_logit_alignment import FrozenNLAGradientCalibration
from .sdh_materializer import (
    E2E_V0_PROTOCOL_ID,
    build_feasibility_sdh_state_payload,
    build_frozen_sdh_state_payload,
)
from .sdh_mechanism import (
    FrozenTargetGradientCalibration,
    SDHObservation,
    SDHObservationEngine,
    adapter_parameters,
    compose_sdh_target_objective,
    load_sdh_batch,
)
from .semantic_hiding_carrier import SemanticHidingCarrier
from .semantic_hiding_validation import (
    compute_hiding_metrics,
    evaluate_hiding_gate,
    hiding_pretrain_step,
    reveal_loss,
)


ARM_SWITCHES = {
    "T0": {"dlfc": False, "cicr": False, "cgr": False, "nla": False},
    "T1": {"dlfc": True, "cicr": True, "cgr": False, "nla": False},
    "P0": {"dlfc": True, "cicr": True, "cgr": True, "nla": False},
    "P1": {"dlfc": True, "cicr": True, "cgr": True, "nla": True},
}

E2E_V0_SPEC_ID = E2E_V0_PROTOCOL_ID
DGCAIP_SPEC_ID = "TAUSB-SDH-DGCAIP-CGR-E20-v2"
DGCAIP_R3_DIAG_SPEC_ID = "TAUSB-SDH-DGCAIP-R3-DIAG-v1"
DGCAIP_R4_DIAG_SPEC_ID = "TAUSB-SDH-DGCAIP-R4-D0-BINDING-FIX-v1"
DGCAIP_P4_E20_SPEC_ID = "TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1"
DGCAIP_DATASET_CGR_PROXY_SPEC_ID = (
    "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"
)
DGCAIP_STRICT_ROUTE_V2_SPEC_ID = "TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2"
DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID = (
    "TAUSB-SDH-DGCAIP-COMPONENT-ALIGNED-ROUTE-v3"
)
DGCAIP_RELAXED_GATE_V4_SPEC_ID = (
    "TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4"
)
DGCAIP_P1_DETERMINISM_AUDIT_SPEC_ID = (
    "TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1"
)
DGCAIP_P1_DET_RESIZE_FIX_SPEC_ID = "TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1"
DGCAIP_DIAG_SPEC_IDS = {DGCAIP_R3_DIAG_SPEC_ID, DGCAIP_R4_DIAG_SPEC_ID}
DGCAIP_AUDIT_SPEC_IDS = {
    DGCAIP_P1_DETERMINISM_AUDIT_SPEC_ID,
    DGCAIP_P1_DET_RESIZE_FIX_SPEC_ID,
}
DGCAIP_SPEC_IDS = {
    DGCAIP_SPEC_ID,
    DGCAIP_P4_E20_SPEC_ID,
    DGCAIP_DATASET_CGR_PROXY_SPEC_ID,
    DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
    DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
    DGCAIP_RELAXED_GATE_V4_SPEC_ID,
    *DGCAIP_DIAG_SPEC_IDS,
    *DGCAIP_AUDIT_SPEC_IDS,
}
E2E_V0_R2_CHECKPOINT_SHA256 = (
    "a765e27a62bb1a1939aaae487ff6e61ec405f457056d2329c1c49f91e02c9f36"
)
E2E_V0_R2_METRICS_SHA256 = (
    "c7d1b120ffbadeb7385be41669dda704b00a2cee60940e3c3d97112e24e59246"
)
E2E_V0_SURROGATE_CHECKPOINT_SHA256 = (
    "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89"
)
E2E_V0_TRAIN_IMAGE_MANIFEST_SHA256 = (
    "4954727df8686532a788668fd815092112ac3e3ee1414eba83b616e683708fbd"
)
E2E_V0_TRAIN_LABEL_MANIFEST_SHA256 = (
    "3cd05ad1ab6a546bf2afd5e63cb6c3ff6667064d80af129dd819325625b9d848"
)
E2E_V0_ALLOWED_FAILED_HIDING_CHECKS = frozenset(
    ("rms_diversity", "delta_high_frequency")
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_size_manifest_sha256(paths: Sequence[Path]) -> str:
    records = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return _canonical_json_sha256(records)


def _path_content_manifest_sha256(paths: Sequence[Path]) -> str:
    records = [
        {"name": path.name, "sha256": _file_sha256(path)}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return _canonical_json_sha256(records)


def _float_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _load_secret(path: Path, expected_tensor_sha256: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Secret image not found: %s" % path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (256, 256):
        raise ValueError("Secret preprocessing must already be 256x256.")
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    actual = _float_tensor_sha256(tensor)
    if actual != str(expected_tensor_sha256).lower():
        raise ValueError("Secret float tensor hash mismatch for %s." % path.name)
    return tensor


def _person_paths(image_dir: Path, label_dir: Path, target_class_id: int) -> List[Path]:
    paths = []
    for raw in list_images(str(image_dir)):
        path = Path(raw)
        annotations = read_yolo_annotations(label_path_for_image(str(path), str(label_dir)))
        if image_has_target(annotations, target_class_id):
            paths.append(path)
    return paths


def deterministic_person_split(
    paths: Sequence[Path],
    *,
    label_dir: Path,
    target_class_id: int,
    calibration_count: int,
    heldout_count: int,
    seed: int,
) -> Dict[str, List[Path]]:
    person_only = []
    cooccur = []
    for path in paths:
        annotations = read_yolo_annotations(label_path_for_image(str(path), str(label_dir)))
        bucket = (
            cooccur
            if any(int(item["cls"]) != int(target_class_id) for item in annotations)
            else person_only
        )
        key = hashlib.sha256(("%d:%s" % (seed, path.stem)).encode("ascii")).hexdigest()
        bucket.append((key, path))
    person_only.sort(key=lambda item: item[0])
    cooccur.sort(key=lambda item: item[0])

    def take_balanced(count: int, offsets: Tuple[int, int]) -> Tuple[List[Path], Tuple[int, int]]:
        first_count = count // 2
        second_count = count - first_count
        selected = (
            [item[1] for item in person_only[offsets[0] : offsets[0] + first_count]]
            + [item[1] for item in cooccur[offsets[1] : offsets[1] + second_count]]
        )
        if len(selected) != count:
            raise ValueError("Insufficient balanced person-only/cooccur images.")
        selected.sort(key=lambda path: path.stem)
        return selected, (offsets[0] + first_count, offsets[1] + second_count)

    calibration, offsets = take_balanced(calibration_count, (0, 0))
    heldout, _ = take_balanced(heldout_count, offsets)
    if set(calibration).intersection(heldout):
        raise RuntimeError("Calibration and held-out person splits overlap.")
    return {"calibration": calibration, "heldout": heldout}


def _split_hash(split: Mapping[str, Sequence[Path]]) -> str:
    payload = {
        name: [path.stem for path in values] for name, values in sorted(split.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _first_person_host(path: Path, label_dir: Path, device: torch.device) -> torch.Tensor:
    image = load_image_rgb_float(str(path))
    annotations = read_yolo_annotations(label_path_for_image(str(path), str(label_dir)))
    targets = [item for item in annotations if int(item["cls"]) == 14]
    if not targets:
        raise ValueError("Person host path has no person label.")
    x1, y1, x2, y2 = _bbox_to_pixels(targets[0]["bbox"], image.shape[1], image.shape[0])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Person host crop is empty.")
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(crop).permute(2, 0, 1).float().to(device)


def _batches(values: Sequence[Path], batch_size: int) -> List[List[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [list(values[index : index + batch_size]) for index in range(0, len(values), batch_size)]


def _time_guard(start: float, max_seconds: float, stage: str) -> None:
    if time.monotonic() - start > max_seconds:
        raise TimeoutError("%s exceeded the frozen cost cap." % stage)


def _balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = labels.bool()
    predictions = predictions.bool()
    positive = labels
    negative = ~labels
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("Balanced accuracy requires both classes.")
    sensitivity = (predictions[positive] == labels[positive]).float().mean()
    specificity = (predictions[negative] == labels[negative]).float().mean()
    return float(((sensitivity + specificity) * 0.5).cpu())


def _binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool().cpu()
    scores = scores.float().cpu()
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both positive and negative samples.")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2.0)
        / float(positive * negative)
    )


def _fit_linear_probe(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or labels.shape[0] != features.shape[0]:
        raise ValueError("Linear probe features/labels do not align.")
    design = torch.cat(
        (features.double(), torch.ones((features.shape[0], 1), dtype=torch.float64)),
        dim=1,
    )
    target = labels.double()
    ridge = 1.0e-3 * torch.eye(design.shape[1], dtype=torch.float64)
    return torch.linalg.solve(design.T @ design + ridge, design.T @ target)


def _secret_dlfc_features(
    paths: Sequence[Path],
    *,
    label_dir: Path,
    carrier: SemanticHidingCarrier,
    primary_secret: torch.Tensor,
    extractor: DetectorLFCExtractor,
    device: torch.device,
    batch_size: int,
    start: float,
    max_seconds: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_rows = []
    cooccur_labels = []
    non_target_labels = []
    for path_batch in _batches(paths, batch_size):
        _time_guard(start, max_seconds, "hiding detector leakage probe")
        hosts = torch.stack(
            [_first_person_host(path, label_dir, device) for path in path_batch]
        )
        output = carrier(
            hosts,
            primary_secret.expand(hosts.shape[0], -1, -1, -1),
        )
        extracted = extractor.extract(output.delta)
        feature_rows.append(torch.cat(extracted.classification, dim=1).detach().cpu())
        for path in path_batch:
            annotations = read_yolo_annotations(
                label_path_for_image(str(path), str(label_dir))
            )
            active = {int(item["cls"]) for item in annotations if int(item["cls"]) != 14}
            cooccur_labels.append(bool(active))
            non_target_labels.append([class_id in active for class_id in range(20) if class_id != 14])
    return (
        torch.cat(feature_rows),
        torch.tensor(cooccur_labels, dtype=torch.bool),
        torch.tensor(non_target_labels, dtype=torch.bool),
    )


def _leakage_probe(
    calibration: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    heldout: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, Any]:
    calibration_features, calibration_cooccur, calibration_classes = calibration
    heldout_features, heldout_cooccur, heldout_classes = heldout
    cooccur_weights = _fit_linear_probe(
        calibration_features, calibration_cooccur.float().unsqueeze(1)
    )
    heldout_design = torch.cat(
        (
            heldout_features.double(),
            torch.ones((heldout_features.shape[0], 1), dtype=torch.float64),
        ),
        dim=1,
    )
    cooccur_scores = (heldout_design @ cooccur_weights).squeeze(1)
    balanced = _balanced_accuracy(heldout_cooccur, cooccur_scores >= 0.5)
    aurocs = []
    evaluated_classes = []
    for index, class_id in enumerate(class_id for class_id in range(20) if class_id != 14):
        train_labels = calibration_classes[:, index]
        test_labels = heldout_classes[:, index]
        if not (bool(train_labels.any()) and bool((~train_labels).any())):
            continue
        if not (bool(test_labels.any()) and bool((~test_labels).any())):
            continue
        weights = _fit_linear_probe(
            calibration_features, train_labels.float().unsqueeze(1)
        )
        scores = (heldout_design @ weights).squeeze(1)
        aurocs.append(_binary_auroc(test_labels, scores))
        evaluated_classes.append(class_id)
    if not aurocs:
        raise ValueError("No active non-target class supports held-out AUROC.")
    return {
        "cooccur_balanced_accuracy": balanced,
        "cooccur_majority_balanced_baseline": 0.5,
        "cooccur_excess": balanced - 0.5,
        "non_target_macro_auroc": float(np.mean(aurocs)),
        "evaluated_non_target_classes": evaluated_classes,
        "pass": bool(balanced <= 0.60 and float(np.mean(aurocs)) <= 0.60),
    }


def validate_sdh_experiment_config(config: Mapping[str, Any]) -> None:
    required = {"spec", "dataset", "model", "secrets", "hiding", "mechanism", "runtime"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError("Missing SDH experiment config sections: %s" % missing)
    spec_id = str(config["spec"].get("spec_id", ""))
    legacy_spec_id = "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3"
    subband_spec_id = "TAUSB-SDH-HIDING-SB-v1"
    if spec_id not in {
        legacy_spec_id,
        subband_spec_id,
        E2E_V0_SPEC_ID,
        *DGCAIP_SPEC_IDS,
    }:
        raise ValueError("SDH experiment spec_id mismatch.")
    if int(config["spec"].get("seed", -1)) != 0:
        raise ValueError("SDH first experiment seed must be 0.")
    if int(config["dataset"].get("target_class_id", -1)) != 14:
        raise ValueError("SDH target class must be person=14.")
    if int(config["model"].get("num_classes", -1)) != 20:
        raise ValueError("SDH surrogate must be VOC20.")
    if int(config["model"].get("image_size", -1)) != 640:
        raise ValueError("SDH mechanism image size must be 640.")
    if abs(float(config["mechanism"].get("epsilon", -1)) - 16.0 / 255.0) > 1e-9:
        raise ValueError("SDH epsilon must remain 16/255.")
    if int(config["mechanism"].get("calibration_batches", -1)) < 16:
        raise ValueError("SDH requires at least 16 calibration batches.")
    if int(config["mechanism"].get("heldout_batches", -1)) < 24:
        raise ValueError("SDH requires at least 24 held-out batches.")
    expected_steps = 1 if spec_id in DGCAIP_AUDIT_SPEC_IDS else 8
    if int(config["mechanism"].get("optimization_steps", -1)) != expected_steps:
        raise ValueError(
            "SDH optimization_steps must remain %d for the selected Spec."
            % expected_steps
        )
    if int(config["mechanism"].get("max_backtracks", -1)) != 5:
        raise ValueError("SDH CGR max_backtracks must remain 5.")
    if float(config["mechanism"].get("probability_drop_tolerance", -1)) != 0.005:
        raise ValueError("SDH probability-drop tolerance must remain 0.005.")
    if config["mechanism"].get("eot_enabled") is not False:
        raise ValueError("SDH first round forbids EOT.")
    if config["mechanism"].get("jnd_enabled") is not False:
        raise ValueError("SDH first round forbids JND.")
    hf_subband_scale = float(config["hiding"].get("hf_subband_scale", 1.0))
    if not math.isfinite(hf_subband_scale) or not 0.0 <= hf_subband_scale <= 1.0:
        raise ValueError("SDH hiding.hf_subband_scale must be finite in [0,1].")
    rms_cv_gate_enabled = config["hiding"].get("rms_cv_gate_enabled", True)
    if spec_id == subband_spec_id:
        if hf_subband_scale != 0.25:
            raise ValueError("TAUSB-SDH-HIDING-SB-v1 freezes hf_subband_scale=0.25.")
        if rms_cv_gate_enabled is not False:
            raise ValueError("TAUSB-SDH-HIDING-SB-v1 keeps RMS CV descriptive only.")
    elif hf_subband_scale != 1.0 or rms_cv_gate_enabled is not True:
        raise ValueError("Legacy SDH config requires the exact scale=1 RMS-gated protocol.")
    feasibility_keys = (
        "source_artifact_root",
        "source_metrics_sha256",
        "source_checkpoint_sha256",
        "allow_failed_scientific_gates",
    )
    if spec_id == E2E_V0_SPEC_ID or spec_id in DGCAIP_SPEC_IDS:
        if config["hiding"].get("allow_failed_scientific_gates") is not True:
            raise ValueError("E2E V0 requires allow_failed_scientific_gates=true.")
        if not str(config["hiding"].get("source_artifact_root", "")).strip():
            raise ValueError("E2E V0 requires a read-only hiding source_artifact_root.")
        if str(config["hiding"].get("source_metrics_sha256", "")).lower() != (
            E2E_V0_R2_METRICS_SHA256
        ):
            raise ValueError("E2E V0 hiding metrics hash must match frozen r2.")
        if str(config["hiding"].get("source_checkpoint_sha256", "")).lower() != (
            E2E_V0_R2_CHECKPOINT_SHA256
        ):
            raise ValueError("E2E V0 hiding checkpoint hash must match frozen r2.")
        if int(config["dataset"].get("expected_train_images", -1)) != 16551:
            raise ValueError("E2E V0 requires exactly 16551 training images.")
        if int(config["dataset"].get("expected_person_images", -1)) != 6095:
            raise ValueError("E2E V0 requires exactly 6095 person training images.")
        if str(config["dataset"].get("train_image_manifest_sha256", "")).lower() != (
            E2E_V0_TRAIN_IMAGE_MANIFEST_SHA256
        ):
            raise ValueError("E2E V0 training image manifest hash mismatch.")
        if str(config["dataset"].get("train_label_manifest_sha256", "")).lower() != (
            E2E_V0_TRAIN_LABEL_MANIFEST_SHA256
        ):
            raise ValueError("E2E V0 training label manifest hash mismatch.")
        if str(config["model"].get("surrogate_checkpoint_sha256", "")).lower() != (
            E2E_V0_SURROGATE_CHECKPOINT_SHA256
        ):
            raise ValueError("E2E V0 surrogate checkpoint hash mismatch.")
        source_root = _resolve(
            Path("."), str(config["hiding"]["source_artifact_root"])
        )
        output_root = _resolve(Path("."), str(config["runtime"].get("artifact_root", "")))
        if source_root == output_root:
            raise ValueError("E2E V0 hiding input and mechanism output roots must differ.")
    elif any(key in config["hiding"] for key in feasibility_keys):
        raise ValueError("Failed-scientific-gate loading is restricted to the E2E V0 Spec.")
    for section, keys in {
        "dataset": ("root", "train_images", "train_labels"),
        "model": ("surrogate_checkpoint",),
        "secrets": ("manifest", "primary_id", "primary_source_sha256"),
        "runtime": ("artifact_root", "device"),
    }.items():
        for key in keys:
            if not str(config[section].get(key, "")).strip():
                raise ValueError("SDH config requires %s.%s." % (section, key))
    if spec_id in DGCAIP_SPEC_IDS:
        dgcaip = config.get("dgcaip")
        if not isinstance(dgcaip, Mapping):
            raise ValueError("DG-CAIP Spec requires a dgcaip config section.")
        run_mode = str(dgcaip.get("run_mode", ""))
        if spec_id == DGCAIP_DATASET_CGR_PROXY_SPEC_ID:
            expected_run_modes = {
                "dataset_risk_scan",
                "short_victim_risk_scan",
                "strict_mechanism",
                "proxy_victim_audit",
                "production_e20",
            }
        elif spec_id in {
            DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
            DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
            DGCAIP_RELAXED_GATE_V4_SPEC_ID,
        }:
            expected_run_modes = {"strict_mechanism"}
        elif spec_id == DGCAIP_P4_E20_SPEC_ID:
            expected_run_modes = {"production_e20"}
        elif spec_id in DGCAIP_DIAG_SPEC_IDS:
            expected_run_modes = {"r3_diag"}
        elif spec_id in DGCAIP_AUDIT_SPEC_IDS:
            expected_run_modes = {"p1_determinism_audit"}
        else:
            expected_run_modes = {"d0", "mechanism"}
        if run_mode not in expected_run_modes:
            raise ValueError(
                "DG-CAIP run_mode is invalid for the selected Spec."
            )
        frozen = {
            "temperature": 2.0,
            "classification_tolerance": 0.005,
            "box_tolerance": 0.02,
            "alignment_tolerance": 0.05,
            "js_backtracking_epsilon": 1.0e-9,
        }
        for key, expected in frozen.items():
            if float(dgcaip.get(key, float("nan"))) != expected:
                raise ValueError("DG-CAIP %s must remain %s." % (key, expected))
        if spec_id in {
            DGCAIP_DATASET_CGR_PROXY_SPEC_ID,
            DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
            DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
            DGCAIP_RELAXED_GATE_V4_SPEC_ID,
        }:
            ranking = config.get("dataset_ranking")
            strict_route = config.get("strict_route")
            agreement = config.get("proxy_agreement")
            if not all(
                isinstance(section, Mapping)
                for section in (ranking, strict_route, agreement)
            ):
                raise ValueError(
                    "Dataset-CGR-Proxy requires ranking, route, and agreement sections."
                )
            expected_values = {
                "dataset_ranking.js_weight": (ranking, "js_weight", 0.7),
                "dataset_ranking.kl_weight": (ranking, "kl_weight", 0.3),
                "dataset_ranking.top_fraction": (ranking, "top_fraction", 0.25),
                "dataset_ranking.minimum_coverage": (ranking, "minimum_coverage", 0.90),
                "dataset_ranking.high_risk_replay_fraction": (
                    ranking,
                    "high_risk_replay_fraction",
                    0.50,
                ),
                "proxy_agreement.minimum_spearman": (
                    agreement,
                    "minimum_spearman",
                    0.40,
                ),
                "proxy_agreement.minimum_top_overlap": (
                    agreement,
                    "minimum_top_overlap",
                    0.50,
                ),
                "proxy_agreement.minimum_coverage": (
                    agreement,
                    "minimum_coverage",
                    0.90,
                ),
            }
            if spec_id in {
                DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
                DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
                DGCAIP_RELAXED_GATE_V4_SPEC_ID,
            }:
                expected_values.update(
                    {
                        "strict_route.repair_floor_fraction": (
                            strict_route,
                            "repair_floor_fraction",
                            0.0,
                        ),
                        "strict_route.max_repair_norm_ratio": (
                            strict_route,
                            "max_repair_norm_ratio",
                            0.0,
                        ),
                        "strict_route.minimum_target_progress": (
                            strict_route,
                            "minimum_target_progress",
                            0.60,
                        ),
                        "strict_route.svd_relative_tolerance": (
                            strict_route,
                            "svd_relative_tolerance",
                            1.0e-6,
                        ),
                    }
                )
            else:
                expected_values.update(
                    {
                        "strict_route.repair_floor_fraction": (
                            strict_route,
                            "repair_floor_fraction",
                            0.05,
                        ),
                        "strict_route.max_repair_norm_ratio": (
                            strict_route,
                            "max_repair_norm_ratio",
                            0.25,
                        ),
                    }
                )
            for name, (section, key, expected) in expected_values.items():
                if float(section.get(key, float("nan"))) != expected:
                    raise ValueError("%s must remain %s." % (name, expected))
            expected_projection_iterations = (
                128
                if spec_id
                in {
                    DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
                    DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
                    DGCAIP_RELAXED_GATE_V4_SPEC_ID,
                }
                else 64
            )
            if int(strict_route.get("max_projection_iterations", -1)) != (
                expected_projection_iterations
            ):
                raise ValueError(
                    "strict_route.max_projection_iterations must remain %d."
                    % expected_projection_iterations
                )
            if spec_id == DGCAIP_STRICT_ROUTE_V2_SPEC_ID and str(
                strict_route.get("mode", "")
            ) != "nonworsening_target_progress_v2":
                raise ValueError("Strict-route v2 mode mismatch.")
            if spec_id == DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID and str(
                strict_route.get("mode", "")
            ) != "component_aligned_target_progress_v3":
                raise ValueError("Component-aligned route v3 mode mismatch.")
            if spec_id == DGCAIP_RELAXED_GATE_V4_SPEC_ID:
                if str(strict_route.get("mode", "")) != (
                    "component_aligned_target_progress_v3"
                ):
                    raise ValueError("Relaxed promotion gate v4 route mode mismatch.")
                if float(
                    strict_route.get("nonlinear_comparison_tolerance", float("nan"))
                ) != 1.0e-6:
                    raise ValueError(
                        "v4 nonlinear comparison tolerance must remain 1e-6."
                    )
                if float(
                    strict_route.get("minimum_accepted_update_ratio", float("nan"))
                ) != 0.50:
                    raise ValueError(
                        "v4 minimum accepted-update ratio must remain 0.50."
                    )
            snapshots = config["model"].get("protection_surrogate_snapshots")
            expected_snapshot_ids = (
                ["v3"]
                if run_mode == "short_victim_risk_scan"
                else ["e1", "e5", "e20"]
            )
            if not isinstance(snapshots, list) or [
                str(item.get("id", "")) for item in snapshots
            ] != expected_snapshot_ids:
                raise ValueError(
                    "Protection snapshot IDs must be %s."
                    % "/".join(expected_snapshot_ids)
                )
            for item in snapshots:
                if not str(item.get("checkpoint", "")).strip():
                    raise ValueError("Protection snapshot checkpoint is missing.")
                digest = str(item.get("sha256", ""))
                if len(digest) != 64 or set(digest) == {"0"}:
                    raise ValueError("Protection snapshot SHA256 is invalid.")
            if run_mode == "strict_mechanism":
                for key in ("risk_bank", "replay_manifest"):
                    if not str(ranking.get(key, "")).strip():
                        raise ValueError("Strict mechanism %s path is missing." % key)
                for key in (
                    "risk_bank_file_sha256",
                    "risk_bank_canonical_sha256",
                    "replay_manifest_file_sha256",
                ):
                    digest = str(ranking.get(key, ""))
                    if len(digest) != 64 or set(digest) == {"0"}:
                        raise ValueError("Strict mechanism %s is invalid." % key)
            if run_mode == "proxy_victim_audit":
                for role in ("proxy", "victim"):
                    if not str(
                        agreement.get("%s_risk_bank" % role, "")
                    ).strip():
                        raise ValueError("Proxy-victim audit risk-bank path is missing.")
                    digest = str(
                        agreement.get("%s_risk_bank_file_sha256" % role, "")
                    )
                    if len(digest) != 64 or set(digest) == {"0"}:
                        raise ValueError("Proxy-victim audit file SHA256 is invalid.")
            if run_mode == "short_victim_risk_scan":
                if not str(dgcaip.get("source_carrier_state", "")).strip():
                    raise ValueError("Short-victim scan source carrier state is missing.")
                digest = str(dgcaip.get("source_carrier_state_sha256", ""))
                if len(digest) != 64 or set(digest) == {"0"}:
                    raise ValueError("Short-victim source carrier SHA256 is invalid.")
        elif float(dgcaip.get("protection_ratio", float("nan"))) != 0.25:
            raise ValueError("DG-CAIP protection_ratio must remain 0.25.")
        if int(dgcaip.get("minimum_rank_instances", -1)) != 4:
            raise ValueError("DG-CAIP minimum_rank_instances must remain 4.")
        split_hash = str(dgcaip.get("expected_split_sha256", ""))
        if len(split_hash) != 64 or set(split_hash) == {"0"}:
            raise ValueError("DG-CAIP requires expected_split_sha256.")
        if not (
            spec_id == DGCAIP_DATASET_CGR_PROXY_SPEC_ID
            and run_mode == "short_victim_risk_scan"
        ):
            if not str(dgcaip.get("source_p1_state", "")).strip():
                raise ValueError("DG-CAIP requires source_p1_state.")
            source_state_hash = str(dgcaip.get("source_p1_state_sha256", ""))
            if len(source_state_hash) != 64 or set(source_state_hash) == {"0"}:
                raise ValueError("DG-CAIP requires source_p1_state_sha256.")
        if run_mode in {
            "mechanism",
            "r3_diag",
            "p1_determinism_audit",
            "production_e20",
        }:
            if not str(dgcaip.get("d0_report", "")).strip():
                raise ValueError("DG-CAIP mechanism requires a passed D0 report.")
            d0_hash = str(dgcaip.get("d0_report_sha256", ""))
            if len(d0_hash) != 64 or set(d0_hash) == {"0"}:
                raise ValueError("DG-CAIP mechanism requires d0_report_sha256.")
            if not str(dgcaip.get("source_p1_metrics", "")).strip():
                raise ValueError("DG-CAIP mechanism requires source_p1_metrics.")
            source_metrics_hash = str(
                dgcaip.get("source_p1_metrics_sha256", "")
            )
            if len(source_metrics_hash) != 64 or set(source_metrics_hash) == {"0"}:
                raise ValueError(
                    "DG-CAIP mechanism requires source_p1_metrics_sha256."
                )
            replay_tolerances = {
                "p1_replay_absolute_tolerance": 1.0e-6,
                "p1_replay_relative_tolerance": 1.0e-4,
            }
            for key, expected in replay_tolerances.items():
                if float(dgcaip.get(key, float("nan"))) != expected:
                    raise ValueError("DG-CAIP %s must remain %s." % (key, expected))
        if (
            spec_id in {DGCAIP_R4_DIAG_SPEC_ID, DGCAIP_P4_E20_SPEC_ID}
            or spec_id in DGCAIP_AUDIT_SPEC_IDS
        ):
            if str(dgcaip.get("expected_d0_spec_id", "")) != DGCAIP_SPEC_ID:
                raise ValueError(
                    "DG-CAIP audit requires expected_d0_spec_id="
                    + DGCAIP_SPEC_ID
                    + "."
                )
        if spec_id == DGCAIP_P4_E20_SPEC_ID:
            if config["runtime"].get("strict_determinism") is not True:
                raise ValueError("DG-CAIP P4 production requires strict determinism.")
            if int(config["mechanism"].get("batch_size", -1)) != 4:
                raise ValueError("DG-CAIP P4 production freezes batch_size=4.")
            if int(config["mechanism"].get("calibration_batches", -1)) != 16:
                raise ValueError("DG-CAIP P4 production freezes 16 calibration batches.")
            if int(config["mechanism"].get("heldout_batches", -1)) != 24:
                raise ValueError("DG-CAIP P4 production freezes 24 held-out batches.")
            if float(config["mechanism"].get("max_seconds", -1)) != 1200.0:
                raise ValueError("DG-CAIP P4 mechanism hard cap must remain 1200 seconds.")
            repair_report = str(dgcaip.get("repair_report", "")).strip()
            if not repair_report:
                raise ValueError("DG-CAIP P4 production requires repair_report.")
            if str(dgcaip.get("repair_report_sha256", "")).lower() != (
                "f05f5f9ca255083d3697af69ad47127c28f8349219e1cf50530edd632bc91b3b"
            ):
                raise ValueError("DG-CAIP P4 repair report hash mismatch.")
        if spec_id in DGCAIP_DIAG_SPEC_IDS:
            diagnostics = dgcaip.get("r3_diagnostics")
            if not isinstance(diagnostics, Mapping) or diagnostics.get("enabled") is not True:
                raise ValueError("R3-DIAG requires r3_diagnostics.enabled=true.")
            if int(config["mechanism"].get("batch_size", -1)) != 4:
                raise ValueError("R3-DIAG freezes mechanism batch_size=4.")
            if int(config["mechanism"].get("calibration_batches", -1)) != 16:
                raise ValueError("R3-DIAG freezes 16 calibration batches.")
            if int(config["mechanism"].get("heldout_batches", -1)) != 24:
                raise ValueError("R3-DIAG freezes 24 held-out batches.")
            if float(config["mechanism"].get("max_seconds", -1)) != 600.0:
                raise ValueError("R3-DIAG hard cap must remain 600 seconds.")
        if spec_id in DGCAIP_AUDIT_SPEC_IDS:
            audit = config.get("audit")
            if not isinstance(audit, Mapping) or audit.get("enabled") is not True:
                raise ValueError("P1 determinism audit requires audit.enabled=true.")
            resize_repair = spec_id == DGCAIP_P1_DET_RESIZE_FIX_SPEC_ID
            frozen_audit = {
                "first_batch_index": 0,
                "paired_repeats": 2,
                "total_hard_cap_seconds": 480 if resize_repair else 300,
                "max_artifact_bytes": 104857600,
            }
            for key, expected in frozen_audit.items():
                if int(audit.get(key, -1)) != expected:
                    raise ValueError("P1 determinism audit freezes audit.%s=%s." % (key, expected))
            if audit.get("zero_parameter_updates") is not True:
                raise ValueError("P1 determinism audit requires zero_parameter_updates=true.")
            expected_baseline = (
                "b2fa96f98ea88d6b347bbbf751768a06e983d47c"
                if resize_repair
                else "4eb064ade919fecec6d1466900442e9f9a9a2bf5"
            )
            if str(audit.get("baseline_commit", "")) != expected_baseline:
                raise ValueError("P1 determinism audit baseline commit changed.")
            expected_normal_lanes = (
                ("reset",) if resize_repair else ("shared", "reset", "fresh")
            )
            if tuple(audit.get("normal_lanes", ())) != expected_normal_lanes:
                raise ValueError("P1 determinism audit normal lanes changed.")
            if tuple(audit.get("strict_lanes", ())) != ("fresh",):
                raise ValueError("P1 determinism audit strict lanes changed.")
            if float(audit.get("absolute_tolerance", float("nan"))) != 1.0e-6:
                raise ValueError("P1 determinism absolute tolerance changed.")
            if float(audit.get("relative_tolerance", float("nan"))) != 1.0e-4:
                raise ValueError("P1 determinism relative tolerance changed.")
            if int(config["mechanism"].get("batch_size", -1)) != 4:
                raise ValueError("P1 determinism audit freezes batch_size=4.")
            if int(config["mechanism"].get("calibration_batches", -1)) != 16:
                raise ValueError("P1 determinism audit freezes 16 calibration batches.")
            if int(config["mechanism"].get("heldout_batches", -1)) != 24:
                raise ValueError("P1 determinism audit freezes 24 held-out batches.")
            if float(config["mechanism"].get("max_seconds", -1)) != 300.0:
                raise ValueError("P1 determinism audit hard cap must remain 300 seconds.")
            expected_exp_id = (
                "TAUSB-SDH-DGCAIP-S0-P1-DET-RESIZE-FIX"
                if resize_repair
                else "TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT"
            )
            if str(config["spec"].get("exp_id", "")) != expected_exp_id:
                raise ValueError("P1 determinism audit ExpID changed.")
            if not str(config["runtime"].get("artifact_root", "")).endswith(
                "/" + expected_exp_id
            ):
                raise ValueError("P1 determinism audit artifact root changed.")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_secret_bank(config: Mapping[str, Any], base: Path) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    manifest_path = _resolve(base, str(config["secrets"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_manifest_sha256 = _canonical_json_sha256(manifest)
    if actual_manifest_sha256 != str(config["secrets"]["manifest_sha256"]):
        raise ValueError("Secret manifest hash does not match SDH config.")
    records = manifest["records"]
    pretrain = [item for item in records if item["role"] == "pretrain"]
    primary = [item for item in records if item["source_id"] == config["secrets"]["primary_id"]]
    if len(pretrain) < 3 or len(primary) != 1:
        raise ValueError("SDH requires at least 3 pretrain secrets and exactly one primary.")
    if primary[0]["source_sha256"] != str(config["secrets"]["primary_source_sha256"]):
        raise ValueError("Primary source hash does not match SDH config.")
    selected = pretrain[:3] + primary
    tensors = []
    repository_root = manifest_path.parents[3]
    for record in selected:
        path = _resolve(repository_root, record["processed_file"])
        if _file_sha256(path) != record["processed_png_sha256"]:
            raise ValueError("Processed secret PNG hash mismatch: %s" % record["source_id"])
        tensors.append(_load_secret(path, record["float32_chw_0_1_le_sha256"]))
    return torch.cat(tensors), 3, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "records": selected,
    }


def _validate_e2e_v0_runtime_inputs(
    config: Mapping[str, Any],
    *,
    config_base: Path,
    image_dir: Path,
    label_dir: Path,
    hiding_state: Mapping[str, Any],
    primary_secret: torch.Tensor,
) -> Dict[str, Any]:
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)
    label_paths = sorted(label_dir.glob("*.txt"), key=lambda path: path.name)
    expected_count = int(config["dataset"]["expected_train_images"])
    if len(image_paths) != expected_count or len(label_paths) != expected_count:
        raise ValueError("E2E V0 training image/label counts do not match the frozen input.")
    image_manifest = _path_size_manifest_sha256(image_paths)
    label_manifest = _path_content_manifest_sha256(label_paths)
    if image_manifest != str(config["dataset"]["train_image_manifest_sha256"]).lower():
        raise ValueError("E2E V0 training image manifest file hash mismatch.")
    if label_manifest != str(config["dataset"]["train_label_manifest_sha256"]).lower():
        raise ValueError("E2E V0 training label manifest file hash mismatch.")

    surrogate_path = _resolve(
        config_base, str(config["model"]["surrogate_checkpoint"])
    )
    surrogate_hash = _file_sha256(surrogate_path)
    if surrogate_hash != str(config["model"]["surrogate_checkpoint_sha256"]).lower():
        raise ValueError("E2E V0 surrogate checkpoint file hash mismatch.")

    secret_bank, primary_index, secret_meta = _load_secret_bank(config, config_base)
    repository_root = Path(secret_meta["manifest_path"]).parents[3]
    for record in secret_meta["records"]:
        source_path = _resolve(repository_root, str(record["source_file"]))
        if _file_sha256(source_path) != str(record["source_sha256"]).lower():
            raise ValueError("Secret source file hash mismatch: %s" % record["source_id"])
    if hiding_state.get("secret_manifest_sha256") != secret_meta["manifest_sha256"]:
        raise ValueError("E2E V0 hiding checkpoint secret manifest mismatch.")
    if int(hiding_state.get("primary_index", -1)) != int(primary_index):
        raise ValueError("E2E V0 hiding checkpoint primary secret index mismatch.")
    expected_primary = secret_bank[primary_index : primary_index + 1]
    actual_primary = primary_secret.detach().cpu().float()
    if _float_tensor_sha256(actual_primary) != str(
        config["secrets"]["primary_tensor_sha256"]
    ).lower():
        raise ValueError("E2E V0 hiding checkpoint primary tensor hash mismatch.")
    if not torch.equal(actual_primary, expected_primary):
        raise ValueError("E2E V0 hiding checkpoint primary tensor differs from manifest.")
    return {
        "train_image_count": len(image_paths),
        "train_label_count": len(label_paths),
        "train_image_manifest_sha256": image_manifest,
        "train_label_manifest_sha256": label_manifest,
        "surrogate_checkpoint_sha256": surrogate_hash,
        "secret_manifest_sha256": secret_meta["manifest_sha256"],
        "primary_secret_tensor_sha256": _float_tensor_sha256(actual_primary),
    }


def run_hiding_pilot(config: Mapping[str, Any], *, config_base: Path) -> Dict[str, Any]:
    validate_sdh_experiment_config(config)
    start = time.monotonic()
    device = torch.device(str(config["runtime"]["device"]))
    dataset_root = _resolve(config_base, str(config["dataset"]["root"]))
    image_dir = dataset_root / str(config["dataset"]["train_images"])
    label_dir = dataset_root / str(config["dataset"]["train_labels"])
    batch_size = int(config["hiding"]["batch_size"])
    calibration_count = int(config["mechanism"]["calibration_batches"]) * int(
        config["mechanism"]["batch_size"]
    )
    heldout_count = int(config["mechanism"]["heldout_batches"]) * int(
        config["mechanism"]["batch_size"]
    )
    paths = _person_paths(image_dir, label_dir, 14)
    if len(paths) != 6095:
        raise ValueError("SDH input person-image count must be exactly 6095.")
    split = deterministic_person_split(
        paths,
        label_dir=label_dir,
        target_class_id=14,
        calibration_count=calibration_count,
        heldout_count=heldout_count,
        seed=0,
    )
    split_hash = _split_hash(split)
    secrets, primary_index, secret_meta = _load_secret_bank(config, config_base)
    secrets = secrets.to(device)
    carrier = SemanticHidingCarrier(
        input_size=256,
        width=64,
        coupling_blocks=4,
        epsilon=16.0 / 255.0,
        hf_subband_scale=float(config["hiding"].get("hf_subband_scale", 1.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        carrier.parameters(), lr=float(config["hiding"]["learning_rate"])
    )
    calibration_hosts = [
        _first_person_host(path, label_dir, device) for path in split["calibration"]
    ]
    history = []
    steps = int(config["hiding"]["steps"])
    rng = random.Random(0)
    for step in range(steps):
        _time_guard(start, float(config["hiding"]["max_seconds"]), "hiding pilot")
        indices = [rng.randrange(len(calibration_hosts)) for _ in range(batch_size)]
        hosts = torch.stack([calibration_hosts[index] for index in indices])
        secret_indices = torch.tensor(
            [rng.randrange(primary_index) for _ in range(batch_size)], device=device
        )
        batch_secrets = secrets[secret_indices]
        metrics = hiding_pretrain_step(
            carrier,
            optimizer,
            hosts,
            batch_secrets,
            cover_weight=float(config["hiding"]["cover_weight"]),
        )
        metrics["step"] = step + 1
        history.append(metrics)

    heldout_hosts = [
        _first_person_host(path, label_dir, device) for path in split["heldout"]
    ]
    recovered_all = []
    true_all = []
    index_all = []
    primary_deltas = []
    carrier.eval()
    with torch.no_grad():
        for offset in range(0, len(heldout_hosts), batch_size):
            _time_guard(start, float(config["hiding"]["max_seconds"]), "hiding pilot")
            hosts = torch.stack(heldout_hosts[offset : offset + batch_size])
            for secret_index in range(secrets.shape[0]):
                selected = secrets[secret_index : secret_index + 1].expand(
                    hosts.shape[0], -1, -1, -1
                )
                output = carrier(hosts, selected)
                recovered_all.append(output.recovered_secret.cpu())
                true_all.append(selected.cpu())
                index_all.append(
                    torch.full((hosts.shape[0],), secret_index, dtype=torch.long)
                )
                if secret_index == primary_index:
                    primary_deltas.append(output.delta.cpu())
    hiding_metrics = compute_hiding_metrics(
        torch.cat(recovered_all),
        torch.cat(true_all),
        secrets.detach().cpu(),
        torch.cat(index_all),
        torch.cat(primary_deltas),
        primary_index,
        support_outside_max=0.0,
    )
    from ultralytics import YOLO

    wrapper = YOLO(str(_resolve(config_base, str(config["model"]["surrogate_checkpoint"]))))
    surrogate = wrapper.model.to(device).eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    carrier.freeze_for_detector_optimization()
    carrier.eval()
    with DetectorLFCExtractor(surrogate, eps=16.0 / 255.0) as extractor:
        calibration_probe = _secret_dlfc_features(
            split["calibration"],
            label_dir=label_dir,
            carrier=carrier,
            primary_secret=secrets[primary_index : primary_index + 1],
            extractor=extractor,
            device=device,
            batch_size=batch_size,
            start=start,
            max_seconds=float(config["hiding"]["max_seconds"]),
        )
        heldout_probe = _secret_dlfc_features(
            split["heldout"],
            label_dir=label_dir,
            carrier=carrier,
            primary_secret=secrets[primary_index : primary_index + 1],
            extractor=extractor,
            device=device,
            batch_size=batch_size,
            start=start,
            max_seconds=float(config["hiding"]["max_seconds"]),
        )
    leakage = _leakage_probe(calibration_probe, heldout_probe)
    gate = evaluate_hiding_gate(
        hiding_metrics,
        rms_diversity_required=bool(config["hiding"].get("rms_cv_gate_enabled", True)),
    )
    gate["checks"]["dlfc_leakage_probe"] = bool(leakage["pass"])
    gate["required_checks"].append("dlfc_leakage_probe")
    gate["pass"] = bool(gate["pass"] and leakage["pass"])
    gate["leakage_probe"] = leakage
    gate["status"] = "pass" if gate["pass"] else "fail"
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"])) / "hiding"
    artifact_root.mkdir(parents=True, exist_ok=False)
    checkpoint = artifact_root / "hiding_checkpoint.pt"
    torch.save(
        {
            "schema": "tausb.sdh-hiding-checkpoint.v1",
            "carrier_state": {name: value.detach().cpu() for name, value in carrier.state_dict().items()},
            "architecture_sha256": carrier.architecture_sha256(),
            "hf_subband_scale": carrier.hf_subband_scale,
            "split_hash": split_hash,
            "secret_manifest_sha256": secret_meta["manifest_sha256"],
            "primary_index": primary_index,
            "primary_secret": secrets[primary_index : primary_index + 1].detach().cpu(),
            "history": history,
        },
        checkpoint,
    )
    result = {
        "schema": "tausb.sdh-hiding-pilot.v1",
        "spec_id": config["spec"]["spec_id"],
        "split_hash": split_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "elapsed_seconds": time.monotonic() - start,
        "history_first": history[0],
        "history_last": history[-1],
        "gate": gate,
    }
    _write_json(artifact_root / "hiding_metrics.json", result)
    _write_json(
        artifact_root / "split_manifest.json",
        {
            "split_hash": split_hash,
            "calibration": [path.stem for path in split["calibration"]],
            "heldout": [path.stem for path in split["heldout"]],
        },
    )
    return result


def _load_hiding_checkpoint(
    config: Mapping[str, Any],
    *,
    config_base: Path,
    device: torch.device,
) -> Tuple[SemanticHidingCarrier, torch.Tensor, Mapping[str, Any]]:
    spec_id = str(config["spec"].get("spec_id", ""))
    feasibility_mode = spec_id == E2E_V0_SPEC_ID or spec_id in DGCAIP_SPEC_IDS
    root_value = (
        config["hiding"].get("source_artifact_root", "")
        if feasibility_mode
        else config["runtime"]["artifact_root"]
    )
    root = _resolve(config_base, str(root_value))
    metrics_path = root / "hiding" / "hiding_metrics.json"
    checkpoint_path = root / "hiding" / "hiding_checkpoint.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Required hiding artifacts are missing before mechanism.")
    if feasibility_mode:
        expected_metrics_hash = str(
            config["hiding"].get("source_metrics_sha256", "")
        ).lower()
        if expected_metrics_hash != E2E_V0_R2_METRICS_SHA256:
            raise ValueError("E2E V0 hiding metrics hash is not the frozen r2 hash.")
        if _file_sha256(metrics_path) != expected_metrics_hash:
            raise ValueError("E2E V0 hiding metrics file hash mismatch.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate = metrics.get("gate", {})
    if feasibility_mode:
        if config["hiding"].get("allow_failed_scientific_gates") is not True:
            raise ValueError("E2E V0 failed-scientific-gate loading was not enabled.")
        checks = gate.get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError("E2E V0 hiding checks are missing.")
        failed_checks = frozenset(name for name, passed in checks.items() if passed is not True)
        if failed_checks != E2E_V0_ALLOWED_FAILED_HIDING_CHECKS:
            raise ValueError(
                "E2E V0 hiding failures differ from frozen r2: %s."
                % sorted(failed_checks)
            )
        if gate.get("pass") is not False or gate.get("status") != "fail":
            raise ValueError("E2E V0 must preserve the frozen r2 hiding FAIL decision.")
    elif gate.get("pass") is not True:
        raise ValueError("Hiding gate did not pass; mechanism is forbidden.")
    checkpoint_hash = _file_sha256(checkpoint_path)
    if checkpoint_hash != metrics.get("checkpoint_sha256"):
        raise ValueError("Hiding checkpoint hash mismatch.")
    if feasibility_mode:
        expected_checkpoint_hash = str(
            config["hiding"].get("source_checkpoint_sha256", "")
        ).lower()
        if expected_checkpoint_hash != E2E_V0_R2_CHECKPOINT_SHA256:
            raise ValueError("E2E V0 hiding checkpoint hash is not the frozen r2 hash.")
        if checkpoint_hash != expected_checkpoint_hash:
            raise ValueError("E2E V0 hiding checkpoint file hash mismatch.")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if state.get("schema") != "tausb.sdh-hiding-checkpoint.v1":
        raise ValueError("Unsupported hiding checkpoint schema.")
    carrier = SemanticHidingCarrier(
        input_size=256,
        width=64,
        coupling_blocks=4,
        epsilon=16.0 / 255.0,
        hf_subband_scale=float(state.get("hf_subband_scale", 1.0)),
    )
    if carrier.hf_subband_scale != float(config["hiding"].get("hf_subband_scale", 1.0)):
        raise ValueError("Hiding checkpoint subband scale does not match config.")
    carrier.load_state_dict(state["carrier_state"], strict=True)
    if carrier.architecture_sha256() != state["architecture_sha256"]:
        raise ValueError("Hiding checkpoint architecture hash mismatch.")
    carrier.to(device)
    carrier.freeze_for_detector_optimization()
    primary = torch.as_tensor(state["primary_secret"], dtype=torch.float32, device=device)
    if feasibility_mode:
        state = dict(state)
        state["e2e_v0_hiding_provenance"] = {
            "evidence_scope": "end_to_end_feasibility_not_formal_method",
            "hiding_gate_passed": False,
            "failed_hiding_checks": sorted(E2E_V0_ALLOWED_FAILED_HIDING_CHECKS),
            "hiding_metrics_sha256": E2E_V0_R2_METRICS_SHA256,
            "hiding_checkpoint_sha256": E2E_V0_R2_CHECKPOINT_SHA256,
        }
    return carrier, primary, state


def _clone_detector_carrier(
    source: SemanticHidingCarrier, device: torch.device
) -> SemanticHidingCarrier:
    clone = SemanticHidingCarrier(
        input_size=source.input_size,
        width=source.width,
        coupling_blocks=source.coupling_blocks,
        epsilon=source.epsilon,
        hf_subband_scale=source.hf_subband_scale,
    ).to(device)
    clone.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    clone.freeze_for_detector_optimization()
    return clone


def _load_saved_p1_carrier(
    path: Path,
    *,
    base_carrier: SemanticHidingCarrier,
    device: torch.device,
) -> SemanticHidingCarrier:
    if not path.is_file():
        raise FileNotFoundError("P1 carrier state is missing: %s" % path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if not isinstance(saved, Mapping) or saved.get("arm_id") != "P1":
        raise ValueError("Feasibility persistence requires the saved P1 arm state.")
    carrier_state = saved.get("carrier_state")
    if not isinstance(carrier_state, Mapping):
        raise ValueError("Saved P1 carrier state must be a mapping.")
    if any(
        not torch.is_tensor(value) or not torch.isfinite(value).all()
        for value in carrier_state.values()
    ):
        raise ValueError("Saved P1 carrier state contains an invalid tensor.")
    carrier = _clone_detector_carrier(base_carrier, device)
    carrier.load_state_dict(carrier_state, strict=True)
    carrier.freeze_for_detector_optimization()
    return carrier


def _flatten_autograd_norm(
    loss: torch.Tensor, parameters: Sequence[torch.Tensor], *, retain_graph: bool
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flattened = torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )
    if not torch.isfinite(flattened).all():
        raise ValueError("Target component produced a non-finite omega gradient.")
    return float(flattened.norm().detach())


def _component_losses(
    observation: SDHObservation,
    dlfc_bank: DetectorLFCPrototypeBank,
    cicr_bank: FrozenInstanceCICRBank,
) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
    dlfc = dlfc_bank.compute(observation.canonical_dlfc_features)
    cicr = cicr_bank.compute(observation.target_residuals, energy_weight=1.0)
    components = {
        "easy": observation.route.loss,
        "reveal": observation.reveal_loss,
        "rms": observation.rms_loss,
        "dlfc": dlfc.loss,
        "cicr": cicr.direction_loss,
        "floor": cicr.energy_floor_loss,
    }
    return components, dlfc, cicr


def _copy_parameters_(
    parameters: Sequence[torch.Tensor], values: Sequence[torch.Tensor]
) -> None:
    if len(parameters) != len(values):
        raise ValueError("Omega parameter/value count mismatch.")
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            if parameter.shape != value.shape:
                raise ValueError("Omega candidate shape mismatch.")
            parameter.copy_(value)


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _q25(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, 0.25)) if finite else float("nan")


def _summarize_arm(
    observations: Sequence[SDHObservation],
    dlfc_bank: DetectorLFCPrototypeBank,
    cicr_bank: FrozenInstanceCICRBank,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    dlfc_cosines = []
    cicr_cosines = []
    coverages = []
    zero_ratios = []
    floor_pass = []
    nla_losses = []
    probability_by_class: Dict[int, List[float]] = {}
    route_losses = []
    deltas = []
    per_scale_pooling = [[] for _ in range(3)]
    for observation in observations:
        dlfc = dlfc_bank.compute(observation.canonical_dlfc_features)
        cicr = cicr_bank.compute(observation.target_residuals)
        dlfc_cosines.extend(
            float(value) for tensor in dlfc.per_scale_cosine for value in tensor.detach().cpu()
        )
        cicr_cosines.extend(
            float(value)
            for value in cicr.per_instance_cosine.detach().cpu()
            if math.isfinite(float(value))
        )
        coverages.append(cicr.valid_instance_coverage)
        zero_ratios.append(cicr.zero_norm_ratio)
        floor_pass.append(1.0 - cicr.low_energy_ratio)
        for scale, count in enumerate(cicr.per_scale_valid_count):
            per_scale_pooling[scale].append(count / max(cicr.total_instance_count, 1))
        nla_losses.append(float(observation.nla.loss.detach()))
        route_losses.append(float(observation.route.loss.detach()))
        for class_id, drop in observation.per_class_probability_drop.items():
            probability_by_class.setdefault(class_id, []).append(drop)
        deltas.append(torch.cat(observation.rendered.canonical_deltas).detach().cpu())
    summary = {
        "dlfc_cosine_median": _median(dlfc_cosines),
        "dlfc_cosine_q25": _q25(dlfc_cosines),
        "cicr_cosine_median": _median(cicr_cosines),
        "cicr_cosine_q25": _q25(cicr_cosines),
        "valid_instance_coverage": _median(coverages),
        "per_scale_pooling_coverage": [_median(values) for values in per_scale_pooling],
        "zero_residual_ratio": _median(zero_ratios),
        "floor_pass_ratio": _median(floor_pass),
        "nla_macro_loss": _median(nla_losses),
        "route_loss": _median(route_losses),
        "probability_drop_by_class": {
            str(class_id): _median(values) for class_id, values in sorted(probability_by_class.items())
        },
        "probability_drop_macro": _median(
            _median(values) for values in probability_by_class.values()
        ),
    }
    return summary, torch.cat(deltas)


def run_mechanism_pilot(config: Mapping[str, Any], *, config_base: Path) -> Dict[str, Any]:
    validate_sdh_experiment_config(config)
    if str(config["spec"].get("spec_id", "")) in DGCAIP_AUDIT_SPEC_IDS:
        raise ValueError(
            "P1 determinism audit must use the dedicated zero-update runner."
        )
    if str(config["spec"].get("spec_id", "")) in {
        DGCAIP_DATASET_CGR_PROXY_SPEC_ID,
        DGCAIP_STRICT_ROUTE_V2_SPEC_ID,
        DGCAIP_COMPONENT_ROUTE_V3_SPEC_ID,
        DGCAIP_RELAXED_GATE_V4_SPEC_ID,
    }:
        from .dgcaip_dataset_risk_experiment import run_dataset_cgr_proxy_stage

        return run_dataset_cgr_proxy_stage(config, config_base=config_base)
    if str(config["spec"].get("spec_id", "")) in DGCAIP_SPEC_IDS:
        from .dgcaip_experiment import run_dgcaip_pilot

        return run_dgcaip_pilot(config, config_base=config_base)
    start = time.monotonic()
    max_seconds = float(config["mechanism"]["max_seconds"])
    device = torch.device(str(config["runtime"]["device"]))
    base_carrier, primary_secret, hiding_state = _load_hiding_checkpoint(
        config, config_base=config_base, device=device
    )
    dataset_root = _resolve(config_base, str(config["dataset"]["root"]))
    image_dir = dataset_root / str(config["dataset"]["train_images"])
    label_dir = dataset_root / str(config["dataset"]["train_labels"])
    runtime_input_hashes = None
    if str(config["spec"].get("spec_id", "")) == E2E_V0_SPEC_ID:
        runtime_input_hashes = _validate_e2e_v0_runtime_inputs(
            config,
            config_base=config_base,
            image_dir=image_dir,
            label_dir=label_dir,
            hiding_state=hiding_state,
            primary_secret=primary_secret,
        )
    batch_size = int(config["mechanism"]["batch_size"])
    paths = _person_paths(image_dir, label_dir, 14)
    if (
        str(config["spec"].get("spec_id", "")) == E2E_V0_SPEC_ID
        and len(paths) != int(config["dataset"]["expected_person_images"])
    ):
        raise ValueError("E2E V0 person-image count does not match the frozen input.")
    split = deterministic_person_split(
        paths,
        label_dir=label_dir,
        target_class_id=14,
        calibration_count=int(config["mechanism"]["calibration_batches"]) * batch_size,
        heldout_count=int(config["mechanism"]["heldout_batches"]) * batch_size,
        seed=0,
    )
    split_hash = _split_hash(split)
    if split_hash != hiding_state["split_hash"]:
        raise ValueError("Mechanism split does not match hiding checkpoint.")
    from ultralytics import YOLO

    wrapper = YOLO(str(_resolve(config_base, str(config["model"]["surrogate_checkpoint"]))))
    model = wrapper.model.to(device).eval()
    engine = SDHObservationEngine(
        model,
        target_class_id=14,
        num_classes=20,
        epsilon=16.0 / 255.0,
        assignment_topk=int(config["mechanism"]["assignment_topk"]),
        pag_layer_ratios=config["mechanism"]["pag_layer_ratios"],
        pag_min_pos=config["mechanism"]["pag_min_pos"],
        box_teacher_weight=float(config["mechanism"]["box_teacher_weight"]),
    )
    calibration_batches = _batches(split["calibration"], batch_size)
    heldout_batches = _batches(split["heldout"], batch_size)
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"])) / "mechanism"
    artifact_root.mkdir(parents=True, exist_ok=False)

    def load(paths_batch: Sequence[Path]):
        return load_sdh_batch(
            paths_batch,
            label_dir=label_dir,
            image_size=640,
            target_class_id=14,
            device=device,
        )

    try:
        initial_observations = []
        with torch.no_grad():
            for paths_batch in calibration_batches:
                _time_guard(start, max_seconds, "mechanism pilot")
                initial_observations.append(
                    engine.observe(load(paths_batch), base_carrier, primary_secret)
                )
        dlfc_bank = DetectorLFCPrototypeBank()
        dlfc_bank.fit(
            [item.canonical_dlfc_features for item in initial_observations],
            split="calibration",
        )
        cicr_bank = FrozenInstanceCICRBank(energy_floor_multiplier=0.5)
        cicr_bank.fit(
            [item.target_residuals for item in initial_observations],
            split="calibration",
        )

        calibration_carrier = _clone_detector_carrier(base_carrier, device)
        omega = adapter_parameters(calibration_carrier)
        gradient_norms: Dict[str, List[float]] = {
            name: [] for name in ("easy", "reveal", "rms", "dlfc", "cicr", "floor")
        }
        warmup_count = int(config["mechanism"]["weight_calibration_batches"])
        warmup_observations = []
        for paths_batch in calibration_batches[:warmup_count]:
            observation = engine.observe(load(paths_batch), calibration_carrier, primary_secret)
            warmup_observations.append(observation)
            components, _, _ = _component_losses(observation, dlfc_bank, cicr_bank)
            for name, loss in components.items():
                gradient_norms[name].append(
                    _flatten_autograd_norm(loss, omega, retain_graph=True)
                )
        target_calibration = FrozenTargetGradientCalibration()
        target_weights = target_calibration.calibrate(
            gradient_norms, split="warmup"
        )

        projected_norms = []
        nla_norms = []
        for observation in warmup_observations:
            components, _, _ = _component_losses(observation, dlfc_bank, cicr_bank)
            objective = compose_sdh_target_objective(
                easy=components["easy"],
                reveal=components["reveal"],
                rms=components["rms"],
                dlfc=components["dlfc"],
                cicr=components["cicr"],
                floor=components["floor"],
                weights=target_weights,
                enable_dlfc=True,
                enable_cicr=True,
            )
            routed = route_multi_parameter_gradients(
                parameters=omega,
                target_loss=objective.loss,
                per_class_nla_losses={
                    str(key): value for key, value in observation.nla.per_class_loss.items()
                },
                nla_loss=observation.nla.loss,
                nla_weight=0.0,
            )
            if routed.nla_norm > 0:
                projected_norms.append(routed.projected_target_norm)
                nla_norms.append(routed.nla_norm)
        nla_calibration = FrozenNLAGradientCalibration(target_ratio=0.25)
        lambda_nla = nla_calibration.calibrate(
            projected_norms, nla_norms, split="warmup"
        )

        initial_heldout = []
        with torch.no_grad():
            for paths_batch in heldout_batches:
                _time_guard(start, max_seconds, "mechanism pilot")
                initial_heldout.append(
                    engine.observe(load(paths_batch), base_carrier, primary_secret)
                )
        initial_summary, initial_deltas = _summarize_arm(
            initial_heldout, dlfc_bank, cicr_bank
        )

        arms = {}
        arm_deltas = {}
        for arm_id, switches in ARM_SWITCHES.items():
            _time_guard(start, max_seconds, "mechanism pilot")
            carrier = _clone_detector_carrier(base_carrier, device)
            parameters = adapter_parameters(carrier)
            step_rows = []
            dlfc_cicr_grad_cosines = []
            backtrack_or_skip = 0
            for step in range(int(config["mechanism"]["optimization_steps"])):
                paths_batch = calibration_batches[step % len(calibration_batches)]
                batch = load(paths_batch)
                observation = engine.observe(batch, carrier, primary_secret)
                components, _, cicr = _component_losses(observation, dlfc_bank, cicr_bank)
                objective = compose_sdh_target_objective(
                    easy=components["easy"],
                    reveal=components["reveal"],
                    rms=components["rms"],
                    dlfc=components["dlfc"] if switches["dlfc"] else None,
                    cicr=components["cicr"] if switches["cicr"] else None,
                    floor=components["floor"] if switches["cicr"] else None,
                    weights=target_weights,
                    enable_dlfc=switches["dlfc"],
                    enable_cicr=switches["cicr"],
                )
                dlfc_gradient = torch.autograd.grad(
                    components["dlfc"], parameters, retain_graph=True, allow_unused=True
                )
                cicr_gradient = torch.autograd.grad(
                    components["cicr"], parameters, retain_graph=True, allow_unused=True
                )
                dlfc_flat = torch.cat([
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(parameters, dlfc_gradient)
                ])
                cicr_flat = torch.cat([
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(parameters, cicr_gradient)
                ])
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        dlfc_flat.reshape(1, -1), cicr_flat.reshape(1, -1), dim=1
                    ).detach()
                ) if float(dlfc_flat.norm()) > 0 and float(cicr_flat.norm()) > 0 else float("nan")
                dlfc_cicr_grad_cosines.append(cosine)
                if switches["cgr"]:
                    routed = route_multi_parameter_gradients(
                        parameters=parameters,
                        target_loss=objective.loss,
                        per_class_nla_losses={
                            str(key): value for key, value in observation.nla.per_class_loss.items()
                        },
                        nla_loss=observation.nla.loss,
                        nla_weight=lambda_nla if switches["nla"] else 0.0,
                    )
                    originals = tuple(parameter.detach().clone() for parameter in parameters)

                    def evaluate(candidate):
                        _copy_parameters_(parameters, candidate)
                        try:
                            with torch.no_grad():
                                current = engine.observe(batch, carrier, primary_secret)
                            return {
                                str(key): value
                                for key, value in current.per_class_probability_drop.items()
                            }
                        finally:
                            _copy_parameters_(parameters, originals)

                    backtracked = backtrack_multi_parameter_update(
                        parameters=parameters,
                        flattened_gradient=routed.gradient,
                        step_size=float(config["mechanism"]["learning_rate"]),
                        evaluate_probability_drops=evaluate,
                        tolerance=0.005,
                        max_backtracks=5,
                    )
                    if backtracked.attempts > 1 or not backtracked.accepted:
                        backtrack_or_skip += 1
                    if backtracked.accepted:
                        _copy_parameters_(parameters, backtracked.candidate)
                    step_rows.append(
                        {
                            "step": step,
                            "route_mode": routed.mode,
                            "constraint_rank": routed.rank,
                            "null_dimension": routed.null_dimension,
                            "attack_retention": routed.attack_retention,
                            "max_projected_row_dot": routed.max_projected_row_dot,
                            "max_final_row_dot": routed.max_final_row_dot,
                            "backtrack_attempts": backtracked.attempts,
                            "accepted": backtracked.accepted,
                        }
                    )
                else:
                    gradients = torch.autograd.grad(
                        objective.loss, parameters, retain_graph=False, allow_unused=True
                    )
                    candidate = tuple(
                        parameter.detach()
                        - float(config["mechanism"]["learning_rate"])
                        * (torch.zeros_like(parameter) if gradient is None else gradient.detach())
                        for parameter, gradient in zip(parameters, gradients)
                    )
                    _copy_parameters_(parameters, candidate)
                    step_rows.append({"step": step, "route_mode": "plain_target"})
            heldout_observations = []
            with torch.no_grad():
                for paths_batch in heldout_batches:
                    _time_guard(start, max_seconds, "mechanism pilot")
                    heldout_observations.append(
                        engine.observe(load(paths_batch), carrier, primary_secret)
                    )
            summary, deltas = _summarize_arm(
                heldout_observations, dlfc_bank, cicr_bank
            )
            summary.update(
                {
                    "arm_id": arm_id,
                    "switches": switches,
                    "gradient_dlfc_cicr_cosine_median": _median(dlfc_cicr_grad_cosines),
                    "gradient_dlfc_cicr_cosine_q25": _q25(dlfc_cicr_grad_cosines),
                    "backtrack_skip_ratio": backtrack_or_skip
                    / int(config["mechanism"]["optimization_steps"]),
                    "steps": step_rows,
                }
            )
            arms[arm_id] = summary
            arm_deltas[arm_id] = deltas
            torch.save(
                {"arm_id": arm_id, "carrier_state": carrier.state_dict()},
                artifact_root / ("%s_state.pt" % arm_id.lower()),
            )

        t0, t1, p0, p1 = (arms[name] for name in ("T0", "T1", "P0", "P1"))
        pattern_t1_t0 = float(
            (arm_deltas["T1"] - arm_deltas["T0"]).square().mean().sqrt()
        )
        pattern_p1_p0 = float(
            (arm_deltas["P1"] - arm_deltas["P0"]).square().mean().sqrt()
        )
        pattern_p0_initial = float(
            (arm_deltas["P0"] - initial_deltas).square().mean().sqrt()
        )
        pattern_p1_initial = float(
            (arm_deltas["P1"] - initial_deltas).square().mean().sqrt()
        )
        active_common = set(p0["probability_drop_by_class"]).intersection(
            p1["probability_drop_by_class"]
        )
        class_nonworse = (
            sum(
                p1["probability_drop_by_class"][key]
                <= p0["probability_drop_by_class"][key] + 1e-12
                for key in active_common
            )
            / max(len(active_common), 1)
        )
        p1_steps = p1["steps"]
        target_checks = {
            "dlfc_gain": t1["dlfc_cosine_median"] - t0["dlfc_cosine_median"] >= 0.10,
            "dlfc_q25": t1["dlfc_cosine_q25"] > 0,
            "cicr_gain": t1["cicr_cosine_median"] - t0["cicr_cosine_median"] >= 0.10,
            "cicr_q25": t1["cicr_cosine_q25"] > 0,
            "coverage": t1["valid_instance_coverage"] >= 0.80,
            "scale_coverage": sum(value >= 0.80 for value in t1["per_scale_pooling_coverage"]) >= 2,
            "zero_ratio": t1["zero_residual_ratio"] <= 0.20,
            "floor_pass": t1["floor_pass_ratio"] >= 0.80,
            "pattern_separation": pattern_t1_t0 >= 0.01,
            "gradient_median": t1["gradient_dlfc_cicr_cosine_median"] >= -0.10,
            "gradient_q25": t1["gradient_dlfc_cicr_cosine_q25"] >= -0.25,
        }
        protection_checks = {
            "nla_reduction": (
                p0["nla_macro_loss"] > 1.0e-12
                and p1["nla_macro_loss"] <= 0.75 * p0["nla_macro_loss"]
            ),
            "probability_drop": p1["probability_drop_macro"] <= 0.005,
            "class_nonworse": class_nonworse >= 0.80,
            "orthogonality": max(
                (float(item.get("max_projected_row_dot", 0.0)) for item in p1_steps),
                default=0.0,
            ) <= 1.0e-5,
            "null_dimension": min(
                (int(item.get("null_dimension", 1)) for item in p1_steps), default=1
            ) > 0,
            "attack_retention": _median(
                item.get("attack_retention", float("nan")) for item in p1_steps
            ) >= 0.20,
            "cicr_preserved": p1["cicr_cosine_median"] >= p0["cicr_cosine_median"] - 0.02,
            "pattern_preserved": pattern_p1_initial >= 0.80 * pattern_p0_initial,
            "backtrack_skip": p1["backtrack_skip_ratio"] < 0.50,
        }
        decision = {
            "target_checks": target_checks,
            "protection_checks": protection_checks,
            "target_pass": all(target_checks.values()),
            "protection_pass": all(protection_checks.values()),
        }
        decision["pass"] = bool(decision["target_pass"] and decision["protection_pass"])

        p1_state_path = artifact_root / "p1_state.pt"
        feasibility_mode = str(config["spec"].get("spec_id", "")) == E2E_V0_SPEC_ID
        p1_delta_linf = float(arm_deltas["P1"].abs().max())
        p1_operational_checks = {
            "delta_finite": bool(torch.isfinite(arm_deltas["P1"]).all()),
            "delta_linf": p1_delta_linf <= 16.0 / 255.0 + 1.0e-6,
            "p1_state_present": p1_state_path.is_file(),
        }
        if not all(p1_operational_checks.values()):
            raise ValueError(
                "P1 operational invariants failed: %s" % p1_operational_checks
            )
        result = {
            "schema": "tausb.sdh-mechanism-pilot.v1",
            "spec_id": config["spec"]["spec_id"],
            "split_hash": split_hash,
            "elapsed_seconds": time.monotonic() - start,
            "target_weight_calibration": target_calibration.state_dict(),
            "nla_calibration": nla_calibration.state_dict(),
            "arms": arms,
            "initial": initial_summary,
            "pattern_t1_t0": pattern_t1_t0,
            "pattern_p1_p0": pattern_p1_p0,
            "pattern_p0_initial": pattern_p0_initial,
            "pattern_p1_initial": pattern_p1_initial,
            "decision": decision,
            "p1_operational_checks": p1_operational_checks,
            "p1_delta_linf": p1_delta_linf,
        }
        if runtime_input_hashes is not None:
            result["runtime_input_hashes"] = runtime_input_hashes
        if feasibility_mode:
            result["feasibility_state"] = {
                "path": str(artifact_root / "p1_feasibility_sdh_state.pt"),
                "evidence_scope": "end_to_end_feasibility_not_formal_method",
                "hiding_gate_passed": False,
                "mechanism_gate_passed": bool(decision["pass"]),
            }
        metrics_path = artifact_root / "mechanism_metrics.json"
        _write_json(metrics_path, result)
        frozen_carrier = _load_saved_p1_carrier(
            p1_state_path,
            base_carrier=base_carrier,
            device=torch.device("cpu"),
        )
        if decision["pass"] and not feasibility_mode:
            payload = build_frozen_sdh_state_payload(
                carrier=frozen_carrier,
                secret=primary_secret.detach().cpu(),
                target_class_id=14,
                secret_source_sha256=config["secrets"]["primary_source_sha256"],
                secret_tensor_sha256=config["secrets"]["primary_tensor_sha256"],
                source_manifest_sha256=config["secrets"]["manifest_sha256"],
                train_split_sha256=config["dataset"]["train_label_manifest_sha256"],
                hiding_gate_passed=True,
                mechanism_gate_passed=True,
            )
            torch.save(payload, artifact_root / "p1_frozen_sdh_state.pt")
        if feasibility_mode:
            hiding_provenance = hiding_state.get("e2e_v0_hiding_provenance")
            if not isinstance(hiding_provenance, Mapping):
                raise ValueError("E2E V0 hiding provenance is missing.")
            payload = build_feasibility_sdh_state_payload(
                carrier=frozen_carrier,
                secret=primary_secret.detach().cpu(),
                target_class_id=14,
                secret_source_sha256=config["secrets"]["primary_source_sha256"],
                secret_tensor_sha256=config["secrets"]["primary_tensor_sha256"],
                source_manifest_sha256=config["secrets"]["manifest_sha256"],
                train_split_sha256=config["dataset"]["train_label_manifest_sha256"],
                mechanism_gate_passed=bool(decision["pass"]),
                hiding_metrics_sha256=hiding_provenance["hiding_metrics_sha256"],
                hiding_checkpoint_sha256=hiding_provenance[
                    "hiding_checkpoint_sha256"
                ],
                hiding_split_sha256=hiding_state["split_hash"],
                mechanism_metrics_sha256=_file_sha256(metrics_path),
                mechanism_decision_sha256=_canonical_json_sha256(decision),
                mechanism_config_sha256=_canonical_json_sha256(config),
                p1_state_sha256=_file_sha256(p1_state_path),
            )
            torch.save(payload, artifact_root / "p1_feasibility_sdh_state.pt")
        return result
    finally:
        engine.close()
