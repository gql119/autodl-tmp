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
from .sdh_materializer import build_frozen_sdh_state_payload
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
    x1, y1, x2, y2 = _bbox_to_pixels(targets[0]["bbox"], image.shape[0], image.shape[1])
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
    if config["spec"].get("spec_id") != "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3":
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
    if int(config["mechanism"].get("optimization_steps", -1)) != 8:
        raise ValueError("SDH matched microtrajectory must remain 8 steps.")
    if int(config["mechanism"].get("max_backtracks", -1)) != 5:
        raise ValueError("SDH CGR max_backtracks must remain 5.")
    if float(config["mechanism"].get("probability_drop_tolerance", -1)) != 0.005:
        raise ValueError("SDH probability-drop tolerance must remain 0.005.")
    if config["mechanism"].get("eot_enabled") is not False:
        raise ValueError("SDH first round forbids EOT.")
    if config["mechanism"].get("jnd_enabled") is not False:
        raise ValueError("SDH first round forbids JND.")
    for section, keys in {
        "dataset": ("root", "train_images", "train_labels"),
        "model": ("surrogate_checkpoint",),
        "secrets": ("manifest", "primary_id", "primary_source_sha256"),
        "runtime": ("artifact_root", "device"),
    }.items():
        for key in keys:
            if not str(config[section].get(key, "")).strip():
                raise ValueError("SDH config requires %s.%s." % (section, key))


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
    gate = evaluate_hiding_gate(hiding_metrics)
    gate["checks"]["dlfc_leakage_probe"] = bool(leakage["pass"])
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
    root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    metrics_path = root / "hiding" / "hiding_metrics.json"
    checkpoint_path = root / "hiding" / "hiding_checkpoint.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Passing hiding artifacts are required before mechanism.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("gate", {}).get("pass") is not True:
        raise ValueError("Hiding gate did not pass; mechanism is forbidden.")
    if _file_sha256(checkpoint_path) != metrics.get("checkpoint_sha256"):
        raise ValueError("Hiding checkpoint hash mismatch.")
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
    )
    carrier.load_state_dict(state["carrier_state"], strict=True)
    if carrier.architecture_sha256() != state["architecture_sha256"]:
        raise ValueError("Hiding checkpoint architecture hash mismatch.")
    carrier.to(device)
    carrier.freeze_for_detector_optimization()
    primary = torch.as_tensor(state["primary_secret"], dtype=torch.float32, device=device)
    return carrier, primary, state


def _clone_detector_carrier(
    source: SemanticHidingCarrier, device: torch.device
) -> SemanticHidingCarrier:
    clone = SemanticHidingCarrier(
        input_size=source.input_size,
        width=source.width,
        coupling_blocks=source.coupling_blocks,
        epsilon=source.epsilon,
    ).to(device)
    clone.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    clone.freeze_for_detector_optimization()
    return clone


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
    start = time.monotonic()
    max_seconds = float(config["mechanism"]["max_seconds"])
    device = torch.device(str(config["runtime"]["device"]))
    base_carrier, primary_secret, hiding_state = _load_hiding_checkpoint(
        config, config_base=config_base, device=device
    )
    dataset_root = _resolve(config_base, str(config["dataset"]["root"]))
    image_dir = dataset_root / str(config["dataset"]["train_images"])
    label_dir = dataset_root / str(config["dataset"]["train_labels"])
    batch_size = int(config["mechanism"]["batch_size"])
    paths = _person_paths(image_dir, label_dir, 14)
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
        if decision["pass"]:
            try:
                p1_state = torch.load(p1_state_path, map_location="cpu", weights_only=False)
            except TypeError:
                p1_state = torch.load(p1_state_path, map_location="cpu")
            frozen_carrier = _clone_detector_carrier(base_carrier, torch.device("cpu"))
            frozen_carrier.load_state_dict(p1_state["carrier_state"], strict=True)
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
        }
        _write_json(artifact_root / "mechanism_metrics.json", result)
        return result
    finally:
        engine.close()
