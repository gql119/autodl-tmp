from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..data_utils import label_path_for_image, read_yolo_annotations
from .constraint_gradient_router import (
    backtrack_multi_parameter_constraints,
    backtrack_multi_parameter_update,
    flatten_loss_gradient,
    route_budgeted_protection_gradients,
    route_multi_parameter_gradients,
)
from .detector_lfc import DetectorLFCPrototypeBank
from .dgcaip import DGCAIPResult, FrozenDGCAIPGradientCalibration
from .dgcaip_dataset_risk import load_risk_bank
from .dgcaip_diagnostics import build_dgcaip_locator_report
from .dgcaip_r3_diagnostics import (
    build_rejection_attribution,
    build_same_process_replay,
)
from .dgcaip_strict_step import (
    partition_nonlinear_constraints,
    run_strict_dgcaip_step,
    strict_component_constraint_losses,
    strict_constraint_losses,
)
from .instance_cicr import FrozenInstanceCICRBank
from .non_target_logit_alignment import FrozenNLAGradientCalibration
from .p1_determinism_audit import backend_manifest, payload_sha256
from .sdh_experiment import (
    _batches,
    _clone_detector_carrier,
    _component_losses,
    _copy_parameters_,
    _file_sha256,
    _flatten_autograd_norm,
    _load_hiding_checkpoint,
    _load_saved_p1_carrier,
    _median,
    _person_paths,
    _resolve,
    _split_hash,
    _summarize_arm,
    _time_guard,
    _validate_e2e_v0_runtime_inputs,
    _write_json,
    _canonical_json_sha256,
    deterministic_person_split,
)
from .sdh_materializer import build_dgcaip_p4_candidate_state_payload
from .sdh_mechanism import (
    FrozenTargetGradientCalibration,
    SDHObservation,
    SDHObservationEngine,
    adapter_parameters,
    compose_sdh_target_objective,
    load_sdh_batch,
)


DGCAIP_ARMS = {
    "P1-R": "off",
    "P2-CAIP": "caip",
    "P3-DIST": "dist",
    "P4-DGCAIP": "dgcaip",
}
R3_DIAGNOSTIC_ARMS = {
    "P1-A": "off",
    "P1-B": "off",
    "P2-CAIP": "caip",
    "P4-DGCAIP": "dgcaip",
}

P1_REPLAY_SCALARS = (
    "nla_macro_loss",
    "probability_drop_macro",
    "route_loss",
    "valid_instance_coverage",
    "cicr_cosine_median",
    "dlfc_cosine_median",
    "backtrack_skip_ratio",
)


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _accepted_target_progress_pass(
    accepted_steps: Sequence[Mapping[str, Any]],
    *,
    minimum: float,
    tolerance: float,
) -> bool:
    return bool(
        accepted_steps
        and _median(float(item["target_progress"]) for item in accepted_steps)
        + float(tolerance)
        >= float(minimum)
    )


def _v4_layered_gate_decision(
    integrity_checks: Mapping[str, bool],
    mechanism_checks: Mapping[str, bool],
    *,
    accepted_update_ratio: float,
    minimum_accepted_update_ratio: float,
    target_progress_pass: bool,
) -> Dict[str, Any]:
    if set(integrity_checks).intersection(mechanism_checks):
        raise ValueError("v4 gate layers must use distinct check names.")
    if (
        not math.isfinite(float(accepted_update_ratio))
        or not math.isfinite(float(minimum_accepted_update_ratio))
        or not 0.0 <= float(accepted_update_ratio) <= 1.0
        or not 0.0 <= float(minimum_accepted_update_ratio) <= 1.0
    ):
        raise ValueError("v4 accepted-update ratios must be finite in [0, 1].")
    promotion_checks = {
        "accepted_update_ratio": float(accepted_update_ratio)
        >= float(minimum_accepted_update_ratio),
        "final_target_progress": bool(target_progress_pass),
    }
    checks = {
        **integrity_checks,
        **mechanism_checks,
        **promotion_checks,
    }
    return {
        "checks": checks,
        "runtime_pass": all(integrity_checks.values()),
        "mechanism_valid": all(mechanism_checks.values()),
        "promotion_pass": all(promotion_checks.values()),
        "pass": all(checks.values()),
    }


def _support_outside_linf(observations: Sequence[SDHObservation]) -> float:
    if not observations:
        raise ValueError("Support audit requires at least one observation.")
    values = []
    for item in observations:
        perturbation = item.rendered.perturbation.detach()
        outside = torch.logical_not(
            item.rendered.union_support.detach()
        ).to(dtype=perturbation.dtype)
        values.append(float((perturbation * outside).abs().max()))
    return max(values)


def _frozen_snapshot(
    *,
    base_carrier: torch.nn.Module,
    engine: SDHObservationEngine,
    dlfc_bank: DetectorLFCPrototypeBank,
    cicr_bank: FrozenInstanceCICRBank,
    target_calibration: FrozenTargetGradientCalibration,
    nla_calibration: FrozenNLAGradientCalibration,
    dg_calibration: FrozenDGCAIPGradientCalibration,
) -> Dict[str, str]:
    return {
        "base_carrier": payload_sha256(base_carrier.state_dict()),
        "surrogate": payload_sha256(engine.model.state_dict()),
        "dlfc_bank": payload_sha256(dlfc_bank.state_dict()),
        "cicr_bank": payload_sha256(cicr_bank.state_dict()),
        "target_calibration": payload_sha256(target_calibration.state_dict()),
        "nla_calibration": payload_sha256(nla_calibration.state_dict()),
        "dgcaip_calibration": payload_sha256(dg_calibration.state_dict()),
    }


def _validate_d0_report_binding(
    d0_report: Mapping[str, Any],
    config: Mapping[str, Any],
    dg_config: Mapping[str, Any],
) -> None:
    if not bool(d0_report.get("decision", {}).get("pass")):
        raise ValueError("DG-CAIP mechanism is blocked by the D0 gate.")
    expected_spec_id = str(
        dg_config.get("expected_d0_spec_id", config["spec"]["spec_id"])
    )
    if d0_report.get("spec_id") != expected_spec_id:
        raise ValueError("DG-CAIP D0 report SpecID mismatch.")
    if d0_report.get("split_hash") != dg_config["expected_split_sha256"]:
        raise ValueError("DG-CAIP D0 report split hash mismatch.")
    if d0_report.get("source_p1_state_sha256") != str(
        dg_config["source_p1_state_sha256"]
    ).lower():
        raise ValueError("DG-CAIP D0 report source P1 hash mismatch.")


def _p1_replay_report(
    observed: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Dict[str, Any]:
    reference_arms = reference_metrics.get("arms", {})
    if not isinstance(reference_arms, Mapping) or not isinstance(
        reference_arms.get("P1"), Mapping
    ):
        raise ValueError("Historical P1 metrics do not contain arms.P1.")
    reference = reference_arms["P1"]
    comparisons: Dict[str, Dict[str, Any]] = {}

    def compare(name: str, actual: Any, expected: Any) -> None:
        actual_value = float(actual)
        expected_value = float(expected)
        limit = float(absolute_tolerance) + float(relative_tolerance) * abs(
            expected_value
        )
        error = abs(actual_value - expected_value)
        comparisons[name] = {
            "observed": actual_value,
            "reference": expected_value,
            "absolute_error": error,
            "limit": limit,
            "pass": math.isfinite(actual_value)
            and math.isfinite(expected_value)
            and error <= limit,
        }

    for key in P1_REPLAY_SCALARS:
        if key not in observed or key not in reference:
            raise ValueError("P1 replay metric is missing: %s" % key)
        compare(key, observed[key], reference[key])

    observed_by_class = {
        str(key): value
        for key, value in observed.get("probability_drop_by_class", {}).items()
    }
    reference_by_class = {
        str(key): value
        for key, value in reference.get("probability_drop_by_class", {}).items()
    }
    if set(observed_by_class) != set(reference_by_class):
        raise ValueError("P1 replay non-target class coverage changed.")
    for class_id in sorted(reference_by_class, key=int):
        compare(
            "probability_drop_by_class.%s" % class_id,
            observed_by_class[class_id],
            reference_by_class[class_id],
        )

    observed_steps = observed.get("steps", [])
    reference_steps = reference.get("steps", [])
    if len(observed_steps) != len(reference_steps):
        raise ValueError("P1 replay optimization-step count changed.")
    structural_checks: Dict[str, bool] = {}
    for index, (actual_step, expected_step) in enumerate(
        zip(observed_steps, reference_steps)
    ):
        for key in (
            "route_mode",
            "accepted",
            "backtrack_attempts",
            "constraint_rank",
            "null_dimension",
        ):
            check_name = "steps.%d.%s" % (index, key)
            structural_checks[check_name] = actual_step.get(key) == expected_step.get(key)
        for key in (
            "attack_retention",
            "max_projected_row_dot",
            "max_final_row_dot",
        ):
            compare(
                "steps.%d.%s" % (index, key),
                actual_step[key],
                expected_step[key],
            )

    return {
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "numeric_comparisons": comparisons,
        "structural_checks": structural_checks,
        "pass": all(item["pass"] for item in comparisons.values())
        and all(structural_checks.values()),
    }


def _parameter_sha256(parameters: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        array = (
            parameter.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _batch_sha256(paths: Sequence[Path], label_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        label_path = Path(label_path_for_image(str(path), str(label_dir)))
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(label_path.name.encode("utf-8"))
        digest.update(_file_sha256(label_path).encode("ascii"))
    return digest.hexdigest()


def _vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denominator = float(left.norm() * right.norm())
    if denominator <= 1.0e-12:
        return 1.0 if float(left.norm() + right.norm()) <= 1.0e-12 else 0.0
    return float(torch.dot(left, right) / denominator)


def _is_cooccurring(path: Path, label_dir: Path, target_class_id: int) -> bool:
    annotations = read_yolo_annotations(
        label_path_for_image(str(path), str(label_dir))
    )
    classes = {int(item["cls"]) for item in annotations}
    return int(target_class_id) in classes and any(
        class_id != int(target_class_id) for class_id in classes
    )


def _dgcaip_component_losses(result: DGCAIPResult) -> Dict[str, torch.Tensor]:
    attributes = {
        "classification": "classification_loss",
        "box": "box_loss",
        "alignment": "alignment_loss",
        "distribution": "distribution_loss",
    }
    output = {}
    for name, attribute in attributes.items():
        per_class = []
        for class_id in result.active_classes:
            values = [
                term.weight * getattr(term, attribute)
                for term in result.instances
                if term.class_id == class_id
            ]
            if values:
                per_class.append(torch.stack(values).mean())
        if per_class:
            output[name] = torch.stack(per_class).mean()
        else:
            output[name] = result.loss * 0.0
    return output


def _combined_protection_losses(
    observation: SDHObservation,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if observation.dgcaip is None:
        raise ValueError("Combined protection requires a DG-CAIP observation.")
    class_ids = sorted(
        set(observation.nla.per_class_loss).union(
            observation.dgcaip.per_class_loss
        )
    )
    zero = observation.nla.loss * 0.0 + observation.dgcaip.loss * 0.0
    per_class = {}
    for class_id in class_ids:
        per_class[str(class_id)] = (
            observation.nla.per_class_loss.get(class_id, zero)
            + observation.dgcaip.per_class_loss.get(class_id, zero)
        )
    loss = torch.stack(tuple(per_class.values())).mean() if per_class else zero
    return per_class, loss


def _dgcaip_class_metrics(result: DGCAIPResult) -> Dict[str, float]:
    metrics = {}
    for class_id in result.active_classes:
        terms = [term for term in result.instances if term.class_id == class_id]
        for name, attribute in {
            "probability": "classification_loss",
            "iou": "box_loss",
            "alignment": "alignment_loss",
            "js": "distribution_loss",
        }.items():
            metrics[f"{class_id}:{name}"] = float(
                torch.stack([getattr(term, attribute) for term in terms]).mean().detach()
            )
    return metrics


def _constraint_limits(
    result: DGCAIPResult,
    *,
    include_js: bool,
    js_epsilon: float,
) -> Dict[str, float]:
    values = _dgcaip_class_metrics(result)
    limits = {}
    for key, value in values.items():
        metric = key.rsplit(":", 1)[1]
        if metric == "js":
            if include_js:
                limits[key] = value + float(js_epsilon)
        else:
            limits[key] = 0.0
    return limits


def _filter_metrics(values: Mapping[str, float], limits: Mapping[str, float]) -> Dict[str, float]:
    if not set(limits).issubset(values):
        missing = sorted(set(limits).difference(values))
        raise ValueError("Candidate DG-CAIP metrics are missing: %s" % missing)
    return {name: float(values[name]) for name in limits}


def _strict_candidate_metrics(
    values: Mapping[str, float],
    baselines: Mapping[str, float],
) -> Dict[str, float]:
    """Treat a lost clean-TAL constraint as a rejected candidate, not a crash."""

    return {
        name: float(values.get(name, float(baseline) + 1.0))
        for name, baseline in baselines.items()
    }


def _strict_component_candidate_metrics(
    values: Mapping[str, float],
    baselines: Mapping[str, float],
) -> Dict[str, float]:
    """Require the v3 candidate registry to match the routed registry exactly."""

    if set(values) != set(baselines):
        missing = sorted(set(baselines).difference(values))
        unexpected = sorted(set(values).difference(baselines))
        raise ValueError(
            "Component-aligned candidate constraint keys differ: "
            "missing=%s unexpected=%s" % (missing, unexpected)
        )
    return {name: float(values[name]) for name in sorted(baselines)}


def _instance_metric_map(
    observations: Sequence[SDHObservation],
) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    output = {}
    for observation in observations:
        if observation.dgcaip is None:
            raise ValueError("DG-CAIP held-out summary requires instance metrics.")
        for term in observation.dgcaip.instances:
            key = (
                observation.image_ids[term.batch_index],
                term.gt_index,
                term.class_id,
            )
            if key in output:
                raise ValueError("Duplicate DG-CAIP held-out instance key.")
            output[key] = {
                "probability": float(term.classification_damage.detach()),
                "iou": float(term.box_damage.detach()),
                "alignment": float(term.alignment_damage.detach()),
                "js": float(term.distribution_loss.detach()),
                "geometry_risk": term.geometry_risk,
            }
    return output


def _cohort_keys(
    metrics: Mapping[Tuple[str, int, int], Mapping[str, float]],
    *,
    highest: bool,
) -> Tuple[Tuple[str, int, int], ...]:
    ordered = sorted(metrics, key=lambda key: metrics[key]["js"])
    count = max(1, math.ceil(len(ordered) / 4.0))
    selected = ordered[-count:] if highest else ordered[:count]
    return tuple(selected)


def _cohort_summary(
    metrics: Mapping[Tuple[str, int, int], Mapping[str, float]],
    keys: Sequence[Tuple[str, int, int]],
) -> Dict[str, float]:
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise ValueError("Held-out DG-CAIP cohort changed across arms.")
    return {
        name: float(np.mean([metrics[key][name] for key in keys]))
        for name in ("probability", "iou", "alignment", "js")
    }


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 1.0e-12:
        return 0.0 if candidate <= 1.0e-12 else -1.0
    return (baseline - candidate) / baseline


def _mean_damage(summary: Mapping[str, float]) -> float:
    return float(np.mean([summary[name] for name in ("probability", "iou", "alignment")]))


def _gradient_cosine(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> float:
    first = torch.autograd.grad(
        first_loss, parameters, retain_graph=True, allow_unused=True
    )
    second = torch.autograd.grad(
        second_loss, parameters, retain_graph=True, allow_unused=True
    )
    first_flat = torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, first)
        ]
    )
    second_flat = torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, second)
        ]
    )
    if float(first_flat.norm()) == 0.0 or float(second_flat.norm()) == 0.0:
        return 0.0
    return float(
        torch.nn.functional.cosine_similarity(
            first_flat.reshape(1, -1), second_flat.reshape(1, -1), dim=1
        ).detach()
    )


def _prepare_experiment(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Tuple[Any, torch.Tensor, Mapping[str, Any], Path, Path, Mapping[str, List[Path]]]:
    device = torch.device(str(config["runtime"]["device"]))
    carrier, secret, hiding_state = _load_hiding_checkpoint(
        config, config_base=config_base, device=device
    )
    dataset_root = _resolve(config_base, str(config["dataset"]["root"]))
    image_dir = dataset_root / str(config["dataset"]["train_images"])
    label_dir = dataset_root / str(config["dataset"]["train_labels"])
    _validate_e2e_v0_runtime_inputs(
        config,
        config_base=config_base,
        image_dir=image_dir,
        label_dir=label_dir,
        hiding_state=hiding_state,
        primary_secret=secret,
    )
    paths = _person_paths(image_dir, label_dir, 14)
    if len(paths) != int(config["dataset"]["expected_person_images"]):
        raise ValueError("DG-CAIP person-image count does not match the frozen input.")
    batch_size = int(config["mechanism"]["batch_size"])
    split = deterministic_person_split(
        paths,
        label_dir=label_dir,
        target_class_id=14,
        calibration_count=int(config["mechanism"]["calibration_batches"]) * batch_size,
        heldout_count=int(config["mechanism"]["heldout_batches"]) * batch_size,
        seed=0,
    )
    if _split_hash(split) != hiding_state["split_hash"]:
        raise ValueError("DG-CAIP split does not match the frozen hiding checkpoint.")
    return carrier, secret, hiding_state, image_dir, label_dir, split


def _load_engine(
    config: Mapping[str, Any],
    *,
    config_base: Path,
    checkpoint_path: Path | None = None,
) -> SDHObservationEngine:
    from ultralytics import YOLO

    device = torch.device(str(config["runtime"]["device"]))
    resolved_checkpoint = checkpoint_path or _resolve(
        config_base, str(config["model"]["surrogate_checkpoint"])
    )
    wrapper = YOLO(str(resolved_checkpoint))
    model = wrapper.model.to(device).eval()
    dgcaip = config["dgcaip"]
    return SDHObservationEngine(
        model,
        target_class_id=14,
        num_classes=20,
        epsilon=16.0 / 255.0,
        assignment_topk=int(config["mechanism"]["assignment_topk"]),
        pag_layer_ratios=config["mechanism"]["pag_layer_ratios"],
        pag_min_pos=config["mechanism"]["pag_min_pos"],
        box_teacher_weight=float(config["mechanism"]["box_teacher_weight"]),
        dgcaip_temperature=float(dgcaip["temperature"]),
        dgcaip_classification_tolerance=float(
            dgcaip["classification_tolerance"]
        ),
        dgcaip_box_tolerance=float(dgcaip["box_tolerance"]),
        dgcaip_alignment_tolerance=float(dgcaip["alignment_tolerance"]),
        dgcaip_minimum_rank_instances=int(dgcaip["minimum_rank_instances"]),
    )


def run_dgcaip_pilot(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Dict[str, Any]:
    start = time.monotonic()
    max_seconds = float(config["mechanism"]["max_seconds"])
    device = torch.device(str(config["runtime"]["device"]))
    base_carrier, primary_secret, hiding_state, image_dir, label_dir, split = _prepare_experiment(
        config, config_base=config_base
    )
    dg_config = config["dgcaip"]
    production_e20 = str(dg_config.get("run_mode", "")) == "production_e20"
    spec_id = str(config["spec"].get("spec_id", ""))
    relaxed_gate_v4 = spec_id == "TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4"
    strict_dataset = (
        spec_id
        in {
            "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1",
            "TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2",
            "TAUSB-SDH-DGCAIP-COMPONENT-ALIGNED-ROUTE-v3",
            "TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4",
        }
        and str(dg_config.get("run_mode", "")) == "strict_mechanism"
    )
    strict_route_mode = str(
        config.get("strict_route", {}).get("mode", "repair_budget_v1")
    )
    r3_diagnostics = dg_config.get("r3_diagnostics", {})
    r3_enabled = bool(r3_diagnostics.get("enabled", False))
    arm_modes = (
        {"P5-DATASET-STRICT": "dgcaip"}
        if strict_dataset
        else (R3_DIAGNOSTIC_ARMS if r3_enabled else DGCAIP_ARMS)
    )
    source_p1_path = _resolve(config_base, str(dg_config["source_p1_state"]))
    if _file_sha256(source_p1_path) != str(dg_config["source_p1_state_sha256"]).lower():
        raise ValueError("DG-CAIP source P1 state hash mismatch.")
    dataset_ranks = None
    strict_replay_ids: Tuple[str, ...] = ()
    risk_bank = None
    if strict_dataset:
        ranking_config = config["dataset_ranking"]
        risk_bank_path = _resolve(
            config_base, str(ranking_config["risk_bank"])
        )
        if _file_sha256(risk_bank_path) != str(
            ranking_config["risk_bank_file_sha256"]
        ).lower():
            raise ValueError("Strict mechanism risk-bank file hash mismatch.")
        risk_bank = load_risk_bank(
            risk_bank_path,
            expected_spec_id="TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1",
            expected_sha256=str(ranking_config["risk_bank_canonical_sha256"]),
        )
        dataset_ranks = risk_bank.rank_mapping()
        replay_path = _resolve(
            config_base, str(ranking_config["replay_manifest"])
        )
        if _file_sha256(replay_path) != str(
            ranking_config["replay_manifest_file_sha256"]
        ).lower():
            raise ValueError("Strict mechanism replay-manifest hash mismatch.")
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
        if (
            replay_payload.get("schema") != "tausb.dgcaip-dataset-replay.v1"
            or replay_payload.get("spec_id")
            != "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"
            or replay_payload.get("risk_bank_canonical_sha256")
            != risk_bank.canonical_sha256
        ):
            raise ValueError("Strict mechanism replay manifest is not bound to the bank.")
        strict_replay_ids = tuple(str(item) for item in replay_payload["image_ids"])
        expected_slots = int(config["mechanism"]["optimization_steps"]) * int(
            config["mechanism"]["batch_size"]
        )
        if len(strict_replay_ids) != expected_slots:
            raise ValueError("Strict mechanism replay slot count mismatch.")
        base_carrier = _load_saved_p1_carrier(
            source_p1_path,
            base_carrier=base_carrier,
            device=device,
        )
    repair_report_path = None
    backend = None
    if production_e20:
        repair_report_path = _resolve(
            config_base, str(dg_config["repair_report"])
        )
        if _file_sha256(repair_report_path) != str(
            dg_config["repair_report_sha256"]
        ).lower():
            raise ValueError("DG-CAIP repair report hash mismatch.")
        backend = backend_manifest()
        if not (
            backend["cublas_workspace_config"] == ":4096:8"
            and backend["deterministic_algorithms"]
            and not backend["deterministic_warn_only"]
            and backend["cudnn_deterministic"]
            and not backend["cudnn_benchmark"]
            and backend["cuda_matmul_allow_tf32"] is False
            and backend["cudnn_allow_tf32"] is False
        ):
            raise RuntimeError("DG-CAIP production strict backend is not active.")
    engine = _load_engine(config, config_base=config_base)
    protection_engines: Dict[str, SDHObservationEngine] = {}
    protection_snapshot_hashes: Dict[str, str] = {}
    if strict_dataset:
        for snapshot in config["model"]["protection_surrogate_snapshots"]:
            snapshot_id = str(snapshot["id"])
            checkpoint = _resolve(config_base, str(snapshot["checkpoint"]))
            actual_hash = _file_sha256(checkpoint)
            if actual_hash != str(snapshot["sha256"]).lower():
                raise ValueError(
                    "Strict protection snapshot hash mismatch: %s" % snapshot_id
                )
            protection_snapshot_hashes[snapshot_id] = actual_hash
            protection_engines[snapshot_id] = _load_engine(
                config,
                config_base=config_base,
                checkpoint_path=checkpoint,
            )
    batch_size = int(config["mechanism"]["batch_size"])
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    run_mode = str(dg_config["run_mode"])
    output_root = artifact_root / run_mode
    output_root.mkdir(parents=True, exist_ok=False)

    def load(paths_batch: Sequence[Path]):
        return load_sdh_batch(
            paths_batch,
            label_dir=label_dir,
            image_size=640,
            target_class_id=14,
            device=device,
        )

    try:
        if run_mode == "d0":
            p1_carrier = _load_saved_p1_carrier(
                source_p1_path,
                base_carrier=base_carrier,
                device=device,
            )
            cooccurring = [
                path
                for path in split["heldout"]
                if _is_cooccurring(path, label_dir, 14)
            ]
            observations = []
            with torch.no_grad():
                for paths_batch in _batches(cooccurring, batch_size):
                    _time_guard(start, max_seconds, "DG-CAIP D0")
                    observations.append(
                        engine.observe(
                            load(paths_batch),
                            p1_carrier,
                            primary_secret,
                            dgcaip_mode="dist",
                        )
                    )
            locator = build_dgcaip_locator_report(
                [
                    observation.dgcaip
                    for observation in observations
                    if observation.dgcaip is not None
                ]
            )
            result = {
                "schema": "tausb.dgcaip-d0-run.v1",
                "spec_id": config["spec"]["spec_id"],
                "split_hash": _split_hash(split),
                "source_p1_state_sha256": _file_sha256(source_p1_path),
                "elapsed_seconds": time.monotonic() - start,
                "locator": locator,
                "decision": {"pass": locator["decision"] == "pass"},
            }
            _write_json(output_root / "d0_locator.json", result)
            return result

        if strict_dataset:
            d0_path = None
            source_p1_metrics_path = None
            source_p1_metrics = {}
        else:
            d0_path = _resolve(config_base, str(dg_config["d0_report"]))
            if _file_sha256(d0_path) != str(dg_config["d0_report_sha256"]).lower():
                raise ValueError("DG-CAIP D0 report hash mismatch.")
            d0_report = json.loads(d0_path.read_text(encoding="utf-8"))
            _validate_d0_report_binding(d0_report, config, dg_config)
            source_p1_metrics_path = _resolve(
                config_base, str(dg_config["source_p1_metrics"])
            )
            if _file_sha256(source_p1_metrics_path) != str(
                dg_config["source_p1_metrics_sha256"]
            ).lower():
                raise ValueError("DG-CAIP source P1 metrics hash mismatch.")
            source_p1_metrics = json.loads(
                source_p1_metrics_path.read_text(encoding="utf-8")
            )

        calibration_batches = _batches(split["calibration"], batch_size)
        heldout_batches = _batches(split["heldout"], batch_size)
        if strict_dataset:
            by_id = {
                path.stem: path for path in _person_paths(image_dir, label_dir, 14)
            }
            missing_replay = sorted(set(strict_replay_ids).difference(by_id))
            if missing_replay:
                raise ValueError("Strict replay references unknown training images.")
            optimization_batches = _batches(
                [by_id[image_id] for image_id in strict_replay_ids],
                batch_size,
            )
        else:
            optimization_batches = calibration_batches
        initial_observations = []
        with torch.no_grad():
            for paths_batch in calibration_batches:
                _time_guard(start, max_seconds, "DG-CAIP mechanism")
                initial_observations.append(
                    engine.observe(
                        load(paths_batch),
                        base_carrier,
                        primary_secret,
                        dgcaip_mode="dist",
                    )
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
        target_norms: Dict[str, List[float]] = {
            name: [] for name in ("easy", "reveal", "rms", "dlfc", "cicr", "floor")
        }
        dg_norms: Dict[str, List[float]] = {
            name: []
            for name in ("classification", "box", "alignment", "distribution")
        }
        warmup_observations = []
        warmup_count = int(config["mechanism"]["weight_calibration_batches"])
        for paths_batch in calibration_batches[:warmup_count]:
            observation = engine.observe(
                load(paths_batch),
                calibration_carrier,
                primary_secret,
                dgcaip_mode="dist",
            )
            warmup_observations.append(observation)
            components, _, _ = _component_losses(observation, dlfc_bank, cicr_bank)
            for name, loss in components.items():
                target_norms[name].append(
                    _flatten_autograd_norm(loss, omega, retain_graph=True)
                )
            if observation.dgcaip is None:
                raise RuntimeError("DG-CAIP warm-up observation is missing.")
            dg_components = _dgcaip_component_losses(observation.dgcaip)
            for name, loss in dg_components.items():
                dg_norms[name].append(
                    _flatten_autograd_norm(loss, omega, retain_graph=True)
                )
        target_calibration = FrozenTargetGradientCalibration()
        target_weights = target_calibration.calibrate(target_norms, split="warmup")
        dg_calibration = FrozenDGCAIPGradientCalibration()
        dg_weights = dg_calibration.calibrate(dg_norms, split="warmup")

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
                    str(key): value
                    for key, value in observation.nla.per_class_loss.items()
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
                initial_heldout.append(
                    engine.observe(
                        load(paths_batch),
                        base_carrier,
                        primary_secret,
                        dgcaip_mode="dist",
                        dgcaip_component_weights=dg_weights,
                    )
                )
        initial_summary, initial_deltas = _summarize_arm(
            initial_heldout, dlfc_bank, cicr_bank
        )
        initial_metric_map = _instance_metric_map(initial_heldout)
        q4_keys = _cohort_keys(initial_metric_map, highest=True)
        q1_keys = _cohort_keys(initial_metric_map, highest=False)
        frozen_before = _frozen_snapshot(
            base_carrier=base_carrier,
            engine=engine,
            dlfc_bank=dlfc_bank,
            cicr_bank=cicr_bank,
            target_calibration=target_calibration,
            nla_calibration=nla_calibration,
            dg_calibration=dg_calibration,
        )
        if strict_dataset:
            frozen_before.update(
                {
                    "protection_snapshot_%s" % snapshot_id: payload_sha256(
                        snapshot_engine.model.state_dict()
                    )
                    for snapshot_id, snapshot_engine in sorted(
                        protection_engines.items()
                    )
                }
            )

        arms = {}
        arm_deltas = {}
        arm_states = {}
        arm_initial_hashes = {}
        arm_final_hashes = {}
        arm_batch_hashes = {}
        arm_route_gradients = {}
        for arm_id, dg_mode in arm_modes.items():
            carrier = _clone_detector_carrier(base_carrier, device)
            parameters = adapter_parameters(carrier)
            arm_initial_hashes[arm_id] = _parameter_sha256(parameters)
            arm_batch_hashes[arm_id] = []
            arm_route_gradients[arm_id] = []
            step_rows = []
            gradient_cosines = []
            backtrack_or_skip = 0
            backtracked_steps = 0
            skipped_updates = 0
            for step in range(int(config["mechanism"]["optimization_steps"])):
                _time_guard(start, max_seconds, "DG-CAIP mechanism")
                paths_batch = optimization_batches[step % len(optimization_batches)]
                batch = load(paths_batch)
                batch_sha256 = (
                    _batch_sha256(paths_batch, label_dir)
                    if r3_enabled or production_e20 or strict_dataset
                    else None
                )
                if r3_enabled or production_e20 or strict_dataset:
                    arm_batch_hashes[arm_id].append(str(batch_sha256))
                observation = engine.observe(
                    batch,
                    carrier,
                    primary_secret,
                    dgcaip_mode=dg_mode,
                    dgcaip_component_weights=dg_weights,
                    dgcaip_dataset_percentile_ranks=(
                        dataset_ranks if strict_dataset else None
                    ),
                )
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
                gradient_cosines.append(
                    _gradient_cosine(components["dlfc"], components["cicr"], parameters)
                )
                originals = tuple(
                    parameter.detach().clone() for parameter in parameters
                )
                if dg_mode == "off":
                    routed = route_multi_parameter_gradients(
                        parameters=parameters,
                        target_loss=objective.loss,
                        per_class_nla_losses={
                            str(key): value
                            for key, value in observation.nla.per_class_loss.items()
                        },
                        nla_loss=observation.nla.loss,
                        nla_weight=lambda_nla,
                    )

                    def evaluate_p1(candidate):
                        _copy_parameters_(parameters, candidate)
                        try:
                            with torch.no_grad():
                                current = engine.observe(
                                    batch, carrier, primary_secret
                                )
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
                        evaluate_probability_drops=evaluate_p1,
                        tolerance=0.005,
                        max_backtracks=5,
                    )
                    protection_ratio = (
                        float(routed.nla_norm * lambda_nla / max(routed.projected_target_norm, 1.0e-12))
                    )
                elif strict_dataset:
                    if observation.dgcaip is None:
                        raise RuntimeError("Strict DG-CAIP observation is missing.")
                    target_gradient = flatten_loss_gradient(
                        objective.loss,
                        parameters,
                        retain_graph=False,
                    ).detach()
                    del observation, objective, components
                    safe_constraint_gradients = {}
                    violated_constraint_gradients = {}
                    current_metrics = {}
                    component_row_digest = ""
                    component_safe_row_count = 0
                    component_violated_row_count = 0
                    for snapshot_id, snapshot_engine in sorted(
                        protection_engines.items()
                    ):
                        protected = snapshot_engine.observe(
                            batch,
                            carrier,
                            primary_secret,
                            dgcaip_mode="dgcaip",
                            dgcaip_component_weights=dg_weights,
                            dgcaip_dataset_percentile_ranks=dataset_ranks,
                        )
                        if protected.dgcaip is None:
                            raise RuntimeError(
                                "Strict protection snapshot observation is missing."
                            )
                        if (
                            strict_route_mode
                            == "component_aligned_target_progress_v3"
                        ):
                            component_losses = {
                                "%s/%s" % (snapshot_id, name): loss
                                for name, loss in strict_component_constraint_losses(
                                    protected
                                ).items()
                            }
                            component_metrics = {
                                name: float(loss.detach())
                                for name, loss in component_losses.items()
                            }
                            snapshot_safe, snapshot_violated = (
                                partition_nonlinear_constraints(
                                    component_metrics,
                                    js_epsilon=float(
                                        dg_config["js_backtracking_epsilon"]
                                    ),
                                )
                            )
                            current_metrics.update(component_metrics)
                            safe_losses = {
                                name: component_losses[name]
                                for name in snapshot_safe
                            }
                            violated_losses = {
                                name: component_losses[name]
                                for name in snapshot_violated
                            }
                        else:
                            current_metrics.update(
                                {
                                    "%s/%s" % (snapshot_id, name): value
                                    for name, value in _dgcaip_class_metrics(
                                        protected.dgcaip
                                    ).items()
                                }
                            )
                            snapshot_safe, snapshot_violated = {}, {}
                            legacy_safe, legacy_violated = strict_constraint_losses(
                                protected
                            )
                            safe_losses = {
                                "%s/%s" % (snapshot_id, name): loss
                                for name, loss in legacy_safe.items()
                            }
                            violated_losses = {
                                "%s/%s" % (snapshot_id, name): loss
                                for name, loss in legacy_violated.items()
                            }
                        gradient_rows = [
                            (
                                safe_constraint_gradients,
                                name,
                                loss,
                            )
                            for name, loss in sorted(safe_losses.items())
                        ] + [
                            (
                                violated_constraint_gradients,
                                name,
                                loss,
                            )
                            for name, loss in sorted(violated_losses.items())
                        ]
                        requiring_grad = [
                            index
                            for index, (_, _, loss) in enumerate(gradient_rows)
                            if loss.requires_grad
                        ]
                        last_gradient = max(requiring_grad, default=-1)
                        for index, (destination, name, loss) in enumerate(
                            gradient_rows
                        ):
                            destination[name] = flatten_loss_gradient(
                                loss,
                                parameters,
                                retain_graph=index != last_gradient,
                            ).detach()
                        del protected, safe_losses, violated_losses, gradient_rows
                        if (
                            strict_route_mode
                            == "component_aligned_target_progress_v3"
                        ):
                            del component_losses, component_metrics
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    if strict_route_mode == "component_aligned_target_progress_v3":
                        component_safe_row_count = len(safe_constraint_gradients)
                        component_violated_row_count = len(
                            violated_constraint_gradients
                        )
                        component_row_digest = hashlib.sha256(
                            "\n".join(sorted(current_metrics)).encode("utf-8")
                        ).hexdigest()

                    def evaluate_strict(candidate):
                        _copy_parameters_(parameters, candidate)
                        try:
                            with torch.no_grad():
                                candidate_values = {}
                                for snapshot_id, snapshot_engine in sorted(
                                    protection_engines.items()
                                ):
                                    current = snapshot_engine.observe(
                                        batch,
                                        carrier,
                                        primary_secret,
                                        dgcaip_mode="dgcaip",
                                        dgcaip_component_weights=dg_weights,
                                        dgcaip_dataset_percentile_ranks=dataset_ranks,
                                    )
                                    if current.dgcaip is None:
                                        continue
                                    if (
                                        strict_route_mode
                                        == "component_aligned_target_progress_v3"
                                    ):
                                        candidate_values.update(
                                            {
                                                "%s/%s" % (snapshot_id, name): float(
                                                    loss.detach()
                                                )
                                                for name, loss in strict_component_constraint_losses(
                                                    current
                                                ).items()
                                            }
                                        )
                                    else:
                                        candidate_values.update(
                                            {
                                                "%s/%s" % (snapshot_id, name): value
                                                for name, value in _dgcaip_class_metrics(
                                                    current.dgcaip
                                                ).items()
                                            }
                                        )
                            if (
                                strict_route_mode
                                == "component_aligned_target_progress_v3"
                            ):
                                return _strict_component_candidate_metrics(
                                    candidate_values, current_metrics
                                )
                            return _strict_candidate_metrics(
                                candidate_values, current_metrics
                            )
                        finally:
                            _copy_parameters_(parameters, originals)

                    strict_step = run_strict_dgcaip_step(
                        parameters=parameters,
                        target_loss=None,
                        observation=None,
                        target_gradient=target_gradient,
                        safe_constraint_gradients=safe_constraint_gradients,
                        violated_constraint_gradients=violated_constraint_gradients,
                        current_metrics=current_metrics,
                        evaluate_constraints=evaluate_strict,
                        step_size=float(config["mechanism"]["learning_rate"]),
                        js_epsilon=float(dg_config["js_backtracking_epsilon"]),
                        repair_floor_fraction=float(
                            config["strict_route"]["repair_floor_fraction"]
                        ),
                        max_repair_norm_ratio=float(
                            config["strict_route"]["max_repair_norm_ratio"]
                        ),
                        max_projection_iterations=int(
                            config["strict_route"]["max_projection_iterations"]
                        ),
                        svd_relative_tolerance=float(
                            config["strict_route"].get(
                                "svd_relative_tolerance", 1.0e-4
                            )
                        ),
                        route_mode=strict_route_mode,
                        minimum_target_progress=float(
                            config["strict_route"].get(
                                "minimum_target_progress", 0.60
                            )
                        ),
                        max_backtracks=5,
                        nonlinear_comparison_tolerance=float(
                            config.get("strict_route", {}).get(
                                "nonlinear_comparison_tolerance", 1.0e-9
                            )
                        ),
                        record_trace=True,
                    )
                    routed = strict_step.route
                    backtracked = strict_step.backtracking
                    protection_ratio = routed.repair_norm / max(
                        routed.target_norm, 1.0e-12
                    )
                else:
                    per_class_protection, protection_loss = _combined_protection_losses(
                        observation
                    )
                    routed = route_budgeted_protection_gradients(
                        parameters=parameters,
                        target_loss=objective.loss,
                        per_class_protection_losses=per_class_protection,
                        protection_loss=protection_loss,
                        protection_ratio=float(dg_config["protection_ratio"]),
                    )
                    if observation.dgcaip is None:
                        raise RuntimeError("DG-CAIP arm observation is missing.")
                    include_js = dg_mode in {"dist", "dgcaip"}
                    limits = _constraint_limits(
                        observation.dgcaip,
                        include_js=include_js,
                        js_epsilon=float(dg_config["js_backtracking_epsilon"]),
                    )

                    def evaluate_dgcaip(candidate):
                        _copy_parameters_(parameters, candidate)
                        try:
                            with torch.no_grad():
                                current = engine.observe(
                                    batch,
                                    carrier,
                                    primary_secret,
                                    dgcaip_mode=dg_mode,
                                    dgcaip_component_weights=dg_weights,
                                )
                            if current.dgcaip is None:
                                raise RuntimeError("DG-CAIP candidate metrics are missing.")
                            return _filter_metrics(
                                _dgcaip_class_metrics(current.dgcaip), limits
                            )
                        finally:
                            _copy_parameters_(parameters, originals)

                    if limits:
                        backtracked = backtrack_multi_parameter_constraints(
                            parameters=parameters,
                            flattened_gradient=routed.gradient,
                            step_size=float(config["mechanism"]["learning_rate"]),
                            evaluate_constraints=evaluate_dgcaip,
                            limits=limits,
                            max_backtracks=5,
                            record_trace=r3_enabled,
                        )
                    else:
                        backtracked = backtrack_multi_parameter_update(
                            parameters=parameters,
                            flattened_gradient=routed.gradient,
                            step_size=float(config["mechanism"]["learning_rate"]),
                            evaluate_probability_drops=lambda _: {},
                            tolerance=0.005,
                            max_backtracks=5,
                        )
                    protection_ratio = routed.explicit_protection_norm_ratio
                if r3_enabled or strict_dataset:
                    arm_route_gradients[arm_id].append(
                        routed.gradient.detach().cpu().clone()
                    )
                if backtracked.attempts > 1 or not backtracked.accepted:
                    backtrack_or_skip += 1
                if backtracked.attempts > 1:
                    backtracked_steps += 1
                if not backtracked.accepted:
                    skipped_updates += 1
                if backtracked.accepted:
                    _copy_parameters_(parameters, backtracked.candidate)
                step_row = {
                    "step": step,
                    "route_mode": routed.mode,
                    "constraint_rank": routed.rank,
                    "null_dimension": routed.null_dimension,
                    "attack_retention": routed.attack_retention,
                    "max_projected_row_dot": (
                        routed.max_safe_final_row_dot
                        if strict_dataset
                        else routed.max_projected_row_dot
                    ),
                    "max_final_row_dot": (
                        routed.max_safe_final_row_dot
                        if strict_dataset
                        else routed.max_final_row_dot
                    ),
                    "explicit_protection_norm_ratio": protection_ratio,
                    "backtrack_attempts": backtracked.attempts,
                    "accepted": backtracked.accepted,
                }
                if strict_dataset:
                    step_row.update(
                        {
                            "min_violated_final_row_dot": (
                                routed.min_violated_final_row_dot
                            ),
                            "repair_floor": routed.repair_floor,
                            "repair_norm": routed.repair_norm,
                            "route_feasible": routed.feasible,
                            "target_gradient_norm": routed.target_norm,
                            "final_gradient_norm": routed.final_norm,
                            "batch_sha256": batch_sha256,
                            "routed_gradient_sha256": _parameter_sha256(
                                (routed.gradient,)
                            ),
                            "backtracking_trace": [
                                asdict(item) for item in backtracked.trace
                            ],
                        }
                    )
                    if strict_route_mode in {
                        "nonworsening_target_progress_v2",
                        "component_aligned_target_progress_v3",
                    }:
                        step_row.update(
                            {
                                "target_progress": routed.target_progress,
                                "target_cosine": routed.target_cosine,
                                "precast_max_safe_row_dot": (
                                    routed.precast_max_safe_row_dot
                                ),
                                "precast_min_violated_row_dot": (
                                    routed.precast_min_violated_row_dot
                                ),
                                "precast_target_progress": (
                                    routed.precast_target_progress
                                ),
                                "solver_dtype": routed.solver_dtype,
                            }
                        )
                    if strict_route_mode == "component_aligned_target_progress_v3":
                        step_row.update(
                            {
                                "constraint_row_schema": (
                                    "snapshot_class_family_v3"
                                ),
                                "safe_component_row_count": (
                                    component_safe_row_count
                                ),
                                "violated_component_row_count": (
                                    component_violated_row_count
                                ),
                                "constraint_row_name_sha256": component_row_digest,
                            }
                        )
                if r3_enabled:
                    serialized_trace = [
                        asdict(item) for item in backtracked.trace
                    ]
                    if dg_mode != "off" and not serialized_trace:
                        serialized_trace = [
                            {
                                "attempt": 0,
                                "step_size": float(backtracked.step_size),
                                "finite": True,
                                "constraints": [],
                                "group_max_margin": {
                                    "probability": None,
                                    "iou": None,
                                    "alignment": None,
                                    "js": None,
                                },
                                "group_violation_count": {
                                    "probability": 0,
                                    "iou": 0,
                                    "alignment": 0,
                                    "js": 0,
                                },
                                "accepted": bool(backtracked.accepted),
                                "reason": str(backtracked.status),
                            }
                        ]
                    step_row.update(
                        {
                            "batch_sha256": batch_sha256,
                            "routed_gradient_sha256": _parameter_sha256(
                                (routed.gradient,)
                            ),
                            "target_gradient_norm": routed.target_norm,
                            "final_gradient_norm": routed.combined_norm,
                            "backtracking_trace": serialized_trace,
                        }
                    )
                step_rows.append(step_row)

            heldout_observations = []
            with torch.no_grad():
                for paths_batch in heldout_batches:
                    heldout_observations.append(
                        engine.observe(
                            load(paths_batch),
                            carrier,
                            primary_secret,
                            dgcaip_mode="dist",
                            dgcaip_component_weights=dg_weights,
                        )
                    )
            summary, deltas = _summarize_arm(
                heldout_observations, dlfc_bank, cicr_bank
            )
            metric_map = _instance_metric_map(heldout_observations)
            summary.update(
                {
                    "fixed_q4": _cohort_summary(metric_map, q4_keys),
                    "fixed_q1": _cohort_summary(metric_map, q1_keys),
                    "gradient_dlfc_cicr_cosine_median": _median(gradient_cosines),
                    "backtrack_skip_ratio": backtrack_or_skip
                    / float(config["mechanism"]["optimization_steps"]),
                    "backtrack_rate": backtracked_steps
                    / float(config["mechanism"]["optimization_steps"]),
                    "actual_skip_ratio": skipped_updates
                    / float(config["mechanism"]["optimization_steps"]),
                    "accepted_update_ratio": 1.0
                    - skipped_updates
                    / float(config["mechanism"]["optimization_steps"]),
                    "steps": step_rows,
                    "full_perturbation_linf": max(
                        float(item.rendered.perturbation.detach().abs().max())
                        for item in heldout_observations
                    ),
                    "support_outside_linf": _support_outside_linf(
                        heldout_observations
                    ),
                }
            )
            arms[arm_id] = summary
            arm_deltas[arm_id] = deltas
            arm_states[arm_id] = {
                key: value.detach().cpu().clone()
                for key, value in carrier.state_dict().items()
            }
            if r3_enabled or production_e20 or strict_dataset:
                arm_final_hashes[arm_id] = _parameter_sha256(parameters)

        if len(set(arm_initial_hashes.values())) != 1:
            raise ValueError("DG-CAIP arms do not share the same adapter initialization.")

        frozen_after = _frozen_snapshot(
            base_carrier=base_carrier,
            engine=engine,
            dlfc_bank=dlfc_bank,
            cicr_bank=cicr_bank,
            target_calibration=target_calibration,
            nla_calibration=nla_calibration,
            dg_calibration=dg_calibration,
        )
        if strict_dataset:
            frozen_after.update(
                {
                    "protection_snapshot_%s" % snapshot_id: payload_sha256(
                        snapshot_engine.model.state_dict()
                    )
                    for snapshot_id, snapshot_engine in sorted(
                        protection_engines.items()
                    )
                }
            )

        if strict_dataset:
            arm_id = "P5-DATASET-STRICT"
            strict_arm = arms[arm_id]
            strict_steps = strict_arm["steps"]
            accepted_steps = [item for item in strict_steps if item["accepted"]]
            safe_dot_pass = all(
                float(item["max_final_row_dot"]) <= 1.0e-5
                for item in accepted_steps
            )
            if strict_route_mode in {
                "nonworsening_target_progress_v2",
                "component_aligned_target_progress_v3",
            }:
                repair_dot_pass = all(
                    float(item["min_violated_final_row_dot"]) >= -1.0e-6
                    for item in accepted_steps
                )
                progress_tolerance = (
                    1.0e-6
                    if strict_route_mode
                    == "component_aligned_target_progress_v3"
                    else 0.0
                )
                target_progress_pass = _accepted_target_progress_pass(
                    accepted_steps,
                    minimum=float(
                        config["strict_route"]["minimum_target_progress"]
                    ),
                    tolerance=progress_tolerance,
                )
            else:
                repair_dot_pass = all(
                    float(item["min_violated_final_row_dot"])
                    + 1.0e-6
                    >= float(item["repair_floor"])
                    for item in accepted_steps
                    if float(item["repair_floor"]) > 0.0
                )
                target_progress_pass = True
            integrity_checks = {
                "risk_bank_bound": bool(
                    risk_bank is not None
                    and dataset_ranks is not None
                    and len(dataset_ranks) == risk_bank.covered_instance_count
                ),
                "replay_slots_bound": len(strict_replay_ids)
                == int(config["mechanism"]["optimization_steps"]) * batch_size,
                "finite": _all_finite(strict_arm)
                and _all_finite(arm_states[arm_id]),
                "frozen_modules_unchanged": frozen_before == frozen_after,
            }
            mechanism_checks = {
                "at_least_one_update": bool(accepted_steps),
                "final_safe_orthogonality": bool(
                    accepted_steps and safe_dot_pass
                ),
                "violated_repair_direction": bool(
                    accepted_steps and repair_dot_pass
                ),
                "null_dimension": min(
                    (int(item["null_dimension"]) for item in accepted_steps),
                    default=0,
                )
                > 0,
                "adapter_changed": arm_final_hashes[arm_id]
                != arm_initial_hashes[arm_id],
            }
            if relaxed_gate_v4:
                decision = _v4_layered_gate_decision(
                    integrity_checks,
                    mechanism_checks,
                    accepted_update_ratio=float(
                        strict_arm["accepted_update_ratio"]
                    ),
                    minimum_accepted_update_ratio=float(
                        config["strict_route"]["minimum_accepted_update_ratio"]
                    ),
                    target_progress_pass=target_progress_pass,
                )
            else:
                checks = {
                    **integrity_checks,
                    **mechanism_checks,
                    "attack_retention": _median(
                        item["attack_retention"] for item in strict_steps
                    )
                    >= 0.60,
                    "backtrack_skip": strict_arm["backtrack_skip_ratio"] < 0.70,
                }
                if strict_route_mode in {
                    "nonworsening_target_progress_v2",
                    "component_aligned_target_progress_v3",
                }:
                    checks["final_target_progress"] = target_progress_pass
                decision = {"checks": checks, "pass": all(checks.values())}
            result = {
                "schema": (
                    "tausb.dgcaip-dataset-strict-mechanism.v4"
                    if relaxed_gate_v4
                    else (
                        "tausb.dgcaip-dataset-strict-mechanism.v3"
                        if strict_route_mode
                        == "component_aligned_target_progress_v3"
                        else (
                            "tausb.dgcaip-dataset-strict-mechanism.v2"
                            if strict_route_mode
                            == "nonworsening_target_progress_v2"
                            else "tausb.dgcaip-dataset-strict-mechanism.v1"
                        )
                    )
                ),
                "spec_id": config["spec"]["spec_id"],
                "split_hash": _split_hash(split),
                "source_p1_state_sha256": _file_sha256(source_p1_path),
                "risk_bank_canonical_sha256": risk_bank.canonical_sha256,
                "risk_bank_file_sha256": _file_sha256(risk_bank_path),
                "replay_manifest_file_sha256": _file_sha256(replay_path),
                "protection_snapshot_sha256": protection_snapshot_hashes,
                "target_weight_calibration": target_calibration.state_dict(),
                "nla_calibration": nla_calibration.state_dict(),
                "dgcaip_calibration": dg_calibration.state_dict(),
                "initial": initial_summary,
                "arms": arms,
                "decision": decision,
                "diagnostics": {
                    "attack_retention_median": _median(
                        item["attack_retention"] for item in strict_steps
                    ),
                    "backtrack_rate": strict_arm["backtrack_rate"],
                    "actual_skip_ratio": strict_arm["actual_skip_ratio"],
                    "accepted_update_ratio": strict_arm["accepted_update_ratio"],
                    "nonlinear_comparison_tolerance": float(
                        config.get("strict_route", {}).get(
                            "nonlinear_comparison_tolerance", 1.0e-9
                        )
                    ),
                },
                "elapsed_seconds": time.monotonic() - start,
            }
            if strict_route_mode in {
                "nonworsening_target_progress_v2",
                "component_aligned_target_progress_v3",
            }:
                result["strict_route_mode"] = strict_route_mode
            if strict_route_mode == "component_aligned_target_progress_v3":
                result["constraint_row_schema"] = "snapshot_class_family_v3"
            _write_json(
                output_root / "backtracking_trace.json",
                {
                    "schema": (
                        "tausb.dgcaip-dataset-strict-backtracking.v4"
                        if relaxed_gate_v4
                        else (
                            "tausb.dgcaip-dataset-strict-backtracking.v3"
                            if strict_route_mode
                            == "component_aligned_target_progress_v3"
                            else (
                                "tausb.dgcaip-dataset-strict-backtracking.v2"
                                if strict_route_mode
                                == "nonworsening_target_progress_v2"
                                else "tausb.dgcaip-dataset-strict-backtracking.v1"
                            )
                        )
                    ),
                    arm_id: strict_steps,
                },
            )
            torch.save(
                {
                    "schema": (
                        "tausb.dgcaip-dataset-strict-state.v4"
                        if relaxed_gate_v4
                        else (
                            "tausb.dgcaip-dataset-strict-state.v3"
                            if strict_route_mode
                            == "component_aligned_target_progress_v3"
                            else (
                                "tausb.dgcaip-dataset-strict-state.v2"
                                if strict_route_mode
                                == "nonworsening_target_progress_v2"
                                else "tausb.dgcaip-dataset-strict-state.v1"
                            )
                        )
                    ),
                    "spec_id": config["spec"]["spec_id"],
                    "arm_id": arm_id,
                    "carrier_state": arm_states[arm_id],
                    "decision": result["decision"],
                    "risk_bank_canonical_sha256": risk_bank.canonical_sha256,
                    "source_p1_state_sha256": _file_sha256(source_p1_path),
                },
                output_root / "p5_dataset_strict_state.pt",
            )
            _write_json(output_root / "mechanism_metrics.json", result)
            return result

        if r3_enabled:
            expected_arms = {"P1-A", "P1-B", "P2-CAIP", "P4-DGCAIP"}
            if set(arms) != expected_arms:
                raise ValueError("R3-DIAG arm set changed.")
            route_cosines = [
                _vector_cosine(p2_gradient, p4_gradient)
                for p2_gradient, p4_gradient in zip(
                    arm_route_gradients["P2-CAIP"],
                    arm_route_gradients["P4-DGCAIP"],
                )
            ]
            h1 = build_rejection_attribution(
                p2_steps=arms["P2-CAIP"]["steps"],
                p4_steps=arms["P4-DGCAIP"]["steps"],
                routed_gradient_cosines=route_cosines,
            )
            same_process_numeric = _p1_replay_report(
                arms["P1-B"],
                {"arms": {"P1": arms["P1-A"]}},
                absolute_tolerance=float(
                    dg_config["p1_replay_absolute_tolerance"]
                ),
                relative_tolerance=float(
                    dg_config["p1_replay_relative_tolerance"]
                ),
            )
            h2 = build_same_process_replay(
                p1_a_initial_sha256=arm_initial_hashes["P1-A"],
                p1_b_initial_sha256=arm_initial_hashes["P1-B"],
                p1_a_batch_sha256=arm_batch_hashes["P1-A"],
                p1_b_batch_sha256=arm_batch_hashes["P1-B"],
                replay_report=same_process_numeric,
            )
            shared_batches = len(
                {tuple(values) for values in arm_batch_hashes.values()}
            ) == 1
            checks = {
                "trace_complete": bool(h1["trace_complete"]),
                "active_trace_decisions_match": bool(
                    h1["active_trace_decisions_match"]
                ),
                "shared_initial_adapter": len(
                    set(arm_initial_hashes.values())
                ) == 1,
                "shared_batch_sequence": shared_batches,
                "p1_replay_inputs_match": bool(
                    h2["initial_adapter_match"]
                    and h2["batch_sequence_match"]
                ),
                "h1_label_emitted": h1["label"]
                in {
                    "caip_common_infeasibility",
                    "js_incremental_blocker",
                    "ranking_route_shift",
                    "inconclusive_mixed",
                },
                "h2_label_emitted": h2["label"]
                in {
                    "within_process_replay_pass",
                    "within_process_nondeterminism",
                    "replay_invalid_input_mismatch",
                },
            }
            result = {
                "schema": "tausb.dgcaip-r3-diagnostic.v1",
                "spec_id": config["spec"]["spec_id"],
                "split_hash": _split_hash(split),
                "source_p1_state_sha256": _file_sha256(source_p1_path),
                "source_p1_metrics_sha256": _file_sha256(
                    source_p1_metrics_path
                ),
                "d0_report_sha256": _file_sha256(d0_path),
                "target_weight_calibration": target_calibration.state_dict(),
                "nla_calibration": nla_calibration.state_dict(),
                "dgcaip_calibration": dg_calibration.state_dict(),
                "arm_initial_adapter_sha256": arm_initial_hashes,
                "arm_final_adapter_sha256": arm_final_hashes,
                "arm_batch_sha256": arm_batch_hashes,
                "initial": initial_summary,
                "arms": arms,
                "h1": h1,
                "h2": h2,
                "decision": {
                    "checks": checks,
                    "pass": all(checks.values()),
                },
                "elapsed_seconds": time.monotonic() - start,
            }
            _write_json(
                output_root / "backtracking_trace.json",
                {
                    "schema": "tausb.dgcaip-r3-backtracking-trace.v1",
                    "P2-CAIP": arms["P2-CAIP"]["steps"],
                    "P4-DGCAIP": arms["P4-DGCAIP"]["steps"],
                },
            )
            _write_json(output_root / "p1_same_process_replay.json", h2)
            _write_json(output_root / "rejection_attribution.json", h1)
            _write_json(output_root / "mechanism_metrics.json", result)
            return result

        p1 = arms["P1-R"]
        p2 = arms["P2-CAIP"]
        p3 = arms["P3-DIST"]
        p4 = arms["P4-DGCAIP"]
        p4_steps = p4["steps"]
        q4_improvements = {
            name: _relative_improvement(p2["fixed_q4"][name], p4["fixed_q4"][name])
            for name in ("probability", "iou", "alignment", "js")
        }
        q1_nonworse = all(
            p4["fixed_q1"][name] <= 1.10 * p2["fixed_q1"][name] + 1.0e-12
            for name in ("probability", "iou", "alignment")
        )
        p4_retention = _median(item["attack_retention"] for item in p4_steps)
        p1_retention = _median(item["attack_retention"] for item in p1["steps"])
        pattern_p1 = float(
            (arm_deltas["P1-R"] - initial_deltas).square().mean().sqrt()
        )
        pattern_p4 = float(
            (arm_deltas["P4-DGCAIP"] - initial_deltas).square().mean().sqrt()
        )
        p3_damage = _mean_damage(p3["fixed_q4"])
        p4_damage = _mean_damage(p4["fixed_q4"])
        p1_replay = _p1_replay_report(
            p1,
            source_p1_metrics,
            absolute_tolerance=float(dg_config["p1_replay_absolute_tolerance"]),
            relative_tolerance=float(dg_config["p1_replay_relative_tolerance"]),
        )
        checks = {
            "q4_three_of_four_improve_20pct": sum(
                value >= 0.20 for value in q4_improvements.values()
            ) >= 3,
            "ranking_gain_over_uniform_dist": _relative_improvement(
                p3_damage, p4_damage
            ) >= 0.10,
            "q1_nonworse": q1_nonworse,
            "attack_retention": p4_retention >= 0.70
            and p4_retention >= p1_retention - 0.05,
            "cicr_preserved": p4["cicr_cosine_median"]
            >= p1["cicr_cosine_median"] - 0.02,
            "pattern_preserved": pattern_p4 >= 0.80 * pattern_p1,
            "orthogonality": max(
                (float(item["max_projected_row_dot"]) for item in p4_steps),
                default=0.0,
            ) <= 1.0e-5,
            "null_dimension": min(
                (int(item["null_dimension"]) for item in p4_steps), default=0
            ) > 0,
            "protection_budget": all(
                0.20 <= float(item["explicit_protection_norm_ratio"]) <= 0.30
                for item in p4_steps
                if item["route_mode"] != "projected_target"
            ),
            "backtrack_skip": p4["backtrack_skip_ratio"] < 0.50,
        }
        if not production_e20:
            checks["p1_replay"] = bool(p1_replay["pass"])
        decision = {
            "checks": checks,
            "pass": all(checks.values()),
            "q4_improvements": q4_improvements,
            "p4_vs_p3_q4_damage_improvement": _relative_improvement(
                p3_damage, p4_damage
            ),
        }
        accepted_p4_steps = [item for item in p4_steps if item["accepted"]]
        shared_batches = len(
            {tuple(values) for values in arm_batch_hashes.values()}
        ) == 1
        verification_carrier = _clone_detector_carrier(
            base_carrier, torch.device("cpu")
        )
        verification_carrier.load_state_dict(
            arm_states["P4-DGCAIP"], strict=True
        )
        verification_hash = _parameter_sha256(
            adapter_parameters(verification_carrier)
        )
        state_integrity_checks = {
            "strict_backend": bool(
                not production_e20
                or (
                    backend is not None
                    and backend["cublas_workspace_config"] == ":4096:8"
                    and backend["deterministic_algorithms"]
                    and not backend["deterministic_warn_only"]
                )
            ),
            "shared_initial_adapter": len(set(arm_initial_hashes.values())) == 1,
            "shared_batch_sequence": bool(not production_e20 or shared_batches),
            "p4_state_finite": _all_finite(arm_states["P4-DGCAIP"]),
            "p4_metrics_finite": _all_finite(p4),
            "p4_update_accepted": len(accepted_p4_steps) >= 1,
            "p4_adapter_changed": bool(
                not production_e20
                or arm_final_hashes["P4-DGCAIP"]
                != arm_initial_hashes["P4-DGCAIP"]
            ),
            "p4_linf": float(p4["full_perturbation_linf"])
            <= 16.0 / 255.0 + 1.0e-6,
            "p4_support": float(p4["support_outside_linf"]) == 0.0,
            "frozen_modules_unchanged": frozen_before == frozen_after,
            "p4_state_roundtrip": bool(
                not production_e20
                or verification_hash == arm_final_hashes["P4-DGCAIP"]
            ),
            "orthogonality": max(
                (
                    float(item["max_projected_row_dot"])
                    for item in accepted_p4_steps
                ),
                default=float("inf"),
            )
            <= 1.0e-5,
            "null_dimension": min(
                (int(item["null_dimension"]) for item in accepted_p4_steps),
                default=0,
            )
            > 0,
        }
        state_integrity = {
            "schema": "tausb.dgcaip-state-integrity.v1",
            "checks": state_integrity_checks,
            "pass": all(state_integrity_checks.values()),
            "accepted_p4_steps": len(accepted_p4_steps),
            "frozen_before": frozen_before,
            "frozen_after": frozen_after,
            "strict_backend": backend,
        }
        result = {
            "schema": "tausb.dgcaip-mechanism.v1",
            "spec_id": config["spec"]["spec_id"],
            "split_hash": _split_hash(split),
            "source_p1_state_sha256": _file_sha256(source_p1_path),
            "source_p1_metrics_sha256": _file_sha256(source_p1_metrics_path),
            "d0_report_sha256": _file_sha256(d0_path),
            "target_weight_calibration": target_calibration.state_dict(),
            "nla_calibration": nla_calibration.state_dict(),
            "dgcaip_calibration": dg_calibration.state_dict(),
            "arm_initial_adapter_sha256": arm_initial_hashes,
            "arm_final_adapter_sha256": arm_final_hashes,
            "arm_batch_sha256": arm_batch_hashes,
            "shared_initial_adapter_sha256": next(iter(arm_initial_hashes.values())),
            "initial": initial_summary,
            "arms": arms,
            "p1_replay": p1_replay,
            "p1_replay_role": (
                "historical_reference_only_pre_deterministic_resize"
                if production_e20
                else "scientific_gate"
            ),
            "decision": decision,
            "state_integrity": state_integrity,
            "elapsed_seconds": time.monotonic() - start,
        }
        metrics_path = output_root / "mechanism_metrics.json"
        _write_json(metrics_path, result)
        save_p4 = bool(
            state_integrity["pass"] if production_e20 else decision["pass"]
        )
        if save_p4:
            p4_state_path = output_root / "p4_dgcaip_state.pt"
            torch.save(
                {
                    "schema": "tausb.dgcaip-state.v1",
                    "arm_id": "P4-DGCAIP",
                    "carrier_state": arm_states["P4-DGCAIP"],
                    "source_p1_state_sha256": _file_sha256(source_p1_path),
                    "source_p1_metrics_sha256": _file_sha256(
                        source_p1_metrics_path
                    ),
                    "d0_report_sha256": _file_sha256(d0_path),
                    "mechanism_metrics_sha256": _file_sha256(metrics_path),
                    "mechanism_config_sha256": _canonical_json_sha256(config),
                    "state_integrity_gate_passed": bool(
                        state_integrity["pass"]
                    ),
                    "mechanism_scientific_gate_passed": bool(
                        decision["pass"]
                    ),
                    "dgcaip_calibration": dg_calibration.state_dict(),
                },
                p4_state_path,
            )
            if production_e20:
                hiding_provenance = hiding_state.get(
                    "e2e_v0_hiding_provenance"
                )
                if not isinstance(hiding_provenance, Mapping):
                    raise ValueError("DG-CAIP P4 hiding provenance is missing.")
                candidate = build_dgcaip_p4_candidate_state_payload(
                    carrier=verification_carrier,
                    secret=primary_secret.detach().cpu(),
                    target_class_id=14,
                    secret_source_sha256=config["secrets"][
                        "primary_source_sha256"
                    ],
                    secret_tensor_sha256=config["secrets"][
                        "primary_tensor_sha256"
                    ],
                    source_manifest_sha256=config["secrets"][
                        "manifest_sha256"
                    ],
                    train_split_sha256=config["dataset"][
                        "train_label_manifest_sha256"
                    ],
                    mechanism_scientific_gate_passed=bool(decision["pass"]),
                    provenance_hashes={
                        "hiding_metrics_sha256": hiding_provenance[
                            "hiding_metrics_sha256"
                        ],
                        "hiding_checkpoint_sha256": hiding_provenance[
                            "hiding_checkpoint_sha256"
                        ],
                        "hiding_split_sha256": hiding_state["split_hash"],
                        "mechanism_metrics_sha256": _file_sha256(metrics_path),
                        "mechanism_scientific_decision_sha256": _canonical_json_sha256(
                            decision
                        ),
                        "state_integrity_decision_sha256": _canonical_json_sha256(
                            state_integrity
                        ),
                        "mechanism_config_sha256": _canonical_json_sha256(
                            config
                        ),
                        "p4_state_sha256": _file_sha256(p4_state_path),
                        "source_p1_state_sha256": _file_sha256(
                            source_p1_path
                        ),
                        "source_p1_metrics_sha256": _file_sha256(
                            source_p1_metrics_path
                        ),
                        "d0_report_sha256": _file_sha256(d0_path),
                        "repair_report_sha256": _file_sha256(
                            repair_report_path
                        ),
                    },
                )
                torch.save(
                    candidate,
                    output_root / "p4_dgcaip_candidate_sdh_state.pt",
                )
        return result
    finally:
        engine.close()
