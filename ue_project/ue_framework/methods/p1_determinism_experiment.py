from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from .constraint_gradient_router import route_multi_parameter_gradients
from .detector_lfc import DetectorLFCPrototypeBank
from .dgcaip import FrozenDGCAIPGradientCalibration
from .dgcaip_experiment import (
    _dgcaip_component_losses,
    _load_engine,
    _prepare_experiment,
    _validate_d0_report_binding,
)
from .instance_cicr import FrozenInstanceCICRBank
from .non_target_logit_alignment import FrozenNLAGradientCalibration
from .p1_determinism_audit import (
    ABSOLUTE_TOLERANCE,
    AUDIT_SPEC_ID,
    RELATIVE_TOLERANCE,
    TensorTrace,
    backend_manifest,
    capture_engine_snapshot,
    capture_module_snapshot,
    capture_rng_snapshot,
    choose_primary_label,
    compare_traces,
    enable_strict_determinism,
    engine_snapshot_manifest,
    module_snapshot_manifest,
    payload_sha256,
    restore_engine_snapshot,
    restore_module_snapshot,
    restore_rng_snapshot,
    rng_manifest,
)
from .sdh_experiment import (
    _batches,
    _clone_detector_carrier,
    _component_losses,
    _copy_parameters_,
    _file_sha256,
    _flatten_autograd_norm,
    _resolve,
    _split_hash,
    _time_guard,
    _write_json,
    validate_sdh_experiment_config,
)
from .sdh_mechanism import (
    FrozenTargetGradientCalibration,
    SDHBatch,
    adapter_parameters,
    compose_sdh_target_objective,
    load_sdh_batch,
)


def _batch_to_device(batch: SDHBatch, device: torch.device) -> SDHBatch:
    images = batch.images.detach().clone().to(device)
    yolo_batch = {
        name: (value.detach().clone().to(device) if torch.is_tensor(value) else value)
        for name, value in batch.yolo_batch.items()
    }
    yolo_batch["img"] = images
    return SDHBatch(
        images=images,
        yolo_batch=yolo_batch,
        boxes_by_image=tuple(value.detach().clone().to(device) for value in batch.boxes_by_image),
        image_ids=batch.image_ids,
        person_cooccur=batch.person_cooccur,
    )


def _batch_tensors(batch: SDHBatch) -> Dict[str, Any]:
    return {
        "images": batch.images,
        "batch_idx": batch.yolo_batch["batch_idx"],
        "cls": batch.yolo_batch["cls"],
        "bboxes": batch.yolo_batch["bboxes"],
        "boxes_by_image": batch.boxes_by_image,
    }


def _flatten_gradient(
    loss: torch.Tensor, parameters: Sequence[torch.Tensor]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    flattened = torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )
    if not torch.isfinite(flattened).all():
        raise ValueError("P1 audit component gradient is non-finite.")
    return flattened


def _state_manifest(
    *,
    engine: Any,
    carrier: torch.nn.Module,
    dlfc_bank: DetectorLFCPrototypeBank,
    cicr_bank: FrozenInstanceCICRBank,
    target_weights: Mapping[str, float],
    dgcaip_weights: Mapping[str, float],
    lambda_nla: float,
) -> Dict[str, Any]:
    model = module_snapshot_manifest(capture_module_snapshot(engine.model))
    carrier_state = module_snapshot_manifest(capture_module_snapshot(carrier))
    engine_state = engine_snapshot_manifest(capture_engine_snapshot(engine))
    rng = rng_manifest(capture_rng_snapshot())
    payload = {
        "model": model,
        "carrier": carrier_state,
        "engine": engine_state,
        "dlfc_bank_sha256": payload_sha256(dlfc_bank.state_dict()),
        "cicr_bank_sha256": payload_sha256(cicr_bank.state_dict()),
        "target_weights_sha256": payload_sha256(dict(target_weights)),
        "dgcaip_weights_sha256": payload_sha256(dict(dgcaip_weights)),
        "lambda_nla": float(lambda_nla),
        "rng": rng,
        "backend": backend_manifest(),
    }
    payload["sha256"] = payload_sha256(payload)
    return payload


def _trace_input_match(report: Mapping[str, Any]) -> bool:
    comparisons = report.get("comparisons", {})
    required = [
        value
        for key, value in comparisons.items()
        if key.startswith("input.cpu/") or key.startswith("input.cuda/")
    ]
    return bool(required) and all(bool(value.get("exact_match")) for value in required)


def _pair_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    comparison = compare_traces(
        right["trace"],
        left["trace"],
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    comparison.update(
        {
            "input_match": _trace_input_match(comparison),
            "initial_state_match": left["state_pre"]["sha256"]
            == right["state_pre"]["sha256"],
            "parameter_state_unchanged": bool(
                left["parameter_state_unchanged"]
                and right["parameter_state_unchanged"]
            ),
            "left_state_pre_sha256": left["state_pre"]["sha256"],
            "right_state_pre_sha256": right["state_pre"]["sha256"],
            "left_state_post_sha256": left["state_post"]["sha256"],
            "right_state_post_sha256": right["state_post"]["sha256"],
        }
    )
    return comparison


def _serialize_repeat(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trace": value["trace"].manifest(),
        "state_pre": value["state_pre"],
        "state_post": value["state_post"],
        "parameter_state_unchanged": value["parameter_state_unchanged"],
        "parameter_state_mutations": value["parameter_state_mutations"],
        "image_ids": list(value["image_ids"]),
    }


def _run_once(
    *,
    engine: Any,
    carrier: torch.nn.Module,
    cpu_batch: SDHBatch,
    secret: torch.Tensor,
    dlfc_bank: DetectorLFCPrototypeBank,
    cicr_bank: FrozenInstanceCICRBank,
    target_weights: Mapping[str, float],
    dgcaip_weights: Mapping[str, float],
    lambda_nla: float,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
    trace = TensorTrace()
    trace.add("input.cpu", _batch_tensors(cpu_batch))
    batch = _batch_to_device(cpu_batch, device)
    trace.add("input.cuda", {**_batch_tensors(batch), "secret": secret})
    state_pre = _state_manifest(
        engine=engine,
        carrier=carrier,
        dlfc_bank=dlfc_bank,
        cicr_bank=cicr_bank,
        target_weights=target_weights,
        dgcaip_weights=dgcaip_weights,
        lambda_nla=lambda_nla,
    )

    observation = engine.observe(
        batch,
        carrier,
        secret,
        dgcaip_mode="off",
        trace_callback=trace.add,
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
    trace.add(
        "loss.components",
        {
            **components,
            "objective": objective.loss,
            "nla": observation.nla.loss,
            "nla_by_class": observation.nla.per_class_loss,
        },
    )
    parameters = adapter_parameters(carrier)
    routed = route_multi_parameter_gradients(
        parameters=parameters,
        target_loss=objective.loss,
        per_class_nla_losses={
            str(key): value for key, value in observation.nla.per_class_loss.items()
        },
        nla_loss=observation.nla.loss,
        nla_weight=lambda_nla,
        trace_callback=trace.add,
    )
    component_gradients = {
        name: _flatten_gradient(loss, parameters)
        for name, loss in components.items()
    }
    component_gradients["objective"] = _flatten_gradient(objective.loss, parameters)
    component_gradients["nla_total"] = _flatten_gradient(
        observation.nla.loss,
        parameters,
    )
    component_gradients.update(
        {
            "nla_class_%s" % key: _flatten_gradient(loss, parameters)
            for key, loss in observation.nla.per_class_loss.items()
        }
    )
    trace.add("grad.components", component_gradients)
    trace.add(
        "route.matrix",
        {
            "active_class_ids": torch.tensor(
                [int(value) for value in routed.active_classes],
                device=device,
                dtype=torch.long,
            )
        },
    )

    rng_before_candidate = capture_rng_snapshot()
    candidate_carrier = _clone_detector_carrier(carrier, device)
    candidate_parameters = adapter_parameters(candidate_carrier)
    candidate_values = tuple(
        parameter.detach() - float(learning_rate) * gradient.detach()
        for parameter, gradient in zip(parameters, routed.parameter_gradients)
    )
    _copy_parameters_(candidate_parameters, candidate_values)
    restore_rng_snapshot(rng_before_candidate)
    with torch.no_grad():
        candidate_observation = engine.observe(batch, candidate_carrier, secret)
    class_ids = sorted(candidate_observation.per_class_probability_drop)
    trace.add(
        "candidate.eval",
        {
            "class_ids": torch.tensor(class_ids, device=device, dtype=torch.long),
            "probability_drop": torch.tensor(
                [candidate_observation.per_class_probability_drop[key] for key in class_ids],
                device=device,
                dtype=torch.float32,
            ),
        },
    )

    state_post = _state_manifest(
        engine=engine,
        carrier=carrier,
        dlfc_bank=dlfc_bank,
        cicr_bank=cicr_bank,
        target_weights=target_weights,
        dgcaip_weights=dgcaip_weights,
        lambda_nla=lambda_nla,
    )
    mutation_checks = {
        "model": state_pre["model"] == state_post["model"],
        "carrier": state_pre["carrier"] == state_post["carrier"],
        "dlfc_bank": state_pre["dlfc_bank_sha256"] == state_post["dlfc_bank_sha256"],
        "cicr_bank": state_pre["cicr_bank_sha256"] == state_post["cicr_bank_sha256"],
        "target_weights": (
            state_pre["target_weights_sha256"]
            == state_post["target_weights_sha256"]
        ),
        "dgcaip_weights": (
            state_pre["dgcaip_weights_sha256"]
            == state_post["dgcaip_weights_sha256"]
        ),
    }
    mutations = sorted(name for name, matches in mutation_checks.items() if not matches)
    return {
        "trace": trace,
        "state_pre": state_pre,
        "state_post": state_post,
        "parameter_state_unchanged": not mutations,
        "parameter_state_mutations": mutations,
        "image_ids": batch.image_ids,
    }


def _prepare_context(
    config: Mapping[str, Any],
    *,
    config_base: Path,
    start: float,
) -> Dict[str, Any]:
    device = torch.device(str(config["runtime"]["device"]))
    base_carrier, primary_secret, _, _, label_dir, split = _prepare_experiment(
        config, config_base=config_base
    )
    engine = _load_engine(config, config_base=config_base)
    batch_size = int(config["mechanism"]["batch_size"])
    calibration_batches = _batches(split["calibration"], batch_size)
    heldout_batches = _batches(split["heldout"], batch_size)

    def load(paths_batch: Sequence[Path], target_device: torch.device) -> SDHBatch:
        return load_sdh_batch(
            paths_batch,
            label_dir=label_dir,
            image_size=640,
            target_class_id=14,
            device=target_device,
        )

    initial_observations = []
    with torch.no_grad():
        for paths_batch in calibration_batches:
            _time_guard(start, 300.0, "P1 determinism calibration")
            initial_observations.append(
                engine.observe(
                    load(paths_batch, device),
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
    target_norms = {
        name: [] for name in ("easy", "reveal", "rms", "dlfc", "cicr", "floor")
    }
    dgcaip_norms = {
        name: []
        for name in ("classification", "box", "alignment", "distribution")
    }
    warmup_observations = []
    warmup_count = int(config["mechanism"]["weight_calibration_batches"])
    for paths_batch in calibration_batches[:warmup_count]:
        _time_guard(start, 300.0, "P1 determinism warmup")
        observation = engine.observe(
            load(paths_batch, device),
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
            raise RuntimeError("P1 audit DG-CAIP warm-up observation is missing.")
        for name, loss in _dgcaip_component_losses(observation.dgcaip).items():
            dgcaip_norms[name].append(
                _flatten_autograd_norm(loss, omega, retain_graph=True)
            )
    target_calibration = FrozenTargetGradientCalibration()
    target_weights = target_calibration.calibrate(target_norms, split="warmup")
    dgcaip_calibration = FrozenDGCAIPGradientCalibration()
    dgcaip_weights = dgcaip_calibration.calibrate(dgcaip_norms, split="warmup")

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
    lambda_nla = nla_calibration.calibrate(projected_norms, nla_norms, split="warmup")
    with torch.no_grad():
        for paths_batch in heldout_batches:
            _time_guard(start, 300.0, "P1 determinism held-out prelude")
            engine.observe(
                load(paths_batch, device),
                base_carrier,
                primary_secret,
                dgcaip_mode="dist",
                dgcaip_component_weights=dgcaip_weights,
            )
    prelude_observation_count = (
        len(calibration_batches) + warmup_count + len(heldout_batches)
    )
    if int(engine._counter) != prelude_observation_count:
        raise RuntimeError("P1 audit R4 prelude observation count changed.")
    cpu_batch = load(calibration_batches[0], torch.device("cpu"))
    return {
        "device": device,
        "base_carrier": base_carrier,
        "secret": primary_secret,
        "engine": engine,
        "cpu_batch": cpu_batch,
        "dlfc_bank": dlfc_bank,
        "cicr_bank": cicr_bank,
        "target_weights": target_weights,
        "target_calibration_state": target_calibration.state_dict(),
        "dgcaip_weights": dgcaip_weights,
        "dgcaip_calibration_state": dgcaip_calibration.state_dict(),
        "lambda_nla": lambda_nla,
        "nla_calibration_state": nla_calibration.state_dict(),
        "prelude_observation_count": prelude_observation_count,
        "split_hash": _split_hash(split),
        "label_dir": label_dir,
    }


def _validate_bound_artifacts(config: Mapping[str, Any], *, config_base: Path) -> None:
    dgcaip = config["dgcaip"]
    for path_key, hash_key in (
        ("source_p1_state", "source_p1_state_sha256"),
        ("source_p1_metrics", "source_p1_metrics_sha256"),
        ("d0_report", "d0_report_sha256"),
    ):
        path = _resolve(config_base, str(dgcaip[path_key]))
        if _file_sha256(path) != str(dgcaip[hash_key]).lower():
            raise ValueError("P1 audit frozen artifact hash mismatch: %s" % path_key)
    d0_path = _resolve(config_base, str(dgcaip["d0_report"]))
    report = json.loads(d0_path.read_text(encoding="utf-8"))
    _validate_d0_report_binding(report, config, dgcaip)


def _common_repeat_arguments(
    context: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "cpu_batch": context["cpu_batch"],
        "secret": context["secret"],
        "dlfc_bank": context["dlfc_bank"],
        "cicr_bank": context["cicr_bank"],
        "target_weights": context["target_weights"],
        "dgcaip_weights": context["dgcaip_weights"],
        "lambda_nla": context["lambda_nla"],
        "learning_rate": float(config["mechanism"]["learning_rate"]),
        "device": context["device"],
    }


def _run_shared_pair(
    context: Mapping[str, Any], config: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    values = []
    for _ in range(2):
        carrier = _clone_detector_carrier(context["base_carrier"], context["device"])
        values.append(
            _run_once(
                engine=context["engine"],
                carrier=carrier,
                **_common_repeat_arguments(context, config),
            )
        )
    traces = {"A": _serialize_repeat(values[0]), "B": _serialize_repeat(values[1])}
    return traces, _pair_payload(*values)


def _run_reset_pair(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    model_snapshot: Any,
    engine_snapshot: Any,
    rng_snapshot: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    carrier = _clone_detector_carrier(context["base_carrier"], context["device"])
    carrier_snapshot = capture_module_snapshot(carrier)
    values = []
    for _ in range(2):
        restore_rng_snapshot(rng_snapshot)
        restore_module_snapshot(context["engine"].model, model_snapshot)
        restore_module_snapshot(carrier, carrier_snapshot)
        restore_engine_snapshot(context["engine"], engine_snapshot)
        values.append(
            _run_once(
                engine=context["engine"],
                carrier=carrier,
                **_common_repeat_arguments(context, config),
            )
        )
    traces = {"A": _serialize_repeat(values[0]), "B": _serialize_repeat(values[1])}
    return traces, _pair_payload(*values)


def _run_fresh_pair(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_base: Path,
    model_snapshot: Any,
    engine_snapshot: Any,
    rng_snapshot: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_carrier_snapshot = capture_module_snapshot(context["base_carrier"])
    values = []
    for _ in range(2):
        restore_rng_snapshot(rng_snapshot)
        engine = _load_engine(config, config_base=config_base)
        try:
            restore_module_snapshot(engine.model, model_snapshot)
            restore_engine_snapshot(engine, engine_snapshot)
            carrier = _clone_detector_carrier(context["base_carrier"], context["device"])
            restore_module_snapshot(carrier, base_carrier_snapshot)
            restore_rng_snapshot(rng_snapshot)
            values.append(
                _run_once(
                    engine=engine,
                    carrier=carrier,
                    **_common_repeat_arguments(context, config),
                )
            )
        finally:
            engine.close()
    traces = {"A": _serialize_repeat(values[0]), "B": _serialize_repeat(values[1])}
    return traces, _pair_payload(*values)


def run_p1_determinism_lane(
    config: Mapping[str, Any],
    *,
    config_base: Path,
    mode: str,
) -> Dict[str, Any]:
    if mode not in {"normal", "strict"}:
        raise ValueError("P1 determinism lane must be normal or strict.")
    validate_sdh_experiment_config(config)
    if str(config["spec"]["spec_id"]) != AUDIT_SPEC_ID:
        raise ValueError("P1 determinism runner received the wrong SpecID.")
    _validate_bound_artifacts(config, config_base=config_base)
    if mode == "strict":
        enable_strict_determinism()
    elif os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise RuntimeError(
            "Normal P1 audit lane requires CUBLAS_WORKSPACE_CONFIG to be unset."
        )
    start = time.monotonic()
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    if mode == "normal":
        artifact_root.mkdir(parents=True, exist_ok=False)
    elif not artifact_root.is_dir():
        raise FileNotFoundError("Normal audit lane must create the artifact root first.")
    trace_path = artifact_root / ("p1_trace_%s.json" % mode)
    if trace_path.exists():
        raise FileExistsError("P1 audit lane output already exists: %s" % trace_path)
    context = _prepare_context(config, config_base=config_base, start=start)
    try:
        if context["split_hash"] != str(config["dgcaip"]["expected_split_sha256"]):
            raise ValueError("P1 audit split does not match expected_split_sha256.")
        model_snapshot = capture_module_snapshot(context["engine"].model)
        engine_snapshot = capture_engine_snapshot(context["engine"])
        rng_snapshot = capture_rng_snapshot()
        traces: Dict[str, Any] = {}
        pairs: Dict[str, Any] = {}
        if mode == "normal":
            traces["shared"], pairs["shared"] = _run_shared_pair(context, config)
            traces["reset"], pairs["reset"] = _run_reset_pair(
                context,
                config,
                model_snapshot=model_snapshot,
                engine_snapshot=engine_snapshot,
                rng_snapshot=rng_snapshot,
            )
        traces["fresh"], pairs["fresh"] = _run_fresh_pair(
            context,
            config,
            config_base=config_base,
            model_snapshot=model_snapshot,
            engine_snapshot=engine_snapshot,
            rng_snapshot=rng_snapshot,
        )
        input_valid = all(bool(value.get("input_match")) for value in pairs.values())
        state_pairs = [pairs["fresh"]]
        if mode == "normal":
            state_pairs.append(pairs["reset"])
        state_valid = all(
            bool(value.get("initial_state_match"))
            and bool(value.get("parameter_state_unchanged"))
            for value in state_pairs
        )
        result = {
            "schema": "tausb.p1-determinism-lane.v1",
            "spec_id": AUDIT_SPEC_ID,
            "mode": mode,
            "backend": backend_manifest(),
            "split_hash": context["split_hash"],
            "image_ids": list(context["cpu_batch"].image_ids),
            "target_weight_calibration": context["target_calibration_state"],
            "dgcaip_weight_calibration": context["dgcaip_calibration_state"],
            "nla_calibration": context["nla_calibration_state"],
            "prelude_observation_count": context["prelude_observation_count"],
            "input_valid": input_valid,
            "state_valid": state_valid,
            "pairs": pairs,
            "traces": traces,
            "elapsed_seconds": time.monotonic() - start,
        }
        _write_json(trace_path, result)
        return result
    finally:
        context["engine"].close()


def summarize_p1_determinism_audit(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Dict[str, Any]:
    validate_sdh_experiment_config(config)
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    normal_path = artifact_root / "p1_trace_normal.json"
    strict_path = artifact_root / "p1_trace_strict.json"
    strict_error_path = artifact_root / "strict_operator_error.json"
    normal = json.loads(normal_path.read_text(encoding="utf-8"))
    if strict_path.is_file():
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
    elif strict_error_path.is_file():
        strict = json.loads(strict_error_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("Strict P1 audit evidence is missing.")
    decision = choose_primary_label(normal, strict)
    normal_shared_a = normal.get("traces", {}).get("shared", {}).get("A", {})
    normal_shared_trace = normal_shared_a.get("trace", {})
    input_state_manifest = {
        "schema": "tausb.p1-determinism-input-state.v1",
        "spec_id": AUDIT_SPEC_ID,
        "image_ids": normal.get("image_ids", []),
        "split_hash": normal.get("split_hash"),
        "input_tensors": {
            name: value
            for name, value in normal_shared_trace.items()
            if name.startswith("input.cpu/") or name.startswith("input.cuda/")
        },
        "reference_state_pre": normal_shared_a.get("state_pre"),
        "normal_pair_state": {
            name: {
                "input_match": value.get("input_match"),
                "initial_state_match": value.get("initial_state_match"),
                "left_state_pre_sha256": value.get("left_state_pre_sha256"),
                "right_state_pre_sha256": value.get("right_state_pre_sha256"),
            }
            for name, value in normal.get("pairs", {}).items()
        },
        "strict_pair_state": {
            name: {
                "input_match": value.get("input_match"),
                "initial_state_match": value.get("initial_state_match"),
                "left_state_pre_sha256": value.get("left_state_pre_sha256"),
                "right_state_pre_sha256": value.get("right_state_pre_sha256"),
            }
            for name, value in strict.get("pairs", {}).items()
        },
    }
    first_divergence = {
        "schema": "tausb.p1-determinism-first-divergence.v1",
        "spec_id": AUDIT_SPEC_ID,
        "normal": {
            name: {
                "first_divergent_stage": value.get("first_divergent_stage"),
                "first_non_allclose_stage": value.get("first_non_allclose_stage"),
                "bitwise_only_drift": value.get("bitwise_only_drift"),
            }
            for name, value in normal.get("pairs", {}).items()
        },
        "strict": {
            name: {
                "first_divergent_stage": value.get("first_divergent_stage"),
                "first_non_allclose_stage": value.get("first_non_allclose_stage"),
                "bitwise_only_drift": value.get("bitwise_only_drift"),
            }
            for name, value in strict.get("pairs", {}).items()
        },
        "strict_operator_error": strict.get("operator_error"),
    }
    mechanical_checks = {
        "normal_pairs_complete": set(normal.get("pairs", {}))
        == {"shared", "reset", "fresh"},
        "strict_complete_or_operator_error": bool(strict.get("operator_error"))
        or set(strict.get("pairs", {})) == {"fresh"},
        "input_valid": bool(decision["input_valid"]),
        "state_valid": bool(decision["state_valid"]),
        "exactly_one_label": bool(decision["label"]),
    }
    result = {
        "schema": "tausb.p1-determinism-audit-summary.v1",
        "spec_id": AUDIT_SPEC_ID,
        "normal_trace_sha256": _file_sha256(normal_path),
        "strict_evidence_sha256": _file_sha256(
            strict_path if strict_path.is_file() else strict_error_path
        ),
        "decision": {
            "checks": mechanical_checks,
            "mechanical_pass": all(mechanical_checks.values()),
            **decision,
        },
    }
    _write_json(artifact_root / "input_state_manifest.json", input_state_manifest)
    _write_json(artifact_root / "first_divergence_report.json", first_divergence)
    _write_json(artifact_root / "determinism_audit_summary.json", result)
    return result


def is_deterministic_operator_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "deterministic" in message and (
        "not have" in message
        or "does not have" in message
        or "nondeterministic" in message
        or "determinism" in message
    )
