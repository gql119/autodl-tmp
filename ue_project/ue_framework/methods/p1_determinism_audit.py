from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


AUDIT_SPEC_ID = "TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1"
ABSOLUTE_TOLERANCE = 1.0e-6
RELATIVE_TOLERANCE = 1.0e-4

TRACE_STAGE_ORDER = (
    "input.cpu",
    "input.cuda",
    "render",
    "clean.forward",
    "clean.tal",
    "poison.forward",
    "loss.components",
    "grad.components",
    "route.matrix",
    "route.projector",
    "route.final",
    "candidate.eval",
)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor_sha256 requires a tensor.")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _update_hash(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        digest.update(b"tensor\0")
        digest.update(tensor_sha256(value).encode("ascii"))
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    elif isinstance(value, np.generic):
        digest.update(b"numpy-scalar\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for item in value:
            _update_hash(digest, item)
    elif isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(value)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError("Unsupported hash payload type: %s" % type(value).__name__)


def payload_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def _tensor_manifest(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach()
    floating = tensor.is_floating_point() or tensor.is_complex()
    finite = bool(torch.isfinite(tensor).all()) if floating else True
    output: Dict[str, Any] = {
        "sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
    }
    if tensor.numel() and floating and finite:
        real = tensor.real if tensor.is_complex() else tensor
        output.update(
            {
                "min": float(real.min().detach().cpu()),
                "max": float(real.max().detach().cpu()),
                "l2": float(real.double().norm().detach().cpu()),
            }
        )
    return output


def _flatten_tensors(
    prefix: str,
    value: Any,
    output: Dict[str, torch.Tensor],
    devices: Dict[str, str],
) -> None:
    if torch.is_tensor(value):
        if prefix in output:
            raise ValueError("Duplicate trace tensor: %s" % prefix)
        devices[prefix] = str(value.device)
        output[prefix] = value.detach().cpu().contiguous().clone()
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = "%s.%s" % (prefix, key) if prefix else str(key)
            _flatten_tensors(child, value[key], output, devices)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = "%s.%d" % (prefix, index) if prefix else str(index)
            _flatten_tensors(child, item, output, devices)
        return
    raise TypeError("Trace values must be tensors or tensor containers: %s" % prefix)


class TensorTrace:
    def __init__(self) -> None:
        self._tensors: Dict[str, torch.Tensor] = {}
        self._devices: Dict[str, str] = {}

    def add(self, stage: str, values: Mapping[str, Any]) -> None:
        if stage not in TRACE_STAGE_ORDER:
            raise ValueError("Unknown P1 audit trace stage: %s" % stage)
        if not isinstance(values, Mapping) or not values:
            raise ValueError("Trace stage must contain at least one tensor.")
        flattened: Dict[str, torch.Tensor] = {}
        devices: Dict[str, str] = {}
        _flatten_tensors("", values, flattened, devices)
        for name, tensor in flattened.items():
            key = "%s/%s" % (stage, name)
            if key in self._tensors:
                raise ValueError("Duplicate trace key: %s" % key)
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError("P1 audit trace contains a non-finite tensor: %s" % key)
            self._tensors[key] = tensor
            self._devices[key] = devices[name]

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._tensors

    def manifest(self) -> Dict[str, Any]:
        return {
            key: {**_tensor_manifest(value), "device": self._devices[key]}
            for key, value in self._ordered_items()
        }

    def _ordered_items(self):
        stage_rank = {name: index for index, name in enumerate(TRACE_STAGE_ORDER)}
        return sorted(
            self._tensors.items(),
            key=lambda item: (stage_rank[item[0].split("/", 1)[0]], item[0]),
        )


def _compare_tensor(
    observed: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Dict[str, Any]:
    same_shape = tuple(observed.shape) == tuple(reference.shape)
    same_dtype = observed.dtype == reference.dtype
    output: Dict[str, Any] = {
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "exact_match": same_shape and same_dtype and torch.equal(observed, reference),
        "observed_sha256": tensor_sha256(observed),
        "reference_sha256": tensor_sha256(reference),
    }
    if not same_shape or not same_dtype:
        output.update({"allclose": False, "max_abs": None, "max_rel": None})
        return output
    if observed.is_floating_point() or observed.is_complex():
        left = observed.to(torch.float64)
        right = reference.to(torch.float64)
        difference = (left - right).abs()
        maximum_absolute = float(difference.max()) if difference.numel() else 0.0
        relative = difference / right.abs().clamp_min(1.0e-12)
        maximum_relative = float(relative.max()) if relative.numel() else 0.0
        output.update(
            {
                "allclose": bool(
                    torch.allclose(
                        left,
                        right,
                        atol=float(absolute_tolerance),
                        rtol=float(relative_tolerance),
                        equal_nan=False,
                    )
                ),
                "max_abs": maximum_absolute,
                "max_rel": maximum_relative,
            }
        )
    else:
        exact = bool(torch.equal(observed, reference))
        output.update(
            {
                "allclose": exact,
                "max_abs": 0.0 if exact else None,
                "max_rel": 0.0 if exact else None,
            }
        )
    return output


def compare_traces(
    observed: TensorTrace,
    reference: TensorTrace,
    *,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    observed_keys = set(observed.tensors)
    reference_keys = set(reference.tensors)
    if observed_keys != reference_keys:
        return {
            "valid": False,
            "pass": False,
            "missing_observed": sorted(reference_keys - observed_keys),
            "missing_reference": sorted(observed_keys - reference_keys),
            "first_divergent_stage": "trace.schema",
            "first_non_allclose_stage": "trace.schema",
            "comparisons": {},
        }
    stage_rank = {name: index for index, name in enumerate(TRACE_STAGE_ORDER)}
    keys = sorted(
        observed_keys,
        key=lambda key: (stage_rank[key.split("/", 1)[0]], key),
    )
    comparisons = {}
    first_exact: Optional[str] = None
    first_numeric: Optional[str] = None
    for key in keys:
        comparison = _compare_tensor(
            observed.tensors[key],
            reference.tensors[key],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        comparisons[key] = comparison
        if not comparison["exact_match"] and first_exact is None:
            first_exact = key
        if not comparison["allclose"] and first_numeric is None:
            first_numeric = key
    return {
        "valid": True,
        "pass": first_numeric is None,
        "bitwise_pass": first_exact is None,
        "bitwise_only_drift": first_exact is not None and first_numeric is None,
        "first_divergent_stage": first_exact,
        "first_non_allclose_stage": first_numeric,
        "comparisons": comparisons,
    }


@dataclass
class RNGSnapshot:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: Tuple[torch.Tensor, ...]


def capture_rng_snapshot() -> RNGSnapshot:
    cuda_states: Tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(value.cpu().clone() for value in torch.cuda.get_rng_state_all())
    return RNGSnapshot(
        python=copy.deepcopy(random.getstate()),
        numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.get_rng_state().cpu().clone(),
        torch_cuda=cuda_states,
    )


def restore_rng_snapshot(snapshot: RNGSnapshot) -> None:
    random.setstate(copy.deepcopy(snapshot.python))
    np.random.set_state(copy.deepcopy(snapshot.numpy))
    torch.set_rng_state(snapshot.torch_cpu.clone())
    if snapshot.torch_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG without CUDA.")
        torch.cuda.set_rng_state_all([value.clone() for value in snapshot.torch_cuda])


def rng_manifest(snapshot: RNGSnapshot) -> Dict[str, Any]:
    fields = {
        "python": snapshot.python,
        "numpy": snapshot.numpy,
        "torch_cpu": snapshot.torch_cpu,
        "torch_cuda": snapshot.torch_cuda,
    }
    return {"sha256": payload_sha256(fields), "cuda_device_count": len(snapshot.torch_cuda)}


@dataclass
class ModuleSnapshot:
    state: Dict[str, torch.Tensor]
    training: Dict[str, bool]
    gradients: Dict[str, Optional[torch.Tensor]]


def capture_module_snapshot(module: torch.nn.Module) -> ModuleSnapshot:
    return ModuleSnapshot(
        state={name: value.detach().cpu().clone() for name, value in module.state_dict().items()},
        training={name: bool(item.training) for name, item in module.named_modules()},
        gradients={
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        },
    )


def restore_module_snapshot(module: torch.nn.Module, snapshot: ModuleSnapshot) -> None:
    module.load_state_dict(snapshot.state, strict=True)
    modules = dict(module.named_modules())
    if set(modules) != set(snapshot.training):
        raise ValueError("Module topology changed during P1 audit replay.")
    for name, training in snapshot.training.items():
        modules[name].train(training)
    parameters = dict(module.named_parameters())
    if set(parameters) != set(snapshot.gradients):
        raise ValueError("Module parameter topology changed during P1 audit replay.")
    for name, gradient in snapshot.gradients.items():
        parameters[name].grad = (
            None
            if gradient is None
            else gradient.to(
                device=parameters[name].device,
                dtype=parameters[name].dtype,
            ).clone()
        )


def module_snapshot_manifest(snapshot: ModuleSnapshot) -> Dict[str, Any]:
    return {
        "state_sha256": payload_sha256(snapshot.state),
        "training_sha256": payload_sha256(snapshot.training),
        "gradients_sha256": payload_sha256(snapshot.gradients),
        "state_tensor_count": len(snapshot.state),
        "parameter_count": len(snapshot.gradients),
    }


@dataclass
class EngineSnapshot:
    counter: int
    dlfc_counter: int
    last_real_assign: Dict[str, torch.Tensor]


def _assert_capture_idle(capture: Any, *, name: str) -> None:
    if capture._active_tag is not None or capture._records:
        raise RuntimeError("%s capture is not idle." % name)
    if capture._closed:
        raise RuntimeError("%s capture is already closed." % name)


def capture_engine_snapshot(engine: Any) -> EngineSnapshot:
    _assert_capture_idle(engine.capture, name="engine")
    _assert_capture_idle(engine.dlfc_extractor.capture, name="dlfc")
    assignments = {
        str(name): value.detach().cpu().clone()
        for name, value in engine.hijacked.last_real_assign.items()
        if torch.is_tensor(value)
    }
    return EngineSnapshot(
        counter=int(engine._counter),
        dlfc_counter=int(engine.dlfc_extractor._counter),
        last_real_assign=assignments,
    )


def restore_engine_snapshot(engine: Any, snapshot: EngineSnapshot) -> None:
    _assert_capture_idle(engine.capture, name="engine")
    _assert_capture_idle(engine.dlfc_extractor.capture, name="dlfc")
    engine._counter = int(snapshot.counter)
    engine.dlfc_extractor._counter = int(snapshot.dlfc_counter)
    device = next(engine.model.parameters()).device
    engine.hijacked.last_real_assign = {
        name: value.to(device=device) for name, value in snapshot.last_real_assign.items()
    }


def engine_snapshot_manifest(snapshot: EngineSnapshot) -> Dict[str, Any]:
    return {
        "counter": snapshot.counter,
        "dlfc_counter": snapshot.dlfc_counter,
        "last_real_assign_sha256": payload_sha256(snapshot.last_real_assign),
        "last_real_assign_keys": sorted(snapshot.last_real_assign),
        "capture_idle": True,
    }


def backend_manifest() -> Dict[str, Any]:
    cuda_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_devices": devices,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            getattr(
                torch,
                "is_deterministic_algorithms_warn_only_enabled",
                lambda: False,
            )()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": None if cuda_matmul is None else bool(cuda_matmul),
        "cudnn_allow_tf32": None if cudnn_tf32 is None else bool(cudnn_tf32),
    }


def enable_strict_determinism() -> Dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "Strict audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA init."
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return backend_manifest()


def _pair_pass(payload: Mapping[str, Any], name: str) -> bool:
    pair = payload.get("pairs", {}).get(name, {})
    return bool(pair.get("valid")) and bool(pair.get("pass"))


def choose_primary_label(
    normal: Mapping[str, Any],
    strict: Mapping[str, Any],
) -> Dict[str, Any]:
    input_valid = bool(normal.get("input_valid")) and bool(strict.get("input_valid", True))
    state_valid = bool(normal.get("state_valid")) and bool(strict.get("state_valid", True))
    shared_pass = _pair_pass(normal, "shared")
    reset_pass = _pair_pass(normal, "reset")
    fresh_pass = _pair_pass(normal, "fresh")
    strict_fresh_pass = _pair_pass(strict, "fresh")
    strict_error = strict.get("operator_error")

    if not input_valid:
        label = "invalid_input_tensor_mismatch"
    elif not state_valid:
        label = "invalid_initial_state_mismatch"
    elif not shared_pass and reset_pass:
        label = "rng_state_dependency"
    elif not reset_pass and fresh_pass:
        label = "shared_engine_state_dependency"
    elif strict_error or (not fresh_pass and strict_fresh_pass):
        label = "cuda_nondeterministic_operator"
    else:
        fresh_report = normal.get("pairs", {}).get("fresh", {})
        first = str(fresh_report.get("first_non_allclose_stage") or "")
        if first.startswith("clean.tal/"):
            label = "tal_assignment_instability"
        elif first.startswith("route.matrix/") or first.startswith("route.projector/"):
            label = "svd_subspace_instability"
        elif first.startswith(
            (
                "render/",
                "clean.forward/",
                "poison.forward/",
                "loss.components/",
                "grad.components/",
            )
        ):
            label = "upstream_forward_backward_drift"
        elif shared_pass and reset_pass and fresh_pass and strict_fresh_pass:
            label = "baseline_not_reproduced"
        else:
            label = "unresolved_first_divergence"

    allowed = {
        "invalid_input_tensor_mismatch",
        "invalid_initial_state_mismatch",
        "rng_state_dependency",
        "shared_engine_state_dependency",
        "cuda_nondeterministic_operator",
        "tal_assignment_instability",
        "svd_subspace_instability",
        "upstream_forward_backward_drift",
        "baseline_not_reproduced",
        "unresolved_first_divergence",
    }
    if label not in allowed:
        raise RuntimeError("P1 audit emitted an unregistered label.")
    return {
        "label": label,
        "input_valid": input_valid,
        "state_valid": state_valid,
        "pair_pass": {
            "normal_shared": shared_pass,
            "normal_reset": reset_pass,
            "normal_fresh": fresh_pass,
            "strict_fresh": strict_fresh_pass,
        },
        "strict_operator_error": bool(strict_error),
    }
