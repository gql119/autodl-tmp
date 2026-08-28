from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from ue_framework.methods.constraint_gradient_router import (
    route_multi_parameter_gradients,
)
from ue_framework.methods.p1_determinism_audit import (
    AUDIT_SPEC_ID,
    TensorTrace,
    capture_module_snapshot,
    capture_rng_snapshot,
    choose_primary_label,
    compare_traces,
    enable_strict_determinism,
    module_snapshot_manifest,
    restore_module_snapshot,
    restore_rng_snapshot,
)
from ue_framework.methods.p1_determinism_experiment import (
    is_deterministic_operator_error,
)
from ue_framework.methods.sdh_experiment import (
    DGCAIP_P1_DETERMINISM_AUDIT_SPEC_ID,
    run_mechanism_pilot,
    validate_sdh_experiment_config,
)
from ue_framework.methods.semantic_hiding_carrier import (
    SemanticHidingCarrier,
    render_person_box_carrier,
)
from ue_framework.tools.run_p1_determinism_audit import (
    build_strict_operator_evidence,
)


CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_p1_determinism_audit_v1.yaml"
)


def _trace(value: torch.Tensor, *, stage: str = "render") -> TensorTrace:
    trace = TensorTrace()
    trace.add("input.cpu", {"x": torch.zeros(1)})
    trace.add("input.cuda", {"x": torch.zeros(1)})
    trace.add(stage, {"value": value})
    return trace


def test_trace_comparator_separates_bitwise_and_numeric_drift() -> None:
    reference = _trace(torch.tensor([1.0, 2.0]))
    bitwise = _trace(torch.tensor([1.0 + 5.0e-7, 2.0]))
    numeric = _trace(torch.tensor([1.01, 2.0]))

    bitwise_report = compare_traces(bitwise, reference)
    assert reference.manifest()["input.cpu/x"]["device"] == "cpu"
    assert bitwise_report["pass"] is True
    assert bitwise_report["bitwise_pass"] is False
    assert bitwise_report["bitwise_only_drift"] is True
    assert bitwise_report["first_divergent_stage"] == "render/value"
    assert bitwise_report["first_non_allclose_stage"] is None

    numeric_report = compare_traces(numeric, reference)
    assert numeric_report["pass"] is False
    assert numeric_report["first_non_allclose_stage"] == "render/value"


def test_trace_schema_mismatch_fails_closed() -> None:
    left = _trace(torch.ones(1))
    right = _trace(torch.ones(1))
    right.add("clean.tal", {"fg_mask": torch.ones(1, dtype=torch.bool)})
    report = compare_traces(right, left)
    assert report["valid"] is False
    assert report["first_non_allclose_stage"] == "trace.schema"


def test_trace_rejects_nonfinite_tensor() -> None:
    trace = TensorTrace()
    with pytest.raises(ValueError, match="non-finite"):
        trace.add("render", {"value": torch.tensor([float("nan")])})


def test_module_and_rng_snapshot_restore_detects_single_state_change() -> None:
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout())
    module.eval()
    module[0].weight.grad = torch.ones_like(module[0].weight)
    snapshot = capture_module_snapshot(module)
    manifest = module_snapshot_manifest(snapshot)

    with torch.no_grad():
        module[0].weight.add_(1.0)
    module.train()
    module[0].weight.grad = None
    assert module_snapshot_manifest(capture_module_snapshot(module)) != manifest
    restore_module_snapshot(module, snapshot)
    assert module_snapshot_manifest(capture_module_snapshot(module)) == manifest

    rng = capture_rng_snapshot()
    expected = torch.rand(4)
    restore_rng_snapshot(rng)
    assert torch.equal(torch.rand(4), expected)


def test_conv_constructor_advances_rng_and_snapshot_restores_it() -> None:
    torch.manual_seed(91)
    before = capture_rng_snapshot()
    torch.nn.Conv2d(3, 4, 3)
    after = capture_rng_snapshot()
    assert not torch.equal(before.torch_cpu, after.torch_cpu)
    restore_rng_snapshot(before)
    assert torch.equal(capture_rng_snapshot().torch_cpu, before.torch_cpu)


def _pair(passed: bool, *, first: str | None = None):
    return {
        "valid": True,
        "pass": passed,
        "first_non_allclose_stage": first,
    }


@pytest.mark.parametrize(
    ("normal", "strict", "expected"),
    [
        (
            {"input_valid": False, "state_valid": True, "pairs": {}},
            {"input_valid": True, "state_valid": True, "pairs": {}},
            "invalid_input_tensor_mismatch",
        ),
        (
            {"input_valid": True, "state_valid": False, "pairs": {}},
            {"input_valid": True, "state_valid": True, "pairs": {}},
            "invalid_initial_state_mismatch",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {"shared": _pair(False), "reset": _pair(True), "fresh": _pair(True)},
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(True)}},
            "rng_state_dependency",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {"shared": _pair(False), "reset": _pair(False), "fresh": _pair(True)},
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(True)}},
            "shared_engine_state_dependency",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {"shared": _pair(False), "reset": _pair(False), "fresh": _pair(False)},
            },
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {},
                "operator_error": {"error": "x"},
            },
            "cuda_nondeterministic_operator",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {
                    "shared": _pair(False),
                    "reset": _pair(False),
                    "fresh": _pair(False, first="clean.tal/fg_mask"),
                },
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(False)}},
            "tal_assignment_instability",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {
                    "shared": _pair(False),
                    "reset": _pair(False),
                    "fresh": _pair(False, first="route.projector/projected_target_gradient"),
                },
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(False)}},
            "svd_subspace_instability",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {"shared": _pair(True), "reset": _pair(True), "fresh": _pair(True)},
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(True)}},
            "baseline_not_reproduced",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {
                    "shared": _pair(False),
                    "reset": _pair(False),
                    "fresh": _pair(False, first="grad.components/objective"),
                },
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(False)}},
            "upstream_forward_backward_drift",
        ),
        (
            {
                "input_valid": True,
                "state_valid": True,
                "pairs": {
                    "shared": _pair(False),
                    "reset": _pair(False),
                    "fresh": _pair(False, first="route.final/combined_gradient"),
                },
            },
            {"input_valid": True, "state_valid": True, "pairs": {"fresh": _pair(False)}},
            "unresolved_first_divergence",
        ),
    ],
)
def test_label_decision_tree_is_exactly_one(normal, strict, expected) -> None:
    decision = choose_primary_label(normal, strict)
    assert decision["label"] == expected
    assert isinstance(decision["label"], str)


def test_route_trace_uses_small_row_gram_and_projector_action() -> None:
    parameter = torch.tensor([0.4, -0.2, 0.7], requires_grad=True)
    target = (parameter.square()).sum()
    class_losses = {
        "1": parameter[0] + 2.0 * parameter[1],
        "3": parameter[2] - parameter[1],
    }
    captured = {}

    def callback(stage, values):
        captured.setdefault(stage, {}).update(values)

    result = route_multi_parameter_gradients(
        parameters=(parameter,),
        target_loss=target,
        per_class_nla_losses=class_losses,
        nla_loss=sum(class_losses.values()),
        nla_weight=0.25,
        trace_callback=callback,
    )
    plain = route_multi_parameter_gradients(
        parameters=(parameter,),
        target_loss=target,
        per_class_nla_losses=class_losses,
        nla_loss=sum(class_losses.values()),
        nla_weight=0.25,
    )
    assert captured["route.projector"]["row_gram"].shape == (2, 2)
    assert "row_space_projector" not in captured["route.projector"]
    assert torch.allclose(
        captured["route.projector"]["removed_target_component"],
        result.target_gradient - result.projected_target_gradient,
    )
    assert torch.equal(result.gradient, plain.gradient)


def test_renderer_trace_is_default_off_and_observational() -> None:
    torch.manual_seed(11)
    carrier = SemanticHidingCarrier(input_size=16, width=8, coupling_blocks=1)
    carrier.freeze_for_detector_optimization()
    images = torch.rand(1, 3, 20, 20)
    boxes = (torch.tensor([[2.0, 3.0, 18.0, 19.0]]),)
    secret = torch.rand(1, 3, 16, 16)
    plain = render_person_box_carrier(images, boxes, carrier, secret)
    traced = {}
    observed = render_person_box_carrier(
        images,
        boxes,
        carrier,
        secret,
        trace_callback=lambda stage, values: traced.update({stage: values}),
    )
    assert torch.equal(plain.poisoned, observed.poisoned)
    assert torch.equal(plain.perturbation, observed.perturbation)
    assert set(traced["render"]) == {
        "hosts",
        "canonical_deltas",
        "resized_patches",
        "perturbation",
        "poisoned",
    }


def test_audit_config_is_frozen_and_generic_runner_is_blocked() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert AUDIT_SPEC_ID == DGCAIP_P1_DETERMINISM_AUDIT_SPEC_ID
    assert config["mechanism"]["optimization_steps"] == 1
    assert config["mechanism"]["max_seconds"] == 300
    assert config["audit"]["normal_lanes"] == ["shared", "reset", "fresh"]
    with pytest.raises(ValueError, match="dedicated zero-update runner"):
        run_mechanism_pilot(config, config_base=Path("/tmp/project"))

    changed = copy.deepcopy(config)
    changed["audit"]["total_hard_cap_seconds"] = 301
    with pytest.raises(ValueError, match="total_hard_cap_seconds"):
        validate_sdh_experiment_config(changed)


def test_strict_mode_requires_pre_cuda_cublas_binding(monkeypatch) -> None:
    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )()
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        enable_strict_determinism()
    try:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        state = enable_strict_determinism()
        assert state["deterministic_algorithms"] is True
    finally:
        torch.use_deterministic_algorithms(old_algorithms, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def test_deterministic_operator_error_classifier_is_narrow() -> None:
    assert is_deterministic_operator_error(
        RuntimeError("operation does not have a deterministic implementation")
    )
    assert not is_deterministic_operator_error(RuntimeError("CUDA out of memory"))


def test_strict_operator_error_becomes_registered_evidence() -> None:
    error = RuntimeError("operation does not have a deterministic implementation")
    evidence = build_strict_operator_evidence(
        {"spec": {"spec_id": AUDIT_SPEC_ID}},
        error,
        "synthetic-stack",
    )
    assert evidence["spec_id"] == AUDIT_SPEC_ID
    assert evidence["validation_status"] == "not_completed_due_operator_error"
    assert set(evidence["pairs"]) == set()
    assert evidence["operator_error"]["traceback"] == "synthetic-stack"


def test_controller_has_global_hard_cap_and_shutdown_trap() -> None:
    pre_run = (
        Path(__file__).parents[2]
        / "research_workspace"
        / "experiments"
        / "TAUSB-SDH-DGCAIP-S0-P1-DET-AUDIT"
        / "pre_run"
    )
    controller = (pre_run / "p1_determinism_controller_v1.sh").read_text(
        encoding="utf-8"
    )
    launcher = (pre_run / "p1_determinism_tmux_launch_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "trap shutdown_once EXIT" in controller
    assert "mountpoint -q" in controller
    assert "HARD_CAP_SECONDS=300" in controller
    assert (
        "timeout --signal=TERM --kill-after=15s '${HARD_CAP_SECONDS}s'"
        in launcher
    )
