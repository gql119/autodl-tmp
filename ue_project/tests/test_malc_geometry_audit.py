from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ue_framework.methods.malc_geometry_audit as geometry_module
from ue_framework.methods.constraint_gradient_router import ConstraintTerm
from ue_framework.methods.malc_geometry_audit import (
    PrototypeGeometryAccumulator,
    classify_first_bad_boundary,
    component_gradient_geometry,
    detached_cosine,
    summarize_gradient_geometry,
)
from ue_framework.methods.sirc_malc_geometry import SIRCMALCGeometryWorkflow


def _residuals(values: torch.Tensor) -> SimpleNamespace:
    valid = torch.ones(values.shape[0], dtype=torch.bool)
    return SimpleNamespace(vectors=(values,), pooling_valid=(valid,))


def test_prototype_resultant_and_leave_one_batch_are_exact_for_unimodal_data() -> None:
    audit = PrototypeGeometryAccumulator(num_scales=1)
    audit.update(_residuals(torch.tensor([[1.0, 0.0], [2.0, 0.0]])))
    audit.update(_residuals(torch.tensor([[3.0, 0.0]])))

    result = audit.finalize(reference_prototypes=(torch.tensor([1.0, 0.0]),))

    scale = result["scales"][0]
    assert scale["valid"] is True
    assert scale["coverage"] == pytest.approx(1.0)
    assert scale["resultant"] == pytest.approx(1.0)
    assert scale["loo_cosines"] == pytest.approx([1.0, 1.0])
    assert scale["reference_cosine"] == pytest.approx(1.0)


def test_prototype_cancellation_zero_and_missing_batches_fail_closed() -> None:
    cancelled = PrototypeGeometryAccumulator(num_scales=1)
    cancelled.update(_residuals(torch.tensor([[1.0, 0.0]])))
    cancelled.update(_residuals(torch.tensor([[-1.0, 0.0]])))
    record = cancelled.finalize()["scales"][0]
    assert record["valid"] is False
    assert record["invalid_reason"] == "direction_cancellation"

    zeros = PrototypeGeometryAccumulator(num_scales=1)
    zeros.update(_residuals(torch.zeros(2, 2)))
    assert zeros.finalize()["scales"][0]["invalid_reason"] == "no_nonzero_directions"

    missing = PrototypeGeometryAccumulator(num_scales=1)
    with pytest.raises(RuntimeError, match="at least one batch"):
        missing.finalize()


def test_component_geometry_reuses_one_row_space_and_exposes_selective_suppression() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
    losses = {
        "easy_cls": parameter[0],
        "malc": parameter[1],
        "rms": parameter.sum(),
    }
    constraint = ConstraintTerm(
        name="class_3_cls",
        margin=parameter[1] + 1.0,
        tolerance=0.005,
    )

    result = component_gradient_geometry(
        parameter=parameter,
        losses=losses,
        constraints=(constraint,),
        near_boundary=0.005,
        svd_relative_tolerance=1e-4,
    ).record

    assert result["components"]["easy_cls"]["retention"] == pytest.approx(1.0)
    assert result["components"]["malc"]["retention"] == pytest.approx(0.0)
    assert result["components"]["rms"]["retention"] == pytest.approx(2 ** -0.5)
    assert result["components"]["easy_cls"]["rank"] == 1
    assert result["components"]["malc"]["active_constraints"] == ["class_3_cls"]
    assert max(
        result["components"][name]["max_projected_row_dot"]
        for name in ("easy_cls", "malc", "rms")
    ) <= 1e-5


def test_component_geometry_builds_the_cgr_projector_once(monkeypatch) -> None:
    calls = 0
    original = geometry_module.route_coefficient_gradient

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(geometry_module, "route_coefficient_gradient", counted)
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
    component_gradient_geometry(
        parameter=parameter,
        losses={
            "easy_cls": parameter[0],
            "malc": parameter[1],
            "rms": parameter.sum(),
        },
        constraints=(
            ConstraintTerm("class_3_cls", parameter[1] + 1.0, 0.005),
        ),
        near_boundary=0.005,
        svd_relative_tolerance=1e-4,
    )
    assert calls == 1


def test_component_geometry_rank_zero_and_disconnected_component() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
    losses = {
        "easy_cls": parameter[0],
        "malc": parameter[1],
        "rms": parameter.sum(),
    }
    result = component_gradient_geometry(
        parameter=parameter,
        losses=losses,
        constraints=(),
        near_boundary=0.005,
        svd_relative_tolerance=1e-4,
    ).record
    assert all(
        result["components"][name]["retention"] == pytest.approx(1.0)
        for name in losses
    )

    disconnected = dict(losses)
    disconnected["malc"] = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="disconnected"):
        component_gradient_geometry(
            parameter=parameter,
            losses=disconnected,
            constraints=(),
            near_boundary=0.005,
            svd_relative_tolerance=1e-4,
        )


def test_component_geometry_full_rank_fails_closed_without_graph_records() -> None:
    for _ in range(16):
        parameter = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
        losses = {
            "easy_cls": parameter[0],
            "malc": parameter[1],
            "rms": parameter.sum(),
        }
        constraints = (
            ConstraintTerm("class_1_cls", parameter[0] + 1.0, 0.005),
            ConstraintTerm("class_2_cls", parameter[1] + 1.0, 0.005),
        )
        geometry = component_gradient_geometry(
            parameter=parameter,
            losses=losses,
            constraints=constraints,
            near_boundary=0.005,
            svd_relative_tolerance=1e-4,
        )
        assert all(
            geometry.record["components"][name]["rank"] == 2
            and geometry.record["components"][name]["retention"]
            == pytest.approx(0.0)
            for name in losses
        )
        assert all(
            tensor.device.type == "cpu" and not tensor.requires_grad
            for tensor in geometry.gradients.values()
        )

        def assert_json_tree(value) -> None:
            assert not torch.is_tensor(value)
            if isinstance(value, dict):
                for child in value.values():
                    assert_json_tree(child)
            elif isinstance(value, list):
                for child in value:
                    assert_json_tree(child)

        assert_json_tree(geometry.record)


def test_gradient_summary_uses_unique_cross_batch_pairs() -> None:
    records = []
    for index, vector in enumerate(([1.0, 0.0], [1.0, 0.0], [-1.0, 0.0])):
        records.append(
            {
                "components": {
                    name: {
                        "raw_gradient": vector if name == "malc" else [0.0, 1.0],
                        "raw_norm": 1.0,
                        "projected_norm": 1.0,
                        "retention": 1.0,
                    }
                    for name in ("easy_cls", "malc", "rms")
                },
                "pairwise_cosines": {
                    "malc_vs_easy": 0.0,
                    "malc_vs_rms": 0.0,
                    "easy_vs_rms": 1.0,
                },
                "batch_index": index,
            }
        )
    summary = summarize_gradient_geometry(records)
    assert summary["cross_batch_malc_cosines"] == pytest.approx([1.0, -1.0, -1.0])
    assert len(summary["cross_batch_malc_cosines"]) == 3


def _decision_inputs() -> tuple[dict, dict, dict]:
    prototype = {
        "calibration": {
            "batch_count": 16,
            "effective_scale_count": 3,
            "effective_resultant_median": 0.8,
            "effective_loo_q25_median": 0.95,
        },
        "heldout": {"batch_count": 24},
    }
    gradient = {
        "batch_count": 16,
        "cross_batch_malc_median": 0.5,
        "cross_batch_malc_q25": 0.2,
        "malc_vs_easy_median": 0.0,
        "malc_vs_rms_median": 0.0,
        "components": {
            "easy_cls": {"retention_median": 0.9},
            "malc": {"retention_median": 0.9},
        },
    }
    micro = {"steps": 8, "D_theta": 1.0, "D_pattern": 0.1}
    return prototype, gradient, micro


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (("prototype", "effective_resultant_median", 0.1), "prototype_incoherence"),
        (("gradient", "cross_batch_malc_median", 0.0), "cross_batch_malc_conflict"),
        (("gradient", "malc_vs_easy_median", -0.2), "objective_gradient_conflict"),
        (("rho", "retention_median", 0.1), "cgr_selective_suppression"),
        (("micro", "D_theta", 0.1), "carrier_update_sink"),
        (("none", "none", 0.0), "unresolved_by_probe"),
    ],
)
def test_decision_table_has_one_preregistered_first_boundary(mutation, expected) -> None:
    prototype, gradient, micro = _decision_inputs()
    target, key, value = mutation
    if target == "prototype":
        prototype["calibration"][key] = value
    elif target == "gradient":
        gradient[key] = value
    elif target == "rho":
        gradient["components"]["malc"][key] = value
    elif target == "micro":
        micro[key] = value
        micro["D_pattern"] = 0.001
    decision = classify_first_bad_boundary(
        prototype_geometry=prototype,
        gradient_geometry=gradient,
        microtrajectory=micro,
    )
    assert decision["valid"] is True
    assert decision["first_bad_boundary"] == expected


def test_detached_cosine_rejects_zero_vectors() -> None:
    with pytest.raises(RuntimeError, match="undefined"):
        detached_cosine(torch.zeros(2), torch.ones(2))


class _TinyCarrier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor([0.1, -0.1]))

    def forward(self) -> torch.Tensor:
        return self.coefficients.reshape(1, 1, 1, 2)


def test_microtrajectory_uses_matched_initial_state_batches_and_snapshots(monkeypatch) -> None:
    workflow = SIRCMALCGeometryWorkflow.__new__(SIRCMALCGeometryWorkflow)
    workflow.geometry = {"microtrajectory_steps": 8}
    workflow.config = {"carrier": {"epsilon": 0.1}}
    workflow.base = SimpleNamespace(
        _carrier=lambda _arm: _TinyCarrier(),
        _load_batch=lambda paths: SimpleNamespace(image_ids=list(paths)),
    )
    workflow._progress = lambda *_args, **_kwargs: None

    def fake_step(*, arm_id, enable_malc, batch, carrier, **_kwargs):
        before = carrier.coefficients.detach().clone()
        increment = torch.tensor([0.01, 0.0])
        if enable_malc:
            increment = increment + torch.tensor([0.0, 0.005])
        with torch.no_grad():
            carrier.coefficients.add_(increment)
        update = increment.double()
        return {
            "arm_id": arm_id,
            "image_ids": list(batch.image_ids),
            "coefficient_hash_before": str(before.tolist()),
            "actual_update": update.tolist(),
        }, update

    monkeypatch.setattr(workflow, "_micro_arm_step", fake_step)
    batches = [[f"image-{index}"] for index in range(8)]
    result = workflow._microtrajectory(
        warm_coefficients=torch.tensor([0.1, -0.1]),
        prototype_calibration=object(),
        gradient_calibration=object(),
        calibration_batches=batches,
        status={},
    )

    assert result["steps"] == 8
    assert [item["step"] for item in result["pattern_snapshots"]] == [0, 4, 8]
    assert [item["image_ids"] for item in result["records"]] == batches
    assert all(item["A0"]["image_ids"] == item["A1"]["image_ids"] for item in result["records"])
    assert result["D_theta"] > 0
    assert result["D_pattern"] > 0
