from datetime import datetime

import pytest
import torch

from oa_lgc.artifacts import create_run_dir, unique_run_id
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier


def _image(size=32):
    return torch.full((3, size, size), 0.5)


def _delta(value=0.04):
    return torch.full((3, 8, 8), value, requires_grad=True)


def test_object_aligned_single_target():
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}], _delta(), CarrierConfig()
    )
    assert result.poisoned.shape == (3, 32, 32)
    assert result.metrics["target_instances"] == result.metrics["applied_instances"] == 1
    assert result.metrics["perturbed_area"] > 0


def test_object_aligned_multiple_targets():
    annotations = [
        {"cls": 14, "bbox": [0.25, 0.5, 0.3, 0.5]},
        {"cls": 14, "bbox": [0.75, 0.5, 0.3, 0.5]},
    ]
    result = apply_object_aligned_carrier(_image(), annotations, _delta(), CarrierConfig())
    assert result.metrics["applied_instances"] == 2
    assert int((result.perturbation.abs().sum(dim=0) > 0).sum()) > 100


def test_object_aligned_non_target_exclusion():
    annotations = [
        {"cls": 14, "bbox": [0.5, 0.5, 0.7, 0.7]},
        {"cls": 7, "bbox": [0.5, 0.5, 0.2, 0.2]},
    ]
    result = apply_object_aligned_carrier(
        _image(), annotations, _delta(), CarrierConfig(non_target_dilation=1)
    )
    assert result.metrics["non_target_overlap_ratio"] > 0
    assert result.metrics["direct_non_target_perturbation_max"] == 0
    assert float((result.perturbation.detach() * result.non_target_mask).abs().max()) == 0


def test_object_aligned_empty_valid_mask():
    annotations = [
        {"cls": 14, "bbox": [0.5, 0.5, 0.2, 0.2]},
        {"cls": 1, "bbox": [0.5, 0.5, 0.4, 0.4]},
    ]
    result = apply_object_aligned_carrier(
        _image(), annotations, _delta(), CarrierConfig(non_target_dilation=0, min_valid_fraction=0.01)
    )
    assert result.metrics["applied_instances"] == 0
    assert result.instance_metrics[0]["invalid_reason"] == "valid_fraction_below_threshold"
    assert float(result.perturbation.abs().max()) == 0


def test_object_aligned_small_box():
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.01, 0.01]}], _delta(), CarrierConfig()
    )
    assert result.metrics["applied_instances"] == 1
    assert result.instance_metrics[0]["box_width"] >= 1
    assert result.instance_metrics[0]["box_height"] >= 1


def test_object_aligned_boundary_clip():
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.0, 0.0, 0.4, 0.6]}], _delta(), CarrierConfig()
    )
    row = result.instance_metrics[0]
    assert row["x1"] == row["y1"] == 0
    assert row["x2"] <= 32 and row["y2"] <= 32


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_object_aligned_interpolation_modes(mode):
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.6, 0.3]}], _delta(), CarrierConfig(interpolation=mode)
    )
    assert result.metrics["finite"]
    assert result.metrics["interpolation"] == mode


def test_object_aligned_soft_mask():
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}], _delta(),
        CarrierConfig(soft_mask=True, soft_edge_pixels=3.0),
    )
    values = result.valid_support[result.valid_support > 0]
    assert torch.any(values < 1.0)
    assert torch.any(values == 1.0)


def test_object_aligned_gradient_to_delta_only():
    model = torch.nn.Conv2d(3, 1, 1, bias=False)
    model.requires_grad_(False)
    delta = _delta()
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}], delta, CarrierConfig()
    )
    model(result.poisoned.unsqueeze(0)).square().mean().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_object_aligned_area_metrics():
    result = apply_object_aligned_carrier(
        _image(), [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.25]}], _delta(), CarrierConfig()
    )
    assert 0 < result.metrics["perturbed_area"] <= result.metrics["actual_support_area"] < 1
    assert result.metrics["max_abs_perturbation"] <= 16 / 255 + 1e-8


def test_object_aligned_no_history_overwrite(tmp_path):
    run_id = unique_run_id("L1", 0, datetime(2026, 7, 13, 12, 0, 0))
    created = create_run_dir(tmp_path, run_id)
    assert created.exists()
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, run_id)


def test_unique_run_id():
    first = unique_run_id("L1", 0, datetime(2026, 7, 13, 12, 0, 0, 1))
    second = unique_run_id("L1", 0, datetime(2026, 7, 13, 12, 0, 0, 2))
    assert first != second
