from __future__ import annotations

import copy
from pathlib import Path

import torch

from mini_csdem.dataset import load_config
from mini_csdem.non_target_preservation import align_non_target_units, compute_non_target_preservation


ROOT = Path(__file__).resolve().parents[1]
TARGET = 14
WEIGHTS = {"logits": 1.0, "box": 0.5, "dfl": 0.25, "assignment": 0.5}


def make_state(delta: torch.Tensor | None = None, labels=(6, TARGET, 3), gt_indices=(0, 1, 2)):
    units, classes, reg_max = 3, 20, 16
    base_scores = torch.linspace(-1.0, 1.0, units * classes).reshape(1, units, classes)
    base_boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0], [7.0, 7.0, 9.0, 9.0]]])
    base_dfl = torch.linspace(-0.5, 0.5, units * 4 * reg_max).reshape(1, units, 4 * reg_max)
    if delta is None:
        scores, boxes, dfl = base_scores, base_boxes, base_dfl
    else:
        score_pattern = torch.linspace(-1.0, 1.0, classes).view(1, 1, classes)
        dfl_pattern = torch.linspace(-1.0, 1.0, 4 * reg_max).view(1, 1, 4 * reg_max)
        scores = base_scores + delta * score_pattern
        boxes = base_boxes + delta * torch.tensor([1.0, 0.5, 1.5, 1.0]).view(1, 1, 4)
        dfl = base_dfl + delta * dfl_pattern
    label_tensor = torch.tensor([labels])
    gt_tensor = torch.tensor([gt_indices])
    target_boxes = base_boxes * 8.0
    target_scores = torch.zeros(1, units, classes)
    for index, label in enumerate(labels):
        target_scores[0, index, label] = 0.8
    return {
        "pred_scores": scores,
        "pred_bboxes": boxes,
        "pred_distri": dfl,
        "anchor_points": torch.zeros(units, 2),
        "stride_tensor": torch.full((units, 1), 8.0),
        "imgsz": torch.tensor([64.0, 64.0]),
        "target_labels": label_tensor,
        "target_gt_idx": gt_tensor,
        "target_bboxes": target_boxes,
        "target_scores": target_scores,
        "fg_mask": torch.ones(1, units, dtype=torch.bool),
    }


def test_clean_teacher_has_no_gradient_and_poison_has_gradient():
    delta = torch.tensor(0.1, requires_grad=True)
    clean = make_state()
    poison = make_state(delta)
    result = compute_non_target_preservation(clean, poison, TARGET, WEIGHTS)
    result.total_loss.backward()
    assert delta.grad is not None and delta.grad.abs() > 0
    assert all(not tensor.requires_grad for tensor in [clean["pred_scores"], clean["pred_bboxes"], clean["pred_distri"]])


def test_each_preservation_component_has_delta_gradient():
    for name in ["logits_loss", "box_loss", "dfl_loss", "assignment_loss"]:
        delta = torch.tensor(0.1, requires_grad=True)
        result = compute_non_target_preservation(make_state(), make_state(delta), TARGET, WEIGHTS)
        gradient = torch.autograd.grad(getattr(result, name), delta)[0]
        assert torch.isfinite(gradient)
        assert gradient.abs() > 0, name


def test_target_class_units_are_excluded():
    result = compute_non_target_preservation(make_state(), make_state(torch.tensor(0.1)), TARGET, WEIGHTS)
    assert TARGET not in result.alignment.class_alignment_counts
    assert result.alignment.matched_count == 2


def test_different_gt_indices_are_not_matched():
    clean = make_state()
    poison = make_state(torch.tensor(0.1), gt_indices=(9, 1, 2))
    alignment = align_non_target_units(clean, poison, TARGET)
    assert alignment.matched_count == 1
    assert alignment.gt_index_mismatch_count == 1


def test_flattened_anchor_and_fpn_level_are_recorded_together():
    alignment = align_non_target_units(make_state(), make_state(torch.tensor(0.1)), TARGET)
    assert torch.equal(alignment.anchor_indices, torch.tensor([0, 2]))
    assert alignment.level_indices.shape == alignment.anchor_indices.shape
    assert all(anchor >= 0 for anchor in alignment.anchor_indices.tolist())


def test_no_non_target_instances_returns_finite_zero():
    clean = make_state(labels=(TARGET, TARGET, TARGET))
    poison = make_state(torch.tensor(0.1), labels=(TARGET, TARGET, TARGET))
    result = compute_non_target_preservation(clean, poison, TARGET, WEIGHTS)
    assert result.total_loss.item() == 0.0
    assert torch.isfinite(result.total_loss)
    assert result.alignment.matched_count == 0


def test_no_successful_alignment_safely_skips():
    clean = make_state()
    poison = make_state(torch.tensor(0.1), gt_indices=(9, 1, 8))
    result = compute_non_target_preservation(clean, poison, TARGET, WEIGHTS)
    assert result.alignment.matched_count == 0
    assert result.total_loss.item() == 0.0


def test_teacher_tensors_are_detached_even_if_supplied_with_grad():
    clean_delta = torch.tensor(0.1, requires_grad=True)
    poison_delta = torch.tensor(0.2, requires_grad=True)
    result = compute_non_target_preservation(make_state(clean_delta), make_state(poison_delta), TARGET, WEIGHTS)
    result.total_loss.backward()
    assert clean_delta.grad is None
    assert poison_delta.grad is not None


def test_stage1_configuration_keeps_preservation_disabled():
    cfg = load_config(ROOT / "configs/mini_csdem/stage1.yaml")
    assert not cfg["features"]["enable_non_target_preservation"]
    assert "preservation" not in cfg or not cfg["preservation"].get("enabled", False)


def test_zero_preservation_weights_leave_target_objective_unchanged():
    delta = torch.tensor(0.1, requires_grad=True)
    result = compute_non_target_preservation(
        make_state(), make_state(delta), TARGET, {key: 0.0 for key in WEIGHTS}
    )
    target_loss = delta.square()
    assert torch.allclose(target_loss + result.total_loss, target_loss)


def test_alignment_coverage_uses_clean_non_target_denominator():
    alignment = align_non_target_units(make_state(), make_state(torch.tensor(0.1)), TARGET)
    assert alignment.clean_non_target_count == 2
    assert alignment.poison_non_target_count == 2
    assert alignment.coverage == 1.0


def test_disabled_component_weight_does_not_enter_total():
    delta = torch.tensor(0.1, requires_grad=True)
    logits_only = {"logits": 1.0, "box": 0.0, "dfl": 0.0, "assignment": 0.0}
    result = compute_non_target_preservation(make_state(), make_state(delta), TARGET, logits_only)
    assert torch.allclose(result.total_loss, result.logits_loss)
