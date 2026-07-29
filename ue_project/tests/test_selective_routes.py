from __future__ import annotations

import pytest
import torch

from ue_framework.methods.shadow_tal import (
    DifferentiableShadowTAL,
    build_non_target_constraints,
    compute_target_route,
)


def _boxes() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [5.0, 5.0, 9.0, 9.0], [2.0, 2.0, 6.0, 6.0]]]
    )


def test_easy_and_evasion_routes_are_mutually_exclusive() -> None:
    logits = torch.zeros((1, 3, 20), requires_grad=True)
    boxes = _boxes().clone().requires_grad_()
    gate = torch.tensor([[True, False, False]])

    easy = compute_target_route(
        route="easy_cls",
        adv_class_logits=logits,
        adv_boxes=boxes,
        clean_boxes=_boxes(),
        target_gate=gate,
        target_class_id=14,
        num_classes=20,
    )
    easy.loss.backward(retain_graph=True)
    assert easy.status == "active"
    assert float(easy.evasion.detach()) == 0.0
    assert logits.grad[0, 0, 14] < 0
    logits.grad.zero_()

    shadow = DifferentiableShadowTAL(target_class_id=14, topk=1)
    gt_labels = torch.tensor([[[14.0]]])
    gt_boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]]])
    mask_gt = torch.ones((1, 1, 1))
    evasion = compute_target_route(
        route="tal_evasion",
        adv_class_logits=logits,
        adv_boxes=boxes,
        clean_boxes=_boxes(),
        target_gate=gate,
        target_class_id=14,
        num_classes=20,
        shadow_tal=shadow,
        gt_labels=gt_labels,
        gt_bboxes=gt_boxes,
        mask_gt=mask_gt,
    )
    evasion.loss.backward()
    assert float(evasion.easy_classification.detach()) == 0.0
    assert logits.grad[0, 0, 14] > 0
    assert torch.isfinite(boxes.grad).all()


def test_route_rejects_objectness_augmented_logits() -> None:
    with pytest.raises(ValueError, match="class-only"):
        compute_target_route(
            route="easy_cls",
            adv_class_logits=torch.zeros((1, 2, 21)),
            adv_boxes=torch.zeros((1, 2, 4)),
            clean_boxes=torch.zeros((1, 2, 4)),
            target_gate=torch.ones((1, 2), dtype=torch.bool),
            target_class_id=14,
            num_classes=20,
        )


def test_non_target_constraints_are_per_class_and_one_sided() -> None:
    clean_logits = torch.zeros((1, 3, 20))
    clean_logits[0, 0, 7] = 1.0
    clean_logits[0, 1, 3] = 0.5
    adv_logits = clean_logits.clone()
    adv_logits[0, 0, 7] = 2.0
    adv_logits[0, 1, 3] = -1.0
    adv_logits.requires_grad_()

    clean_boxes = _boxes()
    adv_boxes = clean_boxes.clone()
    adv_boxes[0, 1] += torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv_boxes.requires_grad_()
    labels = torch.tensor([[7, 3, 14]])
    foreground = torch.tensor([[True, True, True]])

    result = build_non_target_constraints(
        clean_class_logits=clean_logits,
        adv_class_logits=adv_logits,
        clean_boxes=clean_boxes,
        adv_boxes=adv_boxes,
        assigned_gt_boxes=clean_boxes,
        assigned_labels=labels,
        real_foreground=foreground,
        target_class_id=14,
        num_classes=20,
    )
    assert result.status == "active"
    assert [item.class_id for item in result.constraints] == [3, 7]
    by_class = {item.class_id: item for item in result.constraints}
    assert float(by_class[7].cls_violation.detach()) == 0.0
    assert by_class[3].cls_violation > 0
    assert by_class[3].box_violation > 0

    total = sum(
        item.cls_violation + item.box_violation
        for item in result.constraints
    )
    total.backward()
    assert torch.isfinite(adv_logits.grad).all()
    assert torch.isfinite(adv_boxes.grad).all()


def test_person_only_constraints_are_not_applicable() -> None:
    result = build_non_target_constraints(
        clean_class_logits=torch.zeros((1, 2, 20)),
        adv_class_logits=torch.zeros((1, 2, 20)),
        clean_boxes=torch.zeros((1, 2, 4)),
        adv_boxes=torch.zeros((1, 2, 4)),
        assigned_gt_boxes=torch.zeros((1, 2, 4)),
        assigned_labels=torch.tensor([[14, 14]]),
        real_foreground=torch.ones((1, 2), dtype=torch.bool),
        target_class_id=14,
        num_classes=20,
    )
    assert result.status == "not_applicable"
    assert result.constraints == ()
