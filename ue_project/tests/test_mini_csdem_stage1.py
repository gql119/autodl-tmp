from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from mini_csdem.gt_conditioned_partition import TargetPartition
from mini_csdem.object_aligned_perturbation import ObjectAlignedPerturbation
from mini_csdem.selective_detection_loss import target_only_detection_loss
from runners.run_mini_csdem import state_sha256


ROOT = Path(__file__).resolve().parents[1]
TARGET = 14


def batch(classes, boxes, batch_indices=None):
    if batch_indices is None:
        batch_indices = [0] * len(classes)
    return {
        "cls": torch.tensor(classes, dtype=torch.float32).view(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "batch_idx": torch.tensor(batch_indices, dtype=torch.float32),
        "batch_size": max(batch_indices, default=0) + 1,
    }


def test_no_person_image_is_unchanged():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    images = torch.rand(1, 3, 32, 32)
    result = perturbation(images, batch([3], [[0.5, 0.5, 0.5, 0.5]]), TARGET)
    assert torch.equal(result.images, images)


def test_only_person_box_changes_and_epsilon_is_respected():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    images = torch.full((1, 3, 32, 32), 0.5)
    result = perturbation(images, batch([TARGET], [[0.5, 0.5, 0.5, 0.5]]), TARGET)
    difference = result.images - images
    assert difference[:, :, 8:24, 8:24].abs().sum() > 0
    outside = difference.clone()
    outside[:, :, 8:24, 8:24] = 0
    assert outside.abs().max() == 0
    assert difference.abs().max() <= 8 / 255 + 1.0e-7


def test_multiple_person_boxes_share_the_same_object_pattern():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    images = torch.full((1, 3, 32, 32), 0.5)
    boxes = [[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]
    result = perturbation(images, batch([TARGET, TARGET], boxes), TARGET, exclude_non_target_overlap=False)
    difference = result.images - images
    assert torch.allclose(difference[0, :, 4:12, 4:12], difference[0, :, 20:28, 20:28], atol=1.0e-7)
    assert result.target_instances == 2


def test_non_target_overlap_exclusion_reduces_effective_area():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    images = torch.full((1, 3, 32, 32), 0.5)
    rows = batch([TARGET, 6], [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25]])
    result = perturbation(images, rows, TARGET, exclude_non_target_overlap=True)
    assert result.effective_target_pixels < result.raw_target_pixels


class FakeBBoxLoss:
    def __call__(
        self,
        pred_distri,
        pred_bboxes,
        _anchor_points,
        target_bboxes,
        _target_scores,
        _score_sum,
        selected,
        _imgsz,
        _stride_tensor,
    ):
        return F.smooth_l1_loss(pred_bboxes[selected], target_bboxes[selected]), pred_distri[selected].square().mean()


class FakeCriterion:
    bbox_loss = FakeBBoxLoss()


class FakeAdapter:
    criterion = FakeCriterion()

    @staticmethod
    def _hyp_gain(_name):
        return 1.0


def make_loss_state(source: torch.Tensor):
    scores = torch.stack([source, source * 2, source * 3]).view(1, 3, 1).expand(-1, -1, 20)
    return {
        "pred_scores": scores,
        "pred_distri": source.expand(1, 3, 4),
        "pred_bboxes": source.expand(1, 3, 4),
        "anchor_points": torch.zeros(3, 2),
        "stride_tensor": torch.ones(3, 1),
        "target_scores": torch.zeros(1, 3, 20),
        "imgsz": torch.tensor([32.0, 32.0]),
    }


def make_partition():
    return TargetPartition(
        unit_mask=torch.tensor([[True, False, False]]),
        target_labels=torch.tensor([[TARGET, 6, -1]]),
        target_bboxes=torch.ones(1, 3, 4),
        target_scores=torch.tensor([[1.0, 1.0, 0.0]]),
        target_gt_idx=torch.tensor([[0, 1, -1]]),
        tal_positive_count=1,
        fallback_positive_count=0,
    )


def test_target_units_are_person_and_non_target_unit_is_excluded():
    partition = make_partition()
    assert torch.all(partition.target_labels[partition.unit_mask] == TARGET)
    source = torch.tensor(0.25, requires_grad=True)
    base = target_only_detection_loss(FakeAdapter(), make_loss_state(source), partition)["total_loss"]
    changed_state = make_loss_state(source)
    changed_state["pred_scores"] = changed_state["pred_scores"].clone()
    changed_state["pred_scores"][0, 1, 6] = 1000.0
    changed = target_only_detection_loss(FakeAdapter(), changed_state, partition)["total_loss"]
    assert torch.allclose(base, changed)


def test_target_loss_gradient_reaches_object_delta():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    source = perturbation.delta_object.mean()
    loss = target_only_detection_loss(FakeAdapter(), make_loss_state(source), make_partition())["total_loss"]
    loss.backward()
    assert perturbation.delta_object.grad is not None
    assert torch.isfinite(perturbation.delta_object.grad).all()
    assert perturbation.delta_object.grad.abs().sum() > 0


def test_empty_target_loss_can_be_safely_skipped():
    source = torch.tensor(0.25)
    partition = make_partition()
    partition.unit_mask[:] = False
    loss = target_only_detection_loss(FakeAdapter(), make_loss_state(source), partition)["total_loss"]
    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_frozen_surrogate_is_not_modified_by_delta_update():
    surrogate = torch.nn.Linear(2, 1)
    before = {name: value.detach().clone() for name, value in surrogate.state_dict().items()}
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    delta = torch.tensor([1.0, -1.0], requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=0.1)
    surrogate(delta).square().sum().backward()
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in surrogate.state_dict().items())


def test_projection_clamps_object_delta():
    perturbation = ObjectAlignedPerturbation(8, 8 / 255)
    with torch.no_grad():
        perturbation.delta_object.fill_(1.0)
    perturbation.project_()
    assert perturbation.delta_object.abs().max() <= 8 / 255


def test_victim_is_freshly_initialized_not_surrogate_reuse():
    config = ROOT / "configs/voc_yolov8n_20cls.yaml"
    torch.manual_seed(0)
    surrogate = YOLO(str(config)).model
    original_hash = state_sha256(surrogate)
    with torch.no_grad():
        next(surrogate.parameters()).add_(1.0)
    trained_surrogate_hash = state_sha256(surrogate)
    torch.manual_seed(0)
    victim = YOLO(str(config)).model
    assert victim is not surrogate
    assert state_sha256(victim) == original_hash
    assert state_sha256(victim) != trained_surrogate_hash
