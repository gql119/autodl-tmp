import torch

from ue_framework.core.localized_support import LocalizedSupportBuilder


def _targets(cls, boxes):
    return {
        "cls": torch.tensor(cls, dtype=torch.long),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "batch_idx": torch.zeros(len(cls), dtype=torch.long),
        "batch_size": 1,
    }


def test_support_not_full_image():
    images = torch.zeros((1, 3, 64, 64))
    out = LocalizedSupportBuilder(protected_class_id=14).build(
        images, _targets([14], [[0.5, 0.5, 0.25, 0.25]])
    )
    assert 0.0 < out.statistics["support_area_ratio"] < 1.0


def test_delta_is_zero_outside_support():
    images = torch.zeros((1, 3, 32, 32))
    out = LocalizedSupportBuilder(protected_class_id=14).build(
        images, _targets([14], [[0.5, 0.5, 0.25, 0.25]])
    )
    delta = torch.ones((1, 3, 32, 32))
    masked = LocalizedSupportBuilder.apply_support(delta, out.valid_support_mask)
    outside = masked * (1.0 - out.valid_support_mask)
    assert torch.max(torch.abs(outside)).item() == 0.0


def test_authorized_core_is_excluded():
    images = torch.zeros((1, 3, 64, 64))
    out = LocalizedSupportBuilder(protected_class_id=14, ambiguous_iou_threshold=0.95).build(
        images,
        _targets(
            [14, 1],
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.25, 0.25],
            ],
        ),
    )
    overlap = out.valid_support_mask * out.authorized_core_mask
    assert overlap.sum().item() == 0.0
    assert out.authorized_core_mask.sum().item() > 0.0


def test_ambiguous_region_is_excluded():
    images = torch.zeros((1, 3, 64, 64))
    out = LocalizedSupportBuilder(protected_class_id=14, ambiguous_iou_threshold=0.1).build(
        images,
        _targets(
            [14, 1],
            [
                [0.5, 0.5, 0.45, 0.45],
                [0.52, 0.52, 0.45, 0.45],
            ],
        ),
    )
    assert out.ambiguous_mask.sum().item() > 0.0
    assert (out.valid_support_mask * out.ambiguous_mask).sum().item() == 0.0


def test_support_resize_preserves_binary_semantics():
    mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    resized = LocalizedSupportBuilder.resize_mask(mask, (5, 5))
    assert set(torch.unique(resized).tolist()) <= {0.0, 1.0}


def test_empty_protected_batch_returns_empty_support():
    images = torch.zeros((1, 3, 64, 64))
    out = LocalizedSupportBuilder(protected_class_id=14).build(
        images, _targets([1], [[0.5, 0.5, 0.25, 0.25]])
    )
    assert out.valid_support_mask.sum().item() == 0.0
