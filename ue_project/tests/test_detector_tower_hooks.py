from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ue_framework.methods.detector_tower_hooks import YOLODetectTowerCapture


def _branch(in_channels: int, hidden: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.Conv2d(hidden, out_channels, 1),
    )


class TinyDetect(nn.Module):
    def __init__(self, *, num_classes: int = 20) -> None:
        super().__init__()
        self.nc = num_classes
        self.cv3 = nn.ModuleList(
            [_branch(8, 6, num_classes) for _ in range(3)]
        )
        self.cv2 = nn.ModuleList([_branch(8, 5, 64) for _ in range(3)])

    def forward(self, features: list[torch.Tensor]):
        return [
            torch.cat((box(feature), cls(feature)), dim=1)
            for feature, box, cls in zip(features, self.cv2, self.cv3)
        ]


class TinyYOLO(nn.Module):
    def __init__(self, *, num_classes: int = 20) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Identity() for _ in range(22)]
        modules.append(TinyDetect(num_classes=num_classes))
        self.model = nn.ModuleList(modules)
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, images: torch.Tensor):
        feature = self.stem(images)
        scales = [
            feature,
            F.avg_pool2d(feature, 2),
            F.avg_pool2d(feature, 4),
        ]
        return self.model[22](scales)


def test_capture_records_three_classification_and_box_scales() -> None:
    model = TinyYOLO()
    capture = YOLODetectTowerCapture(model)
    images = torch.rand((2, 3, 32, 32), requires_grad=True)

    with capture.record("clean"):
        model(images)
    features = capture.take("clean")

    assert len(features.classification) == 3
    assert len(features.box) == 3
    assert [tuple(item.shape) for item in features.classification] == [
        (2, 6, 32, 32),
        (2, 6, 16, 16),
        (2, 6, 8, 8),
    ]
    assert [tuple(item.shape) for item in features.box] == [
        (2, 5, 32, 32),
        (2, 5, 16, 16),
        (2, 5, 8, 8),
    ]
    sum(item.mean() for item in features.classification).backward()
    assert images.grad is not None
    capture.close()

    hook_count = sum(
        len(branch[-1]._forward_pre_hooks)
        for tower in (model.model[22].cv3, model.model[22].cv2)
        for branch in tower
    )
    assert hook_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        with capture.record("adv"):
            model(images)


def test_capture_fails_closed_on_wrong_class_head() -> None:
    with pytest.raises(ValueError, match="class count"):
        YOLODetectTowerCapture(TinyYOLO(num_classes=19))


def test_capture_rejects_incomplete_forward() -> None:
    model = TinyYOLO()
    with YOLODetectTowerCapture(model) as capture:
        with capture.record("empty"):
            pass
        with pytest.raises(RuntimeError, match="Incomplete"):
            capture.take("empty")
