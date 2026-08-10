from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ue_framework.methods.detector_lfc import (
    DetectorLFCExtractor,
    DetectorLFCFeatures,
    DetectorLFCPrototypeBank,
)


def _branch(in_channels: int, hidden: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.Conv2d(hidden, out_channels, 1),
    )


class _TinyDetect(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = 20
        self.cv3 = nn.ModuleList([_branch(8, 6, 20) for _ in range(3)])
        self.cv2 = nn.ModuleList([_branch(8, 5, 64) for _ in range(3)])

    def forward(self, features):
        return [
            torch.cat((box(feature), cls(feature)), dim=1)
            for feature, box, cls in zip(features, self.cv2, self.cv3)
        ]


class _TinyYOLO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        modules = [nn.Identity() for _ in range(22)]
        modules.append(_TinyDetect())
        self.model = nn.ModuleList(modules)
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, images):
        feature = self.stem(images)
        scales = [feature, F.avg_pool2d(feature, 2), F.avg_pool2d(feature, 4)]
        return self.model[22](scales)


def _features(value: float = 1.0) -> DetectorLFCFeatures:
    return DetectorLFCFeatures(
        classification=tuple(
            F.normalize(torch.full((2, channels), value), dim=1)
            for channels in (3, 4, 5)
        )
    )


def test_detector_lfc_is_detector_native_and_backpropagates_to_delta() -> None:
    model = _TinyYOLO()
    delta = (torch.rand((3, 3, 32, 32)) - 0.5).requires_grad_(True)
    with DetectorLFCExtractor(model, eps=0.5) as extractor:
        features = extractor.extract(delta)
        assert [tuple(item.shape) for item in features.classification] == [
            (3, 6),
            (3, 6),
            (3, 6),
        ]
        bank = DetectorLFCPrototypeBank()
        bank.fit([features], split="calibration")
        changed = extractor.extract(delta * 0.8)
        result = bank.compute(changed)
        result.loss.backward()

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_prototypes_are_calibration_only_and_frozen_after_fit() -> None:
    bank = DetectorLFCPrototypeBank()
    with pytest.raises(ValueError, match="calibration split"):
        bank.fit([_features()], split="heldout")
    bank.fit([_features()], split="calibration")
    assert bank.calibration_count == 2
    with pytest.raises(RuntimeError, match="already frozen"):
        bank.fit([_features()], split="calibration")


def test_equal_scale_weighting_and_metrics() -> None:
    bank = DetectorLFCPrototypeBank()
    bank.fit([_features()], split="calibration")
    features = DetectorLFCFeatures(
        classification=(
            _features().classification[0],
            -_features().classification[1],
            _features().classification[2],
        )
    )
    result = bank.compute(features)
    assert result.loss.item() == pytest.approx(2.0 / 3.0, abs=1.0e-6)
    assert [item.item() for item in result.per_scale_loss] == pytest.approx(
        [0.0, 2.0, 0.0], abs=1.0e-6
    )
    metrics = result.detached_metrics()
    assert metrics["dlfc_p3_coverage"] == 1.0
    assert metrics["dlfc_p4_cosine_median"] == pytest.approx(-1.0)


def test_zero_and_constant_deltas_fail_closed() -> None:
    with DetectorLFCExtractor(_TinyYOLO(), eps=0.5) as extractor:
        with pytest.raises(ValueError, match="Zero"):
            extractor.extract(torch.zeros((2, 3, 32, 32)))
        with pytest.raises(ValueError, match="Constant"):
            extractor.extract(torch.ones((2, 3, 32, 32)) * 0.1)


def test_prototype_state_roundtrip() -> None:
    source = DetectorLFCPrototypeBank()
    source.fit([_features()], split="calibration")
    restored = DetectorLFCPrototypeBank()
    restored.load_state_dict(source.state_dict())
    assert restored.calibration_count == 2
    assert restored.compute(_features()).loss.item() == pytest.approx(0.0, abs=1e-6)
