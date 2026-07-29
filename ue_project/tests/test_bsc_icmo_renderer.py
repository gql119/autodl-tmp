from __future__ import annotations

import torch
import torch.nn.functional as F

from ue_framework.methods.instance_canonical_carrier import (
    affine_canonical_pattern,
    apply_canonical_pattern,
    render_canonical_pattern,
    warp_canonical_patch,
)


def _smooth_pattern(size: int = 32) -> torch.Tensor:
    axis = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        (
            0.04 * torch.sin(2.0 * torch.pi * xx),
            0.03 * torch.cos(2.0 * torch.pi * yy),
            0.02 * (xx + yy),
        )
    )


def _rect_supports(
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> torch.Tensor:
    supports = torch.zeros((len(boxes), 1, height, width))
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        supports[index, :, y1:y2, x1:x2] = 1
    return supports


def _ncc(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.flatten() - first.mean()
    second = second.flatten() - second.mean()
    return float(
        (first @ second / first.norm().clamp_min(1e-12) / second.norm().clamp_min(1e-12))
    )


def test_instance_warp_reconstructs_object_relative_pattern() -> None:
    pattern = _smooth_pattern()
    boxes = [(4, 7, 36, 39), (42, 5, 58, 29), (8, 43, 56, 63)]
    reconstructions = []
    for box in boxes:
        patch = warp_canonical_patch(pattern, box)
        reconstruction = F.interpolate(
            patch.unsqueeze(0),
            size=pattern.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        reconstructions.append(reconstruction)
    similarities = [
        _ncc(first, second)
        for index, first in enumerate(reconstructions)
        for second in reconstructions[index + 1 :]
    ]
    assert min(similarities) >= 0.98


def test_global_and_instance_renderers_are_not_the_same_path() -> None:
    pattern = _smooth_pattern()
    boxes = [(2, 4, 26, 44), (38, 18, 62, 58)]
    supports = _rect_supports(boxes, 64, 64)
    instance = render_canonical_pattern(
        pattern,
        image_size=(64, 64),
        boxes=boxes,
        instance_supports=supports,
        mode="instance",
    )
    global_control = render_canonical_pattern(
        pattern,
        image_size=(64, 64),
        boxes=boxes,
        instance_supports=supports,
        mode="global",
    )
    assert not torch.allclose(
        instance.spatial_pattern,
        global_control.spatial_pattern,
    )

    instance_crops = []
    global_crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        for source, output in (
            (instance.spatial_pattern, instance_crops),
            (global_control.spatial_pattern, global_crops),
        ):
            output.append(
                F.interpolate(
                    source[:, y1:y2, x1:x2].unsqueeze(0),
                    size=pattern.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
    assert _ncc(instance_crops[0], instance_crops[1]) >= 0.98
    assert _ncc(global_crops[0], global_crops[1]) < 0.90


def test_overlap_is_mean_and_support_outside_is_zero() -> None:
    pattern = _smooth_pattern()
    boxes = [(4, 4, 28, 28), (16, 16, 40, 40)]
    supports = _rect_supports(boxes, 48, 48)
    rendered = render_canonical_pattern(
        pattern,
        image_size=(48, 48),
        boxes=boxes,
        instance_supports=supports,
        mode="instance",
    )
    patches = [
        F.pad(
            warp_canonical_patch(pattern, box),
            (box[0], 48 - box[2], box[1], 48 - box[3]),
        )
        for box in boxes
    ]
    expected_overlap = (patches[0][:, 16:28, 16:28] + patches[1][:, 16:28, 16:28]) / 2
    assert torch.allclose(
        rendered.spatial_pattern[:, 16:28, 16:28],
        expected_overlap,
        atol=1e-7,
    )
    assert rendered.overlap_count.max() == 2
    assert torch.count_nonzero(
        rendered.spatial_pattern * (1 - rendered.union_support)
    ) == 0

    invalid_support = supports.clone()
    invalid_support[0, :, 0, 0] = 1
    try:
        render_canonical_pattern(
            pattern,
            image_size=(48, 48),
            boxes=boxes,
            instance_supports=invalid_support,
            mode="instance",
        )
    except ValueError as error:
        assert "inside its box" in str(error)
    else:
        raise AssertionError("Out-of-box support must fail closed.")


def test_apply_is_bounded_supported_and_has_finite_gradient() -> None:
    pattern = _smooth_pattern().requires_grad_()
    boxes = [[(8, 8, 40, 52)]]
    supports = [_rect_supports(boxes[0], 64, 64)]
    images = torch.full((1, 3, 64, 64), 0.5)
    _, perturbation, rendered = apply_canonical_pattern(
        images,
        pattern,
        boxes_by_image=boxes,
        supports_by_image=supports,
        mode="instance",
        epsilon=16.0 / 255.0,
    )
    assert perturbation.abs().amax() <= 16.0 / 255.0
    assert torch.count_nonzero(
        perturbation * (1 - rendered[0].union_support)
    ) == 0
    perturbation.square().mean().backward()
    assert pattern.grad is not None
    assert torch.isfinite(pattern.grad).all()
    assert float(pattern.grad.norm()) > 0


def test_fixed_affine_audits_change_pattern_and_keep_shape() -> None:
    pattern = _smooth_pattern()
    transforms = (
        {"scale": 0.9},
        {"scale": 1.1},
        {"translate_x": -0.05},
        {"translate_x": 0.05},
        {"translate_y": -0.05},
        {"translate_y": 0.05},
    )
    for transform in transforms:
        audited = affine_canonical_pattern(pattern, **transform)
        assert audited.shape == pattern.shape
        assert torch.isfinite(audited).all()
        assert not torch.equal(audited, pattern)
