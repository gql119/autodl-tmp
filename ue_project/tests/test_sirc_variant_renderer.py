from __future__ import annotations

import torch

from ue_framework.methods.instance_canonical_carrier import (
    apply_canonical_pattern,
    apply_variant_canonical_patterns,
)


def _support(
    height: int,
    width: int,
    boxes: list[tuple[int, int, int, int]],
) -> torch.Tensor:
    supports = torch.zeros((len(boxes), 1, height, width))
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        supports[index, 0, y1:y2, x1:x2] = 1
    return supports


def test_each_image_uses_one_selected_variant_for_all_instances() -> None:
    images = torch.full((2, 3, 12, 12), 0.5)
    patterns = torch.stack(
        (
            torch.full((3, 4, 4), 0.02),
            torch.full((3, 4, 4), -0.03),
        )
    ).requires_grad_()
    boxes = [
        [(1, 1, 5, 5), (6, 6, 11, 11)],
        [(2, 2, 10, 10)],
    ]
    supports = [
        _support(12, 12, boxes[0]),
        _support(12, 12, boxes[1]),
    ]
    poisoned, perturbation, rendered = apply_variant_canonical_patterns(
        images,
        patterns,
        variant_indices=(0, 1),
        boxes_by_image=boxes,
        supports_by_image=supports,
        mode="instance",
        epsilon=16 / 255,
        jnd_floor=1.0,
    )
    assert torch.all(perturbation[0, :, 1:5, 1:5] > 0)
    assert torch.all(perturbation[0, :, 6:11, 6:11] > 0)
    assert torch.all(perturbation[1, :, 2:10, 2:10] < 0)
    for index, support in enumerate(supports):
        union = support.sum(dim=0).bool().expand_as(perturbation[index])
        outside_max = perturbation[index][~union].detach().abs().amax()
        assert float(outside_max) == 0
        assert torch.equal(rendered[index].union_support.bool(), union[:1])
    poisoned.sum().backward()
    assert patterns.grad is not None
    assert torch.isfinite(patterns.grad).all()
    assert bool((patterns.grad.flatten(1).norm(dim=1) > 0).all())


def test_single_variant_path_is_numerically_identical_to_legacy_apply() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    images = torch.rand((2, 3, 10, 10), generator=generator)
    pattern = 0.02 * torch.randn((3, 6, 6), generator=generator)
    boxes = [[(1, 2, 7, 9)], [(2, 1, 9, 8)]]
    supports = [
        _support(10, 10, boxes[0]),
        _support(10, 10, boxes[1]),
    ]
    legacy = apply_canonical_pattern(
        images,
        pattern,
        boxes_by_image=boxes,
        supports_by_image=supports,
        mode="instance",
        epsilon=16 / 255,
    )
    variant = apply_variant_canonical_patterns(
        images,
        pattern.unsqueeze(0),
        variant_indices=(0, 0),
        boxes_by_image=boxes,
        supports_by_image=supports,
        mode="instance",
        epsilon=16 / 255,
    )
    assert torch.equal(legacy[0], variant[0])
    assert torch.equal(legacy[1], variant[1])
    for old, new in zip(legacy[2], variant[2]):
        assert torch.equal(old.spatial_pattern, new.spatial_pattern)
        assert torch.equal(old.union_support, new.union_support)
        assert torch.equal(old.overlap_count, new.overlap_count)


def test_variant_indices_fail_closed() -> None:
    images = torch.zeros((1, 3, 8, 8))
    patterns = torch.zeros((2, 3, 4, 4))
    boxes = [[(1, 1, 7, 7)]]
    supports = [_support(8, 8, boxes[0])]
    try:
        apply_variant_canonical_patterns(
            images,
            patterns,
            variant_indices=(2,),
            boxes_by_image=boxes,
            supports_by_image=supports,
            mode="instance",
            epsilon=16 / 255,
        )
    except ValueError as error:
        assert "out-of-range" in str(error)
    else:
        raise AssertionError("Out-of-range variant index did not fail closed.")


def test_variant_render_and_jnd_are_exactly_deterministic() -> None:
    generator = torch.Generator(device="cpu").manual_seed(29)
    images = torch.rand((2, 3, 12, 12), generator=generator)
    patterns = 0.03 * torch.randn(
        (2, 3, 5, 5),
        generator=generator,
    )
    boxes = [[(1, 1, 8, 10)], [(3, 2, 11, 11)]]
    supports = [
        _support(12, 12, boxes[0]),
        _support(12, 12, boxes[1]),
    ]
    kwargs = {
        "variant_indices": (0, 1),
        "boxes_by_image": boxes,
        "supports_by_image": supports,
        "mode": "instance",
        "epsilon": 16 / 255,
    }
    first = apply_variant_canonical_patterns(images, patterns, **kwargs)
    second = apply_variant_canonical_patterns(images, patterns, **kwargs)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert float(first[1].abs().amax()) <= 16 / 255 + 1e-7
