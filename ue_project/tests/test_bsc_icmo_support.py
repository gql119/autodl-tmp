from __future__ import annotations

import numpy as np

from ue_framework.support import (
    build_forced_pseudo_instance_masks,
    build_support_mask,
)


def test_forced_pseudo_instance_union_matches_legacy_fallback() -> None:
    annotations = [
        {"cls": 14, "bbox": (0.35, 0.45, 0.30, 0.50)},
        {
            "cls": 14,
            "bbox": (0.62, 0.55, 0.24, 0.32),
            "polygon": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)],
        },
        {"cls": 11, "bbox": (0.5, 0.5, 0.8, 0.8)},
    ]
    masks = build_forced_pseudo_instance_masks((80, 96), annotations, 14)
    assert len(masks) == 2
    assert all(mask.shape == (80, 96) for mask in masks)
    assert all(mask.dtype == np.float32 for mask in masks)

    without_polygons = [
        {key: value for key, value in ann.items() if key != "polygon"}
        for ann in annotations
    ]
    legacy = build_support_mask(
        (80, 96),
        without_polygons,
        target_class_id=14,
        support_type="mask",
    )
    union = np.maximum.reduce(masks)
    assert np.array_equal(union, legacy)

    polygon_aware = build_support_mask(
        (80, 96),
        annotations,
        target_class_id=14,
        support_type="mask",
    )
    assert not np.array_equal(union, polygon_aware)
