import os
from typing import List

import numpy as np

from ..support import build_support_mask


def get_target_instance_support(
    image_path: str,
    image_np: np.ndarray,
    annotations: List[dict],
    target_class_id: int,
    use_prebaked_instance_mask: bool,
    instance_mask_dir: str,
    strict_instance_mask: bool = False,
) -> np.ndarray:
    """
    Build target support for detection baselines (TAP/LSP) without any detection-aware modules.

    Priority:
    1) prebaked instance mask (if enabled and exists)
    2) target bbox union fallback (unless strict_instance_mask=True)
    """
    h, w = image_np.shape[:2]
    zero = np.zeros((h, w), dtype=np.float32)

    mask_path = None
    if use_prebaked_instance_mask and instance_mask_dir and image_path:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        cand = os.path.join(instance_mask_dir, stem + ".png")
        if os.path.isfile(cand):
            mask_path = cand

    if mask_path is not None:
        return build_support_mask(
            image_shape=image_np.shape,
            annotations=annotations,
            target_class_id=int(target_class_id),
            support_type="mask",
            ring_width=4,
            mask_path=mask_path,
        ).astype(np.float32)

    if strict_instance_mask:
        return zero

    # Required baseline fallback: bbox union mask.
    return build_support_mask(
        image_shape=image_np.shape,
        annotations=annotations,
        target_class_id=int(target_class_id),
        support_type="bbox",
        ring_width=4,
        mask_path=None,
    ).astype(np.float32)

