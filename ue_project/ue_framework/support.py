from typing import List, Optional

import cv2
import numpy as np



def _bbox_to_pixels(bbox, w: int, h: int):
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2.0) * w)
    y1 = int((cy - bh / 2.0) * h)
    x2 = int((cx + bw / 2.0) * w)
    y2 = int((cy + bh / 2.0) * h)
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2



def _draw_instance_mask(mask: np.ndarray, ann: dict) -> None:
    h, w = mask.shape
    if ann.get("polygon"):
        pts = np.array(ann["polygon"], dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 2:
            pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], color=1)
            return

    _draw_pseudo_bbox_ellipse(mask, ann["bbox"])


def _draw_pseudo_bbox_ellipse(mask: np.ndarray, bbox) -> None:
    h, w = mask.shape
    x1, y1, x2, y2 = _bbox_to_pixels(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return

    # Fallback pseudo-instance mask when polygon is unavailable.
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(1, int((x2 - x1) * 0.45))
    ry = max(1, int((y2 - y1) * 0.45))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, thickness=-1)


def build_forced_pseudo_instance_masks(
    image_shape,
    annotations: List[dict],
    target_class_id: int,
) -> List[np.ndarray]:
    h, w = image_shape[:2]
    masks = []
    for ann in annotations:
        if int(ann["cls"]) != int(target_class_id):
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        _draw_pseudo_bbox_ellipse(mask, ann["bbox"])
        masks.append(mask.astype(np.float32))
    return masks



def _read_offline_topology_mask(mask_path: Optional[str], h: int, w: int) -> Optional[np.ndarray]:
    if not mask_path:
        return None

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None

    if mask_img.shape[:2] != (h, w):
        # Nearest neighbor keeps 0/1/2 topology values.
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_img



def build_support_mask(
    image_shape,
    annotations: List[dict],
    target_class_id: int,
    support_type: str,
    ring_width: int = 4,
    mask_path: str = None,
):
    h, w = image_shape[:2]
    base = np.zeros((h, w), dtype=np.float32)

    # BBox support always follows annotation boxes and ignores offline topology mask.
    if support_type == "bbox":
        target_anns = [a for a in annotations if int(a["cls"]) == int(target_class_id)]
        if len(target_anns) == 0:
            return base
        for ann in target_anns:
            x1, y1, x2, y2 = _bbox_to_pixels(ann["bbox"], w, h)
            if x2 > x1 and y2 > y1:
                base[y1:y2, x1:x2] = 1.0
        return base

    if support_type in {"mask", "mask_ring"}:
        # Topology mask priority: 0=bg, 1=core, 2=ring
        mask_img = _read_offline_topology_mask(mask_path, h, w)
        if mask_img is not None:
            if support_type == "mask":
                return (mask_img == 1).astype(np.float32)
            return (mask_img == 2).astype(np.float32)

        # Original fallback path: pseudo instance mask from annotations.
        target_anns = [a for a in annotations if int(a["cls"]) == int(target_class_id)]
        if len(target_anns) == 0:
            return base

        inst = np.zeros((h, w), dtype=np.uint8)
        for ann in target_anns:
            _draw_instance_mask(inst, ann)
        inst_f = inst.astype(np.float32)
        if support_type == "mask":
            return inst_f

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
        dil = cv2.dilate(inst, kernel, iterations=1)
        ero = cv2.erode(inst, kernel, iterations=1)
        ring = np.clip(dil.astype(np.float32) - ero.astype(np.float32), 0.0, 1.0)
        return ring

    raise ValueError(f"Unsupported support_type: {support_type}")
