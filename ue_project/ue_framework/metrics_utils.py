import math
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


VOC20_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def compute_psnr(clean: np.ndarray, poisoned: np.ndarray) -> float:
    mse = float(np.mean((clean - poisoned) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_lpips_batch(clean_batch: List[np.ndarray], poisoned_batch: List[np.ndarray]) -> Optional[float]:
    try:
        import lpips  # type: ignore
    except Exception:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    
    # [致命修复] 强制对齐图像尺寸为 256x256，防止 VOC 数据集分辨率差异导致 np.stack 崩溃
    target_size = (256, 256)
    def _align(batch):
        return [
            cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) 
            if img.shape[:2] != target_size else img 
            for img in batch
        ]

    clean_batch = _align(clean_batch)
    poisoned_batch = _align(poisoned_batch)

    with torch.no_grad():
        c = torch.from_numpy(np.stack(clean_batch)).permute(0, 3, 1, 2).float().to(device)
        p = torch.from_numpy(np.stack(poisoned_batch)).permute(0, 3, 1, 2).float().to(device)
        c = c * 2.0 - 1.0
        p = p * 2.0 - 1.0
        score = loss_fn(c, p).mean().item()
    return float(score)


def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def extract_map50_per_class(metrics_obj, num_classes: int, *, strict: bool = False):
    """Return AP50 in dataset class-id order.

    Ultralytics may compact AP arrays to classes present in the evaluated
    split. In that case ``ap_class_index`` is mandatory; array position is not
    silently treated as a class id.
    """
    box = getattr(metrics_obj, "box", None)
    if box is None:
        if strict:
            raise ValueError("Ultralytics metrics object has no box metrics.")
        return [float("nan")] * num_classes

    values = None
    for attr in ("ap50", "maps50"):
        arr = getattr(box, attr, None)
        if arr is not None:
            arr = np.array(arr).astype(float).reshape(-1)
            if arr.size:
                values = arr
                break
    if values is None:
        if strict:
            raise ValueError("Ultralytics box metrics contain no AP50 array.")
        return [float("nan")] * num_classes

    class_indices = getattr(box, "ap_class_index", None)
    if class_indices is None:
        class_indices = getattr(metrics_obj, "ap_class_index", None)
    if class_indices is None:
        if values.size != num_classes:
            if strict:
                raise ValueError(
                    "Compacted AP50 values require explicit ap_class_index."
                )
            return [float("nan")] * num_classes
        indices = np.arange(num_classes, dtype=np.int64)
    else:
        indices = np.asarray(class_indices).astype(np.int64).reshape(-1)
        if indices.size != values.size:
            raise ValueError("AP50 values and ap_class_index lengths differ.")

    if len(set(int(value) for value in indices)) != indices.size:
        raise ValueError("ap_class_index contains duplicate class ids.")
    if np.any(indices < 0) or np.any(indices >= num_classes):
        raise ValueError("ap_class_index contains an out-of-range class id.")
    result = np.full(num_classes, np.nan, dtype=np.float64)
    result[indices] = values
    if strict:
        if not np.isfinite(result).all():
            missing = np.where(~np.isfinite(result))[0].tolist()
            raise ValueError(f"Full AP50 mapping is incomplete; missing class ids {missing}.")
        if np.any(result < 0) or np.any(result > 1):
            raise ValueError("AP50 values must lie in [0,1].")
    return result.tolist()


def named_voc20_ap50(metrics_obj) -> Dict[str, float]:
    values = extract_map50_per_class(metrics_obj, len(VOC20_CLASS_NAMES), strict=True)
    return {
        name: float(values[index])
        for index, name in enumerate(VOC20_CLASS_NAMES)
    }


def compute_non_target_map(ap50_per_class: List[float], target_class_id: int) -> float:
    vals = [v for i, v in enumerate(ap50_per_class) if i != target_class_id and not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def build_pareto_scores(metrics: Dict, ref_target: float, ref_non_target: float) -> Dict:
    m_target = safe_float(metrics.get("mAP50_target"))
    m_non = safe_float(metrics.get("mAP50_non_target"))

    if math.isnan(m_target):
        collapse = float("nan")
    elif ref_target > 0:
        collapse = max(0.0, min(1.0, 1.0 - m_target / ref_target))
    else:
        collapse = max(0.0, 1.0 - m_target)

    if math.isnan(m_non):
        retain = float("nan")
    elif ref_non_target > 0:
        retain = max(0.0, min(1.5, m_non / ref_non_target))
    else:
        retain = m_non

    return {
        "target_collapse_score": collapse,
        "non_target_retention_score": retain,
    }
