# ue_framework/utils/robust_transforms.py

from io import BytesIO
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return arr


def apply_jpeg_compression(img: np.ndarray, quality: int = 80) -> np.ndarray:
    arr = _to_uint8_rgb(img)

    buffer = BytesIO()
    Image.fromarray(arr).save(
        buffer,
        format="JPEG",
        quality=int(quality),
        optimize=False,
    )
    buffer.seek(0)

    out = Image.open(buffer).convert("RGB")
    return np.asarray(out).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    arr = _to_uint8_rgb(img)

    if int(ksize) % 2 == 0:
        raise ValueError(f"Gaussian blur ksize must be odd, got {ksize}")
    if int(ksize) < 3:
        raise ValueError(f"Gaussian blur ksize must be >= 3, got {ksize}")

    try:
        import cv2

        out = cv2.GaussianBlur(
            arr,
            ksize=(int(ksize), int(ksize)),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
        return out.astype(np.uint8)
    except Exception:
        # PIL fallback. radius=1.0 is a close practical fallback for k=3.
        from PIL import ImageFilter

        radius = max(0.1, float(ksize) / 3.0)
        out = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(out).astype(np.uint8)


def apply_poison_post_transform(
    img: np.ndarray,
    cfg: Dict,
    has_poison_support: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply robustness transform after poison generation and before saving.

    This is for robustness evaluation only:
    - Ours optimization remains unchanged.
    - Transform is applied to poisoned target-support images before victim training.
    """

    platform_cfg = cfg.get("platform", {}) if isinstance(cfg, dict) else {}
    tcfg = platform_cfg.get("poison_post_transform", {}) or {}

    enabled = bool(tcfg.get("enabled", False))
    transform_type = str(tcfg.get("type", "none")).lower()

    info = {
        "post_transform_enabled": bool(enabled),
        "post_transform_type": transform_type,
    }

    if (not enabled) or transform_type in ("", "none", "null"):
        return img, info

    apply_to = str(tcfg.get("apply_to", "poisoned_only")).lower()
    if apply_to == "poisoned_only" and not bool(has_poison_support):
        info["post_transform_applied"] = False
        return img, info
    
    if transform_type in ("jpeg", "jpeg_compression", "jpeg30"):
        quality = int(tcfg.get("jpeg_quality", 30))
        out = apply_jpeg_compression(img, quality=quality)
        info.update(
            {
                "post_transform_applied": True,
                "post_transform_type": "jpeg",
                "jpeg_quality": quality,
            }
        )
        return out, info
    
    if transform_type in ("jpeg", "jpeg_compression", "jpeg50"):
        quality = int(tcfg.get("jpeg_quality", 50))
        out = apply_jpeg_compression(img, quality=quality)
        info.update(
            {
                "post_transform_applied": True,
                "post_transform_type": "jpeg",
                "jpeg_quality": quality,
            }
        )
        return out, info

    if transform_type in ("jpeg", "jpeg_compression", "jpeg80"):
        quality = int(tcfg.get("jpeg_quality", 80))
        out = apply_jpeg_compression(img, quality=quality)
        info.update(
            {
                "post_transform_applied": True,
                "post_transform_type": "jpeg",
                "jpeg_quality": quality,
            }
        )
        return out, info

    if transform_type in ("gaussian_blur", "blur", "gaussian"):
        ksize = int(tcfg.get("blur_ksize", 3))
        sigma = float(tcfg.get("blur_sigma", 0.0))
        out = apply_gaussian_blur(img, ksize=ksize, sigma=sigma)
        info.update(
            {
                "post_transform_applied": True,
                "post_transform_type": "gaussian_blur",
                "blur_ksize": ksize,
                "blur_sigma": sigma,
            }
        )
        return out, info

    raise ValueError(f"Unknown poison_post_transform type: {transform_type}")