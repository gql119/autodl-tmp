import math
from typing import Dict, Optional

import numpy as np
import torch


def to_float01(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32)
        max_v = float(np.max(out)) if out.size > 0 else 0.0
        min_v = float(np.min(out)) if out.size > 0 else 0.0
        if max_v > 1.5 or min_v < 0.0:
            out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_image_poison_stats(
    clean_img,
    poison_img,
    support_mask=None,
    threshold: float = 1.0 / 255.0,
) -> Dict[str, float]:
    clean = to_float01(clean_img)
    poison = to_float01(poison_img)
    diff = poison - clean
    abs_diff = np.abs(diff)

    delta_linf = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    mse = float(np.mean(diff ** 2)) if diff.size > 0 else 0.0
    if mse <= 1e-12:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(1.0 / np.sqrt(mse)))

    changed = np.any(abs_diff > float(threshold), axis=-1) if abs_diff.ndim == 3 else (abs_diff > float(threshold))
    changed_pixel_ratio = float(np.mean(changed)) if changed.size > 0 else 0.0

    support_area_ratio = 0.0
    if support_mask is not None:
        support_arr = np.asarray(support_mask).astype(np.float32)
        support_area_ratio = float(np.mean(support_arr > 0.5)) if support_arr.size > 0 else 0.0

    poisoned = int(delta_linf > float(threshold))
    return {
        "delta_linf": delta_linf,
        "mse": mse,
        "psnr": psnr,
        "changed_pixel_ratio": changed_pixel_ratio,
        "support_area_ratio": support_area_ratio,
        "poisoned": poisoned,
        "is_poisoned": bool(poisoned),
    }


class LPIPSComputer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.available = False
        self.model = None
        self.warn = ""
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(self.device).eval()
            self.available = True
        except Exception as e:  # pragma: no cover
            self.warn = str(e)
            self.available = False
            self.model = None
            print(f"[WARN][LPIPS] unavailable: {e}")

    @torch.no_grad()
    def __call__(self, clean_img, poison_img) -> Optional[float]:
        if not self.available or self.model is None:
            return None

        clean = to_float01(clean_img)
        poison = to_float01(poison_img)

        clean_t = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        poison_t = torch.from_numpy(poison).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        clean_t = clean_t * 2.0 - 1.0
        poison_t = poison_t * 2.0 - 1.0
        val = float(self.model(clean_t, poison_t).item())
        if math.isnan(val) or math.isinf(val):
            return None
        return val
