import hashlib
import math
import random
from typing import List

import numpy as np
import torch

from .base import BasePoisonGenerator, PoisonResult
from .support_utils import get_target_instance_support


def build_lsp_pattern(
    h: int,
    w: int,
    c: int,
    target_class_id: int,
    patch_size: int,
    seed: int,
    eps: float,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build linearly separable synthetic shortcut pattern for a target class.
    Returns tensor shape [1, C, H, W], linf-normalized to eps.
    """
    if device is None:
        device = torch.device("cpu")

    low_h = int(math.ceil(float(h) / float(max(1, patch_size))))
    low_w = int(math.ceil(float(w) / float(max(1, patch_size))))

    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + int(target_class_id) * 1009)

    # Rademacher-like synthetic code.
    low = torch.randint(0, 2, (1, c, low_h, low_w), generator=g, device=device, dtype=torch.int64).float()
    low = low * 2.0 - 1.0

    # TODO: Optional covariance mixing for richer synthetic codebook.
    up = torch.nn.functional.interpolate(low, size=(h, w), mode="nearest")
    up = up - up.mean(dim=(2, 3), keepdim=True)
    ch_std = up.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    up = up / ch_std

    linf = up.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    up = up / linf * float(eps)
    return up


class LSPMaskPoisoner(BasePoisonGenerator):
    """
    LSP-det-mask baseline:
    class-conditional synthetic shortcut restricted to target instance support.
    No detection-aware modules are used.
    """

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.patch_size = int(method_cfg.get("patch_size", 8))
        self.pattern_seed = int(method_cfg.get("pattern_seed", 0))
        self.class_conditional = bool(method_cfg.get("class_conditional", True))
        self.target_only = bool(method_cfg.get("target_only", True))
        self.normalize = str(method_cfg.get("normalize", "linf"))
        self.per_image_jitter = bool(method_cfg.get("per_image_jitter", False))
        self.jitter_strength = float(method_cfg.get("jitter_strength", 0.10))
        self.use_prebaked_instance_mask = bool(method_cfg.get("use_prebaked_instance_mask", True))
        self.strict_instance_mask = bool(method_cfg.get("strict_instance_mask", False))
        self.instance_mask_dir = str(cfg.get("data", {}).get("instance_mask_dir", "") or "")
        self._sanity_printed = False

    def _build_jitter(self, h: int, w: int, c: int, image_path: str) -> torch.Tensor:
        key = (image_path or "none").encode("utf-8")
        digest = hashlib.md5(key).hexdigest()[:8]
        seed = int(digest, 16) + int(self.pattern_seed) * 97 + int(self.target_class_id) * 193
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        j = torch.randn((1, c, h, w), generator=g, device=self.device, dtype=torch.float32)
        j = j / j.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return j

    def generate(
        self,
        image: np.ndarray,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ) -> PoisonResult:
        del steps, support_type
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        support_np = get_target_instance_support(
            image_path=image_path,
            image_np=image,
            annotations=annotations,
            target_class_id=self.target_class_id,
            use_prebaked_instance_mask=self.use_prebaked_instance_mask,
            instance_mask_dir=self.instance_mask_dir,
            strict_instance_mask=self.strict_instance_mask,
        )
        ring_np = np.zeros_like(support_np, dtype=np.float32)

        if float(support_np.sum()) <= 0.0:
            zero = np.zeros_like(image, dtype=np.float32)
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=support_np,
                ring_mask=ring_np,
                losses={},
                extras={
                    "method": "lsp_mask",
                    "poisoned": 0,
                    "is_poisoned": False,
                    "support_area_ratio": 0.0,
                    "changed_pixel_ratio": 0.0,
                    "patch_size": int(self.patch_size),
                    "pattern_seed": int(self.pattern_seed),
                    "delta_linf": 0.0,
                    "target_class_id": int(self.target_class_id),
                    "objective": "synthetic_class_shortcut",
                    "per_image_jitter": bool(self.per_image_jitter),
                    "psnr": None,
                    "lpips": None,
                },
            )

        img = self._to_tensor(image)
        h, w = int(image.shape[0]), int(image.shape[1])
        c = int(image.shape[2])
        eps_val = float(eps)

        class_id_for_pattern = int(self.target_class_id) if self.class_conditional else 0
        delta = build_lsp_pattern(
            h=h,
            w=w,
            c=c,
            target_class_id=class_id_for_pattern,
            patch_size=int(self.patch_size),
            seed=int(self.pattern_seed),
            eps=eps_val,
            device=self.device,
        )

        if self.per_image_jitter:
            jitter = self._build_jitter(h=h, w=w, c=c, image_path=image_path or "")
            delta = delta + float(self.jitter_strength) * jitter

        if str(self.normalize).lower() == "linf":
            delta = delta / delta.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            delta = delta * eps_val

        support = torch.from_numpy(support_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        support3 = support.repeat(1, c, 1, 1)
        delta = delta * support3
        delta = torch.clamp(delta, -eps_val, eps_val)

        poisoned = torch.clamp(img + delta, 0.0, 1.0)
        delta_linf = float(torch.max(torch.abs(delta)).item())
        support_area_ratio = float(np.mean(support_np > 0.5))
        changed_pixel_ratio = float(
            torch.mean((torch.max(torch.abs(delta), dim=1).values > (1.0 / 255.0)).float()).item()
        )
        poisoned_flag = int(delta_linf > (1.0 / 255.0))

        if not self._sanity_printed:
            print("[LSP-mask]")
            print(f"  support_area_ratio={support_area_ratio:.6f}")
            print(f"  patch_size={int(self.patch_size)}")
            print(f"  pattern_seed={int(self.pattern_seed)}")
            print(f"  delta_linf={delta_linf:.6f}")
            print(f"  per_image_jitter={bool(self.per_image_jitter)}")
            print(f"  target_class_id={int(self.target_class_id)}")
            self._sanity_printed = True

        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=self._to_numpy(delta),
            support_mask=support_np,
            ring_mask=ring_np,
            losses={},
            extras={
                "method": "lsp_mask",
                "poisoned": poisoned_flag,
                "is_poisoned": bool(poisoned_flag),
                "support_area_ratio": support_area_ratio,
                "changed_pixel_ratio": changed_pixel_ratio,
                "patch_size": int(self.patch_size),
                "pattern_seed": int(self.pattern_seed),
                "delta_linf": delta_linf,
                "target_class_id": int(self.target_class_id),
                "objective": "synthetic_class_shortcut",
                "per_image_jitter": bool(self.per_image_jitter),
                "psnr": None,
                "lpips": None,
            },
        )
