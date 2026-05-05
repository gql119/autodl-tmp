import random
from typing import List

import numpy as np
import torch

from .base import BasePoisonGenerator, PoisonResult
from .support_utils import get_target_instance_support


class TAPMaskPoisoner(BasePoisonGenerator):
    """
    TAP-det-mask baseline:
    PGD-style adversarial poisoning constrained to target instance support.

    Final baseline definition:
    - target instance support only;
    - target-only adversarial score-suppression proxy;
    - maximize -target_score under L_inf budget;
    - no full-label fallback;
    - no standard detector loss fallback;
    - no TAL / PAG / FPN-aware selection / RLCP / ALCE / DSNP modules.
    """

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)

        self.alpha = float(method_cfg.get("alpha", 2.0 / 255.0))
        self.pgd_steps = int(method_cfg.get("pgd_steps", 40))
        self.random_start = bool(method_cfg.get("random_start", True))
        self.objective = str(method_cfg.get("objective", "untargeted_det_loss"))

        self.use_target_only_labels = bool(method_cfg.get("use_target_only_labels", True))
        self.use_global_steps = bool(method_cfg.get("use_global_steps", False))

        self.use_prebaked_instance_mask = bool(method_cfg.get("use_prebaked_instance_mask", True))
        self.strict_instance_mask = bool(method_cfg.get("strict_instance_mask", False))
        self.instance_mask_dir = str(cfg.get("data", {}).get("instance_mask_dir", "") or "")

        self._sanity_printed = False

    def _proxy_target_only_loss(self, adv_img: torch.Tensor) -> torch.Tensor:
        """
        Target-only adversarial proxy loss.

        We maximize this loss with PGD:
            loss = -mean(sigmoid(target_score))

        Gradient ascent on this objective suppresses the surrogate detector's
        target-class response.
        """
        preds = self._forward_raw(adv_img)
        target_scores = self._target_scores(preds)
        return -torch.mean(torch.sigmoid(target_scores))

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
        del support_type

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

        steps_eff = int(
            steps if self.use_global_steps and int(steps) > 0 else self.pgd_steps
        )

        if float(support_np.sum()) <= 0.0:
            zero = np.zeros_like(image, dtype=np.float32)
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=support_np,
                ring_mask=ring_np,
                losses={
                    "tap_loss_start": 0.0,
                    "tap_loss_end": 0.0,
                },
                extras={
                    "method": "tap_mask",
                    "poisoned": 0,
                    "is_poisoned": False,
                    "support_area_ratio": 0.0,
                    "tap_loss_start": 0.0,
                    "tap_loss_end": 0.0,
                    "delta_linf": 0.0,
                    "steps_eff": int(steps_eff),
                    "pgd_steps": int(steps_eff),
                    "objective": str(self.objective),
                    "use_target_only_labels": True,
                    "det_loss_mode": "none_empty_support",
                    "psnr": None,
                    "lpips": None,
                },
            )

        img = self._to_tensor(image)

        support = torch.from_numpy(support_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        support3 = support.repeat(1, 3, 1, 1)

        eps_val = float(eps)
        alpha = float(self.alpha)

        if self.random_start:
            delta = (torch.rand_like(img) * 2.0 - 1.0) * eps_val
            delta = delta * support3
            delta = delta.detach().requires_grad_(True)
        else:
            delta = torch.zeros_like(img, requires_grad=True)

        loss_start = None
        loss_end = None
        det_loss_mode = "proxy_target_score_target_only"

        for _ in range(max(1, steps_eff)):
            adv = torch.clamp(img + delta, 0.0, 1.0)

            loss = self._proxy_target_only_loss(adv)

            if loss_start is None:
                loss_start = float(loss.item())
            loss_end = float(loss.item())

            if delta.grad is not None:
                delta.grad.zero_()

            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                if grad is None:
                    raise RuntimeError("[TAP] delta.grad is None during proxy backward.")

                # TAP proxy baseline:
                # maximize -target_score to suppress target confidence.
                delta.data = delta.data + alpha * grad.sign()
                delta.data = torch.clamp(delta.data, -eps_val, eps_val)
                delta.data = delta.data * support3
                delta.data = torch.clamp(img + delta.data, 0.0, 1.0) - img

        with torch.no_grad():
            pert = torch.clamp(delta * support3, -eps_val, eps_val)
            poisoned = torch.clamp(img + pert, 0.0, 1.0)

        delta_linf = float(torch.max(torch.abs(pert)).item())
        support_area_ratio = float(np.mean(support_np > 0.5))
        poisoned_flag = int(delta_linf > (1.0 / 255.0))
        objective_name = str(self.objective)

        if not self._sanity_printed:
            print("[TAP-mask]")
            print(f"  support_area_ratio={support_area_ratio:.6f}")
            print(f"  loss_start={float(loss_start or 0.0):.6f}")
            print(f"  loss_end={float(loss_end or 0.0):.6f}")
            print(f"  delta_linf={delta_linf:.6f}")
            print(f"  steps_eff={int(steps_eff)}")
            print(f"  objective={objective_name}")
            print("  use_target_only_labels=True")
            print(f"  det_loss_mode={det_loss_mode}")
            self._sanity_printed = True

        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=self._to_numpy(pert),
            support_mask=support_np,
            ring_mask=ring_np,
            losses={
                "tap_loss_start": float(loss_start or 0.0),
                "tap_loss_end": float(loss_end or 0.0),
            },
            extras={
                "method": "tap_mask",
                "poisoned": poisoned_flag,
                "is_poisoned": bool(poisoned_flag),
                "support_area_ratio": support_area_ratio,
                "tap_loss_start": float(loss_start or 0.0),
                "tap_loss_end": float(loss_end or 0.0),
                "delta_linf": delta_linf,
                "steps_eff": int(steps_eff),
                "pgd_steps": int(steps_eff),
                "objective": objective_name,
                "use_target_only_labels": True,
                "det_loss_mode": det_loss_mode,
                "psnr": None,
                "lpips": None,
            },
        )