import random
from typing import List, Tuple

import numpy as np
import torch

from .base import PoisonResult
from .em_mask import EMMaskPoisoner

try:
    import kornia.augmentation as K
except Exception:  # pragma: no cover
    K = None


class REMMaskPoisoner(EMMaskPoisoner):
    """
    REM-mask final baseline:
    EM target-score proxy + EOT (no inner PGD / eta branch).
    """

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.alpha = float(method_cfg.get("alpha", method_cfg.get("step_size", self.alpha)))
        self.rem_steps = int(method_cfg.get("rem_steps", 10))
        self.objective = str(method_cfg.get("objective", "target_score_proxy_eot"))
        self.eot_samples = int(method_cfg.get("eot_samples", method_cfg.get("rem_eot_samples", 4)))
        if K is not None:
            self.eot_aug = K.AugmentationSequential(
                K.RandomAffine(degrees=12.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=0.7),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
                same_on_batch=False,
            ).to(self.device)
        else:
            self.eot_aug = None

        self._rem_sanity_printed = False

    def _sanity_tag(self) -> str:
        return "REM-mask"

    def _loss_prefix(self) -> str:
        return "rem"

    def _objective_base_name(self) -> str:
        return "target_score_proxy_eot"

    def _steps_key(self) -> str:
        return "rem_steps"

    def _default_steps(self) -> int:
        return int(self.rem_steps)

    def _apply_eot(self, x: torch.Tensor) -> torch.Tensor:
        if self.eot_aug is None:
            gain = 0.9 + 0.2 * torch.rand(1, device=x.device)
            return torch.clamp(x * gain, 0.0, 1.0)
        return self.eot_aug(x)

    def _should_use_standard_det_loss(self) -> bool:
        return False

    def _compute_objective_loss_batched(
        self,
        adv_img: torch.Tensor,
        annotations_list: List[List[dict]],
        orig_shapes: List[Tuple[int, int]],
        mode: str,
    ):
        del annotations_list, orig_shapes, mode
        eot_n = max(1, int(self.eot_samples))
        loss = torch.zeros((), device=adv_img.device, dtype=adv_img.dtype)
        for _ in range(eot_n):
            aug_adv = self._apply_eot(adv_img)
            loss = loss + self._target_score_proxy_loss(aug_adv)
        loss = loss / float(eot_n)
        return loss, "target_score_proxy_eot"

    def generate_batch(
        self,
        images: List[np.ndarray],
        annotations_list: List[List[dict]],
        image_paths: List[str],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
    ) -> List[PoisonResult]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Suppress parent one-time print; REM prints its own required sanity block.
        old_parent_print_flag = bool(getattr(self, "_batch_sanity_printed", False))
        self._batch_sanity_printed = True
        results = super().generate_batch(
            images=images,
            annotations_list=annotations_list,
            image_paths=image_paths,
            seed=seed,
            steps=steps,
            eps=eps,
            support_type=support_type,
        )
        self._batch_sanity_printed = True

        loss_key_start = "rem_loss_start"
        loss_key_end = "rem_loss_end"
        support_vals = []
        changed_vals = []
        delta_vals = []
        loss_start_vals = []
        loss_end_vals = []

        for r in results:
            if r.extras is None:
                r.extras = {}
            r.extras["objective"] = "target_score_proxy_eot"
            r.extras["det_loss_mode"] = "target_score_proxy_eot"
            r.extras["use_target_only_labels"] = False
            r.extras["eot_samples"] = int(self.eot_samples)
            r.extras["steps_eff"] = int(r.extras.get("steps_eff", self._resolve_steps(steps)))
            r.extras["rem_steps"] = int(r.extras.get("rem_steps", r.extras["steps_eff"]))
            r.extras.setdefault("psnr", None)
            r.extras.setdefault("lpips", None)

            support_vals.append(float(r.extras.get("support_area_ratio", 0.0)))
            changed_vals.append(float(r.extras.get("changed_pixel_ratio", 0.0)))
            delta_vals.append(float(r.extras.get("delta_linf", 0.0)))
            loss_start_vals.append(float(r.extras.get(loss_key_start, 0.0)))
            loss_end_vals.append(float(r.extras.get(loss_key_end, 0.0)))

        if not self._rem_sanity_printed:
            print("[REM-mask]")
            print(f"  support_area_ratio={float(np.mean(support_vals)) if support_vals else 0.0:.6f}")
            print(f"  changed_pixel_ratio={float(np.mean(changed_vals)) if changed_vals else 0.0:.6f}")
            print(f"  rem_loss_start={float(np.mean(loss_start_vals)) if loss_start_vals else 0.0:.6f}")
            print(f"  rem_loss_end={float(np.mean(loss_end_vals)) if loss_end_vals else 0.0:.6f}")
            print(f"  delta_linf={float(np.max(delta_vals)) if delta_vals else 0.0:.6f}")
            print(
                f"  steps_eff={int(results[0].extras.get('steps_eff', self._resolve_steps(steps))) if results else 0}"
            )
            print(f"  eot_samples={int(self.eot_samples)}")
            print("  objective=target_score_proxy_eot")
            print("  use_target_only_labels=False")
            print("  det_loss_mode=target_score_proxy_eot")
            self._rem_sanity_printed = True

        if not old_parent_print_flag:
            self._batch_sanity_printed = True
        return results
