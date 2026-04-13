import random
from typing import List

import torch

from .em import EMPoisonGenerator

try:
    import kornia.augmentation as K
except Exception:  # pragma: no cover
    K = None


class REMPoisonGenerator(EMPoisonGenerator):
    """Detection-adapted REM baseline: EM-mask + EOT only."""

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        if K is not None:
            self.eot_aug = K.AugmentationSequential(
                K.RandomAffine(degrees=12.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=0.7),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
                same_on_batch=False,
            ).to(self.device)
        else:
            self.eot_aug = None

    def _apply_eot(self, x: torch.Tensor) -> torch.Tensor:
        if self.eot_aug is None:
            gain = 0.9 + 0.2 * torch.rand(1, device=x.device)
            return torch.clamp(x * gain, 0.0, 1.0)
        return self.eot_aug(x)

    def generate(
        self,
        image,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        support_np, ring_np = self._build_support(image.shape, annotations, support_type=support_type, image_path=image_path)
        if float(support_np.mean()) <= 0.0:
            zero = image * 0.0
            from .base import PoisonResult

            return PoisonResult(image.copy(), zero, support_np, ring_np, {"loss_rem": 0.0, "grad_norm": 0.0})

        img = self._to_tensor(image)
        support = torch.from_numpy(support_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        support3 = support.repeat(1, 3, 1, 1)

        delta = torch.zeros_like(img, requires_grad=True)
        step_size = float(self.method_cfg.get("step_size", 2 / 255))
        noise_scale = float(self.method_cfg.get("noise_scale", 1.0))
        eot_samples = int(self.method_cfg.get("eot_samples", 4))

        loss_val = 0.0
        grad_norm = 0.0
        for _ in range(int(steps)):
            adv = torch.clamp(img + delta * support3 * noise_scale, 0.0, 1.0)
            loss = 0.0
            for _e in range(max(1, eot_samples)):
                aug_adv = self._apply_eot(adv)
                preds = self._forward_raw(aug_adv)
                target_scores = self._target_scores(preds)
                loss = loss + (-torch.mean(target_scores))
            loss = loss / max(1, eot_samples)
            loss_val = float(loss.item())

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                grad_norm = float(torch.norm(grad).item()) if grad is not None else 0.0
                if not torch.isfinite(torch.tensor(grad_norm)):
                    print("[WARN][REM] grad norm is NaN/Inf")
                elif grad_norm < 1e-12:
                    print("[WARN][REM] grad norm is near zero")
                delta -= step_size * torch.sign(grad)
                delta *= support3
                delta.clamp_(-eps, eps)

        with torch.no_grad():
            pert = torch.clamp(delta * support3 * noise_scale, -eps, eps)
            poisoned = torch.clamp(img + pert, 0.0, 1.0)

        from .base import PoisonResult

        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=self._to_numpy(pert),
            support_mask=support_np,
            ring_mask=ring_np,
            losses={"loss_rem": loss_val, "grad_norm": grad_norm},
        )

