import random
from typing import List

import torch

from .base import BasePoisonGenerator, PoisonResult


class EMPoisonGenerator(BasePoisonGenerator):
    """Detection-adapted EM baseline with strict support control."""

    def generate(
        self,
        image,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ) -> PoisonResult:
        random.seed(seed)
        torch.manual_seed(seed)

        support_np, ring_np = self._build_support(image.shape, annotations, support_type=support_type, image_path=image_path)
        if float(support_np.mean()) <= 0.0:
            zero = image * 0.0
            return PoisonResult(image.copy(), zero, support_np, ring_np, {"loss_em": 0.0, "grad_norm": 0.0})

        img = self._to_tensor(image)
        support = torch.from_numpy(support_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        support3 = support.repeat(1, 3, 1, 1)

        delta = torch.zeros_like(img, requires_grad=True)
        step_size = float(self.method_cfg.get("step_size", 2 / 255))
        noise_scale = float(self.method_cfg.get("noise_scale", 1.0))

        loss_val = 0.0
        grad_norm = 0.0
        for _ in range(int(steps)):
            adv = torch.clamp(img + delta * support3 * noise_scale, 0.0, 1.0)
            preds = self._forward_raw(adv)
            target_scores = self._target_scores(preds)

            # EM spirit: minimize training error proxy on poisoned samples.
            loss = -torch.mean(target_scores)
            loss_val = float(loss.item())

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                grad_norm = float(torch.norm(grad).item()) if grad is not None else 0.0
                if not torch.isfinite(torch.tensor(grad_norm)):
                    print("[WARN][EM] grad norm is NaN/Inf")
                elif grad_norm < 1e-12:
                    print("[WARN][EM] grad norm is near zero")
                delta -= step_size * torch.sign(grad)
                delta *= support3
                delta.clamp_(-eps, eps)

        with torch.no_grad():
            pert = torch.clamp(delta * support3 * noise_scale, -eps, eps)
            poisoned = torch.clamp(img + pert, 0.0, 1.0)

        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=self._to_numpy(pert),
            support_mask=support_np,
            ring_mask=ring_np,
            losses={"loss_em": loss_val, "grad_norm": grad_norm},
        )

