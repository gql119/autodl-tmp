"""Experiment B2: retain B0 computation and add a one-step adaptive learner."""

import torch
import torch.nn.functional as F

from .adaptive_learner import AdaptiveLearner
from .legacy_kproto_ret import LegacyKProtoTrainer


class B2AdaptiveTrainer(LegacyKProtoTrainer):
    def __init__(self, cfg, method_cfg, device, frozen_reference, adaptive_model):
        super().__init__(cfg, method_cfg, device, frozen_reference)
        adaptive_cfg = method_cfg["adaptive"]
        if int(adaptive_cfg["inner_steps"]) != 1:
            raise ValueError("B2 first version requires exactly one inner step")
        self.lambda_adapt = float(adaptive_cfg["lambda_adapt"])
        self.lambda_nt = float(adaptive_cfg["lambda_nt"])
        self.adaptive_batch_size = int(adaptive_cfg["batch_size"])
        if self.adaptive_batch_size < 1 or self.adaptive_batch_size > self.universal_batch_size:
            raise ValueError("Invalid adaptive inner batch size")
        self._adaptive_calls = 0
        self.adaptive = AdaptiveLearner(
            adaptive_model, self.feature_layer, self.target_class_id,
            self.imgsz, float(adaptive_cfg["inner_lr"]),
        )
        frozen_state = frozen_reference.state_dict()
        adaptive_state = adaptive_model.state_dict()
        if frozen_state.keys() != adaptive_state.keys() or any(
            not torch.equal(value, adaptive_state[name])
            for name, value in frozen_state.items()
        ):
            raise ValueError("Frozen reference and adaptive learner must start at the same checkpoint")
        self._last_poison = None

    def _compose_delta_batched(self, *args, **kwargs):
        result = super()._compose_delta_batched(*args, **kwargs)
        self._last_poison = result[2]
        return result

    def _batch_loss(self, items, prototypes):
        result = super()._batch_loss(items, prototypes)
        if result is None:
            return None
        frozen_loss, metrics = result
        poison = self._last_poison
        pad_h, pad_w = poison.shape[-2:]
        clean_images, nt_masks, selected_items = [], [], []
        frozen_feature = self.multi_features[self.feature_layer]
        start = self._adaptive_calls % len(items)
        selected_indices = [(start + offset) % len(items) for offset in range(
            min(self.adaptive_batch_size, len(items))
        )]
        self._adaptive_calls += 1
        for index in selected_indices:
            image, annotations, support = items[index]
            h, w = image.shape[:2]
            selected_items.append((image, annotations, support))
            clean_images.append(F.pad(
                torch.from_numpy(image).permute(2, 0, 1).float(),
                (0, pad_w - w, 0, pad_h - h), value=0.5,
            ))
            _, _, non_targets = self._masks(
                annotations, support, (h, w), frozen_feature[index:index + 1], (pad_h, pad_w)
            )
            nt_masks.append(non_targets)
        clean = torch.stack(clean_images).to(self.device)
        selected_poison = poison[selected_indices]
        target, non_target, adaptive_metrics = self.adaptive.losses(
            clean, selected_poison, selected_items, nt_masks, self._pooled
        )
        adaptive_loss = self.lambda_adapt * (target + self.lambda_nt * non_target)
        total = frozen_loss + adaptive_loss
        grad_frozen = torch.autograd.grad(frozen_loss, poison, retain_graph=True)[0]
        grad_adapt = torch.autograd.grad(adaptive_loss, poison, retain_graph=True)[0]
        grad_total = grad_frozen + grad_adapt
        norm_product = grad_frozen.norm() * grad_adapt.norm()
        grad_cosine = (
            torch.sum(grad_frozen * grad_adapt) / norm_product.clamp_min(1e-12)
        )
        metrics.update(adaptive_metrics)
        metrics.update({
            "L_frozen": frozen_loss,
            "L_adaptive": adaptive_loss,
            "L_total": total,
            "grad_delta_frozen": grad_frozen.norm(),
            "grad_delta_adaptive": grad_adapt.norm(),
            "grad_delta_total": grad_total.norm(),
            "cos_grad_frozen_adaptive": grad_cosine,
            "adaptive_batch_size": torch.tensor(float(len(selected_items)), device=self.device),
        })
        return total, metrics
