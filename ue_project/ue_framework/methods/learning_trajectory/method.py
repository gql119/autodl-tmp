from __future__ import annotations

from typing import Dict, Optional

import torch

from ue_framework.core import ClassConditionedRouter, DetectorAdapter

from .class_conditioned_loss import compute_class_conditioned_detection_loss
from .gradient_extractor import extract_gradient_vector
from .meta_objective import build_meta_query_loss
from .trajectory_objective import build_p1_trajectory_loss
from .virtual_update import make_virtual_parameters, parameter_leak_max_abs_diff, snapshot_parameters


class LearningTrajectoryMethod:
    def __init__(self, adapter: DetectorAdapter, config: Dict) -> None:
        self.adapter = adapter
        self.config = config
        protected_class_id = int(config.get("protected_class_id", adapter.protected_class_id))
        num_classes = int(config.get("num_classes", adapter.num_classes))
        routing_cfg = config.get("class_routing", {})
        authorized = config.get("authorized_class_ids", "auto")
        self.router = ClassConditionedRouter(
            protected_class_id=protected_class_id,
            authorized_class_ids=authorized,
            num_classes=num_classes,
            exclude_ambiguous=bool(routing_cfg.get("exclude_ambiguous", True)),
        )

    def compute_p1_step(
        self,
        clean_images: torch.Tensor,
        delta: torch.Tensor,
        batch: Dict,
    ) -> Dict[str, object]:
        traj_cfg = self.config.get("trajectory", {})
        scope = str(traj_cfg.get("parameter_scope", "head"))
        normalize = bool(traj_cfg.get("normalize_per_parameter", True))
        eps = float(traj_cfg.get("eps", 1.0e-8))
        selected = self.adapter.get_named_trainable_parameters(scope)

        poisoned_images = (clean_images + delta).clamp(0.0, 1.0)
        clean_predictions = self.adapter.forward(clean_images)
        poison_predictions = self.adapter.forward(poisoned_images)

        clean_losses = compute_class_conditioned_detection_loss(
            self.adapter,
            clean_predictions,
            batch,
            self.router,
            include_background_negatives=bool(self.config.get("class_routing", {}).get("include_background_negatives", False)),
        )
        poison_losses = compute_class_conditioned_detection_loss(
            self.adapter,
            poison_predictions,
            batch,
            self.router,
            include_background_negatives=bool(self.config.get("class_routing", {}).get("include_background_negatives", False)),
        )

        # Poison gradients keep create_graph=True so the trajectory loss remains differentiable to delta.
        g_protected_clean = extract_gradient_vector(
            clean_losses["protected_total_loss"],
            selected,
            create_graph=False,
            retain_graph=True,
            normalize_per_parameter=normalize,
            eps=eps,
        )
        g_authorized_clean = extract_gradient_vector(
            clean_losses["authorized_total_loss"],
            selected,
            create_graph=False,
            retain_graph=True,
            normalize_per_parameter=normalize,
            eps=eps,
        )
        g_protected_poison = extract_gradient_vector(
            poison_losses["protected_total_loss"],
            selected,
            create_graph=True,
            retain_graph=True,
            normalize_per_parameter=normalize,
            eps=eps,
        )
        g_authorized_poison = extract_gradient_vector(
            poison_losses["authorized_total_loss"],
            selected,
            create_graph=True,
            retain_graph=True,
            normalize_per_parameter=normalize,
            eps=eps,
        )

        p1 = build_p1_trajectory_loss(
            protected_clean=g_protected_clean,
            protected_poison=g_protected_poison,
            authorized_clean=g_authorized_clean,
            authorized_poison=g_authorized_poison,
            lambda_protected=float(traj_cfg.get("lambda_protected", 1.0)),
            lambda_authorized=float(traj_cfg.get("lambda_authorized", 1.0)),
            use_protected=bool(traj_cfg.get("use_protected", True)),
            use_authorized=bool(traj_cfg.get("use_authorized", True)),
            eps=eps,
        )

        logs = self._tensor_logs(p1)
        logs.update(self._loss_logs(clean_losses, suffix="clean"))
        logs.update(self._loss_logs(poison_losses, suffix="poison"))
        logs.update(
            {
                "norm_g_protected_clean": float(g_protected_clean.norm.detach().item()),
                "norm_g_protected_poison": float(g_protected_poison.norm.detach().item()),
                "norm_g_authorized_clean": float(g_authorized_clean.norm.detach().item()),
                "norm_g_authorized_poison": float(g_authorized_poison.norm.detach().item()),
                "effective_gradient_parameter_count": float(
                    min(
                        g_protected_clean.effective_parameter_count,
                        g_protected_poison.effective_parameter_count,
                        g_authorized_clean.effective_parameter_count,
                        g_authorized_poison.effective_parameter_count,
                    )
                ),
            }
        )
        logs.update(self._delta_logs(delta))
        return {"loss": p1["loss"], "logs": logs}

    def compute_p2_step(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        support_batch: Dict,
        query_batch: Dict,
        delta: torch.Tensor,
    ) -> Dict[str, object]:
        vu_cfg = self.config.get("virtual_update", {})
        meta_cfg = self.config.get("meta", {})
        scope = str(vu_cfg.get("parameter_scope", "head"))
        lr = float(vu_cfg.get("lr", 0.001))
        selected = self.adapter.get_named_trainable_parameters(scope)
        snapshot = snapshot_parameters(self.adapter.model)

        support_poison = (support_images + delta).clamp(0.0, 1.0)
        support_predictions = self.adapter.forward(support_poison)
        support_loss = self.adapter.compute_detection_loss(
            support_predictions,
            support_batch,
            class_filter=None,
            return_components=False,
        )
        virtual = make_virtual_parameters(
            self.adapter.model,
            selected,
            support_loss=support_loss,
            lr=lr,
            create_graph=True,
        )
        support_class_losses = compute_class_conditioned_detection_loss(
            self.adapter,
            support_predictions,
            support_batch,
            self.router,
        )

        query_before = self.adapter.forward(query_images)
        before_losses = compute_class_conditioned_detection_loss(self.adapter, query_before, query_batch, self.router)

        query_after = self.adapter.forward_with_parameters(query_images, virtual.updated_parameters)
        after_losses = compute_class_conditioned_detection_loss(self.adapter, query_after, query_batch, self.router)
        meta = build_meta_query_loss(
            protected_query_loss=after_losses["protected_total_loss"],
            authorized_query_loss=after_losses["authorized_total_loss"],
            lambda_protected_query=float(meta_cfg.get("lambda_protected_query", 1.0)),
        )
        total_loss = float(meta_cfg.get("lambda_meta", 1.0)) * meta["meta_loss"]

        p1_regularizer: Optional[Dict[str, object]] = None
        if bool(meta_cfg.get("use_p1_regularizer", False)):
            p1_regularizer = self.compute_p1_step(support_images, delta, support_batch)
            total_loss = total_loss + float(meta_cfg.get("lambda_p1", 0.2)) * p1_regularizer["loss"]

        meta_grad = torch.autograd.grad(total_loss, delta, retain_graph=True, allow_unused=True)
        meta_grad_norm = (
            torch.zeros((), device=delta.device, dtype=delta.dtype)
            if meta_grad[0] is None
            else meta_grad[0].detach().norm()
        )

        leak = parameter_leak_max_abs_diff(self.adapter.model, snapshot)
        logs = {
            "support_total_loss": float(support_loss.detach().item()),
            "support_protected_loss": float(support_class_losses["protected_total_loss"].detach().item()),
            "support_authorized_loss": float(support_class_losses["authorized_total_loss"].detach().item()),
            "query_protected_loss_before_update": float(before_losses["protected_total_loss"].detach().item()),
            "query_protected_loss_after_update": float(after_losses["protected_total_loss"].detach().item()),
            "query_authorized_loss_before_update": float(before_losses["authorized_total_loss"].detach().item()),
            "query_authorized_loss_after_update": float(after_losses["authorized_total_loss"].detach().item()),
            "delta_query_protected_loss": float(
                (after_losses["protected_total_loss"] - before_losses["protected_total_loss"]).detach().item()
            ),
            "delta_query_authorized_loss": float(
                (after_losses["authorized_total_loss"] - before_losses["authorized_total_loss"]).detach().item()
            ),
            "virtual_update_grad_norm": float(virtual.support_grad_norm.detach().item()),
            "virtual_parameter_update_norm": float(virtual.update_norm.detach().item()),
            "meta_gradient_norm_to_delta": float(meta_grad_norm.item()),
            "parameter_leak_max_abs_diff": float(leak),
        }

        if bool(meta_cfg.get("enable_clean_counterfactual", True)):
            logs.update(
                self._clean_counterfactual_logs(
                    support_images,
                    query_images,
                    support_batch,
                    query_batch,
                    selected,
                    lr,
                    after_losses,
                )
            )
            logs["parameter_leak_max_abs_diff"] = max(
                logs["parameter_leak_max_abs_diff"],
                float(parameter_leak_max_abs_diff(self.adapter.model, snapshot)),
            )

        if p1_regularizer is not None:
            logs.update({f"p1_{k}": v for k, v in p1_regularizer["logs"].items() if isinstance(v, (float, int))})

        return {"loss": total_loss, "logs": logs}

    def _clean_counterfactual_logs(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        support_batch: Dict,
        query_batch: Dict,
        selected,
        lr: float,
        poison_update_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        clean_support_pred = self.adapter.forward(support_images)
        clean_support_loss = self.adapter.compute_detection_loss(clean_support_pred, support_batch)
        clean_virtual = make_virtual_parameters(
            self.adapter.model,
            selected,
            support_loss=clean_support_loss,
            lr=lr,
            create_graph=True,
        )
        query_clean_update = self.adapter.forward_with_parameters(query_images, clean_virtual.updated_parameters)
        clean_update_losses = compute_class_conditioned_detection_loss(self.adapter, query_clean_update, query_batch, self.router)
        protected_gap = poison_update_losses["protected_total_loss"] - clean_update_losses["protected_total_loss"]
        authorized_gap = (
            poison_update_losses["authorized_total_loss"] - clean_update_losses["authorized_total_loss"]
        ).abs()
        return {
            "query_protected_loss_clean_update": float(clean_update_losses["protected_total_loss"].detach().item()),
            "query_authorized_loss_clean_update": float(clean_update_losses["authorized_total_loss"].detach().item()),
            "protected_learning_gap": float(protected_gap.detach().item()),
            "authorized_learning_gap": float(authorized_gap.detach().item()),
        }

    @staticmethod
    def _tensor_logs(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {key: float(value.detach().item()) for key, value in values.items() if torch.is_tensor(value)}

    @staticmethod
    def _loss_logs(losses: Dict[str, torch.Tensor], suffix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for side in ["protected", "authorized"]:
            for comp in ["cls", "box", "dfl"]:
                key = f"{side}_{comp}_loss"
                if key in losses:
                    out[f"{key}_{suffix}"] = float(losses[key].detach().item())
        for key in ["protected_positive_count", "authorized_positive_count", "ambiguous_positive_count"]:
            if key in losses:
                out[key] = float(losses[key].detach().item())
        return out

    @staticmethod
    def _delta_logs(delta: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            abs_delta = delta.detach().abs()
            return {
                "mean_abs_delta": float(abs_delta.mean().item()),
                "max_abs_delta": float(abs_delta.max().item()),
                "saturation_ratio": float((abs_delta >= abs_delta.max().clamp_min(1.0e-12)).float().mean().item()),
                "perturbed_area_ratio": float((abs_delta.max(dim=1).values > (1.0 / 255.0)).float().mean().item()),
            }
