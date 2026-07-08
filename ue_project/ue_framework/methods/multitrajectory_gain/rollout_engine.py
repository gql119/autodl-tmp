from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ue_framework.core.localized_support import LocalizedSupportBuilder
from ue_framework.core.supervision_decomposer import SupervisionDecomposer
from ue_framework.methods.learning_trajectory.virtual_update import parameter_leak_max_abs_diff, snapshot_parameters

from .functional_optimizer import clone_parameter_dict, functional_sgd_step, init_functional_sgd_state
from .learning_gain import LearningGainMetrics, compute_learning_gain_objective, compute_learning_gain_objective_v2
from .trajectory_state import FunctionalOptimizerState, TrajectoryBatchSequence


@dataclass
class RolloutOutput:
    loss: torch.Tensor
    metrics: LearningGainMetrics
    initial_query_losses: Dict[str, torch.Tensor]
    clean_query_losses: Dict[str, torch.Tensor]
    poison_query_losses: Dict[str, torch.Tensor]
    logs: Dict[str, float]
    per_step: List[Dict[str, float]]


class J3RolloutEngine:
    def __init__(
        self,
        adapter,
        decomposer: SupervisionDecomposer,
        support_builder: LocalizedSupportBuilder,
        selected_parameter_scope: str = "head",
        steps: int = 3,
        learning_rate: float = 1.0e-4,
        momentum: float = 0.9,
        weight_decay: float = 5.0e-4,
        nesterov: bool = False,
        eps: float = 16.0 / 255.0,
        protected_margin: float = 0.10,
        protected_clean_gain_min: float = 1.0e-4,
        gain_denominator_floor: float = 1.0e-4,
        lambda_protected: float = 1.0,
        lambda_authorized: float = 1.0,
        lambda_shared: float = 1.0,
        lambda_regularization: float = 1.0,
        objective_version: str = "v1",
        robust_scales: Dict[str, float] | None = None,
        authorized_tolerance: float = 0.10,
        shared_tolerance: float = 0.10,
        authorized_clean_gain_min: float = 1.0e-4,
        shared_clean_gain_min: float = 1.0e-4,
        protected_support_min_batches: int = 2,
    ) -> None:
        self.adapter = adapter
        self.decomposer = decomposer
        self.support_builder = support_builder
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.eps = float(eps)
        self.protected_margin = float(protected_margin)
        self.protected_clean_gain_min = float(protected_clean_gain_min)
        self.gain_denominator_floor = float(gain_denominator_floor)
        self.lambda_protected = float(lambda_protected)
        self.lambda_authorized = float(lambda_authorized)
        self.lambda_shared = float(lambda_shared)
        self.lambda_regularization = float(lambda_regularization)
        self.objective_version = str(objective_version)
        self.robust_scales = robust_scales
        self.authorized_tolerance = float(authorized_tolerance)
        self.shared_tolerance = float(shared_tolerance)
        self.authorized_clean_gain_min = float(authorized_clean_gain_min)
        self.shared_clean_gain_min = float(shared_clean_gain_min)
        self.protected_support_min_batches = int(protected_support_min_batches)
        self.selected_named_parameters = self.adapter.get_named_trainable_parameters(selected_parameter_scope)

    def initial_parameter_dict(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.selected_named_parameters}

    def run(self, sequence: TrajectoryBatchSequence, delta: torch.Tensor, create_graph: bool = True) -> RolloutOutput:
        if len(sequence.support_batches) < self.steps:
            raise ValueError(f"sequence has {len(sequence.support_batches)} support batches, need {self.steps}")
        snapshot = snapshot_parameters(self.adapter.model)
        initial_params = clone_parameter_dict(self.initial_parameter_dict(), detach=True)
        clean_params = clone_parameter_dict(initial_params, detach=True)
        poison_params = clone_parameter_dict(initial_params, detach=True)
        clean_state = init_functional_sgd_state(clean_params)
        poison_state = init_functional_sgd_state(poison_params)

        initial_query = self._query_losses(sequence.query_batch, initial_params, grad_enabled=False)
        per_step: List[Dict[str, float]] = []
        support_ratios: List[float] = []
        outside_max_values: List[float] = []
        support_counts = {"protected_support_batches": 0.0, "authorized_support_batches": 0.0, "shared_support_batches": 0.0}

        for step_idx in range(self.steps):
            support = sequence.support_batches[step_idx]
            clean_params, clean_state, clean_diag = self._rollout_step(
                support,
                clean_params,
                clean_state,
                delta=None,
                create_graph=False,
            )
            poison_params, poison_state, poison_diag = self._rollout_step(
                support,
                poison_params,
                poison_state,
                delta=delta,
                create_graph=create_graph,
            )
            support_ratios.append(poison_diag["valid_support_ratio"])
            outside_max_values.append(poison_diag["outside_support_max_abs_delta"])
            if clean_diag.get("protected_positive_count", 0.0) > 0.0:
                support_counts["protected_support_batches"] += 1.0
            if clean_diag.get("authorized_positive_count", 0.0) > 0.0:
                support_counts["authorized_support_batches"] += 1.0
            if clean_diag.get("shared_positive_count", 0.0) > 0.0:
                support_counts["shared_support_batches"] += 1.0
            per_step.append(
                {
                    "step": float(step_idx),
                    **{f"clean_{k}": v for k, v in clean_diag.items() if k.endswith("count") or k.endswith("loss")},
                    **{f"poison_{k}": v for k, v in poison_diag.items() if k.endswith("count") or k.endswith("loss")},
                    "assignment_overlap": self._assignment_overlap(clean_diag, poison_diag),
                    "valid_support_ratio": poison_diag["valid_support_ratio"],
                    "outside_support_max_abs_delta": poison_diag["outside_support_max_abs_delta"],
                }
            )

        clean_query = self._query_losses(sequence.query_batch, clean_params, grad_enabled=False)
        poison_query = self._query_losses(sequence.query_batch, poison_params, grad_enabled=create_graph)
        query_counts = poison_query["counts"]
        if self.objective_version == "v2":
            if self.robust_scales is None:
                raise ValueError("objective_version='v2' requires robust_scales")
            metrics = compute_learning_gain_objective_v2(
                initial_query["losses"],
                clean_query["losses"],
                poison_query["losses"],
                query_counts=query_counts,
                support_counts=support_counts,
                robust_scales=self.robust_scales,
                protected_margin=self.protected_margin,
                authorized_tolerance=self.authorized_tolerance,
                shared_tolerance=self.shared_tolerance,
                protected_clean_gain_min=self.protected_clean_gain_min,
                authorized_clean_gain_min=self.authorized_clean_gain_min,
                shared_clean_gain_min=self.shared_clean_gain_min,
                lambda_protected=self.lambda_protected,
                lambda_authorized=self.lambda_authorized,
                lambda_shared=self.lambda_shared,
                protected_support_min_batches=self.protected_support_min_batches,
            )
        else:
            metrics = compute_learning_gain_objective(
                initial_query["losses"],
                clean_query["losses"],
                poison_query["losses"],
                query_counts=query_counts,
                protected_margin=self.protected_margin,
                protected_clean_gain_min=self.protected_clean_gain_min,
                gain_denominator_floor=self.gain_denominator_floor,
                lambda_protected=self.lambda_protected,
                lambda_authorized=self.lambda_authorized,
                lambda_shared=self.lambda_shared,
            )
        regularization = delta.pow(2).mean()
        total = metrics.total_loss + self.lambda_regularization * regularization
        leak = parameter_leak_max_abs_diff(self.adapter.model, snapshot)
        logs = {
            **metrics.statistics,
            "regularization": float(regularization.detach().item()),
            "total_loss": float(total.detach().item()),
            "surrogate_parameter_max_abs_diff": leak,
            "support_ratio_mean": float(sum(support_ratios) / max(len(support_ratios), 1)),
            "outside_support_max_abs_delta": float(max(outside_max_values) if outside_max_values else 0.0),
            "steps_executed": float(self.steps),
            "clean_poison_initial_parameter_max_abs_diff": 0.0,
            **support_counts,
        }
        return RolloutOutput(
            loss=total,
            metrics=metrics,
            initial_query_losses=initial_query["losses"],
            clean_query_losses=clean_query["losses"],
            poison_query_losses=poison_query["losses"],
            logs=logs,
            per_step=per_step,
        )

    def _rollout_step(
        self,
        support,
        params: Dict[str, torch.Tensor],
        state: FunctionalOptimizerState,
        delta: torch.Tensor | None,
        create_graph: bool,
    ) -> Tuple[Dict[str, torch.Tensor], FunctionalOptimizerState, Dict[str, float]]:
        images = support.images
        support_stats = {"valid_support_ratio": 0.0, "outside_support_max_abs_delta": 0.0}
        if delta is not None:
            support_out = self.support_builder.build(images, support.batch)
            masked_delta = self.support_builder.apply_support(delta, support_out.valid_support_mask)
            images = (images + masked_delta).clamp(0.0, 1.0)
            outside = masked_delta * (1.0 - support_out.valid_support_mask)
            support_stats = {
                "valid_support_ratio": float(support_out.statistics["valid_support_ratio"]),
                "outside_support_max_abs_delta": float(outside.detach().abs().max().item()),
            }
        predictions = self.adapter.forward_with_parameters(images, params)
        full = self.adapter.compute_detection_loss(predictions, support.batch, class_filter=None, return_components=True)
        dec = self.decomposer.decompose(predictions, support.batch)
        updated, next_state, _grads = functional_sgd_step(
            params,
            full["total_loss"],
            state,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            create_graph=create_graph,
        )
        diag = {
            "support_total_loss": float(full["total_loss"].detach().item()),
            "protected_positive_count": dec.statistics["protected_positive_count"],
            "authorized_positive_count": dec.statistics["authorized_positive_count"],
            "ambiguous_positive_count": dec.statistics["ambiguous_positive_count"],
            "shared_positive_count": dec.statistics["shared_positive_count"],
            **support_stats,
        }
        return updated, next_state, diag

    def _query_losses(self, batch_data, params: Dict[str, torch.Tensor], grad_enabled: bool) -> Dict[str, Dict]:
        with torch.enable_grad() if grad_enabled else torch.no_grad():
            predictions = self.adapter.forward_with_parameters(batch_data.images, params)
            dec = self.decomposer.decompose(predictions, batch_data.batch)
        losses = {
            "protected": dec.protected_total if grad_enabled else dec.protected_total.detach(),
            "authorized": dec.authorized_total if grad_enabled else dec.authorized_total.detach(),
            "shared": dec.shared_total if grad_enabled else dec.shared_total.detach(),
        }
        return {"losses": losses, "counts": dec.statistics}

    @staticmethod
    def _assignment_overlap(clean_diag: Dict[str, float], poison_diag: Dict[str, float]) -> float:
        clean_total = (
            clean_diag.get("protected_positive_count", 0.0)
            + clean_diag.get("authorized_positive_count", 0.0)
            + clean_diag.get("shared_positive_count", 0.0)
        )
        poison_total = (
            poison_diag.get("protected_positive_count", 0.0)
            + poison_diag.get("authorized_positive_count", 0.0)
            + poison_diag.get("shared_positive_count", 0.0)
        )
        return float(min(clean_total, poison_total) / max(max(clean_total, poison_total), 1.0))
