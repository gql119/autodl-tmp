from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch


LOSS_LOG_FIELDS = (
    "step",
    "L_core",
    "L_protect",
    "L_carrier",
    "L_auth",
    "L_delta",
    "weighted_carrier",
    "weighted_auth",
    "weighted_reg",
    "gradient_norm",
    "max_abs_delta",
    "mean_abs_delta",
    "saturation_ratio",
    "valid_target_gain",
    "valid_authorized_class_count",
)


@dataclass(frozen=True)
class CoreObjectiveConfig:
    lambda_carrier: float = 1.0
    lambda_auth: float = 1.0
    lambda_reg: float = 1e-3
    gradient_clip_norm: float = 1.0
    eps: float = 16.0 / 255.0


@dataclass
class CoreObjectiveResult:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]


def compose_core_objective(
    protect_loss: torch.Tensor,
    carrier_loss: torch.Tensor,
    authorized_loss: torch.Tensor,
    delta_obj: torch.Tensor,
    config: CoreObjectiveConfig,
) -> CoreObjectiveResult:
    delta_regularizer = delta_obj.square().mean()
    weighted_carrier = float(config.lambda_carrier) * carrier_loss
    weighted_auth = float(config.lambda_auth) * authorized_loss
    weighted_reg = float(config.lambda_reg) * delta_regularizer
    total = protect_loss + weighted_carrier + weighted_auth + weighted_reg
    if not bool(torch.isfinite(total.detach()).all()):
        raise FloatingPointError("non-finite OA-LGC core objective")
    return CoreObjectiveResult(
        total,
        {
            "L_protect": protect_loss,
            "L_carrier": carrier_loss,
            "L_auth": authorized_loss,
            "L_delta": delta_regularizer,
            "weighted_carrier": weighted_carrier,
            "weighted_auth": weighted_auth,
            "weighted_reg": weighted_reg,
        },
    )


def project_delta_(delta_obj: torch.Tensor, eps: float) -> None:
    with torch.no_grad():
        delta_obj.clamp_(min=-float(eps), max=float(eps))


def update_delta(
    result: CoreObjectiveResult,
    delta_obj: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    config: CoreObjectiveConfig,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    if delta_obj.grad is None:
        raise RuntimeError("OA-LGC core gradient did not reach delta_obj")
    if not bool(torch.isfinite(delta_obj.grad).all()):
        raise FloatingPointError("non-finite delta_obj gradient")
    gradient_norm = float(delta_obj.grad.detach().norm().item())
    torch.nn.utils.clip_grad_norm_([delta_obj], max_norm=float(config.gradient_clip_norm))
    optimizer.step()
    project_delta_(delta_obj, config.eps)
    return gradient_norm


def delta_metrics(delta_obj: torch.Tensor, eps: float) -> dict[str, float]:
    detached = delta_obj.detach()
    return {
        "mean_abs_delta": float(detached.abs().mean().item()),
        "max_abs_delta": float(detached.abs().max().item()),
        "saturation_ratio": float((detached.abs() >= float(eps) - 1e-8).float().mean().item()),
        "finite": bool(torch.isfinite(detached).all()),
    }


def save_delta_checkpoint(
    path: str | os.PathLike[str],
    delta_obj: torch.Tensor,
    config: CoreObjectiveConfig,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"delta checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta_obj": delta_obj.detach().cpu(),
            "eps": float(config.eps),
            "shape": tuple(delta_obj.shape),
            "metadata": dict(metadata or {}),
        },
        destination,
    )


def load_delta_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"delta_obj", "eps", "shape", "metadata"}
    if set(payload) != required:
        raise RuntimeError(f"invalid delta checkpoint schema: {sorted(payload)}")
    if tuple(payload["delta_obj"].shape) != tuple(payload["shape"]):
        raise RuntimeError("delta checkpoint shape metadata mismatch")
    if float(payload["delta_obj"].abs().max()) > float(payload["eps"]) + 1e-7:
        raise RuntimeError("delta checkpoint exceeds epsilon budget")
    return payload

