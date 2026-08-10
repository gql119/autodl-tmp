from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .constraint_gradient_router import ConstraintTerm, route_coefficient_gradient
from .instance_canonical_carrier import tensor_sha256


COMPONENT_NAMES = ("easy_cls", "malc", "rms")
BOUNDARY_ORDER = (
    "prototype_incoherence",
    "cross_batch_malc_conflict",
    "objective_gradient_conflict",
    "cgr_selective_suppression",
    "carrier_update_sink",
)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("Geometry statistics contain non-finite values.")
    return float(torch.quantile(tensor, q))


def detached_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    left_flat = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right_flat = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("Cosine inputs must have identical shapes.")
    denominator = left_flat.norm() * right_flat.norm()
    if not math.isfinite(float(denominator)) or float(denominator) <= epsilon:
        raise RuntimeError("Cosine is undefined for zero or non-finite vectors.")
    value = float(torch.dot(left_flat, right_flat) / denominator)
    if not math.isfinite(value):
        raise RuntimeError("Cosine result is non-finite.")
    return max(-1.0, min(1.0, value))


def prototype_bank_hash(bank: Any) -> str:
    payload = {
        "direction_prototypes": [
            tensor_sha256(value) for value in bank.direction_prototypes
        ],
        "median_rms": [float(value) for value in bank.median_rms],
        "energy_floors": [float(value) for value in bank.energy_floors],
        "epsilon": float(bank.epsilon),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class PrototypeGeometryAccumulator:
    """Store detached direction batches for resultant and LOO diagnostics."""

    def __init__(self, *, num_scales: int, epsilon: float = 1e-12) -> None:
        if num_scales <= 0 or epsilon <= 0:
            raise ValueError("Prototype geometry dimensions are invalid.")
        self.num_scales = int(num_scales)
        self.epsilon = float(epsilon)
        self._directions: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_scales)
        ]
        self._total = [0 for _ in range(self.num_scales)]
        self._pooling_valid = [0 for _ in range(self.num_scales)]
        self._batch_count = 0

    def update(self, residuals: Any) -> None:
        if len(residuals.vectors) != self.num_scales:
            raise ValueError("Residual scale count differs from geometry audit.")
        for scale, (vectors, pooling_valid) in enumerate(
            zip(residuals.vectors, residuals.pooling_valid)
        ):
            if vectors.ndim != 2 or pooling_valid.shape != (vectors.shape[0],):
                raise ValueError(f"Scale {scale} residual geometry is misaligned.")
            selected = vectors.detach().to(device="cpu", dtype=torch.float64)[
                pooling_valid.detach().to(device="cpu", dtype=torch.bool)
            ]
            self._total[scale] += int(vectors.shape[0])
            self._pooling_valid[scale] += int(selected.shape[0])
            if selected.numel() and not torch.isfinite(selected).all():
                raise ValueError(f"Scale {scale} residuals contain non-finite values.")
            if selected.numel():
                norms = selected.norm(dim=1)
                selected = selected[norms > self.epsilon]
                norms = norms[norms > self.epsilon]
                directions = selected / norms.unsqueeze(1)
            else:
                directions = selected
            self._directions[scale].append(directions)
        self._batch_count += 1

    def finalize(
        self,
        *,
        reference_prototypes: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if self._batch_count == 0:
            raise RuntimeError("Prototype geometry requires at least one batch.")
        if reference_prototypes is not None and len(reference_prototypes) != self.num_scales:
            raise ValueError("Reference prototype scale count mismatch.")
        scale_records = []
        for scale, batches in enumerate(self._directions):
            valid_batches = [value for value in batches if value.numel()]
            directions = (
                torch.cat(valid_batches, dim=0)
                if valid_batches
                else torch.empty((0, 0), dtype=torch.float64)
            )
            direction_count = int(directions.shape[0])
            coverage = direction_count / max(self._total[scale], 1)
            record: dict[str, Any] = {
                "scale": scale,
                "batch_count": self._batch_count,
                "total_vector_count": self._total[scale],
                "pooling_valid_count": self._pooling_valid[scale],
                "pooling_valid_coverage": self._pooling_valid[scale]
                / max(self._total[scale], 1),
                "direction_count": direction_count,
                "coverage": float(coverage),
                "valid": False,
                "resultant": None,
                "loo_cosines": [],
                "loo_q25": None,
                "reference_cosine": None,
                "invalid_reason": None,
            }
            if direction_count == 0:
                record["invalid_reason"] = "no_nonzero_directions"
                scale_records.append(record)
                continue
            mean = directions.mean(dim=0)
            resultant = float(mean.norm())
            record["resultant"] = resultant
            if not math.isfinite(resultant) or resultant <= self.epsilon:
                record["invalid_reason"] = "direction_cancellation"
                scale_records.append(record)
                continue
            full_direction = mean / mean.norm()
            loo_values = []
            for leave_out in range(self._batch_count):
                retained = [
                    value
                    for index, value in enumerate(batches)
                    if index != leave_out and value.numel()
                ]
                if not retained:
                    continue
                retained_mean = torch.cat(retained, dim=0).mean(dim=0)
                if float(retained_mean.norm()) <= self.epsilon:
                    continue
                loo_values.append(
                    detached_cosine(full_direction, retained_mean, epsilon=self.epsilon)
                )
            record["loo_cosines"] = loo_values
            record["loo_q25"] = _quantile(loo_values, 0.25)
            if reference_prototypes is not None:
                try:
                    record["reference_cosine"] = detached_cosine(
                        full_direction,
                        reference_prototypes[scale],
                        epsilon=self.epsilon,
                    )
                except RuntimeError:
                    record["reference_cosine"] = None
            record["valid"] = True
            scale_records.append(record)

        effective = [
            item for item in scale_records
            if item["valid"] and item["coverage"] >= 0.80
        ]
        return {
            "batch_count": self._batch_count,
            "scales": scale_records,
            "effective_scale_count": len(effective),
            "effective_resultant_median": _quantile(
                [float(item["resultant"]) for item in effective], 0.5
            ),
            "effective_loo_q25_median": _quantile(
                [float(item["loo_q25"]) for item in effective if item["loo_q25"] is not None],
                0.5,
            ),
        }


@dataclass(frozen=True)
class ComponentGeometryBatch:
    record: dict[str, Any]
    gradients: Mapping[str, torch.Tensor]


def component_gradient_geometry(
    *,
    parameter: torch.Tensor,
    losses: Mapping[str, torch.Tensor],
    constraints: Sequence[ConstraintTerm],
    near_boundary: float,
    svd_relative_tolerance: float,
    epsilon: float = 1e-12,
) -> ComponentGeometryBatch:
    if tuple(losses) != COMPONENT_NAMES:
        raise ValueError(f"Component losses must be ordered as {COMPONENT_NAMES}.")
    routes = {}
    gradients = {}
    matrices = []
    for name in COMPONENT_NAMES:
        route = route_coefficient_gradient(
            parameter=parameter,
            target_loss=losses[name],
            constraints=constraints,
            near_boundary=near_boundary,
            svd_relative_tolerance=svd_relative_tolerance,
            epsilon=epsilon,
        )
        raw = route.target_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        projected = route.projected_target_gradient.detach().to(
            device="cpu", dtype=torch.float64
        ).reshape(-1)
        raw_norm = float(raw.norm())
        if not math.isfinite(raw_norm) or raw_norm <= epsilon:
            raise RuntimeError(f"Component {name} has a zero or non-finite gradient.")
        projected_norm = float(projected.norm())
        routes[name] = {
            "raw_norm": raw_norm,
            "projected_norm": projected_norm,
            "retention": projected_norm / raw_norm,
            "projected_cosine": (
                detached_cosine(raw, projected, epsilon=epsilon)
                if projected_norm > epsilon
                else None
            ),
            "rank": int(route.rank),
            "null_dimension": int(route.null_dimension),
            "active_constraints": list(route.active_constraints),
            "violated_constraints": list(route.violated_constraints),
            "max_projected_row_dot": float(route.max_projected_row_dot),
            "raw_gradient": raw.tolist(),
            "projected_gradient": projected.tolist(),
        }
        gradients[name] = raw
        matrices.append(route.constraint_matrix.detach().to(device="cpu", dtype=torch.float64))

    reference_matrix = matrices[0]
    for matrix in matrices[1:]:
        if matrix.shape != reference_matrix.shape or not torch.equal(
            matrix, reference_matrix
        ):
            raise RuntimeError("Components did not reuse an identical CGR row matrix.")
    route_metadata = [
        (
            routes[name]["rank"],
            routes[name]["null_dimension"],
            routes[name]["active_constraints"],
            routes[name]["violated_constraints"],
        )
        for name in COMPONENT_NAMES
    ]
    if any(item != route_metadata[0] for item in route_metadata[1:]):
        raise RuntimeError("Components did not reuse identical CGR active rows.")
    matrix_hash = tensor_sha256(reference_matrix.float())
    pairwise = {
        "malc_vs_easy": detached_cosine(gradients["malc"], gradients["easy_cls"]),
        "malc_vs_rms": detached_cosine(gradients["malc"], gradients["rms"]),
        "easy_vs_rms": detached_cosine(gradients["easy_cls"], gradients["rms"]),
    }
    return ComponentGeometryBatch(
        record={
            "components": routes,
            "pairwise_cosines": pairwise,
            "constraint_matrix_shape": list(reference_matrix.shape),
            "constraint_matrix_hash": matrix_hash,
        },
        gradients=gradients,
    )


def summarize_gradient_geometry(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Gradient geometry requires at least one batch.")
    malc_gradients = [
        torch.tensor(
            record["components"]["malc"]["raw_gradient"],
            dtype=torch.float64,
        )
        for record in records
    ]
    cross_batch = [
        detached_cosine(malc_gradients[left], malc_gradients[right])
        for left in range(len(malc_gradients))
        for right in range(left + 1, len(malc_gradients))
    ]
    component_summary = {}
    for name in COMPONENT_NAMES:
        retentions = [float(record["components"][name]["retention"]) for record in records]
        component_summary[name] = {
            "raw_norm_median": _quantile(
                [float(record["components"][name]["raw_norm"]) for record in records],
                0.5,
            ),
            "projected_norm_median": _quantile(
                [
                    float(record["components"][name]["projected_norm"])
                    for record in records
                ],
                0.5,
            ),
            "retention_median": _quantile(retentions, 0.5),
            "retention_q25": _quantile(retentions, 0.25),
        }
    return {
        "batch_count": len(records),
        "batches": list(records),
        "cross_batch_malc_cosines": cross_batch,
        "cross_batch_malc_median": _quantile(cross_batch, 0.5),
        "cross_batch_malc_q25": _quantile(cross_batch, 0.25),
        "malc_vs_easy_median": _quantile(
            [float(record["pairwise_cosines"]["malc_vs_easy"]) for record in records],
            0.5,
        ),
        "malc_vs_rms_median": _quantile(
            [float(record["pairwise_cosines"]["malc_vs_rms"]) for record in records],
            0.5,
        ),
        "components": component_summary,
    }


def classify_first_bad_boundary(
    *,
    prototype_geometry: Mapping[str, Any],
    gradient_geometry: Mapping[str, Any],
    microtrajectory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    calibration = prototype_geometry.get("calibration", {})
    effective_count = int(calibration.get("effective_scale_count", 0))
    validity_issues = []
    if int(calibration.get("batch_count", 0)) != 16:
        validity_issues.append("calibration_batch_count")
    heldout = prototype_geometry.get("heldout", {})
    if int(heldout.get("batch_count", 0)) != 24:
        validity_issues.append("heldout_batch_count")
    if effective_count == 0:
        validity_issues.append("no_effective_prototype_scale")
    if int(gradient_geometry.get("batch_count", 0)) != 16:
        validity_issues.append("gradient_batch_count")

    resultant = calibration.get("effective_resultant_median")
    loo_q25 = calibration.get("effective_loo_q25_median")
    cross_median = gradient_geometry.get("cross_batch_malc_median")
    cross_q25 = gradient_geometry.get("cross_batch_malc_q25")
    malc_easy = gradient_geometry.get("malc_vs_easy_median")
    malc_rms = gradient_geometry.get("malc_vs_rms_median")
    components = gradient_geometry.get("components", {})
    rho_malc = components.get("malc", {}).get("retention_median")
    rho_easy = components.get("easy_cls", {}).get("retention_median")
    required = {
        "resultant": resultant,
        "loo_q25": loo_q25,
        "cross_median": cross_median,
        "cross_q25": cross_q25,
        "malc_easy": malc_easy,
        "malc_rms": malc_rms,
        "rho_malc": rho_malc,
        "rho_easy": rho_easy,
    }
    if any(value is None or not math.isfinite(float(value)) for value in required.values()):
        validity_issues.append("missing_or_nonfinite_primary_metric")

    triggers = {name: False for name in BOUNDARY_ORDER}
    if not validity_issues:
        triggers["prototype_incoherence"] = (
            float(resultant) < 0.20 or float(loo_q25) < 0.80
        )
        triggers["cross_batch_malc_conflict"] = (
            float(cross_median) <= 0.0 or float(cross_q25) < -0.10
        )
        triggers["objective_gradient_conflict"] = (
            float(malc_easy) < -0.10 or float(malc_rms) < -0.10
        )
        ratio = float(rho_malc) / max(float(rho_easy), 1e-12)
        triggers["cgr_selective_suppression"] = (
            float(rho_malc) < 0.20
            or (float(rho_easy) >= 0.20 and ratio < 0.50)
        )
        if microtrajectory is not None:
            if int(microtrajectory.get("steps", 0)) != 8:
                validity_issues.append("microtrajectory_step_count")
            d_theta = microtrajectory.get("D_theta")
            d_pattern = microtrajectory.get("D_pattern")
            if (
                d_theta is None
                or d_pattern is None
                or not math.isfinite(float(d_theta))
                or not math.isfinite(float(d_pattern))
            ):
                validity_issues.append("microtrajectory_metric")
            elif not any(triggers[name] for name in BOUNDARY_ORDER[:4]):
                triggers["carrier_update_sink"] = (
                    float(d_theta) < 0.25 and float(d_pattern) < 0.01
                )

    if validity_issues:
        boundary = None
    else:
        boundary = next(
            (name for name in BOUNDARY_ORDER if triggers[name]),
            "unresolved_by_probe",
        )
    return {
        "valid": not validity_issues,
        "validity_issues": validity_issues,
        "first_bad_boundary": boundary,
        "triggered_boundaries": [name for name in BOUNDARY_ORDER if triggers[name]],
        "all_trigger_flags": triggers,
        "boundary_order": list(BOUNDARY_ORDER) + ["unresolved_by_probe"],
        "claim_boundary": "seed0_surrogate_geometry_only_not_fresh_victim_ue",
    }
