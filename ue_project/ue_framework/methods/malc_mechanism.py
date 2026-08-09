from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .malc import MALCResult


@dataclass(frozen=True)
class MALCMechanismBatch:
    malc: MALCResult
    cgr_attack_retention: float
    cgr_max_projected_row_dot: float
    cgr_selected_mode: str
    non_target_target_energy_ratio: float
    box_residual_energy: float
    size_groups: tuple[str, ...]
    cooccur_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        total = self.malc.total_instance_count
        if len(self.size_groups) != total or len(self.cooccur_flags) != total:
            raise ValueError("Mechanism group labels must match MALC instances.")
        if self.cgr_selected_mode not in {
            "target",
            "projected_target",
            "repair_only",
            "skip",
        }:
            raise ValueError("Unknown CGR selected mode.")
        values = (
            self.cgr_attack_retention,
            self.cgr_max_projected_row_dot,
            self.non_target_target_energy_ratio,
            self.box_residual_energy,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Mechanism batch diagnostics must be finite and non-negative.")


def _finite_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty((0,), dtype=torch.float64)
    tensor = torch.cat(
        [value.detach().to(device="cpu", dtype=torch.float64) for value in values]
    )
    return tensor[torch.isfinite(tensor)]


def _median(values: Sequence[float]) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("Mechanism metric sequence must be finite and non-empty.")
    return float(torch.quantile(tensor, 0.5))


def _group_summary(
    values: list[tuple[str, float, float]],
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for name, cosine, log_energy in values:
        groups.setdefault(name, []).append((cosine, log_energy))
    summary = {}
    for name, records in sorted(groups.items()):
        cosines = torch.tensor([item[0] for item in records], dtype=torch.float64)
        log_energies = torch.tensor(
            [item[1] for item in records], dtype=torch.float64
        )
        valid_cosines = cosines[torch.isfinite(cosines)]
        valid_energy = log_energies[torch.isfinite(log_energies)]
        median_cosine: float | None = (
            float(torch.quantile(valid_cosines, 0.5))
            if valid_cosines.numel()
            else None
        )
        if valid_energy.numel():
            center = torch.quantile(valid_energy, 0.5)
            energy_mad: float | None = float(
                torch.quantile((valid_energy - center).abs(), 0.5)
            )
        else:
            energy_mad = None
        summary[name] = {
            "count": len(records),
            "coverage": float(valid_cosines.numel()) / max(len(records), 1),
            "residual_cosine_median": median_cosine,
            "log_energy_mad": energy_mad,
        }
    return summary


def aggregate_malc_mechanism_batches(
    batches: Sequence[MALCMechanismBatch],
    *,
    split: str,
) -> dict[str, Any]:
    if split != "heldout":
        raise ValueError("Mechanism gate aggregation requires split='heldout'.")
    if not batches:
        raise ValueError("Mechanism gate requires held-out batches.")
    scale_count = len(batches[0].malc.per_scale_valid_count)
    if scale_count == 0 or any(
        len(batch.malc.per_scale_valid_count) != scale_count for batch in batches
    ):
        raise ValueError("Mechanism batches must share a non-empty scale layout.")

    cosine_values = _finite_values(
        [batch.malc.per_instance_cosine for batch in batches]
    )
    energy_values = _finite_values(
        [batch.malc.per_instance_log_energy for batch in batches]
    )
    if not cosine_values.numel() or not energy_values.numel():
        raise ValueError("Mechanism gate requires finite residual diagnostics.")
    energy_center = torch.quantile(energy_values, 0.5)
    log_energy_mad = float(
        torch.quantile((energy_values - energy_center).abs(), 0.5)
    )

    scale_validity_frequency = tuple(
        sum(batch.malc.per_scale_valid_count[scale] > 0 for batch in batches)
        / len(batches)
        for scale in range(scale_count)
    )
    group_rows: list[tuple[str, float, float]] = []
    for batch in batches:
        cosines = batch.malc.per_instance_cosine.detach().cpu().tolist()
        energies = batch.malc.per_instance_log_energy.detach().cpu().tolist()
        for size, cooccur, cosine, energy in zip(
            batch.size_groups,
            batch.cooccur_flags,
            cosines,
            energies,
        ):
            group_rows.append((f"size:{size}", float(cosine), float(energy)))
            group_rows.append((
                "context:cooccur" if cooccur else "context:person_only",
                float(cosine),
                float(energy),
            ))

    return {
        "split": "heldout",
        "heldout_batch_count": len(batches),
        "residual_cosine_median": float(torch.quantile(cosine_values, 0.5)),
        "residual_cosine_q25": float(torch.quantile(cosine_values, 0.25)),
        "log_energy_mad": log_energy_mad,
        "valid_instance_coverage": sum(
            batch.malc.valid_instance_coverage for batch in batches
        ) / len(batches),
        "zero_norm_ratio": sum(batch.malc.zero_norm_ratio for batch in batches)
        / len(batches),
        "floor_pass_ratio": sum(batch.malc.floor_pass_ratio for batch in batches)
        / len(batches),
        "scale_validity_frequency": scale_validity_frequency,
        "valid_scale_count_at_0_80": sum(
            value >= 0.80 for value in scale_validity_frequency
        ),
        "cgr_max_projected_row_dot": max(
            batch.cgr_max_projected_row_dot for batch in batches
        ),
        "cgr_attack_retention_median": _median(
            [batch.cgr_attack_retention for batch in batches]
        ),
        "cgr_repair_skip_ratio": sum(
            batch.cgr_selected_mode in {"repair_only", "skip"}
            for batch in batches
        ) / len(batches),
        "non_target_target_energy_ratio_median": _median(
            [batch.non_target_target_energy_ratio for batch in batches]
        ),
        "box_residual_energy_median": _median(
            [batch.box_residual_energy for batch in batches]
        ),
        "groups": _group_summary(group_rows),
    }


def _flatten_mapping(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            flattened.update(_flatten_mapping(child, prefix=path))
        else:
            flattened[path] = child
    return flattened


def assert_matched_mechanism_configs(
    a0_config: Mapping[str, Any],
    a1_config: Mapping[str, Any],
    *,
    malc_key: str = "method.enable_malc",
) -> None:
    flat_a0 = _flatten_mapping(a0_config)
    flat_a1 = _flatten_mapping(a1_config)
    if set(flat_a0) != set(flat_a1):
        raise ValueError("A0/A1 mechanism config keys differ.")
    differences = {
        key for key in flat_a0 if flat_a0[key] != flat_a1[key]
    }
    if differences != {malc_key}:
        raise ValueError(
            f"A0/A1 may differ only at {malc_key}; got {sorted(differences)}."
        )
    if bool(flat_a0[malc_key]) or not bool(flat_a1[malc_key]):
        raise ValueError("A0 must disable MALC and A1 must enable MALC.")


def evaluate_malc_mechanism_gate(
    a0: Mapping[str, Any],
    a1: Mapping[str, Any],
) -> dict[str, Any]:
    success = {
        "cosine_gain": (
            float(a1["residual_cosine_median"])
            - float(a0["residual_cosine_median"])
        ) >= 0.10,
        "cosine_q25_positive": float(a1["residual_cosine_q25"]) > 0.0,
        "log_energy_mad": float(a1["log_energy_mad"])
        <= 0.90 * float(a0["log_energy_mad"]),
        "coverage": float(a1["valid_instance_coverage"]) >= 0.80,
        "zero_norm": float(a1["zero_norm_ratio"]) <= 0.20,
        "floor_pass": float(a1["floor_pass_ratio"]) >= 0.80,
        "scale_validity": int(a1["valid_scale_count_at_0_80"]) >= 2,
        "cgr_orthogonality": float(a1["cgr_max_projected_row_dot"])
        <= 1e-5,
        "cgr_attack_retention": float(a1["cgr_attack_retention_median"])
        >= 0.20,
        "cgr_repair_skip": float(a1["cgr_repair_skip_ratio"]) < 0.50,
    }
    failure = {
        "non_target_target_energy_leakage": float(
            a1["non_target_target_energy_ratio_median"]
        ) > 1.25 * float(a0["non_target_target_energy_ratio_median"]),
        "box_residual_energy_leakage": float(a1["box_residual_energy_median"])
        > 1.25 * float(a0["box_residual_energy_median"]),
    }
    return {
        "success_signals": success,
        "failure_signals": failure,
        "pass": all(success.values()) and not any(failure.values()),
        "allow_fresh_victim": all(success.values()) and not any(failure.values()),
    }


def write_malc_mechanism_report(
    output_path: str | Path,
    *,
    a0: Mapping[str, Any],
    a1: Mapping[str, Any],
    gate: Mapping[str, Any],
    split_hash: str,
) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Mechanism report path must end in .json.")
    if path.exists():
        raise FileExistsError(f"Mechanism report already exists: {path}")
    if not split_hash:
        raise ValueError("Mechanism report split_hash must be non-empty.")
    payload = {
        "schema_version": 1,
        "evidence_scope": "heldout_mechanism_only_not_fresh_victim_ue",
        "split_hash": split_hash,
        "A0": dict(a0),
        "A1": dict(a1),
        "gate": dict(gate),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
