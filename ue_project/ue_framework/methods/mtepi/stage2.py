from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateLayer:
    name: str
    module_path: str
    role: str
    stride: int
    channels: int
    height: int
    width: int
    hook_position: str = "module_output_after_bn_activation"
    after_bn: bool = True
    after_activation: bool = True
    shared_downstream: bool = True


@dataclass(frozen=True)
class ChannelScoreThresholds:
    d_t_min: float
    d_a_max: float
    d_s_max: float
    protected_positive_ratio_min: float
    min_samples: int
    clean_energy_min: float
    beta_a: float = 1.0
    beta_s: float = 1.0
    ratio_lambda_s: float = 1.0
    ratio_eps: float = 1.0e-8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boxes_to_feature_mask(
    boxes_xywhn: torch.Tensor,
    feature_hw: tuple[int, int],
    *,
    exclude_boxes_xywhn: torch.Tensor | None = None,
    ambiguous_boxes_xywhn: torch.Tensor | None = None,
    soft: bool = False,
) -> torch.Tensor:
    """Map normalized GT boxes to a feature-space spatial mask.

    Boxes use normalized xywh coordinates. The mask is binary by default and is
    aligned by cell centers, which keeps boundary behavior deterministic.
    """

    h, w = int(feature_hw[0]), int(feature_hw[1])
    device = boxes_xywhn.device
    dtype = torch.float32
    yy = (torch.arange(h, device=device, dtype=dtype) + 0.5) / max(float(h), 1.0)
    xx = (torch.arange(w, device=device, dtype=dtype) + 0.5) / max(float(w), 1.0)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    mask = torch.zeros((h, w), device=device, dtype=dtype)
    for box in boxes_xywhn.reshape(-1, 4):
        cx, cy, bw, bh = [float(v) for v in box.detach().cpu().tolist()]
        x1, x2 = cx - 0.5 * bw, cx + 0.5 * bw
        y1, y2 = cy - 0.5 * bh, cy + 0.5 * bh
        inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
        mask = torch.maximum(mask, inside.to(dtype))

    for boxes in [exclude_boxes_xywhn, ambiguous_boxes_xywhn]:
        if boxes is None or boxes.numel() == 0:
            continue
        excl = boxes_to_feature_mask(boxes.to(device), feature_hw, soft=soft)
        mask = mask * (1.0 - excl.clamp(0.0, 1.0))

    if soft:
        return mask.clamp(0.0, 1.0)
    return (mask > 0.0).to(dtype)


def localized_channel_ablation(features: torch.Tensor, channel_indices: Sequence[int], spatial_mask: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"features must be [B,C,H,W], got {tuple(features.shape)}")
    mask = spatial_mask.to(device=features.device, dtype=features.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(features.shape[0], -1, -1)
    if tuple(mask.shape[-2:]) != tuple(features.shape[-2:]):
        raise ValueError(f"mask spatial shape {tuple(mask.shape[-2:])} does not match features {tuple(features.shape[-2:])}")
    out = features.clone()
    for channel in channel_indices:
        c = int(channel)
        if c < 0 or c >= features.shape[1]:
            raise IndexError(f"channel {c} outside [0,{features.shape[1]})")
        out[:, c, :, :] = out[:, c, :, :] * (1.0 - mask)
    return out


@contextmanager
def channel_ablation_hook(module: torch.nn.Module, channel_indices: Sequence[int], spatial_mask: torch.Tensor) -> Iterator[None]:
    def _hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            return output
        return localized_channel_ablation(output, channel_indices, spatial_mask)

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def ablation_delta(base: Mapping[str, float], ablated: Mapping[str, float]) -> Dict[str, float]:
    return {
        "Delta_t": float(ablated.get("protected", 0.0)) - float(base.get("protected", 0.0)),
        "Delta_a": float(ablated.get("authorized", 0.0)) - float(base.get("authorized", 0.0)),
        "Delta_s": float(ablated.get("shared", 0.0)) - float(base.get("shared", 0.0)),
    }


def _group_rows(rows: Iterable[Mapping[str, float]]) -> Dict[tuple[str, int], List[Mapping[str, float]]]:
    grouped: Dict[tuple[str, int], List[Mapping[str, float]]] = {}
    for row in rows:
        key = (str(row["layer"]), int(row["channel"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def constraint_first_rank(rows: Iterable[Mapping[str, float]], thresholds: ChannelScoreThresholds) -> List[Dict[str, float]]:
    ranked = []
    for (layer, channel), group in _group_rows(rows).items():
        dt = np.array([float(r["Delta_t"]) for r in group], dtype=np.float64)
        da = np.array([abs(float(r["Delta_a"])) for r in group], dtype=np.float64)
        ds = np.array([abs(float(r["Delta_s"])) for r in group], dtype=np.float64)
        energy = np.array([float(r.get("clean_energy", 0.0)) for r in group], dtype=np.float64)
        n = int(len(group))
        mean_dt = float(dt.mean()) if n else 0.0
        mean_abs_da = float(da.mean()) if n else 0.0
        mean_abs_ds = float(ds.mean()) if n else 0.0
        pos_ratio = float((dt > 0.0).mean()) if n else 0.0
        mean_energy = float(energy.mean()) if n else 0.0
        hard_pass = (
            mean_dt > thresholds.d_t_min
            and mean_abs_da < thresholds.d_a_max
            and mean_abs_ds < thresholds.d_s_max
            and pos_ratio >= thresholds.protected_positive_ratio_min
            and n >= thresholds.min_samples
            and mean_energy >= thresholds.clean_energy_min
        )
        diff_score = mean_dt - thresholds.beta_a * mean_abs_da - thresholds.beta_s * mean_abs_ds
        ratio_score = float(np.maximum(dt, 0.0).mean()) / (
            mean_abs_da + thresholds.ratio_lambda_s * mean_abs_ds + thresholds.ratio_eps
        )
        ranked.append(
            {
                "layer": layer,
                "channel": channel,
                "sample_count": n,
                "mean_Delta_t": mean_dt,
                "median_Delta_t": float(np.median(dt)) if n else 0.0,
                "std_Delta_t": float(dt.std()) if n else 0.0,
                "mean_abs_Delta_a": mean_abs_da,
                "mean_abs_Delta_s": mean_abs_ds,
                "protected_positive_ratio": pos_ratio,
                "clean_energy_mean": mean_energy,
                "hard_pass": bool(hard_pass),
                "diff_score": float(diff_score),
                "ratio_score_log_only": float(ratio_score),
                "channel_type": classify_channel(mean_dt, mean_abs_da, mean_abs_ds, pos_ratio, thresholds),
            }
        )
    ranked.sort(key=lambda r: (not r["hard_pass"], -float(r["diff_score"]), str(r["layer"]), int(r["channel"])))
    return ranked


def classify_channel(mean_dt: float, mean_abs_da: float, mean_abs_ds: float, pos_ratio: float, thresholds: ChannelScoreThresholds) -> str:
    if (
        mean_dt > thresholds.d_t_min
        and mean_abs_da < thresholds.d_a_max
        and mean_abs_ds < thresholds.d_s_max
        and pos_ratio >= thresholds.protected_positive_ratio_min
    ):
        return "target-selective"
    if mean_abs_da >= thresholds.d_a_max and mean_dt <= thresholds.d_t_min:
        return "authorized-selective"
    if mean_abs_ds >= thresholds.d_s_max or (mean_dt > thresholds.d_t_min and mean_abs_da >= thresholds.d_a_max):
        return "shared"
    return "inactive/noisy"


def jaccard_overlap(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def build_consensus_pathways(rankings_by_checkpoint: Mapping[str, Sequence[Mapping[str, float]]], top_k: int) -> List[Dict[str, float]]:
    checkpoint_names = list(rankings_by_checkpoint.keys())
    if not checkpoint_names:
        return []
    sets = []
    row_by_key: Dict[tuple[str, int], Mapping[str, float]] = {}
    for rows in rankings_by_checkpoint.values():
        selected = [r for r in rows if bool(r.get("hard_pass", False))][: int(top_k)]
        keys = {(str(r["layer"]), int(r["channel"])) for r in selected}
        sets.append(keys)
        for r in selected:
            row_by_key[(str(r["layer"]), int(r["channel"]))] = r
    consensus_keys = set.intersection(*sets) if sets else set()
    return [
        {
            "layer": layer,
            "channel": channel,
            "checkpoint_support_count": len(checkpoint_names),
            "diff_score": float(row_by_key[(layer, channel)].get("diff_score", 0.0)),
        }
        for layer, channel in sorted(consensus_keys)
    ]


def cross_checkpoint_transfer_matrix(rankings_by_checkpoint: Mapping[str, Sequence[Mapping[str, float]]], top_k: int) -> List[Dict[str, float]]:
    names = list(rankings_by_checkpoint.keys())
    selected = {}
    for name, rows_in in rankings_by_checkpoint.items():
        passed = [r for r in rows_in if bool(r.get("hard_pass", False))][: int(top_k)]
        selected[name] = {(str(r["layer"]), int(r["channel"])) for r in passed}
    rows = []
    for src in names:
        for dst in names:
            rows.append({"source": src, "target": dst, "topk_jaccard": jaccard_overlap(selected[src], selected[dst])})
    return rows


def bootstrap_mean_ci(values: Sequence[float], num_bootstrap: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(arr, size=(int(num_bootstrap), arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "n": int(arr.size),
    }


def build_checkpoint_manifest(checkpoints: Sequence[Mapping[str, object]], root: Path) -> Dict:
    entries = []
    valid = []
    for item in checkpoints:
        path = root / str(item["path"])
        exists = path.exists()
        entry = {
            "name": str(item["name"]),
            "role": str(item.get("role", "unknown")),
            "path": str(item["path"]),
            "exists": bool(exists),
            "sha256": sha256_file(path) if exists and path.is_file() else "",
            "training_run_id": str(item.get("training_run_id", "")),
            "initialization_checksum": str(item.get("initialization_checksum", "")),
            "class_space": str(item.get("class_space", "")),
            "architecture": str(item.get("architecture", "")),
            "epoch": item.get("epoch", "unknown"),
            "manifest": str(item.get("manifest", "")),
        }
        entry["legal_role"] = bool(exists and entry["role"] in {"early", "middle", "late"} and entry["class_space"] == "VOC20")
        entries.append(entry)
        if entry["legal_role"]:
            valid.append(entry)

    roles = {e["role"] for e in valid}
    run_ids = {e["training_run_id"] for e in valid}
    init_ids = {e["initialization_checksum"] for e in valid}
    manifests = {e["manifest"] for e in valid}
    archs = {e["architecture"] for e in valid}
    legal_same_trajectory = (
        {"early", "middle", "late"}.issubset(roles)
        and len(run_ids) == 1
        and len(init_ids) == 1
        and len(manifests) == 1
        and len(archs) == 1
        and "" not in run_ids
        and "" not in init_ids
        and "" not in manifests
        and "" not in archs
    )
    reasons = []
    if not {"early", "middle", "late"}.issubset(roles):
        reasons.append("missing legal early/middle/late checkpoints from one continuous run")
    if len(run_ids) != 1 or "" in run_ids:
        reasons.append("training_run_id is absent or inconsistent")
    if len(init_ids) != 1 or "" in init_ids:
        reasons.append("initialization_checksum is absent or inconsistent")
    if len(manifests) != 1 or "" in manifests:
        reasons.append("training manifest is absent or inconsistent")
    if len(archs) != 1 or "" in archs:
        reasons.append("architecture is absent or inconsistent")
    return {
        "checkpoints": entries,
        "legal_same_trajectory": bool(legal_same_trajectory),
        "valid_checkpoint_count": int(len(valid)),
        "roles_present": sorted(roles),
        "failure_reasons": reasons,
    }


def stage2_gate(
    checkpoint_manifest: Mapping[str, object],
    ranked_rows: Sequence[Mapping[str, object]],
    topk_ap_rows: Sequence[Mapping[str, object]],
    transfer_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    consensus: Sequence[Mapping[str, object]],
    *,
    min_target_selective_channels: int,
    min_authorized_retention: float,
    min_topk_protected_drop: float,
    min_transfer_jaccard: float,
) -> Dict[str, object]:
    reasons = []
    if not bool(checkpoint_manifest.get("legal_same_trajectory", False)):
        reasons.append("no legal same-trajectory early/middle/late checkpoint set")
    target_rows = [r for r in ranked_rows if bool(r.get("hard_pass", False)) and r.get("channel_type") == "target-selective"]
    if len(target_rows) < int(min_target_selective_channels):
        reasons.append("insufficient target-selective functional channels")
    if not consensus:
        reasons.append("no legal consensus pathway")
    if topk_ap_rows:
        best_drop = max(float(r.get("protected_ap_drop", 0.0)) for r in topk_ap_rows)
        best_retention = max(float(r.get("authorized_retention", 0.0)) for r in topk_ap_rows)
        if best_drop < float(min_topk_protected_drop):
            reasons.append("Top-k ablation did not produce required protected AP drop")
        if best_retention < float(min_authorized_retention):
            reasons.append("Top-k ablation did not retain authorized mAP")
    else:
        reasons.append("Top-k AP ablation curve not available")
    if transfer_rows:
        off_diag = [float(r["topk_jaccard"]) for r in transfer_rows if r["source"] != r["target"]]
        if off_diag and min(off_diag) < float(min_transfer_jaccard):
            reasons.append("cross-checkpoint pathway overlap below threshold")
    else:
        reasons.append("cross-checkpoint transfer matrix not available")
    if not bootstrap_rows:
        reasons.append("bootstrap confidence intervals not available")
    return {"gate": "PASS" if not reasons else "FAIL", "reasons": reasons}
