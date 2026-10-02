"""Strict TSFL selection and Safe-BG projection helpers for B2-TO-Safe."""
from __future__ import annotations

import math

import torch


def select_strict_tsfl(target_sensitivity, class_sensitivity, candidate_fraction=0.30,
                       safety_quantile=0.50, worst_k=3, selected_fraction=0.15,
                       min_selected=4, eps=1e-12):
    """Apply target-gate, non-target-gate, then stable target/NT ratio ranking."""
    if target_sensitivity.ndim != 1 or not class_sensitivity:
        raise ValueError("Strict TSFL expects C target scores and per-class NT scores")
    if not 0 < candidate_fraction <= 1 or not 0 <= safety_quantile <= 1:
        raise ValueError("Invalid Strict TSFL gate settings")
    if not 0 < selected_fraction <= 1 or not 1 <= worst_k <= len(class_sensitivity):
        raise ValueError("Invalid Strict TSFL ranking settings")
    classes = sorted(class_sensitivity)
    values = torch.stack([class_sensitivity[c] for c in classes]).to(target_sensitivity)
    if values.shape[1:] != target_sensitivity.shape or not torch.isfinite(values).all():
        raise ValueError("Invalid or nonfinite class-wise sensitivity tensor")
    if not torch.isfinite(target_sensitivity).all() or target_sensitivity.mean() <= 0:
        raise ValueError("Invalid target sensitivity")
    if (values.mean(1) <= 0).any():
        raise ValueError("A non-target class has zero sensitivity")
    st_hat = target_sensitivity / (target_sensitivity.mean() + eps)
    nt_class_hat = values / (values.mean(1, keepdim=True) + eps)
    k = min(worst_k, len(classes))
    nt_worst_hat = nt_class_hat.topk(k, dim=0, largest=True, sorted=True).values.mean(0)
    target_threshold = torch.quantile(st_hat, 1.0 - candidate_fraction)
    nt_threshold = torch.quantile(nt_worst_hat, safety_quantile)
    target_mask = st_hat >= target_threshold
    safe_mask = nt_worst_hat <= nt_threshold
    eligible = target_mask & safe_mask
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_idx.numel() == 0:
        raise ValueError("Strict TSFL gates leave no eligible channels")
    ratio = st_hat / (nt_worst_hat + eps)
    requested = max(1, math.ceil(selected_fraction * target_sensitivity.numel()))
    count = min(requested, int(eligible_idx.numel()))
    order = torch.argsort(ratio[eligible_idx], descending=True, stable=True)
    indices = eligible_idx[order[:count]].sort().values
    if indices.numel() < min_selected:
        raise ValueError(f"Strict TSFL selected {indices.numel()} channels; minimum is {min_selected}")
    selected = torch.zeros_like(target_mask)
    selected[indices] = True
    unselected = ~selected
    conflicts = {
        int(channel): [classes[i] for i in torch.argsort(nt_class_hat[:, channel], descending=True, stable=True)[:k].tolist()]
        for channel in indices.tolist()
    }
    summary = {
        "total_channels": int(target_sensitivity.numel()),
        "target_candidates": int(target_mask.sum()),
        "safe_candidates": int(eligible.sum()),
        "final_selected_channels": int(indices.numel()),
        "final_selected_fraction": float(indices.numel() / target_sensitivity.numel()),
        "target_candidate_fraction": float(candidate_fraction),
        "target_gate_quantile": float(1.0 - candidate_fraction),
        "target_gate_threshold": float(target_threshold),
        "nt_safety_quantile": float(safety_quantile),
        "nt_safety_threshold": float(nt_threshold),
        "sensitivity_worst_k": int(k),
        "selected_channel_fraction": float(selected_fraction),
        "mean_S_t_selected": float(st_hat[selected].mean()),
        "mean_S_t_unselected": float(st_hat[unselected].mean()),
        "mean_S_nt_worst_selected": float(nt_worst_hat[selected].mean()),
        "mean_S_nt_worst_unselected": float(nt_worst_hat[unselected].mean()),
        "mean_ratio_selected": float(ratio[selected].mean()),
        "mean_ratio_unselected": float(ratio[unselected].mean()) if unselected.any() else None,
        "selected_channel_indices": indices.tolist(),
        "selected_channel_top_conflicting_classes": conflicts,
    }
    return {
        "indices": indices,
        "S_target_normalized": st_hat,
        "S_nt_class_normalized": nt_class_hat,
        "S_nt_worst_normalized": nt_worst_hat,
        "ratio": ratio,
        "target_candidate_mask": target_mask,
        "safe_candidate_mask": eligible,
        "summary": summary,
        "classes": classes,
    }


def safe_background_direction(target_clean, background, class_bases, active_k=3,
                              projector_rtol=1e-6, eps=1e-12):
    """Project each clean BG direction outside its top-k active NT subspaces.

    Inputs are frozen selected-space vectors [N,D]. The function intentionally
    does not fallback when projection removes most of a direction; it reports
    that collapse through the returned retention and collapse mask.
    """
    if target_clean.ndim != 2 or background.shape != target_clean.shape or not class_bases:
        raise ValueError("Safe-BG expects matching [N,D] inputs and class bases")
    classes = sorted(class_bases)
    if not 1 <= active_k <= len(classes):
        raise ValueError("Invalid Safe-BG active_k")
    if projector_rtol <= 0 or eps <= 0:
        raise ValueError("Invalid Safe-BG numerical tolerance")
    d_bg = (background - target_clean).detach()
    if not torch.isfinite(d_bg).all():
        raise ValueError("Nonfinite raw background direction")
    directions, retentions, residual_norms, removed, active_rows, ratios_rows, active_ratio_rows, orth_cosines = [], [], [], [], [], [], [], []
    for d in d_bg:
        norm_bg = d.norm()
        if norm_bg <= eps:
            raise ValueError("Raw background direction has zero norm")
        ratios = []
        for cls in classes:
            basis = class_bases[cls].to(device=d.device, dtype=d.dtype)
            if basis.ndim != 2 or basis.shape[0] != d.numel() or not torch.isfinite(basis).all():
                raise ValueError(f"Invalid selected-space basis for class {cls}")
            proj = basis @ (basis.transpose(0, 1) @ d)
            ratios.append(proj.square().sum() / (norm_bg.square() + eps))
        ratios_t = torch.stack(ratios)
        # Classes are already sorted, so stable descending argsort also gives
        # deterministic class-ID tie breaking without a device-to-host sync.
        order = torch.argsort(ratios_t, descending=True, stable=True)[:active_k]
        active = [classes[i] for i in order.tolist()]
        active_ratios = ratios_t.index_select(0, order)
        concatenated = torch.cat([class_bases[c].to(device=d.device, dtype=d.dtype) for c in active], dim=1)
        u, singular, _ = torch.linalg.svd(concatenated, full_matrices=False)
        if singular.numel() == 0 or not torch.isfinite(singular).all():
            raise ValueError("Unstable active non-target projector SVD")
        keep = singular > singular.max() * projector_rtol
        if not keep.any():
            raise ValueError("Active non-target projector has zero rank")
        u_active = u[:, keep]
        projected = u_active @ (u_active.transpose(0, 1) @ d)
        residual = d - projected
        norm_safe = residual.norm()
        retention = norm_safe / (norm_bg + eps)
        if not torch.isfinite(retention):
            raise ValueError("Nonfinite Safe-BG direction retention")
        residual_norms.append(norm_safe)
        scaled = torch.where(norm_safe > eps, residual * (norm_bg / norm_safe.clamp_min(eps)), torch.zeros_like(residual))
        orth_cos = (scaled @ projected) / (scaled.norm() * projected.norm()).clamp_min(eps)
        directions.append(scaled)
        retentions.append(retention)
        removed.append(projected.norm())
        active_rows.append(active)
        ratios_rows.append(ratios_t)
        active_ratio_rows.append(active_ratios)
        orth_cosines.append(orth_cos)
    return target_clean.detach() + torch.stack(directions), {
        "d_bg_norm": torch.stack([d.norm() for d in d_bg]),
        "d_safe_norm": torch.stack(residual_norms),
        "d_safe_scaled_norm": torch.stack([d.norm() for d in directions]),
        "retention": torch.stack(retentions),
        "removed_norm": torch.stack(removed),
        "removed_ratio": torch.stack(removed) / (torch.stack([d.norm() for d in d_bg]) + eps),
        "active_classes": active_rows,
        "projection_ratios": torch.stack(ratios_rows),
        "active_projection_ratios": torch.stack(active_ratio_rows),
        "orth_cosine": torch.stack(orth_cosines),
        "collapsed": torch.stack(retentions) < 0.05,
    }
