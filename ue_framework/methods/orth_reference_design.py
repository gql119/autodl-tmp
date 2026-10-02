"""Unintegrated TO proposal: normalize leakage by frozen clean feature energy.

This is a design helper only. The production trainer still uses normalized_orth;
the angular loss remains an admission metric even if this proposal is integrated.
"""
import torch


def reference_energy_orth(displacement, clean_features, bases, overlap_weights,
                          worst_k=3, eps=1e-12):
    if displacement.ndim != 2 or clean_features.shape != displacement.shape:
        raise ValueError('Clean reference and displacement must have matching NxD shapes')
    if overlap_weights.shape != displacement.shape[:1]:
        raise ValueError('Invalid overlap weight dimensions')
    if not bases or not 1 <= worst_k <= len(bases) or eps <= 0:
        raise ValueError('Invalid worst-k, epsilon, or empty subspaces')
    if not torch.isfinite(clean_features).all() or not torch.isfinite(displacement).all():
        raise ValueError('Nonfinite feature input')
    if not torch.isfinite(overlap_weights).all() or (overlap_weights < 0).any() or overlap_weights.sum() <= 0:
        raise ValueError('Invalid overlap weights')
    energy = clean_features.detach().square().sum(-1)
    if (energy <= eps).any():
        raise ValueError('Clean reference energy is zero or too small')
    classes = sorted(bases)
    ratios = torch.stack([(displacement @ bases[c]).square().sum(-1) /
                          energy for c in classes], -1)
    conflicts = (ratios * overlap_weights[:, None]).mean(0)
    order = torch.argsort(conflicts.detach(), descending=True, stable=True)[:worst_k]
    return conflicts[order].mean(), conflicts, [classes[int(i)] for i in order]
