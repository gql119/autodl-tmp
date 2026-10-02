"""Fixed clean-training semantic coordinates for B2-T/TO; no adaptive changes."""
import hashlib
import math
from pathlib import Path

import torch

from .adaptive_learner import AdaptiveLearner

CLASSES = ('aeroplane bicycle bird boat bottle bus car cat chair cow diningtable '
           'dog horse motorbike person pottedplant sheep sofa train tvmonitor').split()


def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            d.update(b)
    return d.hexdigest()


def select_channels(target, non_target, fraction=.2, specificity=1., eps=1e-12):
    if not 0 < fraction <= 1 or target.shape != non_target.shape:
        raise ValueError('Invalid TSFL coordinates')
    if not torch.isfinite(target).all() or not torch.isfinite(non_target).all():
        raise ValueError('Nonfinite sensitivity')
    if target.mean() <= 0 or non_target.mean() <= 0:
        raise ValueError('Zero sensitivity: evidence is not connected')
    st, sn = target / (target.mean() + eps), non_target / (non_target.mean() + eps)
    score = st - specificity * sn
    # ceil prevents underselecting the requested fraction; stable ties use channel ID.
    indices = torch.argsort(score, descending=True, stable=True)[:max(1, math.ceil(fraction * len(score)))]
    return indices.sort().values, st, sn, score


def gt_evidence(decoded, annotations, image_shape, imgsz):
    """B2 task-alignment formula per real GT; original -> centered letterbox geometry."""
    h, w = image_shape
    r = min(imgsz / h, imgsz / w)
    nh, nw = max(1, round(h*r)), max(1, round(w*r))
    top, left = (imgsz-nh)//2, (imgsz-nw)//2
    result = []
    for ann in annotations:
        cx, cy, bw, bh = ann['bbox']
        gt = decoded.new_tensor([[cx*w*r+left, cy*h*r+top, bw*w*r, bh*h*r]])
        c = int(ann['cls'])
        probs = decoded[0, 4+c].clamp(1e-6, 1.)
        align = probs[:, None].sqrt() * AdaptiveLearner._iou(decoded[0, :4].T, gt).clamp_min(1e-6).pow(6)
        result.append((c, align.topk(min(100, align.shape[0]), dim=0).values.mean()))
    return result


def class_subspace(features, variance=.90, min_rank=4, max_rank_fraction=.25):
    z = torch.stack(features).double()
    n, channels = z.shape
    if n < 2 or not 0 < max_rank_fraction <= 1 or not 0 < variance <= 1:
        raise ValueError('Insufficient class features or invalid PCA configuration')
    mean = z.mean(0)
    _, s, vh = torch.linalg.svd(z-mean, full_matrices=False)
    energy = s.square()
    if energy.sum() <= 0:
        raise ValueError('Degenerate class features')
    cumulative = energy.cumsum(0)/energy.sum()
    needed = int(torch.searchsorted(cumulative, torch.tensor(variance, dtype=cumulative.dtype)))+1
    cap = min(n-1, channels, max(1, math.floor(channels*max_rank_fraction)))
    rank = min(cap, max(min_rank, needed))
    u = vh[:rank].T
    # SVD sign convention for reproducible serialized coordinates.
    pivots = u.abs().argmax(0)
    signs = u[pivots, torch.arange(rank)].sign()
    u = u * signs
    return {'basis': u.float(), 'mean': mean.float(), 'count': n, 'rank': rank,
            'rank_for_variance': needed, 'max_rank': cap,
            'explained_variance': float(cumulative[rank-1]), 'variance_capped': rank < needed}


def normalized_orth(displacement, bases, overlap_weights, worst_k=3, eps=1e-12):
    """Rank worst classes with the same overlap-weighted objective that is optimized."""
    classes = sorted(bases)
    if displacement.ndim != 2 or overlap_weights.ndim != 1 or overlap_weights.shape[0] != displacement.shape[0]:
        raise ValueError('Invalid displacement or overlap-weight dimensions')
    if not torch.isfinite(overlap_weights).all() or (overlap_weights < 0).any() or overlap_weights.sum() <= 0:
        raise ValueError('Overlap weights must be finite, nonnegative, and nonzero')
    ratios = torch.stack([(displacement @ bases[c]).square().sum(-1)/
                          (displacement.square().sum(-1)+eps) for c in classes], -1)
    # Rank by the actual weighted objective. Previously classes were ranked on
    # the unweighted mean, while only the final scalar applied overlap weights.
    weighted_ratios = ratios * overlap_weights[:, None]
    class_conflicts = weighted_ratios.mean(0)
    order = torch.argsort(class_conflicts.detach(), descending=True, stable=True)[:min(worst_k, len(classes))]
    value = class_conflicts[order].mean()
    return value, class_conflicts, [classes[int(i)] for i in order]


def project_bases_to_channels(bases, channel_indices, rank_rtol=1e-6):
    """Represent each full feature subspace in the selected TSFL coordinates.

    Restricting the rows of a basis changes its columns' orthogonality. SVD
    therefore returns an orthonormal basis for the projected span and drops
    directions that vanish (or are numerically negligible) after restriction.
    """
    if not 0 < rank_rtol < 1:
        raise ValueError('rank_rtol must be between 0 and 1')
    projected = {}
    if not bases:
        raise ValueError('At least one semantic subspace is required')
    first_basis = next(iter(bases.values()))
    indices = torch.as_tensor(channel_indices, dtype=torch.long, device=first_basis.device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError('Invalid channel indices')
    for cls, basis in bases.items():
        if basis.ndim != 2 or basis.device != indices.device or int(indices.min()) < 0 or int(indices.max()) >= basis.shape[0]:
            raise ValueError('Invalid channel projection for semantic subspace')
        restricted = basis.index_select(0, indices)
        u, singular_values, _ = torch.linalg.svd(restricted, full_matrices=False)
        if singular_values.numel() == 0 or not torch.isfinite(singular_values).all():
            raise ValueError('Invalid projected semantic subspace')
        threshold = singular_values.max() * rank_rtol
        keep = singular_values > threshold
        if not bool(keep.any()):
            raise ValueError(f'Projected semantic subspace is rank zero for class {cls}')
        projected[int(cls)] = u[:, keep].contiguous()
    return projected


def semantic_identity(cfg, root):
    s = cfg['methods']['legacy_kproto_ret'].get('semantic', {})
    if not s.get('enabled', False):
        return None
    root = Path(root).resolve()
    artifacts = {}
    for name in ('target_specific_channels.pt', 'non_target_subspaces.pt', 'coordinates_identity.json'):
        p = Path(s['artifact_dir']) / name
        if not p.resolve().is_relative_to(root):
            raise ValueError('Semantic artifacts must stay in isolated project')
        artifacts[name] = sha256(p)
    methods = root / 'ue_framework' / 'methods'
    result = {'artifacts': artifacts, 'semantic_code_sha256': sha256(methods/'b2_semantic.py'),
            'coordinate_code_sha256': sha256(methods/'semantic_coordinates.py'),
            'reference_orth_code_sha256': sha256(methods/'orth_reference_design.py'),
            'lambda_orth': float(s['lambda_orth'])}
    if s.get('orth_objective', 'angular') != 'angular':
        result['orth_objective'] = s['orth_objective']
    return result
