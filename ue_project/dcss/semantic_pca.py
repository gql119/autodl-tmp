from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SemanticPCAResult:
    basis: torch.Tensor
    mean: torch.Tensor
    eigenvalues: torch.Tensor
    explained_ratio: torch.Tensor
    cumulative_explained_ratio: torch.Tensor
    sample_count: int


def fit_semantic_pca(
    features: torch.Tensor,
    variance_threshold: float = 0.90,
    fixed_rank: Optional[int] = None,
) -> SemanticPCAResult:
    x = features.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("semantic PCA requires a [N,C] tensor with N >= 2")
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite feature in semantic PCA")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0,1]")

    mean = x.mean(dim=0)
    centered = x - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values.square() / max(1, x.shape[0] - 1)
    total = eigenvalues.sum().clamp_min(torch.finfo(x.dtype).eps)
    ratios = eigenvalues / total
    cumulative = torch.cumsum(ratios, dim=0)
    max_rank = min(x.shape[0] - 1, x.shape[1], vh.shape[0])
    if fixed_rank is None:
        rank = int(torch.searchsorted(cumulative, torch.tensor(variance_threshold, dtype=x.dtype)).item()) + 1
    else:
        rank = int(fixed_rank)
    rank = min(max(1, rank), max_rank)
    basis = vh[:rank].T.contiguous()
    return SemanticPCAResult(basis, mean, eigenvalues, ratios, cumulative, int(x.shape[0]))


def fit_semantic_pca_from_statistics(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    sample_count: int,
    variance_threshold: float = 0.90,
    fixed_rank: Optional[int] = None,
) -> SemanticPCAResult:
    if sample_count < 2:
        raise ValueError("semantic PCA statistics require at least two samples")
    cov = 0.5 * (
        covariance.detach().to(device="cpu", dtype=torch.float64)
        + covariance.detach().to(device="cpu", dtype=torch.float64).T
    )
    values, vectors = torch.linalg.eigh(cov)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0.0)
    vectors = vectors[:, order]
    total = values.sum().clamp_min(torch.finfo(values.dtype).eps)
    ratios = values / total
    cumulative = torch.cumsum(ratios, dim=0)
    max_rank = min(sample_count - 1, cov.shape[0])
    if fixed_rank is None:
        rank = int(torch.searchsorted(cumulative, torch.tensor(variance_threshold, dtype=values.dtype)).item()) + 1
    else:
        rank = int(fixed_rank)
    rank = min(max(1, rank), max_rank)
    return SemanticPCAResult(
        vectors[:, :rank].contiguous(),
        mean.detach().to(device="cpu", dtype=torch.float64),
        values,
        ratios,
        cumulative,
        int(sample_count),
    )
