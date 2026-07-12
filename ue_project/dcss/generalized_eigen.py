from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeneralizedEigenResult:
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    condition_number: float
    min_regularized_eigenvalue: float
    orthogonality_error: float


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def solve_discriminative_subspace(
    target_second_moment: torch.Tensor,
    non_target_second_moment: torch.Tensor,
    semantic_basis: torch.Tensor,
    rank: int,
    regularization: float = 1e-4,
) -> GeneralizedEigenResult:
    ct = _symmetric(target_second_moment.detach().to(device="cpu", dtype=torch.float64))
    cnt = _symmetric(non_target_second_moment.detach().to(device="cpu", dtype=torch.float64))
    s = semantic_basis.detach().to(device="cpu", dtype=torch.float64)
    if ct.ndim != 2 or ct.shape[0] != ct.shape[1] or cnt.shape != ct.shape:
        raise ValueError("target and non-target moments must be matching square matrices")
    if s.ndim != 2 or s.shape[0] != ct.shape[0]:
        raise ValueError("semantic basis dimension mismatch")
    if rank <= 0 or rank > s.shape[1]:
        raise ValueError(f"rank must be in [1,{s.shape[1]}]")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if not torch.isfinite(ct).all() or not torch.isfinite(cnt).all() or not torch.isfinite(s).all():
        raise FloatingPointError("non-finite generalized eigen input")

    s, _ = torch.linalg.qr(s, mode="reduced")
    a = _symmetric(s.T @ ct @ s)
    b = _symmetric(s.T @ cnt @ s) + regularization * torch.eye(s.shape[1], dtype=s.dtype)
    b_values, b_vectors = torch.linalg.eigh(b)
    min_b = float(b_values.min().item())
    if min_b <= 0 or not torch.isfinite(b_values).all():
        raise RuntimeError(f"regularized non-target matrix is not SPD: min_eigenvalue={min_b:.6e}")
    condition = float((b_values.max() / b_values.min()).item())
    whitening = b_vectors @ torch.diag(b_values.rsqrt()) @ b_vectors.T
    whitened_a = _symmetric(whitening @ a @ whitening)
    values, vectors = torch.linalg.eigh(whitened_a)
    order = torch.argsort(values, descending=True)[:rank]
    q = s @ whitening @ vectors[:, order]
    q, _ = torch.linalg.qr(q, mode="reduced")
    ortho_error = float((q.T @ q - torch.eye(rank, dtype=q.dtype)).abs().max().item())
    if ortho_error > 1e-4:
        raise RuntimeError(f"subspace orthogonality check failed: max_error={ortho_error:.6e}")
    return GeneralizedEigenResult(q, values[order], condition, min_b, ortho_error)


def solve_no_semantic_subspace(
    target_second_moment: torch.Tensor,
    non_target_second_moment: torch.Tensor,
    rank: int,
    regularization: float = 1e-4,
) -> GeneralizedEigenResult:
    """Solve C_t q = lambda (C_nt + mu I) q without a semantic projector."""
    target = target_second_moment.detach()
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target moment must be square")
    identity = torch.eye(target.shape[0], dtype=target.dtype, device=target.device)
    return solve_discriminative_subspace(
        target_second_moment,
        non_target_second_moment,
        identity,
        rank,
        regularization,
    )


def top_target_subspace(target_second_moment: torch.Tensor, rank: int) -> torch.Tensor:
    matrix = _symmetric(target_second_moment.detach().to(device="cpu", dtype=torch.float64))
    values, vectors = torch.linalg.eigh(matrix)
    q = vectors[:, torch.argsort(values, descending=True)[:rank]]
    return torch.linalg.qr(q, mode="reduced").Q


def random_subspace(dimension: int, rank: int, seed: int) -> torch.Tensor:
    if not 0 < rank <= dimension:
        raise ValueError("rank must be positive and no larger than dimension")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn((dimension, rank), generator=generator, dtype=torch.float64)
    return torch.linalg.qr(values, mode="reduced").Q
