from typing import Dict

import torch


def projection_energy(second_moment: torch.Tensor, basis: torch.Tensor) -> float:
    c = second_moment.to(dtype=torch.float64, device="cpu")
    q = basis.to(dtype=torch.float64, device="cpu")
    return float(torch.trace(q.T @ c @ q).item())


def selectivity_ratio(target_moment: torch.Tensor, non_target_moment: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> float:
    return projection_energy(target_moment, basis) / (projection_energy(non_target_moment, basis) + eps)


def semantic_overlap(basis: torch.Tensor, semantic_basis: torch.Tensor) -> float:
    q = basis.to(dtype=torch.float64, device="cpu")
    s = semantic_basis.to(dtype=torch.float64, device="cpu")
    return float((s.T @ q).square().sum().item() / q.shape[1])


def projection_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    qa = first.to(dtype=torch.float64, device="cpu")
    qb = second.to(dtype=torch.float64, device="cpu")
    return float((qa.T @ qb).square().sum().item() / min(qa.shape[1], qb.shape[1]))


def principal_angles_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    qa = torch.linalg.qr(first.to(dtype=torch.float64, device="cpu"), mode="reduced").Q
    qb = torch.linalg.qr(second.to(dtype=torch.float64, device="cpu"), mode="reduced").Q
    singular_values = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(singular_values))


def random_baseline_summary(values: torch.Tensor, dcss_value: float) -> Dict[str, float]:
    values = values.to(dtype=torch.float64, device="cpu")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0
    z_score = (float(dcss_value) - mean) / std if std > 0 else float("inf")
    return {"random_mean": mean, "random_std": std, "dcss_value": float(dcss_value), "z_score": z_score}
