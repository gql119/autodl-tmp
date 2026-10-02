import torch

from ue_framework.methods.safebg import safe_background_direction, select_strict_tsfl


def test_strict_tsfl_applies_both_gates_before_ratio_ranking_deterministically():
    target = torch.arange(1, 41, dtype=torch.float64)
    nt = {c: torch.arange(40, 0, -1, dtype=torch.float64) for c in range(19)}
    first = select_strict_tsfl(target, nt)
    second = select_strict_tsfl(target.clone(), {c: v.clone() for c, v in nt.items()})
    assert torch.equal(first["indices"], second["indices"])
    assert first["summary"]["target_candidates"] >= 12
    assert first["summary"]["final_selected_channels"] >= 4
    assert first["summary"]["mean_S_t_selected"] > first["summary"]["mean_S_t_unselected"]
    assert first["summary"]["mean_S_nt_worst_selected"] < first["summary"]["mean_S_nt_worst_unselected"]
    assert first["summary"]["final_selected_channels"] <= first["summary"]["safe_candidates"]


def test_safe_background_is_orthogonal_and_preserves_noncollapsed_direction_norm():
    dtype = torch.float64
    clean = torch.zeros((1, 4), dtype=dtype)
    background = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=dtype)
    bases = {0: torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=dtype),
             1: torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=dtype),
             2: torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=dtype),
             3: torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=dtype)}
    safe, diag = safe_background_direction(clean, background, bases, active_k=3)
    assert diag["active_classes"] == [[0, 1, 2]]
    assert torch.allclose(diag["retention"], torch.tensor([0.5], dtype=dtype), atol=1e-10)
    assert torch.allclose(diag["d_safe_norm"], torch.tensor([1.0], dtype=dtype), atol=1e-10)
    assert torch.allclose(diag["d_safe_scaled_norm"], background.norm(dim=1), atol=1e-10)
    assert torch.allclose(safe.norm(dim=1), background.norm(dim=1), atol=1e-10)
    assert abs(float(diag["orth_cosine"][0])) < 1e-10
    poison = torch.tensor([[0.2, -0.1, 0.3, 0.4]], dtype=dtype, requires_grad=True)
    loss = 1.0 - torch.nn.functional.cosine_similarity(poison, safe.detach(), dim=-1).mean()
    grad = torch.autograd.grad(loss, poison)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
