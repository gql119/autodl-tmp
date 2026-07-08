import torch

from ue_framework.methods.multitrajectory_gain import compute_gradient_leakage_matrix


def test_gradient_leakage_diagnostics_returns_matrix_and_norms():
    layer = torch.nn.Linear(2, 1, bias=False)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x).sum()
    losses = {
        "protected": y,
        "authorized": y * 2.0,
        "shared": -y,
    }
    result = compute_gradient_leakage_matrix(losses, layer.named_parameters())
    assert tuple(result.matrix.shape) == (3, 3)
    assert abs(result.matrix[0, 0].item() - 1.0) < 1.0e-6
    assert result.matrix[0, 1].item() > 0.99
    assert result.matrix[0, 2].item() < -0.99
    assert result.gradient_norms["protected"] > 0.0
    assert result.effective_parameter_count == 1
    assert result.none_gradient_count["protected"] == 0
