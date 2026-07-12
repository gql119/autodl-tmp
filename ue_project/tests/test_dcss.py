import os
import tempfile

import torch
import yaml

from dcss.diagnostics import principal_angles_degrees, projection_similarity, selectivity_ratio, semantic_overlap
from dcss.feature_hooks import FeatureHookBank
from dcss.generalized_eigen import random_subspace, solve_discriminative_subspace, solve_no_semantic_subspace
from dcss.resume import apply_relative_overrides, build_resume_run_dir, diagnostic_gate, mean_finite_metric
from dcss.losses import dcss_stage1_loss, non_target_leakage, subspace_energies
from dcss.semantic_pca import fit_semantic_pca
from dcss.statistics import RunningCovariance
from dcss.subspace_io import load_subspaces, save_subspaces
from dcss.stage15 import constrained_direction, gradient_component_stats, gradient_cosine, object_aligned_warp
from dcss.unit_partition import partition_tal_units


def test_feature_gradient_shape():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 5, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(5, 2, 1),
    )
    hooks = FeatureHookBank(model, ["0"])
    image = torch.randn(2, 3, 8, 8, requires_grad=True)
    output = model(image)
    feature = hooks.outputs["0"]
    gradient = torch.autograd.grad(output.square().mean(), feature)[0]
    hooks.close()
    assert gradient.shape == feature.shape == (2, 5, 8, 8)
    assert gradient.abs().sum() > 0


def test_target_non_target_partition():
    features = {
        "p3": torch.zeros(1, 4, 2, 2),
        "p4": torch.zeros(1, 4, 1, 2),
        "p5": torch.zeros(1, 4, 1, 1),
    }
    fg = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    labels = torch.tensor([[14, 14, 3, 14, 7, 7, -1]])
    scores = torch.zeros(1, 7, 20)
    scores[0, :, 14] = torch.arange(7, dtype=torch.float32)
    result = partition_tal_units(fg, labels, scores, 14, ["p3", "p4", "p5"], features, [0.7, 0.6, 0.4], [1, 1, 1])
    assert result.stats["num_target_positive"] == 3
    assert result.stats["num_non_target_positive"] == 3
    assert set(result.non_target_class_gates) == {3, 7}
    assert 0.0 < result.stats["target_unit_coverage"] <= 1.0


def test_running_covariance():
    torch.manual_seed(0)
    values = torch.randn(37, 6, dtype=torch.float64)
    running = RunningCovariance(6)
    running.update(values[:11])
    running.update(values[11:])
    centered = values - values.mean(dim=0)
    assert running.count == 37
    assert torch.allclose(running.mean, values.mean(dim=0), atol=1e-12)
    assert torch.allclose(running.covariance(), centered.T @ centered / 37, atol=1e-12)
    assert torch.allclose(running.second_moment(), values.T @ values / 37, atol=1e-12)


def test_semantic_pca():
    torch.manual_seed(1)
    coefficients = torch.randn(200, 2, dtype=torch.float64)
    features = torch.zeros(200, 6, dtype=torch.float64)
    features[:, :2] = coefficients
    features[:, 2:] = 0.01 * torch.randn(200, 4, dtype=torch.float64)
    features += 4.0
    result = fit_semantic_pca(features, variance_threshold=0.90)
    expected = torch.eye(6, dtype=torch.float64)[:, :2]
    assert result.basis.shape[1] == 2
    assert projection_similarity(result.basis, expected) > 0.99
    assert torch.allclose(result.mean, features.mean(dim=0))


def _synthetic_moments():
    target = torch.diag(torch.tensor([9.0, 7.0, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64))
    non_target = torch.diag(torch.tensor([0.1, 0.1, 8.0, 6.0, 0.2, 0.2], dtype=torch.float64))
    semantic = torch.eye(6, dtype=torch.float64)[:, :4]
    return target, non_target, semantic


def test_generalized_eigen_solver():
    target, non_target, semantic = _synthetic_moments()
    result = solve_discriminative_subspace(target, non_target, semantic, rank=2, regularization=1e-3)
    expected = torch.eye(6, dtype=torch.float64)[:, :2]
    assert projection_similarity(result.basis, expected) > 0.999
    assert selectivity_ratio(target, non_target, result.basis) > 50
    assert semantic_overlap(result.basis, semantic) > 0.999


def test_subspace_orthogonality():
    target, non_target, semantic = _synthetic_moments()
    q = solve_discriminative_subspace(target, non_target, semantic, rank=2).basis
    assert (q.T @ q - torch.eye(2, dtype=q.dtype)).abs().max() <= 1e-10


def test_projection_energy():
    q = torch.eye(4)[:, :2]
    shift = torch.tensor([[3.0, 4.0, 5.0, 0.0]])
    projected, outside, total = subspace_energies(shift, q)
    assert projected.item() == 25.0
    assert outside.item() == 25.0
    assert total.item() == 50.0


def test_outside_subspace_energy():
    q = torch.eye(3)[:, :1]
    in_space = torch.tensor([[2.0, 0.0, 0.0]])
    _, outside, _ = subspace_energies(in_space, q)
    assert outside.item() == 0.0


def test_non_target_leakage():
    q = torch.eye(3)[:, :1]
    protected = torch.tensor([[0.0, 3.0, 4.0]])
    leaking = torch.tensor([[2.0, 0.0, 0.0]])
    assert non_target_leakage(protected, q).item() == 0.0
    assert non_target_leakage(leaking, q).item() == 4.0


def test_loss_gradient_direction():
    q = torch.eye(4)[:, :2]
    target = torch.tensor([[0.2, 0.1, 0.5, 0.0]], requires_grad=True)
    non_target = torch.tensor([[0.4, 0.2, 0.0, 0.0]], requires_grad=True)
    before_projected, before_outside, _ = subspace_energies(target.detach(), q)
    before_leakage = non_target_leakage(non_target.detach(), q)
    loss, _ = dcss_stage1_loss(
        target,
        non_target,
        q,
        margin=1.0,
        lambda_energy=2.0,
        lambda_outside=1.0,
        lambda_leakage=1.0,
        lambda_logits=0.0,
    )
    loss.backward()
    with torch.no_grad():
        target_after = target - 0.1 * target.grad
        non_target_after = non_target - 0.1 * non_target.grad
    after_projected, after_outside, _ = subspace_energies(target_after, q)
    after_leakage = non_target_leakage(non_target_after, q)
    assert after_projected.item() > before_projected.item()
    assert after_outside.item() <= before_outside.item()
    assert after_leakage.item() <= before_leakage.item()


def test_loss_zero_initialization_has_finite_gradient():
    q = torch.eye(4)[:, :2]
    target = torch.zeros(3, 4, requires_grad=True)
    non_target = torch.zeros(2, 4, requires_grad=True)
    loss, _ = dcss_stage1_loss(target, non_target, q, margin=1.0)
    loss.backward()
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(non_target.grad).all()


def test_subspace_save_load():
    q = random_subspace(6, 2, seed=3)
    payload = {"layers": {"model.15": {"subspaces": {"dcss": {2: q}}}}}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "subspaces.pt")
        save_subspaces(path, payload)
        loaded = load_subspaces(path)
    assert torch.equal(loaded["layers"]["model.15"]["subspaces"]["dcss"][2], q)


def test_no_semantic_generalized_eigen_solver():
    target, non_target, _ = _synthetic_moments()
    result = solve_no_semantic_subspace(target, non_target, rank=2, regularization=1e-3)
    expected = torch.eye(6, dtype=torch.float64)[:, :2]
    assert projection_similarity(result.basis, expected) > 0.999
    assert result.orthogonality_error <= 1e-10


def test_no_semantic_subspace_save_load():
    target, non_target, _ = _synthetic_moments()
    q = solve_no_semantic_subspace(target, non_target, rank=2).basis
    payload = {"layers": {"model.15": {"subspaces": {"no_pt": {2: q}}}}}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "subspace.pt")
        save_subspaces(path, payload)
        loaded = load_subspaces(path)
    assert torch.equal(loaded["layers"]["model.15"]["subspaces"]["no_pt"][2], q)


def test_principal_angles():
    first = torch.eye(4, dtype=torch.float64)[:, :2]
    second = torch.stack([torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 0.0])], dim=1)
    angles = principal_angles_degrees(first, second)
    assert torch.allclose(angles, torch.tensor([0.0, 90.0], dtype=torch.float64), atol=1e-6)


def test_relative_weight_override_keeps_base_unchanged():
    base = {"dcss": {"energy_margin": 2.0, "lambda_leakage": 3.0, "lambda_energy": 4.0}}
    changed = apply_relative_overrides(base, 0.5, 4.0)
    assert changed["dcss"]["energy_margin"] == 1.0
    assert changed["dcss"]["lambda_leakage"] == 12.0
    assert changed["dcss"]["lambda_energy"] == 4.0
    assert base["dcss"]["energy_margin"] == 2.0


def test_diagnostic_gate_thresholds():
    passing = diagnostic_gate({
        "target_unit_coverage": 0.50,
        "target_projected_energy": 0.3601,
        "non_target_leakage": 0.1803,
        "R_shift": 2.0,
        "budget_consistent": True,
    })
    assert passing["pass"]
    failing = diagnostic_gate({
        "target_unit_coverage": 0.49,
        "target_projected_energy": 0.3601,
        "non_target_leakage": 0.1804,
        "R_shift": 2.0,
        "budget_consistent": True,
    })
    assert not failing["pass"]


def test_diagnostic_metrics_aggregation_ignores_non_finite_values():
    rows = [{"energy": "0.2"}, {"energy": "nan"}, {"energy": 0.6}, {"other": 1.0}]
    assert mean_finite_metric(rows, "energy") == 0.4


def test_stage1_legacy_config_is_compatible_with_relative_override():
    path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "dcss", "stage1", "E4_dcss_r8_seed0", "config.yaml")
    with open(path, encoding="utf-8") as file:
        legacy = yaml.safe_load(file)
    changed = apply_relative_overrides(legacy, 0.5, 2.0)
    assert changed["dcss"]["energy_margin"] == 0.5
    assert changed["dcss"]["lambda_leakage"] == 2.0
    assert changed["dcss"]["subspace_path"] == legacy["dcss"]["subspace_path"]


def test_resume_run_directory_cannot_target_historical_stage1():
    with tempfile.TemporaryDirectory() as directory:
        path = build_resume_run_dir(directory, "diagnostic", "D1")
        assert os.path.dirname(path) == os.path.abspath(directory)
        assert os.path.basename(path) == "diagnostic_D1"
        try:
            build_resume_run_dir(directory, "diagnostic", "../E4_dcss_r8_seed0")
        except ValueError:
            pass
        else:
            raise AssertionError("path traversal must be rejected")


def test_projected_coefficient_metrics():
    from dcss.stage1 import projected_coefficient_metrics
    shifts=torch.tensor([[1.,0.,0.],[2.,0.,0.],[0.,1.,0.]])
    values=projected_coefficient_metrics(shifts,torch.eye(3)[:,:2])
    assert set(values)=={"projected_coefficient_pairwise_cosine","projected_coefficient_norm_cv","projected_covariance_effective_rank"}
    assert values["projected_covariance_effective_rank"] >= 1


def test_gradient_component_norms_and_conflict_cosine():
    first=torch.tensor([3.,4.]); second=-first
    assert gradient_component_stats(first)=={"l2":5.0,"l1":7.0,"max_abs":4.0}
    assert gradient_cosine(first,second) < -0.999


def test_constrained_direction_synthetic_and_feasibility():
    target=torch.tensor([1.,1.]); constraints=[torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    direction,status=constrained_direction(target,constraints)
    assert status["status"]=="feasible"
    assert all(float((g*direction).sum()) <= 1e-6 for g in constraints)


def test_constrained_no_silent_fallback():
    try:
        constrained_direction(torch.tensor([1.]),[torch.tensor([1.])],max_iterations=0)
    except (RuntimeError, UnboundLocalError):
        pass
    else:
        raise AssertionError("solver failure must not silently return weighted update")


def test_object_aligned_warp_and_non_target_overlap_exclusion():
    delta=torch.ones(3,8,8)
    annotations=[{"cls":14,"bbox":[0.5,0.5,0.5,0.5]},{"cls":7,"bbox":[0.5,0.5,0.1,0.1]}]
    canvas,support,non_target,metrics=object_aligned_warp(delta,annotations,32,14,dilation=1)
    assert canvas.shape==(3,32,32) and float((canvas*non_target).abs().max())==0
    assert 0 < metrics["valid_support_area"] < 1 and metrics["non_target_overlap_ratio"]==0
