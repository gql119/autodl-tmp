import torch

from ue_framework.methods.semantic_hiding_carrier import SemanticHidingCarrier
from ue_framework.methods.semantic_hiding_validation import (
    HidingMetrics,
    compute_hiding_metrics,
    evaluate_hiding_gate,
    hiding_pretrain_step,
    phase_scramble,
    retrieval_statistics,
    ssim_per_sample,
)


def test_ssim_and_retrieval_are_exact_for_bank_members():
    torch.manual_seed(3)
    bank = torch.rand(3, 3, 16, 16)
    true_indices = torch.tensor([2, 0, 1])
    recovered = bank[true_indices].clone()
    stats = retrieval_statistics(recovered, bank, true_indices)
    assert float(stats["top1_accuracy"]) == 1.0
    assert torch.all(stats["relative_l1_margin"] == 1.0)
    assert torch.allclose(ssim_per_sample(recovered, recovered), torch.ones(3), atol=1e-5)


def test_gate_passes_controlled_fixture_and_rejects_collapse():
    good = HidingMetrics(
        retrieval_top1=1.0,
        primary_recovery_ssim_median=0.8,
        primary_relative_l1_margin_median=0.7,
        pairwise_pixel_cosine_median=0.3,
        channel_rms_cv=(0.1, 0.2, 0.15),
        high_frequency_energy_median=0.2,
        linf=16.0 / 255.0,
        support_outside_max=0.0,
        all_finite=True,
    )
    assert evaluate_hiding_gate(good)["pass"] is True
    collapsed = HidingMetrics(
        retrieval_top1=1.0 / 3.0,
        primary_recovery_ssim_median=0.1,
        primary_relative_l1_margin_median=-0.2,
        pairwise_pixel_cosine_median=0.999,
        channel_rms_cv=(0.0, 0.0, 0.0),
        high_frequency_energy_median=0.2,
        linf=16.0 / 255.0,
        support_outside_max=0.0,
        all_finite=True,
    )
    decision = evaluate_hiding_gate(collapsed)
    assert decision["pass"] is False
    assert decision["checks"]["pixel_diversity"] is False
    assert decision["checks"]["rms_diversity"] is False


def test_compute_metrics_handles_primary_subset_and_low_frequency_deltas():
    torch.manual_seed(4)
    bank = torch.rand(3, 3, 32, 32)
    true_indices = torch.tensor([2, 2, 0, 1])
    recovered = bank[true_indices].clone()
    yy = torch.linspace(-1.0, 1.0, 32)[None, :]
    xx = torch.linspace(-1.0, 1.0, 32)[:, None]
    patterns = []
    for index in range(4):
        value = torch.sin((index + 1) * xx) + torch.cos((index + 2) * yy)
        patterns.append(value.expand(3, -1, -1) * (0.01 + index * 0.002))
    deltas = torch.stack(patterns)
    metrics = compute_hiding_metrics(
        recovered,
        bank[true_indices],
        bank,
        true_indices,
        deltas,
        primary_index=2,
        high_radius=8.0,
    )
    assert metrics.retrieval_top1 == 1.0
    assert metrics.primary_recovery_ssim_median > 0.99
    assert metrics.primary_relative_l1_margin_median > 0.99
    assert metrics.linf < 0.1
    assert metrics.support_outside_max == 0.0


def test_phase_scramble_is_deterministic_and_changes_structure():
    torch.manual_seed(9)
    image = torch.rand(2, 3, 16, 16)
    first = phase_scramble(image, seed=21)
    second = phase_scramble(image, seed=21)
    assert torch.equal(first, second)
    assert not torch.allclose(first, image)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    original_power = torch.fft.fft2(
        image - image.mean(dim=(-2, -1), keepdim=True)
    ).abs().square().flatten(2)
    scrambled_power = torch.fft.fft2(
        first - first.mean(dim=(-2, -1), keepdim=True)
    ).abs().square().flatten(2)
    original_power = original_power / original_power.sum(dim=2, keepdim=True)
    scrambled_power = scrambled_power / scrambled_power.sum(dim=2, keepdim=True)
    assert torch.allclose(original_power, scrambled_power, atol=2e-5, rtol=2e-4)


def test_pretrain_step_has_finite_nonzero_gradients():
    torch.manual_seed(12)
    carrier = SemanticHidingCarrier(input_size=16, width=4, coupling_blocks=2)
    optimizer = torch.optim.Adam(carrier.parameters(), lr=1e-3)
    hosts = torch.rand(3, 3, 16, 16)
    secrets = torch.rand(3, 3, 16, 16)
    record = hiding_pretrain_step(carrier, optimizer, hosts, secrets)
    assert all(torch.isfinite(torch.tensor(value)) for value in record.values())
    assert record["gradient_l1"] > 0.0
