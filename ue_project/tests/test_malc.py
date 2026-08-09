from __future__ import annotations

import pytest
import torch

from ue_framework.methods.malc import (
    FrozenMALCPrototypeBank,
    multi_scale_assignment_latent_concentration,
    score_weighted_instance_residuals,
    target_class_assignment_scores,
)


def _bank(*channels: int, median: float = 1.0, floor: float = 0.2):
    return FrozenMALCPrototypeBank(
        direction_prototypes=tuple(
            torch.nn.functional.one_hot(
                torch.tensor(0), num_classes=channel
            ).float()
            for channel in channels
        ),
        median_rms=tuple(median for _ in channels),
        energy_floors=tuple(floor for _ in channels),
    )


def test_score_weighting_keeps_instances_separate_and_differentiable() -> None:
    clean = [torch.zeros((1, 2, 1, 4))]
    adv = [
        torch.tensor(
            [[[[1.0, 3.0, 0.0, 0.0]], [[0.0, 0.0, 2.0, 4.0]]]],
            requires_grad=True,
        )
    ]
    residuals = score_weighted_instance_residuals(
        clean,
        adv,
        pag_gate=torch.ones((1, 4), dtype=torch.bool),
        target_gt_idx=torch.tensor([[0, 0, 1, 1]]),
        assigned_scores=torch.tensor([[0.75, 0.25, 0.25, 0.75]]),
        target_gt_indices_by_image=((0, 1),),
    )
    assert torch.allclose(
        residuals.vectors[0], torch.tensor([[1.5, 0.0], [0.0, 3.5]])
    )
    assert torch.equal(residuals.assignment_count[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(residuals.score_mass[0], torch.tensor([1.0, 1.0]))
    result = multi_scale_assignment_latent_concentration(
        residuals,
        FrozenMALCPrototypeBank(
            direction_prototypes=(torch.tensor([1.0, 1.0]),),
            median_rms=(1.0,),
            energy_floors=(0.2,),
        ),
    )
    result.loss.backward()
    assert adv[0].grad is not None
    assert torch.isfinite(adv[0].grad).all()
    assert float(adv[0].grad.norm()) > 0


def test_equal_scale_reduction_is_invariant_to_anchor_duplication() -> None:
    def run(first_scale_width: int) -> torch.Tensor:
        clean = [
            torch.zeros((1, 2, 1, first_scale_width)),
            torch.zeros((1, 2, 1, 1)),
        ]
        first = torch.zeros((1, 2, 1, first_scale_width))
        first[:, 0] = 1.0
        second = torch.tensor([[[[0.0]], [[1.0]]]])
        residuals = score_weighted_instance_residuals(
            clean,
            [first, second],
            pag_gate=torch.ones((1, first_scale_width + 1), dtype=torch.bool),
            target_gt_idx=torch.zeros((1, first_scale_width + 1), dtype=torch.long),
            assigned_scores=torch.ones((1, first_scale_width + 1)),
            target_gt_indices_by_image=((0,),),
        )
        return multi_scale_assignment_latent_concentration(
            residuals,
            FrozenMALCPrototypeBank(
                direction_prototypes=(
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([1.0, 0.0]),
                ),
                median_rms=(2 ** -0.5, 2 ** -0.5),
                energy_floors=(0.1, 0.1),
            ),
        ).loss

    assert torch.allclose(run(1), run(8), atol=1e-7)


def test_low_energy_is_counted_by_floor_and_not_dropped() -> None:
    clean = [torch.zeros((1, 2, 1, 1))]
    adv = [torch.full((1, 2, 1, 1), 0.01, requires_grad=True)]
    residuals = score_weighted_instance_residuals(
        clean,
        adv,
        pag_gate=torch.ones((1, 1), dtype=torch.bool),
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        assigned_scores=torch.ones((1, 1)),
        target_gt_indices_by_image=((0,),),
    )
    result = multi_scale_assignment_latent_concentration(
        residuals, _bank(2, median=0.1, floor=0.05)
    )
    assert result.valid_instance_coverage == 1.0
    assert result.floor_pass_ratio == 0.0
    assert float(result.floor_loss.detach()) > 0
    assert result.valid_scale_count == 1


def test_no_assignment_fails_closed_instead_of_faking_concentration() -> None:
    residuals = score_weighted_instance_residuals(
        [torch.zeros((1, 2, 1, 1))],
        [torch.ones((1, 2, 1, 1))],
        pag_gate=torch.zeros((1, 1), dtype=torch.bool),
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        assigned_scores=torch.ones((1, 1)),
        target_gt_indices_by_image=((0,),),
    )
    with pytest.raises(RuntimeError, match="at least one assigned"):
        multi_scale_assignment_latent_concentration(residuals, _bank(2))


def test_target_score_extraction_and_invalid_scores_fail_closed() -> None:
    scores = torch.zeros((2, 3, 20))
    scores[..., 14] = 0.6
    extracted = target_class_assignment_scores(scores, target_class_id=14)
    assert extracted.shape == (2, 3)
    assert torch.equal(extracted, torch.full((2, 3), 0.6))
    scores[0, 0, 14] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        target_class_assignment_scores(scores, target_class_id=14)


def test_prototype_bank_is_detached_normalized_and_validated() -> None:
    source = torch.tensor([3.0, 4.0], requires_grad=True)
    bank = FrozenMALCPrototypeBank(
        direction_prototypes=(source,),
        median_rms=(0.5,),
        energy_floors=(0.25,),
    )
    assert not bank.direction_prototypes[0].requires_grad
    assert torch.allclose(bank.direction_prototypes[0].norm(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="zero norm"):
        FrozenMALCPrototypeBank(
            direction_prototypes=(torch.zeros(2),),
            median_rms=(0.5,),
            energy_floors=(0.25,),
        )
