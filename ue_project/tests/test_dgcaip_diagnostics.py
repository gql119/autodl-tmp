from __future__ import annotations

import torch

from ue_framework.methods.dgcaip import DGCAIPInstanceTerm, DGCAIPResult
from ue_framework.methods.dgcaip_diagnostics import build_dgcaip_locator_report


def _result(divergences, damages) -> DGCAIPResult:
    terms = []
    for index, (divergence, damage) in enumerate(zip(divergences, damages)):
        terms.append(
            DGCAIPInstanceTerm(
                batch_index=0,
                gt_index=index,
                class_id=1 if index < len(divergences) // 2 else 7,
                positive_count=1,
                geometry_risk=1.0,
                divergence_rank=0.0,
                weight=1.0,
                classification_damage=torch.tensor(float(damage)),
                box_damage=torch.tensor(float(damage) * 2.0),
                alignment_damage=torch.tensor(float(damage) * 3.0),
                classification_loss=torch.tensor(0.0),
                box_loss=torch.tensor(0.0),
                alignment_loss=torch.tensor(0.0),
                distribution_loss=torch.tensor(float(divergence)),
                clean_to_poison_kl=torch.tensor(float(divergence) * 2.0),
            )
        )
    return DGCAIPResult(
        loss=torch.tensor(0.0),
        instances=tuple(terms),
        active_classes=(1, 7),
        per_class_loss={},
        per_class_instance_count={1: len(terms) // 2, 7: len(terms) // 2},
        eligible_instance_count=len(terms),
        covered_instance_count=len(terms),
        coverage=1.0,
    )


def test_locator_passes_when_divergence_orders_instance_damage() -> None:
    report = build_dgcaip_locator_report(
        [_result(range(1, 9), range(1, 9))]
    )
    assert report["decision"] == "pass"
    assert report["spearman_divergence_composite_damage"] == 1.0
    assert report["q4_q1_composite_damage_ratio"] >= 1.5
    assert report["coverage"] == 1.0
    assert report["quartiles"]["Q4"]["count"] == 2
    assert set(report["per_class"]) == {"1", "7"}


def test_locator_fails_when_divergence_has_no_ranking_information() -> None:
    report = build_dgcaip_locator_report(
        [_result([1.0] * 8, range(1, 9))]
    )
    assert report["decision"] == "fail"
    assert report["spearman_divergence_composite_damage"] == 0.0
    assert report["checks"]["spearman"] is False
