from __future__ import annotations

import math

import pytest

from ue_framework.methods.dgcaip_proxy_agreement import (
    evaluate_proxy_victim_agreement,
    worst_case_calibrated_risks,
)


def _risks(values, *, class_id=1):
    return {
        ("image-%d" % index, index, class_id): float(value)
        for index, value in enumerate(values)
    }


def test_identical_proxy_victim_ranking_passes_all_gates() -> None:
    proxy = {**_risks((0.0, 0.2, 0.8, 1.0), class_id=1), **_risks((0.1, 0.3, 0.7, 0.9), class_id=7)}
    victim = dict(proxy)
    result = evaluate_proxy_victim_agreement(proxy, victim)
    assert result.passed
    assert result.matched_coverage == pytest.approx(1.0)
    assert result.macro_spearman == pytest.approx(1.0)
    assert result.macro_top_fraction_overlap == pytest.approx(1.0)


def test_reversed_ranking_and_missing_keys_fail_closed() -> None:
    proxy = _risks((0.0, 0.2, 0.8, 1.0))
    victim = _risks((1.0, 0.8, 0.2, 0.0))
    victim.pop(("image-3", 3, 1))
    result = evaluate_proxy_victim_agreement(proxy, victim)
    assert not result.passed
    assert result.matched_coverage == pytest.approx(0.75)
    assert "matched_coverage_below_gate" in result.failure_reasons
    assert "macro_spearman_below_gate" in result.failure_reasons


def test_ties_are_deterministic_and_max_calibration_is_monotone() -> None:
    proxy = _risks((0.2, 0.2, 0.8, 0.8))
    victim = _risks((0.2, 0.2, 0.8, 0.8))
    result = evaluate_proxy_victim_agreement(proxy, victim)
    assert result.macro_spearman == pytest.approx(1.0)
    calibrated = worst_case_calibrated_risks(
        proxy, {key: min(1.0, value + 0.1) for key, value in victim.items()}
    )
    assert all(calibrated[key] >= proxy[key] for key in proxy)


def test_nonfinite_risk_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_proxy_victim_agreement(
            _risks((0.0, 1.0)),
            {("image-0", 0, 1): math.nan, ("image-1", 1, 1): 1.0},
        )
