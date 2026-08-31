from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


InstanceTuple = Tuple[str, int, int]


@dataclass(frozen=True)
class ClassRiskAgreement:
    class_id: int
    matched_count: int
    union_count: int
    coverage: float
    spearman: float
    top_fraction_overlap: float


@dataclass(frozen=True)
class ProxyVictimAgreement:
    matched_count: int
    union_count: int
    matched_coverage: float
    macro_spearman: float
    macro_top_fraction_overlap: float
    per_class: Mapping[int, ClassRiskAgreement]
    passed: bool
    failure_reasons: Tuple[str, ...]


def _validate_risks(risks: Mapping[InstanceTuple, float], *, name: str) -> None:
    if not risks:
        raise ValueError("%s risk mapping must be non-empty." % name)
    for key, value in risks.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not str(key[0]).strip()
            or int(key[1]) < 0
            or int(key[2]) < 0
        ):
            raise ValueError("%s contains an invalid instance key." % name)
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("%s risks must be finite in [0,1]." % name)


def _mid_ranks(values: Tuple[float, ...]) -> Tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        mid = 0.5 * (start + end - 1)
        for offset in range(start, end):
            output[ordered[offset]] = mid
        start = end
    return tuple(output)


def _pearson(first: Tuple[float, ...], second: Tuple[float, ...]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_centered = tuple(value - first_mean for value in first)
    second_centered = tuple(value - second_mean for value in second)
    numerator = sum(a * b for a, b in zip(first_centered, second_centered))
    denominator = math.sqrt(
        sum(value * value for value in first_centered)
        * sum(value * value for value in second_centered)
    )
    return numerator / denominator if denominator > 0 else 0.0


def _top_keys(
    risks: Mapping[InstanceTuple, float],
    keys: Tuple[InstanceTuple, ...],
    *,
    top_fraction: float,
) -> set[InstanceTuple]:
    count = max(1, math.ceil(len(keys) * top_fraction))
    return set(sorted(keys, key=lambda key: (-float(risks[key]), key))[:count])


def evaluate_proxy_victim_agreement(
    proxy_risks: Mapping[InstanceTuple, float],
    victim_risks: Mapping[InstanceTuple, float],
    *,
    top_fraction: float = 0.25,
    minimum_spearman: float = 0.40,
    minimum_top_overlap: float = 0.50,
    minimum_coverage: float = 0.90,
) -> ProxyVictimAgreement:
    """Evaluate training-only risk transfer without using validation AP."""

    _validate_risks(proxy_risks, name="proxy")
    _validate_risks(victim_risks, name="victim")
    if not 0 < top_fraction <= 1:
        raise ValueError("Agreement top_fraction must lie in (0,1].")
    for threshold in (
        minimum_spearman,
        minimum_top_overlap,
        minimum_coverage,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Agreement thresholds must lie in [0,1].")

    proxy_keys = set(proxy_risks)
    victim_keys = set(victim_risks)
    matched = proxy_keys.intersection(victim_keys)
    union = proxy_keys.union(victim_keys)
    coverage = len(matched) / float(len(union))
    classes = sorted({int(key[2]) for key in matched})
    per_class: Dict[int, ClassRiskAgreement] = {}
    for class_id in classes:
        class_matched = tuple(sorted(key for key in matched if int(key[2]) == class_id))
        class_union = {
            key for key in union if int(key[2]) == class_id
        }
        if len(class_matched) < 2:
            continue
        proxy_values = tuple(float(proxy_risks[key]) for key in class_matched)
        victim_values = tuple(float(victim_risks[key]) for key in class_matched)
        spearman = _pearson(_mid_ranks(proxy_values), _mid_ranks(victim_values))
        proxy_top = _top_keys(
            proxy_risks, class_matched, top_fraction=top_fraction
        )
        victim_top = _top_keys(
            victim_risks, class_matched, top_fraction=top_fraction
        )
        overlap = len(proxy_top.intersection(victim_top)) / float(
            max(len(proxy_top), len(victim_top))
        )
        per_class[class_id] = ClassRiskAgreement(
            class_id=class_id,
            matched_count=len(class_matched),
            union_count=len(class_union),
            coverage=len(class_matched) / float(len(class_union)),
            spearman=spearman,
            top_fraction_overlap=overlap,
        )
    if not per_class:
        macro_spearman = 0.0
        macro_overlap = 0.0
    else:
        macro_spearman = sum(item.spearman for item in per_class.values()) / len(
            per_class
        )
        macro_overlap = sum(
            item.top_fraction_overlap for item in per_class.values()
        ) / len(per_class)
    reasons = []
    if coverage + 1.0e-12 < minimum_coverage:
        reasons.append("matched_coverage_below_gate")
    if macro_spearman + 1.0e-12 < minimum_spearman:
        reasons.append("macro_spearman_below_gate")
    if macro_overlap + 1.0e-12 < minimum_top_overlap:
        reasons.append("macro_top_overlap_below_gate")
    if not per_class:
        reasons.append("no_evaluable_class")
    return ProxyVictimAgreement(
        matched_count=len(matched),
        union_count=len(union),
        matched_coverage=coverage,
        macro_spearman=macro_spearman,
        macro_top_fraction_overlap=macro_overlap,
        per_class=per_class,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def worst_case_calibrated_risks(
    proxy_risks: Mapping[InstanceTuple, float],
    victim_risks: Mapping[InstanceTuple, float],
) -> Dict[InstanceTuple, float]:
    """Return a monotone, capacity-free max calibration after a passed audit."""

    _validate_risks(proxy_risks, name="proxy")
    _validate_risks(victim_risks, name="victim")
    return {
        key: max(float(proxy_risks.get(key, 0.0)), float(victim_risks.get(key, 0.0)))
        for key in sorted(set(proxy_risks).union(victim_risks))
    }
