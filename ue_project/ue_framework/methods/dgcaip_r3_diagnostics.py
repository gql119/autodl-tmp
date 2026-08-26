from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any, Dict, Mapping, Sequence


NON_JS_FAMILIES = ("probability", "iou", "alignment")
ALL_FAMILIES = NON_JS_FAMILIES + ("js",)


def _terminal(step: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trace = step.get("backtracking_trace", ())
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return None
    return trace[-1] if trace else None


def _trace_complete(step: Mapping[str, Any]) -> bool:
    trace = step.get("backtracking_trace", ())
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return False
    if len(trace) != int(step.get("backtrack_attempts", -1)):
        return False
    for attempt_index, attempt in enumerate(trace):
        if int(attempt.get("attempt", -1)) != attempt_index:
            return False
        if not bool(attempt.get("finite", False)):
            return False
        constraints = attempt.get("constraints", ())
        if not isinstance(constraints, Sequence):
            return False
        for item in constraints:
            required = {"name", "family", "value", "limit", "margin", "violated"}
            if not required.issubset(item):
                return False
            if not all(
                math.isfinite(float(item[key]))
                for key in ("value", "limit", "margin")
            ):
                return False
            recomputed = float(item["value"]) > float(item["limit"]) + 1.0e-9
            if bool(item["violated"]) != recomputed:
                return False
        recomputed_accept = all(
            not bool(item["violated"]) for item in constraints
        )
        if bool(attempt.get("accepted")) != recomputed_accept:
            return False
    return bool(step.get("accepted")) == any(
        bool(attempt.get("accepted")) for attempt in trace
    )


def _violation_counts(step: Mapping[str, Any]) -> Counter[str]:
    terminal = _terminal(step)
    if terminal is None:
        return Counter()
    return Counter(
        str(item["family"])
        for item in terminal.get("constraints", ())
        if bool(item.get("violated"))
    )


def _positive_margins(
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float | int | None]]:
    by_family = {family: [] for family in ALL_FAMILIES}
    for step in steps:
        terminal = _terminal(step)
        if terminal is None:
            continue
        for item in terminal.get("constraints", ()):
            margin = float(item["margin"])
            family = str(item["family"])
            if bool(item.get("violated")) and family in by_family:
                by_family[family].append(margin)
    return {
        family: {
            "count": len(values),
            "maximum": max(values) if values else None,
            "median": median(values) if values else None,
        }
        for family, values in by_family.items()
    }


def build_rejection_attribution(
    *,
    p2_steps: Sequence[Mapping[str, Any]],
    p4_steps: Sequence[Mapping[str, Any]],
    routed_gradient_cosines: Sequence[float],
) -> Dict[str, Any]:
    if len(p2_steps) != len(p4_steps):
        raise ValueError("P2/P4 diagnostic step counts must match.")
    if len(routed_gradient_cosines) != len(p2_steps):
        raise ValueError("One routed-gradient cosine is required per step.")
    if not all(math.isfinite(float(value)) for value in routed_gradient_cosines):
        raise ValueError("Routed-gradient cosines must be finite.")

    p2_rejected = [step for step in p2_steps if not bool(step.get("accepted"))]
    p4_rejected = [step for step in p4_steps if not bool(step.get("accepted"))]
    trace_complete = all(
        _trace_complete(step) for step in tuple(p2_steps) + tuple(p4_steps)
    )

    p2_counts = Counter()
    p2_non_js_terminal_steps = 0
    for step in p2_rejected:
        counts = _violation_counts(step)
        p2_counts.update(counts)
        if sum(counts[family] for family in NON_JS_FAMILIES) > 0:
            p2_non_js_terminal_steps += 1

    non_js_total = sum(p2_counts[family] for family in NON_JS_FAMILIES)
    dominant_family = None
    dominant_share = 0.0
    if non_js_total:
        dominant_family = max(
            NON_JS_FAMILIES, key=lambda family: p2_counts[family]
        )
        dominant_share = p2_counts[dominant_family] / float(non_js_total)

    p4_dominant_family_steps = 0
    if dominant_family is not None:
        for p2_step, p4_step in zip(p2_steps, p4_steps):
            if bool(p2_step.get("accepted")) or bool(p4_step.get("accepted")):
                continue
            if _violation_counts(p4_step)[dominant_family] > 0:
                p4_dominant_family_steps += 1

    p4_js_dominant_steps = 0
    p4_counts = Counter()
    for step in p4_rejected:
        counts = _violation_counts(step)
        p4_counts.update(counts)
        total = sum(counts[family] for family in ALL_FAMILIES)
        if total and counts["js"] / float(total) >= 0.60:
            p4_js_dominant_steps += 1

    condition_rejected_count = len(p2_rejected) == 7
    condition_non_js_persists = (
        condition_rejected_count and p2_non_js_terminal_steps >= 5
    )
    condition_family_concentration = (
        dominant_family is not None and dominant_share >= 0.60
    )
    condition_shared_with_p4 = p4_dominant_family_steps >= 5
    common_caip = (
        trace_complete
        and condition_non_js_persists
        and condition_family_concentration
        and condition_shared_with_p4
    )

    if common_caip:
        label = "caip_common_infeasibility"
    elif not condition_family_concentration and p4_js_dominant_steps >= 5:
        label = "js_incremental_blocker"
    elif median(float(value) for value in routed_gradient_cosines) < 0.90:
        label = "ranking_route_shift"
    else:
        label = "inconclusive_mixed"

    return {
        "label": label,
        "trace_complete": trace_complete,
        "active_trace_decisions_match": trace_complete,
        "p2_rejected_steps": len(p2_rejected),
        "p4_rejected_steps": len(p4_rejected),
        "p2_terminal_violation_count": {
            family: p2_counts[family] for family in ALL_FAMILIES
        },
        "p4_terminal_violation_count": {
            family: p4_counts[family] for family in ALL_FAMILIES
        },
        "p2_terminal_positive_margin": _positive_margins(p2_rejected),
        "p4_terminal_positive_margin": _positive_margins(p4_rejected),
        "p2_non_js_terminal_steps": p2_non_js_terminal_steps,
        "p2_dominant_non_js_family": dominant_family,
        "p2_dominant_non_js_share": dominant_share,
        "p4_shared_dominant_family_steps": p4_dominant_family_steps,
        "p4_js_dominant_steps": p4_js_dominant_steps,
        "routed_gradient_cosines": [
            float(value) for value in routed_gradient_cosines
        ],
        "routed_gradient_cosine_median": median(
            float(value) for value in routed_gradient_cosines
        ),
        "conditions": {
            "p2_exactly_seven_rejected": condition_rejected_count,
            "p2_non_js_persists_at_least_five": condition_non_js_persists,
            "p2_family_concentration_at_least_60pct": condition_family_concentration,
            "p4_shared_family_at_least_five": condition_shared_with_p4,
        },
    }


def build_same_process_replay(
    *,
    p1_a_initial_sha256: str,
    p1_b_initial_sha256: str,
    p1_a_batch_sha256: Sequence[str],
    p1_b_batch_sha256: Sequence[str],
    replay_report: Mapping[str, Any],
) -> Dict[str, Any]:
    initial_match = p1_a_initial_sha256 == p1_b_initial_sha256
    batch_match = tuple(p1_a_batch_sha256) == tuple(p1_b_batch_sha256)
    if not initial_match or not batch_match:
        label = "replay_invalid_input_mismatch"
    elif bool(replay_report.get("pass")):
        label = "within_process_replay_pass"
    else:
        label = "within_process_nondeterminism"
    return {
        "label": label,
        "initial_adapter_match": initial_match,
        "batch_sequence_match": batch_match,
        "p1_a_initial_adapter_sha256": p1_a_initial_sha256,
        "p1_b_initial_adapter_sha256": p1_b_initial_sha256,
        "p1_a_batch_sha256": list(p1_a_batch_sha256),
        "p1_b_batch_sha256": list(p1_b_batch_sha256),
        "replay": dict(replay_report),
    }
