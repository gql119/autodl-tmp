from __future__ import annotations

import pytest

from ue_framework.methods.dgcaip_r3_diagnostics import (
    build_rejection_attribution,
    build_same_process_replay,
)


def _step(families, *, accepted=False):
    constraints = []
    for index, family in enumerate(families):
        value = 0.0 if accepted else 1.0
        constraints.append(
            {
                "name": f"{index}:{family}",
                "family": family,
                "value": value,
                "limit": 0.0,
                "margin": value,
                "violated": not accepted,
            }
        )
    return {
        "accepted": accepted,
        "backtrack_attempts": 1,
        "backtracking_trace": [
            {
                "attempt": 0,
                "step_size": 0.1,
                "finite": True,
                "constraints": constraints,
                "group_max_margin": {},
                "group_violation_count": {},
                "accepted": accepted,
                "reason": (
                    "accepted" if accepted else "constraint_limit_exceeded"
                ),
            }
        ],
    }


def _arms(p2_families, p4_families):
    p2 = [_step(p2_families) for _ in range(7)] + [
        _step((), accepted=True)
    ]
    p4 = [_step(p4_families) for _ in range(7)] + [
        _step((), accepted=True)
    ]
    return p2, p4


def test_h1_labels_common_caip_infeasibility() -> None:
    p2, p4 = _arms(("probability",), ("probability", "js"))
    report = build_rejection_attribution(
        p2_steps=p2,
        p4_steps=p4,
        routed_gradient_cosines=[0.95] * 8,
    )
    assert report["label"] == "caip_common_infeasibility"
    assert report["trace_complete"] is True
    assert report["p2_dominant_non_js_family"] == "probability"
    assert report["p2_dominant_non_js_share"] == pytest.approx(1.0)
    assert report["p4_shared_dominant_family_steps"] == 7
    assert report["p2_terminal_positive_margin"]["probability"] == {
        "count": 7,
        "maximum": pytest.approx(1.0),
        "median": pytest.approx(1.0),
    }


def test_h1_labels_js_incremental_blocker() -> None:
    p2, p4 = _arms(
        ("probability", "iou", "alignment"),
        ("probability", "js", "js"),
    )
    report = build_rejection_attribution(
        p2_steps=p2,
        p4_steps=p4,
        routed_gradient_cosines=[0.95] * 8,
    )
    assert report["label"] == "js_incremental_blocker"
    assert report["p4_js_dominant_steps"] == 7


def test_h1_labels_ranking_route_shift() -> None:
    p2, p4 = _arms(
        ("probability", "iou", "alignment"),
        ("probability", "iou", "alignment"),
    )
    report = build_rejection_attribution(
        p2_steps=p2,
        p4_steps=p4,
        routed_gradient_cosines=[0.50] * 8,
    )
    assert report["label"] == "ranking_route_shift"
    assert report["routed_gradient_cosine_median"] == pytest.approx(0.50)


def test_h1_labels_inconclusive_mixed() -> None:
    p2, p4 = _arms(
        ("probability", "iou", "alignment"),
        ("probability", "iou", "alignment"),
    )
    report = build_rejection_attribution(
        p2_steps=p2,
        p4_steps=p4,
        routed_gradient_cosines=[0.95] * 8,
    )
    assert report["label"] == "inconclusive_mixed"


def test_h1_rejects_mismatched_step_counts() -> None:
    with pytest.raises(ValueError, match="step counts"):
        build_rejection_attribution(
            p2_steps=[_step(("probability",))],
            p4_steps=[],
            routed_gradient_cosines=[],
        )


@pytest.mark.parametrize(
    ("initial_b", "batches_b", "replay_pass", "expected"),
    [
        ("same", ("batch-0",), True, "within_process_replay_pass"),
        ("same", ("batch-0",), False, "within_process_nondeterminism"),
        ("different", ("batch-0",), True, "replay_invalid_input_mismatch"),
        ("same", ("batch-1",), True, "replay_invalid_input_mismatch"),
    ],
)
def test_h2_labels_are_fail_closed(
    initial_b, batches_b, replay_pass, expected
) -> None:
    report = build_same_process_replay(
        p1_a_initial_sha256="same",
        p1_b_initial_sha256=initial_b,
        p1_a_batch_sha256=("batch-0",),
        p1_b_batch_sha256=batches_b,
        replay_report={
            "absolute_tolerance": 1.0e-6,
            "relative_tolerance": 1.0e-4,
            "structural_checks": {"steps.0.accepted": replay_pass},
            "numeric_comparisons": {},
            "pass": replay_pass,
        },
    )
    assert report["label"] == expected
