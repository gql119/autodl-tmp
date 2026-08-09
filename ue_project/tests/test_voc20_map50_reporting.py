from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ue_framework.metrics_utils import (
    VOC20_CLASS_NAMES,
    extract_map50_per_class,
    named_voc20_ap50,
)
from ue_framework.stages.aggregate import build_c0_m1_comparison
from ue_framework.stages.evaluate import (
    _extract_metrics_dict,
    _sirc_manifest_provenance,
    _sirc_mechanism_evidence,
)


def _metrics(values, indices=None):
    values = np.asarray(values, dtype=np.float64)
    box = SimpleNamespace(
        ap50=values,
        map50=float(np.mean(values)),
    )
    if indices is not None:
        box.ap_class_index = np.asarray(indices, dtype=np.int64)
    return SimpleNamespace(box=box)


def test_ap50_uses_explicit_class_indices_not_array_position() -> None:
    expected = np.linspace(0.1, 0.9, 20)
    indices = np.asarray([7, 1, 19, 0, 14, 3, 9, 2, 18, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17])
    compact = expected[indices]
    metrics = _metrics(compact, indices)
    mapped = extract_map50_per_class(metrics, 20, strict=True)
    assert np.allclose(mapped, expected)
    named = named_voc20_ap50(metrics)
    assert list(named) == list(VOC20_CLASS_NAMES)
    assert named["person"] == pytest.approx(expected[14])


def test_strict_ap50_rejects_missing_duplicate_and_nonfinite_classes() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        extract_map50_per_class(_metrics([0.5], [14]), 20, strict=True)
    with pytest.raises(ValueError, match="duplicate"):
        extract_map50_per_class(_metrics([0.4, 0.5], [3, 3]), 20, strict=True)
    values = np.full(20, 0.5)
    values[2] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        extract_map50_per_class(_metrics(values), 20, strict=True)


def test_full_metric_summary_contains_named_voc20_macro() -> None:
    values = np.linspace(0.2, 0.8, 20)
    summary = _extract_metrics_dict(
        _metrics(values),
        20,
        14,
        strict=True,
    )
    assert summary["mAP50_target"] == pytest.approx(values[14])
    assert summary["mAP50_all"] == pytest.approx(float(np.mean(values)))
    assert len(summary["ap50_by_class"]) == 20


def _arm(tag: str, values, poisoned_count: int):
    return {
        "method": "sirc_malc_cgr",
        "steps": 40,
        "seed": 0,
        "run_tag": tag,
        "voc20_class_names": list(VOC20_CLASS_NAMES),
        "ap50_by_class": dict(zip(VOC20_CLASS_NAMES, values)),
        "poisoned_count": poisoned_count,
        "actual_linf_max": 16 / 255 if tag == "M1" else 0.0,
        "quality_validation_gaps": ["LPIPS_dependency_unavailable"] if tag == "M1" else [],
    }


def test_c0_m1_comparison_reports_all_19_classes_and_frozen_directions() -> None:
    clean = np.full(20, 0.80)
    poison = np.full(20, 0.76)
    poison[14] = 0.40
    comparison = build_c0_m1_comparison(
        [_arm("C0", clean, 0), _arm("M1", poison, 6095)],
        method="sirc_malc_cgr",
        steps=40,
        seed=0,
    )
    assert len(comparison["per_class"]) == 20
    summary = comparison["summary"]
    assert summary["AP50_person_drop"] == pytest.approx(0.40)
    assert summary["mAP50_non_target_macro_drop"] == pytest.approx(0.04)
    assert summary["non_target_classes_drop_le_0_10"] == 19
    assert summary["fresh_victim_success"] is True
    assert summary["claim_status"] == "tentative_single_seed"


def test_c0_m1_comparison_fails_on_incomplete_mapping_or_wrong_count() -> None:
    clean = np.full(20, 0.80)
    poison = np.full(20, 0.76)
    c0 = _arm("C0", clean, 0)
    m1 = _arm("M1", poison, 6094)
    with pytest.raises(ValueError, match="6095"):
        build_c0_m1_comparison(
            [c0, m1], method="sirc_malc_cgr", steps=40, seed=0
        )
    m1 = _arm("M1", poison, 6095)
    del m1["ap50_by_class"]["dog"]
    with pytest.raises(ValueError, match="all VOC20"):
        build_c0_m1_comparison(
            [c0, m1], method="sirc_malc_cgr", steps=40, seed=0
        )


def _formal_context(tmp_path, *, run_tag="M1"):
    frozen = tmp_path / "mechanism" / "a1_frozen_carrier.pt"
    return SimpleNamespace(
        method="sirc_malc_cgr",
        run_tag=run_tag,
        cfg={
            "methods": {
                "sirc_malc_cgr": {
                    "frozen_carrier_state": str(frozen),
                    "semantic_bank_hash": "a" * 64,
                    "source_manifest_hash": "b" * 64,
                    "split_hash": "c" * 64,
                }
            }
        },
    )


def test_formal_manifest_provenance_includes_all_frozen_input_hashes(tmp_path) -> None:
    ctx = _formal_context(tmp_path)
    row = {
        "is_poisoned": "1",
        "state_content_hash": "d" * 64,
        "semantic_bank_hash": "a" * 64,
        "source_manifest_hash": "b" * 64,
        "split_hash": "c" * 64,
        "variant_index": "2",
        "support_source": "forced_pseudo_fallback",
    }
    provenance = _sirc_manifest_provenance(ctx, [row])
    assert provenance["source_manifest_hash"] == "b" * 64
    tampered = dict(row, source_manifest_hash="e" * 64)
    with pytest.raises(ValueError, match="source_manifest_hash"):
        _sirc_manifest_provenance(ctx, [tampered])


def test_m1_metrics_load_only_a_passing_heldout_mechanism_report(tmp_path) -> None:
    ctx = _formal_context(tmp_path)
    report_path = tmp_path / "mechanism" / "mechanism_report.json"
    report_path.parent.mkdir()
    report = {
        "schema_version": 1,
        "evidence_scope": "heldout_mechanism_only_not_fresh_victim_ue",
        "split_hash": "c" * 64,
        "A0": {"residual_cosine_median": 0.1},
        "A1": {"residual_cosine_median": 0.3},
        "gate": {"pass": True, "allow_fresh_victim": True},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    evidence = _sirc_mechanism_evidence(ctx)["mechanism_diagnostics"]
    assert evidence["evidence_scope"] == report["evidence_scope"]
    assert evidence["gate"]["pass"] is True

    report["gate"]["pass"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="passing MALC mechanism gate"):
        _sirc_mechanism_evidence(ctx)
