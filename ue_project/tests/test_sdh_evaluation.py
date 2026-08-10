from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ue_framework.metrics_utils import VOC20_CLASS_NAMES
from ue_framework.methods.sdh_evaluation import (
    build_learning_preference_audit,
    build_sdh_counterfactual_metrics,
    build_sdh_fresh_victim_comparison,
)
from ue_framework.methods.sdh_counterfactual import (
    aggregate_person_classification_losses,
    build_person_free_transplant_metrics,
    deterministic_person_audit_subset,
)
from ue_framework.stages.evaluate import (
    _sdh_manifest_provenance,
    _sdh_mechanism_evidence,
)


def _metrics(value=0.8, person=None):
    values = {name: float(value) for name in VOC20_CLASS_NAMES}
    if person is not None:
        values["person"] = float(person)
    return {"ap50_by_class": values}


def test_counterfactual_clean_is_primary_and_person_gap_is_correct() -> None:
    result = build_sdh_counterfactual_metrics(
        _metrics(0.75, person=0.30),
        _metrics(0.74, person=0.55),
    )
    assert result["primary_split"] == "clean_val"
    assert result["person_ap50_recovery"] == pytest.approx(0.25)
    assert result["shortcut_recovery_ge_0_20"] is True
    assert result["non_target_macro_shift"] == pytest.approx(-0.01)


def test_learning_preference_uses_frozen_formula_and_required_epochs() -> None:
    audit = build_learning_preference_audit(
        {
            1: {"clean_counterfactual_loss": 1.0, "carrier_loss": 0.95},
            5: {"clean_counterfactual_loss": 0.8, "carrier_loss": 0.7},
            10: {"clean_counterfactual_loss": 0.6, "carrier_loss": 0.5},
            20: {"clean_counterfactual_loss": 0.5, "carrier_loss": 0.4},
        }
    )
    rows = {row["epoch"]: row for row in audit["rows"]}
    assert rows[10]["R_e"] == pytest.approx(1.0 / 6.0)
    assert audit["R_10_positive"] is True
    assert audit["R_20_ge_0_10"] is True
    assert audit["used_for_checkpoint_selection"] is False


def test_fresh_victim_comparison_reports_19_classes_and_all_gates() -> None:
    clean = _metrics(0.8, person=0.8)
    poison = _metrics(0.76, person=0.4)
    counterfactual = build_sdh_counterfactual_metrics(
        poison, _metrics(0.76, person=0.65)
    )
    dynamics = build_learning_preference_audit(
        {
            1: {"clean_counterfactual_loss": 1.0, "carrier_loss": 0.9},
            5: {"clean_counterfactual_loss": 0.8, "carrier_loss": 0.7},
            10: {"clean_counterfactual_loss": 0.6, "carrier_loss": 0.5},
            20: {"clean_counterfactual_loss": 0.5, "carrier_loss": 0.4},
        }
    )
    comparison = build_sdh_fresh_victim_comparison(
        clean, poison, counterfactual, dynamics
    )
    assert len(comparison["per_class"]) == 20
    assert sum(not row["is_target"] for row in comparison["per_class"]) == 19
    assert comparison["summary"]["pass"] is True


def test_missing_voc_class_or_epoch_fails_closed() -> None:
    incomplete = _metrics()
    del incomplete["ap50_by_class"]["dog"]
    with pytest.raises(ValueError, match="all named VOC20"):
        build_sdh_counterfactual_metrics(incomplete, _metrics())
    with pytest.raises(ValueError, match="epochs 1/5/10/20"):
        build_learning_preference_audit(
            {1: {"clean_counterfactual_loss": 1, "carrier_loss": 1}}
        )


def _sdh_ctx(tmp_path, *, run_tag="P1-V"):
    hashes = {
        "secret_tensor_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "train_split_sha256": "3" * 64,
        "secret_source_sha256": "4" * 64,
    }
    return SimpleNamespace(
        method="tausb_sdh",
        run_tag=run_tag,
        cfg={
            "methods": {
                "tausb_sdh": {
                    **hashes,
                    "frozen_sdh_state": str(tmp_path / "p1_frozen_sdh_state.pt"),
                }
            }
        },
    )


def test_sdh_manifest_provenance_requires_single_secret_and_gt_box(tmp_path) -> None:
    ctx = _sdh_ctx(tmp_path)
    row = {
        "is_poisoned": 1,
        "state_content_hash": "5" * 64,
        "semantic_bank_hash": "1" * 64,
        "source_manifest_hash": "2" * 64,
        "split_hash": "3" * 64,
        "secret_source_sha256": "4" * 64,
        "support_source": "person_gt_bbox",
    }
    result = _sdh_manifest_provenance(ctx, [row])
    assert result["single_final_secret"] is True
    assert result["support_source"] == "person_gt_bbox"
    bad = dict(row, support_source="forced_pseudo_fallback")
    with pytest.raises(ValueError, match="person_gt_bbox"):
        _sdh_manifest_provenance(ctx, [bad])


def test_sdh_mechanism_evidence_requires_passing_four_arm_report(tmp_path) -> None:
    ctx = _sdh_ctx(tmp_path)
    report = {
        "schema": "tausb.sdh-mechanism-pilot.v1",
        "split_hash": "6" * 64,
        "arms": {"T0": {}, "T1": {}, "P0": {}, "P1": {}},
        "decision": {"pass": True, "target_pass": True, "protection_pass": True},
    }
    (tmp_path / "mechanism_metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    result = _sdh_mechanism_evidence(ctx)["sdh_mechanism_diagnostics"]
    assert result["decision"]["pass"] is True
    report["decision"]["protection_pass"] = False
    (tmp_path / "mechanism_metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sub-gates"):
        _sdh_mechanism_evidence(ctx)


def test_person_loss_aggregation_weights_real_positive_counts() -> None:
    result = aggregate_person_classification_losses(
        [
            {
                "clean_counterfactual_loss": 1.0,
                "carrier_loss": 0.5,
                "person_positive_count": 2,
            },
            {
                "clean_counterfactual_loss": 2.0,
                "carrier_loss": 1.0,
                "person_positive_count": 1,
            },
        ]
    )
    assert result["clean_counterfactual_loss"] == pytest.approx(4.0 / 3.0)
    assert result["carrier_loss"] == pytest.approx(2.0 / 3.0)
    assert result["person_positive_count"] == 3


def test_person_free_transplant_is_descriptive_and_paired() -> None:
    result = build_person_free_transplant_metrics([0.1, 0.3], [0.4, 0.2])
    assert result["mean_confidence_shift"] == pytest.approx(0.1)
    assert result["false_positive_image_count_shift"] == 0
    assert "descriptive" in result["claim_boundary"]
    with pytest.raises(ValueError, match="paired"):
        build_person_free_transplant_metrics([0.1], [0.1, 0.2])


def test_person_audit_subset_is_balanced_deterministic_and_hashed(tmp_path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    paths = []
    for index, label in enumerate(("14 .5 .5 .2 .2", "14 .5 .5 .2 .2\n7 .2 .2 .1 .1") * 2):
        path = image_dir / ("%03d.jpg" % index)
        path.write_bytes(b"image")
        (label_dir / ("%03d.txt" % index)).write_text(label, encoding="utf-8")
        paths.append(path)
    first, first_hash = deterministic_person_audit_subset(
        paths, label_dir=label_dir, target_class_id=14, per_stratum=2
    )
    second, second_hash = deterministic_person_audit_subset(
        list(reversed(paths)), label_dir=label_dir, target_class_id=14, per_stratum=2
    )
    assert first == second
    assert first_hash == second_hash
    assert len(first) == 4
