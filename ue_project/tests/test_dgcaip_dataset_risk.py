from __future__ import annotations

import random

import pytest

from ue_framework.methods.dgcaip_dataset_risk import (
    DGCAIPInstanceKey,
    DGCAIPRiskRecord,
    build_balanced_replay_image_ids,
    build_dataset_risk_bank,
    load_risk_bank,
    risk_bank_payload,
    write_risk_bank,
)


SPEC_ID = "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"


def _records():
    keys = [
        DGCAIPInstanceKey("a", 1, 1),
        DGCAIPInstanceKey("b", 1, 1),
        DGCAIPInstanceKey("c", 1, 1),
        DGCAIPInstanceKey("d", 2, 7),
        DGCAIPInstanceKey("e", 2, 7),
    ]
    output = []
    for snapshot_index, snapshot_id in enumerate(("e1", "e5", "e20")):
        for index, key in enumerate(keys):
            output.append(
                DGCAIPRiskRecord(
                    key=key,
                    snapshot_id=snapshot_id,
                    js_divergence=float(index + snapshot_index),
                    clean_to_poison_kl=float(2 * index + snapshot_index),
                )
            )
    return keys, output


def test_dataset_risk_bank_is_classwise_deterministic_and_selects_each_class() -> None:
    keys, records = _records()
    bank = build_dataset_risk_bank(
        records,
        spec_id=SPEC_ID,
        expected_snapshot_ids=("e1", "e5", "e20"),
        expected_instance_keys=keys,
    )
    shuffled = list(records)
    random.Random(17).shuffle(shuffled)
    replay = build_dataset_risk_bank(
        shuffled,
        spec_id=SPEC_ID,
        expected_snapshot_ids=("e1", "e5", "e20"),
        expected_instance_keys=list(reversed(keys)),
    )
    assert bank.canonical_sha256 == replay.canonical_sha256
    assert risk_bank_payload(bank) == risk_bank_payload(replay)
    assert bank.coverage == pytest.approx(1.0)
    assert bank.class_counts == {1: 3, 7: 2}
    selected_classes = {
        entry.key.class_id for entry in bank.entries if entry.high_risk
    }
    assert selected_classes == {1, 7}
    assert len(bank.high_risk_keys()) == 2


def test_tied_values_receive_stable_midrank() -> None:
    keys = [DGCAIPInstanceKey(name, 0, 3) for name in ("a", "b", "c")]
    bank = build_dataset_risk_bank(
        [
            DGCAIPRiskRecord(keys[0], "only", 1.0, 1.0),
            DGCAIPRiskRecord(keys[1], "only", 1.0, 1.0),
            DGCAIPRiskRecord(keys[2], "only", 3.0, 3.0),
        ],
        spec_id=SPEC_ID,
        expected_snapshot_ids=("only",),
    )
    ranks = bank.rank_mapping()
    assert ranks[keys[0].as_tuple()] == pytest.approx(0.25)
    assert ranks[keys[1].as_tuple()] == pytest.approx(0.25)
    assert ranks[keys[2].as_tuple()] == pytest.approx(1.0)


def test_dataset_risk_bank_rejects_duplicates_unknown_snapshots_and_low_coverage() -> None:
    key = DGCAIPInstanceKey("a", 0, 1)
    record = DGCAIPRiskRecord(key, "e1", 0.1, 0.2)
    with pytest.raises(ValueError, match="Duplicate"):
        build_dataset_risk_bank(
            [record, record],
            spec_id=SPEC_ID,
            expected_snapshot_ids=("e1",),
        )
    with pytest.raises(ValueError, match="unexpected snapshot"):
        build_dataset_risk_bank(
            [record],
            spec_id=SPEC_ID,
            expected_snapshot_ids=("e5",),
        )
    with pytest.raises(ValueError, match="coverage"):
        build_dataset_risk_bank(
            [record],
            spec_id=SPEC_ID,
            expected_snapshot_ids=("e1",),
            expected_instance_keys=(key, DGCAIPInstanceKey("b", 0, 1)),
            minimum_coverage=0.9,
        )


def test_risk_bank_round_trip_verifies_canonical_hash(tmp_path) -> None:
    keys, records = _records()
    bank = build_dataset_risk_bank(
        records,
        spec_id=SPEC_ID,
        expected_snapshot_ids=("e1", "e5", "e20"),
        expected_instance_keys=keys,
    )
    path = tmp_path / "bank.json"
    write_risk_bank(path, bank)
    restored = load_risk_bank(
        path,
        expected_spec_id=SPEC_ID,
        expected_sha256=bank.canonical_sha256,
    )
    assert restored == bank
    path.write_text(
        path.read_text().replace(SPEC_ID, SPEC_ID + "-tampered"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash"):
        load_risk_bank(path)


def test_balanced_replay_is_deterministic_and_has_exact_slots() -> None:
    keys, records = _records()
    bank = build_dataset_risk_bank(
        records,
        spec_id=SPEC_ID,
        expected_snapshot_ids=("e1", "e5", "e20"),
        expected_instance_keys=keys,
    )
    images = tuple(key.image_id for key in keys)
    first = build_balanced_replay_image_ids(bank, images, total_slots=8, seed=3)
    second = build_balanced_replay_image_ids(bank, images, total_slots=8, seed=3)
    assert first == second
    assert len(first) == 8
    high_images = {entry.key.image_id for entry in bank.entries if entry.high_risk}
    assert sum(image_id in high_images for image_id in first[1::2]) == 4
