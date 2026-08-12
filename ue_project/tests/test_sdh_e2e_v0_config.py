from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ue_framework.config import load_config
from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config
from ue_framework.stages.generate import (
    MANIFEST_FIELDS,
    _canonical_json_sha256,
    resolve_train_image_selection,
)
from ue_framework.tools.bind_tausb_sdh_e2e_v0 import _bound_config


ROOT = Path(__file__).parents[1]
FORMAL = ROOT / "ue_framework" / "configs" / (
    "exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
)
MECHANISM = ROOT / "ue_framework" / "configs" / "tausb_sdh_e2e_v0_mechanism.yaml"
HASH_KEYS = (
    "frozen_sdh_state_sha256",
    "hiding_metrics_sha256",
    "hiding_checkpoint_sha256",
    "hiding_split_sha256",
    "mechanism_metrics_sha256",
    "mechanism_decision_sha256",
    "mechanism_config_sha256",
    "p1_state_sha256",
)


def test_sdh_manifest_schema_preserves_secret_source_hash() -> None:
    assert "secret_source_sha256" in MANIFEST_FIELDS


def _hashes() -> dict:
    return {name: ("%x" % (index + 1)) * 64 for index, name in enumerate(HASH_KEYS)}


def _write_bound_config(tmp_path, *, pilot_kind: str, arm_id: str) -> Path:
    base = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    config = _bound_config(
        base,
        arm_id=arm_id,
        pilot_kind=pilot_kind,
        dataset_root=tmp_path / "voc",
        state_path=tmp_path / "p1_feasibility_sdh_state.pt",
        hashes=_hashes(),
        selection_path=str(tmp_path / "selection.json"),
        selection_hash="a" * 64,
        run_root=str(tmp_path / (pilot_kind + "-" + arm_id)),
    )
    path = tmp_path / (pilot_kind + "-" + arm_id + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("pilot_kind", "arm_id", "epochs", "count", "ratio"),
    [
        ("smoke", "C0", 1, 0, 0.0),
        ("smoke", "M1", 1, 40, 1.0),
        ("e20", "C0", 20, 0, 0.0),
        ("e20", "M1", 20, 6095, 1.0),
    ],
)
def test_bound_v0_pair_configs_validate_and_keep_arm_identity(
    tmp_path, pilot_kind, arm_id, epochs, count, ratio
) -> None:
    path = _write_bound_config(tmp_path, pilot_kind=pilot_kind, arm_id=arm_id)
    config = load_config(str(path))
    assert config["victim"]["epochs"] == epochs
    assert config["experiment"]["expected_poisoned_count"] == count
    assert config["experiment"]["poisoning_ratio"] == ratio
    assert config["methods"]["tausb_sdh"]["require_hiding_gate_pass"] is False
    assert config["methods"]["tausb_sdh"]["require_mechanism_gate_pass"] is False
    if pilot_kind == "smoke":
        assert config["data"]["train_selection_manifest"]
        assert config["data"]["materialization_layout"] == "full_png_v1"
    else:
        assert config["data"]["train_selection_manifest"] == ""
        assert config["data"]["materialization_layout"] == "sparse_mixed_list_v1"


def test_legacy_e20_full_png_config_remains_loadable(tmp_path) -> None:
    path = _write_bound_config(tmp_path, pilot_kind="e20", arm_id="M1")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["data"]["materialization_layout"] = "full_png_v1"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = load_config(str(path))
    assert loaded["data"]["materialization_layout"] == "full_png_v1"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("experiment", "poisoning_ratio", 0.5), "poisoning_ratio"),
        (("experiment", "expected_poisoned_count", 41), "expected_poisoned_count"),
        (("victim", "epochs", 20), "victim epochs"),
        (("methods", "support_type", "mask"), "person GT bbox"),
        (("methods", "enable_cgr", False), "all method gates"),
        (("methods", "binding_status", "pending"), "not bound"),
        (("methods", "p1_state_sha256", "pending"), "SHA-256 p1_state"),
    ],
)
def test_v0_config_fails_closed_on_protocol_mutation(tmp_path, mutation, match) -> None:
    path = _write_bound_config(tmp_path, pilot_kind="smoke", arm_id="M1")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    section, key, value = mutation
    if section == "methods":
        config[section]["tausb_sdh"][key] = value
    else:
        config[section][key] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(str(path))


def test_v0_mechanism_config_binds_frozen_r2_and_fresh_output() -> None:
    config = yaml.safe_load(MECHANISM.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["hiding"]["source_checkpoint_sha256"].startswith("a765e27a")
    assert config["hiding"]["source_metrics_sha256"].startswith("c7d1b120")
    assert config["hiding"]["source_artifact_root"] != config["runtime"]["artifact_root"]


def test_selection_manifest_is_hash_and_label_bound(tmp_path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    images = []
    records = []
    for stem, label, has_target in (
        ("000001", "14 0.5 0.5 0.2 0.2\n", True),
        ("000002", "7 0.5 0.5 0.2 0.2\n", False),
    ):
        image_path = image_dir / (stem + ".jpg")
        image_path.write_bytes(b"image-" + stem.encode("ascii"))
        label_path = label_dir / (stem + ".txt")
        label_path.write_text(label, encoding="utf-8")
        images.append(str(image_path))
        records.append(
            {
                "stem": stem,
                "has_target": has_target,
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": "tausb.sdh-e2e-v0-train-selection.v1",
        "records": records,
    }
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = _canonical_json_sha256(manifest)

    selected, actual_hash = resolve_train_image_selection(
        images,
        train_label_dir=str(label_dir),
        selection_manifest_path=str(manifest_path),
        expected_manifest_sha256=manifest_hash,
        target_class_id=14,
    )
    assert [Path(path).stem for path in selected] == ["000001", "000002"]
    assert actual_hash == manifest_hash

    tampered = copy.deepcopy(manifest)
    tampered["records"][0]["label_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="label hash mismatch"):
        resolve_train_image_selection(
            images,
            train_label_dir=str(label_dir),
            selection_manifest_path=str(manifest_path),
            expected_manifest_sha256=_canonical_json_sha256(tampered),
            target_class_id=14,
        )
