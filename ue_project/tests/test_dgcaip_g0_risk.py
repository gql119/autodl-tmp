from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config
from ue_framework.tools.run_tausb_dgcaip_c0_snapshots import _file_sha256
from ue_framework.tools.run_tausb_dgcaip_g0_risk import (
    RUN_MODE,
    SPEC_ID,
    WALL_SECONDS,
    _validate_binding,
    _verify_result,
)


CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_dataset_cgr_proxy_g0_risk_v1.yaml"
)

EXPECTED_SNAPSHOTS = {
    "e1": "6ebacf59d7fa27ae8d30bb86571d5f089392e19d52ba9ffd7fd204faa70c5ae1",
    "e5": "cfaf454563e7ac81676468ec09fb08a94718a9902c5ee7057ee3db0d63202fc4",
    "e20": "e660ed4b2f36e8b866f89a4f88a02e3d3a7eed6f2727f99573cc3c4d8bfaad53",
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _bound_fixture(tmp_path: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    snapshots = []
    manifest_snapshots = {}
    for snapshot_id in ("e1", "e5", "e20"):
        checkpoint = tmp_path / (snapshot_id + ".pt")
        checkpoint.write_bytes(("checkpoint-" + snapshot_id).encode("ascii"))
        digest = _file_sha256(checkpoint)
        snapshots.append(
            {"id": snapshot_id, "checkpoint": str(checkpoint), "sha256": digest}
        )
        manifest_snapshots[snapshot_id] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
        }
    config["model"]["protection_surrogate_snapshots"] = snapshots

    source_p1 = tmp_path / "p1_state.pt"
    source_p1.write_bytes(b"p1-state")
    config["dgcaip"]["source_p1_state"] = str(source_p1)
    config["dgcaip"]["source_p1_state_sha256"] = _file_sha256(source_p1)

    manifest_path = tmp_path / "c0_snapshot_manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "passed",
            "spec_id": SPEC_ID,
            "execution_commit": "f5e223a73d17939402de7613f2152e50a77b07b8",
            "snapshots": manifest_snapshots,
        },
    )
    config["bindings"].update(
        {
            "c0_snapshot_manifest": str(manifest_path),
            "c0_snapshot_manifest_sha256": _file_sha256(manifest_path),
            "c0_execution_commit": "f5e223a73d17939402de7613f2152e50a77b07b8",
        }
    )
    config_path = tmp_path / "g0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def test_g0_config_freezes_exact_c0_snapshot_binding() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["spec"]["spec_id"] == SPEC_ID
    assert config["dgcaip"]["run_mode"] == RUN_MODE
    assert config["runtime"]["device"] == "cuda:0"
    assert config["runtime"]["artifact_root"].endswith("G0-RISK-R1")
    assert {
        item["id"]: item["sha256"]
        for item in config["model"]["protection_surrogate_snapshots"]
    } == EXPECTED_SNAPSHOTS
    assert config["bindings"]["c0_execution_commit"] == (
        "f5e223a73d17939402de7613f2152e50a77b07b8"
    )
    assert WALL_SECONDS == 65 * 60


def test_g0_binding_accepts_exact_files_and_rejects_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    config, config_path = _bound_fixture(tmp_path)
    result = _validate_binding(config, config_path)
    assert result["snapshot_sha256"] == {
        item["id"]: item["sha256"]
        for item in config["model"]["protection_surrogate_snapshots"]
    }

    Path(config["model"]["protection_surrogate_snapshots"][1]["checkpoint"]).write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="snapshot file hash mismatch: e5"):
        _validate_binding(config, config_path)


def test_g0_binding_rejects_manifest_execution_commit_mismatch(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    manifest_path = Path(config["bindings"]["c0_snapshot_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_commit"] = "0" * 40
    _write_json(manifest_path, manifest)
    config["bindings"]["c0_snapshot_manifest_sha256"] = _file_sha256(manifest_path)
    with pytest.raises(ValueError, match="execution commit mismatch"):
        _validate_binding(config, config_path)


def test_g0_result_is_preserved_when_scientific_gate_is_false(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifact_root = tmp_path / "artifacts"
    config["runtime"]["artifact_root"] = str(artifact_root)
    config_path = tmp_path / "g0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_root = artifact_root / RUN_MODE
    output_root.mkdir(parents=True)
    bank_path = output_root / "dgcaip_risk_bank.json"
    replay_path = output_root / "dgcaip_replay_manifest.json"
    raw_path = output_root / "dgcaip_risk_records.jsonl"
    _write_json(bank_path, {"bank": "ok"})
    _write_json(replay_path, {"replay": "ok"})
    raw_path.write_text('{"record": 1}\n', encoding="utf-8")
    snapshot_hashes = {
        item["id"]: item["sha256"]
        for item in config["model"]["protection_surrogate_snapshots"]
    }
    _write_json(
        output_root / "dgcaip_risk_manifest.json",
        {
            "coverage": 0.91,
            "snapshot_sha256": snapshot_hashes,
            "risk_bank_file_sha256": _file_sha256(bank_path),
            "replay_manifest_sha256": _file_sha256(replay_path),
            "expected_instance_count": 100,
            "covered_instance_count": 91,
            "person_cooccurrence_image_count": 50,
            "decision": {"pass": False},
        },
    )
    result = _verify_result(config_path)
    assert result["coverage"] == pytest.approx(0.91)
    assert result["gate_pass"] is False
    assert result["covered_instance_count"] == 91


def test_g0_result_rejects_risk_bank_tamper(tmp_path: Path) -> None:
    config, _ = _bound_fixture(tmp_path)
    artifact_root = tmp_path / "result"
    config["runtime"]["artifact_root"] = str(artifact_root)
    config_path = tmp_path / "result.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_root = artifact_root / RUN_MODE
    output_root.mkdir(parents=True)
    bank_path = output_root / "dgcaip_risk_bank.json"
    replay_path = output_root / "dgcaip_replay_manifest.json"
    raw_path = output_root / "dgcaip_risk_records.jsonl"
    bank_path.write_text("bank", encoding="utf-8")
    replay_path.write_text("replay", encoding="utf-8")
    raw_path.write_text("record\n", encoding="utf-8")
    _write_json(
        output_root / "dgcaip_risk_manifest.json",
        {
            "coverage": 1.0,
            "snapshot_sha256": {
                item["id"]: item["sha256"]
                for item in config["model"]["protection_surrogate_snapshots"]
            },
            "risk_bank_file_sha256": _file_sha256(bank_path),
            "replay_manifest_sha256": _file_sha256(replay_path),
            "expected_instance_count": 1,
            "covered_instance_count": 1,
            "person_cooccurrence_image_count": 1,
            "decision": {"pass": True},
        },
    )
    bank_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="risk bank file hash differs"):
        _verify_result(config_path)
