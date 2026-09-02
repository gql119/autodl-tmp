from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ue_framework.methods.dgcaip_dataset_risk import (
    DGCAIPInstanceKey,
    DGCAIPRiskRecord,
    build_dataset_risk_bank,
    write_risk_bank,
)
from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config
from ue_framework.tools.run_tausb_dgcaip_c0_snapshots import _file_sha256
from ue_framework.tools.run_tausb_dgcaip_g1_strict import (
    RUN_MODE,
    SPEC_ID,
    V2_SPEC_ID,
    V3_SPEC_ID,
    V4_SPEC_ID,
    WALL_SECONDS,
    _minimum_free_bytes,
    _validate_binding,
    _verify_result,
)


CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_dataset_cgr_proxy_g1_strict_v1.yaml"
)
V2_CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_strict_route_v2_g1.yaml"
)
V3_CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_component_route_v3_g1.yaml"
)
V4_CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_relaxed_gate_v4_g1.yaml"
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _bound_fixture(tmp_path: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    snapshot_hashes = {}
    snapshots = []
    for snapshot_id in ("e1", "e5", "e20"):
        path = tmp_path / (snapshot_id + ".pt")
        path.write_bytes(("snapshot-" + snapshot_id).encode("ascii"))
        digest = _file_sha256(path)
        snapshots.append({"id": snapshot_id, "checkpoint": str(path), "sha256": digest})
        snapshot_hashes[snapshot_id] = digest
    config["model"]["protection_surrogate_snapshots"] = snapshots

    source_p1 = tmp_path / "p1.pt"
    source_p1.write_bytes(b"p1")
    source_p1_hash = _file_sha256(source_p1)
    config["dgcaip"]["source_p1_state"] = str(source_p1)
    config["dgcaip"]["source_p1_state_sha256"] = source_p1_hash

    keys = [DGCAIPInstanceKey("image-%d" % index, 0, 1) for index in range(4)]
    records = [
        DGCAIPRiskRecord(key, snapshot_id, float(index), float(index + 1))
        for snapshot_id in ("e1", "e5", "e20")
        for index, key in enumerate(keys)
    ]
    bank = build_dataset_risk_bank(
        records,
        spec_id=SPEC_ID,
        expected_snapshot_ids=("e1", "e5", "e20"),
        expected_instance_keys=keys,
    )
    bank_path = tmp_path / "risk_bank.json"
    write_risk_bank(bank_path, bank)
    replay_path = tmp_path / "replay.json"
    _write_json(
        replay_path,
        {
            "schema": "tausb.dgcaip-dataset-replay.v1",
            "spec_id": SPEC_ID,
            "risk_bank_canonical_sha256": bank.canonical_sha256,
            "image_ids": ["image-%d" % (index % 4) for index in range(32)],
        },
    )
    config["dataset_ranking"].update(
        {
            "risk_bank": str(bank_path),
            "risk_bank_file_sha256": _file_sha256(bank_path),
            "risk_bank_canonical_sha256": bank.canonical_sha256,
            "replay_manifest": str(replay_path),
            "replay_manifest_file_sha256": _file_sha256(replay_path),
        }
    )

    manifest_path = tmp_path / "risk_manifest.json"
    _write_json(
        manifest_path,
        {
            "spec_id": SPEC_ID,
            "coverage": 1.0,
            "decision": {"pass": True},
            "snapshot_sha256": snapshot_hashes,
            "source_carrier_state_sha256": source_p1_hash,
            "risk_bank_file_sha256": _file_sha256(bank_path),
            "risk_bank_canonical_sha256": bank.canonical_sha256,
            "replay_manifest_sha256": _file_sha256(replay_path),
        },
    )
    controller_path = tmp_path / "controller.json"
    _write_json(
        controller_path,
        {
            "status": "completed",
            "execution_commit": "cc0f9b42e265100a835985bfc4ab3e95411470dd",
            "result": {
                "risk_manifest_sha256": _file_sha256(manifest_path),
                "gate_pass": True,
            },
        },
    )
    config["bindings"].update(
        {
            "g0_risk_manifest": str(manifest_path),
            "g0_risk_manifest_sha256": _file_sha256(manifest_path),
            "g0_controller_status": str(controller_path),
            "g0_controller_status_sha256": _file_sha256(controller_path),
        }
    )
    config_path = tmp_path / "g1.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def test_g1_config_freezes_strict_route_and_exact_g0_artifacts() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["spec"]["spec_id"] == SPEC_ID
    assert config["dgcaip"]["run_mode"] == RUN_MODE
    assert config["mechanism"]["optimization_steps"] == 8
    assert config["mechanism"]["max_seconds"] == 1100
    assert config["dataset_ranking"]["risk_bank_file_sha256"] == (
        "21cf001ed69b030a6dce1a7e9ea67b07de45f0f41189a9c828fe9b9e3488fabe"
    )
    assert config["dataset_ranking"]["risk_bank_canonical_sha256"] == (
        "3dcc755fc7629cc5d2b37bd7b6931088001bf0ca0d3976343d7420d4236eb5fc"
    )
    assert config["dataset_ranking"]["replay_manifest_file_sha256"] == (
        "e5dd31cac06d038f4fc305970a9a60a2e2f34b3ae61e55af7405568fbbb7e457"
    )
    assert WALL_SECONDS == 20 * 60


def test_v4_right_sizes_only_the_short_g1_disk_gate() -> None:
    assert _minimum_free_bytes(V4_SPEC_ID) == 4 * 1024 ** 3
    assert _minimum_free_bytes(V3_SPEC_ID) == 5 * 1024 ** 3
    assert _minimum_free_bytes(SPEC_ID) == 5 * 1024 ** 3


def test_g1_v2_config_replaces_only_the_strict_route_geometry() -> None:
    legacy = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    revised = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(revised)
    assert revised["spec"]["spec_id"] == V2_SPEC_ID
    assert revised["strict_route"] == {
        "mode": "nonworsening_target_progress_v2",
        "repair_floor_fraction": 0.0,
        "max_repair_norm_ratio": 0.0,
        "minimum_target_progress": 0.60,
        "max_projection_iterations": 128,
        "svd_relative_tolerance": 1.0e-6,
    }
    for section in ("dataset", "model", "secrets", "hiding", "mechanism", "dgcaip", "dataset_ranking", "proxy_agreement", "bindings"):
        assert revised[section] == legacy[section]
    assert revised["runtime"]["artifact_root"].endswith("SCGR-V2-G1-R1")


def test_g1_v3_config_changes_only_identity_root_and_route_mode() -> None:
    legacy = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    revised = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(revised)
    assert revised["spec"]["spec_id"] == V3_SPEC_ID
    assert revised["strict_route"] == {
        "mode": "component_aligned_target_progress_v3",
        "repair_floor_fraction": 0.0,
        "max_repair_norm_ratio": 0.0,
        "minimum_target_progress": 0.60,
        "max_projection_iterations": 128,
        "svd_relative_tolerance": 1.0e-6,
    }
    for section in (
        "dataset",
        "model",
        "secrets",
        "hiding",
        "mechanism",
        "dgcaip",
        "dataset_ranking",
        "proxy_agreement",
        "bindings",
    ):
        assert revised[section] == legacy[section]
    assert revised["runtime"]["artifact_root"].endswith("SCGR-V3-G1-R1")


def test_g1_v4_config_changes_only_gate_identity_and_registered_relaxations() -> None:
    legacy = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    revised = yaml.safe_load(V4_CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(revised)
    assert revised["spec"]["spec_id"] == V4_SPEC_ID
    assert revised["strict_route"] == {
        **legacy["strict_route"],
        "nonlinear_comparison_tolerance": 1.0e-6,
        "minimum_accepted_update_ratio": 0.50,
    }
    for section in (
        "dataset",
        "model",
        "secrets",
        "hiding",
        "mechanism",
        "dgcaip",
        "dataset_ranking",
        "proxy_agreement",
        "bindings",
    ):
        assert revised[section] == legacy[section]
    assert revised["runtime"]["artifact_root"].endswith("SCGR-V4-G1-R1")


def test_g1_binding_accepts_v2_with_the_frozen_v1_g0_chain(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V2_SPEC_ID
    config["strict_route"] = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))[
        "strict_route"
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = _validate_binding(config, config_path)
    assert result["spec_id"] == V2_SPEC_ID
    assert result["risk_bank_canonical_sha256"] == config["dataset_ranking"][
        "risk_bank_canonical_sha256"
    ]


def test_g1_binding_accepts_v3_with_the_frozen_v1_g0_chain(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V3_SPEC_ID
    config["strict_route"] = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))[
        "strict_route"
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = _validate_binding(config, config_path)
    assert result["spec_id"] == V3_SPEC_ID
    assert result["replay_slots"] == 32


def test_g1_binding_accepts_v4_with_the_frozen_v1_g0_chain(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V4_SPEC_ID
    config["strict_route"] = yaml.safe_load(
        V4_CONFIG.read_text(encoding="utf-8")
    )["strict_route"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = _validate_binding(config, config_path)
    assert result["spec_id"] == V4_SPEC_ID
    assert result["replay_slots"] == 32


def test_g1_binding_accepts_exact_g0_chain_and_rejects_snapshot_tamper(
    tmp_path: Path,
) -> None:
    config, config_path = _bound_fixture(tmp_path)
    result = _validate_binding(config, config_path)
    assert result["coverage"] == pytest.approx(1.0)
    assert result["replay_slots"] == 32
    assert result["risk_bank_canonical_sha256"] == config["dataset_ranking"][
        "risk_bank_canonical_sha256"
    ]

    Path(config["model"]["protection_surrogate_snapshots"][0]["checkpoint"]).write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="snapshot file hash mismatch: e1"):
        _validate_binding(config, config_path)


def test_g1_binding_rejects_failed_g0_controller(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    controller_path = Path(config["bindings"]["g0_controller_status"])
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["result"]["gate_pass"] = False
    _write_json(controller_path, controller)
    config["bindings"]["g0_controller_status_sha256"] = _file_sha256(controller_path)
    with pytest.raises(ValueError, match="terminal binding did not pass"):
        _validate_binding(config, config_path)


def test_g1_result_preserves_scientific_gate_failure(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    config["runtime"]["artifact_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    strict_root = output_root / RUN_MODE
    strict_root.mkdir(parents=True)
    trace_path = strict_root / "backtracking_trace.json"
    state_path = strict_root / "p5_dataset_strict_state.pt"
    _write_json(trace_path, {"trace": []})
    state_path.write_bytes(b"candidate")
    _write_json(
        strict_root / "mechanism_metrics.json",
        {
            "schema": "tausb.dgcaip-dataset-strict-mechanism.v1",
            "spec_id": SPEC_ID,
            "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
            "risk_bank_canonical_sha256": config["dataset_ranking"][
                "risk_bank_canonical_sha256"
            ],
            "risk_bank_file_sha256": config["dataset_ranking"][
                "risk_bank_file_sha256"
            ],
            "replay_manifest_file_sha256": config["dataset_ranking"][
                "replay_manifest_file_sha256"
            ],
            "protection_snapshot_sha256": {
                item["id"]: item["sha256"]
                for item in config["model"]["protection_surrogate_snapshots"]
            },
            "elapsed_seconds": 12.0,
            "decision": {"checks": {"finite": True, "attack_retention": False}, "pass": False},
        },
    )
    result = _verify_result(config_path)
    assert result["gate_pass"] is False
    assert result["checks"]["attack_retention"] is False


def test_g1_result_verifies_v2_schema_and_route_mode(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V2_SPEC_ID
    config["strict_route"] = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))[
        "strict_route"
    ]
    output_root = tmp_path / "outputs-v2"
    config["runtime"]["artifact_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    strict_root = output_root / RUN_MODE
    strict_root.mkdir(parents=True)
    _write_json(strict_root / "backtracking_trace.json", {"trace": []})
    (strict_root / "p5_dataset_strict_state.pt").write_bytes(b"candidate-v2")
    _write_json(
        strict_root / "mechanism_metrics.json",
        {
            "schema": "tausb.dgcaip-dataset-strict-mechanism.v2",
            "strict_route_mode": "nonworsening_target_progress_v2",
            "spec_id": V2_SPEC_ID,
            "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
            "risk_bank_canonical_sha256": config["dataset_ranking"][
                "risk_bank_canonical_sha256"
            ],
            "risk_bank_file_sha256": config["dataset_ranking"][
                "risk_bank_file_sha256"
            ],
            "replay_manifest_file_sha256": config["dataset_ranking"][
                "replay_manifest_file_sha256"
            ],
            "protection_snapshot_sha256": {
                item["id"]: item["sha256"]
                for item in config["model"]["protection_surrogate_snapshots"]
            },
            "elapsed_seconds": 3.0,
            "decision": {
                "checks": {"finite": True, "final_target_progress": True},
                "pass": True,
            },
        },
    )
    result = _verify_result(config_path)
    assert result["gate_pass"] is True
    assert result["checks"]["final_target_progress"] is True


def test_g1_result_verifies_v3_component_schema(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V3_SPEC_ID
    config["strict_route"] = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))[
        "strict_route"
    ]
    output_root = tmp_path / "outputs-v3"
    config["runtime"]["artifact_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    strict_root = output_root / RUN_MODE
    strict_root.mkdir(parents=True)
    _write_json(strict_root / "backtracking_trace.json", {"trace": []})
    (strict_root / "p5_dataset_strict_state.pt").write_bytes(b"candidate-v3")
    _write_json(
        strict_root / "mechanism_metrics.json",
        {
            "schema": "tausb.dgcaip-dataset-strict-mechanism.v3",
            "strict_route_mode": "component_aligned_target_progress_v3",
            "constraint_row_schema": "snapshot_class_family_v3",
            "spec_id": V3_SPEC_ID,
            "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
            "risk_bank_canonical_sha256": config["dataset_ranking"][
                "risk_bank_canonical_sha256"
            ],
            "risk_bank_file_sha256": config["dataset_ranking"][
                "risk_bank_file_sha256"
            ],
            "replay_manifest_file_sha256": config["dataset_ranking"][
                "replay_manifest_file_sha256"
            ],
            "protection_snapshot_sha256": {
                item["id"]: item["sha256"]
                for item in config["model"]["protection_surrogate_snapshots"]
            },
            "elapsed_seconds": 3.0,
            "decision": {
                "checks": {"finite": True, "final_target_progress": True},
                "pass": True,
            },
        },
    )
    result = _verify_result(config_path)
    assert result["gate_pass"] is True


def test_g1_result_verifies_v4_layered_decision(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    config["spec"]["spec_id"] = V4_SPEC_ID
    config["strict_route"] = yaml.safe_load(
        V4_CONFIG.read_text(encoding="utf-8")
    )["strict_route"]
    output_root = tmp_path / "outputs-v4"
    config["runtime"]["artifact_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    strict_root = output_root / RUN_MODE
    strict_root.mkdir(parents=True)
    _write_json(strict_root / "backtracking_trace.json", {"trace": []})
    (strict_root / "p5_dataset_strict_state.pt").write_bytes(b"candidate-v4")
    _write_json(
        strict_root / "mechanism_metrics.json",
        {
            "schema": "tausb.dgcaip-dataset-strict-mechanism.v4",
            "strict_route_mode": "component_aligned_target_progress_v3",
            "constraint_row_schema": "snapshot_class_family_v3",
            "spec_id": V4_SPEC_ID,
            "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
            "risk_bank_canonical_sha256": config["dataset_ranking"][
                "risk_bank_canonical_sha256"
            ],
            "risk_bank_file_sha256": config["dataset_ranking"][
                "risk_bank_file_sha256"
            ],
            "replay_manifest_file_sha256": config["dataset_ranking"][
                "replay_manifest_file_sha256"
            ],
            "protection_snapshot_sha256": {
                item["id"]: item["sha256"]
                for item in config["model"]["protection_surrogate_snapshots"]
            },
            "elapsed_seconds": 3.0,
            "decision": {
                "checks": {
                    "finite": True,
                    "accepted_update_ratio": True,
                    "final_target_progress": True,
                },
                "runtime_pass": True,
                "mechanism_valid": True,
                "promotion_pass": True,
                "pass": True,
            },
        },
    )
    result = _verify_result(config_path)
    assert result["gate_pass"] is True
    assert result["checks"]["accepted_update_ratio"] is True

    metrics_path = strict_root / "mechanism_metrics.json"
    inconsistent = json.loads(metrics_path.read_text(encoding="utf-8"))
    inconsistent["decision"]["promotion_pass"] = False
    _write_json(metrics_path, inconsistent)
    with pytest.raises(ValueError, match="decision layers are inconsistent"):
        _verify_result(config_path)


def test_g1_result_rejects_output_snapshot_drift(tmp_path: Path) -> None:
    config, config_path = _bound_fixture(tmp_path)
    output_root = tmp_path / "drift"
    config["runtime"]["artifact_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    strict_root = output_root / RUN_MODE
    strict_root.mkdir(parents=True)
    _write_json(strict_root / "backtracking_trace.json", {})
    (strict_root / "p5_dataset_strict_state.pt").write_bytes(b"candidate")
    _write_json(
        strict_root / "mechanism_metrics.json",
        {
            "schema": "tausb.dgcaip-dataset-strict-mechanism.v1",
            "spec_id": SPEC_ID,
            "source_p1_state_sha256": config["dgcaip"]["source_p1_state_sha256"],
            "risk_bank_canonical_sha256": config["dataset_ranking"][
                "risk_bank_canonical_sha256"
            ],
            "risk_bank_file_sha256": config["dataset_ranking"][
                "risk_bank_file_sha256"
            ],
            "replay_manifest_file_sha256": config["dataset_ranking"][
                "replay_manifest_file_sha256"
            ],
            "protection_snapshot_sha256": {"e1": "0" * 64},
            "elapsed_seconds": 1.0,
            "decision": {"checks": {"finite": True}, "pass": True},
        },
    )
    with pytest.raises(ValueError, match="snapshot binding differs"):
        _verify_result(config_path)
