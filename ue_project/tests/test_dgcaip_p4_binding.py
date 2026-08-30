from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from ue_framework.config import load_config
from ue_framework.methods.sdh_materializer import (
    build_dgcaip_p4_candidate_state_payload,
)
from ue_framework.methods.semantic_hiding_carrier import SemanticHidingCarrier
from ue_framework.tools.bind_tausb_sdh_dgcaip_p4_e20 import (
    build_bound_config,
    validate_dgcaip_p4_binding,
)
from ue_framework.tools.bind_tausb_sdh_e2e_v0 import _canonical_json_sha256


BASE = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding_fixture(tmp_path: Path):
    mechanism_root = tmp_path / "mechanism_root"
    output = mechanism_root / "production_e20"
    output.mkdir(parents=True)
    mechanism_config = {"spec": {"spec_id": "test"}, "runtime": {}}
    config_path = tmp_path / "mechanism.yaml"
    config_path.write_text(yaml.safe_dump(mechanism_config), encoding="utf-8")
    scientific = {"checks": {"signal": False}, "pass": False}
    integrity = {"checks": {"finite": True}, "pass": True}
    metrics = {
        "schema": "tausb.dgcaip-mechanism.v1",
        "decision": scientific,
        "state_integrity": integrity,
    }
    metrics_path = output / "mechanism_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    fixed = {
        "source_p1_state_sha256": "1" * 64,
        "source_p1_metrics_sha256": "2" * 64,
        "d0_report_sha256": "3" * 64,
    }
    raw = {
        "schema": "tausb.dgcaip-state.v1",
        "arm_id": "P4-DGCAIP",
        "carrier_state": {"adapter": torch.ones(1)},
        "state_integrity_gate_passed": True,
        "mechanism_scientific_gate_passed": False,
        "mechanism_metrics_sha256": _file_sha256(metrics_path),
        "mechanism_config_sha256": _canonical_json_sha256(mechanism_config),
        **fixed,
    }
    raw_path = output / "p4_dgcaip_state.pt"
    torch.save(raw, raw_path)
    torch.manual_seed(23)
    carrier = SemanticHidingCarrier(input_size=32, width=8, coupling_blocks=2)
    provenance = {
        "hiding_metrics_sha256": "4" * 64,
        "hiding_checkpoint_sha256": "5" * 64,
        "hiding_split_sha256": "6" * 64,
        "mechanism_metrics_sha256": _file_sha256(metrics_path),
        "mechanism_scientific_decision_sha256": _canonical_json_sha256(scientific),
        "state_integrity_decision_sha256": _canonical_json_sha256(integrity),
        "mechanism_config_sha256": _canonical_json_sha256(mechanism_config),
        "p4_state_sha256": _file_sha256(raw_path),
        "repair_report_sha256": "7" * 64,
        **fixed,
    }
    candidate = build_dgcaip_p4_candidate_state_payload(
        carrier=carrier,
        secret=torch.rand((1, 3, 32, 32)),
        target_class_id=14,
        secret_source_sha256="a" * 64,
        secret_tensor_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        train_split_sha256="d" * 64,
        mechanism_scientific_gate_passed=False,
        provenance_hashes=provenance,
    )
    torch.save(candidate, output / "p4_dgcaip_candidate_sdh_state.pt")
    return mechanism_root, config_path


def test_p4_binding_accepts_integrity_pass_even_when_scientific_gate_fails(
    tmp_path,
) -> None:
    mechanism_root, config_path = _binding_fixture(tmp_path)
    binding = validate_dgcaip_p4_binding(mechanism_root, config_path)
    assert binding["metrics"]["decision"]["pass"] is False
    assert binding["metrics"]["state_integrity"]["pass"] is True
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config = build_bound_config(
        base,
        arm_id="M1",
        dataset_root=tmp_path / "voc",
        state_path=binding["state_path"],
        hashes=binding["hashes"],
        run_root=str(tmp_path / "run-E20-M1"),
    )
    bound_path = tmp_path / "bound.yaml"
    bound_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    load_config(str(bound_path))
    method = config["methods"]["tausb_sdh"]
    assert method["protocol_id"] == "TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1"
    assert method["p4_state_sha256"] == binding["hashes"]["p4_state_sha256"]


def test_p4_binding_rejects_integrity_failure(tmp_path) -> None:
    mechanism_root, config_path = _binding_fixture(tmp_path)
    metrics_path = mechanism_root / "production_e20" / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["state_integrity"]["pass"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity gate"):
        validate_dgcaip_p4_binding(mechanism_root, config_path)
