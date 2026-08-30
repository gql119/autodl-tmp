from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml

from ue_framework.config import load_config
from ue_framework.methods.sdh_materializer import (
    DGCAIP_P4_ARM_ID,
    DGCAIP_P4_EVIDENCE_SCOPE,
    DGCAIP_P4_MATERIALIZATION_MODE,
    DGCAIP_P4_PROTOCOL_ID,
    DGCAIP_P4_PROVENANCE_HASH_KEYS,
    FROZEN_SDH_STATE_SCHEMA,
)
from ue_framework.tools.bind_tausb_sdh_e2e_v0 import (
    _bound_config,
    _canonical_json_sha256,
    _file_sha256,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind one integrity-passed DG-CAIP P4 state to paired sparse E20."
    )
    parser.add_argument("--mechanism-root", required=True)
    parser.add_argument("--mechanism-config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root-prefix", required=True)
    return parser.parse_args()


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping checkpoint: %s" % path)
    return payload


def validate_dgcaip_p4_binding(
    mechanism_root: Path,
    mechanism_config_path: Path,
) -> Dict[str, Any]:
    output_root = mechanism_root / "production_e20"
    state_path = output_root / "p4_dgcaip_candidate_sdh_state.pt"
    raw_path = output_root / "p4_dgcaip_state.pt"
    metrics_path = output_root / "mechanism_metrics.json"
    for path in (state_path, raw_path, metrics_path, mechanism_config_path):
        if not path.is_file():
            raise FileNotFoundError("DG-CAIP P4 binding input is missing: %s" % path)

    state = _torch_load(state_path)
    raw = _torch_load(raw_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mechanism_config = yaml.safe_load(
        mechanism_config_path.read_text(encoding="utf-8")
    )
    required = {
        "schema": FROZEN_SDH_STATE_SCHEMA,
        "arm_id": DGCAIP_P4_ARM_ID,
        "source_arm_id": DGCAIP_P4_ARM_ID,
        "protocol_id": DGCAIP_P4_PROTOCOL_ID,
        "materialization_mode": DGCAIP_P4_MATERIALIZATION_MODE,
        "evidence_scope": DGCAIP_P4_EVIDENCE_SCOPE,
        "allow_failed_scientific_gates": True,
        "state_integrity_gate_passed": True,
        "hiding_gate_passed": False,
    }
    for key, expected in required.items():
        if state.get(key) != expected:
            raise ValueError("DG-CAIP P4 candidate %s mismatch." % key)
    if raw.get("schema") != "tausb.dgcaip-state.v1":
        raise ValueError("DG-CAIP P4 raw state schema mismatch.")
    if raw.get("arm_id") != DGCAIP_P4_ARM_ID:
        raise ValueError("DG-CAIP P4 raw state arm mismatch.")
    if raw.get("state_integrity_gate_passed") is not True:
        raise ValueError("DG-CAIP P4 raw state did not pass integrity.")
    carrier_state = raw.get("carrier_state")
    if not isinstance(carrier_state, Mapping) or any(
        not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
        for value in carrier_state.values()
    ):
        raise ValueError("DG-CAIP P4 raw carrier state is invalid.")
    if metrics.get("schema") != "tausb.dgcaip-mechanism.v1":
        raise ValueError("DG-CAIP P4 mechanism metrics schema mismatch.")
    integrity = metrics.get("state_integrity")
    scientific = metrics.get("decision")
    if not isinstance(integrity, Mapping) or integrity.get("pass") is not True:
        raise ValueError("DG-CAIP P4 mechanism integrity gate did not pass.")
    if not isinstance(scientific, Mapping) or not isinstance(scientific.get("pass"), bool):
        raise ValueError("DG-CAIP P4 scientific decision is missing.")
    if state.get("mechanism_scientific_gate_passed") is not bool(scientific["pass"]):
        raise ValueError("DG-CAIP P4 scientific gate provenance mismatch.")

    actual_hashes = {
        "frozen_sdh_state_sha256": _file_sha256(state_path),
        "mechanism_metrics_sha256": _file_sha256(metrics_path),
        "mechanism_scientific_decision_sha256": _canonical_json_sha256(scientific),
        "state_integrity_decision_sha256": _canonical_json_sha256(integrity),
        "mechanism_config_sha256": _canonical_json_sha256(mechanism_config),
        "p4_state_sha256": _file_sha256(raw_path),
        **{
            name: str(state[name])
            for name in DGCAIP_P4_PROVENANCE_HASH_KEYS
            if name not in {
                "mechanism_metrics_sha256",
                "mechanism_scientific_decision_sha256",
                "state_integrity_decision_sha256",
                "mechanism_config_sha256",
                "p4_state_sha256",
            }
        },
    }
    for name in DGCAIP_P4_PROVENANCE_HASH_KEYS:
        if str(state.get(name, "")) != actual_hashes[name]:
            raise ValueError("DG-CAIP P4 candidate %s source mismatch." % name)
    for name in (
        "source_p1_state_sha256",
        "source_p1_metrics_sha256",
        "d0_report_sha256",
        "mechanism_metrics_sha256",
        "mechanism_config_sha256",
    ):
        if str(raw.get(name, "")) != actual_hashes[name]:
            raise ValueError("DG-CAIP P4 raw state %s mismatch." % name)
    if raw.get("mechanism_scientific_gate_passed") is not bool(scientific["pass"]):
        raise ValueError("DG-CAIP P4 raw scientific flag mismatch.")
    return {
        "state": state,
        "state_path": state_path,
        "raw_path": raw_path,
        "metrics": metrics,
        "hashes": actual_hashes,
    }


def build_bound_config(
    base: Mapping[str, Any],
    *,
    arm_id: str,
    dataset_root: Path,
    state_path: Path,
    hashes: Mapping[str, str],
    run_root: str,
) -> Dict[str, Any]:
    config = _bound_config(
        base,
        arm_id=arm_id,
        pilot_kind="e20",
        dataset_root=dataset_root,
        state_path=state_path,
        hashes=hashes,
        selection_path="",
        selection_hash="",
        run_root=run_root,
        victim_epochs=20,
    )
    method = config["methods"]["tausb_sdh"]
    method.update(
        {
            "protocol_id": DGCAIP_P4_PROTOCOL_ID,
            "materialization_mode": DGCAIP_P4_MATERIALIZATION_MODE,
            "evidence_scope": DGCAIP_P4_EVIDENCE_SCOPE,
            "allow_failed_scientific_gates": True,
            "binding_status": "bound",
            "require_hiding_gate_pass": False,
            "require_mechanism_gate_pass": False,
            **hashes,
        }
    )
    return config


def main() -> int:
    args = _arguments()
    mechanism_root = Path(args.mechanism_root).resolve()
    mechanism_config_path = Path(args.mechanism_config).resolve()
    base_path = Path(args.base_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("DG-CAIP P4 binding output exists: %s" % output_dir)
    binding = validate_dgcaip_p4_binding(mechanism_root, mechanism_config_path)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = []
    for arm_id in ("C0", "M1"):
        config = build_bound_config(
            base,
            arm_id=arm_id,
            dataset_root=dataset_root,
            state_path=binding["state_path"],
            hashes=binding["hashes"],
            run_root="%s-E20-%s" % (args.run_root_prefix.rstrip("-"), arm_id),
        )
        path = output_dir / ("e20-%s.yaml" % arm_id.lower())
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(str(path))
        configs.append(
            {
                "arm_id": arm_id,
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "canonical_sha256": _canonical_json_sha256(config),
                "run_root": config["platform"]["run_root"],
            }
        )
    report = {
        "schema": "tausb.dgcaip-p4-e20-binding.v1",
        "protocol_id": DGCAIP_P4_PROTOCOL_ID,
        "mechanism_root": str(mechanism_root),
        "frozen_sdh_state": str(binding["state_path"]),
        "state_content_hash": binding["state"]["state_content_hash"],
        "state_integrity_gate_passed": True,
        "mechanism_scientific_gate_passed": bool(
            binding["metrics"]["decision"]["pass"]
        ),
        "binding_scope": "e20_only",
        "hashes": binding["hashes"],
        "configs": configs,
    }
    (output_dir / "binding_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
