from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

import torch
import yaml

from ue_framework.config import load_config
from ue_framework.methods.sdh_materializer import (
    E2E_V0_EVIDENCE_SCOPE,
    E2E_V0_MATERIALIZATION_MODE,
    E2E_V0_PROTOCOL_ID,
    FROZEN_SDH_STATE_SCHEMA,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind one completed E2E V0 mechanism state to paired smoke/E20 configs."
    )
    parser.add_argument("--mechanism-root", required=True)
    parser.add_argument("--mechanism-config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-root-prefix",
        default="/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping in %s." % path)
    return payload


def _has_person(label_path: Path) -> bool:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(float(parts[0])) == 14:
            return True
    return False


def build_smoke_selection_manifest(dataset_root: Path) -> Dict[str, Any]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("VOC train image/label directories are missing.")
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != 16551:
        raise ValueError("E2E V0 requires exactly 16551 VOC train images.")
    target = []
    person_free = []
    rows = {}
    for image_path in images:
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            raise FileNotFoundError("VOC label is missing for %s." % image_path.stem)
        has_target = _has_person(label_path)
        key = hashlib.sha256(("0:%s" % image_path.stem).encode("ascii")).hexdigest()
        record = {
            "stem": image_path.stem,
            "has_target": has_target,
            "label_sha256": _file_sha256(label_path),
        }
        rows[image_path.stem] = record
        (target if has_target else person_free).append((key, image_path.stem))
    if len(target) != 6095:
        raise ValueError("E2E V0 requires exactly 6095 person train images.")
    target.sort()
    person_free.sort()
    selected_stems = [stem for _, stem in target[:40]] + [
        stem for _, stem in person_free[:160]
    ]
    selected_stems.sort()
    records = [rows[stem] for stem in selected_stems]
    return {
        "schema": "tausb.sdh-e2e-v0-train-selection.v1",
        "protocol_id": E2E_V0_PROTOCOL_ID,
        "seed": 0,
        "target_class_id": 14,
        "selection_rule": "sha256(0:stem);first40_person+first160_person_free",
        "selected_count": len(records),
        "target_count": sum(int(item["has_target"]) for item in records),
        "person_free_count": sum(int(not item["has_target"]) for item in records),
        "records": records,
    }


def _validate_mechanism_binding(
    mechanism_root: Path,
    mechanism_config_path: Path,
) -> Dict[str, Any]:
    mechanism_dir = mechanism_root / "mechanism"
    state_path = mechanism_dir / "p1_feasibility_sdh_state.pt"
    metrics_path = mechanism_dir / "mechanism_metrics.json"
    p1_path = mechanism_dir / "p1_state.pt"
    for path in (state_path, metrics_path, p1_path, mechanism_config_path):
        if not path.is_file():
            raise FileNotFoundError("Binding input is missing: %s" % path)
    state = _torch_load(state_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mechanism_config = yaml.safe_load(
        mechanism_config_path.read_text(encoding="utf-8")
    )
    required = {
        "schema": FROZEN_SDH_STATE_SCHEMA,
        "arm_id": "P1",
        "protocol_id": E2E_V0_PROTOCOL_ID,
        "materialization_mode": E2E_V0_MATERIALIZATION_MODE,
        "evidence_scope": E2E_V0_EVIDENCE_SCOPE,
        "hiding_gate_passed": False,
        "allow_failed_scientific_gates": True,
    }
    for key, expected in required.items():
        if state.get(key) != expected:
            raise ValueError("Feasibility state %s does not match the V0 contract." % key)
    actual_hashes = {
        "frozen_sdh_state_sha256": _file_sha256(state_path),
        "mechanism_metrics_sha256": _file_sha256(metrics_path),
        "p1_state_sha256": _file_sha256(p1_path),
        "mechanism_decision_sha256": _canonical_json_sha256(metrics["decision"]),
        "mechanism_config_sha256": _canonical_json_sha256(mechanism_config),
    }
    for key in (
        "mechanism_metrics_sha256",
        "p1_state_sha256",
        "mechanism_decision_sha256",
        "mechanism_config_sha256",
    ):
        if state.get(key) != actual_hashes[key]:
            raise ValueError("Feasibility state %s does not match its source artifact." % key)
    actual_hashes.update(
        {
            key: str(state[key])
            for key in (
                "hiding_metrics_sha256",
                "hiding_checkpoint_sha256",
                "hiding_split_sha256",
            )
        }
    )
    return {
        "state": state,
        "state_path": state_path,
        "metrics": metrics,
        "hashes": actual_hashes,
    }


def _bound_config(
    base: Mapping[str, Any],
    *,
    arm_id: str,
    pilot_kind: str,
    dataset_root: Path,
    state_path: Path,
    hashes: Mapping[str, str],
    selection_path: str,
    selection_hash: str,
    run_root: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(base)
    smoke = pilot_kind == "smoke"
    m1 = arm_id == "M1"
    config["experiment"].update(
        {
            "pilot_kind": pilot_kind,
            "arm_id": arm_id,
            "poisoning_ratio": 1.0 if m1 else 0.0,
            "expected_poisoned_count": 40 if smoke and m1 else (6095 if m1 else 0),
            "expected_train_images": 200 if smoke else 16551,
            "expected_target_images": 40 if smoke else 6095,
            "steps": [40],
            "seeds": [0],
        }
    )
    config["data"].update(
        {
            "dataset_root": str(dataset_root),
            "allow_pseudo_mask_fallback": False,
            "train_selection_manifest": selection_path if smoke else "",
            "train_selection_manifest_sha256": selection_hash if smoke else "",
        }
    )
    config["platform"].update(
        {
            "mode": "cloud",
            "run_root": run_root,
            "resume": False,
            "zip_after_stage": False,
            "save_every_n_epochs": 1 if smoke else 5,
            "pack_every_n_epochs": 1 if smoke else 5,
        }
    )
    config["victim"].update(
        {
            "epochs": 1 if smoke else 20,
            "imgsz": 640,
            "batch": 36,
            "optimizer": "SGD",
        }
    )
    method = config["methods"]["tausb_sdh"]
    method.update(
        {
            "protocol_id": E2E_V0_PROTOCOL_ID,
            "materialization_mode": E2E_V0_MATERIALIZATION_MODE,
            "allow_failed_scientific_gates": True,
            "binding_status": "bound",
            "evidence_scope": E2E_V0_EVIDENCE_SCOPE,
            "support_type": "bbox",
            "require_hiding_gate_pass": False,
            "require_mechanism_gate_pass": False,
            "frozen_sdh_state": str(state_path),
            **hashes,
        }
    )
    return config


def main() -> int:
    args = _arguments()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Binding output directory already exists: %s" % output_dir)
    mechanism_root = Path(args.mechanism_root).resolve()
    mechanism_config_path = Path(args.mechanism_config).resolve()
    base_path = Path(args.base_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    binding = _validate_mechanism_binding(mechanism_root, mechanism_config_path)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    selection = build_smoke_selection_manifest(dataset_root)
    selection_hash = _canonical_json_sha256(selection)
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "smoke_train_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    config_records: List[Dict[str, str]] = []
    for pilot_kind, arm_id, suffix in (
        ("smoke", "C0", "SMOKE-C0"),
        ("smoke", "M1", "SMOKE-M1"),
        ("e20", "C0", "E20-C0"),
        ("e20", "M1", "E20-M1"),
    ):
        config = _bound_config(
            base,
            arm_id=arm_id,
            pilot_kind=pilot_kind,
            dataset_root=dataset_root,
            state_path=binding["state_path"],
            hashes=binding["hashes"],
            selection_path=str(selection_path),
            selection_hash=selection_hash,
            run_root="%s-%s" % (args.run_root_prefix.rstrip("-"), suffix),
        )
        path = output_dir / ("%s-%s.yaml" % (pilot_kind, arm_id.lower()))
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(str(path))
        config_records.append(
            {
                "pilot_kind": pilot_kind,
                "arm_id": arm_id,
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "canonical_sha256": _canonical_json_sha256(config),
                "run_root": config["platform"]["run_root"],
            }
        )
    report = {
        "schema": "tausb.sdh-e2e-v0-binding-report.v1",
        "protocol_id": E2E_V0_PROTOCOL_ID,
        "mechanism_root": str(mechanism_root),
        "frozen_sdh_state": str(binding["state_path"]),
        "hashes": binding["hashes"],
        "state_content_hash": binding["state"]["state_content_hash"],
        "mechanism_gate_passed": bool(binding["state"]["mechanism_gate_passed"]),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_hash,
        "configs": config_records,
    }
    (output_dir / "binding_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
