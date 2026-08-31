from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from ..data_utils import label_path_for_image, read_yolo_annotations
from .dgcaip_dataset_risk import (
    DGCAIPInstanceKey,
    DGCAIPRiskRecord,
    build_balanced_replay_image_ids,
    build_dataset_risk_bank,
    load_risk_bank,
    risk_bank_payload,
    write_risk_bank,
)
from .dgcaip_proxy_agreement import (
    evaluate_proxy_victim_agreement,
    worst_case_calibrated_risks,
)
from .dgcaip_experiment import (
    _is_cooccurring,
    _load_engine,
    _prepare_experiment,
    run_dgcaip_pilot,
)
from .sdh_experiment import (
    _batches,
    _clone_detector_carrier,
    _file_sha256,
    _load_saved_p1_carrier,
    _person_paths,
    _resolve,
    _time_guard,
    _write_json,
)
from .sdh_mechanism import load_sdh_batch


DATASET_CGR_PROXY_SPEC_ID = "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"


def _load_strict_candidate_carrier(
    path: Path,
    *,
    base_carrier,
    device: torch.device,
):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if (
        not isinstance(saved, Mapping)
        or saved.get("schema") != "tausb.dgcaip-dataset-strict-state.v1"
        or saved.get("arm_id") != "P5-DATASET-STRICT"
        or not bool(dict(saved.get("decision", {})).get("pass", False))
    ):
        raise ValueError("Short-victim scan requires a passed strict P5 state.")
    carrier_state = saved.get("carrier_state")
    if not isinstance(carrier_state, Mapping) or any(
        not torch.is_tensor(value) or not torch.isfinite(value).all()
        for value in carrier_state.values()
    ):
        raise ValueError("Strict P5 carrier state contains invalid tensors.")
    carrier = _clone_detector_carrier(base_carrier, device)
    carrier.load_state_dict(carrier_state, strict=True)
    carrier.freeze_for_detector_optimization()
    return carrier


def _snapshot_configs(config: Mapping[str, Any]) -> Sequence[Mapping[str, str]]:
    snapshots = config["model"].get("protection_surrogate_snapshots")
    run_mode = str(config["dgcaip"].get("run_mode", ""))
    expected_ids = ["v3"] if run_mode == "short_victim_risk_scan" else ["e1", "e5", "e20"]
    if (
        not isinstance(snapshots, list)
        or [str(item.get("id", "")) for item in snapshots] != expected_ids
    ):
        raise ValueError(
            "Dataset-risk scan snapshot IDs must be %s."
            % "/".join(expected_ids)
        )
    return snapshots


def _expected_non_target_keys(
    paths: Sequence[Path],
    *,
    label_dir: Path,
    target_class_id: int,
) -> tuple[DGCAIPInstanceKey, ...]:
    keys = []
    for path in paths:
        annotations = read_yolo_annotations(
            label_path_for_image(str(path), str(label_dir))
        )
        for gt_index, annotation in enumerate(annotations):
            class_id = int(annotation["cls"])
            if class_id != int(target_class_id):
                keys.append(DGCAIPInstanceKey(path.stem, gt_index, class_id))
    if len(keys) != len(set(keys)):
        raise RuntimeError("Dataset-risk expected instance keys are not unique.")
    return tuple(keys)


def run_dataset_risk_scan(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Dict[str, Any]:
    start = time.monotonic()
    max_seconds = float(config["mechanism"]["max_seconds"])
    device = torch.device(str(config["runtime"]["device"]))
    base_carrier, primary_secret, _, image_dir, label_dir, _ = _prepare_experiment(
        config, config_base=config_base
    )
    dg_config = config["dgcaip"]
    short_victim_scan = str(dg_config["run_mode"]) == "short_victim_risk_scan"
    if short_victim_scan:
        source_state_path = _resolve(
            config_base, str(dg_config["source_carrier_state"])
        )
        expected_source_hash = str(dg_config["source_carrier_state_sha256"])
        if _file_sha256(source_state_path) != expected_source_hash.lower():
            raise ValueError("Short-victim source carrier-state hash mismatch.")
        carrier = _load_strict_candidate_carrier(
            source_state_path,
            base_carrier=base_carrier,
            device=device,
        )
    else:
        source_state_path = _resolve(
            config_base, str(dg_config["source_p1_state"])
        )
        expected_source_hash = str(dg_config["source_p1_state_sha256"])
        if _file_sha256(source_state_path) != expected_source_hash.lower():
            raise ValueError("Dataset-risk source P1 state hash mismatch.")
        carrier = _load_saved_p1_carrier(
            source_state_path,
            base_carrier=base_carrier,
            device=device,
        )
    all_person = _person_paths(image_dir, label_dir, 14)
    cooccurring = tuple(
        path for path in all_person if _is_cooccurring(path, label_dir, 14)
    )
    if not cooccurring:
        raise RuntimeError("Dataset-risk scan found no person-cooccurrence images.")
    expected_keys = _expected_non_target_keys(
        cooccurring, label_dir=label_dir, target_class_id=14
    )
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    output_root = artifact_root / str(dg_config["run_mode"])
    output_root.mkdir(parents=True, exist_ok=False)
    batch_size = int(config["mechanism"]["batch_size"])
    records = []
    raw_rows = []
    snapshot_hashes = {}
    for snapshot in _snapshot_configs(config):
        snapshot_id = str(snapshot["id"])
        checkpoint = _resolve(config_base, str(snapshot["checkpoint"]))
        actual_hash = _file_sha256(checkpoint)
        if actual_hash != str(snapshot["sha256"]).lower():
            raise ValueError("Protection snapshot hash mismatch: %s" % snapshot_id)
        snapshot_hashes[snapshot_id] = actual_hash
        engine = _load_engine(
            config, config_base=config_base, checkpoint_path=checkpoint
        )
        with torch.no_grad():
            for paths_batch in _batches(cooccurring, batch_size):
                _time_guard(start, max_seconds, "DG-CAIP dataset risk scan")
                batch = load_sdh_batch(
                    paths_batch,
                    label_dir=label_dir,
                    image_size=640,
                    target_class_id=14,
                    device=device,
                )
                observation = engine.observe(
                    batch,
                    carrier,
                    primary_secret,
                    dgcaip_mode="dist",
                )
                if observation.dgcaip is None:
                    raise RuntimeError("Dataset-risk observation is missing DG-CAIP.")
                for term in observation.dgcaip.instances:
                    key = DGCAIPInstanceKey(
                        observation.image_ids[term.batch_index],
                        term.gt_index,
                        term.class_id,
                    )
                    js = float(term.distribution_loss.detach().cpu())
                    kl = float(term.clean_to_poison_kl.detach().cpu())
                    records.append(
                        DGCAIPRiskRecord(
                            key=key,
                            snapshot_id=snapshot_id,
                            js_divergence=js,
                            clean_to_poison_kl=kl,
                        )
                    )
                    raw_rows.append(
                        {
                            "image_id": key.image_id,
                            "gt_index": key.gt_index,
                            "class_id": key.class_id,
                            "snapshot_id": snapshot_id,
                            "js_divergence": js,
                            "clean_to_poison_kl": kl,
                            "assigned_probability_drop": float(
                                term.classification_damage.detach().cpu()
                            ),
                            "iou_drop": float(term.box_damage.detach().cpu()),
                            "tal_alignment_drop": float(
                                term.alignment_damage.detach().cpu()
                            ),
                            "positive_count": term.positive_count,
                            "geometry_risk": term.geometry_risk,
                        }
                    )
        del engine
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranking = config["dataset_ranking"]
    bank = build_dataset_risk_bank(
        records,
        spec_id=DATASET_CGR_PROXY_SPEC_ID,
        expected_snapshot_ids=tuple(snapshot_hashes),
        expected_instance_keys=expected_keys,
        js_weight=float(ranking["js_weight"]),
        kl_weight=float(ranking["kl_weight"]),
        top_fraction=float(ranking["top_fraction"]),
        minimum_coverage=float(ranking["minimum_coverage"]),
    )
    raw_path = output_root / "dgcaip_risk_records.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in sorted(
                raw_rows,
                key=lambda row: (
                    row["snapshot_id"],
                    row["class_id"],
                    row["image_id"],
                    row["gt_index"],
                ),
            )
        ),
        encoding="utf-8",
    )
    bank_path = output_root / "dgcaip_risk_bank.json"
    write_risk_bank(bank_path, bank)
    stable_covered_keys = {
        entry.key.as_tuple()
        for entry in bank.entries
        if entry.snapshot_coverage >= 2.0 / 3.0
    }
    expected_by_image = {}
    for key in expected_keys:
        expected_by_image.setdefault(key.image_id, set()).add(key.as_tuple())
    replay_population = tuple(
        path.stem
        for path in cooccurring
        if expected_by_image[path.stem].issubset(stable_covered_keys)
    )
    if not replay_population:
        raise RuntimeError("No fully covered cooccurrence image is replay-eligible.")
    replay_ids = build_balanced_replay_image_ids(
        bank,
        replay_population,
        total_slots=int(config["mechanism"]["optimization_steps"]) * batch_size,
        high_risk_fraction=float(ranking["high_risk_replay_fraction"]),
        seed=int(config["spec"]["seed"]),
    )
    replay_payload = {
        "schema": "tausb.dgcaip-dataset-replay.v1",
        "spec_id": DATASET_CGR_PROXY_SPEC_ID,
        "risk_bank_canonical_sha256": bank.canonical_sha256,
        "image_ids": list(replay_ids),
    }
    replay_path = output_root / "dgcaip_replay_manifest.json"
    _write_json(replay_path, replay_payload)
    manifest = {
        "schema": "tausb.dgcaip-dataset-risk-manifest.v1",
        "spec_id": DATASET_CGR_PROXY_SPEC_ID,
        "source_carrier_state_sha256": _file_sha256(source_state_path),
        "snapshot_sha256": snapshot_hashes,
        "person_cooccurrence_image_count": len(cooccurring),
        "replay_eligible_image_count": len(replay_population),
        "expected_instance_count": len(expected_keys),
        "covered_instance_count": bank.covered_instance_count,
        "stable_snapshot_instance_count": len(stable_covered_keys),
        "coverage": bank.coverage,
        "risk_bank_canonical_sha256": bank.canonical_sha256,
        "risk_bank_file_sha256": _file_sha256(bank_path),
        "raw_records_sha256": _file_sha256(raw_path),
        "replay_manifest_sha256": _file_sha256(replay_path),
        "elapsed_seconds": time.monotonic() - start,
        "decision": {
            "pass": bank.coverage
            >= float(ranking["minimum_coverage"]),
        },
    }
    _write_json(output_root / "dgcaip_risk_manifest.json", manifest)
    return {
        **manifest,
        "risk_bank": risk_bank_payload(bank),
        "replay": replay_payload,
    }


def run_proxy_victim_agreement_audit(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Dict[str, Any]:
    agreement_config = config["proxy_agreement"]
    proxy_path = _resolve(
        config_base, str(agreement_config["proxy_risk_bank"])
    )
    victim_path = _resolve(
        config_base, str(agreement_config["victim_risk_bank"])
    )
    for name, path in (("proxy", proxy_path), ("victim", victim_path)):
        expected_hash = str(agreement_config[f"{name}_risk_bank_file_sha256"])
        if _file_sha256(path) != expected_hash.lower():
            raise ValueError("%s risk-bank file hash mismatch." % name)
    proxy_bank = load_risk_bank(
        proxy_path, expected_spec_id=DATASET_CGR_PROXY_SPEC_ID
    )
    victim_bank = load_risk_bank(
        victim_path, expected_spec_id=DATASET_CGR_PROXY_SPEC_ID
    )
    proxy_risks = proxy_bank.rank_mapping()
    victim_risks = victim_bank.rank_mapping()
    agreement = evaluate_proxy_victim_agreement(
        proxy_risks,
        victim_risks,
        top_fraction=float(config["dataset_ranking"]["top_fraction"]),
        minimum_spearman=float(agreement_config["minimum_spearman"]),
        minimum_top_overlap=float(agreement_config["minimum_top_overlap"]),
        minimum_coverage=float(agreement_config["minimum_coverage"]),
    )
    calibrated = worst_case_calibrated_risks(proxy_risks, victim_risks)
    artifact_root = _resolve(config_base, str(config["runtime"]["artifact_root"]))
    output_root = artifact_root / "proxy_victim_audit"
    output_root.mkdir(parents=True, exist_ok=False)
    calibrated_payload = {
        "schema": "tausb.dgcaip-calibrated-risk-mapping.v1",
        "spec_id": DATASET_CGR_PROXY_SPEC_ID,
        "proxy_risk_bank_canonical_sha256": proxy_bank.canonical_sha256,
        "victim_risk_bank_canonical_sha256": victim_bank.canonical_sha256,
        "entries": [
            {
                "image_id": key[0],
                "gt_index": key[1],
                "class_id": key[2],
                "risk": calibrated[key],
            }
            for key in sorted(calibrated)
        ],
    }
    encoded = json.dumps(
        calibrated_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    calibrated_payload["canonical_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json(output_root / "calibrated_risk_mapping.json", calibrated_payload)
    report = {
        "schema": "tausb.dgcaip-proxy-victim-agreement.v1",
        "spec_id": DATASET_CGR_PROXY_SPEC_ID,
        "matched_count": agreement.matched_count,
        "union_count": agreement.union_count,
        "matched_coverage": agreement.matched_coverage,
        "macro_spearman": agreement.macro_spearman,
        "macro_top_fraction_overlap": agreement.macro_top_fraction_overlap,
        "per_class": {
            str(class_id): {
                "matched_count": item.matched_count,
                "union_count": item.union_count,
                "coverage": item.coverage,
                "spearman": item.spearman,
                "top_fraction_overlap": item.top_fraction_overlap,
            }
            for class_id, item in sorted(agreement.per_class.items())
        },
        "decision": {
            "pass": agreement.passed,
            "failure_reasons": list(agreement.failure_reasons),
        },
        "calibrated_risk_mapping_sha256": calibrated_payload[
            "canonical_sha256"
        ],
    }
    _write_json(output_root / "proxy_victim_agreement.json", report)
    return report


def run_dataset_cgr_proxy_stage(
    config: Mapping[str, Any],
    *,
    config_base: Path,
) -> Dict[str, Any]:
    run_mode = str(config["dgcaip"].get("run_mode", ""))
    if run_mode in {"dataset_risk_scan", "short_victim_risk_scan"}:
        return run_dataset_risk_scan(config, config_base=config_base)
    if run_mode == "strict_mechanism":
        return run_dgcaip_pilot(config, config_base=config_base)
    if run_mode == "proxy_victim_audit":
        return run_proxy_victim_agreement_audit(
            config, config_base=config_base
        )
    raise ValueError("Dataset-CGR-Proxy run mode is not implemented yet: %s" % run_mode)
