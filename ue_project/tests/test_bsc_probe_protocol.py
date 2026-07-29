from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pytest
import torch
import yaml

from ue_framework.methods.bsc_rc_gr_probe import (
    canonical_hash,
    evaluate_phase_a,
    evaluate_phase_b,
    evaluate_phase_c,
    load_background_sources,
    load_required_shared_split,
    validate_probe_config,
)


def _config() -> dict:
    path = (
        Path(__file__).parents[1]
        / "ue_framework"
        / "configs"
        / "exp_voc_person_tausb_bsc_rc_gr_probe.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_probe_config_and_shared_split_are_hash_bound(tmp_path: Path) -> None:
    config = _config()
    validate_probe_config(config)
    paths = [tmp_path / f"image_{index:03d}.jpg" for index in range(4)]
    payload = {
        "protocol_id": "TAUSB-ALCE-CTX-AUDIT-v1-shared",
        "seed": 2028,
        "calibration": ["image_000", "image_001"],
        "heldout": ["image_002", "image_003"],
        "label_hash": "a" * 64,
    }
    payload["split_hash"] = canonical_hash(payload)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    first = load_required_shared_split(
        split_path,
        target_images=paths,
        required_protocol_prefix="TAUSB-ALCE-CTX-AUDIT-v1",
    )
    second = load_required_shared_split(
        split_path,
        target_images=list(reversed(paths)),
        required_protocol_prefix="TAUSB-ALCE-CTX-AUDIT-v1",
    )
    assert first == second
    assert not set(first["calibration"]) & set(first["heldout"])

    payload["heldout"] = ["image_001", "image_003"]
    payload["split_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "split_hash"}
    )
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_required_shared_split(
            split_path,
            target_images=paths,
            required_protocol_prefix="TAUSB-ALCE-CTX-AUDIT-v1",
        )


def test_background_source_map_is_local_only_and_hash_checked(
    tmp_path: Path,
) -> None:
    manifest = []
    local_map = {}
    for index in range(8):
        path = tmp_path / f"source_{index}.png"
        image = torch.full((8, 8, 3), index * 10, dtype=torch.uint8).numpy()
        assert cv2.imwrite(str(path), image)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_id = f"background-{index}"
        manifest.append(
            {
                "source_id": source_id,
                "sha256": digest,
                "width": 8,
                "height": 8,
                "license_note": "owned",
                "person_free": True,
            }
        )
        local_map[source_id] = str(path.resolve())
    manifest_path = tmp_path / "manifest.json"
    map_path = tmp_path / "local.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    map_path.write_text(json.dumps(local_map), encoding="utf-8")

    images, loaded_manifest, manifest_hash = load_background_sources(
        manifest_path,
        map_path,
    )
    assert len(images) == 8
    assert loaded_manifest == manifest
    assert len(manifest_hash) == 64

    manifest[0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_background_sources(manifest_path, map_path)

    manifest[0]["sha256"] = hashlib.sha256(
        Path(local_map[manifest[0]["source_id"]]).read_bytes()
    ).hexdigest()
    manifest[0]["width"] = 9
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions mismatch"):
        load_background_sources(manifest_path, map_path)


def _phase_a_metric(
    *,
    cicr: float,
    q25: float,
    nt_ratio: float,
    box: float,
    low: float,
    mid: float,
    high: float,
    correlation: float,
) -> dict:
    return {
        "heldout_cicr_median": cicr,
        "heldout_cicr_q25": q25,
        "non_target_target_energy_ratio": nt_ratio,
        "box_residual_energy": box,
        "source_max_abs_correlation": correlation,
        "basis_hash": "a" * 64,
        "coefficient_hash": "b" * 64,
        "finite": True,
        "target_residual_zero_norm_ratio": 0.0,
        "materialized_spectrum_energy": {
            "low": low,
            "mid": mid,
            "high": high,
            "dc": 0.0,
        },
    }


def test_phase_gates_enforce_success_signals() -> None:
    phase_a = evaluate_phase_a(
        {
            "C0": _phase_a_metric(
                cicr=0.30,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.20,
                mid=0.60,
                high=0.20,
                correlation=0.10,
            ),
            "C1-L": _phase_a_metric(
                cicr=0.35,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.90,
                mid=0.05,
                high=0.05,
                correlation=0.80,
            ),
            "C2-L": _phase_a_metric(
                cicr=0.42,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.75,
                mid=0.10,
                high=0.15,
                correlation=0.20,
            ),
            "C2-LM": _phase_a_metric(
                cicr=0.45,
                q25=0.12,
                nt_ratio=0.50,
                box=0.20,
                low=0.30,
                mid=0.50,
                high=0.20,
                correlation=0.20,
            ),
        },
        split_hash="c" * 64,
        source_manifest_hash="d" * 64,
    )
    assert phase_a["pass"]
    assert phase_a["best_background_carrier"] == "C2-LM"

    semantic_failure_metrics = {
        name: dict(value)
        for name, value in {
            "C0": _phase_a_metric(
                cicr=0.30,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.20,
                mid=0.60,
                high=0.20,
                correlation=0.10,
            ),
            "C1-L": _phase_a_metric(
                cicr=0.60,
                q25=0.20,
                nt_ratio=0.50,
                box=0.20,
                low=0.90,
                mid=0.05,
                high=0.05,
                correlation=0.90,
            ),
            "C2-L": _phase_a_metric(
                cicr=0.45,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.80,
                mid=0.10,
                high=0.10,
                correlation=0.10,
            ),
            "C2-LM": _phase_a_metric(
                cicr=0.45,
                q25=0.10,
                nt_ratio=0.50,
                box=0.20,
                low=0.40,
                mid=0.40,
                high=0.20,
                correlation=0.10,
            ),
        }.items()
    }
    semantic_failure = evaluate_phase_a(
        semantic_failure_metrics,
        split_hash="c" * 64,
        source_manifest_hash="d" * 64,
    )
    assert not semantic_failure["pass"]
    assert semantic_failure["failure_signals"]["semantic_dependence"]

    common = {
        "group_cicr_median": {
            "person_only": 0.50,
            "person_cooccur": 0.45,
            "small": 0.42,
            "medium": 0.48,
            "large": 0.50,
        },
        "route_effect": 0.20,
        "attack_retention_median": 0.50,
        "attack_retention_q25": 0.20,
        "projected_violation_ratio": 0.01,
        "gradient_alignment_median": 0.20,
        "low_retention_ratio": 0.10,
        "target_residual_zero_norm_ratio": 0.0,
        "group_non_target_target_energy_ratio": {
            "person_only": 0.20,
            "person_cooccur": 0.25,
        },
    }
    phase_b = evaluate_phase_b(
        {
            "A1": {**common, "heldout_cicr_median": 0.30},
            "A3": {**common, "heldout_cicr_median": 0.40},
        },
        best_background_arm="A3",
    )
    assert phase_b["pass"]

    phase_c = evaluate_phase_c(
        {
            "actual_violation_rate": 0.40,
            "heldout_cicr_median": 0.40,
        },
        {
            "actual_violation_rate": 0.15,
            "heldout_cicr_median": 0.39,
            "repair_skip_ratio": 0.20,
            "null_dimension_median": 12,
            "finite": True,
        },
    )
    assert phase_c["pass"]
