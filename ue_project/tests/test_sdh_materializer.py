from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from ue_framework.methods.sdh_materializer import (
    SDHMaterializer,
    DGCAIP_P4_PROVENANCE_HASH_KEYS,
    build_dgcaip_p4_candidate_state_payload,
    build_feasibility_sdh_state_payload,
    build_frozen_sdh_state_payload,
    load_frozen_sdh_state,
)
from ue_framework.methods.semantic_hiding_carrier import SemanticHidingCarrier


HASHES = {
    "secret_source_sha256": "a" * 64,
    "secret_tensor_sha256": "b" * 64,
    "source_manifest_sha256": "c" * 64,
    "train_split_sha256": "d" * 64,
}


def _payload(*, hiding=True, mechanism=True, hf_subband_scale=1.0):
    torch.manual_seed(7)
    carrier = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        hf_subband_scale=hf_subband_scale,
    )
    secret = torch.rand((1, 3, 32, 32))
    return build_frozen_sdh_state_payload(
        carrier=carrier,
        secret=secret,
        target_class_id=14,
        hiding_gate_passed=hiding,
        mechanism_gate_passed=mechanism,
        **HASHES,
    )


def _cfg(path):
    return {
        "experiment": {"target_class_id": 14, "eps": 16 / 255},
        "surrogate": {"num_classes": 20, "imgsz": 640, "eot_samples": 1},
    }, {"support_type": "bbox", "frozen_sdh_state": str(path), **HASHES}


def test_state_gates_and_content_hash_fail_closed(tmp_path) -> None:
    failed = tmp_path / "failed.pt"
    torch.save(_payload(hiding=False), failed)
    with pytest.raises(ValueError, match="hiding_gate_passed"):
        load_frozen_sdh_state(
            str(failed), device=torch.device("cpu"), expected_target_class_id=14,
            expected_epsilon=16 / 255, expected_hashes=HASHES,
        )
    payload = _payload()
    first_key = next(iter(payload["model_state"]))
    payload["model_state"][first_key] = payload["model_state"][first_key].clone()
    payload["model_state"][first_key].reshape(-1)[0] += 1.0
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_frozen_sdh_state(
            str(tampered), device=torch.device("cpu"), expected_target_class_id=14,
            expected_epsilon=16 / 255, expected_hashes=HASHES,
        )


def test_materializer_is_deterministic_bounded_and_gt_box_local(tmp_path) -> None:
    path = tmp_path / "p1.pt"
    torch.save(_payload(), path)
    cfg, method_cfg = _cfg(path)
    materializer = SDHMaterializer(cfg, method_cfg, torch.device("cpu"), torch.nn.Identity())
    image = np.full((40, 48, 3), 0.5, dtype=np.float32)
    annotations = [
        {"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.6]},
        {"cls": 7, "bbox": [0.2, 0.2, 0.1, 0.1]},
    ]
    kwargs = dict(
        image=image, annotations=annotations, seed=1, steps=40,
        eps=16 / 255, support_type="bbox", image_path="000001.jpg",
    )
    first = materializer.generate(**kwargs)
    second = materializer.generate(**kwargs)
    assert np.array_equal(first.poisoned_image, second.poisoned_image)
    assert np.array_equal(first.perturbation, second.perturbation)
    assert first.extras["support_source"] == "person_gt_bbox"
    assert np.max(np.abs(first.perturbation[first.support_mask == 0])) == 0
    assert np.max(np.abs(first.perturbation)) <= 16 / 255 + 1e-7


def test_materializer_rejects_non_bbox_support(tmp_path) -> None:
    path = tmp_path / "p1.pt"
    torch.save(_payload(), path)
    cfg, method_cfg = _cfg(path)
    materializer = SDHMaterializer(cfg, method_cfg, torch.device("cpu"), torch.nn.Identity())
    with pytest.raises(ValueError, match="person GT boxes"):
        materializer.generate(
            np.zeros((32, 32, 3), dtype=np.float32),
            [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}],
            0, 40, 16 / 255, "mask",
        )


def test_frozen_state_round_trips_non_default_subband_scale(tmp_path) -> None:
    path = tmp_path / "sb25.pt"
    torch.save(_payload(hf_subband_scale=0.25), path)
    loaded = load_frozen_sdh_state(
        str(path),
        device=torch.device("cpu"),
        expected_target_class_id=14,
        expected_epsilon=16 / 255,
        expected_hashes=HASHES,
    )
    assert loaded.carrier.hf_subband_scale == 0.25


def _v0_state(tmp_path):
    torch.manual_seed(13)
    carrier = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        epsilon=16 / 255,
    )
    provenance_hashes = {
        "hiding_metrics_sha256": "e" * 64,
        "hiding_checkpoint_sha256": "f" * 64,
        "hiding_split_sha256": "1" * 64,
        "mechanism_metrics_sha256": "2" * 64,
        "mechanism_decision_sha256": "3" * 64,
        "mechanism_config_sha256": "4" * 64,
        "p1_state_sha256": "5" * 64,
    }
    payload = build_feasibility_sdh_state_payload(
        carrier=carrier,
        secret=torch.rand((1, 3, 32, 32)),
        target_class_id=14,
        mechanism_gate_passed=False,
        **HASHES,
        **provenance_hashes,
    )
    path = tmp_path / "p1_feasibility_sdh_state.pt"
    torch.save(payload, path)
    state_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, state_hash, provenance_hashes


def test_v0_materializer_loads_exact_failed_gate_state_and_emits_provenance(
    tmp_path,
) -> None:
    path, state_hash, provenance_hashes = _v0_state(tmp_path)
    cfg, method_cfg = _cfg(path)
    method_cfg.update(
        {
            "protocol_id": "TAUSB-SDH-E2E-V0-MAP50-v1",
            "materialization_mode": "p1_feasibility_state",
            "allow_failed_scientific_gates": True,
            "frozen_sdh_state_sha256": state_hash,
            **provenance_hashes,
        }
    )
    materializer = SDHMaterializer(
        cfg, method_cfg, torch.device("cpu"), torch.nn.Identity()
    )
    result = materializer.generate(
        np.full((40, 48, 3), 0.5, dtype=np.float32),
        [{"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.6]}],
        0,
        40,
        16 / 255,
        "bbox",
    )
    assert result.extras["evidence_scope"] == (
        "end_to_end_feasibility_not_formal_method"
    )
    assert result.extras["hiding_gate_passed"] is False
    assert result.extras["mechanism_gate_passed"] is False
    assert result.extras["frozen_sdh_state_sha256"] == state_hash


def test_v0_materializer_rejects_wrong_mechanism_provenance_hash(tmp_path) -> None:
    path, state_hash, provenance_hashes = _v0_state(tmp_path)
    cfg, method_cfg = _cfg(path)
    method_cfg.update(
        {
            "protocol_id": "TAUSB-SDH-E2E-V0-MAP50-v1",
            "materialization_mode": "p1_feasibility_state",
            "allow_failed_scientific_gates": True,
            "frozen_sdh_state_sha256": state_hash,
            **provenance_hashes,
        }
    )
    method_cfg["mechanism_metrics_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mechanism_metrics_sha256"):
        SDHMaterializer(cfg, method_cfg, torch.device("cpu"), torch.nn.Identity())


def _p4_candidate_state(tmp_path):
    torch.manual_seed(17)
    carrier = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        epsilon=16 / 255,
    )
    provenance_hashes = {
        name: format(index + 1, "x") * 64
        for index, name in enumerate(DGCAIP_P4_PROVENANCE_HASH_KEYS)
    }
    payload = build_dgcaip_p4_candidate_state_payload(
        carrier=carrier,
        secret=torch.rand((1, 3, 32, 32)),
        target_class_id=14,
        mechanism_scientific_gate_passed=False,
        provenance_hashes=provenance_hashes,
        **HASHES,
    )
    path = tmp_path / "p4_dgcaip_candidate_sdh_state.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), provenance_hashes


def test_p4_candidate_keeps_arm_identity_and_allows_scientific_fail(tmp_path) -> None:
    path, state_hash, provenance_hashes = _p4_candidate_state(tmp_path)
    cfg, method_cfg = _cfg(path)
    method_cfg.update(
        {
            "protocol_id": "TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1",
            "materialization_mode": "p4_dgcaip_candidate_state",
            "allow_failed_scientific_gates": True,
            "frozen_sdh_state_sha256": state_hash,
            **provenance_hashes,
        }
    )
    materializer = SDHMaterializer(
        cfg, method_cfg, torch.device("cpu"), torch.nn.Identity()
    )
    result = materializer.generate(
        np.full((40, 48, 3), 0.5, dtype=np.float32),
        [{"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.6]}],
        0,
        40,
        16 / 255,
        "bbox",
    )
    assert result.extras["source_arm_id"] == "P4-DGCAIP"
    assert result.extras["state_integrity_gate_passed"] is True
    assert result.extras["mechanism_scientific_gate_passed"] is False
    assert result.extras["evidence_scope"] == (
        "diagnostic_candidate_ap50_evaluation"
    )


def test_p4_candidate_cannot_be_loaded_as_legacy_p1(tmp_path) -> None:
    path, _, _ = _p4_candidate_state(tmp_path)
    with pytest.raises(ValueError, match="arm_id=P1"):
        load_frozen_sdh_state(
            str(path),
            device=torch.device("cpu"),
            expected_target_class_id=14,
            expected_epsilon=16 / 255,
            expected_hashes=HASHES,
        )
