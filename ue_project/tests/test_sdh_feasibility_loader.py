from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
import torch
import yaml

from ue_framework.methods import sdh_experiment
from ue_framework.methods.sdh_materializer import (
    E2E_V0_EVIDENCE_SCOPE,
    E2E_V0_MATERIALIZATION_MODE,
    build_feasibility_sdh_state_payload,
    load_frozen_sdh_state,
)
from ue_framework.methods.semantic_hiding_carrier import SemanticHidingCarrier


R2_CHECKPOINT_HASH = sdh_experiment.E2E_V0_R2_CHECKPOINT_SHA256
R2_METRICS_HASH = sdh_experiment.E2E_V0_R2_METRICS_SHA256


class _FakeCarrier(torch.nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        width: int,
        coupling_blocks: int,
        epsilon: float,
        hf_subband_scale: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.coupling_blocks = coupling_blocks
        self.epsilon = epsilon
        self.hf_subband_scale = hf_subband_scale

    def architecture_sha256(self) -> str:
        return "e" * 64

    def freeze_for_detector_optimization(self) -> None:
        return None


def _write_hiding_artifacts(
    root: Path, *, extra_failure: Optional[str] = None
) -> None:
    hiding = root / "hiding"
    hiding.mkdir(parents=True)
    checks = {
        "delta_high_frequency": False,
        "dlfc_leakage_probe": True,
        "finite": True,
        "linf": True,
        "pixel_diversity": True,
        "primary_l1_margin": True,
        "primary_recovery_ssim": True,
        "retrieval_top1": True,
        "rms_diversity": False,
        "support": True,
    }
    if extra_failure is not None:
        checks[extra_failure] = False
    (hiding / "hiding_metrics.json").write_text(
        json.dumps(
            {
                "schema": "tausb.sdh-hiding-pilot.v1",
                "checkpoint_sha256": R2_CHECKPOINT_HASH,
                "gate": {"checks": checks, "pass": False, "status": "fail"},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "schema": "tausb.sdh-hiding-checkpoint.v1",
            "carrier_state": {},
            "architecture_sha256": "e" * 64,
            "hf_subband_scale": 1.0,
            "split_hash": "f" * 64,
            "primary_secret": torch.zeros((1, 3, 256, 256)),
        },
        hiding / "hiding_checkpoint.pt",
    )


def _v0_loader_config(root: Path) -> dict:
    return {
        "spec": {"spec_id": sdh_experiment.E2E_V0_SPEC_ID},
        "hiding": {
            "source_artifact_root": str(root),
            "source_metrics_sha256": R2_METRICS_HASH,
            "source_checkpoint_sha256": R2_CHECKPOINT_HASH,
            "allow_failed_scientific_gates": True,
            "hf_subband_scale": 1.0,
        },
        "runtime": {"artifact_root": str(root / "new-output")},
    }


def _patch_hashes(monkeypatch) -> None:
    def fake_hash(path: Path) -> str:
        if path.name == "hiding_metrics.json":
            return R2_METRICS_HASH
        if path.name == "hiding_checkpoint.pt":
            return R2_CHECKPOINT_HASH
        raise AssertionError("unexpected hash path: %s" % path)

    monkeypatch.setattr(sdh_experiment, "_file_sha256", fake_hash)
    monkeypatch.setattr(sdh_experiment, "SemanticHidingCarrier", _FakeCarrier)


def test_exact_r2_feasibility_loader_preserves_failed_gate(tmp_path, monkeypatch) -> None:
    _write_hiding_artifacts(tmp_path)
    _patch_hashes(monkeypatch)

    carrier, secret, state = sdh_experiment._load_hiding_checkpoint(
        _v0_loader_config(tmp_path),
        config_base=tmp_path,
        device=torch.device("cpu"),
    )

    assert isinstance(carrier, _FakeCarrier)
    assert carrier.hf_subband_scale == 1.0
    assert secret.shape == (1, 3, 256, 256)
    assert state["split_hash"] == "f" * 64
    assert state["e2e_v0_hiding_provenance"] == {
        "evidence_scope": "end_to_end_feasibility_not_formal_method",
        "hiding_gate_passed": False,
        "failed_hiding_checks": ["delta_high_frequency", "rms_diversity"],
        "hiding_metrics_sha256": R2_METRICS_HASH,
        "hiding_checkpoint_sha256": R2_CHECKPOINT_HASH,
    }


def test_v0_loader_rejects_unexpected_failed_hiding_check(tmp_path, monkeypatch) -> None:
    _write_hiding_artifacts(tmp_path, extra_failure="support")
    _patch_hashes(monkeypatch)

    with pytest.raises(ValueError, match="failures differ from frozen r2"):
        sdh_experiment._load_hiding_checkpoint(
            _v0_loader_config(tmp_path),
            config_base=tmp_path,
            device=torch.device("cpu"),
        )


def test_v0_loader_rejects_wrong_frozen_hash_before_loading(tmp_path, monkeypatch) -> None:
    _write_hiding_artifacts(tmp_path)
    _patch_hashes(monkeypatch)
    config = _v0_loader_config(tmp_path)
    config["hiding"]["source_metrics_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="not the frozen r2 hash"):
        sdh_experiment._load_hiding_checkpoint(
            config,
            config_base=tmp_path,
            device=torch.device("cpu"),
        )


def test_formal_loader_still_rejects_the_same_failed_gate(tmp_path, monkeypatch) -> None:
    _write_hiding_artifacts(tmp_path)
    _patch_hashes(monkeypatch)
    config = {
        "spec": {"spec_id": "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3"},
        "hiding": {"hf_subband_scale": 1.0},
        "runtime": {"artifact_root": str(tmp_path)},
    }

    with pytest.raises(ValueError, match="Hiding gate did not pass"):
        sdh_experiment._load_hiding_checkpoint(
            config,
            config_base=tmp_path,
            device=torch.device("cpu"),
        )


def test_v0_experiment_config_is_exact_and_formal_cannot_opt_in() -> None:
    retry_path = (
        Path(__file__).parents[1]
        / "ue_framework"
        / "configs"
        / "tausb_sdh_mechanism_v3_r2.yaml"
    )
    config = yaml.safe_load(retry_path.read_text(encoding="utf-8"))
    config["spec"]["spec_id"] = sdh_experiment.E2E_V0_SPEC_ID
    config["runtime"]["artifact_root"] = "/root/tausb-sdh-runs/e2e-v0-mechanism"
    config["dataset"].update(
        {
            "expected_train_images": 16551,
            "expected_person_images": 6095,
            "train_image_manifest_sha256": (
                sdh_experiment.E2E_V0_TRAIN_IMAGE_MANIFEST_SHA256
            ),
        }
    )
    config["model"]["surrogate_checkpoint_sha256"] = (
        sdh_experiment.E2E_V0_SURROGATE_CHECKPOINT_SHA256
    )
    config["hiding"].update(
        {
            "source_artifact_root": (
                "/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-r2"
            ),
            "source_metrics_sha256": R2_METRICS_HASH,
            "source_checkpoint_sha256": R2_CHECKPOINT_HASH,
            "allow_failed_scientific_gates": True,
        }
    )
    sdh_experiment.validate_sdh_experiment_config(config)

    config["spec"]["spec_id"] = "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3"
    with pytest.raises(ValueError, match="restricted to the E2E V0 Spec"):
        sdh_experiment.validate_sdh_experiment_config(config)


def test_v0_experiment_config_rejects_unbound_surrogate() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "ue_framework"
        / "configs"
        / "tausb_sdh_e2e_v0_mechanism.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["surrogate_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="surrogate checkpoint hash mismatch"):
        sdh_experiment.validate_sdh_experiment_config(config)


def test_v0_runtime_inputs_are_content_bound(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "dataset" / "images" / "train"
    label_dir = tmp_path / "dataset" / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(2):
        (image_dir / ("%06d.jpg" % index)).write_bytes(b"image" + bytes([index]))
        (label_dir / ("%06d.txt" % index)).write_text(
            "14 0.5 0.5 0.2 0.3\n", encoding="utf-8"
        )
    surrogate = tmp_path / "surrogate.pt"
    surrogate.write_bytes(b"surrogate")
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "a" / "b" / "c" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    primary = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    source_hash = sdh_experiment._file_sha256(source)
    manifest_hash = "a" * 64

    def fake_secret_bank(config, base):
        return primary.clone(), 0, {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "records": [
                {
                    "source_id": "primary",
                    "source_file": "source.jpg",
                    "source_sha256": source_hash,
                }
            ],
        }

    monkeypatch.setattr(sdh_experiment, "_load_secret_bank", fake_secret_bank)
    config = {
        "dataset": {
            "expected_train_images": 2,
            "train_image_manifest_sha256": sdh_experiment._path_size_manifest_sha256(
                list(image_dir.glob("*.jpg"))
            ),
            "train_label_manifest_sha256": sdh_experiment._path_content_manifest_sha256(
                list(label_dir.glob("*.txt"))
            ),
        },
        "model": {
            "surrogate_checkpoint": str(surrogate),
            "surrogate_checkpoint_sha256": sdh_experiment._file_sha256(surrogate),
        },
        "secrets": {
            "primary_tensor_sha256": sdh_experiment._float_tensor_sha256(primary),
        },
    }
    hashes = sdh_experiment._validate_e2e_v0_runtime_inputs(
        config,
        config_base=tmp_path,
        image_dir=image_dir,
        label_dir=label_dir,
        hiding_state={"secret_manifest_sha256": manifest_hash, "primary_index": 0},
        primary_secret=primary,
    )
    assert hashes["train_image_count"] == 2
    assert hashes["train_label_count"] == 2
    assert hashes["surrogate_checkpoint_sha256"] == config["model"][
        "surrogate_checkpoint_sha256"
    ]

    surrogate.write_bytes(b"different")
    with pytest.raises(ValueError, match="surrogate checkpoint file hash mismatch"):
        sdh_experiment._validate_e2e_v0_runtime_inputs(
            config,
            config_base=tmp_path,
            image_dir=image_dir,
            label_dir=label_dir,
            hiding_state={"secret_manifest_sha256": manifest_hash, "primary_index": 0},
            primary_secret=primary,
        )


def _feasibility_payload(carrier: SemanticHidingCarrier) -> dict:
    return build_feasibility_sdh_state_payload(
        carrier=carrier,
        secret=torch.zeros((1, 3, carrier.input_size, carrier.input_size)),
        target_class_id=14,
        secret_source_sha256="a" * 64,
        secret_tensor_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        train_split_sha256="d" * 64,
        mechanism_gate_passed=False,
        hiding_metrics_sha256=R2_METRICS_HASH,
        hiding_checkpoint_sha256=R2_CHECKPOINT_HASH,
        hiding_split_sha256="e" * 64,
        mechanism_metrics_sha256="f" * 64,
        mechanism_decision_sha256="1" * 64,
        mechanism_config_sha256="2" * 64,
        p1_state_sha256="3" * 64,
    )


def test_feasibility_payload_keeps_real_gate_flags_and_formal_loader_rejects_it(
    tmp_path,
) -> None:
    carrier = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        epsilon=16 / 255,
    )
    payload = _feasibility_payload(carrier)
    assert payload["arm_id"] == "P1"
    assert payload["hiding_gate_passed"] is False
    assert payload["mechanism_gate_passed"] is False
    assert payload["materialization_mode"] == E2E_V0_MATERIALIZATION_MODE
    assert payload["evidence_scope"] == E2E_V0_EVIDENCE_SCOPE

    path = tmp_path / "p1_feasibility_sdh_state.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hiding_gate_passed"):
        load_frozen_sdh_state(
            str(path),
            device=torch.device("cpu"),
            expected_target_class_id=14,
            expected_epsilon=16 / 255,
            expected_hashes={
                "secret_source_sha256": "a" * 64,
                "secret_tensor_sha256": "b" * 64,
                "source_manifest_sha256": "c" * 64,
                "train_split_sha256": "d" * 64,
            },
        )


def test_feasibility_payload_rejects_nonfinite_p1_state() -> None:
    carrier = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        epsilon=16 / 255,
    )
    with torch.no_grad():
        next(carrier.parameters()).reshape(-1)[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite tensor"):
        _feasibility_payload(carrier)


def test_saved_p1_loader_uses_the_p1_arm_and_rejects_t0(tmp_path) -> None:
    torch.manual_seed(11)
    base = SemanticHidingCarrier(
        input_size=32,
        width=8,
        coupling_blocks=2,
        epsilon=16 / 255,
    )
    p1_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    first_key = next(iter(p1_state))
    p1_state[first_key].reshape(-1)[0] += 0.25
    path = tmp_path / "p1_state.pt"
    torch.save({"arm_id": "P1", "carrier_state": p1_state}, path)

    loaded = sdh_experiment._load_saved_p1_carrier(
        path,
        base_carrier=base,
        device=torch.device("cpu"),
    )
    assert torch.equal(loaded.state_dict()[first_key], p1_state[first_key])
    assert not torch.equal(loaded.state_dict()[first_key], base.state_dict()[first_key])

    torch.save({"arm_id": "T0", "carrier_state": p1_state}, path)
    with pytest.raises(ValueError, match="saved P1 arm state"):
        sdh_experiment._load_saved_p1_carrier(
            path,
            base_carrier=base,
            device=torch.device("cpu"),
        )
