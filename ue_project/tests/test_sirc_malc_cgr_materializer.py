from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ue_framework.methods.sirc_malc_cgr import (
    SIRCMALCCGRMaterializer,
    build_frozen_sirc_state_payload,
    load_frozen_sirc_state,
    resolve_sirc_malc_effective_method,
)


def _payload(*, gate_passed: bool = True):
    generator = torch.Generator(device="cpu").manual_seed(77)
    bases = torch.randn((4, 16, 8, 8), generator=generator)
    scales = torch.randn((4, 16), generator=generator)
    coefficients = 0.1 * torch.randn((16, 3), generator=generator)
    return build_frozen_sirc_state_payload(
        semantic_bases=bases,
        semantic_scales=scales,
        coefficients=coefficients,
        gamma=2.0,
        epsilon=16 / 255,
        variant_seed=2102,
        jnd_floor=0.5,
        target_class_id=14,
        semantic_bank_hash="a" * 64,
        source_manifest_hash="b" * 64,
        split_hash="c" * 64,
        mechanism_gate_passed=gate_passed,
    )


def _config(state_path: str):
    return {
        "experiment": {
            "target_class_id": 14,
            "eps": 16 / 255,
        },
        "surrogate": {
            "num_classes": 20,
            "imgsz": 640,
            "eot_samples": 1,
        },
        "methods": {},
    }, {
        "support_type": "mask",
        "enable_sirc_carrier": True,
        "enable_malc": True,
        "enable_cgr": True,
        "require_mechanism_pass": True,
        "frozen_carrier_state": state_path,
        "semantic_bank_hash": "a" * 64,
        "source_manifest_hash": "b" * 64,
        "split_hash": "c" * 64,
        "variant_seed": 2102,
        "jnd_floor": 0.5,
    }


def test_frozen_state_gate_and_content_hash_fail_closed(tmp_path) -> None:
    path = tmp_path / "failed_gate.pt"
    torch.save(_payload(gate_passed=False), path)
    with pytest.raises(ValueError, match="gate did not pass"):
        load_frozen_sirc_state(
            str(path),
            device="cpu",
            expected_target_class_id=14,
            expected_epsilon=16 / 255,
        )

    tampered = _payload()
    tampered["coefficients"] = tampered["coefficients"].clone()
    tampered["coefficients"][0, 0] += 1
    path = tmp_path / "tampered.pt"
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_frozen_sirc_state(
            str(path),
            device="cpu",
            expected_target_class_id=14,
            expected_epsilon=16 / 255,
        )


def test_materializer_is_deterministic_bounded_and_support_local(tmp_path) -> None:
    state_path = tmp_path / "a1.pt"
    torch.save(_payload(), state_path)
    cfg, method_cfg = _config(str(state_path))
    materializer = SIRCMALCCGRMaterializer(
        cfg,
        method_cfg,
        torch.device("cpu"),
        torch.nn.Identity(),
    )
    image = np.full((40, 48, 3), 0.5, dtype=np.float32)
    annotations = [
        {"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.6]},
        {"cls": 7, "bbox": [0.2, 0.2, 0.1, 0.1]},
    ]
    kwargs = {
        "image": image,
        "annotations": annotations,
        "seed": 0,
        "steps": 40,
        "eps": 16 / 255,
        "support_type": "mask",
        "image_path": "/data/000001.jpg",
    }
    first = materializer.generate(**kwargs)
    second = materializer.generate(**kwargs)
    assert np.array_equal(first.poisoned_image, second.poisoned_image)
    assert np.array_equal(first.perturbation, second.perturbation)
    assert first.extras["variant_index"] == second.extras["variant_index"]
    assert first.extras["support_source"] == "forced_pseudo_fallback"
    assert first.extras["source_manifest_hash"] == "b" * 64
    outside = first.support_mask == 0
    assert np.max(np.abs(first.perturbation[outside])) == 0
    assert np.max(np.abs(first.perturbation)) <= 16 / 255 + 1e-7


def test_materializer_rejects_eot_and_formal_feature_off(tmp_path) -> None:
    state_path = tmp_path / "a1.pt"
    torch.save(_payload(), state_path)
    cfg, method_cfg = _config(str(state_path))
    cfg_eot = copy.deepcopy(cfg)
    cfg_eot["surrogate"]["eot_samples"] = 2
    with pytest.raises(ValueError, match="forbids EOT"):
        SIRCMALCCGRMaterializer(
            cfg_eot,
            method_cfg,
            torch.device("cpu"),
            torch.nn.Identity(),
        )

    control = dict(method_cfg)
    control["enable_malc"] = False
    with pytest.raises(ValueError, match="MALC and CGR enabled"):
        SIRCMALCCGRMaterializer(
            cfg,
            control,
            torch.device("cpu"),
            torch.nn.Identity(),
        )


def test_all_v2_switches_off_selects_exact_legacy_route() -> None:
    assert resolve_sirc_malc_effective_method(
        {
            "enable_sirc_carrier": False,
            "enable_malc": False,
            "enable_cgr": False,
        }
    ) == "tausb_mask"
    assert resolve_sirc_malc_effective_method(
        {
            "enable_sirc_carrier": True,
            "enable_malc": False,
            "enable_cgr": False,
        }
    ) == "sirc_malc_cgr"
