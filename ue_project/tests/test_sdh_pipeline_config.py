from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ue_framework.config import SUPPORTED_METHODS, load_config
from ue_framework.methods.factory import build_generator
from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config


FORMAL = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
)
MECHANISM = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_mechanism_v3.yaml"
)


def test_formal_sdh_config_is_registered_and_has_no_legacy_features() -> None:
    config = load_config(str(FORMAL))
    method = config["methods"]["tausb_sdh"]
    assert "tausb_sdh" in SUPPORTED_METHODS
    assert method["support_type"] == "bbox"
    assert all(
        method[name] is True
        for name in (
            "enable_deep_hiding",
            "enable_dlfc",
            "enable_cicr",
            "enable_cgr",
            "enable_nla_loss",
        )
    )
    assert config["surrogate"]["eot_samples"] == 1
    assert "jnd_floor" not in method
    assert "carrier_basis_mode" not in method
    assert "enable_malc" not in method


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("methods", "tausb_sdh", "enable_deep_hiding", False), "all method gates"),
        (("methods", "tausb_sdh", "support_type", "mask"), "person GT bbox"),
        (("surrogate", "eot_samples", None, 2), "forbids EOT"),
    ],
)
def test_formal_config_fails_closed(tmp_path, mutation, match) -> None:
    payload = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    first, second, third, value = mutation
    if third is None:
        payload[first][second] = value
    else:
        payload[first][second][third] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(str(path))


def test_factory_never_silently_maps_sdh_to_legacy() -> None:
    with pytest.raises(FileNotFoundError, match="Frozen SDH state"):
        build_generator(
            "tausb_sdh",
            {"experiment": {"target_class_id": 14, "eps": 16 / 255},
             "surrogate": {"num_classes": 20, "imgsz": 640}},
            {
                "frozen_sdh_state": "definitely-missing.pt",
                "secret_source_sha256": "a" * 64,
                "secret_tensor_sha256": "b" * 64,
                "source_manifest_sha256": "c" * 64,
                "train_split_sha256": "d" * 64,
            },
            "cpu",
            __import__("torch").nn.Identity(),
        )


def test_capped_mechanism_config_freezes_cost_and_protocol() -> None:
    config = yaml.safe_load(MECHANISM.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(config)
    assert config["hiding"]["max_seconds"] == 1200
    assert config["mechanism"]["max_seconds"] == 900
    assert config["mechanism"]["calibration_batches"] == 16
    assert config["mechanism"]["heldout_batches"] == 24
    assert config["mechanism"]["optimization_steps"] == 8
    assert config["mechanism"]["max_backtracks"] == 5
    assert config["mechanism"]["eot_enabled"] is False
    assert config["mechanism"]["jnd_enabled"] is False
