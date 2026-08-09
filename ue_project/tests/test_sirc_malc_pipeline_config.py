from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ue_framework.config import load_config
from ue_framework.stages.generate import resolve_effective_generation_method
from ue_framework.methods.malc_mechanism import (
    assert_matched_mechanism_configs,
)
from ue_framework.methods.sirc_probe import validate_sirc_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "ue_framework/configs/exp_voc_person_sirc_malc_cgr_map50_v2.yaml"
MECHANISM = ROOT / "ue_framework/configs/exp_voc_person_sirc_malc_mechanism_v2.yaml"


def test_formal_config_freezes_victim_no_eot_and_hashes() -> None:
    config = load_config(str(FORMAL))
    assert config["experiment"]["target_class_id"] == 14
    assert config["experiment"]["seeds"] == [0]
    assert config["surrogate"]["eot_samples"] == 1
    assert config["victim"]["epochs"] == 200
    assert config["victim"]["imgsz"] == 640
    assert config["victim"]["batch"] == 36
    assert config["victim"]["optimizer"] == "SGD"
    method = config["methods"]["sirc_malc_cgr"]
    assert method["enable_malc"] is True
    assert method["enable_cgr"] is True
    assert all(len(method[key]) == 64 for key in (
        "semantic_bank_hash",
        "source_manifest_hash",
        "split_hash",
    ))


def test_mechanism_config_is_no_eot_and_a0_a1_are_exactly_matched() -> None:
    config = yaml.safe_load(MECHANISM.read_text(encoding="utf-8"))
    validate_sirc_config(config)
    assert config["eot"]["enabled"] is False
    assert config["eot"]["samples"] == 1
    a0 = deepcopy(config)
    a1 = deepcopy(config)
    a0["method"]["enable_malc"] = False
    a1["method"]["enable_malc"] = True
    assert_matched_mechanism_configs(a0, a1)
    a1["optimization"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="differ only"):
        assert_matched_mechanism_configs(a0, a1)


def test_existing_tausb_config_does_not_activate_v2_validation() -> None:
    legacy = ROOT / "ue_framework/configs/exp_voc_person_tausb_formal.yaml"
    config = load_config(str(legacy))
    assert config["methods"]["tausb_mask"]["support_type"] == "mask"


def test_all_v2_switches_off_reuses_the_exact_legacy_method_config() -> None:
    config = load_config(str(FORMAL))
    method = config["methods"]["sirc_malc_cgr"]
    method["enable_sirc_carrier"] = False
    method["enable_malc"] = False
    method["enable_cgr"] = False
    effective, effective_config = resolve_effective_generation_method(
        "sirc_malc_cgr",
        config,
    )
    assert effective == "tausb_mask"
    assert effective_config is config["methods"]["tausb_mask"]
