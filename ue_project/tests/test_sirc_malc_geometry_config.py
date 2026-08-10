from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ue_framework.methods.sirc_malc_geometry import validate_geometry_config
from ue_framework.methods.sirc_probe import validate_sirc_config


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "ue_framework/configs/exp_voc_person_malc_grad_geometry_audit_v1.yaml"
V2 = ROOT / "ue_framework/configs/exp_voc_person_sirc_malc_mechanism_v2.yaml"


def test_geometry_config_freezes_bounded_no_eot_probe() -> None:
    config = yaml.safe_load(GEOMETRY.read_text(encoding="utf-8"))
    validate_geometry_config(config)
    validate_sirc_config(config)
    assert config["geometry"] == {
        "calibration_images": 64,
        "heldout_images": 96,
        "microtrajectory_steps": 8,
        "run_microtrajectory": True,
    }
    assert config["runtime"]["artifact_root"].endswith(
        "TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry"
    )
    assert config["eot"]["enabled"] is False
    assert config["method"]["enable_cgr"] is True


def test_geometry_config_rejects_silent_protocol_reduction() -> None:
    config = yaml.safe_load(GEOMETRY.read_text(encoding="utf-8"))
    reduced = deepcopy(config)
    reduced["geometry"]["calibration_images"] = 8
    with pytest.raises(ValueError, match="exactly 64"):
        validate_geometry_config(reduced)


def test_v2_config_validation_remains_accepted() -> None:
    config = yaml.safe_load(V2.read_text(encoding="utf-8"))
    validate_sirc_config(config)
