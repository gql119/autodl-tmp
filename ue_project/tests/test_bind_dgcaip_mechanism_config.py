from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config
from ue_framework.tools.bind_dgcaip_mechanism_config import (
    bind_dgcaip_mechanism_config,
)


ROOT = Path(__file__).parents[1]
TEMPLATE = (
    ROOT
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_mechanism_v2.template.yaml"
)
D0_CONFIG = (
    ROOT / "ue_framework" / "configs" / "tausb_sdh_dgcaip_d0_v2.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d0_payload(*, passed: bool = True):
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    return {
        "schema": "tausb.dgcaip-d0-run.v1",
        "spec_id": template["spec"]["spec_id"],
        "split_hash": template["dgcaip"]["expected_split_sha256"],
        "source_p1_state_sha256": template["dgcaip"][
            "source_p1_state_sha256"
        ],
        "decision": {"pass": passed},
    }


def test_d0_config_is_runnable_but_mechanism_template_is_not() -> None:
    d0 = yaml.safe_load(D0_CONFIG.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(d0)
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="d0_report_sha256"):
        validate_sdh_experiment_config(template)


def test_binder_requires_passed_matching_d0_and_writes_hashed_config(
    tmp_path: Path,
) -> None:
    d0_path = tmp_path / "d0_locator.json"
    d0_path.write_text(json.dumps(_d0_payload()), encoding="utf-8")
    output = tmp_path / "mechanism.yaml"
    remote_reference = "/root/autodl-tmp/dgcaip/d0/d0_locator.json"
    report = bind_dgcaip_mechanism_config(
        TEMPLATE, d0_path, output, d0_reference=remote_reference
    )
    bound = yaml.safe_load(output.read_text(encoding="utf-8"))
    validate_sdh_experiment_config(bound)
    assert bound["dgcaip"]["d0_report"] == remote_reference
    assert bound["dgcaip"]["d0_report_sha256"] == _sha256(d0_path)
    assert report["output_sha256"] == _sha256(output)
    assert output.with_suffix(".yaml.binding.json").is_file()


@pytest.mark.parametrize("mutation", ("fail", "split", "source"))
def test_binder_fails_closed_on_invalid_d0(tmp_path: Path, mutation: str) -> None:
    payload = copy.deepcopy(_d0_payload())
    if mutation == "fail":
        payload["decision"]["pass"] = False
    elif mutation == "split":
        payload["split_hash"] = "f" * 64
    else:
        payload["source_p1_state_sha256"] = "e" * 64
    d0_path = tmp_path / "d0_locator.json"
    d0_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="DG-CAIP"):
        bind_dgcaip_mechanism_config(
            TEMPLATE, d0_path, tmp_path / "mechanism.yaml"
        )
