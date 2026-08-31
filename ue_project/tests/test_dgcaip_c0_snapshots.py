from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ue_framework.tools.run_tausb_dgcaip_c0_snapshots import (
    MAX_TMP_ROOT_BYTES,
    SNAPSHOT_EPOCHS,
    WALL_SECONDS,
    _validate_config,
    validate_checkpoint_epoch,
    validate_tmp_root_path,
)


CONFIG = (
    Path(__file__).parents[1]
    / "ue_framework"
    / "configs"
    / "tausb_sdh_dgcaip_dataset_cgr_proxy_c0_snapshots_v1.yaml"
)


def test_c0_snapshot_config_binds_exact_epoch_schedule_and_clean_arm() -> None:
    cfg = _validate_config(CONFIG)
    assert cfg["experiment"]["arm_id"] == "C0"
    assert cfg["experiment"]["poisoning_ratio"] == 0.0
    assert cfg["platform"]["save_every_n_epochs"] == 1
    assert cfg["platform"]["pack_every_n_epochs"] == 20
    assert cfg["platform"]["zip_after_stage"] is False
    assert cfg["victim"]["epochs"] == 20
    assert cfg["data"]["materialization_layout"] == "sparse_mixed_list_v1"
    assert SNAPSHOT_EPOCHS == {"e1": 0, "e5": 4, "e20": 19}
    assert WALL_SECONDS == 45 * 60


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        (
            "experiment",
            "arm_id",
            "M1",
            "expected_poisoned_count|Config experiment.arm_id",
        ),
        ("platform", "save_every_n_epochs", 5, "save_every_n_epochs"),
        ("platform", "zip_after_stage", True, "zip_after_stage"),
        ("victim", "epochs", 21, "victim epochs|victim.epochs"),
        ("data", "materialization_layout", "full_png_v1", "materialization_layout"),
    ],
)
def test_c0_snapshot_config_fails_closed(
    tmp_path: Path, section: str, key: str, value: object, match: str
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _validate_config(path)


@pytest.mark.parametrize("snapshot_id", ("e1", "e5", "e20"))
def test_checkpoint_epoch_metadata_uses_post_epoch_index(snapshot_id: str) -> None:
    expected = SNAPSHOT_EPOCHS[snapshot_id]
    validate_checkpoint_epoch({"epoch": expected}, expected)


def test_checkpoint_epoch_metadata_rejects_epoch5_as_e5() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        validate_checkpoint_epoch({"epoch": 5}, SNAPSHOT_EPOCHS["e5"])


@pytest.mark.parametrize("payload", ({}, {"epoch": True}, {"epoch": 4.0}))
def test_checkpoint_epoch_metadata_requires_integer(payload) -> None:
    with pytest.raises(ValueError, match="not an integer"):
        validate_checkpoint_epoch(payload, 4)


def test_tmp_root_keeps_af_unix_socket_headroom(tmp_path: Path) -> None:
    short = Path("/root/autodl-tmp/t/dg0r2")
    assert len(str(short).encode()) <= MAX_TMP_ROOT_BYTES
    validate_tmp_root_path(short)

    long_path = tmp_path / ("x" * (MAX_TMP_ROOT_BYTES + 1))
    with pytest.raises(ValueError, match="AF_UNIX"):
        validate_tmp_root_path(long_path)
