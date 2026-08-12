from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from ue_framework.sparse_dataset import (
    SPARSE_MIXED_LIST_LAYOUT,
    audit_sparse_training_list,
    file_sha256,
    png_roundtrip_exact,
    save_png_and_reload,
    write_train_path_list,
    yolo_label_path_for_image,
)
from ue_framework.tools.bind_tausb_sdh_e2e_v0 import _bound_config
from ue_framework.tools.run_tausb_sdh_sparse_e20 import (
    DISK_RESERVE_BYTES,
    c0_ap50_is_interpretable,
    validate_sparse_pair,
)
from ue_framework.stages.train_victim import _probe_sparse_dataloader


ROOT = Path(__file__).parents[1]
FORMAL = ROOT / "ue_framework" / "configs" / (
    "exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
)
HASH_KEYS = (
    "frozen_sdh_state_sha256",
    "hiding_metrics_sha256",
    "hiding_checkpoint_sha256",
    "hiding_split_sha256",
    "mechanism_metrics_sha256",
    "mechanism_decision_sha256",
    "mechanism_config_sha256",
    "p1_state_sha256",
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 20, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _record(source: Path, effective: Path, label: Path, poisoned: bool) -> dict:
    return {
        "stem": source.stem,
        "image_path": str(effective.resolve()),
        "source_image_path": str(source.resolve()),
        "label_path": str(label.resolve()),
        "source_image_sha256": file_sha256(str(source)),
        "label_sha256": file_sha256(str(label)),
        "saved_image_sha256": file_sha256(str(effective)) if poisoned else "",
        "is_poisoned": "1" if poisoned else "0",
        "has_target": "1" if poisoned else "0",
        "support_outside_linf": "0.00000000",
    }


def test_sparse_list_audit_accepts_mixed_sources_and_fails_closed(tmp_path) -> None:
    clean = tmp_path / "voc" / "images" / "train" / "clean.jpg"
    clean_label = tmp_path / "voc" / "labels" / "train" / "clean.txt"
    target = tmp_path / "voc" / "images" / "train" / "target.jpg"
    source_target_label = tmp_path / "voc" / "labels" / "train" / "target.txt"
    poisoned = tmp_path / "run" / "images" / "poisoned" / "target.png"
    poisoned_label = tmp_path / "run" / "labels" / "poisoned" / "target.txt"
    _write_rgb(clean, 30)
    _write_rgb(target, 60)
    _write_rgb(poisoned, 61)
    clean_label.parent.mkdir(parents=True, exist_ok=True)
    clean_label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    source_target_label.write_text("14 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    poisoned_label.parent.mkdir(parents=True, exist_ok=True)
    poisoned_label.write_bytes(source_target_label.read_bytes())

    rows = [
        _record(clean, clean, clean_label, False),
        _record(target, poisoned, poisoned_label, True),
    ]
    path_list = tmp_path / "run" / "train-images.txt"
    write_train_path_list(str(path_list), [str(clean), str(poisoned)])
    report = audit_sparse_training_list(
        str(path_list),
        rows,
        expected_total=2,
        expected_poisoned=1,
        expected_target=1,
        target_class_id=14,
        num_classes=20,
    )
    assert report["poisoned_png_count"] == 1
    assert report["original_jpeg_count"] == 1
    assert yolo_label_path_for_image(str(poisoned)) == str(poisoned_label.resolve())

    ctx = SimpleNamespace(
        cfg={"victim": {"imgsz": 32}},
    )
    probe = _probe_sparse_dataloader(ctx, report)
    assert probe["dataset_count"] == 2
    assert probe["batch_image_shape"][0] == 1

    rows[1]["saved_image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PNG hash mismatch"):
        audit_sparse_training_list(
            str(path_list), rows, expected_total=2, expected_poisoned=1,
            expected_target=1, target_class_id=14, num_classes=20,
        )


def test_png_save_reload_and_clean_roundtrip_are_exact(tmp_path) -> None:
    rng = np.random.default_rng(4)
    uint8 = rng.integers(0, 256, size=(16, 13, 3), dtype=np.uint8)
    image = uint8.astype(np.float32) / 255.0
    assert png_roundtrip_exact(image)
    loaded, digest = save_png_and_reload(str(tmp_path / "poison.png"), image)
    assert np.array_equal(np.rint(loaded * 255).astype(np.uint8), uint8)
    assert digest == hashlib.sha256((tmp_path / "poison.png").read_bytes()).hexdigest()


def test_e20_binder_enables_sparse_layout_only_for_full_voc(tmp_path) -> None:
    base = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    hashes = {name: str(index + 1) * 64 for index, name in enumerate(HASH_KEYS)}
    e20 = _bound_config(
        base,
        arm_id="M1",
        pilot_kind="e20",
        dataset_root=tmp_path / "voc",
        state_path=tmp_path / "p1.pt",
        hashes=hashes,
        selection_path="",
        selection_hash="",
        run_root=str(tmp_path / "e20"),
    )
    smoke = _bound_config(
        base,
        arm_id="M1",
        pilot_kind="smoke",
        dataset_root=tmp_path / "voc",
        state_path=tmp_path / "p1.pt",
        hashes=hashes,
        selection_path=str(tmp_path / "selection.json"),
        selection_hash="a" * 64,
        run_root=str(tmp_path / "smoke"),
    )
    assert e20["data"]["materialization_layout"] == SPARSE_MIXED_LIST_LAYOUT
    assert smoke["data"]["materialization_layout"] == "full_png_v1"


def test_sparse_pair_gate_requires_exact_frozen_counts_and_identity() -> None:
    shared = {
        "ordered_stems_sha256": "a" * 64,
        "label_content_manifest_sha256": "b" * 64,
        "total_count": 16551,
        "target_count": 6095,
        "clean_png_roundtrip_probe_count": 64,
    }
    c0 = {
        **shared,
        "poisoned_count": 0,
        "poisoned_png_count": 0,
        "original_jpeg_count": 16551,
    }
    m1 = {
        **shared,
        "poisoned_count": 6095,
        "poisoned_png_count": 6095,
        "original_jpeg_count": 10456,
    }
    assert validate_sparse_pair(c0, m1)["pass"] is True
    assert DISK_RESERVE_BYTES == 3 * 1024 ** 3
    m1["ordered_stems_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="ordered_stems"):
        validate_sparse_pair(c0, m1)


def test_c0_all_zero_ap50_stops_before_m1_training() -> None:
    assert c0_ap50_is_interpretable({"ap50_by_class": {"person": 0.2}}) is True
    assert c0_ap50_is_interpretable({"ap50_by_class": {"person": 0.0}}) is False
    with pytest.raises(ValueError, match="named AP50"):
        c0_ap50_is_interpretable({})


def test_e20_only_binder_does_not_build_or_write_smoke_selection(
    tmp_path, monkeypatch
) -> None:
    from ue_framework.tools import bind_tausb_sdh_e2e_v0 as binder

    mechanism_root = tmp_path / "mechanism"
    mechanism_config = tmp_path / "mechanism.yaml"
    base_config = tmp_path / "base.yaml"
    dataset_root = tmp_path / "voc"
    output_dir = tmp_path / "binding"
    mechanism_config.write_text("runtime: {}\n", encoding="utf-8")
    base_config.write_text(FORMAL.read_text(encoding="utf-8"), encoding="utf-8")
    fake_binding = {
        "state": {"state_content_hash": "f" * 64, "mechanism_gate_passed": False},
        "state_path": tmp_path / "p1.pt",
        "metrics": {},
        "hashes": {name: str(index + 1) * 64 for index, name in enumerate(HASH_KEYS)},
    }
    monkeypatch.setattr(binder, "_arguments", lambda: type("Args", (), {
        "mechanism_root": str(mechanism_root),
        "mechanism_config": str(mechanism_config),
        "base_config": str(base_config),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "run_root_prefix": str(tmp_path / "runs"),
        "e20_only": True,
    })())
    monkeypatch.setattr(binder, "_validate_mechanism_binding", lambda *args: fake_binding)
    monkeypatch.setattr(
        binder,
        "build_smoke_selection_manifest",
        lambda *args: (_ for _ in ()).throw(AssertionError("smoke builder called")),
    )
    assert binder.main() == 0
    assert (output_dir / "e20-c0.yaml").is_file()
    assert (output_dir / "e20-m1.yaml").is_file()
    assert not (output_dir / "smoke-c0.yaml").exists()
    assert not (output_dir / "smoke_train_selection.json").exists()
    report = json.loads((output_dir / "binding_report.json").read_text(encoding="utf-8"))
    assert report["binding_scope"] == "e20_only"
