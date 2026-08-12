from types import SimpleNamespace
import json

import pytest
import torch

from ue_framework.stages import train_victim


def test_fresh_victim_seed_covers_initialization_and_ultralytics(tmp_path, monkeypatch):
    events = []
    train_kwargs = {}
    completed = {}

    class DummyYOLO:
        def __init__(self, init):
            events.append(("init", init))
            self.model = torch.nn.Linear(2, 2)

        def add_callback(self, *_args, **_kwargs):
            return None

        def train(self, **kwargs):
            train_kwargs.update(kwargs)

    artifact_root = tmp_path / "artifact"
    paths = SimpleNamespace(
        artifact_status_json=str(artifact_root / "status.json"),
        artifact_root=str(artifact_root),
        poisoned_root=str(tmp_path / "poisoned"),
        checkpoints_dir=str(artifact_root / "checkpoints"),
        logs_dir=str(artifact_root / "logs"),
        train_project_dir=str(artifact_root / "train_runs"),
    )
    victim_cfg = {
        "init": "configs/voc_yolov8n_20cls.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 36,
        "optimizer": "SGD",
        "cos_lr": True,
        "close_mosaic": 10,
        "cache": False,
        "amp": True,
        "save_period": 10,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
    }
    ctx = SimpleNamespace(
        cfg={
            "victim": victim_cfg,
            "surrogate": {"ckpt": str(tmp_path / "surrogate.pt")},
            "platform": {
                "resume": False,
                "save_every_n_epochs": 10,
                "pack_every_n_epochs": 10,
            },
        },
        seed=0,
        method="sirc_malc_cgr",
        steps=40,
        run_tag="C0",
        gpu_id=0,
        platform_mode="cloud",
        paths=paths,
        dataset_root=str(tmp_path / "clean"),
        train_img_dir=str(tmp_path / "clean/images/train"),
        val_img_dir=str(tmp_path / "clean/images/val"),
        train_label_dir=str(tmp_path / "clean/labels/train"),
        val_label_dir=str(tmp_path / "clean/labels/val"),
    )

    monkeypatch.setattr(
        train_victim,
        "set_global_seed",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(train_victim, "YOLO", DummyYOLO)
    monkeypatch.setattr(train_victim, "load_or_init_status", lambda *_args: {})
    monkeypatch.setattr(train_victim, "mark_stage_running", lambda *_args: {})
    monkeypatch.setattr(
        train_victim,
        "mark_stage_completed",
        lambda _path, _status, _stage, extra: completed.update(extra),
    )
    monkeypatch.setattr(train_victim, "save_status", lambda *_args: None)
    monkeypatch.setattr(
        train_victim, "_write_train_yaml", lambda _ctx, **_kwargs: "train.yaml"
    )
    monkeypatch.setattr(train_victim, "remove_yolo_cache_files", lambda _paths: 0)
    monkeypatch.setattr(train_victim, "resolve_workers", lambda *_args: 4)
    monkeypatch.setattr(train_victim, "_snapshot_train_state", lambda *_args: 199)
    monkeypatch.setattr(train_victim, "pack_run_artifacts", lambda _paths: "bundle.zip")
    monkeypatch.setattr(train_victim, "atomic_write_json", lambda *_args: None)

    train_victim.run_train_victim(ctx)

    assert events[:3] == [
        ("seed", 0),
        ("seed", 0),
        ("init", "configs/voc_yolov8n_20cls.yaml"),
    ]
    assert train_kwargs["seed"] == 0
    assert completed["fresh_init"]["resume_enabled"] is False
    assert completed["fresh_init"]["surrogate_checkpoint_used_for_victim_init"] is False
    assert completed["fresh_init"]["matches_expected_victim_init"] is None
    assert len(completed["fresh_init"]["victim_init_tensor_sha256"]) == 64


def test_canonical_model_tensor_hash_is_stable_and_weight_sensitive():
    class Wrapper:
        def __init__(self, value):
            self.model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                self.model.weight.fill_(value)
                self.model.bias.zero_()

    first = train_victim._canonical_model_tensor_sha256(Wrapper(1.0))
    same = train_victim._canonical_model_tensor_sha256(Wrapper(1.0))
    changed = train_victim._canonical_model_tensor_sha256(Wrapper(2.0))
    assert first == same
    assert first != changed


def test_canonical_model_tensor_hash_supports_scalar_integer_buffers():
    class ScalarBufferModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("seen", torch.tensor(0, dtype=torch.long))

    wrapper = SimpleNamespace(model=ScalarBufferModel())
    assert len(train_victim._canonical_model_tensor_sha256(wrapper)) == 64


def test_fresh_init_mismatch_is_recorded_before_training(tmp_path, monkeypatch):
    events = []

    class DummyYOLO:
        def __init__(self, _init):
            self.model = torch.nn.Linear(2, 2)

        def add_callback(self, *_args, **_kwargs):
            events.append("callback")

        def train(self, **_kwargs):
            events.append("train")

    artifact = tmp_path / "artifact"
    paths = SimpleNamespace(
        artifact_status_json=str(artifact / "status.json"),
        artifact_root=str(artifact),
        poisoned_root=str(tmp_path / "poisoned"),
        checkpoints_dir=str(artifact / "checkpoints"),
        logs_dir=str(artifact / "logs"),
        train_project_dir=str(artifact / "train_runs"),
    )
    ctx = SimpleNamespace(
        cfg={
            "victim": {
                "init": "model.yaml", "epochs": 200, "imgsz": 640, "batch": 36,
                "optimizer": "SGD", "cos_lr": True, "close_mosaic": 10,
                "cache": False, "amp": True, "save_period": 10,
            },
            "surrogate": {"ckpt": ""},
            "platform": {"resume": False, "save_every_n_epochs": 10},
        },
        seed=0, method="tausb_sdh", steps=40, run_tag="M1", gpu_id=0,
        platform_mode="cloud", paths=paths, dataset_root=str(tmp_path),
        train_img_dir="", val_img_dir="", train_label_dir="", val_label_dir="",
    )
    monkeypatch.setenv("TAUSB_EXPECTED_VICTIM_INIT_TENSOR_SHA256", "0" * 64)
    monkeypatch.setattr(train_victim, "YOLO", DummyYOLO)
    monkeypatch.setattr(train_victim, "load_or_init_status", lambda *_args: {})
    monkeypatch.setattr(train_victim, "mark_stage_running", lambda *_args: {})
    monkeypatch.setattr(train_victim, "save_status", lambda *_args: None)
    monkeypatch.setattr(train_victim, "_write_train_yaml", lambda *_args, **_kwargs: "train.yaml")
    monkeypatch.setattr(train_victim, "remove_yolo_cache_files", lambda _paths: 0)
    monkeypatch.setattr(train_victim, "resolve_workers", lambda *_args: 0)
    with pytest.raises(ValueError, match="paired C0 hash"):
        train_victim.run_train_victim(ctx)
    evidence = json.loads((artifact / "logs/fresh_init.json").read_text(encoding="utf-8"))
    assert evidence["matches_expected_victim_init"] is False
    assert "train" not in events
