from types import SimpleNamespace

from ue_framework.stages import train_victim


def test_fresh_victim_seed_covers_initialization_and_ultralytics(tmp_path, monkeypatch):
    events = []
    train_kwargs = {}

    class DummyYOLO:
        def __init__(self, init):
            events.append(("init", init))

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
    monkeypatch.setattr(train_victim, "mark_stage_completed", lambda *_args: None)
    monkeypatch.setattr(train_victim, "_write_train_yaml", lambda _ctx: "train.yaml")
    monkeypatch.setattr(train_victim, "remove_yolo_cache_files", lambda _paths: 0)
    monkeypatch.setattr(train_victim, "resolve_workers", lambda *_args: 4)
    monkeypatch.setattr(train_victim, "_snapshot_train_state", lambda *_args: 199)
    monkeypatch.setattr(train_victim, "pack_run_artifacts", lambda _paths: "bundle.zip")
    monkeypatch.setattr(train_victim, "atomic_write_json", lambda *_args: None)

    train_victim.run_train_victim(ctx)

    assert events[:2] == [
        ("seed", 0),
        ("init", "configs/voc_yolov8n_20cls.yaml"),
    ]
    assert train_kwargs["seed"] == 0
