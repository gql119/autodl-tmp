from pathlib import Path
from tempfile import TemporaryDirectory

from ue_framework.methods.mtepi import build_checkpoint_manifest


def test_cross_checkpoint_functional_transfer_rejects_missing_same_trajectory_metadata():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = tmp_path / "late.pt"
        ckpt.write_bytes(b"checkpoint")
        manifest = build_checkpoint_manifest(
            [{"name": "late", "role": "late", "path": "late.pt", "class_space": "VOC20", "architecture": "YOLOv8n"}],
            tmp_path,
        )
    assert manifest["legal_same_trajectory"] is False
    assert "late" in manifest["roles_present"]
