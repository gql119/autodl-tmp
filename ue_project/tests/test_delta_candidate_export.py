import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def test_delta_candidate_metadata_roundtrip():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        delta_path = tmp_path / "delta_step_0.pt"
        meta_path = tmp_path / "delta_step_0.json"
        torch.save(torch.zeros(1, 3, 2, 2), delta_path)
        meta = {"name": "delta_step_0", "step": 0, "Delta_t": 1.0, "path": str(delta_path)}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        assert Path(loaded["path"]).exists()
        assert loaded["step"] == 0
