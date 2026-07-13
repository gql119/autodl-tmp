from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np


def unique_run_id(stage: str, seed: int, now: datetime | None = None) -> str:
    instant = now or datetime.now()
    return f"{instant:%Y%m%d_%H%M%S_%f}_{stage}_seed{int(seed)}"


def create_run_dir(root: str | os.PathLike[str], run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    path = Path(root).resolve() / run_id
    if path.exists():
        raise FileExistsError(f"artifact path already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_png(path: str | os.PathLike[str], rgb_float: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    if uint8.ndim == 3 and uint8.shape[2] == 3:
        uint8 = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", uint8)
    if not ok:
        raise RuntimeError(f"failed to encode PNG: {destination}")
    encoded.tofile(str(destination))

