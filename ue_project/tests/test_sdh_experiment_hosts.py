from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ue_framework.methods.sdh_experiment import _first_person_host


@pytest.mark.parametrize(
    ("shape", "bbox"),
    [
        ((500, 375), (0.908, 0.297, 0.1786666667, 0.258)),
        ((375, 500), (0.958, 0.8853333333, 0.048, 0.096)),
        ((334, 500), (0.269, 0.8353293413, 0.146, 0.3233532934)),
    ],
)
def test_first_person_host_handles_non_square_edge_boxes(
    tmp_path: Path,
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float],
) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image_path = image_dir / "host.jpg"
    image = np.full((shape[0], shape[1], 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    (label_dir / "host.txt").write_text(
        "14 " + " ".join(str(value) for value in bbox) + "\n",
        encoding="utf-8",
    )

    host = _first_person_host(image_path, label_dir, torch.device("cpu"))

    assert host.shape == (3, 256, 256)
    assert torch.isfinite(host).all()
