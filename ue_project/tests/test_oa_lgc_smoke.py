from pathlib import Path

import numpy as np
import torch

from oa_lgc.episodes import ImageRecord
from oa_lgc.smoke import run_smoke_chain


def _config():
    return {
        "target_class_id": 14,
        "num_classes": 20,
        "data": {"image_size": 24},
        "episode": {"support_size": 2, "query_size": 2},
        "model": {"pool_size": 2, "hidden_dim": 8},
        "carrier": {
            "object_resolution": 8, "eps": 0.1, "non_target_dilation": 0,
            "min_valid_fraction": 0.01, "interpolation": "bilinear", "soft_mask": True,
            "soft_edge_pixels": 1.0, "box_jitter": 0.0,
        },
        "virtual_update": {"learning_rate": 0.1, "mode": "head_only", "first_order": True, "selected_modules": []},
        "gain": {"rho_t": 0.8, "rho_k": 0.2, "eps": 1e-8, "min_valid_clean_gain": 1e-8, "min_valid_class_gain": 1e-8, "minimum_class_samples": 1},
        "objective": {"lambda_carrier": 1.0, "lambda_auth": 1.0, "lambda_reg": 0.001, "gradient_clip_norm": 1.0, "eps": 0.1},
        "optimization": {"outer_learning_rate": 0.01},
    }


def _records():
    annotations = (
        {"cls": 14, "bbox": [0.5, 0.5, 0.4, 0.5]},
        {"cls": 1, "bbox": [0.2, 0.2, 0.2, 0.2]},
    )
    return [ImageRecord(f"id{index}", f"image{index}", annotations) for index in range(6)]


def _loader(path):
    index = int(path.replace("image", ""))
    generator = np.random.default_rng(index)
    return generator.random((24, 24, 3), dtype=np.float32)


def test_smoke_end_to_end_forward_backward_and_delta_update():
    result = run_smoke_chain(_records(), _config(), virtual_steps=1, outer_steps=1, seed=0, image_loader=_loader)
    assert result.summary["forward_complete"] and result.summary["backward_complete"]
    assert result.summary["delta_updated"] and result.summary["base_model_unchanged"]
    assert result.summary["support_query_overlap_max"] == 0
    assert result.summary["valid_authorized_class_total"] >= 1
    assert result.gradient_rows[0]["gradient_norm"] > 0


def test_smoke_reproducibility():
    first = run_smoke_chain(_records(), _config(), 1, 1, 3, image_loader=_loader)
    second = run_smoke_chain(_records(), _config(), 1, 1, 3, image_loader=_loader)
    assert first.loss_rows == second.loss_rows
    assert torch.equal(first.delta_obj, second.delta_obj)
    assert first.episode_manifests == second.episode_manifests


def test_history_artifacts_untouched(tmp_path):
    sentinel = tmp_path / "historical.txt"
    sentinel.write_text("keep", encoding="utf-8")
    _ = run_smoke_chain(_records(), _config(), 1, 1, 1, image_loader=_loader)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_smoke_j3_and_j5_interfaces():
    for steps in (3, 5):
        result = run_smoke_chain(_records(), _config(), steps, 1, 2, image_loader=_loader)
        assert result.summary["finite"] and result.summary["virtual_steps"] == steps

