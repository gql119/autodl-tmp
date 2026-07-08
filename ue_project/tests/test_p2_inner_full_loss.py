from pathlib import Path

import pytest
import torch

from ue_framework.core import ClassConditionedRouter
from ue_framework.core.yolov8_tal_adapter import YOLOv8TALAdapter
from ue_framework.methods.learning_trajectory import LearningTrajectoryMethod
from ue_framework.methods.learning_trajectory.class_conditioned_loss import compute_class_conditioned_detection_loss
from ue_framework.methods.learning_trajectory.virtual_update import (
    make_virtual_parameters,
    parameter_leak_max_abs_diff,
    snapshot_parameters,
)


def _load_real_adapter():
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    ckpt = Path("checkpoints/voc20_surrogate.pt")
    if not ckpt.is_file():
        pytest.skip("VOC20 surrogate checkpoint is not available.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = YOLO(str(ckpt))
    model = wrapper.model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return YOLOv8TALAdapter(model, num_classes=20, protected_class_id=14), device


def _make_batch(device):
    return {
        "cls": torch.tensor([14.0, 1.0], device=device),
        "bboxes": torch.tensor(
            [
                [0.30, 0.30, 0.24, 0.28],
                [0.72, 0.70, 0.22, 0.24],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
        "batch_size": 2,
    }


def _method_config():
    return {
        "protected_class_id": 14,
        "authorized_class_ids": "auto",
        "num_classes": 20,
        "trajectory": {"parameter_scope": "head", "normalize_per_parameter": True},
        "class_routing": {"exclude_ambiguous": True, "include_background_negatives": False},
        "virtual_update": {"parameter_scope": "head", "lr": 1.0e-4},
        "meta": {"lambda_meta": 1.0, "lambda_protected_query": 1.0, "enable_clean_counterfactual": True},
    }


def test_p2_inner_update_uses_full_detection_loss():
    adapter, device = _load_real_adapter()
    torch.manual_seed(7)
    images = torch.rand((2, 3, 160, 160), device=device)
    delta = (torch.randn((1, 3, 160, 160), device=device) * 0.001).requires_grad_(True)
    batch = _make_batch(device)

    predictions = adapter.forward((images + delta).clamp(0.0, 1.0))
    full_components = adapter.compute_detection_loss(predictions, batch, class_filter=None, return_components=True)
    protected_components = adapter.compute_detection_loss(predictions, batch, class_filter=[14], return_components=True)
    authorized_components = adapter.compute_detection_loss(predictions, batch, class_filter=[1], return_components=True)

    assert full_components["cls_loss"].detach().item() > 0.0
    assert full_components["box_loss"].detach().item() > 0.0
    assert full_components["dfl_loss"].detach().item() > 0.0
    assert not torch.allclose(full_components["total_loss"], protected_components["total_loss"])
    assert not torch.allclose(full_components["total_loss"], authorized_components["total_loss"])

    protected_changed = _make_batch(device)
    protected_changed["bboxes"] = protected_changed["bboxes"].clone()
    protected_changed["bboxes"][0] = torch.tensor([0.55, 0.25, 0.16, 0.20], device=device)
    authorized_changed = _make_batch(device)
    authorized_changed["bboxes"] = authorized_changed["bboxes"].clone()
    authorized_changed["bboxes"][1] = torch.tensor([0.45, 0.78, 0.16, 0.18], device=device)

    protected_changed_loss = adapter.compute_detection_loss(predictions, protected_changed, class_filter=None)
    authorized_changed_loss = adapter.compute_detection_loss(predictions, authorized_changed, class_filter=None)
    assert (protected_changed_loss - full_components["total_loss"]).detach().abs().item() > 1.0e-5
    assert (authorized_changed_loss - full_components["total_loss"]).detach().abs().item() > 1.0e-5

    router = ClassConditionedRouter(protected_class_id=14, authorized_class_ids="auto", num_classes=20)
    outer_losses = compute_class_conditioned_detection_loss(adapter, predictions, batch, router)
    assert outer_losses["protected_positive_count"].item() > 0
    assert outer_losses["authorized_positive_count"].item() > 0
    assert outer_losses["protected_total_loss"].detach().item() > 0.0
    assert outer_losses["authorized_total_loss"].detach().item() > 0.0

    selected = adapter.get_named_trainable_parameters("head")
    snapshot = snapshot_parameters(adapter.model)
    virtual = make_virtual_parameters(
        adapter.model,
        selected,
        support_loss=full_components["total_loss"],
        lr=1.0e-4,
        create_graph=True,
    )
    assert virtual.update_norm.detach().item() > 0.0
    assert parameter_leak_max_abs_diff(adapter.model, snapshot) == 0.0

    method = LearningTrajectoryMethod(adapter, _method_config())
    query_images = torch.rand((2, 3, 160, 160), device=device)
    result = method.compute_p2_step(images, query_images, batch, batch, delta)
    meta_grad = torch.autograd.grad(result["loss"], delta, retain_graph=False, allow_unused=False)[0]
    assert torch.isfinite(meta_grad).all()
    assert meta_grad.detach().norm().item() > 0.0
    assert result["logs"]["support_full_cls_loss"] > 0.0
    assert result["logs"]["support_full_box_loss"] > 0.0
    assert result["logs"]["support_full_dfl_loss"] > 0.0
    assert result["logs"]["parameter_leak_max_abs_diff"] == 0.0
