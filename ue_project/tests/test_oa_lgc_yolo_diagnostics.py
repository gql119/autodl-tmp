from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier
from oa_lgc.yolo_adapter import YOLOFunctionalAdapter
from oa_lgc.yolo_diagnostics import (
    CLASSWISE_DIAGNOSTIC_FIELDS,
    TARGET_DIAGNOSTIC_FIELDS,
    assignment_overlap,
    build_episode_diagnostics,
    coverage_valid_reason,
    positive_coverage,
)
from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations


DATASET_ROOT = Path("F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset")
CHECKPOINT = Path("F:/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt")
DEVICE = torch.device("cuda:0")
VOC_NAMES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow", 10: "diningtable",
    11: "dog", 12: "horse", 13: "motorbike", 14: "person", 15: "pottedplant",
    16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


def _load(source_id: str):
    image_path = DATASET_ROOT / "images" / "train" / f"{source_id}.jpg"
    label_path = DATASET_ROOT / "labels" / "train" / f"{source_id}.txt"
    annotations = read_yolo_annotations(str(label_path))
    image, adjusted = _letterbox_with_annotations(load_image_rgb_float(str(image_path)), annotations, 320)
    return image.to(DEVICE), adjusted


@pytest.fixture(scope="module")
def diagnostic_context():
    if not torch.cuda.is_available():
        pytest.skip("real TAL diagnostics require CUDA")
    if not CHECKPOINT.is_file():
        pytest.skip("VOC surrogate checkpoint is unavailable")
    adapter = YOLOFunctionalAdapter.from_checkpoint(CHECKPOINT, device=DEVICE)
    support_image, support_annotations = _load("000021")
    query_image, query_annotations = _load("000171")
    clean_batch = _batch_from_annotations(support_annotations, support_image, DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    delta = torch.nn.Parameter(torch.randn((3, 32, 32), generator=generator, device=DEVICE) * 1e-3)
    carrier = apply_object_aligned_carrier(
        support_image[0], support_annotations, delta, CarrierConfig()
    )
    poison_batch = _batch_from_annotations(
        support_annotations, carrier.poisoned.unsqueeze(0), DEVICE
    )
    clean = adapter.virtual_update(clean_batch, 1, 1e-4, "classification_head_only", create_graph=False)
    poison = adapter.virtual_update(poison_batch, 1, 1e-4, "classification_head_only", create_graph=False)
    query_batch = _batch_from_annotations(query_annotations, query_image, DEVICE)
    diagnostics = build_episode_diagnostics(
        adapter, query_image, query_batch, clean, poison, class_names=VOC_NAMES
    )
    return adapter, query_image, query_batch, diagnostics


def test_real_tal_positive_count(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    assert diagnostics.target.target_reference_positive_count > 0
    assert int(diagnostics.reference.fg_mask.sum()) > 0


def test_target_coverage_metric(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    target = diagnostics.target
    assert target.target_positive_coverage == positive_coverage(
        target.target_reference_positive_count, target.target_poison_positive_count
    )
    assert target.target_positive_coverage >= 0.5


def test_low_coverage_episode_invalid():
    assert positive_coverage(10, 4) == 0.4
    assert coverage_valid_reason(10, 4) == "target_positive_coverage_below_0.50"
    assert coverage_valid_reason(0, 0) == "no_reference_positive"


def test_box_loss_available(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    assert diagnostics.target.target_box_loss > 0
    assert torch.isfinite(torch.tensor(diagnostics.target.target_box_loss))


def test_dfl_loss_available(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    assert diagnostics.target.target_dfl_loss > 0
    assert torch.isfinite(torch.tensor(diagnostics.target.target_dfl_loss))


def test_classwise_assignment_drift(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    dog = diagnostics.classes[11]
    assert dog.valid
    assert 0.0 <= dog.assignment_drift <= 1.0
    assert dog.box_drift is not None and dog.box_drift >= 0
    assert dog.dfl_drift is not None and dog.dfl_drift >= 0


def test_classwise_box_drift(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    valid = [row for row in diagnostics.classes.values() if row.valid]
    assert valid
    assert all(row.box_drift is not None for row in valid)
    assert all(row.dfl_drift is not None for row in valid)


def test_reference_assignment_fixed(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    target = 14
    fixed_count = int((diagnostics.reference.target_scores[..., target] > 0).sum())
    assert fixed_count == diagnostics.target.target_reference_positive_count
    assert diagnostics.target_fixed_losses["clean_cls"].shape == torch.Size([])
    assert diagnostics.target_fixed_losses["poison_cls"].shape == torch.Size([])


def test_no_proxy_diagnostic_fallback(diagnostic_context):
    adapter, _, _, _ = diagnostic_context
    assert adapter.backend == "real_ultralytics_yolo"
    assert not hasattr(adapter, "proxy")


def test_diagnostic_schema_complete(diagnostic_context):
    _, _, _, diagnostics = diagnostic_context
    assert set(diagnostics.target_dict()) == set(TARGET_DIAGNOSTIC_FIELDS)
    assert set(diagnostics.class_rows()[0]) == set(CLASSWISE_DIAGNOSTIC_FIELDS)
    assert len(diagnostics.class_rows()) == 20
    assert assignment_overlap(
        diagnostics.reference.fg_mask, diagnostics.reference.fg_mask
    ) == 1.0
