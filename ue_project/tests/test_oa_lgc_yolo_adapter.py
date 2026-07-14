from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dcss.stage0_collection import _batch_from_annotations, _letterbox_with_annotations
from oa_lgc.carrier import CarrierConfig, apply_object_aligned_carrier
from oa_lgc.yolo_adapter import YOLOFunctionalAdapter
from ue_framework.data_utils import (
    label_path_for_image,
    list_images,
    load_image_rgb_float,
    read_yolo_annotations,
)


DATASET_ROOT = Path("F:/autodl-tmp/ue_project/outputs/mini_csdem/clean_dataset")
CHECKPOINT = Path("F:/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt")
DEVICE = torch.device("cuda:0")


@pytest.fixture(scope="module")
def real_yolo_context():
    if not torch.cuda.is_available():
        pytest.skip("real-YOLO adapter integration tests require CUDA")
    if not CHECKPOINT.is_file() or not DATASET_ROOT.is_dir():
        pytest.skip("local VOC surrogate checkpoint or mini dataset is unavailable")
    torch.manual_seed(0)
    adapter = YOLOFunctionalAdapter.from_checkpoint(CHECKPOINT, device=DEVICE)
    samples = []
    image_dir = DATASET_ROOT / "images" / "train"
    label_dir = DATASET_ROOT / "labels" / "train"
    for path in list_images(str(image_dir)):
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        if any(int(annotation["cls"]) == 14 for annotation in annotations):
            image, adjusted = _letterbox_with_annotations(load_image_rgb_float(path), annotations, 320)
            samples.append((path, image.to(DEVICE), adjusted))
        if len(samples) == 3:
            break
    if len(samples) < 3:
        pytest.skip("mini dataset has fewer than three target-class images")
    return adapter, samples


def _batch(image: torch.Tensor, annotations: list[dict]) -> dict[str, torch.Tensor]:
    return _batch_from_annotations(annotations, image, DEVICE)


def _poisoned_support(image: torch.Tensor, annotations: list[dict], seed: int = 0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    delta = torch.nn.Parameter(torch.randn((3, 32, 32), generator=generator, device=DEVICE) * 1e-3)
    carrier = apply_object_aligned_carrier(image[0], annotations, delta, CarrierConfig())
    return delta, carrier, _batch(carrier.poisoned.unsqueeze(0), annotations)


def test_real_yolo_forward(real_yolo_context):
    adapter, samples = real_yolo_context
    raw = adapter.forward(samples[0][1], adapter.base_parameters(), adapter.clone_buffers())
    assert raw["boxes"].shape[1] == 64
    assert raw["scores"].shape[1] == 20
    assert all(torch.isfinite(value).all() for value in (raw["boxes"], raw["scores"]))


def test_real_yolo_detection_loss(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    loss = adapter.compute_detection_loss(
        _batch(image, annotations), adapter.base_parameters(), adapter.clone_buffers()
    )
    assert loss.total.requires_grad is False
    assert all(torch.isfinite(value) and float(value) > 0 for value in (loss.box, loss.classification, loss.dfl))


def test_fast_parameter_manifest(real_yolo_context):
    adapter, _ = real_yolo_context
    classification = adapter.parameter_manifest("classification_head_only")
    detection = adapter.parameter_manifest("detection_head")
    selected = adapter.parameter_manifest("selected_neck_and_head")
    full = adapter.parameter_manifest("full_model")
    assert classification.selected_tensors == 24
    assert all(name.startswith("model.22.cv3.") for name in classification.selected_names)
    assert set(classification.selected_names) < set(detection.selected_names)
    assert all(name.startswith("model.22.") for name in detection.selected_names)
    assert set(detection.selected_names) < set(selected.selected_names) < set(full.selected_names)
    assert "model.22.dfl.conv.weight" in detection.omitted_names
    assert len(classification.selected_hash) == 64


def test_functional_update_does_not_mutate_base(real_yolo_context):
    adapter, samples = real_yolo_context
    before = adapter.hash_base_state()
    image, annotations = samples[0][1], samples[0][2]
    trajectory = adapter.virtual_update(
        _batch(image, annotations), 1, 1e-4, "classification_head_only", create_graph=False
    )
    assert trajectory.parameter_delta_norms[0] > 0
    assert adapter.hash_base_state() == before


def test_clean_poison_fast_states_are_independent(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    _, _, poison_batch = _poisoned_support(image, annotations)
    clean = adapter.virtual_update(
        _batch(image, annotations), 1, 1e-4, "classification_head_only", create_graph=False
    )
    poison = adapter.virtual_update(
        poison_batch, 1, 1e-4, "classification_head_only", create_graph=False
    )
    assert adapter.clean_poison_states_independent(clean, poison)


def test_real_yolo_j1_classification_head(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    trajectory = adapter.virtual_update(
        _batch(image, annotations), 1, 1e-4, "classification_head_only", create_graph=True
    )
    assert trajectory.steps == 1
    assert trajectory.create_graph
    assert trajectory.step_losses[0]["box"] > 0
    assert trajectory.step_losses[0]["classification"] > 0
    assert trajectory.step_losses[0]["dfl"] > 0


def test_real_yolo_mixed_gradient_protect_only(real_yolo_context):
    adapter, samples = real_yolo_context
    support_image, support_annotations = samples[0][1], samples[0][2]
    query_image, query_annotations = samples[1][1], samples[1][2]
    delta, _, poison_batch = _poisoned_support(support_image, support_annotations)
    trajectory = adapter.virtual_update(
        poison_batch, 1, 1e-4, "classification_head_only", create_graph=True
    )
    query_batch = _batch(query_image, query_annotations)
    reference = adapter.reference_assignment(query_image, query_batch)
    initial = adapter.compute_classwise_query_loss(
        query_image,
        query_batch,
        adapter.base_parameters(),
        adapter.clone_buffers(),
        reference,
    )
    updated = adapter.compute_classwise_query_loss(
        query_image, query_batch, trajectory.parameters, trajectory.buffers, reference
    )
    assert initial.valid[14] and updated.valid[14]
    protect_only = initial.losses[14] - updated.losses[14]
    gradient = torch.autograd.grad(protect_only, delta)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0


def test_real_yolo_carrier_gradient(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    delta, _, poison_batch = _poisoned_support(image, annotations, seed=1)
    parameters, buffers, _ = adapter.initial_functional_state("classification_head_only")
    carrier_loss = adapter.compute_detection_loss(poison_batch, parameters, buffers).classification
    gradient = torch.autograd.grad(carrier_loss, delta)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0


def test_fixed_query_assignment_consistency(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[1][1], samples[1][2]
    batch = _batch(image, annotations)
    reference = adapter.reference_assignment(image, batch)
    first = adapter.compute_classwise_query_loss(
        image, batch, adapter.base_parameters(), adapter.clone_buffers(), reference
    )
    second = adapter.compute_classwise_query_loss(
        image, batch, adapter.base_parameters(), adapter.clone_buffers(), reference
    )
    assert torch.equal(first.assignment.fg_mask, second.assignment.fg_mask)
    assert torch.equal(first.assignment.target_scores, second.assignment.target_scores)
    assert torch.equal(first.losses[14], second.losses[14])


def test_classwise_positive_loss(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[1][1], samples[1][2]
    batch = _batch(image, annotations)
    reference = adapter.reference_assignment(image, batch)
    result = adapter.compute_classwise_query_loss(
        image, batch, adapter.base_parameters(), adapter.clone_buffers(), reference
    )
    valid_ids = [class_id for class_id, valid in result.valid.items() if valid]
    assert 14 in valid_ids
    assert all(result.positive_count[class_id] > 0 for class_id in valid_ids)
    assert all(result.target_score_mass[class_id] > 0 for class_id in valid_ids)
    assert all(torch.isfinite(result.losses[class_id]) for class_id in valid_ids)


def test_missing_class_invalid(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[1][1], samples[1][2]
    batch = _batch(image, annotations)
    reference = adapter.reference_assignment(image, batch)
    result = adapter.compute_classwise_query_loss(
        image, batch, adapter.base_parameters(), adapter.clone_buffers(), reference
    )
    missing = next(class_id for class_id in range(20) if result.positive_count[class_id] == 0)
    assert not result.valid[missing]
    assert result.invalid_reason[missing] == "no_reference_positive"
    assert missing not in result.losses


def test_functional_buffers_do_not_mutate_base(real_yolo_context):
    adapter, samples = real_yolo_context
    before = {name: value.detach().clone() for name, value in adapter.model.named_buffers()}
    image, annotations = samples[0][1], samples[0][2]
    trajectory = adapter.virtual_update(
        _batch(image, annotations), 1, 1e-4, "detection_head", create_graph=False
    )
    assert all(torch.equal(before[name], value) for name, value in adapter.model.named_buffers())
    assert all(trajectory.buffers[name].data_ptr() != value.data_ptr() for name, value in adapter.model.named_buffers())


def test_no_proxy_fallback(real_yolo_context):
    adapter, _ = real_yolo_context
    assert adapter.backend == "real_ultralytics_yolo"
    assert adapter.functional_call_backend == "torch.func.functional_call"
    assert adapter.model.__class__.__name__ == "DetectionModel"


def test_parameter_hash_unchanged(real_yolo_context):
    adapter, samples = real_yolo_context
    before = adapter.hash_base_state()
    image, annotations = samples[0][1], samples[0][2]
    adapter.virtual_update(_batch(image, annotations), 1, 1e-4, "detection_head", create_graph=False)
    assert adapter.hash_base_state() == before


def test_real_yolo_reproducibility(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    batch = _batch(image, annotations)
    first = adapter.virtual_update(batch, 1, 1e-4, "classification_head_only", create_graph=False)
    second = adapter.virtual_update(batch, 1, 1e-4, "classification_head_only", create_graph=False)
    assert first.step_losses == second.step_losses
    assert first.parameter_delta_norms == second.parameter_delta_norms
    maximum_difference = max(
        float((first.parameters[name] - second.parameters[name]).detach().abs().max())
        for name in first.manifest.selected_names
    )
    assert maximum_difference <= 1e-7


def test_selected_neck_and_head_one_step_runnable(real_yolo_context):
    adapter, samples = real_yolo_context
    image, annotations = samples[0][1], samples[0][2]
    trajectory = adapter.virtual_update(
        _batch(image, annotations), 1, 1e-5, "selected_neck_and_head", create_graph=False
    )
    assert trajectory.parameter_delta_norms[0] > 0
