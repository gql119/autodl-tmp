import torch

from ue_framework.methods.learning_trajectory import compute_class_conditioned_detection_loss

from .helpers import make_batch, make_images, make_toy_components


def test_class_conditioned_loss_isolates_protected_and_authorized_logits():
    _model, adapter, method, _config = make_toy_components(seed=3)
    images = make_images(seed=4)
    batch = make_batch()
    predictions = adapter.forward(images).detach()

    base = compute_class_conditioned_detection_loss(adapter, predictions, batch, method.router)

    person_changed = predictions.clone()
    person_changed[:, 0, 4 + 14] += 5.0
    person_loss = compute_class_conditioned_detection_loss(adapter, person_changed, batch, method.router)

    auth_changed = predictions.clone()
    auth_changed[:, 1, 4 + 1] += 5.0
    auth_loss = compute_class_conditioned_detection_loss(adapter, auth_changed, batch, method.router)

    protected_delta = (person_loss["protected_total_loss"] - base["protected_total_loss"]).abs()
    authorized_delta = (person_loss["authorized_total_loss"] - base["authorized_total_loss"]).abs()
    assert protected_delta > authorized_delta * 10

    authorized_delta_2 = (auth_loss["authorized_total_loss"] - base["authorized_total_loss"]).abs()
    protected_delta_2 = (auth_loss["protected_total_loss"] - base["protected_total_loss"]).abs()
    assert authorized_delta_2 > protected_delta_2 * 10

    cross_changed = predictions.clone()
    cross_changed[:, 1, 4 + 14] += 5.0
    cross_loss = compute_class_conditioned_detection_loss(adapter, cross_changed, batch, method.router)
    cross_authorized_delta = (cross_loss["authorized_total_loss"] - base["authorized_total_loss"]).abs()
    assert cross_authorized_delta == 0.0
