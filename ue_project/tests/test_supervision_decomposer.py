import torch

from ue_framework.core.supervision_decomposer import SupervisionDecomposer


def make_decomposition(pred_scores=None, ambiguous_mask=None, per_box=None, per_dfl=None):
    if pred_scores is None:
        pred_scores = torch.zeros((1, 3, 20), dtype=torch.float32)
    target_scores = torch.zeros_like(pred_scores)
    target_scores[0, 0, 14] = 0.7
    target_scores[0, 1, 1] = 0.6
    target_labels = torch.tensor([[14, 1, 0]], dtype=torch.long)
    fg_mask = torch.tensor([[True, True, False]])
    if ambiguous_mask is None:
        ambiguous_mask = torch.zeros_like(fg_mask)
    if per_box is None:
        per_box = torch.tensor([[0.2, 0.3, 0.0]], dtype=torch.float32)
    if per_dfl is None:
        per_dfl = torch.tensor([[0.4, 0.5, 0.0]], dtype=torch.float32)
    return SupervisionDecomposer(protected_class_id=14, num_classes=20).decompose_from_tensors(
        pred_scores=pred_scores,
        target_scores=target_scores,
        target_labels=target_labels,
        fg_mask=fg_mask,
        ambiguous_mask=ambiguous_mask,
        per_unit_box_loss=per_box,
        per_unit_dfl_loss=per_dfl,
        target_scores_sum=target_scores.sum().clamp_min(1.0),
        batch_size=1,
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
    )


def test_supervision_decomposer_routes_counts():
    dec = make_decomposition()
    assert dec.statistics["protected_positive_count"] == 1.0
    assert dec.statistics["authorized_positive_count"] == 1.0
    assert dec.statistics["background_count"] == 1.0
    assert dec.protected_cls.item() > 0.0
    assert dec.authorized_cls.item() > 0.0
    assert dec.shared_cls.item() > 0.0


def test_supervision_decomposer_reconstructs_total():
    dec = make_decomposition()
    assert dec.reconstruction_error.item() < 1.0e-5
    assert dec.statistics["relative_reconstruction_error"] < 1.0e-5
    assert dec.cls_reconstruction_error.item() < 1.0e-6
    assert dec.box_reconstruction_error.item() < 1.0e-6
    assert dec.dfl_reconstruction_error.item() < 1.0e-6


def test_supervision_decomposer_routes_ambiguous_to_shared():
    ambiguous = torch.tensor([[True, False, False]])
    dec = make_decomposition(ambiguous_mask=ambiguous)
    assert dec.statistics["protected_positive_count"] == 0.0
    assert dec.statistics["authorized_positive_count"] == 1.0
    assert dec.statistics["shared_positive_count"] == 1.0
    assert dec.protected_box.item() == 0.0
    assert dec.shared_box.item() > 0.0
