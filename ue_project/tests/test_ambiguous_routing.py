import torch

from ue_framework.core.supervision_decomposer import SupervisionDecomposer


def test_low_target_score_positive_routes_to_shared():
    pred_scores = torch.zeros((1, 2, 20))
    target_scores = torch.zeros_like(pred_scores)
    target_scores[0, 0, 14] = 0.01
    labels = torch.tensor([[14, 1]])
    fg = torch.tensor([[True, False]])
    dec = SupervisionDecomposer(
        protected_class_id=14,
        num_classes=20,
        target_score_reliability_threshold=0.02,
    ).decompose_from_tensors(
        pred_scores=pred_scores,
        target_scores=target_scores,
        target_labels=labels,
        fg_mask=fg,
        ambiguous_mask=torch.tensor([[True, False]]),
        target_scores_sum=target_scores.sum().clamp_min(1.0),
    )
    assert dec.statistics["ambiguous_positive_count"] == 1.0
    assert dec.statistics["protected_positive_count"] == 0.0
    assert dec.statistics["shared_positive_count"] == 1.0
