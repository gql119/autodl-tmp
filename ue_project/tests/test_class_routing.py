import torch

from ue_framework.core import ClassConditionedRouter
from ue_framework.core.assignment_parser import AssignmentResult


def test_class_routing_protected_authorized_ambiguous_counts():
    assignment = AssignmentResult(
        fg_mask=torch.tensor([[True, True, True, False]]),
        target_gt_idx=torch.tensor([[0, 1, 2, -1]]),
        target_labels=torch.tensor([[14, 3, 14, -1]]),
        target_scores=torch.zeros(1, 4, 20),
        assignment_counts=torch.tensor([[1, 1, 2, 0]]),
        level_ids=torch.tensor([[3, 3, 4, 5]]),
    )
    router = ClassConditionedRouter(protected_class_id=14, num_classes=20, exclude_ambiguous=True)
    result = router.route(assignment)

    assert result.protected_mask.tolist() == [[True, False, False, False]]
    assert result.authorized_mask.tolist() == [[False, True, False, False]]
    assert result.ambiguous_mask.tolist() == [[False, False, True, False]]
    assert result.stats["protected_positive_count"] == 1.0
    assert result.stats["authorized_positive_count"] == 1.0
    assert result.stats["ambiguous_positive_count"] == 1.0
    assert result.stats_by_level["P3"]["authorized_positive_count"] == 1.0
