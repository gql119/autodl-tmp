import torch

from ue_framework.methods.mtepi import boxes_to_feature_mask


def test_feature_space_instance_mask_excludes_ambiguous_overlap():
    person = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    ambiguous = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    full = boxes_to_feature_mask(person, (8, 8))
    excluded = boxes_to_feature_mask(person, (8, 8), ambiguous_boxes_xywhn=ambiguous)
    assert excluded.sum().item() < full.sum().item()
    assert excluded.min().item() >= 0.0
    assert excluded.max().item() <= 1.0
