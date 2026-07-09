import torch

from ue_framework.methods.mtepi import boxes_to_feature_mask


def test_pathway_overlap_exclusion_removes_authorized_core_from_person_mask():
    person = torch.tensor([[0.5, 0.5, 0.75, 0.75]])
    authorized = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    mask = boxes_to_feature_mask(person, (8, 8), exclude_boxes_xywhn=authorized)
    authorized_mask = boxes_to_feature_mask(authorized, (8, 8))
    assert (mask * authorized_mask).sum().item() == 0.0
