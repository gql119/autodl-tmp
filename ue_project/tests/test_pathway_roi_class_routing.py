import torch

from ue_framework.methods.mtepi import boxes_to_feature_mask


def test_pathway_roi_class_routing_keeps_person_and_authorized_masks_separate():
    person_box = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    authorized_box = torch.tensor([[0.75, 0.75, 0.25, 0.25]])
    person_mask = boxes_to_feature_mask(person_box, (4, 4), exclude_boxes_xywhn=authorized_box)
    authorized_mask = boxes_to_feature_mask(authorized_box, (4, 4), exclude_boxes_xywhn=person_box)
    assert (person_mask * authorized_mask).sum().item() == 0.0
    assert person_mask.sum().item() > 0.0
    assert authorized_mask.sum().item() > 0.0
