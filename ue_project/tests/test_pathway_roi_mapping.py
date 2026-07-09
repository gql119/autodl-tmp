import torch

from ue_framework.methods.mtepi import boxes_to_feature_mask


def test_pathway_roi_mapping_uses_normalized_cell_centers():
    box = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    mask = boxes_to_feature_mask(box, (4, 4))
    assert mask.sum().item() == 4.0
    assert mask[1, 1].item() == 1.0
    assert mask[2, 2].item() == 1.0
    assert mask[0, 0].item() == 0.0
