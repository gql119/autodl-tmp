import torch

from ue_framework.methods.mtepi import localized_channel_ablation


def test_localized_channel_ablation_only_masks_selected_channel_and_roi():
    features = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(4, 4)
    mask[1:3, 1:3] = 1.0
    out = localized_channel_ablation(features, [1], mask)
    assert out[:, 1, 1:3, 1:3].abs().sum().item() == 0.0
    assert out[:, 0].sum().item() == features[:, 0].sum().item()
    assert out[:, 2].sum().item() == features[:, 2].sum().item()
    assert out[:, 1, 0, 0].item() == 1.0
