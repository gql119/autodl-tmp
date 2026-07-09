import torch

from ue_framework.methods.mtepi import channel_ablation_hook


def test_channel_ablation_hook_is_temporary_and_keeps_parameters():
    conv = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
    conv.weight.data.fill_(1.0)
    before = conv.weight.detach().clone()
    x = torch.ones(1, 2, 2, 2)
    mask = torch.ones(2, 2)
    with channel_ablation_hook(conv, [0], mask):
        hooked = conv(x)
    restored = conv(x)
    assert torch.equal(conv.weight.detach(), before)
    assert hooked[:, 0].abs().max().item() == 0.0
    assert restored[:, 0].abs().max().item() > 0.0
