import torch

from ue_framework.methods.multitrajectory_gain.early_stopping import HeldoutEarlyStopping


def test_best_delta_restore_restores_best_snapshot():
    stopper = HeldoutEarlyStopping(patience=3)
    delta = torch.tensor([1.0, 2.0])
    stopper.update(0, 1.0, delta)
    delta.add_(10.0)
    stopper.update(1, 0.5, delta)
    stopper.restore_best(delta)
    assert torch.allclose(delta, torch.tensor([1.0, 2.0]))
