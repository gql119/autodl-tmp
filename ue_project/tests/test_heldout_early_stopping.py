import torch

from ue_framework.methods.multitrajectory_gain.early_stopping import HeldoutEarlyStopping


def test_heldout_early_stopping_triggers_after_patience():
    stopper = HeldoutEarlyStopping(patience=2)
    delta = torch.tensor([1.0])
    assert stopper.update(0, 1.0, delta).improved
    assert not stopper.update(1, 0.5, delta).should_stop
    assert stopper.update(2, 0.4, delta).should_stop


def test_heldout_update_does_not_require_backward():
    stopper = HeldoutEarlyStopping(patience=1)
    delta = torch.tensor([2.0], requires_grad=True)
    stopper.update(0, 1.0, delta)
    assert stopper.best_delta is not None
    assert not stopper.best_delta.requires_grad
