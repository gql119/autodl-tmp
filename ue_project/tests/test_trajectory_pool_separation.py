import pytest
import torch

from ue_framework.methods.multitrajectory_gain import BatchData
from ue_framework.methods.multitrajectory_gain.online_sampler import OnlineTrajectorySampler


def _batch(image_id):
    return BatchData(torch.zeros(1, 3, 2, 2), {"batch_size": 1}, [image_id])


def test_train_and_heldout_pools_must_not_overlap():
    train = [_batch("a"), _batch("b"), _batch("c"), _batch("d")]
    heldout = [_batch("d")]
    with pytest.raises(ValueError):
        OnlineTrajectorySampler(train, heldout)


def test_train_and_heldout_pools_are_separate_when_ids_differ():
    sampler = OnlineTrajectorySampler([_batch(str(i)) for i in range(5)], [_batch("held")], support_steps=3)
    assert sampler.sample(0).sequence is not None
