import torch

from ue_framework.methods.multitrajectory_gain import BatchData
from ue_framework.methods.multitrajectory_gain.online_sampler import OnlineTrajectorySampler


def _batch(image_id):
    return BatchData(torch.zeros(1, 3, 2, 2), {"batch_size": 1}, [image_id])


def test_online_sampler_produces_new_trajectory_each_step():
    train = [_batch(f"train_{i}") for i in range(20)]
    heldout = [_batch(f"held_{i}") for i in range(4)]
    sampler = OnlineTrajectorySampler(train, heldout, support_steps=3, seed=4, recent_image_exclusion_window=8)
    first = sampler.sample(0)
    second = sampler.sample(1)
    assert first.sequence is not None
    assert second.sequence is not None
    first_ids = set(sum(first.sequence.support_image_ids, []) + first.sequence.query_image_ids)
    second_ids = set(sum(second.sequence.support_image_ids, []) + second.sequence.query_image_ids)
    assert first_ids != second_ids
    assert not (set(first.sequence.query_image_ids) & set(sum(first.sequence.support_image_ids, [])))
