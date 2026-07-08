import torch

from ue_framework.methods.multitrajectory_gain import BatchData, TrajectorySampler


def _batch(idx):
    return BatchData(
        images=torch.zeros(1, 3, 4, 4),
        batch={"cls": torch.tensor([14.0]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "batch_idx": torch.tensor([0.0]), "batch_size": 1},
        image_ids=[f"{idx:06d}"],
        augmentation_seed=idx,
    )


def test_matched_trajectory_sequence_has_distinct_support_and_query():
    sampler = TrajectorySampler([_batch(i) for i in range(8)], support_steps=3, seed=10)
    seq = sampler.sample(0)
    support_ids = {item for group in seq.support_image_ids for item in group}
    query_ids = set(seq.query_image_ids)
    assert len(seq.support_batches) == 3
    assert not (support_ids & query_ids)
    assert seq.augmentation_seed == seq.batch_order_seed
