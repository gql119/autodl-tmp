from __future__ import annotations

import random
from typing import List, Sequence

from .trajectory_state import BatchData, TrajectoryBatchSequence


class TrajectorySampler:
    def __init__(
        self,
        batches: Sequence[BatchData],
        support_steps: int = 3,
        seed: int = 0,
        exclude_support_query_overlap: bool = True,
    ) -> None:
        self.batches = list(batches)
        self.support_steps = int(support_steps)
        self.seed = int(seed)
        self.exclude_support_query_overlap = bool(exclude_support_query_overlap)

    def sample(self, index: int) -> TrajectoryBatchSequence:
        rng = random.Random(self.seed + int(index))
        if len(self.batches) < self.support_steps + 1:
            raise ValueError("Not enough batches to sample support and query.")
        candidates = list(range(len(self.batches)))
        rng.shuffle(candidates)
        support_ids: List[int] = []
        used_images = set()
        for candidate in candidates:
            image_ids = set(self.batches[candidate].image_ids)
            if image_ids & used_images:
                continue
            support_ids.append(candidate)
            used_images.update(image_ids)
            if len(support_ids) == self.support_steps:
                break
        if len(support_ids) != self.support_steps:
            raise ValueError("Could not sample distinct support batches.")
        query_id = None
        for candidate in candidates:
            if candidate in support_ids:
                continue
            if self.exclude_support_query_overlap and set(self.batches[candidate].image_ids) & used_images:
                continue
            query_id = candidate
            break
        if query_id is None:
            raise ValueError("Could not sample independent query batch.")
        support = [self.batches[i] for i in support_ids]
        query = self.batches[query_id]
        return TrajectoryBatchSequence(
            support_batches=support,
            query_batch=query,
            support_image_ids=[b.image_ids for b in support],
            query_image_ids=query.image_ids,
            augmentation_seed=self.seed + int(index),
            batch_order_seed=self.seed + int(index),
        )
