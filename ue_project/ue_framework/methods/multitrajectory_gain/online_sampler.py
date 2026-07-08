from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .trajectory_state import BatchData, TrajectoryBatchSequence


@dataclass
class OnlineTrajectorySample:
    sequence: TrajectoryBatchSequence | None
    rejection_reason: str | None
    attempts: int


class OnlineTrajectorySampler:
    def __init__(
        self,
        train_batches: Sequence[BatchData],
        heldout_batches: Sequence[BatchData],
        support_steps: int = 3,
        seed: int = 0,
        recent_image_exclusion_window: int = 20,
        max_attempts: int = 100,
    ) -> None:
        self.train_batches = list(train_batches)
        self.heldout_batches = list(heldout_batches)
        self.support_steps = int(support_steps)
        self.seed = int(seed)
        self.max_attempts = int(max_attempts)
        self.recent_images: deque[str] = deque(maxlen=int(recent_image_exclusion_window))
        self.heldout_images = _image_id_set(self.heldout_batches)
        if _image_id_set(self.train_batches) & self.heldout_images:
            raise ValueError("train and held-out image pools overlap")

    def sample(self, outer_step: int) -> OnlineTrajectorySample:
        rng = random.Random(self.seed + int(outer_step))
        for attempt in range(1, self.max_attempts + 1):
            candidates = list(range(len(self.train_batches)))
            rng.shuffle(candidates)
            used: Set[str] = set()
            support_indices: List[int] = []
            reason = "not_enough_distinct_support"
            for idx in candidates:
                image_ids = set(self.train_batches[idx].image_ids)
                if image_ids & used:
                    reason = "support_query_overlap"
                    continue
                if image_ids & set(self.recent_images):
                    reason = "recent_image_reuse"
                    continue
                support_indices.append(idx)
                used.update(image_ids)
                if len(support_indices) == self.support_steps:
                    break
            if len(support_indices) != self.support_steps:
                continue
            query_idx = None
            for idx in candidates:
                if idx in support_indices:
                    continue
                image_ids = set(self.train_batches[idx].image_ids)
                if image_ids & used:
                    reason = "support_query_overlap"
                    continue
                if image_ids & set(self.recent_images):
                    reason = "recent_image_reuse"
                    continue
                query_idx = idx
                break
            if query_idx is None:
                continue
            support = [self.train_batches[i] for i in support_indices]
            query = self.train_batches[query_idx]
            sequence = TrajectoryBatchSequence(
                support_batches=support,
                query_batch=query,
                support_image_ids=[b.image_ids for b in support],
                query_image_ids=query.image_ids,
                augmentation_seed=self.seed + int(outer_step),
                batch_order_seed=self.seed + int(outer_step),
            )
            for image_id in _sequence_image_ids(sequence):
                self.recent_images.append(image_id)
            return OnlineTrajectorySample(sequence, None, attempt)
        return OnlineTrajectorySample(None, reason, self.max_attempts)


def _image_id_set(batches: Iterable[BatchData]) -> Set[str]:
    ids: Set[str] = set()
    for batch in batches:
        ids.update(batch.image_ids)
    return ids


def _sequence_image_ids(sequence: TrajectoryBatchSequence) -> List[str]:
    ids: List[str] = []
    for batch_ids in sequence.support_image_ids:
        ids.extend(batch_ids)
    ids.extend(sequence.query_image_ids)
    return ids
