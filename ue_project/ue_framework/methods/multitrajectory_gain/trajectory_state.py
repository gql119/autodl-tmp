from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class FunctionalOptimizerState:
    momentum_buffers: Dict[str, torch.Tensor]

    def clone(self) -> "FunctionalOptimizerState":
        return FunctionalOptimizerState({name: value.clone() for name, value in self.momentum_buffers.items()})


@dataclass
class RolloutState:
    parameters: Dict[str, torch.Tensor]
    optimizer_state: FunctionalOptimizerState
    step: int


@dataclass
class BatchData:
    images: torch.Tensor
    batch: dict
    image_ids: List[str]
    augmentation_seed: int = 0


@dataclass
class TrajectoryBatchSequence:
    support_batches: List[BatchData]
    query_batch: BatchData
    support_image_ids: List[List[str]]
    query_image_ids: List[str]
    augmentation_seed: int
    batch_order_seed: int = 0
