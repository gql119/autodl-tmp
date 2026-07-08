from __future__ import annotations

import torch

from .rollout_engine import J3RolloutEngine, RolloutOutput
from .trajectory_state import TrajectoryBatchSequence


class J3LearningGainMethod:
    def __init__(self, engine: J3RolloutEngine) -> None:
        self.engine = engine

    def compute_loss(self, sequence: TrajectoryBatchSequence, delta: torch.Tensor) -> RolloutOutput:
        return self.engine.run(sequence, delta, create_graph=True)
