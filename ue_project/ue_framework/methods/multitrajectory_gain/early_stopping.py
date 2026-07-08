from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EarlyStoppingResult:
    improved: bool
    should_stop: bool
    best_step: int
    best_score: float


class HeldoutEarlyStopping:
    def __init__(self, patience: int = 3) -> None:
        self.patience = int(patience)
        self.best_score = float("-inf")
        self.best_step = -1
        self.bad_evals = 0
        self.best_delta: Optional[torch.Tensor] = None

    def update(self, step: int, score: float, delta: torch.Tensor) -> EarlyStoppingResult:
        improved = float(score) > self.best_score
        if improved:
            self.best_score = float(score)
            self.best_step = int(step)
            self.bad_evals = 0
            self.best_delta = delta.detach().clone()
        else:
            self.bad_evals += 1
        return EarlyStoppingResult(improved, self.bad_evals >= self.patience, self.best_step, self.best_score)

    def restore_best(self, delta: torch.Tensor) -> None:
        if self.best_delta is None:
            return
        with torch.no_grad():
            delta.copy_(self.best_delta.to(device=delta.device, dtype=delta.dtype))
