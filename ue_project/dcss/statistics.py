from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RunningCovariance:
    """Streaming mean, centered covariance, and uncentered second moment."""

    dimension: int
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.count = 0
        self.mean = torch.zeros(self.dimension, dtype=self.dtype)
        self.m2 = torch.zeros((self.dimension, self.dimension), dtype=self.dtype)
        self.second_sum = torch.zeros_like(self.m2)

    def update(self, values: torch.Tensor) -> None:
        x = values.detach().to(device="cpu", dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"expected [N,{self.dimension}], got {tuple(x.shape)}")
        if x.shape[0] == 0:
            return
        if not torch.isfinite(x).all():
            raise FloatingPointError("non-finite sample in running covariance")

        batch_n = int(x.shape[0])
        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered
        self.second_sum += x.T @ x

        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.count = batch_n
            return

        total = self.count + batch_n
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (self.count * batch_n / total)
        self.mean += delta * (batch_n / total)
        self.count = total

    def covariance(self, unbiased: bool = False) -> torch.Tensor:
        denominator = self.count - 1 if unbiased else self.count
        if denominator <= 0:
            return torch.zeros_like(self.m2)
        return self.m2 / denominator

    def second_moment(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.second_sum)
        return self.second_sum / self.count

    def state_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "second_sum": self.second_sum,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "RunningCovariance":
        result = cls(int(state["dimension"]), dtype=state["mean"].dtype)
        result.count = int(state["count"])
        result.mean.copy_(state["mean"])
        result.m2.copy_(state["m2"])
        result.second_sum.copy_(state["second_sum"])
        return result
