from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from .assignment_parser import AssignmentResult


@dataclass
class ClassRoutingResult:
    protected_mask: torch.Tensor
    authorized_mask: torch.Tensor
    ambiguous_mask: torch.Tensor
    protected_indices: torch.Tensor
    authorized_indices: torch.Tensor
    stats: Dict[str, float]
    stats_by_level: Dict[str, Dict[str, float]]
    assignment: AssignmentResult


class ClassConditionedRouter:
    def __init__(
        self,
        protected_class_id: int,
        authorized_class_ids: Optional[Iterable[int]] = None,
        num_classes: Optional[int] = None,
        exclude_ambiguous: bool = True,
    ) -> None:
        self.protected_class_id = int(protected_class_id)
        self.num_classes = None if num_classes is None else int(num_classes)
        if authorized_class_ids is None or authorized_class_ids == "auto":
            if self.num_classes is None:
                self.authorized_class_ids: Optional[List[int]] = None
            else:
                self.authorized_class_ids = [i for i in range(self.num_classes) if i != self.protected_class_id]
        else:
            self.authorized_class_ids = [int(v) for v in authorized_class_ids]
        self.exclude_ambiguous = bool(exclude_ambiguous)

    def route(self, assignment: AssignmentResult) -> ClassRoutingResult:
        assignment.validate()
        fg_mask = assignment.fg_mask.bool()
        labels = assignment.target_labels.long()
        ambiguous_mask = fg_mask & (assignment.assignment_counts.long() > 1)

        valid_for_loss = fg_mask & ~ambiguous_mask if self.exclude_ambiguous else fg_mask
        protected_mask = valid_for_loss & (labels == self.protected_class_id)

        if self.authorized_class_ids is None:
            authorized_mask = valid_for_loss & (labels != self.protected_class_id)
        else:
            authorized_ids = torch.tensor(self.authorized_class_ids, device=labels.device, dtype=labels.dtype)
            authorized_mask = valid_for_loss & torch.isin(labels, authorized_ids)

        protected_indices = torch.nonzero(protected_mask, as_tuple=False)
        authorized_indices = torch.nonzero(authorized_mask, as_tuple=False)
        stats = self._stats(protected_mask, authorized_mask, ambiguous_mask, fg_mask)
        stats_by_level = self._stats_by_level(protected_mask, authorized_mask, ambiguous_mask, fg_mask, assignment.level_ids)

        return ClassRoutingResult(
            protected_mask=protected_mask,
            authorized_mask=authorized_mask,
            ambiguous_mask=ambiguous_mask,
            protected_indices=protected_indices,
            authorized_indices=authorized_indices,
            stats=stats,
            stats_by_level=stats_by_level,
            assignment=assignment,
        )

    @staticmethod
    def _stats(
        protected_mask: torch.Tensor,
        authorized_mask: torch.Tensor,
        ambiguous_mask: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Dict[str, float]:
        fg_count = float(fg_mask.sum().item())
        protected_count = float(protected_mask.sum().item())
        authorized_count = float(authorized_mask.sum().item())
        ambiguous_count = float(ambiguous_mask.sum().item())
        denom = max(fg_count, 1.0)
        return {
            "protected_positive_count": protected_count,
            "authorized_positive_count": authorized_count,
            "ambiguous_positive_count": ambiguous_count,
            "protected_positive_ratio": protected_count / denom,
            "authorized_positive_ratio": authorized_count / denom,
        }

    @staticmethod
    def _stats_by_level(
        protected_mask: torch.Tensor,
        authorized_mask: torch.Tensor,
        ambiguous_mask: torch.Tensor,
        fg_mask: torch.Tensor,
        level_ids: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        if level_ids is None:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for level in torch.unique(level_ids).detach().cpu().tolist():
            level_mask = level_ids == int(level)
            out[f"P{int(level)}"] = ClassConditionedRouter._stats(
                protected_mask & level_mask,
                authorized_mask & level_mask,
                ambiguous_mask & level_mask,
                fg_mask & level_mask,
            )
        return out
