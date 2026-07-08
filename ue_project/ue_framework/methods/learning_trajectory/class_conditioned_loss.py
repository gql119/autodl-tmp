from __future__ import annotations

from typing import Dict, Optional

import torch

from ue_framework.core import ClassConditionedRouter, ClassRoutingResult, DetectorAdapter
from ue_framework.core.assignment_parser import AssignmentResult


def compute_class_conditioned_detection_loss(
    adapter: DetectorAdapter,
    predictions: torch.Tensor,
    batch: Dict,
    router: ClassConditionedRouter,
    assignment: Optional[AssignmentResult] = None,
    routing: Optional[ClassRoutingResult] = None,
    include_background_negatives: bool = False,
) -> Dict[str, torch.Tensor]:
    if assignment is None:
        assignment = adapter.get_task_aligned_assignments(predictions, batch)
    if routing is None:
        routing = router.route(assignment)

    protected = adapter.compute_masked_detection_loss(
        predictions=predictions,
        batch=batch,
        assignment=assignment,
        unit_mask=routing.protected_mask,
        include_background_negatives=include_background_negatives,
        background_class_filter=[router.protected_class_id],
    )
    authorized = adapter.compute_masked_detection_loss(
        predictions=predictions,
        batch=batch,
        assignment=assignment,
        unit_mask=routing.authorized_mask,
        include_background_negatives=include_background_negatives,
        background_class_filter=router.authorized_class_ids,
    )

    out: Dict[str, torch.Tensor] = {
        "protected_total_loss": protected["total_loss"],
        "protected_cls_loss": protected["cls_loss"],
        "protected_box_loss": protected["box_loss"],
        "protected_dfl_loss": protected["dfl_loss"],
        "authorized_total_loss": authorized["total_loss"],
        "authorized_cls_loss": authorized["cls_loss"],
        "authorized_box_loss": authorized["box_loss"],
        "authorized_dfl_loss": authorized["dfl_loss"],
    }
    for key, value in routing.stats.items():
        out[key] = torch.as_tensor(value, device=predictions.device if torch.is_tensor(predictions) else None)
    return out
