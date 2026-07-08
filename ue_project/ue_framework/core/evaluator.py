from __future__ import annotations

from typing import Dict


def protected_authorized_metric_view(metrics: Dict[str, float], protected_key: str = "mAP50_target") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if protected_key in metrics:
        out["mAP50_protected"] = float(metrics[protected_key])
    if "mAP50_non_target" in metrics:
        out["mAP50_authorized"] = float(metrics["mAP50_non_target"])
    return out
