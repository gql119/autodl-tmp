"""Baseline namespace for legacy methods.

The current stable TAUSB/ALCE implementation remains in its original modules so
the recorded best configuration can still run unchanged. New trajectory methods
must not import this package.
"""

LEGACY_BASELINES = {
    "legacy_best": "ue_framework.methods.tausb_universal",
    "alce_legacy": "ue_framework.methods.tausb_universal",
    "adv_det": "ue_framework.methods.em",
    "cs_em_det": "ue_framework.methods.em",
    "assignment_disruption": "ue_framework.methods.shadow_tal",
}

__all__ = ["LEGACY_BASELINES"]
