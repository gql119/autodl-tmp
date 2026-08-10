from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from ..metrics_utils import VOC20_CLASS_NAMES


def _strict_named_ap50(metrics: Mapping[str, object], name: str) -> Dict[str, float]:
    values = metrics.get("ap50_by_class")
    if not isinstance(values, Mapping) or set(values) != set(VOC20_CLASS_NAMES):
        raise ValueError("%s must contain all named VOC20 AP50 values." % name)
    output = {class_name: float(values[class_name]) for class_name in VOC20_CLASS_NAMES}
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in output.values()):
        raise ValueError("%s contains a non-finite or out-of-range AP50." % name)
    return output


def build_sdh_counterfactual_metrics(
    clean_val: Mapping[str, object],
    target_carrier_val: Mapping[str, object],
) -> Dict[str, object]:
    clean = _strict_named_ap50(clean_val, "clean_val")
    carrier = _strict_named_ap50(target_carrier_val, "target_carrier_val")
    per_class = {
        name: carrier[name] - clean[name] for name in VOC20_CLASS_NAMES
    }
    non_target = [name for name in VOC20_CLASS_NAMES if name != "person"]
    person_recovery = per_class["person"]
    return {
        "schema": "tausb.sdh-counterfactual.v1",
        "primary_split": "clean_val",
        "counterfactual_split": "target_carrier_val_gt_box_read_only",
        "person_ap50_clean_val": clean["person"],
        "person_ap50_target_carrier_val": carrier["person"],
        "person_ap50_recovery": person_recovery,
        "non_target_macro_shift": float(np.mean([per_class[name] for name in non_target])),
        "per_class_shift_carrier_minus_clean": per_class,
        "shortcut_recovery_ge_0_20": bool(person_recovery >= 0.20),
        "shortcut_failure_lt_0_10": bool(person_recovery < 0.10),
        "claim_boundary": (
            "counterfactual mechanism evidence only; clean_val remains the UE primary metric"
        ),
    }

def learning_preference_ratio(
    *, clean_counterfactual_loss: float, carrier_loss: float
) -> float:
    clean = float(clean_counterfactual_loss)
    carrier = float(carrier_loss)
    if not np.isfinite(clean) or not np.isfinite(carrier):
        raise ValueError("Learning-preference losses must be finite.")
    return (clean - carrier) / (abs(clean) + 1.0e-8)


def build_learning_preference_audit(
    epoch_losses: Mapping[int, Mapping[str, float]],
) -> Dict[str, object]:
    required = (1, 5, 10, 20)
    if set(int(value) for value in epoch_losses) != set(required):
        raise ValueError("Learning-preference audit requires epochs 1/5/10/20 exactly.")
    rows = []
    for epoch in required:
        values = epoch_losses[epoch]
        ratio = learning_preference_ratio(
            clean_counterfactual_loss=float(values["clean_counterfactual_loss"]),
            carrier_loss=float(values["carrier_loss"]),
        )
        rows.append(
            {
                "epoch": epoch,
                "clean_counterfactual_loss": float(values["clean_counterfactual_loss"]),
                "carrier_loss": float(values["carrier_loss"]),
                "R_e": ratio,
            }
        )
    by_epoch = {row["epoch"]: row["R_e"] for row in rows}
    return {
        "schema": "tausb.sdh-learning-preference.v1",
        "rows": rows,
        "R_10_positive": bool(by_epoch[10] > 0),
        "R_20_ge_0_10": bool(by_epoch[20] >= 0.10),
        "failure_R10_R20": bool(by_epoch[10] <= 0 and by_epoch[20] < 0.05),
        "read_only": True,
        "used_for_checkpoint_selection": False,
    }


def build_sdh_fresh_victim_comparison(
    clean: Mapping[str, object],
    poisoned: Mapping[str, object],
    counterfactual: Mapping[str, object],
    dynamics: Mapping[str, object],
) -> Dict[str, object]:
    clean_ap = _strict_named_ap50(clean, "C0")
    poison_ap = _strict_named_ap50(poisoned, "P1-V")
    per_class = []
    for class_id, name in enumerate(VOC20_CLASS_NAMES):
        drop = clean_ap[name] - poison_ap[name]
        if clean_ap[name] <= 0:
            raise ValueError("C0 AP50 is zero; retention is undefined for %s." % name)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": name,
                "C0_AP50": clean_ap[name],
                "P1V_AP50": poison_ap[name],
                "drop_C0_minus_P1V": drop,
                "retention_P1V_over_C0": poison_ap[name] / clean_ap[name],
                "is_target": name == "person",
            }
        )
    non_target = [row for row in per_class if not row["is_target"]]
    person = next(row for row in per_class if row["is_target"])
    macro_drop = float(np.mean([row["drop_C0_minus_P1V"] for row in non_target]))
    recovery = float(counterfactual["person_ap50_recovery"])
    dynamics_rows = {int(row["epoch"]): float(row["R_e"]) for row in dynamics["rows"]}
    checks = {
        "person_drop": person["drop_C0_minus_P1V"] >= 0.30,
        "carrier_recovery": recovery >= 0.20,
        "learning_preference": dynamics_rows[10] > 0 and dynamics_rows[20] >= 0.10,
        "non_target_macro": macro_drop <= 0.05,
        "non_target_count": sum(row["drop_C0_minus_P1V"] <= 0.10 for row in non_target) >= 16,
    }
    return {
        "schema": "tausb.sdh-fresh-victim-comparison.v1",
        "claim_status": "tentative_single_seed",
        "per_class": per_class,
        "summary": {
            "person_drop": person["drop_C0_minus_P1V"],
            "person_carrier_recovery": recovery,
            "non_target_macro_drop": macro_drop,
            "non_target_classes_drop_le_0_10": sum(
                row["drop_C0_minus_P1V"] <= 0.10 for row in non_target
            ),
            "checks": checks,
            "pass": all(checks.values()),
        },
    }
