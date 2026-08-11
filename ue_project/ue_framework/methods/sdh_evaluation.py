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


def build_sdh_e2e_v0_comparison(
    clean: Mapping[str, object],
    poisoned: Mapping[str, object],
) -> Dict[str, object]:
    clean_ap = _strict_named_ap50(clean, "C0")
    poison_ap = _strict_named_ap50(poisoned, "M1")
    required_identity = {
        "protocol_id": "TAUSB-SDH-E2E-V0-MAP50-v1",
        "pilot_kind": "e20",
        "seed": 0,
        "steps": 40,
        "victim_epochs": 20,
        "evidence_scope": "end_to_end_feasibility_not_formal_method",
        "hiding_gate_passed": False,
    }
    for key, expected in required_identity.items():
        if clean.get(key) != expected or poisoned.get(key) != expected:
            raise ValueError("C0/M1 %s does not match the E2E V0 protocol." % key)
    if clean.get("arm_id") != "C0" or poisoned.get("arm_id") != "M1":
        raise ValueError("E2E V0 comparison requires explicit C0 and M1 arm identity.")
    if clean.get("mechanism_gate_passed") != poisoned.get("mechanism_gate_passed"):
        raise ValueError("C0/M1 mechanism gate provenance differs.")
    for key in (
        "clean_val_manifest_sha256",
        "paired_training_protocol_sha256",
        "frozen_sdh_state_sha256",
        "hiding_metrics_sha256",
        "hiding_checkpoint_sha256",
        "hiding_split_sha256",
        "mechanism_metrics_sha256",
        "mechanism_decision_sha256",
        "mechanism_config_sha256",
        "p1_state_sha256",
    ):
        clean_value = str(clean.get(key, ""))
        poison_value = str(poisoned.get(key, ""))
        if len(clean_value) != 64 or clean_value != poison_value:
            raise ValueError("C0/M1 %s is missing or mismatched." % key)
    if int(clean.get("poisoned_count", -1)) != 0:
        raise ValueError("E2E V0 C0 poisoned_count must be zero.")
    if int(poisoned.get("poisoned_count", -1)) != 6095:
        raise ValueError("E2E V0 M1 poisoned_count must be 6095.")
    linf = float(poisoned.get("actual_linf_max", float("nan")))
    if not np.isfinite(linf) or linf > 16.0 / 255.0 + 1.0 / 255.0:
        raise ValueError("E2E V0 M1 Linf is invalid or exceeds tolerance.")
    per_class = []
    for class_id, class_name in enumerate(VOC20_CLASS_NAMES):
        if clean_ap[class_name] <= 0:
            raise ValueError("C0 AP50 is zero; retention is undefined for %s." % class_name)
        drop = clean_ap[class_name] - poison_ap[class_name]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "C0_AP50": clean_ap[class_name],
                "M1_AP50": poison_ap[class_name],
                "drop_C0_minus_M1": drop,
                "retention_M1_over_C0": poison_ap[class_name] / clean_ap[class_name],
                "is_target": class_name == "person",
            }
        )
    person = next(row for row in per_class if row["is_target"])
    non_target = [row for row in per_class if not row["is_target"]]
    person_drop = float(person["drop_C0_minus_M1"])
    non_target_macro_drop = float(
        np.mean([row["drop_C0_minus_M1"] for row in non_target])
    )
    non_target_le_015 = sum(
        row["drop_C0_minus_M1"] <= 0.15 for row in non_target
    )
    non_target_gt_020 = sum(
        row["drop_C0_minus_M1"] > 0.20 for row in non_target
    )
    success_checks = {
        "person_drop_ge_0_10": person_drop >= 0.10,
        "non_target_macro_drop_le_0_08": non_target_macro_drop <= 0.08,
        "non_target_count_drop_le_0_15_ge_15": non_target_le_015 >= 15,
        "poisoned_count_6095": True,
        "linf_within_tolerance": True,
    }
    failure_checks = {
        "person_drop_lt_0_03": person_drop < 0.03,
        "non_target_macro_drop_gt_0_15": non_target_macro_drop > 0.15,
        "non_target_count_drop_gt_0_20_ge_5": non_target_gt_020 >= 5,
    }
    if all(success_checks.values()):
        decision = "directional_feasibility_pass"
    elif any(failure_checks.values()):
        decision = "directional_feasibility_fail"
    else:
        decision = "inconclusive_tradeoff"
    return {
        "schema": "tausb.sdh-e2e-v0-comparison.v1",
        "protocol_id": "TAUSB-SDH-E2E-V0-MAP50-v1",
        "claim_status": "directional_feasibility_single_seed_e20",
        "pilot_decision": decision,
        "mechanism_gate_passed": bool(clean["mechanism_gate_passed"]),
        "hiding_gate_passed": False,
        "evidence_scope": "end_to_end_feasibility_not_formal_method",
        "per_class": per_class,
        "summary": {
            "person_drop": person_drop,
            "non_target_macro_drop": non_target_macro_drop,
            "non_target_classes_drop_le_0_15": non_target_le_015,
            "non_target_classes_drop_gt_0_20": non_target_gt_020,
            "success_checks": success_checks,
            "failure_checks": failure_checks,
        },
        "paired_identity": {
            key: clean[key]
            for key in (
                "clean_val_manifest_sha256",
                "paired_training_protocol_sha256",
                "frozen_sdh_state_sha256",
                "mechanism_metrics_sha256",
                "mechanism_decision_sha256",
                "mechanism_config_sha256",
            )
        },
    }
