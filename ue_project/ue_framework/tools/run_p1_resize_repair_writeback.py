from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import yaml

from ue_framework.methods.constraint_gradient_router import (
    backtrack_multi_parameter_update,
    route_multi_parameter_gradients,
)
from ue_framework.methods.p1_determinism_audit import (
    RESIZE_REPAIR_SPEC_ID,
    capture_module_snapshot,
    enable_strict_determinism,
    module_snapshot_manifest,
    payload_sha256,
)
from ue_framework.methods.p1_determinism_experiment import (
    _batch_to_device,
    _prepare_context,
    _validate_bound_artifacts,
)
from ue_framework.methods.sdh_experiment import (
    _clone_detector_carrier,
    _component_losses,
    _copy_parameters_,
    validate_sdh_experiment_config,
)
from ue_framework.methods.sdh_mechanism import (
    adapter_parameters,
    compose_sdh_target_objective,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two strict P1 writeback steps after resize repair."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parameter_hash(parameters) -> str:
    return payload_sha256(tuple(value.detach().cpu() for value in parameters))


def _module_hash(module: torch.nn.Module) -> str:
    manifest = module_snapshot_manifest(capture_module_snapshot(module))
    return payload_sha256(manifest)


def run_writeback_smoke(config: dict, *, config_base: Path) -> dict:
    validate_sdh_experiment_config(config)
    if str(config["spec"]["spec_id"]) != RESIZE_REPAIR_SPEC_ID:
        raise ValueError("Writeback smoke received the wrong SpecID.")
    _validate_bound_artifacts(config, config_base=config_base)
    started = time.monotonic()
    context = _prepare_context(config, config_base=config_base, start=started)
    try:
        device = context["device"]
        carrier = _clone_detector_carrier(context["base_carrier"], device)
        parameters = adapter_parameters(carrier)
        adapter_before = _parameter_hash(parameters)
        frozen_before = {
            "model": _module_hash(context["engine"].model),
            "hiding_trunk": _module_hash(carrier.hiding_trunk),
            "reveal_decoder": _module_hash(carrier.reveal_decoder),
            "dlfc_bank": payload_sha256(context["dlfc_bank"].state_dict()),
            "cicr_bank": payload_sha256(context["cicr_bank"].state_dict()),
            "target_weights": payload_sha256(dict(context["target_weights"])),
            "dgcaip_weights": payload_sha256(dict(context["dgcaip_weights"])),
            "nla_weight": payload_sha256(float(context["lambda_nla"])),
        }
        steps = []
        accepted_count = 0
        batch = _batch_to_device(context["cpu_batch"], device)
        for step in range(2):
            observation = context["engine"].observe(
                batch,
                carrier,
                context["secret"],
                dgcaip_mode="off",
            )
            components, _, _ = _component_losses(
                observation,
                context["dlfc_bank"],
                context["cicr_bank"],
            )
            objective = compose_sdh_target_objective(
                easy=components["easy"],
                reveal=components["reveal"],
                rms=components["rms"],
                dlfc=components["dlfc"],
                cicr=components["cicr"],
                floor=components["floor"],
                weights=context["target_weights"],
                enable_dlfc=True,
                enable_cicr=True,
            )
            routed = route_multi_parameter_gradients(
                parameters=parameters,
                target_loss=objective.loss,
                per_class_nla_losses={
                    str(key): value
                    for key, value in observation.nla.per_class_loss.items()
                },
                nla_loss=observation.nla.loss,
                nla_weight=float(context["lambda_nla"]),
            )
            originals = tuple(parameter.detach().clone() for parameter in parameters)

            def evaluate(candidate):
                _copy_parameters_(parameters, candidate)
                try:
                    with torch.no_grad():
                        current = context["engine"].observe(
                            batch,
                            carrier,
                            context["secret"],
                            dgcaip_mode="off",
                        )
                    return {
                        str(key): value
                        for key, value in current.per_class_probability_drop.items()
                    }
                finally:
                    _copy_parameters_(parameters, originals)

            backtracked = backtrack_multi_parameter_update(
                parameters=parameters,
                flattened_gradient=routed.gradient,
                step_size=float(config["mechanism"]["learning_rate"]),
                evaluate_probability_drops=evaluate,
                tolerance=float(config["mechanism"]["probability_drop_tolerance"]),
                max_backtracks=int(config["mechanism"]["max_backtracks"]),
            )
            if backtracked.accepted:
                _copy_parameters_(parameters, backtracked.candidate)
                accepted_count += 1
            steps.append(
                {
                    "step": step,
                    "objective": float(objective.loss.detach()),
                    "route_mode": routed.mode,
                    "target_norm": routed.target_norm,
                    "projected_target_norm": routed.projected_target_norm,
                    "nla_norm": routed.nla_norm,
                    "combined_norm": routed.combined_norm,
                    "max_projected_row_dot": routed.max_projected_row_dot,
                    "max_final_row_dot": routed.max_final_row_dot,
                    "backtrack_attempts": backtracked.attempts,
                    "accepted": bool(backtracked.accepted),
                    "step_size": backtracked.step_size,
                    "probability_drop": backtracked.values,
                }
            )

        with torch.no_grad():
            final_observation = context["engine"].observe(
                batch,
                carrier,
                context["secret"],
                dgcaip_mode="off",
            )
        perturbation = final_observation.rendered.perturbation
        outside = perturbation * (
            ~final_observation.rendered.union_support
        ).expand_as(perturbation)
        adapter_after = _parameter_hash(parameters)
        frozen_after = {
            "model": _module_hash(context["engine"].model),
            "hiding_trunk": _module_hash(carrier.hiding_trunk),
            "reveal_decoder": _module_hash(carrier.reveal_decoder),
            "dlfc_bank": payload_sha256(context["dlfc_bank"].state_dict()),
            "cicr_bank": payload_sha256(context["cicr_bank"].state_dict()),
            "target_weights": payload_sha256(dict(context["target_weights"])),
            "dgcaip_weights": payload_sha256(dict(context["dgcaip_weights"])),
            "nla_weight": payload_sha256(float(context["lambda_nla"])),
        }
        frozen_checks = {
            name: frozen_before[name] == frozen_after[name] for name in frozen_before
        }
        finite = all(
            torch.isfinite(torch.tensor(list(item["probability_drop"].values()))).all()
            and all(
                torch.isfinite(torch.tensor(item[name]))
                for name in (
                    "objective",
                    "target_norm",
                    "projected_target_norm",
                    "nla_norm",
                    "combined_norm",
                    "max_projected_row_dot",
                    "max_final_row_dot",
                )
            )
            for item in steps
        )
        linf = float(perturbation.abs().max())
        support_valid = int(torch.count_nonzero(outside)) == 0
        passed = bool(
            accepted_count >= 1
            and adapter_before != adapter_after
            and all(frozen_checks.values())
            and finite
            and linf <= 16.0 / 255.0 + 1.0e-6
            and support_valid
        )
        artifact_root = Path(str(config["runtime"]["artifact_root"]))
        smoke_root = artifact_root / "writeback_smoke"
        smoke_root.mkdir(parents=True, exist_ok=False)
        state_path = smoke_root / "p1_resize_repair_smoke_state.pt"
        torch.save(
            {
                "spec_id": RESIZE_REPAIR_SPEC_ID,
                "carrier_state": carrier.state_dict(),
                "accepted_steps": accepted_count,
            },
            state_path,
        )
        saved = torch.load(state_path, map_location=device)
        verification_carrier = _clone_detector_carrier(context["base_carrier"], device)
        verification_carrier.load_state_dict(saved["carrier_state"], strict=True)
        state_loadable = (
            _parameter_hash(adapter_parameters(verification_carrier)) == adapter_after
        )
        passed = bool(passed and state_loadable)
        status = (
            "passed"
            if passed
            else "algorithmic_no_acceptance"
            if accepted_count == 0
            else "failed_invariant"
        )
        return {
            "schema": "tausb.p1-resize-repair-writeback.v1",
            "spec_id": RESIZE_REPAIR_SPEC_ID,
            "status": status,
            "passed": passed,
            "accepted_steps": accepted_count,
            "adapter_sha256_before": adapter_before,
            "adapter_sha256_after": adapter_after,
            "adapter_changed": adapter_before != adapter_after,
            "frozen_state_checks": frozen_checks,
            "finite": bool(finite),
            "perturbation_linf": linf,
            "support_valid": support_valid,
            "state_loadable": state_loadable,
            "steps": steps,
            "state_path": str(state_path),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        context["engine"].close()


def main() -> int:
    args = _arguments()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = Path(str(config["runtime"]["artifact_root"]))
    result_path = artifact_root / "writeback_smoke.json"
    started = time.time()
    try:
        enable_strict_determinism()
        result = run_writeback_smoke(config, config_base=Path.cwd().resolve())
        result["started_unix"] = started
        result["ended_unix"] = time.time()
        _write_json(result_path, result)
        return 0 if result["passed"] else 2
    except Exception as error:
        _write_json(
            result_path,
            {
                "schema": "tausb.p1-resize-repair-writeback.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "ended_unix": time.time(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
