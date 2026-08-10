from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .bsc_icmo_probe import ICMOBatch, ICMOObservation
from .bsc_rc_gr_probe import _write_probe_json, make_batches
from .malc import multi_scale_assignment_latent_concentration
from .malc_calibration import (
    MALCGradientNormCalibrator,
    MALCPrototypeCalibrator,
)
from .malc_cgr import (
    class_probability_constraint_terms,
    route_malc_cgr_update,
)
from .malc_mechanism import (
    MALCMechanismBatch,
    aggregate_malc_mechanism_batches,
    assert_matched_mechanism_configs,
    detach_malc_result,
    evaluate_malc_mechanism_gate,
    write_malc_mechanism_report,
)
from .semantic_residual_carrier import VariantMatchedCanonicalCarrier
from .sirc_malc_cgr import build_frozen_sirc_state_payload
from .sirc_probe import SIRCProbeWorkflow


class SIRCMALCMechanismWorkflow:
    """Matched A0/A1 detector-native MALC mechanism experiment."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        device_override: str | None = None,
        source_manifest: str | None = None,
        source_local_map: str | None = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        method = self.config.get("method", {})
        if method.get("enable_sirc_carrier") is not True:
            raise ValueError("MALC mechanism requires the shared SIRC carrier.")
        if method.get("enable_cgr") is not True:
            raise ValueError("MALC mechanism requires CGR for both A0 and A1.")
        if self.config.get("eot", {}).get("enabled") is not False:
            raise ValueError("The approved MALC mechanism run forbids EOT.")
        self.base = SIRCProbeWorkflow(
            self.config,
            config_path=config_path,
            device_override=device_override,
            source_manifest=source_manifest,
            source_local_map=source_local_map,
        )
        self.artifact_root = self.base.artifact_root
        self.optimization = self.config["optimization"]

    def close(self) -> None:
        self.base.close()

    def _observe(
        self,
        batch: ICMOBatch,
        carrier: VariantMatchedCanonicalCarrier,
    ) -> ICMOObservation:
        return self.base._observe(batch, carrier, arm_id="I-SV")

    def _rms_loss(self, carrier: VariantMatchedCanonicalCarrier) -> torch.Tensor:
        target = float(self.config["carrier"]["epsilon"]) * float(
            self.config["carrier"]["target_rms_ratio"]
        )
        return (carrier().square().mean().sqrt() / target - 1.0).square()

    @staticmethod
    def _mean_vector_norm(
        vectors: Sequence[torch.Tensor],
        valid: Sequence[torch.Tensor],
    ) -> float:
        values = []
        for tensor, gate in zip(vectors, valid):
            if bool(gate.any()):
                values.extend(float(value) for value in tensor.detach()[gate].norm(dim=1))
        return float(np.mean(values)) if values else 0.0

    def _evaluate_class_margins(
        self,
        candidate: torch.Tensor,
        *,
        batch: ICMOBatch,
        carrier: VariantMatchedCanonicalCarrier,
    ) -> dict[str, float]:
        original = carrier.coefficients.detach().clone()
        try:
            with torch.no_grad():
                carrier.coefficients.copy_(candidate)
                observation = self._observe(batch, carrier)
                return {
                    term.name: float(term.margin)
                    for term in class_probability_constraint_terms(
                        observation.constraints,
                        tolerance=float(self.optimization["cgr_tolerance"]),
                    )
                }
        finally:
            with torch.no_grad():
                carrier.coefficients.copy_(original)

    def _route(
        self,
        *,
        batch: ICMOBatch,
        carrier: VariantMatchedCanonicalCarrier,
        observation: ICMOObservation,
        target_loss: torch.Tensor,
    ):
        return route_malc_cgr_update(
            parameter=carrier.coefficients,
            target_loss=target_loss,
            constraint_set=observation.constraints,
            step_size=float(self.optimization["learning_rate"]),
            evaluate_class_margins=lambda candidate: self._evaluate_class_margins(
                candidate,
                batch=batch,
                carrier=carrier,
            ),
            tolerance=float(self.optimization["cgr_tolerance"]),
            near_boundary=float(self.optimization["cgr_near_boundary"]),
            svd_relative_tolerance=float(
                self.optimization["cgr_svd_relative_tolerance"]
            ),
            max_backtracks=int(self.optimization["cgr_max_backtracks"]),
        )

    def _warm_start(
        self,
        calibration_batches: Sequence[Sequence[str]],
        *,
        smoke: bool,
    ) -> tuple[VariantMatchedCanonicalCarrier, list[dict[str, Any]]]:
        carrier = self.base._carrier("I-SV")
        steps = 1 if smoke else int(self.optimization["warmup_steps"])
        diagnostics = []
        for step in range(steps):
            batch = self.base._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self._observe(batch, carrier)
            routed = self._route(
                batch=batch,
                carrier=carrier,
                observation=observation,
                target_loss=observation.route.loss,
            )
            if routed.accepted:
                with torch.no_grad():
                    carrier.coefficients.copy_(routed.candidate)
            diagnostics.append(
                {
                    "stage": "route_warmup",
                    "step": step,
                    "route_loss": float(observation.route.loss.detach()),
                    "cgr_mode": routed.selected_mode,
                    "cgr_accepted": routed.accepted,
                    "cgr_attempts": routed.attempts,
                    "cgr_attack_retention": routed.route.attack_retention,
                    "cgr_max_projected_row_dot": routed.route.max_projected_row_dot,
                }
            )
        return carrier, diagnostics

    def _fit_calibrations(
        self,
        carrier: VariantMatchedCanonicalCarrier,
        calibration_batches: Sequence[Sequence[str]],
        *,
        smoke: bool,
    ):
        selected = calibration_batches[:1] if smoke else calibration_batches
        prototype = MALCPrototypeCalibrator(
            num_scales=3,
            split_hash=str(self.base.split["split_hash"]),
            energy_floor_multiplier=0.5,
        )
        observations = []
        for paths in selected:
            observation = self._observe(self.base._load_batch(paths), carrier)
            prototype.update(observation.malc_residuals, split="calibration")
            observations.append(observation)
        prototype_calibration = prototype.finalize()

        gradients = MALCGradientNormCalibrator(
            component_names=("easy_cls", "malc", "rms"),
            reference_name="easy_cls",
            clip_min=0.1,
            clip_max=10.0,
            max_clipped_fraction=0.5,
        )
        for observation in observations:
            malc = multi_scale_assignment_latent_concentration(
                observation.malc_residuals,
                prototype_calibration.bank,
            )
            gradients.update(
                {
                    "easy_cls": observation.route.loss,
                    "malc": malc.loss,
                    "rms": self._rms_loss(carrier),
                },
                (carrier.coefficients,),
            )
        return prototype_calibration, gradients.finalize()

    def _optimize_arm(
        self,
        *,
        arm_id: str,
        enable_malc: bool,
        warm_coefficients: torch.Tensor,
        prototype_calibration,
        gradient_calibration,
        calibration_batches: Sequence[Sequence[str]],
        smoke: bool,
    ) -> tuple[VariantMatchedCanonicalCarrier, list[dict[str, Any]]]:
        carrier = self.base._carrier("I-SV")
        with torch.no_grad():
            carrier.coefficients.copy_(warm_coefficients)
        steps = 1 if smoke else int(self.optimization["optimization_steps"])
        diagnostics = []
        for step in range(steps):
            batch = self.base._load_batch(
                calibration_batches[step % len(calibration_batches)]
            )
            observation = self._observe(batch, carrier)
            malc = multi_scale_assignment_latent_concentration(
                observation.malc_residuals,
                prototype_calibration.bank,
            )
            rms_loss = self._rms_loss(carrier)
            target_loss = observation.route.loss
            target_loss = target_loss + float(
                gradient_calibration.weights["rms"]
            ) * rms_loss
            if enable_malc:
                target_loss = target_loss + float(
                    gradient_calibration.weights["malc"]
                ) * malc.loss
            routed = self._route(
                batch=batch,
                carrier=carrier,
                observation=observation,
                target_loss=target_loss,
            )
            if routed.accepted:
                with torch.no_grad():
                    carrier.coefficients.copy_(routed.candidate)
            diagnostics.append(
                {
                    "arm_id": arm_id,
                    "step": step,
                    "enable_malc": bool(enable_malc),
                    "target_loss": float(target_loss.detach()),
                    "easy_cls_loss": float(observation.route.loss.detach()),
                    "malc_loss": float(malc.loss.detach()),
                    "rms_loss": float(rms_loss.detach()),
                    "valid_instance_coverage": malc.valid_instance_coverage,
                    "scale_contribution_share": malc.scale_contribution_share,
                    "cgr_mode": routed.selected_mode,
                    "cgr_accepted": routed.accepted,
                    "cgr_attempts": routed.attempts,
                    "cgr_attack_retention": routed.route.attack_retention,
                    "cgr_max_projected_row_dot": routed.route.max_projected_row_dot,
                }
            )
        return carrier, diagnostics

    def _heldout_batches(
        self,
        *,
        carrier: VariantMatchedCanonicalCarrier,
        enable_malc: bool,
        prototype_calibration,
        gradient_calibration,
        smoke: bool,
    ) -> list[MALCMechanismBatch]:
        batches = make_batches(
            self.base.split["heldout"],
            batch_size=int(self.optimization["batch_size"]),
        )
        if smoke:
            batches = batches[:1]
        records = []
        original = carrier.coefficients.detach().clone()
        for paths in batches:
            batch = self.base._load_batch(paths)
            observation = self._observe(batch, carrier)
            malc = multi_scale_assignment_latent_concentration(
                observation.malc_residuals,
                prototype_calibration.bank,
            )
            target_loss = observation.route.loss + float(
                gradient_calibration.weights["rms"]
            ) * self._rms_loss(carrier)
            if enable_malc:
                target_loss = target_loss + float(
                    gradient_calibration.weights["malc"]
                ) * malc.loss
            routed = self._route(
                batch=batch,
                carrier=carrier,
                observation=observation,
                target_loss=target_loss,
            )
            target_energy = self._mean_vector_norm(
                observation.malc_residuals.vectors,
                observation.malc_residuals.pooling_valid,
            )
            non_target_energy = self._mean_vector_norm(
                observation.non_target_residuals.vectors,
                observation.non_target_residuals.gate_valid,
            )
            box_energy = self._mean_vector_norm(
                observation.box_residuals.vectors,
                observation.box_residuals.gate_valid,
            )
            size_groups = tuple(
                group
                for groups in batch.instance_scale_groups_by_image
                for group in groups
            )
            cooccur_flags = tuple(
                flag
                for flag, groups in zip(
                    batch.person_cooccur,
                    batch.instance_scale_groups_by_image,
                )
                for _ in groups
            )
            records.append(
                MALCMechanismBatch(
                    malc=detach_malc_result(malc),
                    cgr_attack_retention=routed.route.attack_retention,
                    cgr_max_projected_row_dot=routed.route.max_projected_row_dot,
                    cgr_selected_mode=routed.selected_mode,
                    non_target_target_energy_ratio=non_target_energy
                    / max(target_energy, 1e-12),
                    box_residual_energy=box_energy,
                    size_groups=size_groups,
                    cooccur_flags=cooccur_flags,
                )
            )
            if not torch.equal(carrier.coefficients.detach(), original):
                raise RuntimeError("Held-out mechanism evaluation modified carrier state.")
            del observation, malc, target_loss, routed
        return records

    def run(self, *, smoke: bool = False) -> dict[str, Any]:
        status_path = self.artifact_root / "status.json"
        status = {
            "state": "running",
            "stage": "warmup",
            "claim_boundary": "mechanism_only_not_fresh_victim_ue",
            "smoke": bool(smoke),
        }
        _write_probe_json(status_path, status)
        try:
            a0_config = copy.deepcopy(self.config)
            a1_config = copy.deepcopy(self.config)
            a0_config["method"]["enable_malc"] = False
            a1_config["method"]["enable_malc"] = True
            assert_matched_mechanism_configs(a0_config, a1_config)

            calibration_batches = make_batches(
                self.base.split["calibration"],
                batch_size=int(self.optimization["batch_size"]),
            )
            if not calibration_batches:
                raise RuntimeError("Calibration split produced no batches.")
            warm_carrier, warmup_diagnostics = self._warm_start(
                calibration_batches,
                smoke=smoke,
            )
            warm_coefficients = warm_carrier.coefficients.detach().clone()
            status.update({"stage": "calibration"})
            _write_probe_json(status_path, status)
            prototype_calibration, gradient_calibration = self._fit_calibrations(
                warm_carrier,
                calibration_batches,
                smoke=smoke,
            )
            calibration_summary = {
                "prototype_hash": prototype_calibration.calibration_hash,
                "prototype_split_hash": prototype_calibration.split_hash,
                "prototype_counts": prototype_calibration.per_scale_vector_count,
                "gradient_hash": gradient_calibration.calibration_hash,
                "gradient_median_norms": dict(gradient_calibration.median_norms),
                "gradient_weights": dict(gradient_calibration.weights),
                "gradient_clipped_components": gradient_calibration.clipped_components,
            }
            _write_probe_json(
                self.artifact_root / "malc_calibration.json",
                calibration_summary,
            )

            arm_outputs = {}
            carriers = {}
            for arm_id, enabled in (("A0", False), ("A1", True)):
                status.update({"stage": f"optimize_{arm_id}"})
                _write_probe_json(status_path, status)
                carrier, diagnostics = self._optimize_arm(
                    arm_id=arm_id,
                    enable_malc=enabled,
                    warm_coefficients=warm_coefficients,
                    prototype_calibration=prototype_calibration,
                    gradient_calibration=gradient_calibration,
                    calibration_batches=calibration_batches,
                    smoke=smoke,
                )
                heldout = self._heldout_batches(
                    carrier=carrier,
                    enable_malc=enabled,
                    prototype_calibration=prototype_calibration,
                    gradient_calibration=gradient_calibration,
                    smoke=smoke,
                )
                summary = aggregate_malc_mechanism_batches(
                    heldout,
                    split="heldout",
                )
                del heldout
                summary.update(
                    {
                        "arm_id": arm_id,
                        "enable_malc": enabled,
                        "optimization_steps": len(diagnostics),
                    }
                )
                arm_outputs[arm_id] = summary
                carriers[arm_id] = carrier
                _write_probe_json(
                    self.artifact_root / "arms" / f"{arm_id}_diagnostics.json",
                    diagnostics,
                )
                _write_probe_json(
                    self.artifact_root / "arms" / f"{arm_id}_metrics.json",
                    summary,
                )

            gate = evaluate_malc_mechanism_gate(
                arm_outputs["A0"], arm_outputs["A1"]
            )
            write_malc_mechanism_report(
                self.artifact_root / "mechanism_report.json",
                a0=arm_outputs["A0"],
                a1=arm_outputs["A1"],
                gate=gate,
                split_hash=str(self.base.split["split_hash"]),
            )
            frozen_path = None
            if gate["pass"] and not smoke:
                frozen_path = self.artifact_root / "a1_frozen_carrier.pt"
                if frozen_path.exists():
                    raise FileExistsError(f"Frozen A1 state already exists: {frozen_path}")
                payload = build_frozen_sirc_state_payload(
                    semantic_bases=self.base.semantic_bank.semantic_bases,
                    semantic_scales=self.base.semantic_bank.semantic_scales,
                    coefficients=carriers["A1"].coefficients,
                    gamma=self.base.gamma_calibration.gamma,
                    epsilon=float(self.config["carrier"]["epsilon"]),
                    variant_seed=int(self.config["carrier"]["variant_seed"]),
                    jnd_floor=float(self.config["method"]["jnd_floor"]),
                    target_class_id=int(self.config["dataset"]["target_class_id"]),
                    semantic_bank_hash=self.base.semantic_bank.bank_hash,
                    source_manifest_hash=self.base.source_manifest_hash,
                    split_hash=str(self.base.split["split_hash"]),
                    mechanism_gate_passed=True,
                )
                torch.save(payload, frozen_path)
            status.update(
                {
                    "state": "completed" if gate["pass"] else "stopped",
                    "stage": "mechanism_gate",
                    "mechanism_pass": bool(gate["pass"]),
                    "allow_fresh_victim": bool(gate["allow_fresh_victim"]),
                    "frozen_carrier_state": str(frozen_path) if frozen_path else None,
                    "warmup_diagnostics": warmup_diagnostics,
                }
            )
            _write_probe_json(status_path, status)
            return status
        except Exception as error:
            status.update(
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_probe_json(status_path, status)
            raise
