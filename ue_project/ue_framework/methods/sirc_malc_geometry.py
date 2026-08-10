from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .bsc_icmo_probe import ICMOBatch
from .bsc_rc_gr_probe import _write_probe_json, make_batches
from .instance_canonical_carrier import tensor_sha256
from .malc import multi_scale_assignment_latent_concentration
from .malc_calibration import MALCGradientNormCalibrator, MALCPrototypeCalibrator
from .malc_cgr import class_probability_constraint_terms
from .malc_geometry_audit import (
    PrototypeGeometryAccumulator,
    classify_first_bad_boundary,
    component_gradient_geometry,
    detached_cosine,
    prototype_bank_hash,
    summarize_gradient_geometry,
)
from .semantic_residual_carrier import VariantMatchedCanonicalCarrier
from .sirc_malc_mechanism import SIRCMALCMechanismWorkflow


def validate_geometry_config(config: Mapping[str, Any]) -> None:
    spec = config.get("spec", {})
    if spec.get("spec_id") != "TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1":
        raise ValueError("MALC geometry spec_id mismatch.")
    if spec.get("exp_id") != "TAUSB-MALC-GRAD-GEOMETRY-S0":
        raise ValueError("MALC geometry exp_id mismatch.")
    geometry = config.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("MALC geometry config requires a geometry section.")
    exact = {
        "calibration_images": 64,
        "heldout_images": 96,
        "microtrajectory_steps": 8,
    }
    for key, expected in exact.items():
        if int(geometry.get(key, -1)) != expected:
            raise ValueError(f"geometry.{key} must remain exactly {expected}.")
    if not isinstance(geometry.get("run_microtrajectory"), bool):
        raise ValueError("geometry.run_microtrajectory must be boolean.")
    if config.get("eot", {}).get("enabled") is not False:
        raise ValueError("MALC geometry audit forbids EOT.")
    if int(config.get("optimization", {}).get("batch_size", -1)) != 4:
        raise ValueError("MALC geometry batch size must remain 4.")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SIRCMALCGeometryWorkflow(SIRCMALCMechanismWorkflow):
    """Read-only gradient geometry audit plus a matched eight-step A0/A1 trace."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        device_override: str | None = None,
        source_manifest: str | None = None,
        source_local_map: str | None = None,
    ) -> None:
        validate_geometry_config(config)
        super().__init__(
            config,
            config_path=config_path,
            device_override=device_override,
            source_manifest=source_manifest,
            source_local_map=source_local_map,
        )
        self.geometry = self.config["geometry"]
        self._status_path = self.artifact_root / "status.json"

    def _progress(self, status: dict[str, Any], *, stage: str, **values: Any) -> None:
        status.update({"stage": stage, **values})
        if self.base.device.type == "cuda":
            status["cuda_memory_allocated"] = int(
                torch.cuda.memory_allocated(self.base.device)
            )
        _write_probe_json(self._status_path, status)

    def _prototype_calibration(
        self,
        *,
        carrier: VariantMatchedCanonicalCarrier,
        calibration_batches: Sequence[Sequence[str]],
        status: dict[str, Any],
    ):
        calibrator = MALCPrototypeCalibrator(
            num_scales=3,
            split_hash=str(self.base.split["split_hash"]),
            energy_floor_multiplier=0.5,
        )
        accumulator = PrototypeGeometryAccumulator(num_scales=3)
        with torch.no_grad():
            for batch_index, paths in enumerate(calibration_batches):
                observation = self._observe(self.base._load_batch(paths), carrier)
                calibrator.update(observation.malc_residuals, split="calibration")
                accumulator.update(observation.malc_residuals)
                del observation
                self._progress(
                    status,
                    stage="prototype_calibration",
                    batch_index=batch_index + 1,
                    batch_count=len(calibration_batches),
                )
        calibration = calibrator.finalize()
        geometry = accumulator.finalize(
            reference_prototypes=calibration.bank.direction_prototypes
        )
        return calibration, geometry

    def _gradient_calibration_and_geometry(
        self,
        *,
        carrier: VariantMatchedCanonicalCarrier,
        prototype_calibration: Any,
        calibration_batches: Sequence[Sequence[str]],
        status: dict[str, Any],
    ):
        calibrator = MALCGradientNormCalibrator(
            component_names=("easy_cls", "malc", "rms"),
            reference_name="easy_cls",
            clip_min=0.1,
            clip_max=10.0,
            max_clipped_fraction=0.5,
        )
        records = []
        for batch_index, paths in enumerate(calibration_batches):
            batch = self.base._load_batch(paths)
            observation = self._observe(batch, carrier)
            malc = multi_scale_assignment_latent_concentration(
                observation.malc_residuals,
                prototype_calibration.bank,
            )
            losses = {
                "easy_cls": observation.route.loss,
                "malc": malc.loss,
                "rms": self._rms_loss(carrier),
            }
            calibrator.update(losses, (carrier.coefficients,))
            geometry = component_gradient_geometry(
                parameter=carrier.coefficients,
                losses=losses,
                constraints=class_probability_constraint_terms(
                    observation.constraints,
                    tolerance=float(self.optimization["cgr_tolerance"]),
                ),
                near_boundary=float(self.optimization["cgr_near_boundary"]),
                svd_relative_tolerance=float(
                    self.optimization["cgr_svd_relative_tolerance"]
                ),
            )
            record = geometry.record
            record.update(
                {
                    "batch_index": batch_index,
                    "image_ids": list(batch.image_ids),
                    "image_id_hash": _json_digest(list(batch.image_ids)),
                }
            )
            records.append(record)
            del geometry, losses, malc, observation, batch
            self._progress(
                status,
                stage="gradient_geometry",
                batch_index=batch_index + 1,
                batch_count=len(calibration_batches),
            )
        frozen = calibrator.finalize()
        summary = summarize_gradient_geometry(records)
        summary.update(
            {
                "gradient_calibration_hash": frozen.calibration_hash,
                "gradient_median_norms": dict(frozen.median_norms),
                "gradient_weights": dict(frozen.weights),
                "gradient_clipped_components": list(frozen.clipped_components),
            }
        )
        return frozen, summary

    def _heldout_prototype_geometry(
        self,
        *,
        carrier: VariantMatchedCanonicalCarrier,
        prototype_calibration: Any,
        heldout_batches: Sequence[Sequence[str]],
        status: dict[str, Any],
    ) -> dict[str, Any]:
        accumulator = PrototypeGeometryAccumulator(num_scales=3)
        original = carrier.coefficients.detach().clone()
        with torch.no_grad():
            for batch_index, paths in enumerate(heldout_batches):
                observation = self._observe(self.base._load_batch(paths), carrier)
                accumulator.update(observation.malc_residuals)
                del observation
                if not torch.equal(carrier.coefficients.detach(), original):
                    raise RuntimeError("Held-out geometry modified carrier coefficients.")
                self._progress(
                    status,
                    stage="heldout_geometry",
                    batch_index=batch_index + 1,
                    batch_count=len(heldout_batches),
                )
        return accumulator.finalize(
            reference_prototypes=prototype_calibration.bank.direction_prototypes
        )

    @staticmethod
    def _pattern_snapshot(
        step: int,
        a0: VariantMatchedCanonicalCarrier,
        a1: VariantMatchedCanonicalCarrier,
        *,
        epsilon: float,
    ) -> dict[str, Any]:
        with torch.no_grad():
            pattern_a0 = a0().detach()
            pattern_a1 = a1().detach()
            separation = float((pattern_a1 - pattern_a0).square().mean().sqrt())
            snapshot = {
                "step": int(step),
                "A0_pattern_hash": tensor_sha256(pattern_a0),
                "A1_pattern_hash": tensor_sha256(pattern_a1),
                "A0_pattern_rms": float(pattern_a0.square().mean().sqrt()),
                "A1_pattern_rms": float(pattern_a1.square().mean().sqrt()),
                "pattern_separation_rms": separation,
                "normalized_pattern_separation": separation / float(epsilon),
            }
        return snapshot

    def _micro_arm_step(
        self,
        *,
        arm_id: str,
        enable_malc: bool,
        batch: ICMOBatch,
        carrier: VariantMatchedCanonicalCarrier,
        prototype_calibration: Any,
        gradient_calibration: Any,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        before = carrier.coefficients.detach().clone()
        observation = self._observe(batch, carrier)
        malc = multi_scale_assignment_latent_concentration(
            observation.malc_residuals,
            prototype_calibration.bank,
        )
        rms_loss = self._rms_loss(carrier)
        target_loss = observation.route.loss + float(
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
        after = carrier.coefficients.detach().clone()
        update = (after - before).detach().to(device="cpu", dtype=torch.float64)
        record = {
            "arm_id": arm_id,
            "enable_malc": bool(enable_malc),
            "coefficient_hash_before": tensor_sha256(before),
            "coefficient_hash_after": tensor_sha256(after),
            "actual_update": update.reshape(-1).tolist(),
            "actual_update_norm": float(update.norm()),
            "target_loss": float(target_loss.detach()),
            "easy_cls_loss": float(observation.route.loss.detach()),
            "malc_loss": float(malc.loss.detach()),
            "rms_loss": float(rms_loss.detach()),
            "cgr_mode": routed.selected_mode,
            "cgr_accepted": bool(routed.accepted),
            "cgr_attempts": int(routed.attempts),
            "cgr_attack_retention": float(routed.route.attack_retention),
            "cgr_max_projected_row_dot": float(
                routed.route.max_projected_row_dot
            ),
        }
        del routed, target_loss, rms_loss, malc, observation
        return record, update

    def _microtrajectory(
        self,
        *,
        warm_coefficients: torch.Tensor,
        prototype_calibration: Any,
        gradient_calibration: Any,
        calibration_batches: Sequence[Sequence[str]],
        status: dict[str, Any],
    ) -> dict[str, Any]:
        a0 = self.base._carrier("I-SV")
        a1 = self.base._carrier("I-SV")
        with torch.no_grad():
            a0.coefficients.copy_(warm_coefficients)
            a1.coefficients.copy_(warm_coefficients)
        if not torch.equal(a0.coefficients.detach(), a1.coefficients.detach()):
            raise RuntimeError("Matched A0/A1 initial coefficients differ.")
        epsilon = float(self.config["carrier"]["epsilon"])
        snapshots = [self._pattern_snapshot(0, a0, a1, epsilon=epsilon)]
        steps = int(self.geometry["microtrajectory_steps"])
        records = []
        for step in range(steps):
            paths = calibration_batches[step]
            batch = self.base._load_batch(paths)
            a0_record, a0_update = self._micro_arm_step(
                arm_id="A0",
                enable_malc=False,
                batch=batch,
                carrier=a0,
                prototype_calibration=prototype_calibration,
                gradient_calibration=gradient_calibration,
            )
            a1_record, a1_update = self._micro_arm_step(
                arm_id="A1",
                enable_malc=True,
                batch=batch,
                carrier=a1,
                prototype_calibration=prototype_calibration,
                gradient_calibration=gradient_calibration,
            )
            update_cosine = None
            if float(a0_update.norm()) > 1e-12 and float(a1_update.norm()) > 1e-12:
                update_cosine = detached_cosine(a0_update, a1_update)
            coefficient_distance = float(
                (a1.coefficients.detach() - a0.coefficients.detach()).norm()
            )
            records.append(
                {
                    "step": step + 1,
                    "image_ids": list(batch.image_ids),
                    "image_id_hash": _json_digest(list(batch.image_ids)),
                    "A0": a0_record,
                    "A1": a1_record,
                    "update_cosine": update_cosine,
                    "coefficient_distance": coefficient_distance,
                }
            )
            if step + 1 in {4, 8}:
                snapshots.append(
                    self._pattern_snapshot(step + 1, a0, a1, epsilon=epsilon)
                )
            del batch, a0_update, a1_update
            self._progress(
                status,
                stage="microtrajectory",
                step=step + 1,
                step_count=steps,
            )
        warm = warm_coefficients.detach()
        denominator = float((a0.coefficients.detach() - warm).norm())
        d_theta = float(
            (a1.coefficients.detach() - a0.coefficients.detach()).norm()
        ) / max(denominator, 1e-12)
        final_snapshot = snapshots[-1]
        return {
            "steps": steps,
            "initial_coefficients_hash": tensor_sha256(warm_coefficients),
            "batch_sequence_hash": _json_digest(
                [record["image_ids"] for record in records]
            ),
            "records": records,
            "pattern_snapshots": snapshots,
            "D_theta": d_theta,
            "D_pattern": float(final_snapshot["normalized_pattern_separation"]),
            "allow_fresh_victim": False,
        }

    def run(self, *, smoke: bool = False) -> dict[str, Any]:
        if smoke:
            raise ValueError(
                "Geometry smoke must use a dedicated synthetic/local test; the formal "
                "workflow may not silently reduce frozen batch counts."
            )
        status: dict[str, Any] = {
            "state": "running",
            "stage": "input_validation",
            "claim_boundary": "seed0_surrogate_geometry_only_not_fresh_victim_ue",
            "allow_fresh_victim": False,
        }
        _write_probe_json(self._status_path, status)
        try:
            calibration_paths = self.base.split["calibration"]
            heldout_paths = self.base.split["heldout"]
            if len(calibration_paths) != int(self.geometry["calibration_images"]):
                raise RuntimeError("Calibration image count differs from the frozen 64.")
            if len(heldout_paths) != int(self.geometry["heldout_images"]):
                raise RuntimeError("Held-out image count differs from the frozen 96.")
            batch_size = int(self.optimization["batch_size"])
            calibration_batches = make_batches(calibration_paths, batch_size=batch_size)
            heldout_batches = make_batches(heldout_paths, batch_size=batch_size)
            if len(calibration_batches) != 16 or len(heldout_batches) != 24:
                raise RuntimeError("Frozen geometry batch counts must be 16 and 24.")

            input_audit_path = self.artifact_root / "input_audit.json"
            input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
            input_audit["geometry_protocol"] = {
                "calibration_images": len(calibration_paths),
                "calibration_batches": len(calibration_batches),
                "heldout_images": len(heldout_paths),
                "heldout_batches": len(heldout_batches),
                "batch_size": batch_size,
                "warmup_steps": int(self.optimization["warmup_steps"]),
                "microtrajectory_steps": int(self.geometry["microtrajectory_steps"]),
                "run_microtrajectory": bool(self.geometry["run_microtrajectory"]),
                "eot_enabled": bool(self.config["eot"]["enabled"]),
            }
            _write_probe_json(input_audit_path, input_audit)

            self._progress(status, stage="warmup")
            warm_carrier, warmup_diagnostics = self._warm_start(
                calibration_batches,
                smoke=False,
            )
            warm_coefficients = warm_carrier.coefficients.detach().clone()

            prototype_calibration, calibration_geometry = self._prototype_calibration(
                carrier=warm_carrier,
                calibration_batches=calibration_batches,
                status=status,
            )
            bank_hash_before = prototype_bank_hash(prototype_calibration.bank)
            gradient_calibration, gradient_geometry = (
                self._gradient_calibration_and_geometry(
                    carrier=warm_carrier,
                    prototype_calibration=prototype_calibration,
                    calibration_batches=calibration_batches,
                    status=status,
                )
            )
            heldout_geometry = self._heldout_prototype_geometry(
                carrier=warm_carrier,
                prototype_calibration=prototype_calibration,
                heldout_batches=heldout_batches,
                status=status,
            )
            prototype_geometry = {
                "prototype_calibration_hash": prototype_calibration.calibration_hash,
                "prototype_split_hash": prototype_calibration.split_hash,
                "prototype_bank_hash_before": bank_hash_before,
                "calibration": calibration_geometry,
                "heldout": heldout_geometry,
            }
            _write_probe_json(
                self.artifact_root / "prototype_geometry.json",
                prototype_geometry,
            )
            _write_probe_json(
                self.artifact_root / "gradient_geometry.json",
                gradient_geometry,
            )

            microtrajectory = None
            if bool(self.geometry["run_microtrajectory"]):
                microtrajectory = self._microtrajectory(
                    warm_coefficients=warm_coefficients,
                    prototype_calibration=prototype_calibration,
                    gradient_calibration=gradient_calibration,
                    calibration_batches=calibration_batches,
                    status=status,
                )
                _write_probe_json(
                    self.artifact_root / "microtrajectory.json",
                    microtrajectory,
                )

            bank_hash_after = prototype_bank_hash(prototype_calibration.bank)
            prototype_geometry["prototype_bank_hash_after"] = bank_hash_after
            prototype_geometry["prototype_bank_unchanged"] = (
                bank_hash_before == bank_hash_after
            )
            _write_probe_json(
                self.artifact_root / "prototype_geometry.json",
                prototype_geometry,
            )
            if bank_hash_before != bank_hash_after:
                raise RuntimeError("Frozen MALC prototype bank changed during audit.")

            decision = classify_first_bad_boundary(
                prototype_geometry=prototype_geometry,
                gradient_geometry=gradient_geometry,
                microtrajectory=microtrajectory,
            )
            decision.update(
                {
                    "spec_id": self.config["spec"]["spec_id"],
                    "exp_id": self.config["spec"]["exp_id"],
                    "split_hash": self.base.split["split_hash"],
                    "warmup_diagnostics": warmup_diagnostics,
                    "prototype_bank_unchanged": bank_hash_before == bank_hash_after,
                    "quality_metrics": {
                        "PSNR": "N/A_no_materialization",
                        "LPIPS": "N/A_no_materialization",
                        "poisoned_count": "N/A_no_materialization",
                    },
                }
            )
            _write_probe_json(
                self.artifact_root / "diagnostic_decision.json",
                decision,
            )
            status.update(
                {
                    "state": "completed" if decision["valid"] else "stopped",
                    "stage": "diagnostic_decision",
                    "first_bad_boundary": decision["first_bad_boundary"],
                    "valid": bool(decision["valid"]),
                    "allow_fresh_victim": False,
                }
            )
            _write_probe_json(self._status_path, status)
            return status
        except Exception as error:
            status.update(
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "allow_fresh_victim": False,
                }
            )
            _write_probe_json(self._status_path, status)
            raise
