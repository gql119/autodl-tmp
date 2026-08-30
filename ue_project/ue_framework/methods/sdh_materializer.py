from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from .base import BasePoisonGenerator, PoisonResult
from .semantic_hiding_carrier import (
    SemanticHidingCarrier,
    render_person_box_carrier,
)


FROZEN_SDH_STATE_SCHEMA = "tausb_sdh_carrier_v1"
FORMAL_SDH_ARM_ID = "P1"
E2E_V0_PROTOCOL_ID = "TAUSB-SDH-E2E-V0-MAP50-v1"
E2E_V0_MATERIALIZATION_MODE = "p1_feasibility_state"
E2E_V0_EVIDENCE_SCOPE = "end_to_end_feasibility_not_formal_method"
DGCAIP_P4_PROTOCOL_ID = "TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1"
DGCAIP_P4_MATERIALIZATION_MODE = "p4_dgcaip_candidate_state"
DGCAIP_P4_EVIDENCE_SCOPE = "diagnostic_candidate_ap50_evaluation"
DGCAIP_P4_ARM_ID = "P4-DGCAIP"
DGCAIP_P4_PROVENANCE_HASH_KEYS = (
    "hiding_metrics_sha256",
    "hiding_checkpoint_sha256",
    "hiding_split_sha256",
    "mechanism_metrics_sha256",
    "mechanism_scientific_decision_sha256",
    "state_integrity_decision_sha256",
    "mechanism_config_sha256",
    "p4_state_sha256",
    "source_p1_state_sha256",
    "source_p1_metrics_sha256",
    "d0_report_sha256",
    "repair_report_sha256",
)


def _torch_load(path: str, *, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("%s must be a lowercase SHA-256 digest." % name)
    return digest


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sdh_state_content_hash(
    *,
    model_state: Mapping[str, torch.Tensor],
    secret: torch.Tensor,
    architecture_sha256: str,
    target_class_id: int,
    epsilon: float,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(model_state):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(model_state[name]).encode("ascii"))
    digest.update(_tensor_sha256(secret).encode("ascii"))
    digest.update(_require_sha256(architecture_sha256, "architecture_sha256").encode("ascii"))
    digest.update(
        ("target=%d;epsilon=%.17g" % (int(target_class_id), float(epsilon))).encode(
            "ascii"
        )
    )
    return digest.hexdigest()


def build_frozen_sdh_state_payload(
    *,
    carrier: SemanticHidingCarrier,
    secret: torch.Tensor,
    target_class_id: int,
    secret_source_sha256: str,
    secret_tensor_sha256: str,
    source_manifest_sha256: str,
    train_split_sha256: str,
    hiding_gate_passed: bool,
    mechanism_gate_passed: bool,
) -> Dict[str, Any]:
    if secret.shape != (1, 3, carrier.input_size, carrier.input_size):
        raise ValueError("Frozen SDH secret must be [1,3,input_size,input_size].")
    model_state = {
        name: value.detach().cpu().clone() for name, value in carrier.state_dict().items()
    }
    architecture_hash = carrier.architecture_sha256()
    secret_cpu = secret.detach().cpu().float().contiguous()
    content_hash = sdh_state_content_hash(
        model_state=model_state,
        secret=secret_cpu,
        architecture_sha256=architecture_hash,
        target_class_id=target_class_id,
        epsilon=carrier.epsilon,
    )
    return {
        "schema": FROZEN_SDH_STATE_SCHEMA,
        "arm_id": FORMAL_SDH_ARM_ID,
        "hiding_gate_passed": bool(hiding_gate_passed),
        "mechanism_gate_passed": bool(mechanism_gate_passed),
        "enable_deep_hiding": True,
        "enable_dlfc": True,
        "enable_cicr": True,
        "enable_cgr": True,
        "enable_nla_loss": True,
        "eot_enabled": False,
        "jnd_enabled": False,
        "target_class_id": int(target_class_id),
        "epsilon": float(carrier.epsilon),
        "input_size": carrier.input_size,
        "width": carrier.width,
        "coupling_blocks": carrier.coupling_blocks,
        "hf_subband_scale": carrier.hf_subband_scale,
        "architecture_sha256": architecture_hash,
        "secret_source_sha256": _require_sha256(
            secret_source_sha256, "secret_source_sha256"
        ),
        "secret_tensor_sha256": _require_sha256(
            secret_tensor_sha256, "secret_tensor_sha256"
        ),
        "source_manifest_sha256": _require_sha256(
            source_manifest_sha256, "source_manifest_sha256"
        ),
        "train_split_sha256": _require_sha256(
            train_split_sha256, "train_split_sha256"
        ),
        "state_content_hash": content_hash,
        "secret": secret_cpu,
        "model_state": model_state,
    }


def build_feasibility_sdh_state_payload(
    *,
    carrier: SemanticHidingCarrier,
    secret: torch.Tensor,
    target_class_id: int,
    secret_source_sha256: str,
    secret_tensor_sha256: str,
    source_manifest_sha256: str,
    train_split_sha256: str,
    mechanism_gate_passed: bool,
    hiding_metrics_sha256: str,
    hiding_checkpoint_sha256: str,
    hiding_split_sha256: str,
    mechanism_metrics_sha256: str,
    mechanism_decision_sha256: str,
    mechanism_config_sha256: str,
    p1_state_sha256: str,
) -> Dict[str, Any]:
    if int(target_class_id) != 14:
        raise ValueError("E2E V0 feasibility state requires person=14.")
    if abs(float(carrier.epsilon) - 16.0 / 255.0) > 1e-12:
        raise ValueError("E2E V0 feasibility state requires epsilon=16/255.")
    if not torch.isfinite(secret).all() or not (0 <= secret).all() or not (secret <= 1).all():
        raise ValueError("E2E V0 secret must be finite in [0,1].")
    if any(not torch.isfinite(value).all() for value in carrier.state_dict().values()):
        raise ValueError("E2E V0 P1 carrier state contains a non-finite tensor.")
    payload = build_frozen_sdh_state_payload(
        carrier=carrier,
        secret=secret,
        target_class_id=target_class_id,
        secret_source_sha256=secret_source_sha256,
        secret_tensor_sha256=secret_tensor_sha256,
        source_manifest_sha256=source_manifest_sha256,
        train_split_sha256=train_split_sha256,
        hiding_gate_passed=False,
        mechanism_gate_passed=mechanism_gate_passed,
    )
    payload.update(
        {
            "protocol_id": E2E_V0_PROTOCOL_ID,
            "materialization_mode": E2E_V0_MATERIALIZATION_MODE,
            "allow_failed_scientific_gates": True,
            "evidence_scope": E2E_V0_EVIDENCE_SCOPE,
            "failed_hiding_checks": [
                "delta_high_frequency",
                "rms_diversity",
            ],
            "hiding_metrics_sha256": _require_sha256(
                hiding_metrics_sha256, "hiding_metrics_sha256"
            ),
            "hiding_checkpoint_sha256": _require_sha256(
                hiding_checkpoint_sha256, "hiding_checkpoint_sha256"
            ),
            "hiding_split_sha256": _require_sha256(
                hiding_split_sha256, "hiding_split_sha256"
            ),
            "mechanism_metrics_sha256": _require_sha256(
                mechanism_metrics_sha256, "mechanism_metrics_sha256"
            ),
            "mechanism_decision_sha256": _require_sha256(
                mechanism_decision_sha256, "mechanism_decision_sha256"
            ),
            "mechanism_config_sha256": _require_sha256(
                mechanism_config_sha256, "mechanism_config_sha256"
            ),
            "p1_state_sha256": _require_sha256(
                p1_state_sha256, "p1_state_sha256"
            ),
        }
    )
    return payload


def build_dgcaip_p4_candidate_state_payload(
    *,
    carrier: SemanticHidingCarrier,
    secret: torch.Tensor,
    target_class_id: int,
    secret_source_sha256: str,
    secret_tensor_sha256: str,
    source_manifest_sha256: str,
    train_split_sha256: str,
    mechanism_scientific_gate_passed: bool,
    provenance_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    missing = sorted(set(DGCAIP_P4_PROVENANCE_HASH_KEYS).difference(provenance_hashes))
    if missing:
        raise ValueError("DG-CAIP P4 provenance hashes are missing: %s" % missing)
    if any(not torch.isfinite(value).all() for value in carrier.state_dict().values()):
        raise ValueError("DG-CAIP P4 carrier state contains a non-finite tensor.")
    payload = build_frozen_sdh_state_payload(
        carrier=carrier,
        secret=secret,
        target_class_id=target_class_id,
        secret_source_sha256=secret_source_sha256,
        secret_tensor_sha256=secret_tensor_sha256,
        source_manifest_sha256=source_manifest_sha256,
        train_split_sha256=train_split_sha256,
        hiding_gate_passed=False,
        mechanism_gate_passed=bool(mechanism_scientific_gate_passed),
    )
    payload.update(
        {
            "arm_id": DGCAIP_P4_ARM_ID,
            "source_arm_id": DGCAIP_P4_ARM_ID,
            "protocol_id": DGCAIP_P4_PROTOCOL_ID,
            "materialization_mode": DGCAIP_P4_MATERIALIZATION_MODE,
            "allow_failed_scientific_gates": True,
            "evidence_scope": DGCAIP_P4_EVIDENCE_SCOPE,
            "state_integrity_gate_passed": True,
            "mechanism_scientific_gate_passed": bool(
                mechanism_scientific_gate_passed
            ),
            **{
                name: _require_sha256(provenance_hashes[name], name)
                for name in DGCAIP_P4_PROVENANCE_HASH_KEYS
            },
        }
    )
    return payload


@dataclass(frozen=True)
class FrozenSDHState:
    carrier: SemanticHidingCarrier
    secret: torch.Tensor
    target_class_id: int
    epsilon: float
    secret_source_sha256: str
    secret_tensor_sha256: str
    source_manifest_sha256: str
    train_split_sha256: str
    state_content_hash: str
    provenance: Mapping[str, Any]


def load_frozen_sdh_state(
    path: str,
    *,
    device: torch.device,
    expected_target_class_id: int,
    expected_epsilon: float,
    expected_hashes: Mapping[str, str],
    feasibility_contract: Optional[Mapping[str, str]] = None,
) -> FrozenSDHState:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Frozen SDH state not found: %s" % path)
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != FROZEN_SDH_STATE_SCHEMA:
        raise ValueError("Frozen SDH state has an unsupported schema.")
    feasibility_mode = feasibility_contract is not None
    candidate_mode = bool(
        feasibility_mode
        and feasibility_contract.get("protocol_id") == DGCAIP_P4_PROTOCOL_ID
    )
    expected_arm_id = DGCAIP_P4_ARM_ID if candidate_mode else FORMAL_SDH_ARM_ID
    if payload.get("arm_id") != expected_arm_id:
        raise ValueError(
            "SDH materialization requires arm_id=%s." % expected_arm_id
        )
    if candidate_mode:
        if payload.get("protocol_id") != DGCAIP_P4_PROTOCOL_ID:
            raise ValueError("DG-CAIP P4 state protocol_id mismatch.")
        if payload.get("materialization_mode") != DGCAIP_P4_MATERIALIZATION_MODE:
            raise ValueError("DG-CAIP P4 state materialization mode mismatch.")
        if payload.get("evidence_scope") != DGCAIP_P4_EVIDENCE_SCOPE:
            raise ValueError("DG-CAIP P4 state evidence scope mismatch.")
        if payload.get("allow_failed_scientific_gates") is not True:
            raise ValueError("DG-CAIP P4 state did not preserve diagnostic gates.")
        if payload.get("state_integrity_gate_passed") is not True:
            raise ValueError("DG-CAIP P4 state integrity gate did not pass.")
        if payload.get("source_arm_id") != DGCAIP_P4_ARM_ID:
            raise ValueError("DG-CAIP P4 source arm identity mismatch.")
        if not isinstance(payload.get("mechanism_scientific_gate_passed"), bool):
            raise ValueError("DG-CAIP P4 scientific gate flag is missing.")
        expected_state_hash = _require_sha256(
            feasibility_contract.get("frozen_sdh_state_sha256", ""),
            "frozen_sdh_state_sha256",
        )
        if _file_sha256(path) != expected_state_hash:
            raise ValueError("DG-CAIP P4 frozen state file hash mismatch.")
        for name in DGCAIP_P4_PROVENANCE_HASH_KEYS:
            actual = _require_sha256(payload.get(name, ""), name)
            expected = _require_sha256(feasibility_contract.get(name, ""), name)
            if actual != expected:
                raise ValueError("DG-CAIP P4 %s does not match config." % name)
    elif feasibility_mode:
        if payload.get("protocol_id") != E2E_V0_PROTOCOL_ID:
            raise ValueError("E2E V0 state protocol_id mismatch.")
        if payload.get("materialization_mode") != E2E_V0_MATERIALIZATION_MODE:
            raise ValueError("E2E V0 state materialization mode mismatch.")
        if payload.get("allow_failed_scientific_gates") is not True:
            raise ValueError("E2E V0 state did not enable failed scientific gates.")
        if payload.get("evidence_scope") != E2E_V0_EVIDENCE_SCOPE:
            raise ValueError("E2E V0 state evidence scope mismatch.")
        if payload.get("hiding_gate_passed") is not False:
            raise ValueError("E2E V0 must preserve hiding_gate_passed=false.")
        if not isinstance(payload.get("mechanism_gate_passed"), bool):
            raise ValueError("E2E V0 mechanism gate flag is missing.")
        if payload.get("failed_hiding_checks") != [
            "delta_high_frequency",
            "rms_diversity",
        ]:
            raise ValueError("E2E V0 failed hiding checks mismatch.")
        expected_state_hash = _require_sha256(
            feasibility_contract.get("frozen_sdh_state_sha256", ""),
            "frozen_sdh_state_sha256",
        )
        if _file_sha256(path) != expected_state_hash:
            raise ValueError("E2E V0 frozen state file hash mismatch.")
        for name in (
            "hiding_metrics_sha256",
            "hiding_checkpoint_sha256",
            "hiding_split_sha256",
            "mechanism_metrics_sha256",
            "mechanism_decision_sha256",
            "mechanism_config_sha256",
            "p1_state_sha256",
        ):
            actual = _require_sha256(payload.get(name, ""), name)
            expected = _require_sha256(feasibility_contract.get(name, ""), name)
            if actual != expected:
                raise ValueError("E2E V0 %s does not match config." % name)
    else:
        for gate in ("hiding_gate_passed", "mechanism_gate_passed"):
            if payload.get(gate) is not True:
                raise ValueError("Frozen SDH %s is not passed." % gate)
    for switch in (
        "enable_deep_hiding",
        "enable_dlfc",
        "enable_cicr",
        "enable_cgr",
        "enable_nla_loss",
    ):
        if payload.get(switch) is not True:
            raise ValueError("Frozen SDH state requires %s=true." % switch)
    if payload.get("eot_enabled") is not False or payload.get("jnd_enabled") is not False:
        raise ValueError("First-round SDH state forbids EOT and JND.")
    target_class_id = int(payload.get("target_class_id", -1))
    epsilon = float(payload.get("epsilon", float("nan")))
    if target_class_id != int(expected_target_class_id):
        raise ValueError("Frozen SDH target class mismatch.")
    if not math.isfinite(epsilon) or abs(epsilon - float(expected_epsilon)) > 1e-12:
        raise ValueError("Frozen SDH epsilon mismatch.")
    carrier = SemanticHidingCarrier(
        input_size=int(payload.get("input_size", -1)),
        width=int(payload.get("width", -1)),
        coupling_blocks=int(payload.get("coupling_blocks", -1)),
        epsilon=epsilon,
        hf_subband_scale=float(payload.get("hf_subband_scale", 1.0)),
    )
    architecture_hash = _require_sha256(
        payload.get("architecture_sha256", ""), "architecture_sha256"
    )
    if carrier.architecture_sha256() != architecture_hash:
        raise ValueError("Frozen SDH architecture hash mismatch.")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("Frozen SDH model state must be a mapping.")
    if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in model_state.values()):
        raise ValueError("Frozen SDH model state contains an invalid tensor.")
    carrier.load_state_dict(model_state, strict=True)
    carrier.to(device).eval()
    for parameter in carrier.parameters():
        parameter.requires_grad_(False)
    secret = torch.as_tensor(payload.get("secret"), dtype=torch.float32)
    if secret.shape != (1, 3, carrier.input_size, carrier.input_size):
        raise ValueError("Frozen SDH secret shape mismatch.")
    if not torch.isfinite(secret).all() or not (0 <= secret).all() or not (secret <= 1).all():
        raise ValueError("Frozen SDH secret values must be finite in [0,1].")
    hashes = {
        name: _require_sha256(payload.get(name, ""), name)
        for name in (
            "secret_source_sha256",
            "secret_tensor_sha256",
            "source_manifest_sha256",
            "train_split_sha256",
        )
    }
    for name, expected in expected_hashes.items():
        if hashes.get(name) != _require_sha256(expected, name):
            raise ValueError("Frozen SDH %s does not match config." % name)
    content_hash = _require_sha256(payload.get("state_content_hash", ""), "state_content_hash")
    expected_content = sdh_state_content_hash(
        model_state=model_state,
        secret=secret,
        architecture_sha256=architecture_hash,
        target_class_id=target_class_id,
        epsilon=epsilon,
    )
    if content_hash != expected_content:
        raise ValueError("Frozen SDH state content hash mismatch.")
    return FrozenSDHState(
        carrier=carrier,
        secret=secret.to(device),
        target_class_id=target_class_id,
        epsilon=epsilon,
        secret_source_sha256=hashes["secret_source_sha256"],
        secret_tensor_sha256=hashes["secret_tensor_sha256"],
        source_manifest_sha256=hashes["source_manifest_sha256"],
        train_split_sha256=hashes["train_split_sha256"],
        state_content_hash=content_hash,
        provenance=(
            {
                "protocol_id": DGCAIP_P4_PROTOCOL_ID,
                "source_arm_id": DGCAIP_P4_ARM_ID,
                "evidence_scope": DGCAIP_P4_EVIDENCE_SCOPE,
                "hiding_gate_passed": False,
                "mechanism_gate_passed": payload["mechanism_gate_passed"],
                "mechanism_scientific_gate_passed": payload[
                    "mechanism_scientific_gate_passed"
                ],
                "state_integrity_gate_passed": True,
                "frozen_sdh_state_sha256": feasibility_contract[
                    "frozen_sdh_state_sha256"
                ],
                **{
                    name: payload[name]
                    for name in DGCAIP_P4_PROVENANCE_HASH_KEYS
                },
            }
            if candidate_mode
            else
            {
                "protocol_id": E2E_V0_PROTOCOL_ID,
                "evidence_scope": E2E_V0_EVIDENCE_SCOPE,
                "hiding_gate_passed": False,
                "mechanism_gate_passed": payload["mechanism_gate_passed"],
                "frozen_sdh_state_sha256": feasibility_contract[
                    "frozen_sdh_state_sha256"
                ],
                **{
                    name: payload[name]
                    for name in (
                        "hiding_metrics_sha256",
                        "hiding_checkpoint_sha256",
                        "hiding_split_sha256",
                        "mechanism_metrics_sha256",
                        "mechanism_decision_sha256",
                        "mechanism_config_sha256",
                        "p1_state_sha256",
                    )
                },
            }
            if feasibility_mode
            else {
                "protocol_id": "TAUSB-SDH-LFC-CICR-CGR-NLA-MAP50-v3",
                "evidence_scope": "formal_method",
                "hiding_gate_passed": True,
                "mechanism_gate_passed": True,
            }
        ),
    )


class SDHMaterializer(BasePoisonGenerator):
    """Deterministic person-GT-box materializer from one hash-bound P1 state."""

    def __init__(self, cfg, method_cfg, device, surrogate) -> None:
        super().__init__(cfg, method_cfg, device, surrogate)
        protocol_id = str(method_cfg.get("protocol_id", ""))
        feasibility_mode = protocol_id == E2E_V0_PROTOCOL_ID
        candidate_mode = protocol_id == DGCAIP_P4_PROTOCOL_ID
        feasibility_markers = bool(
            method_cfg.get("allow_failed_scientific_gates", False)
            or method_cfg.get("materialization_mode")
            in {E2E_V0_MATERIALIZATION_MODE, DGCAIP_P4_MATERIALIZATION_MODE}
        )
        if feasibility_markers and not (feasibility_mode or candidate_mode):
            raise ValueError(
                "Failed-gate materialization is restricted to an approved protocol."
            )
        if feasibility_mode:
            if method_cfg.get("materialization_mode") != E2E_V0_MATERIALIZATION_MODE:
                raise ValueError("E2E V0 materialization mode mismatch.")
            if method_cfg.get("allow_failed_scientific_gates") is not True:
                raise ValueError("E2E V0 failed-gate materialization was not enabled.")
        if candidate_mode:
            if method_cfg.get("materialization_mode") != DGCAIP_P4_MATERIALIZATION_MODE:
                raise ValueError("DG-CAIP P4 materialization mode mismatch.")
            if method_cfg.get("allow_failed_scientific_gates") is not True:
                raise ValueError("DG-CAIP P4 diagnostic materialization was not enabled.")
        contract = (
            {
                "protocol_id": protocol_id,
                **{
                    name: str(method_cfg.get(name, ""))
                    for name in (
                        "frozen_sdh_state_sha256",
                        *(
                            DGCAIP_P4_PROVENANCE_HASH_KEYS
                            if candidate_mode
                            else (
                                "hiding_metrics_sha256",
                                "hiding_checkpoint_sha256",
                                "hiding_split_sha256",
                                "mechanism_metrics_sha256",
                                "mechanism_decision_sha256",
                                "mechanism_config_sha256",
                                "p1_state_sha256",
                            )
                        ),
                    )
                },
            }
            if feasibility_mode or candidate_mode
            else None
        )
        self.state = load_frozen_sdh_state(
            str(method_cfg.get("frozen_sdh_state", "")),
            device=device,
            expected_target_class_id=self.target_class_id,
            expected_epsilon=self.eps,
            expected_hashes={
                name: str(method_cfg.get(name, ""))
                for name in (
                    "secret_source_sha256",
                    "secret_tensor_sha256",
                    "source_manifest_sha256",
                    "train_split_sha256",
                )
            },
            feasibility_contract=contract,
        )

    def generate(
        self,
        image: np.ndarray,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ) -> PoisonResult:
        if support_type != "bbox":
            raise ValueError("tausb_sdh formal materialization requires person GT boxes.")
        if abs(float(eps) - self.state.epsilon) > 1e-12:
            raise ValueError("Runtime epsilon does not match frozen SDH state.")
        image_tensor = self._to_tensor(image)
        boxes = self._collect_target_gt_boxes_xyxy(annotations, image.shape)
        if boxes.numel() == 0:
            zero = np.zeros_like(image, dtype=np.float32)
            support = np.zeros(image.shape[:2], dtype=np.float32)
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=support,
                ring_mask=support.copy(),
                losses={"L_total": 0.0},
                extras={"is_poisoned": False, "poisoned": 0, "support_source": "none"},
            )
        with torch.no_grad():
            rendered = render_person_box_carrier(
                image_tensor,
                (boxes,),
                self.state.carrier,
                self.state.secret,
            )
        perturbation = self._to_numpy(rendered.perturbation)
        support = rendered.union_support[0, 0].detach().cpu().numpy().astype(np.float32)
        if float(np.max(np.abs(perturbation) * (1.0 - support[..., None]))) != 0.0:
            raise RuntimeError("SDH perturbation leaked outside person GT boxes.")
        linf = float(np.max(np.abs(perturbation)))
        return PoisonResult(
            poisoned_image=self._to_numpy(rendered.poisoned),
            perturbation=perturbation,
            support_mask=support,
            ring_mask=np.zeros_like(support),
            losses={"L_total": 0.0},
            extras={
                "is_poisoned": bool(linf > 0),
                "poisoned": int(linf > 0),
                "support_source": "person_gt_bbox",
                "state_content_hash": self.state.state_content_hash,
                "semantic_bank_hash": self.state.secret_tensor_sha256,
                "source_manifest_hash": self.state.source_manifest_sha256,
                "split_hash": self.state.train_split_sha256,
                "secret_source_sha256": self.state.secret_source_sha256,
                **dict(self.state.provenance),
                "linf": linf,
            },
        )
