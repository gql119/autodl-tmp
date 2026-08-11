from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from .base import BasePoisonGenerator, PoisonResult
from .semantic_hiding_carrier import (
    SemanticHidingCarrier,
    render_person_box_carrier,
)


FROZEN_SDH_STATE_SCHEMA = "tausb_sdh_carrier_v1"
FORMAL_SDH_ARM_ID = "P1"


def _torch_load(path: str, *, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def load_frozen_sdh_state(
    path: str,
    *,
    device: torch.device,
    expected_target_class_id: int,
    expected_epsilon: float,
    expected_hashes: Mapping[str, str],
) -> FrozenSDHState:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Frozen SDH state not found: %s" % path)
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != FROZEN_SDH_STATE_SCHEMA:
        raise ValueError("Frozen SDH state has an unsupported schema.")
    if payload.get("arm_id") != FORMAL_SDH_ARM_ID:
        raise ValueError("Formal SDH materialization requires the P1 state.")
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
    )


class SDHMaterializer(BasePoisonGenerator):
    """Deterministic person-GT-box materializer from a passing frozen P1 state."""

    def __init__(self, cfg, method_cfg, device, surrogate) -> None:
        super().__init__(cfg, method_cfg, device, surrogate)
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
                "linf": linf,
            },
        )
