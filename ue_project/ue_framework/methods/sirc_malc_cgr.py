from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from ..support import build_forced_pseudo_instance_masks
from .base import BasePoisonGenerator, PoisonResult
from .instance_canonical_carrier import (
    apply_variant_canonical_patterns,
    tensor_sha256,
)
from .semantic_residual_carrier import (
    DEFAULT_RADIAL_EDGES,
    VariantMatchedCanonicalCarrier,
    stable_variant_index,
)


FROZEN_STATE_SCHEMA = "sirc_malc_cgr_carrier_v1"
FORMAL_ARM_ID = "A1"


def _torch_load(path: str, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require_hash(value: str, *, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return value


def frozen_state_content_hash(
    *,
    semantic_bases: torch.Tensor,
    semantic_scales: torch.Tensor,
    coefficients: torch.Tensor,
    gamma: float,
    epsilon: float,
    variant_seed: int,
) -> str:
    digest = hashlib.sha256()
    for tensor in (semantic_bases, semantic_scales, coefficients):
        digest.update(tensor_sha256(tensor).encode("ascii"))
    digest.update(
        (
            f"gamma={float(gamma):.17g};epsilon={float(epsilon):.17g};"
            f"variant_seed={int(variant_seed)}"
        ).encode("ascii")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSIRCCarrierState:
    semantic_bases: torch.Tensor
    semantic_scales: torch.Tensor
    coefficients: torch.Tensor
    gamma: float
    epsilon: float
    variant_seed: int
    jnd_floor: float
    target_class_id: int
    semantic_bank_hash: str
    source_manifest_hash: str
    split_hash: str
    state_content_hash: str
    arm_id: str


def build_frozen_sirc_state_payload(
    *,
    semantic_bases: torch.Tensor,
    semantic_scales: torch.Tensor,
    coefficients: torch.Tensor,
    gamma: float,
    epsilon: float,
    variant_seed: int,
    jnd_floor: float,
    target_class_id: int,
    semantic_bank_hash: str,
    source_manifest_hash: str,
    split_hash: str,
    mechanism_gate_passed: bool,
    arm_id: str = FORMAL_ARM_ID,
    radial_edges=DEFAULT_RADIAL_EDGES,
) -> Dict[str, Any]:
    content_hash = frozen_state_content_hash(
        semantic_bases=semantic_bases,
        semantic_scales=semantic_scales,
        coefficients=coefficients,
        gamma=gamma,
        epsilon=epsilon,
        variant_seed=variant_seed,
    )
    return {
        "schema": FROZEN_STATE_SCHEMA,
        "arm_id": str(arm_id),
        "mechanism_gate_passed": bool(mechanism_gate_passed),
        "enable_malc": True,
        "enable_cgr": True,
        "target_class_id": int(target_class_id),
        "epsilon": float(epsilon),
        "gamma": float(gamma),
        "variant_seed": int(variant_seed),
        "jnd_floor": float(jnd_floor),
        "radial_edges": [float(value) for value in radial_edges],
        "semantic_bank_hash": str(semantic_bank_hash),
        "source_manifest_hash": str(source_manifest_hash),
        "split_hash": str(split_hash),
        "state_content_hash": content_hash,
        "semantic_bases": semantic_bases.detach().cpu(),
        "semantic_scales": semantic_scales.detach().cpu(),
        "coefficients": coefficients.detach().cpu(),
    }


def load_frozen_sirc_state(
    path: str,
    *,
    device: torch.device | str,
    expected_target_class_id: int,
    expected_epsilon: float,
    require_mechanism_pass: bool = True,
) -> FrozenSIRCCarrierState:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Frozen SIRC carrier state not found: {path}")
    payload = _torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Frozen SIRC carrier state must be a mapping.")
    if payload.get("schema") != FROZEN_STATE_SCHEMA:
        raise ValueError("Frozen SIRC carrier state has an unsupported schema.")
    if str(payload.get("arm_id", "")) != FORMAL_ARM_ID:
        raise ValueError("Formal materialization requires the A1 carrier state.")
    if require_mechanism_pass and payload.get("mechanism_gate_passed") is not True:
        raise ValueError("A1 mechanism gate did not pass; M1 materialization is forbidden.")
    if payload.get("enable_malc") is not True or payload.get("enable_cgr") is not True:
        raise ValueError("Formal A1 state must record MALC and CGR as enabled.")

    target_class_id = int(payload.get("target_class_id", -1))
    if target_class_id != int(expected_target_class_id):
        raise ValueError("Frozen carrier target_class_id does not match the experiment.")
    epsilon = float(payload.get("epsilon", float("nan")))
    if not math.isfinite(epsilon) or abs(epsilon - float(expected_epsilon)) > 1e-12:
        raise ValueError("Frozen carrier epsilon does not match the experiment.")
    gamma = float(payload.get("gamma", float("nan")))
    jnd_floor = float(payload.get("jnd_floor", float("nan")))
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Frozen carrier gamma must be positive and finite.")
    if not math.isfinite(jnd_floor) or not 0 <= jnd_floor <= 1:
        raise ValueError("Frozen carrier jnd_floor must lie in [0,1].")
    radial_edges = tuple(float(value) for value in payload.get("radial_edges", ()))
    if radial_edges != tuple(DEFAULT_RADIAL_EDGES):
        raise ValueError("Frozen carrier radial edges do not match the approved [2,24] bank.")

    bases = torch.as_tensor(payload.get("semantic_bases"), dtype=torch.float32)
    scales = torch.as_tensor(payload.get("semantic_scales"), dtype=torch.float32)
    coefficients = torch.as_tensor(payload.get("coefficients"), dtype=torch.float32)
    if bases.ndim != 4 or bases.shape[0] != 4 or bases.shape[1] != 16:
        raise ValueError("Frozen semantic_bases must have shape [4,16,R,R].")
    if bases.shape[-2] != bases.shape[-1] or scales.shape != (4, 16):
        raise ValueError("Frozen semantic basis/scale shapes are invalid.")
    if coefficients.shape != (16, 3):
        raise ValueError("Frozen coefficients must contain exactly 48 RGB values.")
    if not all(torch.isfinite(value).all() for value in (bases, scales, coefficients)):
        raise ValueError("Frozen carrier tensors must be finite.")

    semantic_bank_hash = _require_hash(
        payload.get("semantic_bank_hash", ""), name="semantic_bank_hash"
    )
    source_manifest_hash = _require_hash(
        payload.get("source_manifest_hash", ""), name="source_manifest_hash"
    )
    split_hash = _require_hash(payload.get("split_hash", ""), name="split_hash")
    expected_content_hash = frozen_state_content_hash(
        semantic_bases=bases,
        semantic_scales=scales,
        coefficients=coefficients,
        gamma=gamma,
        epsilon=epsilon,
        variant_seed=int(payload.get("variant_seed", 0)),
    )
    state_content_hash = _require_hash(
        payload.get("state_content_hash", ""), name="state_content_hash"
    )
    if state_content_hash != expected_content_hash:
        raise ValueError("Frozen carrier content hash mismatch.")

    return FrozenSIRCCarrierState(
        semantic_bases=bases.to(device),
        semantic_scales=scales.to(device),
        coefficients=coefficients.to(device),
        gamma=gamma,
        epsilon=epsilon,
        variant_seed=int(payload.get("variant_seed", 0)),
        jnd_floor=jnd_floor,
        target_class_id=target_class_id,
        semantic_bank_hash=semantic_bank_hash,
        source_manifest_hash=source_manifest_hash,
        split_hash=split_hash,
        state_content_hash=state_content_hash,
        arm_id=FORMAL_ARM_ID,
    )


def resolve_sirc_malc_effective_method(method_cfg: Mapping[str, Any]) -> str:
    switches = (
        bool(method_cfg.get("enable_sirc_carrier", True)),
        bool(method_cfg.get("enable_malc", True)),
        bool(method_cfg.get("enable_cgr", True)),
    )
    return "tausb_mask" if not any(switches) else "sirc_malc_cgr"


class SIRCMALCCGRMaterializer(BasePoisonGenerator):
    """Materialize an already frozen, gate-passed A1 carrier.

    MALC and CGR act while optimizing the 48 carrier coefficients. They are not
    recomputed while writing the poisoned VOC dataset.
    """

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        if resolve_sirc_malc_effective_method(method_cfg) != "sirc_malc_cgr":
            raise ValueError("All v2 switches are off; dispatch to tausb_mask instead.")
        if method_cfg.get("enable_malc") is not True or method_cfg.get("enable_cgr") is not True:
            raise ValueError("Formal M1 materialization requires MALC and CGR enabled.")
        if int(cfg["surrogate"].get("eot_samples", 1)) != 1:
            raise ValueError("The approved v2 experiment forbids EOT; set eot_samples=1.")
        self.state = load_frozen_sirc_state(
            str(method_cfg.get("frozen_carrier_state", "")),
            device=device,
            expected_target_class_id=self.target_class_id,
            expected_epsilon=self.eps,
            require_mechanism_pass=bool(method_cfg.get("require_mechanism_pass", True)),
        )
        expected_hashes = {
            "semantic_bank_hash": self.state.semantic_bank_hash,
            "source_manifest_hash": self.state.source_manifest_hash,
            "split_hash": self.state.split_hash,
        }
        for key, actual in expected_hashes.items():
            expected = _require_hash(method_cfg.get(key, ""), name=key)
            if expected != actual:
                raise ValueError(f"Configured {key} does not match frozen A1 state.")
        configured_seed = int(method_cfg.get("variant_seed", self.state.variant_seed))
        if configured_seed != self.state.variant_seed:
            raise ValueError("Configured variant_seed does not match frozen A1 state.")
        configured_floor = float(method_cfg.get("jnd_floor", self.state.jnd_floor))
        if abs(configured_floor - self.state.jnd_floor) > 1e-12:
            raise ValueError("Configured jnd_floor does not match frozen A1 state.")
        self.variant_seed = configured_seed
        self.jnd_floor = configured_floor
        carrier = VariantMatchedCanonicalCarrier(
            self.state.semantic_bases,
            self.state.semantic_scales,
            epsilon=self.state.epsilon,
            gamma=self.state.gamma,
            initial_coefficients=self.state.coefficients,
        ).to(device)
        carrier.eval()
        with torch.no_grad():
            self.patterns = carrier().detach()

    @staticmethod
    def _target_boxes(
        annotations: List[dict], target_class_id: int, height: int, width: int
    ) -> List[tuple[int, int, int, int]]:
        boxes = []
        for annotation in annotations:
            if int(annotation["cls"]) != int(target_class_id):
                continue
            cx, cy, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1 = max(0, min(width, int((cx - box_width / 2.0) * width)))
            y1 = max(0, min(height, int((cy - box_height / 2.0) * height)))
            x2 = max(0, min(width, int((cx + box_width / 2.0) * width)))
            y2 = max(0, min(height, int((cy + box_height / 2.0) * height)))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

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
        if support_type != "mask":
            raise ValueError("sirc_malc_cgr only supports forced-pseudo mask materialization.")
        if abs(float(eps) - self.state.epsilon) > 1e-12:
            raise ValueError("Runtime epsilon does not match frozen A1 state.")
        height, width = image.shape[:2]
        boxes = self._target_boxes(annotations, self.target_class_id, height, width)
        masks = build_forced_pseudo_instance_masks(
            image.shape, annotations, self.target_class_id
        )
        if len(boxes) != len(masks):
            raise ValueError("Target box/support count mismatch.")
        if not boxes:
            zero = np.zeros_like(image, dtype=np.float32)
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=np.zeros((height, width), dtype=np.float32),
                ring_mask=np.zeros((height, width), dtype=np.float32),
                losses={"L_total": 0.0},
                extras={"is_poisoned": False, "poisoned": 0, "support_source": "none"},
            )

        image_tensor = self._to_tensor(image)
        supports = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(
            device=self.device, dtype=image_tensor.dtype
        )
        image_id = os.path.splitext(os.path.basename(image_path or str(seed)))[0]
        variant_index = stable_variant_index(
            image_id, seed=self.variant_seed, num_variants=self.patterns.shape[0]
        )
        with torch.no_grad():
            poisoned, perturbation, rendered = apply_variant_canonical_patterns(
                image_tensor,
                self.patterns,
                variant_indices=(variant_index,),
                boxes_by_image=(boxes,),
                supports_by_image=(supports,),
                mode="instance",
                epsilon=self.state.epsilon,
                jnd_floor=self.jnd_floor,
            )
        union = rendered[0].union_support[0].detach().cpu().numpy().astype(np.float32)
        perturbation_np = self._to_numpy(perturbation)
        outside = np.abs(perturbation_np) * (1.0 - union[..., None])
        outside_max = float(np.max(outside))
        if outside_max != 0.0:
            raise RuntimeError("Materialized perturbation leaked outside forced-pseudo support.")
        linf = float(np.max(np.abs(perturbation_np)))
        if linf > self.state.epsilon + 1e-7:
            raise RuntimeError("Materialized perturbation exceeded the frozen epsilon budget.")
        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=perturbation_np,
            support_mask=union,
            ring_mask=np.zeros_like(union),
            losses={"L_total": 0.0},
            extras={
                "is_poisoned": bool(linf > 0),
                "poisoned": int(linf > 0),
                "support_source": "forced_pseudo_fallback",
                "variant_index": int(variant_index),
                "state_content_hash": self.state.state_content_hash,
                "semantic_bank_hash": self.state.semantic_bank_hash,
                "source_manifest_hash": self.state.source_manifest_hash,
                "split_hash": self.state.split_hash,
                "linf": linf,
            },
        )
