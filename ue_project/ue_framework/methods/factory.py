from .em import EMPoisonGenerator
from .ours import OursPoisonGenerator
from .rem import REMPoisonGenerator



def build_generator(method: str, cfg, method_cfg, device, surrogate):
    if method in {"trajectory_p1", "meta_p2", "trajectory_meta_p2"}:
        raise ValueError(
            f"{method} is a batch-level learning-trajectory method. "
            "Use runners/generate_poison.py or runners/run_p0_diagnostics.py; "
            "the legacy per-image generator path is intentionally not used."
        )
    if method in {"em_bbox", "em_mask"}:
        return EMPoisonGenerator(cfg, method_cfg, device, surrogate)
    if method == "rem_mask":
        return REMPoisonGenerator(cfg, method_cfg, device, surrogate)
    if method == "ours_mask":
        return OursPoisonGenerator(cfg, method_cfg, device, surrogate)
    if method == "tausb_mask":
        raise ValueError(
            "tausb_mask requires stage-level universal training before generator init. "
            "Use stages.generate._build_tausb_generator()."
        )
    raise ValueError(f"Unsupported method: {method}")
