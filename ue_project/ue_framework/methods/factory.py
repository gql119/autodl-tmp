from .em import EMPoisonGenerator
from .ours import OursPoisonGenerator
from .rem import REMPoisonGenerator
from .sirc_malc_cgr import SIRCMALCCGRMaterializer



def build_generator(method: str, cfg, method_cfg, device, surrogate):
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
    if method == "sirc_malc_cgr":
        return SIRCMALCCGRMaterializer(cfg, method_cfg, device, surrogate)
    raise ValueError(f"Unsupported method: {method}")
