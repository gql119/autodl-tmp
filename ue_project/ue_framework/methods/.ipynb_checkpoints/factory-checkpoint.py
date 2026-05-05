from .em import EMPoisonGenerator
from .em_mask import EMMaskPoisoner
from .lsp_mask import LSPMaskPoisoner
from .ours import OursPoisonGenerator
from .rem_mask import REMMaskPoisoner
from .tap_mask import TAPMaskPoisoner


def build_generator(method: str, cfg, method_cfg, device, surrogate):
    if method == "em_bbox":
        return EMPoisonGenerator(cfg, method_cfg, device, surrogate)
    if method == "em_mask":
        return EMMaskPoisoner(cfg, method_cfg, device, surrogate)
    if method == "rem_mask":
        return REMMaskPoisoner(cfg, method_cfg, device, surrogate)
    if method == "tap_mask":
        return TAPMaskPoisoner(cfg, method_cfg, device, surrogate)
    if method == "lsp_mask":
        return LSPMaskPoisoner(cfg, method_cfg, device, surrogate)
    if method == "ours_mask":
        return OursPoisonGenerator(cfg, method_cfg, device, surrogate)
    if method == "tausb_mask":
        raise ValueError(
            "tausb_mask requires stage-level universal training before generator init. "
            "Use stages.generate._build_tausb_generator()."
        )
    raise ValueError(f"Unsupported method: {method}")
