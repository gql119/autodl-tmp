from typing import Dict, Iterable

import torch


class FeatureHookBank:
    def __init__(self, model: torch.nn.Module, layer_names: Iterable[str]):
        modules = dict(model.named_modules())
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles = []
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"feature layer not found: {name}")
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 4:
                raise RuntimeError(f"feature hook {name} expected [B,C,H,W] tensor")
            self.outputs[name] = output
        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
