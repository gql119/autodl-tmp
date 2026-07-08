from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class FeatureHookManager:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, module_names: Iterable[str]) -> None:
        self.clear()
        modules = dict(self.model.named_modules())
        for name in module_names:
            if name not in modules:
                raise KeyError(f"Module not found: {name}")

            def _hook(_module, _inputs, output, key=name):
                if torch.is_tensor(output):
                    self.features[key] = output

            self._handles.append(modules[name].register_forward_hook(_hook))

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.features.clear()
