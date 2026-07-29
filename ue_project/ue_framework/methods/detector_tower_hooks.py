from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class DetectTowerFeatures:
    classification: tuple[torch.Tensor, ...]
    box: tuple[torch.Tensor, ...]


class YOLODetectTowerCapture:
    def __init__(
        self,
        model: nn.Module,
        *,
        detect_path: str = "model.22",
        num_scales: int = 3,
        expected_num_classes: int = 20,
    ) -> None:
        self._model = model
        self._detect_path = detect_path
        self._num_scales = int(num_scales)
        self._expected_num_classes = int(expected_num_classes)
        self._active_tag: str | None = None
        self._records: dict[str, dict[tuple[str, int], torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._closed = False

        modules = dict(model.named_modules())
        detect = modules.get(detect_path)
        if detect is None:
            raise ValueError(f"Missing YOLO Detect module: {detect_path}")
        if not hasattr(detect, "cv3") or not hasattr(detect, "cv2"):
            raise ValueError(
                f"{detect_path} must expose decoupled cv3 and cv2 branches."
            )
        if len(detect.cv3) != self._num_scales or len(detect.cv2) != self._num_scales:
            raise ValueError(
                f"{detect_path} must have {self._num_scales} cv3/cv2 scales."
            )

        detect_nc = int(getattr(detect, "nc", self._expected_num_classes))
        if detect_nc != self._expected_num_classes:
            raise ValueError(
                f"Detect class count {detect_nc} != {self._expected_num_classes}."
            )

        for tower_name, branches in (("cv3", detect.cv3), ("cv2", detect.cv2)):
            for scale_index, branch in enumerate(branches):
                if not isinstance(branch, nn.Sequential) or len(branch) < 1:
                    raise ValueError(
                        f"{detect_path}.{tower_name}.{scale_index} must be Sequential."
                    )
                final = branch[-1]
                expected_path = (
                    f"{detect_path}.{tower_name}.{scale_index}.{len(branch) - 1}"
                )
                if modules.get(expected_path) is not final:
                    raise ValueError(
                        f"Module path {expected_path} is not the branch final layer."
                    )
                if not isinstance(final, nn.Conv2d):
                    raise ValueError(f"{expected_path} must be a final Conv2d.")
                if (
                    tower_name == "cv3"
                    and final.out_channels != self._expected_num_classes
                ):
                    raise ValueError(
                        f"{expected_path} out_channels={final.out_channels}; "
                        f"expected {self._expected_num_classes} classes."
                    )
                handle = final.register_forward_pre_hook(
                    self._make_hook(tower_name, scale_index)
                )
                self._handles.append(handle)

    def _make_hook(self, tower_name: str, scale_index: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if self._active_tag is None:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(
                    f"{tower_name}[{scale_index}] pre-hook expected one tensor input."
                )
            feature = inputs[0]
            if feature.ndim != 4:
                raise RuntimeError(
                    f"{tower_name}[{scale_index}] feature must be [B,C,H,W]."
                )
            key = (tower_name, scale_index)
            record = self._records[self._active_tag]
            if key in record:
                raise RuntimeError(
                    f"Duplicate {tower_name}[{scale_index}] capture for "
                    f"tag={self._active_tag}."
                )
            record[key] = feature

        return hook

    @contextmanager
    def record(self, tag: str) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("YOLODetectTowerCapture is closed.")
        if self._active_tag is not None:
            raise RuntimeError("Nested tower capture is not supported.")
        if not tag:
            raise ValueError("Capture tag must be non-empty.")
        self._records[tag] = {}
        self._active_tag = tag
        try:
            yield
        finally:
            self._active_tag = None

    def take(self, tag: str) -> DetectTowerFeatures:
        record = self._records.pop(tag, None)
        if record is None:
            raise KeyError(f"No tower capture for tag={tag}.")
        expected = {
            (tower_name, scale_index)
            for tower_name in ("cv3", "cv2")
            for scale_index in range(self._num_scales)
        }
        if set(record) != expected:
            missing = sorted(expected.difference(record))
            extra = sorted(set(record).difference(expected))
            raise RuntimeError(
                f"Incomplete tower capture for tag={tag}: "
                f"missing={missing} extra={extra}."
            )
        return DetectTowerFeatures(
            classification=tuple(
                record[("cv3", scale_index)]
                for scale_index in range(self._num_scales)
            ),
            box=tuple(
                record[("cv2", scale_index)]
                for scale_index in range(self._num_scales)
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._records.clear()
        self._active_tag = None
        self._closed = True

    def __enter__(self) -> "YOLODetectTowerCapture":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
