from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.tal import make_anchors


@dataclass(frozen=True)
class ParameterManifest:
    mode: str
    selected_names: tuple[str, ...]
    omitted_names: tuple[str, ...]
    selected_tensors: int
    selected_parameters: int
    eligible_parameters: int
    coverage_percent: float
    selected_hash: str


@dataclass
class ReferenceAssignment:
    target_labels: torch.Tensor
    target_bboxes: torch.Tensor
    target_scores: torch.Tensor
    fg_mask: torch.Tensor
    target_gt_idx: torch.Tensor
    anchor_points: torch.Tensor
    stride_tensor: torch.Tensor
    gt_labels: torch.Tensor
    gt_bboxes: torch.Tensor
    mask_gt: torch.Tensor
    image_size: tuple[int, int]


@dataclass
class DetectionLossResult:
    total: torch.Tensor
    box: torch.Tensor
    classification: torch.Tensor
    dfl: torch.Tensor


@dataclass
class ClasswiseQueryLoss:
    losses: dict[int, torch.Tensor]
    valid: dict[int, bool]
    invalid_reason: dict[int, str]
    positive_count: dict[int, int]
    target_score_mass: dict[int, float]
    assignment: ReferenceAssignment


@dataclass
class YOLOVirtualTrajectory:
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    manifest: ParameterManifest
    step_losses: list[dict[str, float]]
    parameter_delta_norms: list[float]
    step_times_seconds: list[float]
    steps: int
    learning_rate: float
    optimizer: str
    momentum: float
    weight_decay: float
    create_graph: bool


class YOLOFunctionalAdapter:
    backend = "real_ultralytics_yolo"
    virtual_buffer_mode = "cloned"
    functional_call_backend = "torch.func.functional_call"

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        target_class_id: int = 14,
        selected_neck_indices: Sequence[int] = (15, 18, 21),
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.target_class_id = int(target_class_id)
        self.selected_neck_indices = tuple(int(index) for index in selected_neck_indices)
        if not hasattr(model, "model") or not model.model:
            raise TypeError("expected an Ultralytics DetectionModel with a non-empty model sequence")
        self.detect_index = len(model.model) - 1
        self.detect_module = model.model[self.detect_index]
        if type(self.detect_module).__name__ != "Detect":
            raise TypeError(f"final module is not Detect: {type(self.detect_module).__name__}")
        if int(self.detect_module.nc) != self.num_classes:
            raise ValueError(
                f"class count mismatch: adapter={self.num_classes}, detect_head={self.detect_module.nc}"
            )
        self._module_names = {id(module): name for name, module in model.named_modules()}
        self.detect_prefix = self._module_name(self.detect_module)
        self.classification_prefix = self._module_name(self.detect_module.cv3)
        self.box_prefix = self._module_name(self.detect_module.cv2)
        self.dfl_prefix = self._module_name(self.detect_module.dfl)
        self._fixed_parameter_names = frozenset(self._parameter_names_for_module(self.detect_module.dfl))
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.criterion = self.model.init_criterion()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str | torch.device = "cuda:0",
        num_classes: int = 20,
        target_class_id: int = 14,
        selected_neck_indices: Sequence[int] = (15, 18, 21),
    ) -> "YOLOFunctionalAdapter":
        wrapper = YOLO(str(Path(checkpoint)))
        model = wrapper.model.to(device)
        sparse_args = dict(model.args) if isinstance(model.args, dict) else dict(vars(model.args))
        train_args = wrapper.ckpt.get("train_args", {}) if isinstance(wrapper.ckpt, dict) else {}
        for name in ("box", "cls", "dfl"):
            if name in train_args:
                sparse_args[name] = train_args[name]
        model.args = get_cfg(overrides=sparse_args)
        if len(wrapper.names) != int(num_classes):
            raise ValueError(f"checkpoint class count mismatch: expected={num_classes}, actual={len(wrapper.names)}")
        return cls(model, num_classes, target_class_id, selected_neck_indices)

    def _module_name(self, module: torch.nn.Module) -> str:
        name = self._module_names.get(id(module))
        if name is None:
            raise RuntimeError(f"module is not registered in model: {type(module).__name__}")
        return name

    def _parameter_names_for_module(self, module: torch.nn.Module) -> tuple[str, ...]:
        prefix = self._module_name(module)
        local_names = {name for name, _ in module.named_parameters(recurse=True)}
        return tuple(f"{prefix}.{name}" if prefix and name else prefix or name for name in local_names)

    def base_parameters(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach() for name, parameter in self.model.named_parameters()}

    def clone_buffers(self, source: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        values = source if source is not None else dict(self.model.named_buffers())
        return {name: value.detach().clone() for name, value in values.items()}

    def select_parameter_names(self, mode: str) -> tuple[str, ...]:
        all_names = tuple(name for name, _ in self.model.named_parameters())
        eligible = set(all_names) - set(self._fixed_parameter_names)
        if mode == "classification_head_only":
            selected = set(self._parameter_names_for_module(self.detect_module.cv3))
        elif mode == "detection_head":
            selected = set(self._parameter_names_for_module(self.detect_module)) - set(self._fixed_parameter_names)
        elif mode == "selected_neck_and_head":
            modules = []
            for index in (*self.selected_neck_indices, self.detect_index):
                if index < 0 or index >= len(self.model.model):
                    raise IndexError(f"selected model layer index is unavailable: {index}")
                modules.append(self.model.model[index])
            selected = set().union(*(set(self._parameter_names_for_module(module)) for module in modules))
            selected -= set(self._fixed_parameter_names)
        elif mode == "full_model":
            selected = eligible
        else:
            raise ValueError(f"unknown fast parameter mode: {mode}")
        unknown = selected - set(all_names)
        if unknown:
            raise RuntimeError(f"parameter selection produced unknown names: {sorted(unknown)}")
        ordered = tuple(name for name in all_names if name in selected)
        if not ordered:
            raise RuntimeError(f"fast parameter mode selected no parameters: {mode}")
        return ordered

    def parameter_manifest(self, mode: str) -> ParameterManifest:
        parameters = dict(self.model.named_parameters())
        selected = self.select_parameter_names(mode)
        selected_set = set(selected)
        omitted = tuple(name for name in parameters if name not in selected_set)
        selected_parameters = sum(parameters[name].numel() for name in selected)
        eligible_parameters = sum(
            parameter.numel() for name, parameter in parameters.items() if name not in self._fixed_parameter_names
        )
        digest = hashlib.sha256()
        for name in selected:
            value = parameters[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return ParameterManifest(
            mode=mode,
            selected_names=selected,
            omitted_names=omitted,
            selected_tensors=len(selected),
            selected_parameters=selected_parameters,
            eligible_parameters=eligible_parameters,
            coverage_percent=100.0 * selected_parameters / max(eligible_parameters, 1),
            selected_hash=digest.hexdigest(),
        )

    def hash_base_state(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def initial_functional_state(
        self, mode: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], ParameterManifest]:
        manifest = self.parameter_manifest(mode)
        selected = set(manifest.selected_names)
        parameters = {
            name: value.detach().clone().requires_grad_(name in selected)
            for name, value in self.model.named_parameters()
        }
        return parameters, self.clone_buffers(), manifest

    def validate_fast_parameters(
        self, parameters: Mapping[str, torch.Tensor], manifest: ParameterManifest
    ) -> None:
        base = dict(self.model.named_parameters())
        if set(parameters) != set(base):
            raise RuntimeError("fast parameter keys do not exactly match base model parameter keys")
        for name, value in parameters.items():
            if value.shape != base[name].shape:
                raise RuntimeError(f"fast parameter shape mismatch: {name}")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite fast parameter: {name}")
            if name in manifest.selected_names and not value.requires_grad:
                raise RuntimeError(f"selected fast parameter is detached: {name}")

    def forward(
        self,
        images: torch.Tensor,
        parameters: Mapping[str, torch.Tensor] | None = None,
        buffers: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        active_parameters = dict(parameters) if parameters is not None else self.base_parameters()
        active_buffers = dict(buffers) if buffers is not None else self.clone_buffers()
        output = functional_call(self.model, (active_parameters, active_buffers), (images,), strict=True)
        if not isinstance(output, dict) or not {"boxes", "scores", "feats"}.issubset(output):
            raise RuntimeError("real YOLO training forward returned an unexpected schema")
        return output

    def compute_detection_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
    ) -> DetectionLossResult:
        components, _ = functional_call(
            self.model, (dict(parameters), dict(buffers)), (dict(batch),), strict=True
        )
        if components.numel() != 3 or not torch.isfinite(components).all():
            raise RuntimeError(f"unexpected native detection loss shape: {tuple(components.shape)}")
        return DetectionLossResult(
            total=components.sum(),
            box=components[0],
            classification=components[1],
            dfl=components[2],
        )

    def virtual_update(
        self,
        support_batch: Mapping[str, torch.Tensor],
        steps: int,
        learning_rate: float,
        mode: str,
        create_graph: bool = True,
    ) -> YOLOVirtualTrajectory:
        if int(steps) <= 0:
            raise ValueError("virtual update steps must be positive")
        parameters, buffers, manifest = self.initial_functional_state(mode)
        base = self.base_parameters()
        losses: list[dict[str, float]] = []
        delta_norms: list[float] = []
        times: list[float] = []
        for _ in range(int(steps)):
            started = time.perf_counter()
            loss = self.compute_detection_loss(support_batch, parameters, buffers)
            selected_values = [parameters[name] for name in manifest.selected_names]
            gradients = torch.autograd.grad(
                loss.total, selected_values, create_graph=bool(create_graph), allow_unused=False
            )
            updated = dict(parameters)
            for name, gradient in zip(manifest.selected_names, gradients):
                updated[name] = parameters[name] - float(learning_rate) * gradient
            parameters = updated
            squared_delta = sum(
                (parameters[name] - base[name]).float().square().sum()
                for name in manifest.selected_names
            )
            losses.append(
                {
                    "total": float(loss.total.detach()),
                    "box": float(loss.box.detach()),
                    "classification": float(loss.classification.detach()),
                    "dfl": float(loss.dfl.detach()),
                }
            )
            delta_norms.append(float(torch.sqrt(squared_delta.detach())))
            times.append(time.perf_counter() - started)
        self.validate_fast_parameters(parameters, manifest)
        return YOLOVirtualTrajectory(
            parameters=parameters,
            buffers=buffers,
            manifest=manifest,
            step_losses=losses,
            parameter_delta_norms=delta_norms,
            step_times_seconds=times,
            steps=int(steps),
            learning_rate=float(learning_rate),
            optimizer="sgd",
            momentum=0.0,
            weight_decay=0.0,
            create_graph=bool(create_graph),
        )

    def extract_tal_diagnostics(
        self, raw_predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
    ) -> ReferenceAssignment:
        criterion = self.model.criterion
        pred_distri = raw_predictions["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = raw_predictions["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(raw_predictions["feats"], criterion.stride, 0.5)
        batch_size = pred_scores.shape[0]
        image_hw = (
            torch.tensor(raw_predictions["feats"][0].shape[2:], device=pred_scores.device, dtype=pred_scores.dtype)
            * criterion.stride[0]
        )
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1
        )
        targets = criterion.preprocess(
            targets.to(pred_scores.device), batch_size, scale_tensor=image_hw[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)
        pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return ReferenceAssignment(
            target_labels=target_labels.detach(),
            target_bboxes=target_bboxes.detach(),
            target_scores=target_scores.detach(),
            fg_mask=fg_mask.detach(),
            target_gt_idx=target_gt_idx.detach(),
            anchor_points=anchor_points.detach(),
            stride_tensor=stride_tensor.detach(),
            gt_labels=gt_labels.detach(),
            gt_bboxes=gt_bboxes.detach(),
            mask_gt=mask_gt.detach(),
            image_size=(int(image_hw[0].item()), int(image_hw[1].item())),
        )

    def reference_assignment(
        self,
        query_images: torch.Tensor,
        query_batch: Mapping[str, torch.Tensor],
    ) -> ReferenceAssignment:
        raw = self.forward(query_images, self.base_parameters(), self.clone_buffers())
        return self.extract_tal_diagnostics(raw, query_batch)

    def compute_classwise_query_loss(
        self,
        query_images: torch.Tensor,
        query_batch: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
        reference: ReferenceAssignment | None = None,
        assignment_mode: str = "fixed_clean_reference",
    ) -> ClasswiseQueryLoss:
        raw = self.forward(query_images, parameters, self.clone_buffers(buffers))
        if assignment_mode == "fixed_clean_reference":
            if reference is None:
                raise ValueError("fixed_clean_reference mode requires a reference assignment")
            assignment = reference
        elif assignment_mode == "recomputed":
            assignment = self.extract_tal_diagnostics(raw, query_batch)
        else:
            raise ValueError(f"unknown query assignment mode: {assignment_mode}")
        scores = raw["scores"].permute(0, 2, 1).contiguous()
        losses: dict[int, torch.Tensor] = {}
        valid: dict[int, bool] = {}
        reasons: dict[int, str] = {}
        counts: dict[int, int] = {}
        masses: dict[int, float] = {}
        for class_id in range(self.num_classes):
            targets = assignment.target_scores[..., class_id].to(dtype=scores.dtype)
            positive = targets > 0
            count = int(positive.sum().item())
            mass = targets[positive].sum()
            counts[class_id] = count
            masses[class_id] = float(mass.detach()) if count else 0.0
            if count == 0 or float(mass.detach()) <= 0.0:
                valid[class_id] = False
                reasons[class_id] = "no_reference_positive"
                continue
            losses[class_id] = F.binary_cross_entropy_with_logits(
                scores[..., class_id][positive], targets[positive], reduction="sum"
            ) / mass.clamp_min(1e-12)
            valid[class_id] = True
            reasons[class_id] = ""
        return ClasswiseQueryLoss(losses, valid, reasons, counts, masses, assignment)

    def clean_poison_states_independent(
        self, clean: YOLOVirtualTrajectory, poison: YOLOVirtualTrajectory
    ) -> bool:
        for name in clean.manifest.selected_names:
            if clean.parameters[name].data_ptr() == poison.parameters[name].data_ptr():
                return False
        for name in clean.buffers:
            if clean.buffers[name].data_ptr() == poison.buffers[name].data_ptr():
                return False
        return True
