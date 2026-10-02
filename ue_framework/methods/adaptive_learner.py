"""One differentiable detector update for the B2 perturbation objective."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.func import functional_call
from ultralytics.cfg import get_cfg

from .legacy_kproto_ret import non_target_retention_loss


class AdaptiveLearner:
    def __init__(self, model, feature_layer, target_class_id, imgsz, inner_lr):
        if inner_lr <= 0:
            raise ValueError("inner_lr must be positive")
        self.model = model
        self.model.eval()
        if isinstance(self.model.args, dict):
            self.model.args = get_cfg(overrides=self.model.args)
        self.model.criterion = self.model.init_criterion()
        self.feature_module = dict(model.named_modules())[feature_layer]
        self.target_class_id = target_class_id
        self.imgsz = imgsz
        self.inner_lr = inner_lr
        # VOC teacher checkpoints may mark every tensor frozen. Restore native
        # detector eligibility on this separate learner, keeping DFL integral fixed.
        head = model.model[-1]
        fixed = {id(p) for p in head.dfl.parameters()} if hasattr(head, "dfl") else set()
        for p in model.parameters():
            p.requires_grad_(id(p) not in fixed)
        self.trainable = tuple(name for name, p in model.named_parameters() if p.requires_grad)
        if not self.trainable:
            raise ValueError("Adaptive detector has no trainable parameters")
        for p in model.parameters():
            p.requires_grad_(False)

    @contextmanager
    def _mode(self, training):
        states = [(module, module.training) for module in self.model.modules()]
        self.model.train(training)
        # Virtual updates have no running-stat state transition.
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        try:
            yield
        finally:
            for module, state in states:
                module.training = state

    def _initial_state(self):
        params = {
            name: value.detach().clone().requires_grad_(name in self.trainable)
            for name, value in self.model.named_parameters()
        }
        buffers = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        return params, buffers

    def _letterbox(self, image):
        _, _, h, w = image.shape
        ratio = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top, left = (self.imgsz - new_h) // 2, (self.imgsz - new_w) // 2
        result = F.pad(
            resized, (left, self.imgsz - new_w - left, top, self.imgsz - new_h - top),
            value=0.447,
        )
        return result, ratio, left, top

    def _batch(self, image, items):
        image, ratio, left, top = self._letterbox(image)
        boxes, classes, indices = [], [], []
        for index, (original, annotations, _) in enumerate(items):
            h, w = original.shape[:2]
            for ann in annotations:
                if ann.get("bbox") is None:
                    continue
                cx, cy, bw, bh = map(float, ann["bbox"])
                boxes.append((
                    (cx * w * ratio + left) / self.imgsz,
                    (cy * h * ratio + top) / self.imgsz,
                    bw * w * ratio / self.imgsz,
                    bh * h * ratio / self.imgsz,
                ))
                classes.append([float(ann["cls"])])
                indices.append(index)
        device = image.device
        return {
            "img": image,
            "bboxes": torch.tensor(boxes, device=device, dtype=torch.float32).reshape(-1, 4),
            "cls": torch.tensor(classes, device=device, dtype=torch.float32).reshape(-1, 1),
            "batch_idx": torch.tensor(indices, device=device, dtype=torch.long),
        }

    def _predict(self, images, params, buffers, *, grad):
        capture = {}

        def hook(_module, _inputs, output):
            capture["feature"] = output

        handle = self.feature_module.register_forward_hook(hook)
        try:
            with self._mode(training=False):
                if grad:
                    result = functional_call(self.model, (params, buffers), (images,), strict=True)
                else:
                    with torch.no_grad():
                        result = functional_call(self.model, (params, buffers), (images,), strict=True)
        finally:
            handle.remove()
        decoded = result[0] if isinstance(result, (tuple, list)) else result
        if decoded.ndim != 3 or decoded.shape[1] < 4 + self.target_class_id + 1:
            raise ValueError("Expected decoded YOLO predictions [B, 4+C, N]")
        return decoded, capture["feature"]

    @staticmethod
    def _iou(boxes, gt):
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)
        gcx, gcy, gw, gh = gt.unbind(-1)
        gt_xyxy = torch.stack((gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2), -1)
        left_top = torch.maximum(xyxy[:, None, :2], gt_xyxy[None, :, :2])
        right_bottom = torch.minimum(xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
        inter = (right_bottom - left_top).clamp_min(0).prod(-1)
        area_boxes = (xyxy[:, 2:] - xyxy[:, :2]).clamp_min(0).prod(-1)
        area_gt = (gt_xyxy[:, 2:] - gt_xyxy[:, :2]).clamp_min(0).prod(-1)
        return inter / (area_boxes[:, None] + area_gt[None] - inter).clamp_min(1e-6)

    def _target_evidence(self, decoded, batch):
        values = []
        for index in range(decoded.shape[0]):
            mask = (batch["batch_idx"] == index) & (
                batch["cls"].flatten().long() == self.target_class_id
            )
            if not bool(mask.any()):
                continue
            boxes = decoded[index, :4].T
            probs = decoded[index, 4 + self.target_class_id].clamp(1e-6, 1.0)
            gt = batch["bboxes"][mask] * self.imgsz
            alignment = probs[:, None].sqrt() * self._iou(boxes, gt).clamp_min(1e-6).pow(6)
            # One task-aligned foreground evidence value for each target GT.
            values.extend(alignment.topk(min(100, alignment.shape[0]), dim=0).values.mean(0).unbind())
        if not values:
            raise ValueError("Adaptive query has no target GT")
        return torch.stack(values).mean()

    def losses(self, clean, poison, items, non_target_masks, pool):
        poison_batch = self._batch(poison, items)
        clean_batch = self._batch(clean, items)
        params, buffers = self._initial_state()
        with self._mode(training=True):
            inner_loss, _ = functional_call(self.model, (params, buffers), (poison_batch,), strict=True)
        inner_loss = inner_loss.sum()
        if not torch.isfinite(inner_loss):
            raise FloatingPointError("Non-finite adaptive detection loss")
        gradients = torch.autograd.grad(
            inner_loss, [params[name] for name in self.trainable],
            create_graph=True, allow_unused=False,
        )
        updated = {**params, **{
            name: params[name] - self.inner_lr * gradient
            for name, gradient in zip(self.trainable, gradients)
        }}
        update_norm = torch.stack([gradient.detach().norm() for gradient in gradients]).norm() * self.inner_lr
        before_pred, before_feat = self._predict(clean_batch["img"], params, buffers, grad=False)
        after_pred, after_feat = self._predict(clean_batch["img"], updated, buffers, grad=True)
        with torch.no_grad():
            poison_pred, _ = self._predict(poison_batch["img"], updated, buffers, grad=False)
        before_score = self._target_evidence(before_pred, clean_batch)
        after_score = self._target_evidence(after_pred, clean_batch)
        poison_score = self._target_evidence(poison_pred, clean_batch)
        before_nt, after_nt = [], []
        for index, masks in enumerate(non_target_masks):
            before_nt.extend(pool(before_feat[index:index + 1], masks))
            after_nt.extend(pool(after_feat[index:index + 1], masks))
        if before_nt:
            nt_loss = non_target_retention_loss(torch.stack(after_nt), torch.stack(before_nt))
        else:
            nt_loss = after_feat.sum() * 0.0
        metrics = {
            "L_adapt_target": after_score,
            "L_adapt_nt": nt_loss,
            "target_score_clean_before": before_score,
            "target_score_clean_after": after_score,
            "target_score_poison_after": poison_score,
            "cue_gap": poison_score - after_score,
            "cue_ratio": poison_score / (after_score + 1e-6),
            "inner_detection_loss": inner_loss,
            "inner_update_norm": update_norm,
        }
        return after_score, nt_loss, metrics
