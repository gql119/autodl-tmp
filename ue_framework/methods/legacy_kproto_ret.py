"""Frozen-surrogate, universal Fourier perturbation with scene-conditioned prototypes."""

import csv
import hashlib
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from .alce_losses import masked_prototype
from .tausb_universal import TAUSBMaskGenerator, TAUSBUniversalTrainer

CACHE_VERSION = 1


def select_prototype(context, prototypes):
    similarities = F.normalize(context.detach(), dim=-1) @ F.normalize(prototypes.detach(), dim=-1).T
    indices = similarities.argmax(dim=-1)
    selected = prototypes.detach()[indices]
    return indices, selected, similarities.gather(1, indices[:, None]).squeeze(1)


def non_target_retention_loss(poisoned, clean):
    if poisoned.numel():
        return (poisoned - clean.detach()).square().sum(dim=-1).mean()
    return poisoned.sum() * 0.0


def kproto_losses(target_poison, target_clean, selected_bg, non_target_poison, non_target_clean, margin):
    bg_similarity = F.cosine_similarity(target_poison, selected_bg.detach(), dim=-1)
    clean_similarity = F.cosine_similarity(target_poison, target_clean.detach(), dim=-1)
    l_bg = (1.0 - bg_similarity).mean()
    l_rel = F.relu(float(margin) + clean_similarity - bg_similarity).mean()
    l_ret = non_target_retention_loss(non_target_poison, non_target_clean)
    return l_bg, l_rel, l_ret, clean_similarity.mean(), bg_similarity.mean()


def deterministic_kmeans(vectors, k, seed, iterations=30):
    x = F.normalize(vectors.float().cpu(), dim=-1)
    if x.shape[0] < k:
        raise ValueError(f"Only {x.shape[0]} valid background contexts for K={k}")
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(x.shape[0], (1,), generator=generator))
    chosen = [first]
    for _ in range(1, k):
        distance = 1.0 - (x @ x[chosen].T).max(dim=1).values
        distance[chosen] = -1.0
        chosen.append(int(distance.argmax()))
    centers = x[chosen]
    for _ in range(iterations):
        labels = (x @ centers.T).argmax(dim=1)
        updated = torch.stack([x[labels == i].mean(dim=0) if (labels == i).any() else centers[i] for i in range(k)])
        updated = F.normalize(updated, dim=-1)
        if torch.allclose(updated, centers, atol=1e-6):
            break
        centers = updated
    return centers


def _rectangle(shape, bbox, scale=1.0):
    h, w = shape
    cx, cy, bw, bh = map(float, bbox)
    x1 = max(0, int((cx - bw * scale / 2) * w))
    x2 = min(w, int(np.ceil((cx + bw * scale / 2) * w)))
    y1 = max(0, int((cy - bh * scale / 2) * h))
    y2 = min(h, int(np.ceil((cy + bh * scale / 2) * h)))
    result = torch.zeros((1, 1, h, w), dtype=torch.float32)
    if x2 > x1 and y2 > y1:
        result[:, :, y1:y2, x1:x2] = 1.0
    return result


def _local_context_mask(instance, r_inner, r_outer):
    """Binary equivalent of the legacy square max-pool annulus, using CPU morphology."""
    inner_radius = max(1, int(r_inner))
    outer_radius = max(inner_radius + 1, int(r_outer))
    source = instance[0, 0].numpy().astype(np.uint8)
    inner = cv2.dilate(source, np.ones((2 * inner_radius + 1,) * 2, dtype=np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    outer = cv2.dilate(source, np.ones((2 * outer_radius + 1,) * 2, dtype=np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return torch.from_numpy((outer.astype(np.float32) - inner.astype(np.float32)).clip(0, 1))[None, None]


def _feature_mask(mask, feature, image_shape, imgsz):
    """Apply the same resize and padding geometry as BasePoisonGenerator._letterbox_tensor."""
    h, w = image_shape
    if (h, w) != (imgsz, imgsz):
        ratio = min(imgsz / h, imgsz / w)
        new_h, new_w = max(1, round(h * ratio)), max(1, round(w * ratio))
        mask = F.interpolate(mask, size=(new_h, new_w), mode="nearest")
        pad_h, pad_w = imgsz - new_h, imgsz - new_w
        mask = F.pad(mask, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return F.interpolate(mask.to(device=feature.device, dtype=feature.dtype), size=feature.shape[-2:], mode="area")


class _StrictTargetSupport:
    def _bbox_half_support(self, image_shape, annotations):
        h, w = image_shape[:2]
        target = np.zeros((h, w), dtype=bool)
        other = np.zeros((h, w), dtype=bool)
        for ann in annotations:
            if ann.get("bbox") is None:
                continue
            box = _rectangle((h, w), ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann.get("cls", -1)) == self.target_class_id:
                target |= box
            else:
                other |= box
        strength = target.astype(np.float32)
        strength[target & other] = 0.5
        return strength

    def _jnd_gain(self, img, current_floor):
        # Keep the inherited +0.0 to +0.1 legacy schedule while making its
        # starting floor configurable for both optimization and generation.
        floor = min(self.jnd_ceiling, self.jnd_floor + float(current_floor) - 0.4)
        return super()._jnd_gain(img, current_floor=floor)

    def _build_support(self, image_shape, annotations, support_type="mask", ring_width=4, image_path=None):
        h, w = image_shape[:2]
        if self.support_mode == "bbox_half_overlap":
            if support_type != "mask":
                raise ValueError(f"Expected mask support argument, got {support_type}")
            support = self._bbox_half_support(image_shape, annotations)
            return support, np.zeros((h, w), dtype=np.float32), (
                "bbox_half_overlap" if np.any(support) else "no_target"
            )
        if self.strict_instance_mask:
            if support_type != "mask":
                raise ValueError(f"Expected mask support, got {support_type}")
            if not any(int(ann.get("cls", -1)) == self.target_class_id for ann in annotations):
                zero = np.zeros((h, w), dtype=np.float32)
                return zero, zero, "no_target"
            mask_path = self._resolve_instance_mask_path(image_path)
            if mask_path is None:
                raise FileNotFoundError(f"Pre-baked target mask missing for {image_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (h, w) or (mask > 2).any():
                raise ValueError(f"Invalid pre-baked target mask: {mask_path}")
            inner = (mask == 1).astype(np.float32)
            if inner.sum() <= 10:
                raise ValueError(f"Empty pre-baked target core: {mask_path}")
            ring = (mask == 2).astype(np.float32)
            source = "prebaked"
        else:
            inner, ring, source = super()._build_support(image_shape, annotations, support_type, ring_width, image_path)
        for ann in annotations:
            if int(ann.get("cls", -1)) != self.target_class_id and ann.get("bbox") is not None:
                inner *= (1.0 - _rectangle((h, w), ann["bbox"])[0, 0].numpy())
        if self.strict_instance_mask and inner.sum() <= 10:
            return inner, ring, "prebaked_excluded"
        return inner, ring, source

    def _compose_delta_batched(self, img_t, inner_t, ring_t, coords, fourier_coeff, suppress_small, current_epoch=0):
        if self.support_mode != "bbox_half_overlap":
            return super()._compose_delta_batched(
                img_t, inner_t, torch.zeros_like(ring_t), coords, fourier_coeff, suppress_small, current_epoch
            )
        binary = (inner_t > 0).to(inner_t.dtype)
        raw, base, _, _, jnd = super()._compose_delta_batched(
            img_t, binary, torch.zeros_like(ring_t), coords, fourier_coeff, suppress_small, current_epoch
        )
        # Two uint8 levels before halving guarantee at least one changed level
        # in overlap after nearest-integer PNG quantization.
        minimum = 2.01 / 255.0
        sign = torch.where(base < 0, -torch.ones_like(base), torch.ones_like(base))
        sign = torch.where((img_t <= 0) & (sign < 0), -sign, sign)
        sign = torch.where((img_t >= 1) & (sign > 0), -sign, sign)
        floored = sign * torch.maximum(base.abs(), base.new_tensor(minimum))
        base = base + (floored - base).detach()
        delta = base * inner_t
        adv = (img_t + delta).clamp(0.0, 1.0)
        return raw * inner_t, delta, adv, binary, jnd


class LegacyKProtoGenerator(_StrictTargetSupport, TAUSBMaskGenerator):
    def __init__(self, cfg, method_cfg, device, surrogate, global_params_path):
        super().__init__(cfg, method_cfg, device, surrogate, global_params_path)
        self.support_mode = str(method_cfg.get("support_mode", "prebaked"))
        metadata = torch.load(global_params_path, map_location="cpu", weights_only=True)
        if metadata.get("method") != "legacy_kproto_ret":
            raise ValueError("Global parameters do not belong to legacy_kproto_ret")
        if abs(float(metadata["eps"]) - self.eps) > 1e-9:
            raise ValueError("Global parameter epsilon differs from current configuration")
        if metadata["feature_layer"] != method_cfg["kproto"]["feature_layer"]:
            raise ValueError("Global parameter feature layer differs from current configuration")
        if metadata.get("support_mode", "prebaked") != self.support_mode:
            raise ValueError("Global parameter support mode differs from current configuration")


class LegacyKProtoTrainer(_StrictTargetSupport, TAUSBUniversalTrainer):
    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.support_mode = str(method_cfg.get("support_mode", "prebaked"))
        kp = method_cfg["kproto"]
        loss = method_cfg["loss"]
        self.feature_layer = str(kp["feature_layer"])
        if self.feature_layer not in self.shape_layers:
            raise ValueError(f"feature_layer must be one of {self.shape_layers}")
        self.num_prototypes = int(kp["num_prototypes"])
        self.margin = float(kp["margin"])
        self.prototype_seed = int(kp["seed"])
        self.context_inner = int(kp["context_inner"])
        self.context_outer = int(kp["context_outer"])
        self.prototype_cache = str(kp.get("prototype_cache", ""))
        self.lambda_bg = float(loss["lambda_bg"])
        self.lambda_rel = float(loss["lambda_rel"])
        self.lambda_ret = float(loss["lambda_ret"])
        self.max_images = int(kp.get("max_images", 0))
        self.max_steps = int(kp.get("max_steps", 0))

    def _feature(self, image, grad):
        self._clear_multi_features()
        if grad:
            self._forward_raw(image)
        else:
            with torch.no_grad():
                self._forward_raw(image)
        return self.multi_features[self.feature_layer]

    def _masks(self, annotations, support, image_shape, feature, padded_shape=None):
        h, w = image_shape
        padded_shape = padded_shape or image_shape

        def resize(mask):
            if padded_shape != image_shape:
                mask = F.pad(mask, (0, padded_shape[1] - w, 0, padded_shape[0] - h))
            return _feature_mask(mask, feature, padded_shape, self.imgsz)

        objects = torch.zeros((1, 1, h, w), dtype=torch.float32)
        for ann in annotations:
            if ann.get("bbox") is not None:
                objects = torch.maximum(objects, _rectangle((h, w), ann["bbox"]))
        target, context, non_target = [], [], []
        support = torch.from_numpy((support > 0).astype(np.float32))[None, None]
        for ann in annotations:
            if ann.get("bbox") is None:
                continue
            box = _rectangle((h, w), ann["bbox"])
            if int(ann.get("cls", -1)) == self.target_class_id:
                instance = box * support
                ctx = _local_context_mask(instance, self.context_inner, self.context_outer)
                ctx = ctx * (1.0 - objects)
                target.append(resize(instance))
                context.append(resize(ctx))
            else:
                non_target.append(resize(box))
        return target, context, non_target

    def _pooled(self, feature, masks):
        values = []
        for mask in masks:
            value, valid = masked_prototype(feature, mask, min_pixels=1.0)
            if bool(valid.item()):
                values.append(value[0])
        return values

    def _cache_key(self, train_img_dir, train_label_dir):
        checkpoint = self.cfg["surrogate"]["ckpt"]
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        inputs = [CACHE_VERSION, self.support_mode, digest.hexdigest(),
                  os.path.realpath(train_img_dir), os.path.realpath(train_label_dir),
                  self.feature_layer, self.num_prototypes, self.prototype_seed,
                  self.context_inner, self.context_outer, self.max_images]
        return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()

    def _build_bank(self, images, label_dir, cache_path, cache_key):
        if os.path.isfile(cache_path):
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
            if saved.get("cache_key") == cache_key:
                return saved["prototypes"].to(self.device)
        contexts = []
        for path in images:
            image = load_image_rgb_float(path)
            annotations = read_yolo_annotations(label_path_for_image(path, label_dir))
            support, _, _ = self._build_support(image.shape, annotations, image_path=path)
            if support.sum() <= 10:
                continue
            tensor = self._to_tensor(image)
            feature = self._feature(tensor, grad=False)
            _, masks, _ = self._masks(annotations, support, image.shape[:2], feature)
            contexts.extend(value.detach().cpu() for value in self._pooled(feature, masks))
        if not contexts:
            raise RuntimeError("No valid object-excluded context features for prototype bank")
        prototypes = deterministic_kmeans(torch.stack(contexts), self.num_prototypes, self.prototype_seed)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"cache_key": cache_key, "prototypes": prototypes, "count": len(contexts)}, cache_path)
        return prototypes.to(self.device)

    def _one_image_loss(self, image, annotations, support, prototypes):
        tensor = self._to_tensor(image)
        support_t = torch.from_numpy(support).float()[None, None].to(self.device)
        raw, delta, poison, _, _ = self._compose_delta_batched(
            tensor, support_t, torch.zeros_like(support_t), self._active_coords,
            self.fourier_coeff[self._active_idx], self.suppress_small, current_epoch=self._epoch
        )
        clean_aug, poison_aug = self._apply_shared_eot_pair_batched(tensor, poison)
        clean = self._feature(clean_aug, grad=False).detach()
        poisoned = self._feature(poison_aug, grad=True)
        target_masks, context_masks, nt_masks = self._masks(annotations, support, image.shape[:2], poisoned)
        target_clean, target_poison, context = [], [], []
        for target_mask, context_mask in zip(target_masks, context_masks):
            h0 = self._pooled(clean, [target_mask])
            hp = self._pooled(poisoned, [target_mask])
            ctx = self._pooled(clean, [context_mask])
            if h0 and hp and ctx:
                target_clean.append(h0[0]); target_poison.append(hp[0]); context.append(ctx[0])
        if not target_poison:
            return None
        target_clean = torch.stack(target_clean)
        target_poison = torch.stack(target_poison)
        context = torch.stack(context)
        indices, selected, similarity = select_prototype(context, prototypes)
        nt_clean = self._pooled(clean, nt_masks)
        nt_poison = self._pooled(poisoned, nt_masks)
        nt_clean_t = torch.stack(nt_clean) if nt_clean else poisoned.new_zeros((0, poisoned.shape[1]))
        nt_poison_t = torch.stack(nt_poison) if nt_poison else poisoned.new_zeros((0, poisoned.shape[1]))
        bg, rel, ret, cos_clean, cos_bg = kproto_losses(
            target_poison, target_clean, selected, nt_poison_t, nt_clean_t, self.margin
        )
        tv = self._tv_loss(raw)
        budget = F.relu(raw.abs().max() - self.eps)
        weighted_bg, weighted_rel, weighted_ret = self.lambda_bg * bg, self.lambda_rel * rel, self.lambda_ret * ret
        total = weighted_bg + weighted_rel + weighted_ret + self.lambda_tv * tv + self.lambda_budget * budget
        metrics = {
            "L_BG": bg, "L_rel": rel, "L_ret": ret, "weighted_L_BG": weighted_bg,
            "weighted_L_rel": weighted_rel, "weighted_L_ret": weighted_ret,
            "L_tv": tv, "L_budget": budget, "L_total": total,
            "cos_target_clean": cos_clean, "cos_target_bg": cos_bg,
            "relative_gap": cos_bg - cos_clean, "non_target_feature_drift": ret,
            "target_feature_drift": (target_poison - target_clean).square().sum(dim=-1).mean(),
            "selected_proto_id": indices[0].float(),
            "selected_proto_ids": ",".join(str(int(i)) for i in indices.detach().cpu().tolist()),
            "context_proto_similarity": similarity.mean(),
            "context_feature_norm": context.norm(dim=-1).mean(),
            "max_abs_delta": delta.abs().max(), "mean_abs_delta": delta.abs().mean(),
            "saturation_ratio": (delta.abs() >= self.eps - 1e-6).float().mean(),
            "support_area_ratio": (support_t > 0).float().mean(),
            "outside_support_max": (delta * (support_t == 0)).abs().max(),
        }
        return total, metrics

    def _batch_loss(self, items, prototypes):
        """One surrogate clean/poisoned forward for a legacy-style padded batch."""
        pad_h = (max(image.shape[0] for image, _, _ in items) + 31) // 32 * 32
        pad_w = (max(image.shape[1] for image, _, _ in items) + 31) // 32 * 32
        images, supports = [], []
        for image, _, support in items:
            h, w = image.shape[:2]
            images.append(F.pad(torch.from_numpy(image).permute(2, 0, 1).float(),
                                (0, pad_w - w, 0, pad_h - h), value=0.5))
            supports.append(F.pad(torch.from_numpy(support)[None].float(),
                                  (0, pad_w - w, 0, pad_h - h)))
        tensor = torch.stack(images).to(self.device)
        support_t = torch.stack(supports).to(self.device)
        raw, delta, poison, _, _ = self._compose_delta_batched(
            tensor, support_t, torch.zeros_like(support_t), self._active_coords,
            self.fourier_coeff[self._active_idx], self.suppress_small, current_epoch=self._epoch
        )
        clean_aug, poison_aug = self._apply_shared_eot_pair_batched(tensor, poison)
        clean = self._feature(clean_aug, grad=False).detach()
        poisoned = self._feature(poison_aug, grad=True)
        image_losses, non_target_pairs = [], []
        target_drift, proto_ids, proto_similarity, context_norm = [], [], [], []
        for b, (image, annotations, support) in enumerate(items):
            h0_map, hp_map = clean[b:b + 1], poisoned[b:b + 1]
            targets, contexts, non_targets = self._masks(
                annotations, support, image.shape[:2], hp_map, (pad_h, pad_w)
            )
            h0, hp, ct = [], [], []
            for target_mask, context_mask in zip(targets, contexts):
                clean_target = self._pooled(h0_map, [target_mask])
                poison_target = self._pooled(hp_map, [target_mask])
                context = self._pooled(h0_map, [context_mask])
                if clean_target and poison_target and context:
                    h0.append(clean_target[0])
                    hp.append(poison_target[0])
                    ct.append(context[0])
            if not hp:
                continue
            h0, hp, ct = torch.stack(h0), torch.stack(hp), torch.stack(ct)
            indices, selected, similarity = select_prototype(ct, prototypes)
            nt0 = self._pooled(h0_map, non_targets)
            ntp = self._pooled(hp_map, non_targets)
            nt0 = torch.stack(nt0) if nt0 else hp_map.new_zeros((0, hp_map.shape[1]))
            ntp = torch.stack(ntp) if ntp else hp_map.new_zeros((0, hp_map.shape[1]))
            image_losses.append(kproto_losses(hp, h0, selected, ntp, nt0, self.margin))
            if ntp.numel():
                non_target_pairs.append((ntp, nt0))
            target_drift.append((hp - h0).square().sum(dim=-1).mean())
            proto_ids.extend(indices.detach().cpu().tolist())
            proto_similarity.append(similarity.mean())
            context_norm.append(ct.norm(dim=-1).mean())
        if not image_losses:
            return None
        bg, rel, _, cos_clean, cos_bg = [
            torch.stack([loss[i] for loss in image_losses]).mean() for i in range(5)
        ]
        if non_target_pairs:
            ret = non_target_retention_loss(
                torch.cat([pair[0] for pair in non_target_pairs]),
                torch.cat([pair[1] for pair in non_target_pairs]),
            )
        else:
            ret = poisoned.sum() * 0.0
        tv = self._tv_loss(raw)
        budget = F.relu(raw.abs().max() - self.eps)
        weighted_bg = self.lambda_bg * bg
        weighted_rel = self.lambda_rel * rel
        weighted_ret = self.lambda_ret * ret
        total = weighted_bg + weighted_rel + weighted_ret + self.lambda_tv * tv + self.lambda_budget * budget
        metrics = {
            "L_BG": bg, "L_rel": rel, "L_ret": ret,
            "weighted_L_BG": weighted_bg, "weighted_L_rel": weighted_rel,
            "weighted_L_ret": weighted_ret, "L_tv": tv, "L_budget": budget,
            "L_total": total, "cos_target_clean": cos_clean, "cos_target_bg": cos_bg,
            "relative_gap": cos_bg - cos_clean, "non_target_feature_drift": ret,
            "target_feature_drift": torch.stack(target_drift).mean(),
            "selected_proto_id": torch.tensor(float(proto_ids[0]), device=self.device),
            "selected_proto_ids": ",".join(str(int(i)) for i in proto_ids),
            "context_proto_similarity": torch.stack(proto_similarity).mean(),
            "context_feature_norm": torch.stack(context_norm).mean(),
            "max_abs_delta": delta.abs().max(), "mean_abs_delta": delta.abs().mean(),
            "saturation_ratio": (delta.abs() >= self.eps - 1e-6).float().mean(),
            "support_area_ratio": (support_t > 0).float().mean(),
            "outside_support_max": (delta * (support_t == 0)).abs().max(),
        }
        return total, metrics

    def train_universal(self, train_img_dir, train_label_dir, global_params_path,
                        diagnostics_csv_path, diagnostics_json_path, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        images = self._collect_target_images(train_img_dir, train_label_dir)
        if self.max_images:
            images = images[:self.max_images]
        if not images:
            raise RuntimeError("No target-class training images")
        if self.strict_instance_mask:
            for path in images:
                image = load_image_rgb_float(path)
                annotations = read_yolo_annotations(label_path_for_image(path, train_label_dir))
                self._build_support(image.shape, annotations, image_path=path)
        cache_path = self.prototype_cache or os.path.join(os.path.dirname(global_params_path), "background_prototypes.pt")
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(os.path.dirname(global_params_path), cache_path)
        prototypes = self._build_bank(images, train_label_dir, cache_path,
                                      self._cache_key(train_img_dir, train_label_dir))
        optimizer = torch.optim.Adam([{"params": [self.fourier_coeff], "lr": self.universal_lr_fourier}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.universal_epochs, eta_min=1e-5
        )
        assert all(not p.requires_grad for p in self.surrogate.parameters())
        os.makedirs(os.path.dirname(diagnostics_csv_path), exist_ok=True)
        step = 0
        support_sources = {}
        latest = {}
        with open(diagnostics_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = None
            for epoch in range(self.universal_epochs):
                self._epoch = epoch
                order = np.random.permutation(len(images))
                for start in range(0, len(order), self.universal_batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    self._active_idx = self._get_active_freq_indices(epoch, step)
                    self._active_coords = [tuple(self.freq_candidate_coords[int(i)]) for i in self._active_idx.cpu().tolist()]
                    items = []
                    for index in order[start:start + self.universal_batch_size]:
                        path = images[int(index)]
                        image = load_image_rgb_float(path)
                        annotations = read_yolo_annotations(label_path_for_image(path, train_label_dir))
                        support, _, source = self._build_support(image.shape, annotations, image_path=path)
                        support_sources[source] = support_sources.get(source, 0) + 1
                        if support.sum() <= 10:
                            continue
                        items.append((image, annotations, support))
                    batch_metrics = []
                    if items:
                        for _ in range(max(1, self.eot_samples)):
                            result = self._batch_loss(items, prototypes)
                            if result is None:
                                continue
                            loss, metrics = result
                            loss.backward()
                            batch_metrics.append({key: float(value.detach()) if torch.is_tensor(value) else value
                                                  for key, value in metrics.items()})
                    if not batch_metrics:
                        continue
                    self.fourier_coeff.grad.div_(len(batch_metrics))
                    grad_norm = float(self.fourier_coeff.grad.norm())
                    if self.enable_adaptive_freq_basis and epoch < self.adaptive_freq_warmup_epochs:
                        with torch.no_grad():
                            self.freq_score[self._active_idx] += self.fourier_coeff.grad[self._active_idx].abs().mean(dim=1)
                            self.freq_usage[self._active_idx] += 1
                    optimizer.step()
                    latest = {key: (";".join(row[key] for row in batch_metrics) if key == "selected_proto_ids"
                                    else sum(row[key] for row in batch_metrics) / len(batch_metrics))
                              for key in batch_metrics[0]}
                    latest.update(epoch=epoch + 1, step=step, grad_norm_fourier=grad_norm)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(latest))
                        writer.writeheader()
                    writer.writerow(latest); handle.flush()
                    step += 1
                    if self.max_steps and step >= self.max_steps:
                        break
                if self.max_steps and step >= self.max_steps:
                    break
                if self.enable_adaptive_freq_basis and epoch + 1 == self.adaptive_freq_warmup_epochs:
                    score = self.freq_score / self.freq_usage.clamp_min(1)
                    self.freq_active_idx = torch.topk(score, self.freq_active_num_bases).indices.sort().values
                scheduler.step()
        if step == 0:
            raise RuntimeError("No valid optimization steps")
        os.makedirs(os.path.dirname(global_params_path), exist_ok=True)
        if self.enable_adaptive_freq_basis and self._epoch + 1 >= self.adaptive_freq_warmup_epochs:
            saved_idx = self.freq_active_idx
        else:
            saved_idx = self._active_idx
        saved_coords = [list(self.freq_candidate_coords[int(i)]) for i in saved_idx.cpu().tolist()]
        torch.save({"coords": saved_coords,
                    "fourier_coeff": self.fourier_coeff.detach()[saved_idx].cpu(),
                    "suppress_small": self.suppress_small.detach().cpu(),
                    "method": "legacy_kproto_ret", "support_mode": self.support_mode,
                    "target_class_id": self.target_class_id,
                    "eps": self.eps, "feature_layer": self.feature_layer,
                    "prototype_cache": cache_path, "num_prototypes": self.num_prototypes,
                    "margin": self.margin,
                    "loss_weights": {"lambda_bg": self.lambda_bg,
                                     "lambda_rel": self.lambda_rel,
                                     "lambda_ret": self.lambda_ret}}, global_params_path)
        with open(diagnostics_json_path, "w", encoding="utf-8") as handle:
            json.dump({"latest": latest, "prototype_cache": cache_path,
                       "global_params_path": global_params_path,
                       "optimization_support_sources": support_sources}, handle, indent=2)
        return global_params_path
