import random
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import BasePoisonGenerator, PoisonResult
from .support_utils import get_target_instance_support


class EMMaskPoisoner(BasePoisonGenerator):
    """
    EM-mask baseline (detection adaptation of error-minimizing poison).
    Only uses:
    - clean image / clean annotations
    - target instance support
    - target-score proxy shortcut objective (final paper baseline)
    """

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.alpha = float(method_cfg.get("alpha", method_cfg.get("step_size", 2.0 / 255.0)))
        self.noise_scale = float(method_cfg.get("noise_scale", 1.0))
        self.em_steps = int(method_cfg.get("em_steps", 10))
        self.random_start = bool(method_cfg.get("random_start", False))
        self.objective = str(method_cfg.get("objective", "target_score_proxy"))
        self.use_target_only_labels = bool(method_cfg.get("use_target_only_labels", False))
        self.use_prebaked_instance_mask = bool(method_cfg.get("use_prebaked_instance_mask", True))
        self.strict_instance_mask = bool(method_cfg.get("strict_instance_mask", False))
        self.instance_mask_dir = str(cfg.get("data", {}).get("instance_mask_dir", "") or "")

        # Batched poison generation knobs
        self.poison_batch_size = int(method_cfg.get("poison_batch_size", 12))
        self.poison_amp = bool(method_cfg.get("poison_amp", False))
        self.use_global_steps = bool(method_cfg.get("use_global_steps", False))

        self._single_sanity_printed = False
        self._batch_sanity_printed = False

    def _sanity_tag(self) -> str:
        return "EM-mask"

    def _loss_prefix(self) -> str:
        return "em"

    def _objective_base_name(self) -> str:
        return str(self.objective)

    def _steps_key(self) -> str:
        return "em_steps"

    def _default_steps(self) -> int:
        return int(self.em_steps)

    def _delta_direction(self) -> float:
        # EM: minimize target detector loss
        return -1.0

    def _resolve_steps(self, steps: int) -> int:
        if self.use_global_steps and int(steps) > 0:
            return int(steps)
        return int(self._default_steps())

    def _letterbox_meta(self, h: int, w: int) -> Tuple[float, int, int, int, int]:
        r = min(self.imgsz / float(h), self.imgsz / float(w))
        new_h = max(1, int(round(h * r)))
        new_w = max(1, int(round(w * r)))
        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        left = pad_w // 2
        return float(r), int(left), int(top), int(new_w), int(new_h)

    def _letterbox_mask_tensor(self, mask_t: torch.Tensor) -> torch.Tensor:
        # mask_t: [1,1,H,W] in {0,1}
        if mask_t.ndim != 4 or mask_t.shape[1] != 1:
            raise ValueError(f"mask_t should be [1,1,H,W], got {tuple(mask_t.shape)}")
        _, _, h, w = mask_t.shape
        if h == self.imgsz and w == self.imgsz:
            return mask_t
        r, left, top, new_w, new_h = self._letterbox_meta(h, w)
        resized = F.interpolate(mask_t, size=(new_h, new_w), mode="nearest")
        bottom = self.imgsz - new_h - top
        right = self.imgsz - new_w - left
        return F.pad(resized, (left, right, top, bottom), mode="constant", value=0.0)

    def _project_letterbox_perturb_to_original(
        self,
        pert_lb: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        # pert_lb: [3,S,S]
        if pert_lb.ndim != 3:
            raise ValueError(f"pert_lb should be [3,S,S], got {tuple(pert_lb.shape)}")
        _, left, top, new_w, new_h = self._letterbox_meta(orig_h, orig_w)
        y1 = int(max(0, min(self.imgsz, top)))
        y2 = int(max(y1 + 1, min(self.imgsz, top + new_h)))
        x1 = int(max(0, min(self.imgsz, left)))
        x2 = int(max(x1 + 1, min(self.imgsz, left + new_w)))

        crop = pert_lb[:, y1:y2, x1:x2].unsqueeze(0)
        proj = F.interpolate(crop, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return proj.squeeze(0)

    def _build_batched_loss_batch(
        self,
        image_lb: torch.Tensor,
        annotations_list: List[List[dict]],
        target_only: bool,
        orig_shapes: List[Tuple[int, int]],
    ) -> Tuple[dict, int]:
        cls_vals = []
        box_vals = []
        batch_idx_vals = []

        for local_b, anns in enumerate(annotations_list):
            orig_h, orig_w = int(orig_shapes[local_b][0]), int(orig_shapes[local_b][1])
            r, left, top, _new_w, _new_h = self._letterbox_meta(orig_h, orig_w)

            for ann in anns:
                cls_id = int(ann.get("cls", -1))
                if target_only and cls_id != int(self.target_class_id):
                    continue

                bbox = ann.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue

                cx, cy, bw, bh = [float(v) for v in bbox]
                abs_cx = cx * float(orig_w)
                abs_cy = cy * float(orig_h)
                abs_bw = bw * float(orig_w)
                abs_bh = bh * float(orig_h)

                cx_lb = abs_cx * r + float(left)
                cy_lb = abs_cy * r + float(top)
                bw_lb = abs_bw * r
                bh_lb = abs_bh * r

                cx_n = float(max(0.0, min(1.0, cx_lb / float(self.imgsz))))
                cy_n = float(max(0.0, min(1.0, cy_lb / float(self.imgsz))))
                bw_n = float(max(0.0, min(1.0, bw_lb / float(self.imgsz))))
                bh_n = float(max(0.0, min(1.0, bh_lb / float(self.imgsz))))
                if bw_n <= 1e-8 or bh_n <= 1e-8:
                    continue

                cls_vals.append([float(cls_id)])
                box_vals.append([cx_n, cy_n, bw_n, bh_n])
                batch_idx_vals.append(int(local_b))

        if len(cls_vals) == 0:
            cls_t = torch.zeros((0, 1), dtype=torch.float32, device=self.device)
            box_t = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            batch_idx_t = torch.zeros((0,), dtype=torch.int64, device=self.device)
        else:
            cls_t = torch.tensor(cls_vals, dtype=torch.float32, device=self.device)
            box_t = torch.tensor(box_vals, dtype=torch.float32, device=self.device)
            batch_idx_t = torch.tensor(batch_idx_vals, dtype=torch.int64, device=self.device)

        batch = {
            "img": image_lb,
            "cls": cls_t,
            "bboxes": box_t,
            "batch_idx": batch_idx_t,
            "batch_size": int(image_lb.shape[0]),
        }
        return batch, int(len(cls_vals))

    @staticmethod
    def _extract_loss_scalar(out) -> Optional[torch.Tensor]:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        return None

    def _try_standard_det_loss_batched(
        self,
        adv_img: torch.Tensor,
        annotations_list: List[List[dict]],
        use_target_only: bool,
        orig_shapes: List[Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], int]:
        if not hasattr(self.surrogate, "loss"):
            return None, 0

        batch, n_labels = self._build_batched_loss_batch(
            image_lb=adv_img,
            annotations_list=annotations_list,
            target_only=use_target_only,
            orig_shapes=orig_shapes,
        )
        if use_target_only and n_labels <= 0:
            return None, 0

        preds_raw = self.surrogate(adv_img)

        loss_val = None
        try:
            loss_val = self._extract_loss_scalar(self.surrogate.loss(batch, preds_raw))
        except Exception:
            loss_val = None
        if loss_val is not None:
            return loss_val, n_labels

        try:
            loss_val = self._extract_loss_scalar(self.surrogate.loss(preds_raw, batch))
        except Exception:
            loss_val = None
        if loss_val is not None:
            return loss_val, n_labels

        try:
            loss_val = self._extract_loss_scalar(self.surrogate.loss(batch))
        except Exception:
            loss_val = None
        return loss_val, n_labels

    def _target_score_proxy_loss(self, adv_img: torch.Tensor) -> torch.Tensor:
        preds = self._forward_raw(adv_img)
        target_scores = self._target_scores(preds)
        return -torch.mean(target_scores)

    def _should_use_standard_det_loss(self) -> bool:
        obj = str(self.objective).strip().lower()
        return obj in {
            "untargeted_det_loss",
            "minimize_target_det_loss",
            "robust_minimize_target_det_loss",
        }

    def _compute_objective_loss_batched(
        self,
        adv_img: torch.Tensor,
        annotations_list: List[List[dict]],
        orig_shapes: List[Tuple[int, int]],
        mode: str,
    ) -> Tuple[torch.Tensor, str]:
        current = mode
        if not self._should_use_standard_det_loss():
            if "eot" in str(self.objective).lower():
                return self._target_score_proxy_loss(adv_img), "target_score_proxy_eot"
            return self._target_score_proxy_loss(adv_img), "target_score_proxy"

        loss = None

        if current == "standard_target_only":
            loss, n_labels = self._try_standard_det_loss_batched(
                adv_img, annotations_list, use_target_only=True, orig_shapes=orig_shapes
            )
            if loss is None or n_labels <= 0:
                current = "standard_full_labels"

        if current == "standard_full_labels":
            loss, _ = self._try_standard_det_loss_batched(
                adv_img, annotations_list, use_target_only=False, orig_shapes=orig_shapes
            )
            if loss is None:
                current = "proxy_target_score"

        if current == "proxy_target_score":
            loss = self._target_score_proxy_loss(adv_img)

        return loss, current

    def _build_identity_result(
        self,
        image: np.ndarray,
        support_np: np.ndarray,
        ring_np: np.ndarray,
        steps_eff: int,
        det_loss_mode: str,
        note: str,
    ) -> PoisonResult:
        zero = np.zeros_like(image, dtype=np.float32)
        support_area_ratio = float(np.mean(support_np > 0.5)) if support_np.size > 0 else 0.0
        loss_start_key = f"{self._loss_prefix()}_loss_start"
        loss_end_key = f"{self._loss_prefix()}_loss_end"
        return PoisonResult(
            poisoned_image=image.copy(),
            perturbation=zero,
            support_mask=support_np,
            ring_mask=ring_np,
            losses={loss_start_key: 0.0, loss_end_key: 0.0},
            extras={
                "method": f"{self._loss_prefix()}_mask",
                "poisoned": 0,
                "is_poisoned": False,
                "support_area_ratio": support_area_ratio,
                "changed_pixel_ratio": 0.0,
                loss_start_key: 0.0,
                loss_end_key: 0.0,
                "delta_linf": 0.0,
                "steps_eff": int(steps_eff),
                self._steps_key(): int(steps_eff),
                "objective": self._objective_base_name(),
                "use_target_only_labels": bool(self.use_target_only_labels),
                "det_loss_mode": det_loss_mode,
                "psnr": None,
                "lpips": None,
                "note": note,
            },
        )

    def _prepare_active_batch(
        self,
        images: List[np.ndarray],
        annotations_list: List[List[dict]],
        image_paths: List[str],
        eps: float,
    ):
        del eps
        active_indices: List[int] = []
        imgs_list: List[torch.Tensor] = []
        supports3_list: List[torch.Tensor] = []
        supports_np: List[np.ndarray] = []
        rings_np: List[np.ndarray] = []
        annotations_active: List[List[dict]] = []
        image_shapes: List[Tuple[int, int]] = []
        inactive_meta: Dict[int, Dict] = {}

        for i, (image, anns, image_path) in enumerate(zip(images, annotations_list, image_paths)):
            support_np = get_target_instance_support(
                image_path=image_path,
                image_np=image,
                annotations=anns,
                target_class_id=self.target_class_id,
                use_prebaked_instance_mask=self.use_prebaked_instance_mask,
                instance_mask_dir=self.instance_mask_dir,
                strict_instance_mask=self.strict_instance_mask,
            )
            ring_np = np.zeros_like(support_np, dtype=np.float32)

            target_ann_count = sum(1 for a in anns if int(a.get("cls", -1)) == int(self.target_class_id))
            if float(support_np.sum()) <= 0.0:
                inactive_meta[i] = {
                    "support": support_np,
                    "ring": ring_np,
                    "note": "empty_support",
                    "reason_mode": "none_empty_support",
                }
                continue
            if self.use_target_only_labels and target_ann_count <= 0:
                inactive_meta[i] = {
                    "support": support_np,
                    "ring": ring_np,
                    "note": "empty_target_labels",
                    "reason_mode": "none_empty_target_labels",
                }
                continue

            img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            img_lb = self._letterbox_tensor(img_t)

            support_t = torch.from_numpy(support_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
            support_lb = self._letterbox_mask_tensor(support_t)
            support3 = support_lb.repeat(1, 3, 1, 1)

            active_indices.append(i)
            imgs_list.append(img_lb)
            supports3_list.append(support3)
            supports_np.append(support_np.astype(np.float32))
            rings_np.append(ring_np.astype(np.float32))
            annotations_active.append(anns)
            image_shapes.append((int(image.shape[0]), int(image.shape[1])))

        if len(imgs_list) > 0:
            imgs = torch.cat(imgs_list, dim=0)
            supports3 = torch.cat(supports3_list, dim=0)
        else:
            imgs = torch.zeros((0, 3, self.imgsz, self.imgsz), dtype=torch.float32, device=self.device)
            supports3 = torch.zeros((0, 3, self.imgsz, self.imgsz), dtype=torch.float32, device=self.device)

        return (
            active_indices,
            imgs,
            supports3,
            supports_np,
            rings_np,
            annotations_active,
            image_shapes,
            inactive_meta,
        )

    def _make_amp_context(self):
        if self.poison_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def generate_batch(
        self,
        images: List[np.ndarray],
        annotations_list: List[List[dict]],
        image_paths: List[str],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
    ) -> List[PoisonResult]:
        del support_type
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        n = len(images)
        results: List[Optional[PoisonResult]] = [None] * n
        steps_eff = self._resolve_steps(steps)
        eps_val = float(eps)
        alpha = float(self.alpha)
        noise_scale = float(self.noise_scale)

        (
            active_indices,
            imgs,
            supports3,
            supports_np,
            rings_np,
            annotations_active,
            image_shapes,
            inactive_meta,
        ) = self._prepare_active_batch(
            images=images,
            annotations_list=annotations_list,
            image_paths=image_paths,
            eps=eps,
        )

        for idx, meta in inactive_meta.items():
            results[idx] = self._build_identity_result(
                image=images[idx],
                support_np=meta["support"],
                ring_np=meta["ring"],
                steps_eff=steps_eff,
                det_loss_mode=meta.get("reason_mode", "none"),
                note=meta.get("note", "inactive"),
            )

        if len(active_indices) == 0:
            return [r for r in results]  # type: ignore[arg-type]

        if self.random_start:
            delta = (torch.rand_like(imgs) * 2.0 - 1.0) * eps_val
            delta = (delta * supports3).detach().requires_grad_(True)
        else:
            delta = torch.zeros_like(imgs, requires_grad=True)

        loss_start = None
        loss_end = None
        if self._should_use_standard_det_loss():
            det_loss_mode = "standard_target_only" if self.use_target_only_labels else "standard_full_labels"
        else:
            det_loss_mode = "target_score_proxy_eot" if "eot" in str(self.objective).lower() else "target_score_proxy"

        for _ in range(max(1, int(steps_eff))):
            adv = torch.clamp(imgs + delta * supports3 * noise_scale, 0.0, 1.0)
            with self._make_amp_context():
                loss, det_loss_mode = self._compute_objective_loss_batched(
                    adv_img=adv,
                    annotations_list=annotations_active,
                    orig_shapes=image_shapes,
                    mode=det_loss_mode,
                )

            if loss_start is None:
                loss_start = float(loss.item())
            loss_end = float(loss.item())

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad
            if grad is None or (not torch.isfinite(grad).all()):
                if delta.grad is not None:
                    delta.grad.zero_()
                det_loss_mode = "target_score_proxy"
                loss = self._target_score_proxy_loss(torch.clamp(imgs + delta * supports3 * noise_scale, 0.0, 1.0))
                if loss_start is None:
                    loss_start = float(loss.item())
                loss_end = float(loss.item())
                loss.backward()
                grad = delta.grad
            if grad is None:
                raise RuntimeError(f"{self._sanity_tag()} batch gradient is None.")

            with torch.no_grad():
                delta = delta + self._delta_direction() * alpha * torch.sign(grad)
                delta = torch.clamp(delta, -eps_val, eps_val)
                delta = delta * supports3
                delta = delta.detach().requires_grad_(True)

        with torch.no_grad():
            pert_lb = torch.clamp(delta * supports3 * noise_scale, -eps_val, eps_val)

        loss_start_key = f"{self._loss_prefix()}_loss_start"
        loss_end_key = f"{self._loss_prefix()}_loss_end"

        delta_linf_vals = []
        support_area_vals = []
        changed_area_vals = []
        for local_i, global_i in enumerate(active_indices):
            orig_h, orig_w = image_shapes[local_i]
            clean_np = images[global_i]
            support_np = supports_np[local_i]
            ring_np = rings_np[local_i]

            pert_orig_t = self._project_letterbox_perturb_to_original(pert_lb[local_i], orig_h=orig_h, orig_w=orig_w)
            pert_orig_np = pert_orig_t.detach().permute(1, 2, 0).cpu().numpy().astype(np.float32)
            pert_orig_np = np.clip(pert_orig_np, -eps_val, eps_val)
            pert_orig_np = pert_orig_np * support_np[..., None]
            poisoned_np = np.clip(clean_np + pert_orig_np, 0.0, 1.0).astype(np.float32)

            delta_linf = float(np.max(np.abs(pert_orig_np)))
            poisoned_flag = int(delta_linf > (1.0 / 255.0))
            support_area_ratio = float(np.mean(support_np > 0.5))
            changed_pixel_ratio = float(np.mean(np.max(np.abs(pert_orig_np), axis=2) > (1.0 / 255.0)))

            delta_linf_vals.append(delta_linf)
            support_area_vals.append(support_area_ratio)
            changed_area_vals.append(changed_pixel_ratio)

            objective_name = self._objective_base_name()
            target_only_used = bool(self.use_target_only_labels and det_loss_mode == "standard_target_only")

            results[global_i] = PoisonResult(
                poisoned_image=poisoned_np,
                perturbation=pert_orig_np,
                support_mask=support_np,
                ring_mask=ring_np,
                losses={
                    loss_start_key: float(loss_start or 0.0),
                    loss_end_key: float(loss_end or 0.0),
                },
                extras={
                    "method": f"{self._loss_prefix()}_mask",
                    "poisoned": poisoned_flag,
                    "is_poisoned": bool(poisoned_flag),
                    "support_area_ratio": support_area_ratio,
                    "changed_pixel_ratio": changed_pixel_ratio,
                    loss_start_key: float(loss_start or 0.0),
                    loss_end_key: float(loss_end or 0.0),
                    "delta_linf": delta_linf,
                    "steps_eff": int(steps_eff),
                    self._steps_key(): int(steps_eff),
                    "objective": objective_name,
                    "use_target_only_labels": target_only_used,
                    "det_loss_mode": det_loss_mode,
                    "psnr": None,
                    "lpips": None,
                },
            )

        if not self._batch_sanity_printed:
            print(f"[{self._sanity_tag()}]")
            print(f"  support_area_ratio={float(np.mean(support_area_vals)) if support_area_vals else 0.0:.6f}")
            print(f"  changed_pixel_ratio={float(np.mean(changed_area_vals)) if changed_area_vals else 0.0:.6f}")
            print(f"  loss_start={float(loss_start or 0.0):.6f}")
            print(f"  loss_end={float(loss_end or 0.0):.6f}")
            print(f"  delta_linf={float(np.max(delta_linf_vals)) if delta_linf_vals else 0.0:.6f}")
            print(f"  steps_eff={int(steps_eff)}")
            print(f"  objective={self._objective_base_name()}")
            print(f"  use_target_only_labels={bool(self.use_target_only_labels)}")
            print(f"  det_loss_mode={det_loss_mode}")
            self._batch_sanity_printed = True

        return [r for r in results]  # type: ignore[arg-type]

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
        results = self.generate_batch(
            images=[image],
            annotations_list=[annotations],
            image_paths=[image_path or ""],
            seed=seed,
            steps=steps,
            eps=eps,
            support_type=support_type,
        )
        return results[0]
