import random
from typing import List

import torch
import torch.nn.functional as F

from .base import BasePoisonGenerator, PoisonResult
from .fourier import build_fourier_pattern, sample_midfreq_coords, spectrum_to_numpy

try:
    import kornia.augmentation as K
except Exception:  # pragma: no cover
    K = None


class OursPoisonGenerator(BasePoisonGenerator):
    """Ours-mask: assignment-aware alignment collapse + shape entanglement + non-target preservation."""

    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)

        self.ring_width = int(method_cfg.get("ring_width", 4))
        self.enable_conditional_sps = bool(method_cfg.get("enable_conditional_sps", True))
        self.enable_midfreq_search = bool(method_cfg.get("enable_midfreq_search", True))
        self.enable_jnd_gain = bool(method_cfg.get("enable_jnd_gain", True))
        self.enable_assignment_shortcut = bool(method_cfg.get("enable_assignment_shortcut", True))
        self.enable_shape_entanglement = bool(method_cfg.get("enable_shape_entanglement", True))

        self.shortcut_num_bases = int(method_cfg.get("shortcut_num_bases", 2))
        self.shortcut_ring_weight = float(method_cfg.get("shortcut_ring_weight", 1.0))
        self.suppress_inner_weight = float(method_cfg.get("suppress_inner_weight", 1.0))
        self.suppress_ring_weight = float(method_cfg.get("suppress_ring_weight", 0.5))

        self.assignment_topk = int(method_cfg.get("assignment_topk", 100))
        self.align_alpha = float(method_cfg.get("align_alpha", 1.0))
        self.align_beta = float(method_cfg.get("align_beta", 6.0))
        self.assign_margin = float(method_cfg.get("assign_margin", 0.4))

        self.lambda_align = float(method_cfg.get("lambda_align", 4.0))
        self.lambda_rank = float(method_cfg.get("lambda_rank", 2.0))
        self.lambda_shape = float(method_cfg.get("lambda_shape", 1.0))
        self.lambda_preserve = float(method_cfg.get("lambda_preserve", 1.0))
        self.lambda_suppress = float(method_cfg.get("lambda_suppress", 0.5))
        self.lambda_jnd = float(method_cfg.get("lambda_jnd", 0.5))
        self.lambda_tv = float(method_cfg.get("lambda_tv", 1e-3))
        self.lambda_budget = float(method_cfg.get("lambda_budget", 10.0))

        self.jnd_floor = float(method_cfg.get("jnd_floor", 0.2))
        self.jnd_ceiling = float(method_cfg.get("jnd_ceiling", 1.0))
        self.eot_samples = int(method_cfg.get("eot_samples", 4))
        self.suppress_step_size = float(method_cfg.get("suppress_step_size", 1.0 / 255.0))
        self.optim_lr = float(method_cfg.get("optim_lr", 0.03))

        if K is not None:
            self.eot_aug = K.AugmentationSequential(
                K.RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(0.85, 1.15), p=0.7),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.7),
                same_on_batch=False,
            ).to(self.device)
        else:
            self.eot_aug = None

        self.mid_feature_module_name = None
        if self.enable_shape_entanglement:
            self.mid_feature_module_name = self._register_mid_feature_hook()

    def _apply_shared_eot_pair(self, clean: torch.Tensor, adv: torch.Tensor):
        if self.eot_aug is not None:
            params = self.eot_aug.forward_parameters(clean.shape)
            clean_aug = self.eot_aug(clean, params=params)
            adv_aug = self.eot_aug(adv, params=params)
        else:
            clean_aug = clean
            adv_aug = adv
            if random.random() < 0.5:
                clean_aug = torch.flip(clean_aug, dims=[3])
                adv_aug = torch.flip(adv_aug, dims=[3])

        gain = 0.92 + 0.16 * torch.rand(1, device=clean.device)
        noise_scale = 0.005 * torch.rand(1, device=clean.device)
        noise = torch.randn_like(clean_aug) * noise_scale

        clean_aug = torch.clamp(clean_aug * gain + noise, 0.0, 1.0)
        adv_aug = torch.clamp(adv_aug * gain + noise, 0.0, 1.0)
        return clean_aug, adv_aug

    def _jnd_gain(self, img: torch.Tensor) -> torch.Tensor:
        if not self.enable_jnd_gain:
            return torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), device=img.device)

        r = img[:, 0:1, :, :]
        g = img[:, 1:2, :, :]
        b = img[:, 2:3, :, :]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=img.device, dtype=img.dtype)
        ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=img.device, dtype=img.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

        mag = mag - mag.amin(dim=(2, 3), keepdim=True)
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        return self.jnd_floor + (self.jnd_ceiling - self.jnd_floor) * mag

    @staticmethod
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dx + dy

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled = (feat * mask).sum(dim=(2, 3), keepdim=True) / denom
        return pooled.squeeze(-1).squeeze(-1)

    @staticmethod
    def _topk_mean(x: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, min(int(k), x.numel()))
        return torch.topk(x.reshape(-1), k=k, largest=True).values.mean()

    def generate(
        self,
        image,
        annotations: List[dict],
        seed: int,
        steps: int,
        eps: float,
        support_type: str,
        image_path: str = None,
    ) -> PoisonResult:
        random.seed(seed)
        torch.manual_seed(seed)

        if support_type != "mask":
            raise ValueError(
                f"ours_mask only supports support_type='mask', but got '{support_type}'."
            )

        inner_np, ring_np = self._build_support(
            image_shape=image.shape,
            annotations=annotations,
            support_type="mask",
            ring_width=self.ring_width,
            image_path=image_path,
        )

        if float(inner_np.mean()) <= 0.0:
            zero = image * 0.0
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=inner_np,
                ring_mask=ring_np,
                losses={"L_total": 0.0},
                extras={"note": "empty_support"},
            )

        img = self._to_tensor(image)
        h, w = image.shape[:2]

        gt_boxes_xyxy = self._collect_target_gt_boxes_xyxy(annotations, image.shape)
        if gt_boxes_xyxy.numel() == 0:
            raise RuntimeError("Target GT boxes are empty while support is non-empty.")

        inner = torch.from_numpy(inner_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        ring = torch.from_numpy(ring_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

        shortcut_support = torch.clamp(inner + self.shortcut_ring_weight * ring, 0.0, 1.0)
        suppress_support = torch.clamp(
            self.suppress_inner_weight * inner + self.suppress_ring_weight * ring,
            0.0,
            1.0,
        )

        shortcut_support3 = shortcut_support.repeat(1, 3, 1, 1)
        suppress_support3 = suppress_support.repeat(1, 3, 1, 1)

        coords = sample_midfreq_coords(
            h=h,
            w=w,
            num_bases=self.shortcut_num_bases,
            seed=seed,
            enable_search=self.enable_midfreq_search,
        )

        amp_param = torch.nn.Parameter(torch.zeros((self.shortcut_num_bases,), device=self.device))
        suppress_param = torch.nn.Parameter(torch.zeros_like(img))
        optimizer = torch.optim.Adam([amp_param, suppress_param], lr=self.optim_lr)

        jnd = self._jnd_gain(img)
        jnd3 = jnd.repeat(1, 3, 1, 1)

        tracked = {}
        for _step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)

            amps = torch.tanh(amp_param) * eps
            pattern = build_fourier_pattern(h, w, coords, amps, self.device)
            pattern3 = pattern.repeat(1, 3, 1, 1)

            if self.enable_conditional_sps:
                shortcut = pattern3 * shortcut_support3 * jnd3
            else:
                shortcut = torch.zeros_like(img)

            suppression = torch.tanh(suppress_param) * eps * 0.30 * suppress_support3
            raw_perturb = shortcut + suppression
            perturb = torch.clamp(raw_perturb, -eps, eps)
            adv = torch.clamp(img + perturb, 0.0, 1.0)

            eot_count = max(1, self.eot_samples)
            L_align = torch.tensor(0.0, device=self.device)
            L_rank = torch.tensor(0.0, device=self.device)
            L_shape = torch.tensor(0.0, device=self.device)
            L_preserve = torch.tensor(0.0, device=self.device)
            L_cls_aux = torch.tensor(0.0, device=self.device)

            align_clean_topk_acc = 0.0
            align_adv_topk_acc = 0.0
            target_prob_clean_acc = 0.0
            target_prob_adv_acc = 0.0
            best_iou_clean_acc = 0.0
            best_iou_adv_acc = 0.0

            for _ in range(eot_count):
                clean_aug, adv_aug = self._apply_shared_eot_pair(img, adv)

                with torch.no_grad():
                    self._clear_mid_feature_cache()
                    preds_clean = self._forward_raw(clean_aug)
                    align_clean, target_prob_clean, best_iou_clean = self._alignment_proxy(
                        preds=preds_clean,
                        gt_boxes_xyxy=gt_boxes_xyxy,
                        alpha=self.align_alpha,
                        beta=self.align_beta,
                        image_shape=image.shape,
                    )
                    k = min(self.assignment_topk, int(align_clean.numel()))
                    align_clean_topk = self._topk_mean(align_clean, k)
                    non_target_clean = self._non_target_logits(preds_clean)
                    if self.enable_shape_entanglement:
                        feat_clean = self._get_mid_feature(require=True, detach=True)
                    else:
                        feat_clean = None

                self._clear_mid_feature_cache()
                preds_adv = self._forward_raw(adv_aug)
                align_adv, target_prob_adv, best_iou_adv = self._alignment_proxy(
                    preds=preds_adv,
                    gt_boxes_xyxy=gt_boxes_xyxy,
                    alpha=self.align_alpha,
                    beta=self.align_beta,
                    image_shape=image.shape,
                )

                align_adv_topk = self._topk_mean(align_adv, k)
                L_align = L_align + align_adv_topk
                L_rank = L_rank + F.relu(align_adv_topk - self.assign_margin * align_clean_topk)

                if self.enable_shape_entanglement:
                    feat_adv = self._get_mid_feature(require=True, detach=False)
                    if feat_clean is None:
                        raise RuntimeError("feat_clean is None while shape entanglement is enabled.")
                    target_mask_feat = F.interpolate(inner, size=feat_adv.shape[-2:], mode="nearest")
                    feat_clean_pool = self._masked_pool(feat_clean, target_mask_feat)
                    feat_adv_pool = self._masked_pool(feat_adv, target_mask_feat)
                    cos = F.cosine_similarity(feat_clean_pool, feat_adv_pool, dim=1).mean()
                    L_shape = L_shape + (1.0 - cos)

                non_target_adv = self._non_target_logits(preds_adv)
                if non_target_adv.numel() > 0 and non_target_clean.numel() > 0:
                    L_preserve = L_preserve + F.mse_loss(
                        torch.sigmoid(non_target_adv),
                        torch.sigmoid(non_target_clean),
                    )

                L_cls_aux = L_cls_aux + target_prob_adv.mean()

                align_clean_topk_acc += float(align_clean_topk.item())
                align_adv_topk_acc += float(align_adv_topk.item())
                target_prob_clean_acc += float(target_prob_clean.mean().item())
                target_prob_adv_acc += float(target_prob_adv.mean().item())
                best_iou_clean_acc += float(best_iou_clean.mean().item())
                best_iou_adv_acc += float(best_iou_adv.mean().item())

            L_align = L_align / eot_count
            L_rank = L_rank / eot_count
            L_shape = L_shape / eot_count
            L_preserve = L_preserve / eot_count
            L_cls_aux = L_cls_aux / eot_count

            if self.enable_jnd_gain and self.enable_conditional_sps:
                L_jnd = torch.mean(torch.abs(shortcut) * (1.0 - jnd3))
            else:
                L_jnd = torch.tensor(0.0, device=self.device)

            L_tv = self._tv_loss(suppression)
            L_budget = F.relu(torch.max(torch.abs(raw_perturb)) - eps)

            total_loss = (
                self.lambda_align * L_align
                + self.lambda_rank * L_rank
                + self.lambda_shape * L_shape
                + self.lambda_preserve * L_preserve
                + self.lambda_suppress * L_cls_aux
                + self.lambda_jnd * L_jnd
                + self.lambda_tv * L_tv
                + self.lambda_budget * L_budget
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite total loss detected in ours_mask optimization.")

            total_loss.backward()

            if amp_param.grad is None:
                raise RuntimeError("amp_param.grad is None in ours_mask optimization.")
            if suppress_param.grad is None:
                raise RuntimeError("suppress_param.grad is None in ours_mask optimization.")

            grad_norm_shortcut = float(torch.norm(amp_param.grad).item())
            grad_norm_suppress = float(torch.norm(suppress_param.grad).item())
            grad_norm_total = float(
                torch.norm(
                    torch.cat(
                        [
                            amp_param.grad.reshape(-1),
                            suppress_param.grad.reshape(-1),
                        ],
                        dim=0,
                    )
                ).item()
            )

            if not torch.isfinite(torch.tensor([grad_norm_total], device=self.device)).all():
                raise RuntimeError("Non-finite gradient norm detected in ours_mask optimization.")

            optimizer.step()

            with torch.no_grad():
                if not torch.isfinite(amp_param).all() or not torch.isfinite(suppress_param).all():
                    raise RuntimeError("Parameters became NaN/Inf after optimizer step.")

            tracked = {
                "L_align": float(L_align.item()),
                "L_rank": float(L_rank.item()),
                "L_shape": float(L_shape.item()),
                "L_preserve": float(L_preserve.item()),
                "L_cls_aux": float(L_cls_aux.item()),
                "L_jnd": float(L_jnd.item()),
                "L_tv": float(L_tv.item()),
                "L_budget": float(L_budget.item()),
                "L_total": float(total_loss.item()),
                "align_clean_topk": align_clean_topk_acc / eot_count,
                "align_adv_topk": align_adv_topk_acc / eot_count,
                "target_prob_clean_mean": target_prob_clean_acc / eot_count,
                "target_prob_adv_mean": target_prob_adv_acc / eot_count,
                "best_iou_clean_mean": best_iou_clean_acc / eot_count,
                "best_iou_adv_mean": best_iou_adv_acc / eot_count,
                "shape_loss": float(L_shape.item()),
                "preserve_loss": float(L_preserve.item()),
                "grad_norm_total": grad_norm_total,
                "grad_norm_shortcut": grad_norm_shortcut,
                "grad_norm_suppress": grad_norm_suppress,
            }

        with torch.no_grad():
            amps = torch.tanh(amp_param) * eps
            pattern = build_fourier_pattern(h, w, coords, amps, self.device).repeat(1, 3, 1, 1)
            if self.enable_conditional_sps:
                shortcut = pattern * shortcut_support3 * jnd3
            else:
                shortcut = torch.zeros_like(img)
            suppression = torch.tanh(suppress_param) * eps * 0.30 * suppress_support3
            pert = torch.clamp(shortcut + suppression, -eps, eps)
            poisoned = torch.clamp(img + pert, 0.0, 1.0)

            spectrum = spectrum_to_numpy(h, w, coords, amps.detach().cpu().numpy())

            if not torch.isfinite(poisoned).all() or not torch.isfinite(pert).all():
                raise RuntimeError("Non-finite poisoned image or perturbation detected at finalize stage.")

        return PoisonResult(
            poisoned_image=self._to_numpy(poisoned),
            perturbation=self._to_numpy(pert),
            support_mask=inner_np,
            ring_mask=ring_np,
            losses=tracked,
            extras={
                "coords": coords,
                "jnd_gain": jnd.squeeze(0).squeeze(0).detach().cpu().numpy(),
                "spectrum": spectrum,
                "pattern": pattern.squeeze(0).mean(dim=0).detach().cpu().numpy(),
                "mid_feature_module_name": self.mid_feature_module_name,
            },
        )

