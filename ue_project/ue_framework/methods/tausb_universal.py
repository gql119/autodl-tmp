import csv
import math
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..data_utils import image_has_target, label_path_for_image, list_images, load_image_rgb_float, read_yolo_annotations
from ..io_utils import atomic_write_json
from ..ultra.hijacked_loss import HijackedV8Loss
from .base import BasePoisonGenerator, PoisonResult
from .fourier import build_fourier_pattern, sample_bandfreq_coords, sample_midfreq_coords, spectrum_to_numpy
from .shadow_tal import DifferentiableShadowTAL
from .alce_acgt import (
    build_non_target_core_mask,
    build_confounder_mask,
    build_pag_gate,
    build_local_context_mask,
    build_scale_adaptive_context_mask,
    project_strict_gate_to_fpn,
    renorm_yolo_bbox_after_padding,
)
from .alce_losses import (
    compute_anchor_losses,
    compute_alsi_score,
    compute_collapse_loss,
    compute_entangle_loss,
    compute_non_target_margin_preserve,
    masked_prototype,
    robust_masked_prototype,
)
from .alce_metrics import compute_confounder_purity, compute_overlap_ratio, safe_mean
from ..support import build_support_mask


try:
    import kornia.augmentation as K
except Exception:  # pragma: no cover
    K = None


# ==========================================
# 🚀 1. 纯净 IO 多进程加载器
# ==========================================
class TAUSBDataset(Dataset):
    def __init__(self, image_paths, label_dir, target_class_id, instance_mask_dir=""):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.target_class_id = target_class_id
        self.instance_mask_dir = instance_mask_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_np = load_image_rgb_float(img_path)
        anns = read_yolo_annotations(label_path_for_image(img_path, self.label_dir))

        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = ""
        if self.instance_mask_dir:
            cand = os.path.join(self.instance_mask_dir, stem + ".png")
            if os.path.isfile(cand):
                mask_path = cand

        return {
            "clean_np": clean_np,
            "anns": anns,
            "img_path": img_path,
            "mask_path": mask_path,
        }


def tausb_collate_fn(batch):
    return batch


# ==========================================
# 🚀 _TAUSBCommon: 核心基类与多轨特征挂载
# ==========================================
class _TAUSBCommon(BasePoisonGenerator):
    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)

        self.ring_width = int(method_cfg.get("ring_width", 4))
        self.shortcut_num_bases = int(method_cfg.get("shortcut_num_bases", 16)) 
        self.suppress_small_size = int(method_cfg.get("suppress_small_size", 32))

        self.align_alpha = float(method_cfg.get("align_alpha", 0.5))
        self.align_beta = float(method_cfg.get("align_beta", 6.0))
        self.assignment_topk = int(method_cfg.get("assignment_topk", 100))

        alce_cfg = method_cfg.get("alce", {}) if isinstance(method_cfg.get("alce", {}), dict) else {}
        self.alce_enabled = bool(alce_cfg.get("enabled", method_cfg.get("alce_enabled", True)))
        self.lambda_ent = float(
            alce_cfg.get("lambda_ent", method_cfg.get("lambda_ent", method_cfg.get("lambda_tsvc", 10.0)))
        )
        self.lambda_anchor = float(
            alce_cfg.get("lambda_anchor", method_cfg.get("lambda_anchor", method_cfg.get("lambda_sem", 5.0)))
        )
        self.lambda_flat = float(alce_cfg.get("lambda_flat", method_cfg.get("lambda_flat", 2.0)))
        self.entangle_tau = float(alce_cfg.get("entangle_tau", method_cfg.get("entangle_tau", 0.1)))
        self.context_expand_ratio = float(
            alce_cfg.get("context_expand_ratio", method_cfg.get("context_expand_ratio", 1.5))
        )
        self.exclude_ring = bool(alce_cfg.get("exclude_ring", method_cfg.get("exclude_ring", True)))
        self.exclude_all_objects = bool(
            alce_cfg.get("exclude_all_objects", method_cfg.get("exclude_all_objects", True))
        )
        self.min_conf_pixels = float(alce_cfg.get("min_conf_pixels", method_cfg.get("min_conf_pixels", 20.0)))
        self.rlcp_core_scale = float(alce_cfg.get("rlcp_core_scale", method_cfg.get("rlcp_core_scale", 0.8)))
        self.rlcp_trim_ratio = float(alce_cfg.get("rlcp_trim_ratio", method_cfg.get("rlcp_trim_ratio", 0.1)))
        self.rlcp_adaptive_alpha = float(
            alce_cfg.get("rlcp_adaptive_alpha", method_cfg.get("rlcp_adaptive_alpha", 0.15))
        )
        self.rlcp_adaptive_beta = float(
            alce_cfg.get("rlcp_adaptive_beta", method_cfg.get("rlcp_adaptive_beta", 0.35))
        )
        self.use_adaptive_context = bool(
            alce_cfg.get("use_adaptive_context", method_cfg.get("use_adaptive_context", True))
        )
        
        self.rlcp_adaptive_inner_min = float(
            alce_cfg.get("rlcp_adaptive_inner_min", method_cfg.get("rlcp_adaptive_inner_min", 8.0))
        )
        self.rlcp_adaptive_inner_max = float(
            alce_cfg.get("rlcp_adaptive_inner_max", method_cfg.get("rlcp_adaptive_inner_max", 16.0))
        )
        self.rlcp_adaptive_outer_min = float(
            alce_cfg.get("rlcp_adaptive_outer_min", method_cfg.get("rlcp_adaptive_outer_min", 24.0))
        )
        self.rlcp_adaptive_outer_max = float(
            alce_cfg.get("rlcp_adaptive_outer_max", method_cfg.get("rlcp_adaptive_outer_max", 36.0))
        )
        self.rlcp_adaptive_min_gap = float(
            alce_cfg.get("rlcp_adaptive_min_gap", method_cfg.get("rlcp_adaptive_min_gap", 8.0))
        )
        

        # 🚀 升级：读取 FPN 分层 PAG 比例 [P3, P4, P5] 和 最小存活数
        self.pag_layer_ratios = alce_cfg.get("pag_layer_ratios", method_cfg.get("pag_layer_ratios", [0.7, 0.6, 0.4]))
        self.pag_min_pos = alce_cfg.get("pag_min_pos", method_cfg.get("pag_min_pos", [8, 6, 4]))

        self.lambda_tsvc = float(method_cfg.get("lambda_tsvc", self.lambda_flat))
        self.lambda_sem = float(method_cfg.get("lambda_sem", self.lambda_anchor))
        self.lambda_preserve = float(alce_cfg.get("lambda_preserve", method_cfg.get("lambda_preserve", 5.0)))
        self.lambda_preserve_feat = float(
            alce_cfg.get("lambda_preserve_feat", method_cfg.get("lambda_preserve_feat", 0.5))
        )
        self.lambda_preserve_logits = float(
            alce_cfg.get("lambda_preserve_logits", method_cfg.get("lambda_preserve_logits", 1.0))
        )
        self.lambda_margin = float(alce_cfg.get("lambda_margin", method_cfg.get("lambda_margin", 1.0)))
        if not self.alce_enabled:
            self.lambda_ent = 0.0
            self.lambda_anchor = self.lambda_sem
            self.lambda_flat = self.lambda_tsvc
        
        self.lambda_tv = float(method_cfg.get("lambda_tv", 0.0))
        self.lambda_budget = float(method_cfg.get("lambda_budget", 0.0))

        self.lambda_freq = float(method_cfg.get("lambda_freq", 1.0))
        self.lambda_supp = float(method_cfg.get("lambda_supp", 0.35))
        self.ring_weight_freq = float(method_cfg.get("shortcut_ring_weight", 1.0))
        self.ring_weight_supp = float(method_cfg.get("suppress_ring_weight", 0.5))

        self.jnd_floor = float(method_cfg.get("jnd_floor", 0.2))
        self.jnd_ceiling = float(method_cfg.get("jnd_ceiling", 1.0))
        self.eot_samples = int(method_cfg.get("eot_samples", 1))

        self.enable_shape_entanglement = bool(method_cfg.get("enable_shape_entanglement", True))
        
        self.freq_amp_buffer = float(method_cfg.get("freq_amp_buffer", 1.0))
        self.supp_amp_buffer = float(method_cfg.get("supp_amp_buffer", 1.0))
        self.tanh_temp = float(method_cfg.get("tanh_temp", 1.0))

        if K is not None:
            self.eot_aug = K.AugmentationSequential(
                K.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.7),
                same_on_batch=False,
            ).to(self.device)
        else:
            self.eot_aug = None

        self.preserve_layers = ["model.4", "model.6"]
        self.shape_layers = ["model.15", "model.18", "model.21"]
        self.multi_features = {}
        self._register_multi_layer_hooks()
        
        self.instance_mask_dir = str(cfg.get("data", {}).get("instance_mask_dir", "") or "")
        self.allow_pseudo_mask_fallback = bool(cfg.get("data", {}).get("allow_pseudo_mask_fallback", True))
        self.legacy_best_reproduce_mode = bool(method_cfg.get("legacy_best_reproduce_mode", False))
        self.force_pseudo_mask_fallback = bool(method_cfg.get("force_pseudo_mask_fallback", False))

        self.use_prebaked_instance_mask = bool(method_cfg.get("use_prebaked_instance_mask", True))
        self.strict_instance_mask = bool(method_cfg.get("strict_instance_mask", False))
        if self.instance_mask_dir and not os.path.isabs(self.instance_mask_dir):
            self.instance_mask_dir = "/" + self.instance_mask_dir.lstrip("/")

        self.is_universal_training = False

    def _register_multi_layer_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.multi_features[name] = output
            return hook
            
        for name, module in self.surrogate.named_modules():
            if name in self.preserve_layers or name in self.shape_layers:
                module.register_forward_hook(get_activation(name))

    def _clear_multi_features(self):
        self.multi_features.clear()

    def _resolve_instance_mask_path(self, image_path: str):
        if self.legacy_best_reproduce_mode and self.force_pseudo_mask_fallback:
            return None
        if not self.use_prebaked_instance_mask:
            return None
        if not self.instance_mask_dir:
            return None
        if not image_path:
            return None

        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(self.instance_mask_dir, stem + ".png")
        if os.path.isfile(mask_path):
            return mask_path
        return None

    def _build_support(self, image_shape, annotations, support_type="mask", ring_width=4, image_path=None):
        h, w = image_shape[:2]
        zero = np.zeros((h, w), dtype=np.float32)

        if support_type != "mask":
            raise ValueError(f"tausb_mask only supports support_type='mask', got {support_type}")

        if self.legacy_best_reproduce_mode and self.force_pseudo_mask_fallback:
            inner_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask",
                ring_width=ring_width,
                mask_path=None,
            )
            ring_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask_ring",
                ring_width=ring_width,
                mask_path=None,
            )
            return inner_mask.astype(np.float32), ring_mask.astype(np.float32), "forced_pseudo_fallback"

        mask_path = self._resolve_instance_mask_path(image_path)

        if mask_path is not None:
            inner_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask",
                ring_width=ring_width,
                mask_path=mask_path,
            )
            ring_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask_ring",
                ring_width=ring_width,
                mask_path=mask_path,
            )
            return inner_mask.astype(np.float32), ring_mask.astype(np.float32), "prebaked"

        if self.strict_instance_mask:
            return zero, zero, "empty_no_prebaked_mask"

        if self.allow_pseudo_mask_fallback:
            inner_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask",
                ring_width=ring_width,
                mask_path=None,
            )
            ring_mask = build_support_mask(
                image_shape=image_shape,
                annotations=annotations,
                target_class_id=self.target_class_id,
                support_type="mask_ring",
                ring_width=ring_width,
                mask_path=None,
            )
            return inner_mask.astype(np.float32), ring_mask.astype(np.float32), "pseudo_fallback"

        return zero, zero, "empty_no_fallback"

    def _apply_shared_eot_pair_batched(self, clean: torch.Tensor, adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = clean.shape[0]
        if self.eot_aug is not None:
            params = self.eot_aug.forward_parameters(clean.shape)
            clean_aug = self.eot_aug(clean, params=params)
            adv_aug = self.eot_aug(adv, params=params)
        else:
            clean_aug = clean
            adv_aug = adv

        gain = 0.92 + 0.16 * torch.rand(B, 1, 1, 1, device=clean.device)
        noise_scale = 0.004 * torch.rand(B, 1, 1, 1, device=clean.device)
        noise = torch.randn_like(clean_aug) * noise_scale
        clean_aug = torch.clamp(clean_aug * gain + noise, 0.0, 1.0)
        adv_aug = torch.clamp(adv_aug * gain + noise, 0.0, 1.0)
        return clean_aug, adv_aug

    def _jnd_gain(self, img: torch.Tensor, current_floor: float) -> torch.Tensor:
        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=img.device, dtype=img.dtype)
        ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=img.device, dtype=img.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

        mag = mag - mag.amin(dim=(2, 3), keepdim=True)
        mag = mag / mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return current_floor + (self.jnd_ceiling - current_floor) * mag

    @staticmethod
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dx + dy

    def _build_global_freq_pattern(self, h: int, w: int, coords: List[Tuple[int, int]], coeff: torch.Tensor) -> torch.Tensor:
        ch_patterns = []
        for c in range(3):
            amps = torch.tanh(coeff[:, c] / self.tanh_temp) * (self.eps * self.freq_amp_buffer)
            base = build_fourier_pattern(self.imgsz, self.imgsz, coords, amps, self.device)
            cur = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
            ch_patterns.append(cur)
        return torch.cat(ch_patterns, dim=1)

    def _compose_delta(self, img_t: torch.Tensor, inner_np: np.ndarray, ring_np: np.ndarray, coords: List[Tuple[int, int]], fourier_coeff: torch.Tensor, suppress_small: torch.Tensor):
        inner = torch.from_numpy(inner_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        ring = torch.from_numpy(ring_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return self._compose_delta_batched(img_t, inner, ring, coords, fourier_coeff, suppress_small, current_epoch=0)

    def _compose_delta_batched(self, img_t: torch.Tensor, inner_t: torch.Tensor, ring_t: torch.Tensor, coords: List[Tuple[int, int]], fourier_coeff: torch.Tensor, suppress_small: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = img_t.shape[-2:]
        
        support_freq = torch.clamp(inner_t + 0.01 * ring_t, 0.0, 1.0)
        support_supp = torch.clamp(inner_t + 0.01 * ring_t, 0.0, 1.0)

        support_freq3 = support_freq.repeat(1, 3, 1, 1)
        support_supp3 = support_supp.repeat(1, 3, 1, 1)

        if getattr(self, 'is_universal_training', False):
            tanh_coeff = 1.5 + min(2.5, (current_epoch / 20.0) * 2.5)
            if current_epoch < 15:
                cur_jnd_floor = 0.4
            else:
                cur_jnd_floor = 0.4 + min(0.1, ((current_epoch - 15) / 15.0) * 0.2)
        else:
            tanh_coeff = 4.0
            cur_jnd_floor = 0.5

        jnd = self._jnd_gain(img_t, current_floor=cur_jnd_floor)
        jnd3 = jnd.repeat(1, 3, 1, 1)

        freq_pattern = self._build_global_freq_pattern(h, w, coords, fourier_coeff)
        freq_pattern = torch.tanh(freq_pattern * tanh_coeff)
        
        shortcut = self.lambda_freq * (freq_pattern * jnd3 * support_freq3)
        
        if getattr(self, 'is_universal_training', False): 
            scale_factor = random.uniform(0.6, 1.0)
            dropout_mask = (torch.rand_like(shortcut) > 0.15).float() 
            shortcut = shortcut * scale_factor * dropout_mask
        
        suppression = 0.0

        raw_perturb = shortcut + suppression
        perturb = torch.clamp(raw_perturb, -self.eps, self.eps)
        adv = torch.clamp(img_t + perturb, 0.0, 1.0)
        return raw_perturb, perturb, adv, support_freq, jnd


class TAUSBMaskGenerator(_TAUSBCommon):
    def __init__(self, cfg, method_cfg, device, surrogate, global_params_path: str):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.is_universal_training = False
        if not os.path.isfile(global_params_path):
            raise FileNotFoundError(f"global params not found: {global_params_path}")
        pack = torch.load(global_params_path, map_location=device)
        self.coords = [tuple(x) for x in pack["coords"]]
        self.fourier_coeff = pack["fourier_coeff"].to(device)
        self.suppress_small = pack["suppress_small"].to(device)

    def generate(self, image: np.ndarray, annotations: List[dict], seed: int, steps: int, eps: float, support_type: str, image_path: str = None) -> PoisonResult:
        if support_type != "mask":
            raise ValueError(f"tausb_mask only supports support_type='mask', got {support_type}")

        inner_np, ring_np, support_source = self._build_support(
            image_shape=image.shape,
            annotations=annotations,
            support_type="mask",
            ring_width=self.ring_width,
            image_path=image_path,
        )
            
        if float(inner_np.sum()) <= 10.0:  
            zero = image * 0.0
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=inner_np,
                ring_mask=ring_np,
                losses={"L_total": 0.0},
                extras={
                    "note": "empty_support", 
                    "is_poisoned": False,
                    "poisoned": 0,
                    "support_source": support_source,
                    "mask_path": self._resolve_instance_mask_path(image_path) or "",
                },
            )

        with torch.no_grad():
            img_t = self._to_tensor(image)
            raw_perturb, perturb, adv, _support, jnd = self._compose_delta(
                img_t=img_t,
                inner_np=inner_np,
                ring_np=ring_np,
                coords=self.coords,
                fourier_coeff=self.fourier_coeff,
                suppress_small=self.suppress_small,
            )
            budget_violation = F.relu(torch.max(torch.abs(raw_perturb)) - self.eps)

            h, w = image.shape[:2]
            spectrum = spectrum_to_numpy(
                self.imgsz,
                self.imgsz,
                self.coords,
                torch.tanh(self.fourier_coeff[:, 0] / self.tanh_temp).detach().cpu().numpy(),
            )
            pattern_vis = self._build_global_freq_pattern(h, w, self.coords, self.fourier_coeff).mean(dim=1)

        return PoisonResult(
            poisoned_image=self._to_numpy(adv),
            perturbation=self._to_numpy(perturb),
            support_mask=inner_np,
            ring_mask=ring_np,
            losses={
                "L_budget": float(budget_violation.item()),
                "L_total": float(budget_violation.item()),
            },
            extras={
                "note": "poisoned",
                "is_poisoned": True,
                "poisoned": 1,
                "coords": self.coords,
                "jnd_gain": jnd.squeeze(0).squeeze(0).detach().cpu().numpy(),
                "spectrum": spectrum,
                "pattern": pattern_vis.squeeze(0).detach().cpu().numpy(),
                "support_source": support_source,
                "mask_path": self._resolve_instance_mask_path(image_path) or "",
            },
        )


class TAUSBUniversalTrainer(_TAUSBCommon):
    def __init__(self, cfg, method_cfg, device, surrogate):
        super().__init__(cfg, method_cfg, device, surrogate)
        self.is_universal_training = True 

        self.universal_epochs = int(method_cfg.get("universal_epochs", 40))
        self.universal_batch_size = int(method_cfg.get("universal_batch_size", 16))
        self.universal_lr_fourier = float(method_cfg.get("universal_lr_fourier", 0.05)) 
        self.universal_lr_suppress = float(method_cfg.get("universal_lr_suppress", 0.002))
        self.phase0_diagnostics_only = bool(method_cfg.get("phase0_diagnostics_only", False))
        self.phase0_max_epochs = int(method_cfg.get("phase0_max_epochs", 1))
        self.save_phase0_probe_only = bool(method_cfg.get("save_phase0_probe_only", False))

        self.enable_adaptive_freq_basis = bool(method_cfg.get("enable_adaptive_freq_basis", False))
        self.freq_candidate_num_bases = int(method_cfg.get("freq_candidate_num_bases", 64))
        self.freq_active_num_bases = int(method_cfg.get("freq_active_num_bases", self.shortcut_num_bases))
        self.adaptive_freq_warmup_epochs = int(method_cfg.get("adaptive_freq_warmup_epochs", 5))
        self.adaptive_freq_explore_mode = str(method_cfg.get("adaptive_freq_explore_mode", "round_robin"))
        self.adaptive_freq_select_metric = str(method_cfg.get("adaptive_freq_select_metric", "grad_norm"))
        self.adaptive_freq_update_once = bool(method_cfg.get("adaptive_freq_update_once", True))
        self.freq_basis_seed = int(method_cfg.get("freq_basis_seed", 0))
        self.freq_score_eps = float(method_cfg.get("freq_score_eps", 1.0e-6))
        self.freq_score_min_attack_grad = float(method_cfg.get("freq_score_min_attack_grad", 1.0e-7))
        self.freq_score_use_logits_collateral = bool(method_cfg.get("freq_score_use_logits_collateral", True))
        self.freq_score_use_feat_collateral = bool(method_cfg.get("freq_score_use_feat_collateral", False))
        self.freq_score_use_margin_collateral = bool(method_cfg.get("freq_score_use_margin_collateral", False))
        self.freq_score_attack_use_weighted_loss = bool(method_cfg.get("freq_score_attack_use_weighted_loss", True))
        self.freq_score_collateral_use_weighted_loss = bool(
            method_cfg.get("freq_score_collateral_use_weighted_loss", True)
        )
        self.enable_cooccur_nt_logits_preserve = bool(
            method_cfg.get("enable_cooccur_nt_logits_preserve", False)
        )
        self.cooccur_nt_logits_lambda = float(method_cfg.get("cooccur_nt_logits_lambda", 0.10))
        self.cooccur_nt_min_clean_conf = float(method_cfg.get("cooccur_nt_min_clean_conf", 0.20))
        self.cooccur_nt_only_on_real_fg = bool(method_cfg.get("cooccur_nt_only_on_real_fg", True))
        self.cooccur_nt_assigned_class_only = bool(method_cfg.get("cooccur_nt_assigned_class_only", True))
        self.cooccur_nt_exclude_target_class = bool(method_cfg.get("cooccur_nt_exclude_target_class", True))
        self.cooccur_nt_warmup_start_epoch = int(method_cfg.get("cooccur_nt_warmup_start_epoch", 10))
        self.cooccur_nt_warmup_end_epoch = int(method_cfg.get("cooccur_nt_warmup_end_epoch", 25))
        self.cooccur_nt_apply_after_freq_selection = bool(
            method_cfg.get("cooccur_nt_apply_after_freq_selection", True)
        )
        self.cooccur_nt_do_not_use_for_freq_score = bool(
            method_cfg.get("cooccur_nt_do_not_use_for_freq_score", True)
        )
        self.cooccur_nt_loss_type = str(method_cfg.get("cooccur_nt_loss_type", "smooth_l1_prob"))
        self.cooccur_nt_drop_tolerance = float(method_cfg.get("cooccur_nt_drop_tolerance", 0.005))
        self.cooccur_nt_hard_top_ratio = float(method_cfg.get("cooccur_nt_hard_top_ratio", 1.0))
        self.cooccur_nt_min_hard_count = int(method_cfg.get("cooccur_nt_min_hard_count", 16))
        self.cooccur_nt_use_positive_drop_only = bool(
            method_cfg.get("cooccur_nt_use_positive_drop_only", True)
        )
        self.cooccur_nt_hard_top_ratio = min(1.0, max(0.0, self.cooccur_nt_hard_top_ratio))
        self.cooccur_nt_min_hard_count = max(1, self.cooccur_nt_min_hard_count)
        self.enable_late_nt_repair = bool(method_cfg.get("enable_late_nt_repair", False))
        self.late_repair_start_epoch = int(method_cfg.get("late_repairQ_start_epoch", 20))
        self.late_repair_ramp_epochs = int(method_cfg.get("late_repair_ramp_epochs", 5))
        self.late_attack_loss_scale_enabled = bool(method_cfg.get("late_attack_loss_scale_enabled", True))
        self.late_lambda_ent_scale = float(method_cfg.get("late_lambda_ent_scale", 0.50))
        self.late_lambda_anchor_scale = float(method_cfg.get("late_lambda_anchor_scale", 0.80))
        self.late_lambda_flat_scale = float(method_cfg.get("late_lambda_flat_scale", 0.80))
        self.late_preserve_logits_scale = float(method_cfg.get("late_preserve_logits_scale", 2.20))
        self.late_preserve_feat_scale = float(method_cfg.get("late_preserve_feat_scale", 1.0))
        self.late_margin_scale = float(method_cfg.get("late_margin_scale", 1.0))
        self.late_cooccur_lambda_scale = float(method_cfg.get("late_cooccur_lambda_scale", 1.0))
        self.late_only_after_freq_selection = bool(method_cfg.get("late_only_after_freq_selection", True))
        self.late_repair_ramp_epochs = max(1, self.late_repair_ramp_epochs)
        self.enable_band_aware_freq_basis = bool(method_cfg.get("enable_band_aware_freq_basis", False))
        self.freq_band_names = [str(x) for x in method_cfg.get("freq_band_names", ["low", "mid", "high"])]
        self.freq_band_candidate_nums = [int(x) for x in method_cfg.get("freq_band_candidate_nums", [16, 32, 16])]
        self.freq_band_active_nums = [int(x) for x in method_cfg.get("freq_band_active_nums", [1, 13, 2])]
        self.freq_band_radius_ranges = method_cfg.get(
            "freq_band_radius_ranges",
            {
                "low": [2, 8],
                "mid": [8, 24],
                "high": [24, 48],
            },
        )
        self.freq_band_adaptive_quota = bool(method_cfg.get("freq_band_adaptive_quota", False))
        self.freq_band_active_min = [int(x) for x in method_cfg.get("freq_band_active_min", [0, 12, 1])]
        self.freq_band_active_max = [int(x) for x in method_cfg.get("freq_band_active_max", [3, 16, 4])]
        self.freq_candidate_meta: List[Dict[str, Any]] = []
        self.freq_band_to_indices: Dict[str, List[int]] = {}
        self._freq_band_radius_mean: Dict[str, float] = {}

        if self.enable_adaptive_freq_basis:
            if self.freq_active_num_bases != self.shortcut_num_bases:
                raise ValueError(
                    f"freq_active_num_bases ({self.freq_active_num_bases}) must equal "
                    f"shortcut_num_bases ({self.shortcut_num_bases}) when adaptive basis is enabled."
                )
            if self.freq_candidate_num_bases < self.freq_active_num_bases:
                raise ValueError(
                    f"freq_candidate_num_bases ({self.freq_candidate_num_bases}) must be >= "
                    f"freq_active_num_bases ({self.freq_active_num_bases})."
                )
            if self.enable_band_aware_freq_basis:
                if len(set(self.freq_band_names)) != len(self.freq_band_names):
                    raise ValueError("freq_band_names must be unique for band-aware adaptive frequency basis.")
                if len(self.freq_band_names) != len(self.freq_band_candidate_nums):
                    raise ValueError("freq_band_names and freq_band_candidate_nums must have the same length.")
                if len(self.freq_band_names) != len(self.freq_band_active_nums):
                    raise ValueError("freq_band_names and freq_band_active_nums must have the same length.")
                if not isinstance(self.freq_band_radius_ranges, dict):
                    raise ValueError("freq_band_radius_ranges must be a dict when enable_band_aware_freq_basis=true.")
                if sum(self.freq_band_candidate_nums) != self.freq_candidate_num_bases:
                    raise ValueError(
                        f"sum(freq_band_candidate_nums) ({sum(self.freq_band_candidate_nums)}) must equal "
                        f"freq_candidate_num_bases ({self.freq_candidate_num_bases})."
                    )
                if sum(self.freq_band_active_nums) != self.freq_active_num_bases:
                    raise ValueError(
                        f"sum(freq_band_active_nums) ({sum(self.freq_band_active_nums)}) must equal "
                        f"freq_active_num_bases ({self.freq_active_num_bases})."
                    )
                for band_name, cand_k, act_k in zip(
                    self.freq_band_names, self.freq_band_candidate_nums, self.freq_band_active_nums
                ):
                    if cand_k <= 0:
                        raise ValueError(
                            f"freq_band_candidate_nums for band '{band_name}' must be > 0, got {cand_k}."
                        )
                    if act_k < 0:
                        raise ValueError(f"freq_band_active_nums for band '{band_name}' must be >= 0, got {act_k}.")
                    if act_k > cand_k:
                        raise ValueError(
                            f"freq_band_active_nums for band '{band_name}' ({act_k}) must be <= "
                            f"freq_band_candidate_nums ({cand_k})."
                        )
                    rr = self.freq_band_radius_ranges.get(band_name)
                    if rr is None or len(rr) != 2:
                        raise ValueError(f"Missing or invalid freq_band_radius_ranges for band '{band_name}'.")
                band_coords, band_meta = sample_bandfreq_coords(
                    h=self.imgsz,
                    w=self.imgsz,
                    band_names=self.freq_band_names,
                    band_num_bases=self.freq_band_candidate_nums,
                    band_radius_ranges=self.freq_band_radius_ranges,
                    seed=self.freq_basis_seed,
                    enable_search=True,
                )
                if len(band_coords) != self.freq_candidate_num_bases:
                    raise RuntimeError(
                        f"band-aware candidate generation returned {len(band_coords)} coords, "
                        f"expected {self.freq_candidate_num_bases}."
                    )
                if len(band_meta) != len(band_coords):
                    raise RuntimeError("band-aware candidate meta length mismatch.")
                self.freq_candidate_coords = [tuple(map(int, t)) for t in band_coords]
                self.freq_band_to_indices = {name: [] for name in self.freq_band_names}
                self.freq_candidate_meta = []
                for idx, meta in enumerate(band_meta):
                    band_name = str(meta.get("band", "mid"))
                    y = int(meta.get("y", 0))
                    x = int(meta.get("x", 0))
                    radius = float(meta.get("radius", 0.0))
                    local_index = len(self.freq_band_to_indices.get(band_name, []))
                    if band_name not in self.freq_band_to_indices:
                        self.freq_band_to_indices[band_name] = []
                    self.freq_band_to_indices[band_name].append(int(idx))
                    self.freq_candidate_meta.append(
                        {
                            "index": int(idx),
                            "band": band_name,
                            "band_id": int(self.freq_band_names.index(band_name)),
                            "local_index": int(local_index),
                            "y": y,
                            "x": x,
                            "radius": radius,
                        }
                    )
                self._freq_band_radius_mean = {}
                for band_name in self.freq_band_names:
                    bidx = self.freq_band_to_indices.get(band_name, [])
                    if not bidx:
                        self._freq_band_radius_mean[band_name] = float("nan")
                        continue
                    r_vals = [float(self.freq_candidate_meta[i]["radius"]) for i in bidx]
                    self._freq_band_radius_mean[band_name] = float(np.mean(r_vals)) if r_vals else float("nan")
            else:
                self.freq_candidate_coords = sample_midfreq_coords(
                    h=self.imgsz,
                    w=self.imgsz,
                    num_bases=self.freq_candidate_num_bases,
                    seed=self.freq_basis_seed,
                    enable_search=True,
                )
                self.freq_band_to_indices = {}
                self.freq_candidate_meta = []
            self.fourier_coeff = torch.nn.Parameter(
                torch.zeros((self.freq_candidate_num_bases, 3), device=self.device)
            )
            if self.enable_band_aware_freq_basis:
                init_active_idx: List[int] = []
                for band_name, k in zip(self.freq_band_names, self.freq_band_active_nums):
                    band_idx = self.freq_band_to_indices.get(band_name, [])
                    init_active_idx.extend([int(i) for i in band_idx[: int(k)]])
                if len(init_active_idx) != self.freq_active_num_bases:
                    raise RuntimeError(
                        f"band-aware init active count mismatch: {len(init_active_idx)} vs {self.freq_active_num_bases}"
                    )
                self.freq_active_idx = torch.tensor(
                    sorted(init_active_idx), device=self.device, dtype=torch.long
                )
            else:
                self.freq_active_idx = torch.arange(self.freq_active_num_bases, device=self.device, dtype=torch.long)
            self.freq_score = torch.zeros((self.freq_candidate_num_bases,), device=self.device, dtype=torch.float32)
            self.freq_usage = torch.zeros((self.freq_candidate_num_bases,), device=self.device, dtype=torch.float32)
            self._adaptive_freq_grad_warned = False
            self._adaptive_freq_score_top_mean = float("nan")
            self._adaptive_freq_mode_warned = False
            self._adaptive_freq_metric_warned = False
            self.coords = [tuple(self.freq_candidate_coords[int(i)]) for i in self.freq_active_idx.detach().cpu().tolist()]
        else:
            self.coords = sample_midfreq_coords(
                h=self.imgsz,
                w=self.imgsz,
                num_bases=self.shortcut_num_bases,
                seed=int(cfg.get("experiment", {}).get("seeds", [0])[0]),
                enable_search=True,
            )
            self.fourier_coeff = torch.nn.Parameter(torch.zeros((self.shortcut_num_bases, 3), device=self.device))
            self.freq_candidate_coords = self.coords
            self.freq_candidate_num_bases = self.shortcut_num_bases
            self.freq_active_num_bases = self.shortcut_num_bases
            self.freq_active_idx = torch.arange(self.shortcut_num_bases, device=self.device, dtype=torch.long)
            self.freq_score = torch.zeros((self.shortcut_num_bases,), device=self.device, dtype=torch.float32)
            self.freq_usage = torch.zeros((self.shortcut_num_bases,), device=self.device, dtype=torch.float32)
            self._adaptive_freq_grad_warned = False
            self._adaptive_freq_score_top_mean = float("nan")
            self._adaptive_freq_mode_warned = False
            self._adaptive_freq_metric_warned = False
            self.enable_band_aware_freq_basis = False

        self.freq_score_eps = max(self.freq_score_eps, 1.0e-12)
        self.freq_score_min_attack_grad = max(self.freq_score_min_attack_grad, 0.0)
        self.freq_attack_grad_score = torch.zeros(
            (self.freq_candidate_num_bases,), device=self.device, dtype=torch.float32
        )
        self.freq_collateral_grad_score = torch.zeros(
            (self.freq_candidate_num_bases,), device=self.device, dtype=torch.float32
        )
        self.freq_ratio_score = torch.zeros(
            (self.freq_candidate_num_bases,), device=self.device, dtype=torch.float32
        )

        self.suppress_small = torch.nn.Parameter(
            torch.zeros((1, 3, self.suppress_small_size, self.suppress_small_size), device=self.device)
        )

        self.shadow_tal = DifferentiableShadowTAL(
            target_class_id=self.target_class_id,
            alpha=self.align_alpha,
            beta=self.align_beta,
            topk=self.assignment_topk,
        )
        self.hijacked = HijackedV8Loss.from_surrogate(
            surrogate,
            num_classes=self.num_classes,
            target_class_id=self.target_class_id,
        )

    def _split_indices_by_band(self, indices: torch.Tensor) -> Dict[str, List[int]]:
        out = {name: [] for name in self.freq_band_names}
        if not self.enable_band_aware_freq_basis:
            return out

        idx_set = set(int(x) for x in indices.detach().cpu().tolist())
        for band_name in self.freq_band_names:
            band_idx = self.freq_band_to_indices.get(band_name, [])
            out[band_name] = [int(i) for i in band_idx if int(i) in idx_set]
        return out

    def _get_band_round_robin_indices(self, step: int) -> torch.Tensor:
        selected: List[torch.Tensor] = []
        for band_name, k in zip(self.freq_band_names, self.freq_band_active_nums):
            act_k = int(k)
            if act_k <= 0:
                continue
            band_indices = self.freq_band_to_indices.get(band_name, [])
            m = len(band_indices)
            if m == 0:
                raise RuntimeError(f"Band '{band_name}' has no candidate frequencies.")
            if act_k > m:
                raise RuntimeError(
                    f"Band '{band_name}' active quota ({act_k}) cannot exceed candidate num ({m})."
                )
            band_tensor = torch.tensor(band_indices, device=self.device, dtype=torch.long)
            start = (step * act_k) % m
            local = (torch.arange(act_k, device=self.device) + start) % m
            selected.append(band_tensor[local.long()])

        if not selected:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        out = torch.cat(selected, dim=0).long()
        if out.numel() != self.freq_active_num_bases:
            raise RuntimeError(
                f"Band round-robin produced {out.numel()} active idx, expected {self.freq_active_num_bases}."
            )
        return out

    def _select_topk_per_band(self, score: torch.Tensor) -> torch.Tensor:
        selected: List[torch.Tensor] = []
        for band_name, k in zip(self.freq_band_names, self.freq_band_active_nums):
            act_k = int(k)
            if act_k <= 0:
                continue
            band_indices = self.freq_band_to_indices.get(band_name, [])
            if not band_indices:
                raise RuntimeError(f"Band '{band_name}' has no candidate frequencies.")
            band_tensor = torch.tensor(band_indices, device=self.device, dtype=torch.long)
            if act_k > band_tensor.numel():
                raise RuntimeError(
                    f"Band '{band_name}' active quota ({act_k}) cannot exceed candidate num ({band_tensor.numel()})."
                )
            band_score = score[band_tensor]
            top_local = torch.topk(band_score, k=act_k, largest=True).indices
            selected.append(band_tensor[top_local.long()])

        if not selected:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        out = torch.cat(selected, dim=0).long()
        if out.numel() != self.freq_active_num_bases:
            raise RuntimeError(
                f"Band top-k produced {out.numel()} active idx, expected {self.freq_active_num_bases}."
            )
        return out

    def _compute_band_score_stats(self, score: torch.Tensor) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        if not self.enable_band_aware_freq_basis:
            return stats
        for band_name, act_k in zip(self.freq_band_names, self.freq_band_active_nums):
            band_indices = self.freq_band_to_indices.get(band_name, [])
            if not band_indices:
                stats[band_name] = {"mean": float("nan"), "top_mean": float("nan")}
                continue
            band_tensor = torch.tensor(band_indices, device=self.device, dtype=torch.long)
            band_score = score[band_tensor]
            mean_val = float(band_score.mean().item())
            if int(act_k) <= 0:
                top_val = float("nan")
            else:
                top_k = int(min(int(act_k), int(band_score.numel())))
                top_val = float(torch.topk(band_score, k=top_k, largest=True).values.mean().item())
            stats[band_name] = {"mean": mean_val, "top_mean": top_val}
        return stats

    @staticmethod
    def _topk_mean(score: torch.Tensor, top_idx: torch.Tensor) -> float:
        if score.numel() == 0 or top_idx.numel() == 0:
            return float("nan")
        return float(score[top_idx.long()].mean().item())

    def _update_adaptive_freq_score(
        self,
        active_idx: torch.Tensor,
        L_attack_score: torch.Tensor,
        L_collateral_score: torch.Tensor,
    ) -> None:
        if not self.enable_adaptive_freq_basis:
            return
        if self.adaptive_freq_select_metric != "attack_collateral_ratio":
            return

        attack_grad = None
        if torch.is_tensor(L_attack_score) and L_attack_score.requires_grad:
            attack_grad = torch.autograd.grad(
                L_attack_score,
                self.fourier_coeff,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if attack_grad is None:
            attack_grad = torch.zeros_like(self.fourier_coeff)

        collateral_grad = None
        if torch.is_tensor(L_collateral_score) and L_collateral_score.requires_grad:
            collateral_grad = torch.autograd.grad(
                L_collateral_score,
                self.fourier_coeff,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if collateral_grad is None:
            collateral_grad = torch.zeros_like(attack_grad)

        reduce_dims = tuple(range(1, attack_grad.ndim))
        attack_score = attack_grad.detach().abs().mean(dim=reduce_dims)
        collateral_score = collateral_grad.detach().abs().mean(dim=reduce_dims)
        ratio_score = attack_score / (collateral_score + self.freq_score_eps)
        ratio_score = torch.where(attack_score >= self.freq_score_min_attack_grad, ratio_score, torch.zeros_like(ratio_score))

        active_idx = active_idx.long()
        self.freq_attack_grad_score[active_idx] += attack_score[active_idx]
        self.freq_collateral_grad_score[active_idx] += collateral_score[active_idx]
        self.freq_ratio_score[active_idx] += ratio_score[active_idx]
        self.freq_score[active_idx] += ratio_score[active_idx]
        self.freq_usage[active_idx] += 1.0

    def _get_active_freq_indices(self, epoch: int, step: int) -> torch.Tensor:
        if not self.enable_adaptive_freq_basis:
            return torch.arange(self.shortcut_num_bases, device=self.device)

        if epoch < self.adaptive_freq_warmup_epochs:
            if self.enable_band_aware_freq_basis:
                if self.adaptive_freq_explore_mode not in {"band_round_robin", "round_robin"} and not self._adaptive_freq_mode_warned:
                    print(
                        f"[AdaptiveFreq][Warning] unsupported adaptive_freq_explore_mode="
                        f"{self.adaptive_freq_explore_mode}, fallback to band_round_robin."
                    )
                    self._adaptive_freq_mode_warned = True
                return self._get_band_round_robin_indices(step=step)
            else:
                if self.adaptive_freq_explore_mode != "round_robin" and not self._adaptive_freq_mode_warned:
                    print(
                        f"[AdaptiveFreq][Warning] unsupported adaptive_freq_explore_mode="
                        f"{self.adaptive_freq_explore_mode}, fallback to round_robin."
                    )
                    self._adaptive_freq_mode_warned = True
                k = self.freq_active_num_bases
                m = self.freq_candidate_num_bases
                start = (step * k) % m
                idx = (torch.arange(k, device=self.device) + start) % m
                return idx.long()

        return self.freq_active_idx.long()

    def _collect_target_images(self, train_img_dir: str, train_label_dir: str) -> List[str]:
        all_images = list_images(train_img_dir)
        out = []
        for p in all_images:
            anns = read_yolo_annotations(label_path_for_image(p, train_label_dir))
            if image_has_target(anns, self.target_class_id):
                out.append(p)
        return out

    def _append_diag_row(self, csv_path: str, row: Dict):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        exists = os.path.isfile(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

    def train_universal(
        self,
        train_img_dir: str,
        train_label_dir: str,
        global_params_path: str,
        diagnostics_csv_path: str,
        diagnostics_json_path: str,
        seed: int,
    ) -> str:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        target_images = self._collect_target_images(train_img_dir, train_label_dir)
        if not target_images:
            raise RuntimeError("No target-class training images found for TAUS-B universal training.")

        import cv2
        cv2.setNumThreads(0)

        optimizer = torch.optim.Adam(
            [
                {"params": [self.fourier_coeff], "lr": self.universal_lr_fourier},
            ]
        )

        run_epochs = self.universal_epochs
        if self.phase0_diagnostics_only:
            run_epochs = min(self.universal_epochs, max(1, self.phase0_max_epochs))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max(1, run_epochs), 
            eta_min=1e-5
        )

        dataset = TAUSBDataset(
            target_images,
            train_label_dir,
            self.target_class_id,
            instance_mask_dir=self.instance_mask_dir,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.universal_batch_size,
            num_workers=16,          
            pin_memory=True,        
            persistent_workers=True,
            shuffle=True,
            collate_fn=tausb_collate_fn
        )

        global_step = 0
        latest_diag = {}

        for epoch in range(run_epochs):
            if epoch < 15:
                cur_lambda_preserve = self.lambda_preserve * 0.3
            elif epoch < 25:
                warmup_ratio = 0.3 + min(0.7, (epoch - 15) / 10.0 * 0.7)
                cur_lambda_preserve = self.lambda_preserve * warmup_ratio
            else:
                cur_lambda_preserve = self.lambda_preserve
                warmup_ratio = min(1.0, (epoch - 25) / 5.0)
            if not self.enable_cooccur_nt_logits_preserve:
                cooccur_lambda_cur = 0.0
            elif self.cooccur_nt_apply_after_freq_selection and epoch < self.adaptive_freq_warmup_epochs:
                cooccur_lambda_cur = 0.0
            elif epoch < self.cooccur_nt_warmup_start_epoch:
                cooccur_lambda_cur = 0.0
            elif epoch >= self.cooccur_nt_warmup_end_epoch:
                cooccur_lambda_cur = self.cooccur_nt_logits_lambda
            else:
                r = (
                    (epoch - self.cooccur_nt_warmup_start_epoch)
                    / max(1, self.cooccur_nt_warmup_end_epoch - self.cooccur_nt_warmup_start_epoch)
                )
                cooccur_lambda_cur = self.cooccur_nt_logits_lambda * float(r)

            late_repair_ratio = 0.0
            if self.enable_late_nt_repair:
                can_apply_late = True
                if self.late_only_after_freq_selection and epoch < self.adaptive_freq_warmup_epochs:
                    can_apply_late = False
                if can_apply_late and epoch >= self.late_repair_start_epoch:
                    late_repair_ratio = min(
                        1.0,
                        float(epoch - self.late_repair_start_epoch + 1)
                        / float(max(1, self.late_repair_ramp_epochs)),
                    )

            if self.enable_late_nt_repair and late_repair_ratio > 0:
                if self.late_attack_loss_scale_enabled:
                    ent_scale_cur = 1.0 + late_repair_ratio * (self.late_lambda_ent_scale - 1.0)
                    anchor_scale_cur = 1.0 + late_repair_ratio * (self.late_lambda_anchor_scale - 1.0)
                    flat_scale_cur = 1.0 + late_repair_ratio * (self.late_lambda_flat_scale - 1.0)
                else:
                    ent_scale_cur = 1.0
                    anchor_scale_cur = 1.0
                    flat_scale_cur = 1.0
                preserve_logits_scale_cur = 1.0 + late_repair_ratio * (self.late_preserve_logits_scale - 1.0)
                preserve_feat_scale_cur = 1.0 + late_repair_ratio * (self.late_preserve_feat_scale - 1.0)
                margin_scale_cur = 1.0 + late_repair_ratio * (self.late_margin_scale - 1.0)
                cooccur_lambda_scale_cur = 1.0 + late_repair_ratio * (self.late_cooccur_lambda_scale - 1.0)
            else:
                ent_scale_cur = 1.0
                anchor_scale_cur = 1.0
                flat_scale_cur = 1.0
                preserve_logits_scale_cur = 1.0
                preserve_feat_scale_cur = 1.0
                margin_scale_cur = 1.0
                cooccur_lambda_scale_cur = 1.0

            effective_lambda_ent = self.lambda_ent * ent_scale_cur
            effective_lambda_anchor = self.lambda_anchor * anchor_scale_cur
            effective_lambda_flat = self.lambda_flat * flat_scale_cur
            effective_cooccur_lambda = cooccur_lambda_cur * cooccur_lambda_scale_cur

            epoch_diag = {
                "align_clean_topk": 0.0,
                "align_adv_topk": 0.0,
                "L_entangle_bg": 0.0,
                "L_anchor": 0.0,
                "L_collapse_aux": 0.0,
                "alsi_score": 0.0,
                "cos_t_conf": 0.0,
                "cos_t_clean": 0.0,
                "M_conf_sum": 0.0,
                "rlcp_conf_mass": 0.0,
                "rlcp_trim_keep_ratio": 0.0,
                "rlcp_core_exclusion_ratio": 0.0,
                "rlcp_adaptive_inner_mean": 0.0,
                "rlcp_adaptive_outer_mean": 0.0,
                "rlcp_adaptive_inner_min": 0.0,
                "rlcp_adaptive_outer_max": 0.0,
                "confounder_purity_ratio": 0.0,
                "overlap_ratio": 0.0,
                "preserve_loss": 0.0,
                "L_margin": 0.0,
                "margin_clean_mean": 0.0,
                "margin_adv_mean": 0.0,
                "gate_positive_ratio": 0.0,
                "pag_positive_ratio": 0.0,
                "pag_threshold": 0.0,
                "pag_mean_target_score": 0.0,
                "pag_fallback_count": 0.0,
                "perturbed_area_ratio_mean": 0.0,
                "L_tv": 0.0,
                "L_budget": 0.0,
                "L_total": 0.0,
                "L_tsvc": 0.0,
                "L_sem": 0.0,
                "sld_score": 0.0,
                "mean_abs_delta": 0.0,
                "max_abs_delta": 0.0,
                "saturation_ratio": 0.0,
                "support_prebaked_ratio": 0.0,
                "support_fallback_ratio": 0.0,
                "support_forced_pseudo_fallback_ratio": 0.0,
                "support_empty_ratio": 0.0,
                "L_cooccur_nt_logits": 0.0,
                "cooccur_nt_lambda_cur": 0.0,
                "cooccur_nt_fg_count": 0.0,
                "cooccur_nt_img_ratio": 0.0,
                "cooccur_nt_clean_prob_mean": 0.0,
                "cooccur_nt_adv_prob_mean": 0.0,
                "cooccur_nt_prob_drop": 0.0,
                "cooccur_nt_prob_drop_raw_mean": 0.0,
                "cooccur_nt_prob_drop_pos_mean": 0.0,
                "cooccur_nt_prob_drop_hinge_mean": 0.0,
                "cooccur_nt_hard_loss_mean": 0.0,
                "cooccur_nt_hard_count": 0.0,
                "late_repair_ratio": 0.0,
                "ent_scale_cur": 1.0,
                "anchor_scale_cur": 1.0,
                "flat_scale_cur": 1.0,
                "preserve_logits_scale_cur": 1.0,
                "preserve_feat_scale_cur": 1.0,
                "margin_scale_cur": 1.0,
                "cooccur_lambda_scale_cur": 1.0,
                "effective_lambda_ent": 0.0,
                "effective_lambda_anchor": 0.0,
                "effective_lambda_flat": 0.0,
                "effective_cooccur_lambda": 0.0,
                "cooccur_nt_loss_type": str(self.cooccur_nt_loss_type),
            }
            batch_count = 0

            for batch_list in loader:
                if not batch_list:
                    continue

                optimizer.zero_grad(set_to_none=True)

                valid_items = []
                batch_prebaked = 0
                batch_fallback = 0
                batch_forced_fallback = 0
                batch_empty = 0
                cooccur_img_flags = []
                
                for data in batch_list:
                    inner_np, ring_np, support_source = self._build_support(
                        image_shape=data["clean_np"].shape,
                        annotations=data["anns"],
                        support_type="mask",
                        ring_width=self.ring_width,
                        image_path=data["img_path"],
                    )
                    if support_source == "prebaked":
                        batch_prebaked += 1
                    elif support_source in ("pseudo_fallback", "forced_pseudo_fallback"):
                        batch_fallback += 1
                        if support_source == "forced_pseudo_fallback":
                            batch_forced_fallback += 1
                    else:
                        batch_empty += 1
                        
                    if float(inner_np.sum()) > 10.0:
                        valid_items.append((data, inner_np, ring_np, support_source))
                        has_target = any(
                            int(ann.get("cls", -1)) == self.target_class_id
                            for ann in data["anns"]
                        )
                        has_non_target = any(
                            int(ann.get("cls", -1)) != self.target_class_id
                            for ann in data["anns"]
                        )
                        cooccur_img_flags.append(bool(has_target and has_non_target))
                
                total_support_count = batch_prebaked + batch_fallback + batch_empty
                if total_support_count > 0:
                    epoch_diag["support_prebaked_ratio"] += batch_prebaked / total_support_count
                    epoch_diag["support_fallback_ratio"] += batch_fallback / total_support_count
                    epoch_diag["support_forced_pseudo_fallback_ratio"] += batch_forced_fallback / total_support_count
                    epoch_diag["support_empty_ratio"] += batch_empty / total_support_count

                if not valid_items:
                    continue

                max_h = max(item[0]["clean_np"].shape[0] for item in valid_items)
                max_w = max(item[0]["clean_np"].shape[1] for item in valid_items)
                pad_h_target = ((max_h + 31) // 32) * 32 
                pad_w_target = ((max_w + 31) // 32) * 32

                img_t_list, inner_t_list, ring_t_list = [], [], []
                bboxes, clss, batch_idx = [], [], []

                for i, (data, inner, ring, support_source) in enumerate(valid_items):
                    img_tsr = torch.from_numpy(data["clean_np"]).float().permute(2, 0, 1)
                    inner_tsr = torch.from_numpy(inner).float().unsqueeze(0)
                    ring_tsr = torch.from_numpy(ring).float().unsqueeze(0)

                    ch, cw = img_tsr.shape[1], img_tsr.shape[2]
                    pad_b = pad_h_target - ch
                    pad_r = pad_w_target - cw

                    if pad_b > 0 or pad_r > 0:
                        img_tsr = F.pad(img_tsr, (0, pad_r, 0, pad_b), value=0.5) 
                        inner_tsr = F.pad(inner_tsr, (0, pad_r, 0, pad_b), value=0.0)
                        ring_tsr = F.pad(ring_tsr, (0, pad_r, 0, pad_b), value=0.0)

                    img_t_list.append(img_tsr)
                    inner_t_list.append(inner_tsr)
                    ring_t_list.append(ring_tsr)

                    for ann in data["anns"]:
                        c = int(ann.get("cls", -1))
                        bb = ann.get("bbox", None)
                        if bb is not None and len(bb) == 4:
                            cx, cy, bw, bh = renorm_yolo_bbox_after_padding(
                                bb[0], bb[1], bb[2], bb[3],
                                orig_w=cw,
                                orig_h=ch,
                                pad_w=pad_w_target,
                                pad_h=pad_h_target,
                            )
                            bboxes.append([cx, cy, bw, bh])
                            clss.append([float(c)])
                            batch_idx.append(i)

                img_t = torch.stack(img_t_list).to(self.device, non_blocking=True)
                inner_t = torch.stack(inner_t_list).to(self.device, non_blocking=True)
                ring_t = torch.stack(ring_t_list).to(self.device, non_blocking=True)
                cooccur_img_mask = torch.tensor(
                    cooccur_img_flags,
                    dtype=torch.bool,
                    device=self.device,
                )
                
                batch_h, batch_w = img_t.shape[-2:]

                single_batch = {
                    "batch_idx": torch.tensor(batch_idx, dtype=torch.long, device=self.device) if batch_idx else torch.zeros((0,), dtype=torch.long, device=self.device),
                    "cls": torch.tensor(clss, dtype=torch.float32, device=self.device) if clss else torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    "bboxes": torch.tensor(bboxes, dtype=torch.float32, device=self.device) if bboxes else torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    "batch_size": len(valid_items),
                    "img": img_t
                }

                M_non_target_core = build_non_target_core_mask(
                    batch=single_batch,
                    pad_h=pad_h_target,
                    pad_w=pad_w_target,
                    target_class_id=self.target_class_id,
                    device=self.device,
                    core_scale=self.rlcp_core_scale,
                ).to(dtype=inner_t.dtype)
                
                if self.use_adaptive_context:
                    local_ctx_t, rlcp_adapt_stats = build_scale_adaptive_context_mask(
                        batch=single_batch,
                        pad_h=pad_h_target,
                        pad_w=pad_w_target,
                        target_class_id=self.target_class_id,
                        device=self.device,
                        alpha=self.rlcp_adaptive_alpha,
                        beta=self.rlcp_adaptive_beta,
                        inner_min=self.rlcp_adaptive_inner_min,
                        inner_max=self.rlcp_adaptive_inner_max,
                        outer_min=self.rlcp_adaptive_outer_min,
                        outer_max=self.rlcp_adaptive_outer_max,
                        min_gap=self.rlcp_adaptive_min_gap,
                    )
                    local_ctx_t = local_ctx_t.to(dtype=inner_t.dtype)
                else:
                    local_ctx_t = build_local_context_mask(
                        inner_mask=inner_t,
                        r_inner=12,
                        r_outer=28,
                    )
                    rlcp_adapt_stats = {
                        "rlcp_adaptive_inner_mean": 0.0,
                        "rlcp_adaptive_outer_mean": 0.0,
                        "rlcp_adaptive_inner_min": 0.0,
                        "rlcp_adaptive_outer_max": 0.0,
                    }
                
                conf_t = build_confounder_mask(
                    local_ctx_mask=local_ctx_t,
                    all_objects_mask=M_non_target_core,
                    ring_mask=ring_t,
                )

                

                epoch_diag["rlcp_conf_mass"] += float(conf_t.sum().item()) 
                epoch_diag["rlcp_adaptive_inner_mean"] += float(
                    rlcp_adapt_stats.get("rlcp_adaptive_inner_mean", 0.0)
                )
                epoch_diag["rlcp_adaptive_outer_mean"] += float(
                    rlcp_adapt_stats.get("rlcp_adaptive_outer_mean", 0.0)
                )
                epoch_diag["rlcp_adaptive_inner_min"] += float(
                    rlcp_adapt_stats.get("rlcp_adaptive_inner_min", 0.0)
                )
                epoch_diag["rlcp_adaptive_outer_max"] += float(
                    rlcp_adapt_stats.get("rlcp_adaptive_outer_max", 0.0)
                )

                if not getattr(self, "_sanity_mask_space_printed", False):
                    inner_sum = inner_t.sum().item()
                    ring_sum = ring_t.sum().item()
                    non_obj_sum = M_non_target_core.sum().item()
                    local_ctx_sum = local_ctx_t.sum().item()
                    conf_raw_sum = conf_t.sum().item()
                    
                    removed_by_objects = (local_ctx_t * M_non_target_core).sum().item()
                    removed_by_ring = (local_ctx_t * ring_t).sum().item()
                    core_exclusion_ratio = removed_by_objects / (local_ctx_sum + 1e-6)
                    
                    print(f"\n🔬 [Mask Space Diagnostics] (Mid-range Annulus)")
                    print(f"   Target Inner Sum: {inner_sum:.1f}")
                    print(f"   Target Ring Sum: {ring_sum:.1f}")
                    print(f"   Non-Target Core Sum: {non_obj_sum:.1f}")
                    print(f"   Local Ctx Annulus Sum: {local_ctx_sum:.1f}")
                    print(f"   Final M_conf Sum: {conf_raw_sum:.1f}")
                    print(f"   ----------------------------------")
                    print(f"   ⚠️ Removed by Non-Target Objects: {removed_by_objects:.1f}")
                    print(f"   ⚠️ Removed by Ring (Should be 0 now): {removed_by_ring:.1f}")
                    print(f"   🔥 Raw Purity Ratio: {(conf_raw_sum / (local_ctx_sum + 1e-6)):.2%}")
                    print(f"   rlcp_core_exclusion_ratio: {core_exclusion_ratio:.2%}")
                    print(f"   context_mode: {'adaptive' if self.use_adaptive_context else 'fixed'}")
                    print(f"   rlcp_adaptive_inner_mean: {rlcp_adapt_stats.get('rlcp_adaptive_inner_mean', 0.0):.3f}")
                    print(f"   rlcp_adaptive_outer_mean: {rlcp_adapt_stats.get('rlcp_adaptive_outer_mean', 0.0):.3f}")
                    print(f"   rlcp_adaptive_inner_min: {rlcp_adapt_stats.get('rlcp_adaptive_inner_min', 0.0):.3f}")
                    print(f"   rlcp_adaptive_outer_max: {rlcp_adapt_stats.get('rlcp_adaptive_outer_max', 0.0):.3f}")
                    print(f"   rlcp_adaptive_inner_max: {rlcp_adapt_stats.get('rlcp_adaptive_inner_max', 0.0):.3f}")
                    print(f"   rlcp_adaptive_outer_min: {rlcp_adapt_stats.get('rlcp_adaptive_outer_min', 0.0):.3f}")
                    self._sanity_mask_space_printed = True

                active_idx = self._get_active_freq_indices(epoch=epoch, step=global_step)
                if self.enable_adaptive_freq_basis:
                    active_idx_list = active_idx.detach().cpu().tolist()
                    coords_active = [tuple(self.freq_candidate_coords[int(i)]) for i in active_idx_list]
                    coeff_active = self.fourier_coeff[active_idx]
                else:
                    coords_active = self.coords
                    coeff_active = self.fourier_coeff

                raw_perturb, perturb, adv, _support, _jnd = self._compose_delta_batched(
                    img_t=img_t,
                    inner_t=inner_t,
                    ring_t=ring_t,
                    coords=coords_active,
                    fourier_coeff=coeff_active,
                    suppress_small=self.suppress_small,
                    current_epoch=epoch  
                )

                eot_count = max(1, self.eot_samples)
                
                L_ent_total = torch.zeros((), device=self.device)
                L_anchor_total = torch.zeros((), device=self.device)
                L_flat_total = torch.zeros((), device=self.device)
                L_preserve_total = torch.zeros((), device=self.device)
                L_preserve_logits_total = torch.zeros((), device=self.device)
                L_preserve_feat_total = torch.zeros((), device=self.device)
                L_margin_total = torch.zeros((), device=self.device)
                L_cooccur_nt_total = torch.zeros((), device=self.device)
                
                align_clean_topk_acc = 0.0
                align_adv_topk_acc = 0.0
                gate_ratio_acc = 0.0
                batch_alsi_score_acc = 0.0
                layer_overlap_vals = []
                layer_conf_sum_vals = []
                layer_conf_purity_vals = []
                layer_cos_conf_vals = []
                layer_cos_clean_vals = []
                layer_trim_keep_vals = []
                layer_core_exclusion_vals = []
                pag_ratio_vals = []
                pag_threshold_vals = []
                pag_mean_score_vals = []
                margin_loss_vals = []
                margin_clean_vals = []
                margin_adv_vals = []
                pag_fallback_vals = []
                cooccur_nt_fg_count_vals = []
                cooccur_nt_clean_prob_vals = []
                cooccur_nt_adv_prob_vals = []
                cooccur_nt_prob_drop_vals = []
                cooccur_nt_img_ratio_vals = []
                cooccur_nt_prob_drop_raw_vals = []
                cooccur_nt_prob_drop_pos_vals = []
                cooccur_nt_prob_drop_hinge_vals = []
                cooccur_nt_hard_loss_vals = []
                cooccur_nt_hard_count_vals = []

                for eot_idx in range(eot_count):
                    clean_aug, adv_aug = self._apply_shared_eot_pair_batched(img_t, adv)

                    with torch.no_grad():
                        self._clear_multi_features()
                        preds_clean = self._forward_raw(clean_aug)
                        features_clean_cache = {k: v.detach() for k, v in self.multi_features.items()}

                        single_batch_probe = dict(single_batch)
                        single_batch_probe["image_shape"] = (batch_h, batch_w)
                        
                        self.hijacked.last_real_assign = {}

                        _ = self.hijacked.get_assigned_targets_and_loss(
                            preds_clean,
                            single_batch_probe,
                        )
                        
                        fg_cached = self.hijacked.last_real_assign.get("fg_mask", None)
                        lbl_cached = self.hijacked.last_real_assign.get("target_labels", None)
                        assert torch.is_tensor(fg_cached), "Fatal: Assigner failed to return tensor fg_mask"
                        assert torch.is_tensor(lbl_cached), "Fatal: Assigner failed to return tensor target_labels"
                        assert fg_cached.shape[0] == img_t.shape[0], "Fatal: Batch dimension mismatch in last_real_assign"

                        cache_clean = self.hijacked.cache_assign_inputs_only(
                            preds=preds_clean,
                            batch=single_batch,
                            image_shape=(batch_h, batch_w),
                            assignment_topk=self.assignment_topk,
                        )

                    if not cache_clean:
                        continue

                    gate = cache_clean.get("fg_mask", None)
                    if gate is not None and gate.numel() > 0:
                        shadow_clean = self.shadow_tal(
                            pred_scores_logits=cache_clean["pred_scores_logits"],
                            pred_bboxes=cache_clean["pred_bboxes"],
                            gt_labels=cache_clean["gt_labels"],
                            gt_bboxes=cache_clean["gt_bboxes"],
                            mask_gt=cache_clean["mask_gt"],
                            gate=gate,
                            topk=self.assignment_topk,
                        )
                        align_clean_topk_acc += float(shadow_clean["topk_alignment"].mean().item())
                        gate_ratio_acc += float(shadow_clean["gate_positive_ratio"].item())

                    self._clear_multi_features()
                    preds_adv = self._forward_raw(adv_aug)
                    features_adv_cache = self.multi_features

                    cache_adv = self.hijacked.cache_assign_inputs_only(
                        preds=preds_adv,
                        batch=single_batch,
                        image_shape=(batch_h, batch_w),
                        assignment_topk=self.assignment_topk,
                    )
                    
                    if gate is not None and gate.numel() > 0:
                        shadow_adv = self.shadow_tal(
                            pred_scores_logits=cache_adv["pred_scores_logits"],
                            pred_bboxes=cache_adv["pred_bboxes"],
                            gt_labels=cache_adv["gt_labels"],
                            gt_bboxes=cache_adv["gt_bboxes"],
                            mask_gt=cache_adv["mask_gt"],
                            gate=gate,
                            topk=self.assignment_topk,
                        )
                        align_adv_topk_acc += float(shadow_adv["topk_alignment"].mean().item())

                    L_entangle_bg = torch.zeros((), device=self.device)
                    L_semantic_anchor = torch.zeros((), device=self.device)
                    L_collapse_aux = torch.zeros((), device=self.device)
                    valid_layers = 0

                    raw_assign = getattr(self.hijacked, "last_real_assign", {})
                    real_assign_clean = {}
                    if (
                        raw_assign
                        and torch.is_tensor(raw_assign.get("fg_mask", None))
                        and torch.is_tensor(raw_assign.get("target_labels", None))
                    ):
                        real_assign_clean = {
                            k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in raw_assign.items()
                        }

                    real_labels = None
                    cooccur_nt_fg_mask = None
                    if real_assign_clean:
                        real_fg = real_assign_clean["fg_mask"].bool()
                        real_labels = real_assign_clean["target_labels"].long()

                        nt_fg_mask = real_fg & (real_labels != self.target_class_id)
                        base_cooccur_mask = nt_fg_mask
                        if not self.cooccur_nt_only_on_real_fg:
                            base_cooccur_mask = (real_labels != self.target_class_id)
                        cooccur_img_mask_2d = cooccur_img_mask[:, None].expand_as(real_fg)
                        cooccur_nt_fg_mask = base_cooccur_mask & cooccur_img_mask_2d
                        if self.cooccur_nt_exclude_target_class:
                            cooccur_nt_fg_mask = cooccur_nt_fg_mask & (real_labels != self.target_class_id)
                        clean_all_logits = cache_clean["pred_scores_logits"]
                        clean_all_prob = torch.sigmoid(clean_all_logits)
                        assigned_labels = real_labels.clamp(min=0, max=self.num_classes - 1)
                        assigned_clean_prob = clean_all_prob.gather(
                            dim=2,
                            index=assigned_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        cooccur_nt_fg_mask = cooccur_nt_fg_mask & (
                            assigned_clean_prob >= self.cooccur_nt_min_clean_conf
                        )
                        strict_gate_1d = real_fg & (real_labels == self.target_class_id)

                        # 🚀 动态计算当前 Batch 下 FPN 各层的 1D 展平尺寸
                        layer_sizes = []
                        for layer_name in self.shape_layers:
                            if layer_name in features_clean_cache:
                                _, _, h, w = features_clean_cache[layer_name].shape
                                layer_sizes.append(h * w)

                        pag_gate_1d, pag_stats = build_pag_gate(
                            strict_gate_1d=strict_gate_1d,
                            target_scores=real_assign_clean.get("target_scores", None),
                            target_class_id=self.target_class_id,
                            top_ratio=self.pag_layer_ratios,
                            min_keep=self.pag_min_pos,
                            layer_sizes=layer_sizes,
                        )
                        if pag_stats:
                            pag_ratio_vals.append(float(pag_stats.get("pag_positive_ratio", 0.0)))
                            pag_threshold_vals.append(float(pag_stats.get("pag_threshold", 0.0)))
                            pag_mean_score_vals.append(float(pag_stats.get("pag_mean_target_score", 0.0)))
                            pag_fallback_vals.append(float(pag_stats.get("pag_fallback_count", 0.0))) 
                        
                        if not getattr(self, "_sanity_gate_printed", False):
                            print(f"\n🔬 [Sanity Check 1] fg positives: {real_fg.sum().item()}")
                            print(f"🔬 [Sanity Check 1] strict target positives: {strict_gate_1d.sum().item()}")
                            print(
                                "🔬 [Sanity Check 1] "
                                f"pag positives: {int(pag_stats.get('pag_positive', 0.0))} "
                                f"(ratio={pag_stats.get('pag_positive_ratio', 0.0):.2%}, "
                                f"threshold={pag_stats.get('pag_threshold', 0.0):.4f}, "
                                f"mean_target_score={pag_stats.get('pag_mean_target_score', 0.0):.4f})"
                            )
                            if "layer_stats" in pag_stats and pag_stats["layer_stats"]:
                                print("🔬 [PAG Layer Stats]")
                                for l_stat in pag_stats["layer_stats"]:
                                    l_idx = l_stat["layer_idx"]
                                    l_name = self.shape_layers[l_idx] if l_idx < len(self.shape_layers) else f"Layer {l_idx}"
                                    print(f"   {l_name}: strict={l_stat['strict']} -> pag={l_stat['pag']} (ratio={l_stat['ratio']:.2%})")
                            self._sanity_gate_printed = True
                    else:
                        strict_gate_1d = None
                        pag_gate_1d = None
                        nt_fg_mask = None

                    if strict_gate_1d is None or pag_gate_1d is None:
                        if not getattr(self, "_sanity_miss_printed", False):
                            print("\n⚠️ [Warning] strict assign info missing, ALSD branch skipped for this batch.")
                            self._sanity_miss_printed = True
                    else:
                        layer_pairs = []
                        for layer_name in self.shape_layers:
                            if layer_name in features_adv_cache and layer_name in features_clean_cache:
                                z_adv_l = features_adv_cache[layer_name]
                                z_clean_l = features_clean_cache[layer_name]
                                layer_pairs.append((layer_name, z_adv_l, z_clean_l))

                        if layer_pairs:
                            assign_maps = project_strict_gate_to_fpn(
                                strict_gate_1d=pag_gate_1d,
                                shape_layers=self.shape_layers,
                                features_cache=features_adv_cache,
                            )
                        else:
                            assign_maps = {}

                        for layer_name, Z_adv, Z_clean in layer_pairs:
                            B_feat, _, H, W = Z_adv.shape
                            assert strict_gate_1d.shape[0] == B_feat, (
                                f"Batch Mismatch: strict_gate={strict_gate_1d.shape[0]}, feature={B_feat}"
                            )
                            if layer_name not in assign_maps:
                                continue

                            M_topology = F.adaptive_avg_pool2d(inner_t, output_size=(H, W)).clamp(0.0, 1.0)
                            M_local_ctx = F.adaptive_avg_pool2d(local_ctx_t, output_size=(H, W)).clamp(0.0, 1.0)
                            M_conf = F.adaptive_avg_pool2d(conf_t, output_size=(H, W)).clamp(0.0, 1.0)
                            M_non_target_core_l = F.adaptive_avg_pool2d(M_non_target_core, output_size=(H, W)).clamp(0.0, 1.0)

                            M_assign_2d = assign_maps[layer_name]
                            M_AL = M_topology * M_assign_2d

                            overlap_ratio = compute_overlap_ratio(M_AL, M_assign_2d)
                            conf_purity = compute_confounder_purity(M_conf, M_local_ctx)
                            core_exclusion_ratio = float(
                                (M_local_ctx * M_non_target_core_l).sum().item() / (M_local_ctx.sum().item() + 1e-6)
                            )
                            layer_overlap_vals.append(overlap_ratio)
                            layer_conf_sum_vals.append(float(M_conf.sum().item()))
                            layer_conf_purity_vals.append(conf_purity)
                            layer_core_exclusion_vals.append(core_exclusion_ratio)

                            z_t_adv, valid_t = masked_prototype(Z_adv, M_AL, min_pixels=1.0)
                            mu_t_clean, valid_clean = masked_prototype(Z_clean, M_AL, min_pixels=1.0)
                            min_conf_l = max(1.0, self.min_conf_pixels * (H * W) / float(batch_h * batch_w))
                            c_conf, valid_conf, trim_keep_ratio = robust_masked_prototype(
                                Z_clean,
                                M_conf,
                                trim_ratio=self.rlcp_trim_ratio,
                                min_pixels=min_conf_l,
                            )
                            valid_joint = valid_t & valid_clean & valid_conf
                            if torch.count_nonzero(valid_joint) == 0:
                                continue

                            valid_layers += 1
                            layer_trim_keep_vals.append(float(trim_keep_ratio[valid_joint].mean().item()))

                            if (
                                epoch == 0
                                and batch_count == 0
                                and eot_idx == 0
                                and not getattr(self, f"_sanity_layer_{layer_name}_printed", False)
                            ):
                                print(f"🔬 [Sanity Check 2&3 - {layer_name}]")
                                print(f"   M_topology sum: {M_topology.sum().item():.1f}")
                                print(f"   M_assign_2d sum: {M_assign_2d.sum().item():.1f}")
                                print(f"   M_AL sum: {M_AL.sum().item():.1f}")
                                print(f"   rlcp_conf_mass: {M_conf.sum().item():.1f}")
                                print(f"   rlcp_trim_keep_ratio: {float(trim_keep_ratio[valid_joint].mean().item()):.2%}")
                                print(f"   rlcp_core_exclusion_ratio: {core_exclusion_ratio:.2%}")
                                print(f"   confounder_purity_ratio: {conf_purity:.2%}")
                                print(f"   overlap ratio: {overlap_ratio:.2%}")
                                setattr(self, f"_sanity_layer_{layer_name}_printed", True)

                            layer_ent, ent_stats = compute_entangle_loss(
                                z_adv=z_t_adv,
                                z_clean=mu_t_clean,
                                z_conf=c_conf,
                                tau=self.entangle_tau,
                                valid_mask=valid_joint,
                            )
                            L_cos, L_energy = compute_anchor_losses(z_t_adv[valid_joint], mu_t_clean[valid_joint])
                            layer_anchor = L_cos + L_energy
                            layer_flat, spatial_var = compute_collapse_loss(Z_adv, M_AL)

                            L_entangle_bg += layer_ent
                            L_semantic_anchor += layer_anchor
                            L_collapse_aux += layer_flat

                            layer_cos_conf_vals.append(ent_stats["cos_t_conf"])
                            layer_cos_clean_vals.append(ent_stats["cos_t_clean"])
                            batch_alsi_score_acc += compute_alsi_score(z_t_adv, spatial_var, valid_mask=valid_joint)

                        if valid_layers > 0:
                            L_entangle_bg = L_entangle_bg / valid_layers
                            L_semantic_anchor = L_semantic_anchor / valid_layers
                            L_collapse_aux = L_collapse_aux / valid_layers
                            batch_alsi_score_acc = batch_alsi_score_acc / valid_layers

                    # ====================================================
                    #  Track A: Non-Target Preserve 
                    # ====================================================
                    M_supp_spatial = inner_t 
                    M_non_supp_spatial = 1.0 - M_supp_spatial
                    L_preserve = torch.zeros((), device=self.device)
                    L_preserve_feat = torch.zeros((), device=self.device)
                    L_preserve_logits = torch.zeros((), device=self.device)
                    L_margin = torch.zeros((), device=self.device)
                    L_cooccur_nt_logits = torch.zeros((), device=self.device)
                    cooccur_nt_fg_count = 0
                    cooccur_nt_clean_prob_mean = 0.0
                    cooccur_nt_adv_prob_mean = 0.0
                    cooccur_nt_prob_drop = 0.0
                    cooccur_nt_prob_drop_raw_mean = 0.0
                    cooccur_nt_prob_drop_pos_mean = 0.0
                    cooccur_nt_prob_drop_hinge_mean = 0.0
                    cooccur_nt_hard_loss_mean = 0.0
                    cooccur_nt_hard_count = 0
                    
                    if cur_lambda_preserve > 0:
                        for layer_name in self.preserve_layers:
                            if layer_name in features_clean_cache and layer_name in features_adv_cache:
                                z_c = features_clean_cache[layer_name]
                                z_a = features_adv_cache[layer_name]
                                M_bg = F.adaptive_avg_pool2d(M_non_supp_spatial, output_size=z_a.shape[-2:])
                                mse_map = F.mse_loss(z_c, z_a, reduction='none').mean(dim=1, keepdim=True)
                                L_preserve_feat = L_preserve_feat + (mse_map * M_bg).sum() / (M_bg.sum() + 1e-6)
                        
                        non_target_indices = torch.arange(self.num_classes, device=self.device) != self.target_class_id
                        clean_non_target_logits = cache_clean["pred_scores_logits"][:, :, non_target_indices]
                        adv_non_target_logits = cache_adv["pred_scores_logits"][:, :, non_target_indices]
                        
                        if nt_fg_mask is not None and nt_fg_mask.any():
                            clean_nt_fg_logits = clean_non_target_logits[nt_fg_mask]
                            adv_nt_fg_logits = adv_non_target_logits[nt_fg_mask]
                            L_preserve_logits = F.mse_loss(
                                torch.sigmoid(adv_nt_fg_logits),
                                torch.sigmoid(clean_nt_fg_logits.detach())
                            )
                        else:
                            L_preserve_logits = torch.zeros((), device=self.device)

                        L_margin, margin_stats = compute_non_target_margin_preserve(
                            clean_non_target_logits=clean_non_target_logits,
                            adv_non_target_logits=adv_non_target_logits,
                            use_smooth_l1=True,
                            valid_mask=nt_fg_mask 
                        )
                        margin_loss_vals.append(float(L_margin.item()))
                        margin_clean_vals.append(float(margin_stats["margin_clean_mean"]))
                        margin_adv_vals.append(float(margin_stats["margin_adv_mean"]))

                        L_preserve = (
                            self.lambda_preserve_feat * preserve_feat_scale_cur * L_preserve_feat
                            + self.lambda_preserve_logits * preserve_logits_scale_cur * L_preserve_logits
                            + self.lambda_margin * margin_scale_cur * L_margin
                        )

                        if (
                            epoch == 0
                            and batch_count == 0
                            and eot_idx == 0
                            and not getattr(self, "_sanity_dsnp_printed", False)
                        ):
                            print("\n🔬 [DSNP-lite Diagnostics]")
                            print(f"   L_margin: {float(L_margin.item()):.6f}")
                            print(f"   margin_clean_mean: {float(margin_stats['margin_clean_mean']):.6f}")
                            print(f"   margin_adv_mean: {float(margin_stats['margin_adv_mean']):.6f}")
                            self._sanity_dsnp_printed = True

                    if (
                        self.enable_cooccur_nt_logits_preserve
                        and cooccur_nt_fg_mask is not None
                        and cooccur_nt_fg_mask.any()
                        and real_labels is not None
                    ):
                        clean_logits_all = cache_clean["pred_scores_logits"]
                        adv_logits_all = cache_adv["pred_scores_logits"]
                        assigned_labels = real_labels.clamp(min=0, max=self.num_classes - 1)
                        if not self.cooccur_nt_assigned_class_only:
                            # Keep v1 fixed to assigned-class preserve only.
                            pass
                        clean_assigned_logits = clean_logits_all.gather(
                            dim=2,
                            index=assigned_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        adv_assigned_logits = adv_logits_all.gather(
                            dim=2,
                            index=assigned_labels.unsqueeze(-1),
                        ).squeeze(-1)

                        clean_prob = torch.sigmoid(clean_assigned_logits[cooccur_nt_fg_mask]).detach()
                        adv_prob = torch.sigmoid(adv_assigned_logits[cooccur_nt_fg_mask])
                        prob_drop = clean_prob - adv_prob

                        if self.cooccur_nt_loss_type == "smooth_l1_prob":
                            L_cooccur_nt_logits = F.smooth_l1_loss(
                                adv_prob,
                                clean_prob,
                                reduction="mean",
                            )
                            hard_vals = torch.zeros_like(prob_drop)
                            cooccur_nt_hard_count = 0
                            cooccur_nt_hard_loss_mean = 0.0
                        elif self.cooccur_nt_loss_type == "prob_drop_hinge":
                            if self.cooccur_nt_use_positive_drop_only:
                                drop_penalty = F.relu(prob_drop - self.cooccur_nt_drop_tolerance)
                            else:
                                drop_penalty = torch.abs(prob_drop)

                            if drop_penalty.numel() > 0:
                                if self.cooccur_nt_hard_top_ratio < 1.0:
                                    k = int(math.ceil(drop_penalty.numel() * self.cooccur_nt_hard_top_ratio))
                                    k = max(self.cooccur_nt_min_hard_count, k)
                                    k = min(k, drop_penalty.numel())
                                    hard_vals = torch.topk(drop_penalty, k=k, largest=True).values
                                    L_cooccur_nt_logits = hard_vals.mean()
                                else:
                                    hard_vals = drop_penalty
                                    L_cooccur_nt_logits = drop_penalty.mean()
                            else:
                                hard_vals = drop_penalty
                                L_cooccur_nt_logits = torch.zeros((), device=self.device)
                            if hard_vals.numel() > 0:
                                cooccur_nt_hard_count = int(hard_vals.numel())
                                cooccur_nt_hard_loss_mean = float(hard_vals.detach().mean().item())
                            else:
                                cooccur_nt_hard_count = 0
                                cooccur_nt_hard_loss_mean = 0.0
                        else:
                            raise ValueError(f"Unsupported cooccur_nt_loss_type: {self.cooccur_nt_loss_type}")

                        cooccur_nt_fg_count = int(cooccur_nt_fg_mask.sum().item())
                        cooccur_nt_clean_prob_mean = float(clean_prob.mean().item())
                        cooccur_nt_adv_prob_mean = float(adv_prob.mean().item())
                        cooccur_nt_prob_drop = cooccur_nt_clean_prob_mean - cooccur_nt_adv_prob_mean
                        prob_drop_det = prob_drop.detach()
                        cooccur_nt_prob_drop_raw_mean = float(prob_drop_det.mean().item())
                        cooccur_nt_prob_drop_pos_mean = float(F.relu(prob_drop_det).mean().item())
                        cooccur_nt_prob_drop_hinge_mean = float(
                            F.relu(prob_drop_det - self.cooccur_nt_drop_tolerance).mean().item()
                        )

                    # 累加 EOT 的 ALCE losses
                    L_ent_total += L_entangle_bg
                    L_anchor_total += L_semantic_anchor
                    L_flat_total += L_collapse_aux
                    L_preserve_total += L_preserve
                    L_preserve_logits_total += L_preserve_logits
                    L_preserve_feat_total += L_preserve_feat
                    L_margin_total += L_margin
                    L_cooccur_nt_total += L_cooccur_nt_logits
                    cooccur_nt_fg_count_vals.append(float(cooccur_nt_fg_count))
                    cooccur_nt_clean_prob_vals.append(cooccur_nt_clean_prob_mean)
                    cooccur_nt_adv_prob_vals.append(cooccur_nt_adv_prob_mean)
                    cooccur_nt_prob_drop_vals.append(cooccur_nt_prob_drop)
                    cooccur_nt_img_ratio_vals.append(float(cooccur_img_mask.float().mean().item()))
                    cooccur_nt_prob_drop_raw_vals.append(cooccur_nt_prob_drop_raw_mean)
                    cooccur_nt_prob_drop_pos_vals.append(cooccur_nt_prob_drop_pos_mean)
                    cooccur_nt_prob_drop_hinge_vals.append(cooccur_nt_prob_drop_hinge_mean)
                    cooccur_nt_hard_loss_vals.append(cooccur_nt_hard_loss_mean)
                    cooccur_nt_hard_count_vals.append(float(cooccur_nt_hard_count))

                # 计算 EOT 平均
                L_ent_final = L_ent_total / eot_count
                L_anchor_final = L_anchor_total / eot_count
                L_flat_final = L_flat_total / eot_count
                L_preserve_final = L_preserve_total / eot_count
                L_preserve_logits_final = L_preserve_logits_total / eot_count
                L_preserve_feat_final = L_preserve_feat_total / eot_count
                L_margin_final_for_score = L_margin_total / eot_count
                L_cooccur_nt_final = L_cooccur_nt_total / eot_count

                L_tv = self._tv_loss(raw_perturb)
                L_budget = F.relu(torch.max(torch.abs(raw_perturb)) - self.eps)

                total_loss = (
                    effective_lambda_ent * L_ent_final
                    + effective_lambda_anchor * L_anchor_final
                    + effective_lambda_flat * L_flat_final
                    + cur_lambda_preserve * L_preserve_final
                    + effective_cooccur_lambda * L_cooccur_nt_final
                    + self.lambda_tv * L_tv
                    + self.lambda_budget * L_budget
                )

                attack_scale_ent = self.lambda_ent if self.freq_score_attack_use_weighted_loss else 1.0
                attack_scale_anchor = self.lambda_anchor if self.freq_score_attack_use_weighted_loss else 1.0
                attack_scale_flat = self.lambda_flat if self.freq_score_attack_use_weighted_loss else 1.0
                L_attack_score = (
                    attack_scale_ent * L_ent_final
                    + attack_scale_anchor * L_anchor_final
                    + attack_scale_flat * L_flat_final
                )

                collateral_terms = []
                if self.freq_score_use_logits_collateral:
                    c_scale = self.lambda_preserve_logits if self.freq_score_collateral_use_weighted_loss else 1.0
                    collateral_terms.append(c_scale * L_preserve_logits_final)
                if self.freq_score_use_feat_collateral:
                    c_scale = self.lambda_preserve_feat if self.freq_score_collateral_use_weighted_loss else 1.0
                    collateral_terms.append(c_scale * L_preserve_feat_final)
                if self.freq_score_use_margin_collateral:
                    c_scale = self.lambda_margin if self.freq_score_collateral_use_weighted_loss else 1.0
                    collateral_terms.append(c_scale * L_margin_final_for_score)
                if collateral_terms:
                    L_collateral_score = collateral_terms[0]
                    for t in collateral_terms[1:]:
                        L_collateral_score = L_collateral_score + t
                else:
                    L_collateral_score = torch.zeros((), device=self.device, dtype=total_loss.dtype)

                metric_for_update = self.adaptive_freq_select_metric
                if (
                    self.enable_adaptive_freq_basis
                    and epoch < self.adaptive_freq_warmup_epochs
                    and metric_for_update not in {"grad_norm", "attack_collateral_ratio"}
                ):
                    if not self._adaptive_freq_metric_warned:
                        print(
                            f"[AdaptiveFreq][Warning] unsupported adaptive_freq_select_metric="
                            f"{self.adaptive_freq_select_metric}, fallback to grad_norm."
                        )
                        self._adaptive_freq_metric_warned = True
                    metric_for_update = "grad_norm"

                if (
                    self.enable_adaptive_freq_basis
                    and epoch < self.adaptive_freq_warmup_epochs
                    and metric_for_update == "attack_collateral_ratio"
                ):
                    self._update_adaptive_freq_score(
                        active_idx=active_idx,
                        L_attack_score=L_attack_score,
                        L_collateral_score=L_collateral_score,
                    )

                total_loss.backward()

                if self.enable_adaptive_freq_basis and epoch < self.adaptive_freq_warmup_epochs:
                    with torch.no_grad():
                        if metric_for_update == "grad_norm":
                            grad = self.fourier_coeff.grad
                            if grad is None:
                                if not self._adaptive_freq_grad_warned:
                                    print("[AdaptiveFreq][Warning] fourier_coeff.grad is None during warmup; skip score update.")
                                    self._adaptive_freq_grad_warned = True
                            else:
                                reduce_dims = tuple(range(1, grad.ndim))
                                grad_score = grad.detach().abs().mean(dim=reduce_dims)
                                self.freq_score[active_idx] += grad_score[active_idx]
                                self.freq_usage[active_idx] += 1.0

                if self.fourier_coeff.grad is not None:
                    grad_norm_fourier = float(torch.norm(self.fourier_coeff.grad).item())
                else:
                    grad_norm_fourier = -1.0  

                if self.suppress_small.grad is not None:
                    grad_norm_suppress = float(torch.norm(self.suppress_small.grad).item())
                else:
                    grad_norm_suppress = -1.0

                optimizer.step()

                if batch_count == 0 and epoch % 5 == 0:
                    vis_dir = os.path.join(os.path.dirname(diagnostics_csv_path), "vis_debug")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    cl_img = img_t[0].permute(1, 2, 0).detach().cpu().numpy()
                    adv_img = adv[0].permute(1, 2, 0).detach().cpu().numpy()
                    noise_img = perturb[0].permute(1, 2, 0).detach().cpu().numpy()
                    mask_img = inner_t[0, 0].detach().cpu().numpy()
                    
                    noise_vis = np.clip((noise_img * 10.0) + 0.5, 0.0, 1.0)
                    
                    cl_bgr = cv2.cvtColor((cl_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    adv_bgr = cv2.cvtColor((adv_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    noise_bgr = cv2.cvtColor((noise_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    mask_bgr = cv2.cvtColor((mask_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    
                    panel = np.concatenate([cl_bgr, adv_bgr, noise_bgr, mask_bgr], axis=1)
                    
                    save_path = os.path.join(vis_dir, f"debug_epoch_{epoch}_step_{global_step}.png")
                    cv2.imwrite(save_path, panel)
                    print(f"  -> [Visualizer] 调试图像已保存至: {save_path}")

                epoch_diag["align_clean_topk"] += align_clean_topk_acc / eot_count
                epoch_diag["align_adv_topk"] += align_adv_topk_acc / eot_count
                epoch_diag["L_entangle_bg"] += float(L_ent_final.item())
                epoch_diag["L_anchor"] += float(L_anchor_final.item())
                epoch_diag["L_collapse_aux"] += float(L_flat_final.item())
                epoch_diag["alsi_score"] += batch_alsi_score_acc / max(1, eot_count)
                epoch_diag["cos_t_conf"] += safe_mean(layer_cos_conf_vals)
                epoch_diag["cos_t_clean"] += safe_mean(layer_cos_clean_vals)
                epoch_diag["M_conf_sum"] += safe_mean(layer_conf_sum_vals)
                
                epoch_diag["rlcp_trim_keep_ratio"] += safe_mean(layer_trim_keep_vals)
                epoch_diag["rlcp_core_exclusion_ratio"] += safe_mean(layer_core_exclusion_vals)
                epoch_diag["confounder_purity_ratio"] += safe_mean(layer_conf_purity_vals)
                epoch_diag["overlap_ratio"] += safe_mean(layer_overlap_vals)
                epoch_diag["preserve_loss"] += float(L_preserve_final.item())
                epoch_diag["L_margin"] += safe_mean(margin_loss_vals)
                epoch_diag["margin_clean_mean"] += safe_mean(margin_clean_vals)
                epoch_diag["margin_adv_mean"] += safe_mean(margin_adv_vals)
                epoch_diag["gate_positive_ratio"] += gate_ratio_acc / eot_count
                epoch_diag["pag_positive_ratio"] += safe_mean(pag_ratio_vals)
                epoch_diag["pag_threshold"] += safe_mean(pag_threshold_vals)
                epoch_diag["pag_mean_target_score"] += safe_mean(pag_mean_score_vals)
                epoch_diag["pag_fallback_count"] += safe_mean(pag_fallback_vals) 
                epoch_diag["L_tv"] += float(L_tv.item())
                epoch_diag["L_budget"] += float(L_budget.item())
                epoch_diag["L_total"] += float(total_loss.item())
                epoch_diag["L_tsvc"] += float(L_flat_final.item())
                epoch_diag["L_sem"] += float(L_anchor_final.item())
                epoch_diag["sld_score"] += batch_alsi_score_acc / max(1, eot_count)
                epoch_diag["L_cooccur_nt_logits"] += float(L_cooccur_nt_final.item())
                epoch_diag["cooccur_nt_lambda_cur"] += float(cooccur_lambda_cur)
                epoch_diag["cooccur_nt_fg_count"] += safe_mean(cooccur_nt_fg_count_vals)
                epoch_diag["cooccur_nt_img_ratio"] += safe_mean(cooccur_nt_img_ratio_vals)
                epoch_diag["cooccur_nt_clean_prob_mean"] += safe_mean(cooccur_nt_clean_prob_vals)
                epoch_diag["cooccur_nt_adv_prob_mean"] += safe_mean(cooccur_nt_adv_prob_vals)
                epoch_diag["cooccur_nt_prob_drop"] += safe_mean(cooccur_nt_prob_drop_vals)
                epoch_diag["cooccur_nt_prob_drop_raw_mean"] += safe_mean(cooccur_nt_prob_drop_raw_vals)
                epoch_diag["cooccur_nt_prob_drop_pos_mean"] += safe_mean(cooccur_nt_prob_drop_pos_vals)
                epoch_diag["cooccur_nt_prob_drop_hinge_mean"] += safe_mean(cooccur_nt_prob_drop_hinge_vals)
                epoch_diag["cooccur_nt_hard_loss_mean"] += safe_mean(cooccur_nt_hard_loss_vals)
                epoch_diag["cooccur_nt_hard_count"] += safe_mean(cooccur_nt_hard_count_vals)
                epoch_diag["late_repair_ratio"] += float(late_repair_ratio)
                epoch_diag["ent_scale_cur"] += float(ent_scale_cur)
                epoch_diag["anchor_scale_cur"] += float(anchor_scale_cur)
                epoch_diag["flat_scale_cur"] += float(flat_scale_cur)
                epoch_diag["preserve_logits_scale_cur"] += float(preserve_logits_scale_cur)
                epoch_diag["preserve_feat_scale_cur"] += float(preserve_feat_scale_cur)
                epoch_diag["margin_scale_cur"] += float(margin_scale_cur)
                epoch_diag["cooccur_lambda_scale_cur"] += float(cooccur_lambda_scale_cur)
                epoch_diag["effective_lambda_ent"] += float(effective_lambda_ent)
                epoch_diag["effective_lambda_anchor"] += float(effective_lambda_anchor)
                epoch_diag["effective_lambda_flat"] += float(effective_lambda_flat)
                epoch_diag["effective_cooccur_lambda"] += float(effective_cooccur_lambda)
                epoch_diag["cooccur_nt_loss_type"] = str(self.cooccur_nt_loss_type)

                batch_max_d = float(torch.max(torch.abs(perturb)).item())
                batch_mean_d = float(torch.mean(torch.abs(perturb)).item())
                
                batch_sat_ratio = float(torch.mean((torch.abs(raw_perturb) > self.eps - 1e-5).float()).item())
                batch_area_mask = (torch.max(torch.abs(perturb), dim=1)[0] > (1.0 / 255.0)).float()
                batch_area_ratio = float(torch.mean(batch_area_mask).item())
                
                epoch_diag["max_abs_delta"] = max(epoch_diag["max_abs_delta"], batch_max_d)
                epoch_diag["mean_abs_delta"] += batch_mean_d
                epoch_diag["saturation_ratio"] += batch_sat_ratio
                epoch_diag["perturbed_area_ratio_mean"] += batch_area_ratio
                
                batch_count += 1
                global_step += 1

            if batch_count > 0:
                for k in list(epoch_diag.keys()):
                    if k != "max_abs_delta" and isinstance(epoch_diag[k], (int, float)):
                        epoch_diag[k] = epoch_diag[k] / float(batch_count)

            freq_score_top_mean = float("nan")
            band_score_stats: Dict[str, Dict[str, float]] = {}
            band_attack_stats: Dict[str, Dict[str, float]] = {}
            band_collateral_stats: Dict[str, Dict[str, float]] = {}
            band_ratio_stats: Dict[str, Dict[str, float]] = {}
            freq_attack_grad_mean = float("nan")
            freq_collateral_grad_mean = float("nan")
            freq_ratio_mean = float("nan")
            freq_attack_grad_top_mean = float("nan")
            freq_collateral_grad_top_mean = float("nan")
            freq_ratio_top_mean = float("nan")
            if self.enable_adaptive_freq_basis:
                with torch.no_grad():
                    score_now = self.freq_score / self.freq_usage.clamp_min(1.0)
                    if self.enable_band_aware_freq_basis:
                        top_idx_now = self._select_topk_per_band(score_now)
                        band_score_stats = self._compute_band_score_stats(score_now)
                    else:
                        top_idx_now = torch.topk(score_now, k=self.freq_active_num_bases, largest=True).indices
                    freq_score_top_mean = float(score_now[top_idx_now].mean().item())
                    if self.adaptive_freq_select_metric == "attack_collateral_ratio":
                        attack_now = self.freq_attack_grad_score / self.freq_usage.clamp_min(1.0)
                        collateral_now = self.freq_collateral_grad_score / self.freq_usage.clamp_min(1.0)
                        ratio_now = self.freq_ratio_score / self.freq_usage.clamp_min(1.0)
                        freq_attack_grad_mean = float(attack_now.mean().item())
                        freq_collateral_grad_mean = float(collateral_now.mean().item())
                        freq_ratio_mean = float(ratio_now.mean().item())
                        freq_attack_grad_top_mean = self._topk_mean(attack_now, top_idx_now)
                        freq_collateral_grad_top_mean = self._topk_mean(collateral_now, top_idx_now)
                        freq_ratio_top_mean = self._topk_mean(ratio_now, top_idx_now)
                        if self.enable_band_aware_freq_basis:
                            band_attack_stats = self._compute_band_score_stats(attack_now)
                            band_collateral_stats = self._compute_band_score_stats(collateral_now)
                            band_ratio_stats = self._compute_band_score_stats(ratio_now)

                if self.adaptive_freq_update_once:
                    need_select_topk = (epoch + 1 == self.adaptive_freq_warmup_epochs)
                else:
                    need_select_topk = (epoch + 1 >= self.adaptive_freq_warmup_epochs)

                if need_select_topk:
                    with torch.no_grad():
                        score = self.freq_score / self.freq_usage.clamp_min(1.0)
                        if self.enable_band_aware_freq_basis:
                            top_idx = self._select_topk_per_band(score)
                        else:
                            top_idx = torch.topk(score, k=self.freq_active_num_bases, largest=True).indices
                        self.freq_active_idx = top_idx.sort().values.detach()

                        mask = torch.zeros(self.freq_candidate_num_bases, device=self.device, dtype=torch.bool)
                        mask[self.freq_active_idx] = True
                        self.fourier_coeff.data[~mask] = 0.0

                        self.coords = [
                            tuple(self.freq_candidate_coords[int(i)])
                            for i in self.freq_active_idx.detach().cpu().tolist()
                        ]
                        self._adaptive_freq_score_top_mean = float(score[self.freq_active_idx].mean().item())
                        freq_score_top_mean = self._adaptive_freq_score_top_mean

                        if self.enable_band_aware_freq_basis:
                            selected_by_band = self._split_indices_by_band(self.freq_active_idx)
                            selected_coords_by_band = {
                                band_name: [
                                    [int(self.freq_candidate_coords[i][0]), int(self.freq_candidate_coords[i][1])]
                                    for i in selected_by_band.get(band_name, [])
                                ]
                                for band_name in self.freq_band_names
                            }
                            band_score_stats = self._compute_band_score_stats(score)
                            selected_attack_top_mean = float("nan")
                            selected_collateral_top_mean = float("nan")
                            selected_ratio_top_mean = float("nan")
                            selected_band_attack_stats: Dict[str, Dict[str, float]] = {}
                            selected_band_collateral_stats: Dict[str, Dict[str, float]] = {}
                            selected_band_ratio_stats: Dict[str, Dict[str, float]] = {}
                            if self.adaptive_freq_select_metric == "attack_collateral_ratio":
                                attack_now = self.freq_attack_grad_score / self.freq_usage.clamp_min(1.0)
                                collateral_now = self.freq_collateral_grad_score / self.freq_usage.clamp_min(1.0)
                                ratio_now = self.freq_ratio_score / self.freq_usage.clamp_min(1.0)
                                selected_attack_top_mean = self._topk_mean(attack_now, self.freq_active_idx)
                                selected_collateral_top_mean = self._topk_mean(collateral_now, self.freq_active_idx)
                                selected_ratio_top_mean = self._topk_mean(ratio_now, self.freq_active_idx)
                                selected_band_attack_stats = self._compute_band_score_stats(attack_now)
                                selected_band_collateral_stats = self._compute_band_score_stats(collateral_now)
                                selected_band_ratio_stats = self._compute_band_score_stats(ratio_now)
                            quota_parts = [
                                f"{band_name}={int(k)}" for band_name, k in zip(self.freq_band_names, self.freq_band_active_nums)
                            ]
                            print("[BandAwareFreq] selected active basis:")
                            print(f"  candidate_num={self.freq_candidate_num_bases}")
                            print(f"  active_num={self.freq_active_num_bases}")
                            print(f"  active_quota: {' '.join(quota_parts)}")
                            for band_name in self.freq_band_names:
                                print(f"  selected_{band_name}_idx={selected_by_band.get(band_name, [])}")
                            for band_name in self.freq_band_names:
                                print(f"  selected_{band_name}_coords={selected_coords_by_band.get(band_name, [])}")
                            for band_name in self.freq_band_names:
                                bstats = band_score_stats.get(band_name, {})
                                print(f"  score_mean_{band_name}={float(bstats.get('mean', float('nan'))):.6e}")
                            for band_name in self.freq_band_names:
                                bstats = band_score_stats.get(band_name, {})
                                print(f"  score_top_mean_{band_name}={float(bstats.get('top_mean', float('nan'))):.6e}")
                            if self.adaptive_freq_select_metric == "attack_collateral_ratio":
                                print("[AdaptiveFreq] metric=attack_collateral_ratio")
                                print(f"  attack_top_mean={selected_attack_top_mean:.6e}")
                                print(f"  collateral_top_mean={selected_collateral_top_mean:.6e}")
                                print(f"  ratio_top_mean={selected_ratio_top_mean:.6e}")
                                for band_name in self.freq_band_names:
                                    b_attack = selected_band_attack_stats.get(band_name, {})
                                    b_coll = selected_band_collateral_stats.get(band_name, {})
                                    b_ratio = selected_band_ratio_stats.get(band_name, {})
                                    print(
                                        f"  band={band_name} attack_top_mean={float(b_attack.get('top_mean', float('nan'))):.6e} "
                                        f"collateral_top_mean={float(b_coll.get('top_mean', float('nan'))):.6e} "
                                        f"ratio_top_mean={float(b_ratio.get('top_mean', float('nan'))):.6e}"
                                    )
                        else:
                            print("[AdaptiveFreq] selected active basis:")
                            print(f"  candidate_num={self.freq_candidate_num_bases}")
                            print(f"  active_num={self.freq_active_num_bases}")
                            print(f"  active_idx={self.freq_active_idx.detach().cpu().tolist()}")
                            print(f"  score_mean={score.mean().item():.6e}")
                            print(f"  score_top_mean={score[self.freq_active_idx].mean().item():.6e}")
                            print(f"  score_max={score.max().item():.6e}")
                            print(f"  score_min={score.min().item():.6e}")
                            if self.adaptive_freq_select_metric == "attack_collateral_ratio":
                                attack_now = self.freq_attack_grad_score / self.freq_usage.clamp_min(1.0)
                                collateral_now = self.freq_collateral_grad_score / self.freq_usage.clamp_min(1.0)
                                ratio_now = self.freq_ratio_score / self.freq_usage.clamp_min(1.0)
                                print("[AdaptiveFreq] metric=attack_collateral_ratio")
                                print(f"  attack_top_mean={self._topk_mean(attack_now, self.freq_active_idx):.6e}")
                                print(f"  collateral_top_mean={self._topk_mean(collateral_now, self.freq_active_idx):.6e}")
                                print(f"  ratio_top_mean={self._topk_mean(ratio_now, self.freq_active_idx):.6e}")

            if self.enable_adaptive_freq_basis:
                freq_mode = "explore" if epoch < self.adaptive_freq_warmup_epochs else "fixed_topk"
                fact = self.freq_active_num_bases
                fcand = self.freq_candidate_num_bases
            else:
                freq_mode = "fixed"
                fact = self.shortcut_num_bases
                fcand = self.shortcut_num_bases

            band_name_triplet = ["low", "mid", "high"]
            if self.enable_adaptive_freq_basis and self.enable_band_aware_freq_basis:
                band_candidate_map = {
                    str(name): int(k) for name, k in zip(self.freq_band_names, self.freq_band_candidate_nums)
                }
                band_active_map = {
                    str(name): int(k) for name, k in zip(self.freq_band_names, self.freq_band_active_nums)
                }
            else:
                band_candidate_map = {}
                band_active_map = {}
            selected_idx_by_band: Dict[str, List[int]] = {name: [] for name in band_name_triplet}
            selected_coords_by_band: Dict[str, List[List[int]]] = {name: [] for name in band_name_triplet}
            band_score_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            band_score_top_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            band_radius_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            band_attack_top_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            band_collateral_top_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            band_ratio_top_mean_map: Dict[str, float] = {name: float("nan") for name in band_name_triplet}
            if self.enable_adaptive_freq_basis and self.enable_band_aware_freq_basis:
                selected_raw = self._split_indices_by_band(self.freq_active_idx)
                for name in band_name_triplet:
                    idx_list = [int(i) for i in selected_raw.get(name, [])]
                    selected_idx_by_band[name] = idx_list
                    selected_coords_by_band[name] = [
                        [int(self.freq_candidate_coords[i][0]), int(self.freq_candidate_coords[i][1])]
                        for i in idx_list
                    ]
                    stats = band_score_stats.get(name, {})
                    band_score_mean_map[name] = float(stats.get("mean", float("nan")))
                    band_score_top_mean_map[name] = float(stats.get("top_mean", float("nan")))
                    band_radius_mean_map[name] = float(self._freq_band_radius_mean.get(name, float("nan")))
                    attack_stats = band_attack_stats.get(name, {})
                    collateral_stats = band_collateral_stats.get(name, {})
                    ratio_stats = band_ratio_stats.get(name, {})
                    band_attack_top_mean_map[name] = float(attack_stats.get("top_mean", float("nan")))
                    band_collateral_top_mean_map[name] = float(collateral_stats.get("top_mean", float("nan")))
                    band_ratio_top_mean_map[name] = float(ratio_stats.get("top_mean", float("nan")))

            row = {
                "epoch": epoch + 1,
                **epoch_diag,
                "grad_norm_fourier": grad_norm_fourier if "grad_norm_fourier" in locals() else float("nan"),
                "grad_norm_suppress": grad_norm_suppress if "grad_norm_suppress" in locals() else float("nan"),
                "adaptive_freq_select_metric": self.adaptive_freq_select_metric,
                "freq_attack_grad_mean": freq_attack_grad_mean,
                "freq_collateral_grad_mean": freq_collateral_grad_mean,
                "freq_ratio_mean": freq_ratio_mean,
                "freq_attack_grad_top_mean": freq_attack_grad_top_mean,
                "freq_collateral_grad_top_mean": freq_collateral_grad_top_mean,
                "freq_ratio_top_mean": freq_ratio_top_mean,
                "freq_band_low_attack_top_mean": band_attack_top_mean_map["low"],
                "freq_band_mid_attack_top_mean": band_attack_top_mean_map["mid"],
                "freq_band_high_attack_top_mean": band_attack_top_mean_map["high"],
                "freq_band_low_collateral_top_mean": band_collateral_top_mean_map["low"],
                "freq_band_mid_collateral_top_mean": band_collateral_top_mean_map["mid"],
                "freq_band_high_collateral_top_mean": band_collateral_top_mean_map["high"],
                "freq_band_low_ratio_top_mean": band_ratio_top_mean_map["low"],
                "freq_band_mid_ratio_top_mean": band_ratio_top_mean_map["mid"],
                "freq_band_high_ratio_top_mean": band_ratio_top_mean_map["high"],
                "enable_cooccur_nt_logits_preserve": bool(self.enable_cooccur_nt_logits_preserve),
                "cooccur_nt_loss_type": str(self.cooccur_nt_loss_type),
                "cooccur_nt_logits_lambda": float(self.cooccur_nt_logits_lambda),
                "cooccur_nt_min_clean_conf": float(self.cooccur_nt_min_clean_conf),
                "cooccur_nt_drop_tolerance": float(self.cooccur_nt_drop_tolerance),
                "cooccur_nt_hard_top_ratio": float(self.cooccur_nt_hard_top_ratio),
                "cooccur_nt_min_hard_count": int(self.cooccur_nt_min_hard_count),
                "cooccur_nt_use_positive_drop_only": bool(self.cooccur_nt_use_positive_drop_only),
                "cooccur_nt_warmup_start_epoch": int(self.cooccur_nt_warmup_start_epoch),
                "cooccur_nt_warmup_end_epoch": int(self.cooccur_nt_warmup_end_epoch),
                "cooccur_nt_apply_after_freq_selection": bool(self.cooccur_nt_apply_after_freq_selection),
                "cooccur_nt_do_not_use_for_freq_score": bool(self.cooccur_nt_do_not_use_for_freq_score),
                "enable_late_nt_repair": bool(self.enable_late_nt_repair),
                "late_repair_start_epoch": int(self.late_repair_start_epoch),
                "late_repair_ramp_epochs": int(self.late_repair_ramp_epochs),
                "late_attack_loss_scale_enabled": bool(self.late_attack_loss_scale_enabled),
                "late_lambda_ent_scale": float(self.late_lambda_ent_scale),
                "late_lambda_anchor_scale": float(self.late_lambda_anchor_scale),
                "late_lambda_flat_scale": float(self.late_lambda_flat_scale),
                "late_preserve_logits_scale": float(self.late_preserve_logits_scale),
                "late_preserve_feat_scale": float(self.late_preserve_feat_scale),
                "late_margin_scale": float(self.late_margin_scale),
                "late_cooccur_lambda_scale": float(self.late_cooccur_lambda_scale),
                "late_only_after_freq_selection": bool(self.late_only_after_freq_selection),
            }
            if self.enable_adaptive_freq_basis:
                row.update(
                    {
                        "freq_mode": freq_mode,
                        "freq_active_num_bases": fact,
                        "freq_candidate_num_bases": fcand,
                        "freq_score_top_mean": freq_score_top_mean,
                    }
                )
            if self.enable_adaptive_freq_basis and self.enable_band_aware_freq_basis:
                row.update(
                    {
                        "enable_band_aware_freq_basis": True,
                        "freq_band_low_candidate_num": int(band_candidate_map.get("low", 0)),
                        "freq_band_mid_candidate_num": int(band_candidate_map.get("mid", 0)),
                        "freq_band_high_candidate_num": int(band_candidate_map.get("high", 0)),
                        "freq_band_low_active_num": int(band_active_map.get("low", 0)),
                        "freq_band_mid_active_num": int(band_active_map.get("mid", 0)),
                        "freq_band_high_active_num": int(band_active_map.get("high", 0)),
                        "freq_band_low_score_mean": band_score_mean_map["low"],
                        "freq_band_mid_score_mean": band_score_mean_map["mid"],
                        "freq_band_high_score_mean": band_score_mean_map["high"],
                        "freq_band_low_score_top_mean": band_score_top_mean_map["low"],
                        "freq_band_mid_score_top_mean": band_score_top_mean_map["mid"],
                        "freq_band_high_score_top_mean": band_score_top_mean_map["high"],
                        "freq_band_low_radius_mean": band_radius_mean_map["low"],
                        "freq_band_mid_radius_mean": band_radius_mean_map["mid"],
                        "freq_band_high_radius_mean": band_radius_mean_map["high"],
                        "freq_selected_low_indices": selected_idx_by_band["low"],
                        "freq_selected_mid_indices": selected_idx_by_band["mid"],
                        "freq_selected_high_indices": selected_idx_by_band["high"],
                        "freq_selected_low_coords": selected_coords_by_band["low"],
                        "freq_selected_mid_coords": selected_coords_by_band["mid"],
                        "freq_selected_high_coords": selected_coords_by_band["high"],
                    }
                )
            self._append_diag_row(diagnostics_csv_path, row)
            latest_diag = row

            epoch_log = (
                f"[ALSD-ALCE][epoch {epoch + 1}/{run_epochs}] "
                f"L_ent={row['L_entangle_bg']:.4f} L_anchor={row['L_anchor']:.4f} "
                f"L_flat={row['L_collapse_aux']:.4f} L_margin={row['L_margin']:.4f} "
                f"ALSI={row['alsi_score']:.2f} "
                f"Purity={row['confounder_purity_ratio']:.2%} PAG={row['pag_positive_ratio']:.2%} "
                f"Sat={row['saturation_ratio']:.2%} "
                f"Late={row['late_repair_ratio']:.2f} EntS={row['ent_scale_cur']:.2f} "
                f"AncS={row['anchor_scale_cur']:.2f} FlatS={row['flat_scale_cur']:.2f} "
                f"LogitS={row['preserve_logits_scale_cur']:.2f} "
                f"Max_D={row['max_abs_delta']:.4f} "
                f"CNP={row['L_cooccur_nt_logits']:.4f} CNPw={row['effective_cooccur_lambda']:.3f} "
                f"CNPcnt={row['cooccur_nt_fg_count']:.1f} CNPdrop={row['cooccur_nt_prob_drop']:.4f} "
                f"CNPhinge={row['cooccur_nt_prob_drop_hinge_mean']:.4f} "
                f"CNPhard={row['cooccur_nt_hard_loss_mean']:.4f} "
                f"CNPhcnt={row['cooccur_nt_hard_count']:.1f}"
            )
            if self.enable_adaptive_freq_basis:
                epoch_log += (
                    f" Freq={freq_mode} FAct={fact} FCand={fcand} FScoreTop={freq_score_top_mean:.3e}"
                )
            print(epoch_log)

            scheduler.step()

        if self.phase0_diagnostics_only:
            print("[LegacyBestReproduce] P0 signature check:")
            checks = [
                ("support_fallback_ratio", "eq", 1.0, 1.0),
                ("support_prebaked_ratio", "eq", 0.0, 0.0),
                ("support_forced_pseudo_fallback_ratio", "eq", 1.0, 1.0),
                ("perturbed_area_ratio_mean", "range", 0.158, 0.165),
                ("overlap_ratio", "range", 0.91, 0.97),
                ("pag_positive_ratio", "range", 0.50, 0.56),
                ("confounder_purity_ratio", "range", 0.845, 0.905),
                ("L_entangle_bg", "range", 3.10, 3.36),
                ("L_anchor", "range", 0.85, 0.97),
                ("L_collapse_aux", "range", 0.162, 0.169),
                ("saturation_ratio", "range", 0.085, 0.125),
                ("max_abs_delta", "range", 0.062745 - 1e-3, 0.062745 + 1e-3),
            ]
            p0_mismatch = False
            for key, mode, low, high in checks:
                val = float(latest_diag.get(key, float("nan")))
                if mode == "eq":
                    ok = (not math.isnan(val)) and abs(val - low) <= 1e-6
                    expected_str = f"{low:.6f}"
                else:
                    ok = (not math.isnan(val)) and (val >= low) and (val <= high)
                    expected_str = f"[{low:.6f}, {high:.6f}]"
                status = "OK" if ok else "Mismatch"
                print(f"  {key}={val:.6f} expected {expected_str} -> {status}")
                if not ok:
                    p0_mismatch = True
            extra_checks = [
                (
                    "cooccur_nt_fg_count",
                    float(latest_diag.get("cooccur_nt_fg_count", float("nan"))),
                    lambda v: (not math.isnan(v)) and (v > 0.0),
                    "> 0",
                ),
                (
                    "cooccur_nt_img_ratio",
                    float(latest_diag.get("cooccur_nt_img_ratio", float("nan"))),
                    lambda v: (not math.isnan(v)) and (v > 0.0),
                    "> 0",
                ),
                (
                    "cooccur_nt_clean_prob_mean",
                    float(latest_diag.get("cooccur_nt_clean_prob_mean", float("nan"))),
                    lambda v: not math.isnan(v),
                    "non-NaN",
                ),
                (
                    "cooccur_nt_adv_prob_mean",
                    float(latest_diag.get("cooccur_nt_adv_prob_mean", float("nan"))),
                    lambda v: not math.isnan(v),
                    "non-NaN",
                ),
            ]
            for key, val, fn_ok, expected_str in extra_checks:
                ok = bool(fn_ok(val))
                status = "OK" if ok else "Mismatch"
                print(f"  {key}={val:.6f} expected {expected_str} -> {status}")
                if not ok:
                    p0_mismatch = True
            metric_val = str(latest_diag.get("adaptive_freq_select_metric", ""))
            metric_ok = metric_val == "attack_collateral_ratio"
            print(
                f"  adaptive_freq_select_metric={metric_val} expected attack_collateral_ratio -> "
                f"{'OK' if metric_ok else 'Mismatch'}"
            )
            if not metric_ok:
                p0_mismatch = True
            quota_vals = (
                int(latest_diag.get("freq_band_low_active_num", -1)),
                int(latest_diag.get("freq_band_mid_active_num", -1)),
                int(latest_diag.get("freq_band_high_active_num", -1)),
            )
            quota_ok = quota_vals == (1, 13, 2)
            print(
                f"  band_active_quota={list(quota_vals)} expected [1, 13, 2] -> "
                f"{'OK' if quota_ok else 'Mismatch'}"
            )
            if not quota_ok:
                p0_mismatch = True
            expected_loss_type = str(self.cooccur_nt_loss_type)
            loss_type_val = str(latest_diag.get("cooccur_nt_loss_type", ""))
            loss_type_ok = loss_type_val == expected_loss_type
            print(
                f"  cooccur_nt_loss_type={loss_type_val} expected {expected_loss_type} -> "
                f"{'OK' if loss_type_ok else 'Mismatch'}"
            )
            if not loss_type_ok:
                p0_mismatch = True
            expected_hard_top_ratio = float(self.cooccur_nt_hard_top_ratio)
            hard_top_ratio_val = float(latest_diag.get("cooccur_nt_hard_top_ratio", float("nan")))
            hard_top_ratio_ok = (not math.isnan(hard_top_ratio_val)) and abs(
                hard_top_ratio_val - expected_hard_top_ratio
            ) <= 1e-9
            print(
                f"  cooccur_nt_hard_top_ratio={hard_top_ratio_val:.6f} expected {expected_hard_top_ratio:.6f} -> "
                f"{'OK' if hard_top_ratio_ok else 'Mismatch'}"
            )
            if not hard_top_ratio_ok:
                p0_mismatch = True
            expected_drop_tol = float(self.cooccur_nt_drop_tolerance)
            drop_tol_val = float(latest_diag.get("cooccur_nt_drop_tolerance", float("nan")))
            drop_tol_ok = (not math.isnan(drop_tol_val)) and abs(drop_tol_val - expected_drop_tol) <= 1e-9
            print(
                f"  cooccur_nt_drop_tolerance={drop_tol_val:.6f} expected {expected_drop_tol:.6f} -> "
                f"{'OK' if drop_tol_ok else 'Mismatch'}"
            )
            if not drop_tol_ok:
                p0_mismatch = True
            late_ratio_val = float(latest_diag.get("late_repair_ratio", float("nan")))
            late_ratio_ok = (not math.isnan(late_ratio_val)) and abs(late_ratio_val - 0.0) <= 1e-9
            print(
                f"  late_repair_ratio={late_ratio_val:.6f} expected 0.000000 -> "
                f"{'OK' if late_ratio_ok else 'Mismatch'}"
            )
            if not late_ratio_ok:
                p0_mismatch = True
            ent_scale_val = float(latest_diag.get("ent_scale_cur", float("nan")))
            anchor_scale_val = float(latest_diag.get("anchor_scale_cur", float("nan")))
            flat_scale_val = float(latest_diag.get("flat_scale_cur", float("nan")))
            ent_scale_ok = (not math.isnan(ent_scale_val)) and abs(ent_scale_val - 1.0) <= 1e-9
            anchor_scale_ok = (not math.isnan(anchor_scale_val)) and abs(anchor_scale_val - 1.0) <= 1e-9
            flat_scale_ok = (not math.isnan(flat_scale_val)) and abs(flat_scale_val - 1.0) <= 1e-9
            print(
                f"  ent_scale_cur={ent_scale_val:.6f} expected 1.000000 -> "
                f"{'OK' if ent_scale_ok else 'Mismatch'}"
            )
            print(
                f"  anchor_scale_cur={anchor_scale_val:.6f} expected 1.000000 -> "
                f"{'OK' if anchor_scale_ok else 'Mismatch'}"
            )
            print(
                f"  flat_scale_cur={flat_scale_val:.6f} expected 1.000000 -> "
                f"{'OK' if flat_scale_ok else 'Mismatch'}"
            )
            if not ent_scale_ok:
                p0_mismatch = True
            if not anchor_scale_ok:
                p0_mismatch = True
            if not flat_scale_ok:
                p0_mismatch = True
            if p0_mismatch:
                print("[LegacyBestReproduce] P0 signature mismatch. Do not run full.")
            else:
                print("[LegacyBestReproduce] P0 signature matched.")

        os.makedirs(os.path.dirname(global_params_path), exist_ok=True)
        if self.enable_adaptive_freq_basis:
            save_idx = self.freq_active_idx.long()
            save_idx_list = [int(i) for i in save_idx.detach().cpu().tolist()]
            save_coords = [tuple(self.freq_candidate_coords[int(i)]) for i in save_idx_list]
            save_fourier_coeff = self.fourier_coeff.detach()[save_idx].cpu()
        else:
            save_idx_list = []
            save_coords = self.coords
            save_fourier_coeff = self.fourier_coeff.detach().cpu()

        save_pack: Dict[str, Any] = {
            "coords": [[int(y), int(x)] for y, x in save_coords],
            "fourier_coeff": save_fourier_coeff,
            "suppress_small": self.suppress_small.detach().cpu(),
            "method": "tausb_mask",
            "target_class_id": self.target_class_id,
        }
        if self.enable_adaptive_freq_basis and self.enable_band_aware_freq_basis:
            save_pack.update(
                {
                    "enable_band_aware_freq_basis": True,
                    "freq_band_names": [str(x) for x in self.freq_band_names],
                    "freq_band_candidate_nums": [int(x) for x in self.freq_band_candidate_nums],
                    "freq_band_active_nums": [int(x) for x in self.freq_band_active_nums],
                    "freq_candidate_meta": self.freq_candidate_meta,
                    "freq_active_meta": [
                        self.freq_candidate_meta[int(i)] for i in save_idx_list if 0 <= int(i) < len(self.freq_candidate_meta)
                    ],
                }
            )

        torch.save(save_pack, global_params_path)

        atomic_write_json(
            diagnostics_json_path,
            {
                "latest": latest_diag,
                "global_params_path": global_params_path,
                "coords": [[int(y), int(x)] for y, x in save_coords],
            },
        )

        return global_params_path
