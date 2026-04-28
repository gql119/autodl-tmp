import csv
import math
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..data_utils import image_has_target, label_path_for_image, list_images, load_image_rgb_float, read_yolo_annotations
from ..io_utils import atomic_write_json
from ..ultra.hijacked_loss import HijackedV8Loss
from .base import BasePoisonGenerator, PoisonResult
from .fourier import build_fourier_pattern, sample_midfreq_coords, spectrum_to_numpy
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

        self.coords = sample_midfreq_coords(
            h=self.imgsz,
            w=self.imgsz,
            num_bases=self.shortcut_num_bases,
            seed=int(cfg.get("experiment", {}).get("seeds", [0])[0]),
            enable_search=True,
        )

        self.fourier_coeff = torch.nn.Parameter(torch.zeros((self.shortcut_num_bases, 3), device=self.device))
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.universal_epochs, 
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

        for epoch in range(self.universal_epochs):
            if epoch < 15:
                cur_lambda_preserve = self.lambda_preserve * 0.3  
            elif epoch < 25:
                warmup_ratio = 0.3 + min(0.7, (epoch - 15) / 10.0 * 0.7)
                cur_lambda_preserve = self.lambda_preserve * warmup_ratio
            else:
                cur_lambda_preserve = self.lambda_preserve
                warmup_ratio = min(1.0, (epoch - 25) / 5.0)

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
                "support_empty_ratio": 0.0,
            }
            batch_count = 0

            for batch_list in loader:
                if not batch_list:
                    continue

                optimizer.zero_grad(set_to_none=True)

                valid_items = []
                batch_prebaked = 0
                batch_fallback = 0
                batch_empty = 0
                
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
                    elif support_source == "pseudo_fallback":
                        batch_fallback += 1
                    else:
                        batch_empty += 1
                        
                    if float(inner_np.sum()) > 10.0:
                        valid_items.append((data, inner_np, ring_np, support_source))
                
                total_support_count = batch_prebaked + batch_fallback + batch_empty
                if total_support_count > 0:
                    epoch_diag["support_prebaked_ratio"] += batch_prebaked / total_support_count
                    epoch_diag["support_fallback_ratio"] += batch_fallback / total_support_count
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

                raw_perturb, perturb, adv, _support, _jnd = self._compose_delta_batched(
                    img_t=img_t,
                    inner_t=inner_t,
                    ring_t=ring_t,
                    coords=self.coords,
                    fourier_coeff=self.fourier_coeff,
                    suppress_small=self.suppress_small,
                    current_epoch=epoch  
                )

                eot_count = max(1, self.eot_samples)
                
                L_ent_total = torch.zeros((), device=self.device)
                L_anchor_total = torch.zeros((), device=self.device)
                L_flat_total = torch.zeros((), device=self.device)
                L_preserve_total = torch.zeros((), device=self.device)
                
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

                    if real_assign_clean:
                        real_fg = real_assign_clean["fg_mask"].bool() 
                        real_labels = real_assign_clean["target_labels"].long()
                        
                        nt_fg_mask = real_fg & (real_labels != self.target_class_id)
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
                    
                    if cur_lambda_preserve > 0:
                        L_preserve_feat = torch.zeros((), device=self.device)
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
                            self.lambda_preserve_feat * L_preserve_feat
                            + self.lambda_preserve_logits * L_preserve_logits
                            + self.lambda_margin * L_margin
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

                    # 累加 EOT 的 ALCE losses
                    L_ent_total += L_entangle_bg
                    L_anchor_total += L_semantic_anchor
                    L_flat_total += L_collapse_aux
                    L_preserve_total += L_preserve

                # 计算 EOT 平均
                L_ent_final = L_ent_total / eot_count
                L_anchor_final = L_anchor_total / eot_count
                L_flat_final = L_flat_total / eot_count
                L_preserve_final = L_preserve_total / eot_count

                L_tv = self._tv_loss(raw_perturb)
                L_budget = F.relu(torch.max(torch.abs(raw_perturb)) - self.eps)

                total_loss = ( 
                    self.lambda_ent * L_ent_final
                    + self.lambda_anchor * L_anchor_final
                    + self.lambda_flat * L_flat_final
                    + cur_lambda_preserve * L_preserve_final 
                    + self.lambda_tv * L_tv  
                    + self.lambda_budget * L_budget          
                )

                total_loss.backward()

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
                    if k != "max_abs_delta":  
                        epoch_diag[k] = epoch_diag[k] / float(batch_count)

            row = {
                "epoch": epoch + 1,
                **epoch_diag,
                "grad_norm_fourier": grad_norm_fourier if "grad_norm_fourier" in locals() else float("nan"),
                "grad_norm_suppress": grad_norm_suppress if "grad_norm_suppress" in locals() else float("nan"),
            }
            self._append_diag_row(diagnostics_csv_path, row)
            latest_diag = row

            print(
                f"[ALSD-ALCE][epoch {epoch + 1}/{self.universal_epochs}] "
                f"L_ent={row['L_entangle_bg']:.4f} L_anchor={row['L_anchor']:.4f} "
                f"L_flat={row['L_collapse_aux']:.4f} L_margin={row['L_margin']:.4f} "
                f"ALSI={row['alsi_score']:.2f} "
                f"Purity={row['confounder_purity_ratio']:.2%} PAG={row['pag_positive_ratio']:.2%} "
                f"Sat={row['saturation_ratio']:.2%} "
                f"Max_D={row['max_abs_delta']:.4f}"
            )

            scheduler.step()

        os.makedirs(os.path.dirname(global_params_path), exist_ok=True)
        torch.save(
            {
                "coords": [[int(y), int(x)] for y, x in self.coords],
                "fourier_coeff": self.fourier_coeff.detach().cpu(),
                "suppress_small": self.suppress_small.detach().cpu(),
                "method": "tausb_mask",
                "target_class_id": self.target_class_id,
            },
            global_params_path,
        )

        atomic_write_json(
            diagnostics_json_path,
            {
                "latest": latest_diag,
                "global_params_path": global_params_path,
                "coords": [[int(y), int(x)] for y, x in self.coords],
            },
        )

        return global_params_path
