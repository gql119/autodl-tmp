import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from ..data_utils import image_has_target, label_path_for_image, load_image_rgb_float, read_yolo_annotations


def _save_rgb_png(path: str, image_rgb_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def tensor_to_signed_rgb(delta: np.ndarray, eps: float, support: Optional[np.ndarray] = None) -> np.ndarray:
    """Map signed perturbation from [-eps, eps] to RGB with gray as zero."""
    eps = max(float(eps), 1e-12)
    vis = (np.clip(delta / eps, -1.0, 1.0) * 0.5 + 0.5) * 255.0
    vis = np.clip(vis, 0.0, 255.0).astype(np.uint8)
    if support is not None:
        mask = support > 0.5
        vis = vis.copy()
        vis[~mask] = np.array([128, 128, 128], dtype=np.uint8)
    return vis


def tensor_to_abs_heatmap(delta: np.ndarray, eps: float, support: Optional[np.ndarray] = None) -> np.ndarray:
    """Simple black-red-yellow-white heatmap for mean absolute perturbation."""
    eps = max(float(eps), 1e-12)
    intensity = np.mean(np.abs(delta), axis=2) / eps
    intensity = np.clip(intensity, 0.0, 1.0)
    if support is not None:
        intensity = intensity * (support > 0.5).astype(np.float32)

    r = np.clip(3.0 * intensity, 0.0, 1.0)
    g = np.clip(3.0 * intensity - 1.0, 0.0, 1.0)
    b = np.clip(3.0 * intensity - 2.0, 0.0, 1.0)
    heat = np.stack([r, g, b], axis=2)
    return (heat * 255.0).astype(np.uint8)


def _add_title(image_rgb_u8: np.ndarray, title: str, title_h: int = 34) -> np.ndarray:
    h, w = image_rgb_u8.shape[:2]
    canvas = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    canvas[title_h:, :, :] = image_rgb_u8
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_panel(
    clean: np.ndarray,
    poisoned: np.ndarray,
    delta_eff: np.ndarray,
    eps: float,
    save_path: str,
    amplify: float = 10.0,
) -> None:
    signed = tensor_to_signed_rgb(delta_eff, eps)
    heat = tensor_to_abs_heatmap(delta_eff, eps)
    amplified = np.clip(clean + float(amplify) * delta_eff, 0.0, 1.0)

    tiles = [
        _add_title(_to_u8(clean), "Clean"),
        _add_title(_to_u8(poisoned), "Poisoned"),
        _add_title(signed, "Signed delta"),
        _add_title(_to_u8(amplified), f"Clean + {float(amplify):g}x delta"),
        _add_title(heat, "Abs delta heatmap"),
    ]
    spacer = np.full((tiles[0].shape[0], 8, 3), 255, dtype=np.uint8)
    panel = tiles[0]
    for tile in tiles[1:]:
        panel = np.concatenate([panel, spacer, tile], axis=1)
    _save_rgb_png(save_path, panel)


def _find_first_target_sample(image_paths: List[str], label_dir: str, target_class_id: int):
    for img_path in image_paths:
        anns = read_yolo_annotations(label_path_for_image(img_path, label_dir))
        if image_has_target(anns, target_class_id):
            return img_path, anns
    return None, []


def _build_tausb_global_delta(generator, imgsz: int, eps: float) -> Optional[np.ndarray]:
    if not all(hasattr(generator, name) for name in ["_build_global_freq_pattern", "coords", "fourier_coeff"]):
        return None
    with torch.no_grad():
        pattern = generator._build_global_freq_pattern(
            int(imgsz),
            int(imgsz),
            generator.coords,
            generator.fourier_coeff,
        )
        tanh_coeff = 4.0
        delta = generator.lambda_freq * torch.tanh(pattern * tanh_coeff)
        delta = torch.clamp(delta, -float(eps), float(eps))
    return delta.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def save_noise_metadata(path: str, metadata: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_universal_noise_visualizations(
    artifact_root: str,
    global_params,
    trainer_or_generator,
    cfg: Dict,
    method_cfg: Dict,
    device,
    image_paths: Optional[List[str]] = None,
    train_label_dir: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    support_type: str = "mask",
) -> None:
    """Save final universal-noise and one effective per-image perturbation visualization."""
    del device  # The generator already owns the device and tensors.
    noise_dir = os.path.join(artifact_root, "noise")
    os.makedirs(noise_dir, exist_ok=True)

    generator = trainer_or_generator
    eps = float(method_cfg.get("eps", cfg.get("experiment", {}).get("eps", getattr(generator, "eps", 16 / 255))))
    imgsz = int(cfg.get("surrogate", {}).get("imgsz", getattr(generator, "imgsz", 640)))
    target_class_id = int(cfg.get("experiment", {}).get("target_class_id", getattr(generator, "target_class_id", -1)))
    global_params_path = global_params if isinstance(global_params, str) else ""

    metadata = {
        "method": "tausb_mask",
        "seed": seed,
        "steps": steps,
        "eps": eps,
        "imgsz": imgsz,
        "target_class_id": target_class_id,
        "global_params_path": global_params_path,
        "strict_instance_mask": bool(method_cfg.get("strict_instance_mask", getattr(generator, "strict_instance_mask", False))),
        "ring_width": int(method_cfg.get("ring_width", getattr(generator, "ring_width", 0))),
        "support_type": support_type,
    }

    global_delta = _build_tausb_global_delta(generator, imgsz=imgsz, eps=eps)
    if global_delta is not None:
        _save_rgb_png(os.path.join(noise_dir, "global_delta_signed.png"), tensor_to_signed_rgb(global_delta, eps))
        _save_rgb_png(os.path.join(noise_dir, "global_delta_abs_heatmap.png"), tensor_to_abs_heatmap(global_delta, eps))
        metadata["global_delta_max_abs"] = float(np.max(np.abs(global_delta)))
        metadata["global_delta_mean_abs"] = float(np.mean(np.abs(global_delta)))
    else:
        metadata["warning"] = "global Fourier parameters unavailable; skipped global delta visualization"

    selected_image_path = None
    anns = []
    if image_paths and train_label_dir:
        selected_image_path, anns = _find_first_target_sample(image_paths, train_label_dir, target_class_id)

    if selected_image_path is None:
        metadata["selected_image_path"] = None
        metadata["warning"] = metadata.get("warning", "no target-class training sample found")
        save_noise_metadata(os.path.join(noise_dir, "noise_metadata.json"), metadata)
        print(f"[NoiseVisualizer] saved noise visualizations to: {noise_dir}")
        if global_delta is not None:
            print("[NoiseVisualizer] global_delta_signed.png")
        print("[NoiseVisualizer][WARN] no target sample found; example effective delta panel skipped")
        return

    clean = load_image_rgb_float(selected_image_path)
    result = generator.generate(
        image=clean,
        annotations=anns,
        seed=int(seed or 0),
        steps=int(steps or 0),
        eps=eps,
        support_type=support_type,
        image_path=selected_image_path,
    )
    poisoned = result.poisoned_image
    delta_eff = result.perturbation
    support = result.support_mask

    _save_rgb_png(
        os.path.join(noise_dir, "effective_delta_signed_example.png"),
        tensor_to_signed_rgb(delta_eff, eps, support=support),
    )
    _save_rgb_png(
        os.path.join(noise_dir, "effective_delta_abs_heatmap_example.png"),
        tensor_to_abs_heatmap(delta_eff, eps, support=support),
    )
    make_panel(
        clean=clean,
        poisoned=poisoned,
        delta_eff=delta_eff,
        eps=eps,
        save_path=os.path.join(noise_dir, "noise_visualization_panel_example.png"),
        amplify=float(method_cfg.get("noise_visualization_amplify", 10.0)),
    )

    diff = np.max(np.abs(delta_eff), axis=2)
    metadata.update(
        {
            "selected_image_path": selected_image_path,
            "support_area_ratio": float(np.mean(support > 0.5)) if support is not None else 0.0,
            "perturbed_area_ratio": float(np.mean(diff > (1.0 / 255.0))),
            "max_abs_delta": float(np.max(np.abs(delta_eff))),
            "mean_abs_delta": float(np.mean(np.abs(delta_eff))),
            "support_prebaked_or_fallback": str((result.extras or {}).get("support_source", "")),
        }
    )
    save_noise_metadata(os.path.join(noise_dir, "noise_metadata.json"), metadata)

    print(f"[NoiseVisualizer] saved noise visualizations to: {noise_dir}")
    print("[NoiseVisualizer] global_delta_signed.png")
    print("[NoiseVisualizer] effective_delta_signed_example.png")
    print("[NoiseVisualizer] noise_visualization_panel_example.png")
