from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ue_framework.core import DetectorAdapter
from ue_framework.methods.learning_trajectory import LearningTrajectoryMethod


class DiagnosticToyDetector(nn.Module):
    def __init__(self, num_classes: int = 20, num_units: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_units = int(num_units)
        self.head = nn.Linear(3, self.num_units * self.num_classes)
        anchors = torch.tensor(
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
                [0.25, 0.75, 0.20, 0.20],
                [0.75, 0.25, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("anchors_xywh", anchors[: self.num_units])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        logits = self.head(pooled).reshape(images.shape[0], self.num_units, self.num_classes)
        boxes = self.anchors_xywh.to(images.device, images.dtype).unsqueeze(0).expand(images.shape[0], -1, -1)
        return torch.cat([boxes, logits], dim=-1)


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_batch(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "cls": torch.tensor([14, 1, 14, 1], dtype=torch.long, device=device),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "batch_idx": torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device),
        "batch_size": 2,
    }


def _device_from_config(cfg: Dict) -> torch.device:
    requested = str(cfg.get("diagnostics", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _prepare_config(cfg: Dict) -> Dict:
    protected = int(cfg.get("protected_class_id", cfg.get("experiment", {}).get("target_class_id", 14)))
    num_classes = int(cfg.get("num_classes", cfg.get("surrogate", {}).get("num_classes", 20)))
    out = dict(cfg)
    out["protected_class_id"] = protected
    out["num_classes"] = num_classes
    out.setdefault("authorized_class_ids", "auto")
    out.setdefault("trajectory", {})
    out.setdefault("class_routing", {})
    out.setdefault("virtual_update", {})
    out.setdefault("meta", {})
    out["trajectory"].setdefault("parameter_scope", "head")
    out["trajectory"].setdefault("normalize_per_parameter", True)
    out["trajectory"].setdefault("lambda_protected", 1.0)
    out["trajectory"].setdefault("lambda_authorized", 1.0)
    out["trajectory"].setdefault("eps", 1.0e-8)
    out["class_routing"].setdefault("exclude_ambiguous", True)
    out["class_routing"].setdefault("include_background_negatives", False)
    out["virtual_update"].setdefault("parameter_scope", "head")
    out["virtual_update"].setdefault("steps", 1)
    out["virtual_update"].setdefault("lr", 0.001)
    out["meta"].setdefault("use_p1_regularizer", out.get("method") == "trajectory_meta_p2")
    out["meta"].setdefault("lambda_meta", 1.0)
    out["meta"].setdefault("lambda_p1", 0.2)
    out["meta"].setdefault("lambda_protected_query", 1.0)
    out["meta"].setdefault("enable_clean_counterfactual", True)
    return out


def run_diagnostics(config_path: str) -> Tuple[Dict[str, float], str]:
    cfg = _prepare_config(_load_config(config_path))
    seed = int(cfg.get("seed", cfg.get("experiment", {}).get("seeds", [0])[0] if cfg.get("experiment", {}).get("seeds") else 0))
    torch.manual_seed(seed)
    device = _device_from_config(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = DiagnosticToyDetector(num_classes=int(cfg["num_classes"])).to(device)
    adapter = DetectorAdapter(
        model,
        num_classes=int(cfg["num_classes"]),
        protected_class_id=int(cfg["protected_class_id"]),
        assignment_topk=int(cfg.get("assignment_topk", 1)),
    )
    method = LearningTrajectoryMethod(adapter, cfg)

    support_images = torch.rand(2, 3, 8, 8, device=device)
    query_images = torch.rand(2, 3, 8, 8, device=device)
    batch = _make_batch(device)

    delta_p1 = (torch.randn_like(support_images) * 0.01).requires_grad_()
    p1 = method.compute_p1_step(support_images, delta_p1, batch)
    p1_grad = torch.autograd.grad(p1["loss"], delta_p1, retain_graph=False, allow_unused=False)[0]

    delta_p2 = (torch.randn_like(support_images) * 0.01).requires_grad_()
    p2 = method.compute_p2_step(support_images, query_images, batch, batch, delta_p2)
    p2_grad = torch.autograd.grad(p2["loss"], delta_p2, retain_graph=False, allow_unused=False)[0]

    peak_gpu_memory = 0.0
    if device.type == "cuda":
        peak_gpu_memory = float(torch.cuda.max_memory_allocated(device))

    metrics: Dict[str, float] = {}
    metrics.update({k: float(v) for k, v in p1["logs"].items() if isinstance(v, (float, int))})
    metrics.update({k: float(v) for k, v in p2["logs"].items() if isinstance(v, (float, int))})
    metrics["p1_gradient_norm_to_delta"] = float(p1_grad.detach().norm().item())
    metrics["p2_gradient_norm_to_delta"] = float(p2_grad.detach().norm().item())
    metrics["peak_gpu_memory"] = peak_gpu_memory

    output_path = str(cfg.get("diagnostics", {}).get("output_path", "outputs/p0_diagnostics/diagnostics.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small P1/P2 autograd diagnostic.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics, output_path = run_diagnostics(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[run_p0_diagnostics] wrote {output_path}")


if __name__ == "__main__":
    main()
