from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from ue_framework.methods.alce_acgt import build_pag_gate, project_strict_gate_to_fpn


@dataclass(frozen=True)
class UnitPartition:
    target_gate: torch.Tensor
    selected_target_gate: torch.Tensor
    non_target_gate: torch.Tensor
    non_target_class_gates: Dict[int, torch.Tensor]
    layer_target_maps: Dict[str, torch.Tensor]
    layer_non_target_maps: Dict[str, torch.Tensor]
    stats: Dict


def partition_tal_units(
    fg_mask: torch.Tensor,
    target_labels: torch.Tensor,
    target_scores: torch.Tensor,
    target_class_id: int,
    layer_names: List[str],
    features: Dict[str, torch.Tensor],
    pag_layer_ratios: Sequence[float],
    pag_min_pos: Sequence[int],
) -> UnitPartition:
    if fg_mask.ndim != 2 or target_labels.shape != fg_mask.shape:
        raise ValueError("TAL fg_mask and target_labels must both be [B,N]")
    fg = fg_mask.bool()
    labels = target_labels.long()
    target = fg & (labels == int(target_class_id))
    non_target = fg & (labels != int(target_class_id)) & (labels >= 0)
    layer_sizes = [int(features[name].shape[-2] * features[name].shape[-1]) for name in layer_names]
    if sum(layer_sizes) != fg.shape[1]:
        raise ValueError(f"TAL/FPN alignment mismatch: anchors={fg.shape[1]}, layers={layer_sizes}")
    selected, pag_stats = build_pag_gate(
        strict_gate_1d=target,
        target_scores=target_scores,
        target_class_id=target_class_id,
        top_ratio=pag_layer_ratios,
        min_keep=pag_min_pos,
        layer_sizes=layer_sizes,
    )
    class_gates = {int(k): non_target & (labels == int(k)) for k in torch.unique(labels[non_target]).tolist()}
    target_maps = project_strict_gate_to_fpn(selected, layer_names, features)
    non_target_maps = project_strict_gate_to_fpn(non_target, layer_names, features)
    selected_count = int(selected.sum().item())
    target_count = int(target.sum().item())
    stats = {
        "num_fg": int(fg.sum().item()),
        "num_target_positive": target_count,
        "num_selected_target": selected_count,
        "num_non_target_positive": int(non_target.sum().item()),
        "target_unit_coverage": selected_count / max(1, target_count),
        "pag_positive_ratio": float(pag_stats.get("pag_positive_ratio", 0.0)),
        "pag": pag_stats,
    }
    return UnitPartition(target, selected, non_target, class_gates, target_maps, non_target_maps, stats)
