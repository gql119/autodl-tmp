from typing import Dict, Iterable, Tuple
import random
import torch
import torch.nn.functional as F


def fixed_batch_order(count: int, seed: int, epoch: int = 0):
    order=list(range(count)); random.Random(int(seed)+int(epoch)).shuffle(order); return order


def gradient_component_stats(gradient: torch.Tensor) -> Dict[str, float]:
    value = gradient.detach().reshape(-1).float()
    return {"l2": float(value.norm()), "l1": float(value.abs().sum()), "max_abs": float(value.abs().max()) if value.numel() else 0.0}


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = first.detach().reshape(-1).float(), second.detach().reshape(-1).float()
    if float(a.norm()) <= 1e-12 or float(b.norm()) <= 1e-12:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0))


def constrained_direction(target_direction: torch.Tensor, constraint_gradients: Iterable[torch.Tensor], max_iterations: int = 200, tolerance: float = 1e-7) -> Tuple[torch.Tensor, Dict]:
    direction = target_direction.detach().clone()
    constraints = [gradient.detach() for gradient in constraint_gradients if float(gradient.detach().norm()) > 1e-12]
    iteration = -1
    for iteration in range(max_iterations):
        maximum = 0.0
        for gradient in constraints:
            violation = float((gradient * direction).sum())
            maximum = max(maximum, violation)
            if violation > tolerance:
                direction = direction - (violation / float(gradient.square().sum().clamp_min(1e-12))) * gradient
        if maximum <= tolerance:
            break
    violations = [float((gradient * direction).sum()) for gradient in constraints]
    feasible = all(value <= tolerance * 10 for value in violations)
    if not feasible or iteration < 0:
        raise RuntimeError("constrained direction solver failed; weighted fallback is forbidden")
    return direction, {"status": "feasible", "iterations": iteration + 1, "constraints": len(constraints), "max_violation": max(violations, default=0.0)}


def object_aligned_warp(delta_object: torch.Tensor, annotations, image_size: int, target_class_id: int, dilation: int = 4):
    if delta_object.ndim != 3:
        raise ValueError("delta_object must be [3,H,W]")
    canvas = delta_object.new_zeros((3, image_size, image_size))
    target_mask = delta_object.new_zeros((1, image_size, image_size))
    non_target = delta_object.new_zeros((1, image_size, image_size))
    boxes = []
    for annotation in annotations:
        xc, yc, width, height = annotation["bbox"]
        x1=max(0,int((xc-width/2)*image_size)); x2=min(image_size,int((xc+width/2)*image_size))
        y1=max(0,int((yc-height/2)*image_size)); y2=min(image_size,int((yc+height/2)*image_size))
        if x2<=x1 or y2<=y1: continue
        if int(annotation["cls"]) == target_class_id: boxes.append((x1,y1,x2,y2))
        else: non_target[:,y1:y2,x1:x2]=1
    if dilation > 0:
        non_target=F.max_pool2d(non_target.unsqueeze(0),2*dilation+1,stride=1,padding=dilation)[0]
    for x1,y1,x2,y2 in boxes:
        pattern=F.interpolate(delta_object.unsqueeze(0),size=(y2-y1,x2-x1),mode="bilinear",align_corners=False)[0]
        valid=1-non_target[:,y1:y2,x1:x2]
        canvas[:,y1:y2,x1:x2]=canvas[:,y1:y2,x1:x2]+pattern*valid
        target_mask[:,y1:y2,x1:x2]=torch.maximum(target_mask[:,y1:y2,x1:x2],valid)
    canvas=canvas*target_mask
    overlap=float((target_mask*non_target).mean())
    return canvas, target_mask, non_target, {"valid_support_area":float(target_mask.mean()),"non_target_overlap_ratio":overlap}
