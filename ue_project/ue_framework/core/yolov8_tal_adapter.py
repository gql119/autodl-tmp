from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from .assignment_parser import AssignmentResult, infer_fpn_level_ids
from .detector_adapter import DetectorAdapter


class YOLOv8TALAdapter(DetectorAdapter):
    """Detector adapter backed by Ultralytics v8DetectionLoss and TAL assignment."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        protected_class_id: int,
        tal_topk: int = 10,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            model=model,
            num_classes=num_classes,
            protected_class_id=protected_class_id,
            assignment_topk=tal_topk,
            eps=eps,
        )
        self.criterion = v8DetectionLoss(model, tal_topk=tal_topk)
        if isinstance(self.criterion.hyp, dict):
            self.criterion.hyp = SimpleNamespace(**self.criterion.hyp)
        for name, value in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
            if not hasattr(self.criterion.hyp, name):
                setattr(self.criterion.hyp, name, value)

    def get_task_aligned_assignments(self, predictions: Dict[str, torch.Tensor], batch: Dict) -> AssignmentResult:
        state = self._build_assignment_state(predictions, batch)
        fg_mask = state["fg_mask"].bool()
        target_labels = state["target_labels"].long()
        target_scores = state["target_scores"]
        target_gt_idx = state["target_gt_idx"].long()
        assignment_counts = fg_mask.long()
        level_ids = infer_fpn_level_ids(fg_mask.shape[1], fg_mask.device).unsqueeze(0).expand_as(fg_mask)
        return AssignmentResult(
            fg_mask=fg_mask,
            target_gt_idx=target_gt_idx,
            target_labels=target_labels,
            target_scores=target_scores,
            assignment_counts=assignment_counts,
            level_ids=level_ids,
        )

    def compute_detection_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict,
        class_filter: Optional[Iterable[int]] = None,
        return_components: bool = False,
        include_background_negatives: bool = False,
    ):
        if class_filter is not None:
            assignment = self.get_task_aligned_assignments(predictions, batch)
            class_ids = torch.tensor([int(c) for c in class_filter], device=assignment.fg_mask.device)
            unit_mask = assignment.fg_mask & torch.isin(assignment.target_labels.long(), class_ids.long())
            components = self.compute_masked_detection_loss(
                predictions,
                batch,
                assignment,
                unit_mask,
                include_background_negatives=include_background_negatives,
                background_class_filter=class_filter,
            )
            return components if return_components else components["total_loss"]

        loss_vec, _detached = self.criterion(predictions, batch)
        if loss_vec.ndim > 0:
            box_loss, cls_loss, dfl_loss = loss_vec[0], loss_vec[1], loss_vec[2]
            total = loss_vec.sum()
        else:
            total = loss_vec
            zero = total * 0.0
            box_loss = zero
            cls_loss = zero
            dfl_loss = zero
        components = {
            "total_loss": total,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
        }
        return components if return_components else total

    def compute_masked_detection_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict,
        assignment: AssignmentResult,
        unit_mask: torch.Tensor,
        include_background_negatives: bool = False,
        background_class_filter: Optional[Iterable[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        state = self._build_assignment_state(predictions, batch)
        pred_distri = state["pred_distri"]
        pred_scores = state["pred_scores"]
        pred_bboxes = state["pred_bboxes"]
        anchor_points = state["anchor_points"]
        stride_tensor = state["stride_tensor"]
        target_bboxes = state["target_bboxes"]
        target_scores = state["target_scores"]
        target_labels = state["target_labels"].long().clamp(min=0, max=self.num_classes - 1)
        imgsz = state["imgsz"]

        selected = unit_mask.bool()
        zero = pred_scores.sum() * 0.0
        cls_loss = zero
        box_loss = zero
        dfl_loss = zero

        if selected.any():
            pos = torch.nonzero(selected, as_tuple=False)
            b_idx = pos[:, 0]
            a_idx = pos[:, 1]
            labels = target_labels[b_idx, a_idx]
            assigned_logits = pred_scores[b_idx, a_idx].gather(1, labels.unsqueeze(1)).squeeze(1)
            assigned_scores = target_scores[b_idx, a_idx].gather(1, labels.unsqueeze(1)).squeeze(1).to(pred_scores.dtype)
            score_sum = assigned_scores.sum().clamp_min(1.0)
            cls_loss = F.binary_cross_entropy_with_logits(assigned_logits, assigned_scores, reduction="sum") / score_sum

            filtered_scores = torch.zeros_like(target_scores)
            filtered_scores[selected] = target_scores[selected]
            box_loss, dfl_loss = self.criterion.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                filtered_scores,
                filtered_scores.sum().clamp_min(1.0),
                selected,
                imgsz,
                stride_tensor,
            )

        if include_background_negatives:
            bg_mask = ~state["fg_mask"].bool()
            if bg_mask.any():
                bg_logits = pred_scores[bg_mask]
                if background_class_filter is not None:
                    class_ids = torch.tensor(
                        [int(c) for c in background_class_filter],
                        device=bg_logits.device,
                        dtype=torch.long,
                    )
                    bg_logits = bg_logits[:, class_ids]
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    bg_logits,
                    torch.zeros_like(bg_logits),
                    reduction="mean",
                )

        box_loss = box_loss * self._hyp_gain("box")
        cls_loss = cls_loss * self._hyp_gain("cls")
        dfl_loss = dfl_loss * self._hyp_gain("dfl")
        return {
            "total_loss": box_loss + cls_loss + dfl_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
        }

    def _hyp_gain(self, name: str) -> float:
        hyp = self.criterion.hyp
        if isinstance(hyp, dict):
            return float(hyp.get(name, 1.0))
        return float(getattr(hyp, name, 1.0))

    def _build_assignment_state(self, predictions: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        preds = self.criterion.parse_output(predictions)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.criterion.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.criterion.device, dtype=dtype)
        imgsz = imgsz * self.criterion.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.criterion.preprocess(
            targets.to(self.criterion.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.criterion.bbox_decode(anchor_points, pred_distri)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        return {
            "pred_distri": pred_distri,
            "pred_scores": pred_scores,
            "pred_bboxes": pred_bboxes,
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
            "imgsz": imgsz,
            "target_labels": target_labels,
            "target_bboxes": target_bboxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask.bool(),
            "target_gt_idx": target_gt_idx,
        }
