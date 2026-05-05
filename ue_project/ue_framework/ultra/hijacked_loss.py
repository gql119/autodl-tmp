import os
from typing import Dict, Tuple
import torch

try:
    from ultralytics.utils.loss import v8DetectionLoss as _V8Base
except Exception:  # pragma: no cover
    _V8Base = object

class HijackedV8Loss(_V8Base):
    """
    TSVC-DA 的安全版特征拦截器 + Strict AL-SVC 的独立探针。
    主要职责：
    1. 为 TSVC-DA 提供稳定的 Proxy 缓存
    2. 独立调用 TaskAlignedAssigner 窃取真实分配信息，彻底绕开 Ultralytics 的 loss.py 报错坑
    """
    def __init__(self, model=None, num_classes: int = 20, target_class_id: int = 14):
        self._super_ready = False
        if _V8Base is not object and model is not None:
            try:
                super().__init__(model)
                self._super_ready = True
            except Exception:
                self._super_ready = False

        self.num_classes = int(num_classes)
        self.target_class_id = int(target_class_id)
        
        self.last_assign_inputs: Dict = {}
        self.last_real_assign: Dict = {}
        
        # 🚀 探针保持开启窃取数据，但关闭阻断开关，确保 AL-SVC 训练能正常进行
        self.enable_strict_assign_probe = True
        self.stop_after_strict_probe = False  # 👈 已安全关闭阻断开关
        self._strict_probe_done = False

    @classmethod
    def from_surrogate(cls, surrogate, num_classes: int, target_class_id: int):
        instance = cls(model=surrogate, num_classes=num_classes, target_class_id=target_class_id)
        instance.surrogate = surrogate  # 存下模型，备用提取锚点
        return instance

    @staticmethod
    def _to_decoded_preds(preds: torch.Tensor) -> torch.Tensor:
        if isinstance(preds, (list, tuple)):
            if len(preds) == 0:
                raise RuntimeError("Empty predictions passed to HijackedV8Loss.")
            if torch.is_tensor(preds[0]) and preds[0].ndim == 3:
                preds = preds[0]
            else:
                raise RuntimeError("Decoded preds should be [B,C,N].")
        return preds

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
        x, y, w, h = boxes_xywh[..., 0], boxes_xywh[..., 1], boxes_xywh[..., 2], boxes_xywh[..., 3]
        return torch.stack([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dim=-1)

    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype)
        lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
        rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0.0) * (box1[:, 3] - box1[:, 1]).clamp(min=0.0))
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0.0) * (box2[:, 3] - box2[:, 1]).clamp(min=0.0))
        return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(eps)

    def _build_gt_tensors(self, batch: Dict, device: torch.device, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = image_shape
        cls = batch.get("cls").to(device)
        bboxes = batch.get("bboxes").to(device)
        batch_idx = batch.get("batch_idx").to(device).to(torch.long).reshape(-1)

        bsz = int(batch.get("batch_size", int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1))
        gt_count = [int((batch_idx == i).sum().item()) for i in range(bsz)]
        max_gt = max(gt_count) if gt_count else 0

        gt_labels = torch.full((bsz, max_gt, 1), -1.0, device=device)
        gt_bboxes = torch.zeros((bsz, max_gt, 4), device=device)
        mask_gt = torch.zeros((bsz, max_gt, 1), device=device)

        for i in range(bsz):
            pos = torch.nonzero(batch_idx == i, as_tuple=False).reshape(-1)
            if pos.numel() == 0: continue
            c, bb = cls[pos], bboxes[pos].clone()
            if torch.max(torch.abs(bb)) <= 1.5:
                bb[:, 0] *= w; bb[:, 1] *= h; bb[:, 2] *= w; bb[:, 3] *= h
            bb = self._xywh_to_xyxy(bb)
            n = pos.numel()
            gt_labels[i, :n, :] = c
            gt_bboxes[i, :n, :] = bb
            mask_gt[i, :n, :] = 1.0

        return gt_labels, gt_bboxes, mask_gt

    def _build_clean_gate(self, pred_scores_logits, pred_bboxes, gt_labels, gt_bboxes, mask_gt, assignment_topk):
        bsz, n_anchor, _ = pred_scores_logits.shape
        target_prob = torch.sigmoid(pred_scores_logits[:, :, self.target_class_id])
        gate = torch.zeros((bsz, n_anchor), device=pred_scores_logits.device, dtype=torch.bool)

        for b in range(bsz):
            valid = mask_gt[b, :, 0] > 0
            valid = valid & (gt_labels[b, :, 0].to(torch.long) == self.target_class_id)
            if not torch.any(valid): continue
            gt_b = gt_bboxes[b, valid]
            iou_mat = self._box_iou(pred_bboxes[b], gt_b)
            best_iou = iou_mat.max(dim=1).values
            align = (target_prob[b].detach() ** 0.5) * (best_iou.detach().clamp(min=0.0) ** 6.0)
            k = max(1, min(int(assignment_topk), int(align.numel())))
            idx = torch.topk(align, k=k, largest=True).indices
            gate[b, idx] = True
        return gate

    def cache_assign_inputs_only(self, preds: torch.Tensor, batch: Dict, image_shape: Tuple[int, int], assignment_topk: int = 100) -> Dict:
        preds_decoded = self._to_decoded_preds(preds)
        device = preds_decoded.device

        cls_start = 4
        cls_end = min(preds_decoded.shape[1], cls_start + self.num_classes)

        pred_scores_logits = preds_decoded[:, cls_start:cls_end, :].permute(0, 2, 1).contiguous()
        pred_bboxes = self._xywh_to_xyxy(preds_decoded[:, :4, :].permute(0, 2, 1).contiguous())

        h, w = image_shape
        x1, y1, x2, y2 = pred_bboxes.unbind(-1)
        pred_bboxes = torch.stack([
            x1.clamp(0.0, float(w)), y1.clamp(0.0, float(h)),
            x2.clamp(0.0, float(w)), y2.clamp(0.0, float(h))
        ], dim=-1)

        gt_labels, gt_bboxes, mask_gt = self._build_gt_tensors(batch, device=device, image_shape=image_shape)
        gate = self._build_clean_gate(
            pred_scores_logits=pred_scores_logits,
            pred_bboxes=pred_bboxes,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
            assignment_topk=assignment_topk,
        )

        self.last_assign_inputs = {
            "pred_scores_logits": pred_scores_logits,
            "pred_bboxes": pred_bboxes,
            "gt_labels": gt_labels,
            "gt_bboxes": gt_bboxes,
            "mask_gt": mask_gt,
            "fg_mask": gate,
        }
        return self.last_assign_inputs

    def get_assigned_targets_and_loss(self, preds, batch):
        # Stable output schema for strict assignment cache.
        self.last_real_assign = {
            "target_labels": None,
            "target_scores": None,
            "fg_mask": None,
            "target_gt_idx": None,
        }
        if not self._super_ready or not hasattr(self, 'assigner') or self.assigner is None:
            return None
            
        preds_decoded = self._to_decoded_preds(preds)
        device = preds_decoded.device
        
        cls_start = 4
        cls_end = min(preds_decoded.shape[1], cls_start + self.num_classes)
        pred_scores = preds_decoded[:, cls_start:cls_end, :].permute(0, 2, 1).contiguous().sigmoid()
        pred_bboxes = self._xywh_to_xyxy(preds_decoded[:, :4, :].permute(0, 2, 1).contiguous())
        
        # ====================================================
        # 🚀 终极数学破局：从锚点数量逆推真实分辨率！
        # 8400 -> 640, 5376 -> 512
        # ====================================================
        num_anchors = pred_scores.shape[1]
        true_size = int(((num_anchors * 1024 / 21) ** 0.5 + 16) // 32) * 32
        true_shape = (true_size, true_size)
        
        # 🚨 必须用真实的 true_shape (如 640x640) 重新缩放 GT，否则框全错位！
        gt_labels, gt_bboxes, mask_gt = self._build_gt_tensors(batch, device=device, image_shape=true_shape)
        
        # 根据推导出的真实尺寸，手搓绝对精准的锚点
        anchor_points = []
        for s in [8, 16, 32]:
            fh, fw = true_size // s, true_size // s
            sy = torch.arange(end=fh, dtype=torch.float32, device=device) + 0.5
            sx = torch.arange(end=fw, dtype=torch.float32, device=device) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchors = torch.stack((sx, sy), dim=-1).view(-1, 2) * s
            anchor_points.append(anchors)
        anchor_points = torch.cat(anchor_points, dim=0)
        
        try:
            # 独立召唤原生分配器，参数尺寸已完美统一！
            out = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach(),
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt
            )
            if isinstance(out, (tuple, list)):
                self.last_real_assign = {
                    "target_labels": out[0] if len(out) > 0 else None,
                    "target_scores": out[2] if len(out) > 2 else None,
                    "fg_mask": out[3] if len(out) > 3 else None,
                    "target_gt_idx": out[4] if len(out) > 4 else None,
                }
            self._run_strict_probe_once()
        except Exception as e:
            import traceback
            print(f"\n🚨 [STRICT PROBE ERROR] TaskAlignedAssigner 独立调用失败，堆栈：")
            traceback.print_exc()
            print("\n")
            
        return None

    def _run_strict_probe_once(self):
        if not self.enable_strict_assign_probe or self._strict_probe_done:
            return
            
        self._strict_probe_done = True
        
        print("\n" + "="*70)
        print("🎯 [STRICT AL-SVC PROBE] Intercepted Real TAL Assigner Outputs")
        print("="*70)
        for k, v in self.last_real_assign.items():
            if torch.is_tensor(v):
                if v.numel() > 0:
                    print(f"  - {k:15s}: shape={tuple(v.shape)}, dtype={v.dtype}, min={v.float().min().item():.2f}, max={v.float().max().item():.2f}")
                else:
                    print(f"  - {k:15s}: shape={tuple(v.shape)}, dtype={v.dtype} (Empty Tensor)")
            else:
                print(f"  - {k:15s}: type={type(v)}")
        print("="*70 + "\n")
        
        if self.stop_after_strict_probe:
            import sys
            print("[STRICT AL-SVC PROBE] Stopping execution as requested.")
            sys.exit(0)
