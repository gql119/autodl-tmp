"""B2 Frozen-only semantic ablations. The original adaptive method is reused via MRO."""
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .b2_adaptive import B2AdaptiveTrainer
from .legacy_kproto_ret import (LegacyKProtoTrainer, _rectangle, select_prototype,
                                kproto_losses, non_target_retention_loss)
from .semantic_coordinates import normalized_orth, project_bases_to_channels, CLASSES, sha256
from .orth_reference_design import reference_energy_orth


class SemanticFrozenTrainer(LegacyKProtoTrainer):
    def _batch_loss(self, items, prototypes):
        """One surrogate clean/poisoned forward for a legacy-style padded batch."""
        pad_h = (max(image.shape[0] for image, _, _ in items) + 31) // 32 * 32
        pad_w = (max(image.shape[1] for image, _, _ in items) + 31) // 32 * 32
        images, supports = [], []
        for image, _, support in items:
            h, w = image.shape[:2]
            images.append(F.pad(torch.from_numpy(image).permute(2, 0, 1).float(),
                                (0, pad_w - w, 0, pad_h - h), value=0.5))
            supports.append(F.pad(torch.from_numpy(support)[None].float(),
                                  (0, pad_w - w, 0, pad_h - h)))
        tensor = torch.stack(images).to(self.device)
        support_t = torch.stack(supports).to(self.device)
        raw, delta, poison, _, _ = self._compose_delta_batched(
            tensor, support_t, torch.zeros_like(support_t), self._active_coords,
            self.fourier_coeff[self._active_idx], self.suppress_small, current_epoch=self._epoch
        )
        clean_aug, poison_aug = self._apply_shared_eot_pair_batched(tensor, poison)
        clean = self._feature(clean_aug, grad=False).detach()
        poisoned = self._feature(poison_aug, grad=True)
        image_losses, non_target_pairs = [], []
        selected_losses, displacements, clean_targets, overlap_weights = [], [], [], []
        class_drifts = {c: [] for c in self.bases}
        target_drift, proto_ids, proto_similarity, context_norm = [], [], [], []
        for b, (image, annotations, support) in enumerate(items):
            h0_map, hp_map = clean[b:b + 1], poisoned[b:b + 1]
            targets, contexts, non_targets = self._masks(
                annotations, support, image.shape[:2], hp_map, (pad_h, pad_w)
            )
            h0, hp, ct = [], [], []
            overlaps = []
            target_anns = [a for a in annotations if int(a["cls"]) == self.target_class_id]
            nt_anns = [a for a in annotations if int(a["cls"]) != self.target_class_id]
            for ann, target_mask, context_mask in zip(target_anns, targets, contexts):
                clean_target = self._pooled(h0_map, [target_mask])
                poison_target = self._pooled(hp_map, [target_mask])
                context = self._pooled(h0_map, [context_mask])
                if clean_target and poison_target and context:
                    h0.append(clean_target[0])
                    hp.append(poison_target[0])
                    ct.append(context[0])
                    box = _rectangle(image.shape[:2], ann["bbox"])
                    overlap = any(bool((box * _rectangle(image.shape[:2], a["bbox"])).any()) for a in nt_anns)
                    overlaps.append(self.overlap_weight if overlap else 1.)
            if not hp:
                continue
            h0, hp, ct = torch.stack(h0), torch.stack(hp), torch.stack(ct)
            indices, selected, similarity = select_prototype(ct, prototypes)
            nt0 = self._pooled(h0_map, non_targets)
            ntp = self._pooled(hp_map, non_targets)
            nt0 = torch.stack(nt0) if nt0 else hp_map.new_zeros((0, hp_map.shape[1]))
            ntp = torch.stack(ntp) if ntp else hp_map.new_zeros((0, hp_map.shape[1]))
            image_losses.append(kproto_losses(hp, h0, selected, ntp, nt0, self.margin))
            selected_losses.append(kproto_losses(hp[:, self.channels], h0[:, self.channels],
                                    selected[:, self.channels], ntp, nt0, self.margin))
            displacements.append(hp - h0.detach())
            clean_targets.append(h0.detach())
            overlap_weights.extend(overlaps)
            for ann, mask in zip(nt_anns, non_targets):
                a = self._pooled(h0_map, [mask]); p = self._pooled(hp_map, [mask])
                if a and p:
                    class_drifts[int(ann["cls"])].append((p[0]-a[0]).square().sum())
            if ntp.numel():
                non_target_pairs.append((ntp, nt0))
            target_drift.append((hp - h0).square().sum(dim=-1).mean())
            proto_ids.extend(indices.detach().cpu().tolist())
            proto_similarity.append(similarity.mean())
            context_norm.append(ct.norm(dim=-1).mean())
        if not image_losses:
            return None
        bg, rel, _, cos_clean, cos_bg = [
            torch.stack([loss[i] for loss in image_losses]).mean() for i in range(5)
        ]
        if non_target_pairs:
            ret = non_target_retention_loss(
                torch.cat([pair[0] for pair in non_target_pairs]),
                torch.cat([pair[1] for pair in non_target_pairs]),
            )
        else:
            ret = poisoned.sum() * 0.0
        bg_full, rel_full, cos_clean_full, cos_bg_full = bg, rel, cos_clean, cos_bg
        bg, rel, _, cos_clean, cos_bg = [torch.stack([x[i] for x in selected_losses]).mean() for i in range(5)]
        displacement = torch.cat(displacements)
        selected_displacement = displacement.index_select(1, self.channels)
        orth, conflicts, worst = normalized_orth(selected_displacement, self.orth_bases,
            displacement.new_tensor(overlap_weights), self.worst_k, self.orth_eps)
        orth_full, _, _ = normalized_orth(displacement, self.bases,
            displacement.new_tensor(overlap_weights), self.worst_k, self.orth_eps)
        reference_orth, _, _ = reference_energy_orth(selected_displacement,
            torch.cat(clean_targets).index_select(1, self.channels), self.orth_bases,
            displacement.new_tensor(overlap_weights), self.worst_k, self.orth_eps)
        orth_objective = reference_orth if self.orth_objective == 'reference_energy' else orth
        tv = self._tv_loss(raw)
        budget = F.relu(raw.abs().max() - self.eps)
        weighted_bg = self.lambda_bg * bg
        weighted_rel = self.lambda_rel * rel
        weighted_ret = self.lambda_ret * ret
        total = weighted_bg + weighted_rel + weighted_ret + self.lambda_orth * orth_objective + self.lambda_tv * tv + self.lambda_budget * budget
        self._semantic_terms = {"BG_sel": weighted_bg, "rel_sel": weighted_rel,
                                "ret": weighted_ret, "orth": self.lambda_orth * orth_objective}
        self._orth_raw = orth_objective
        self._frozen_total = total
        metrics = {
            "L_BG": bg, "L_rel": rel, "L_ret": ret,
            "weighted_L_BG": weighted_bg, "weighted_L_rel": weighted_rel,
            "weighted_L_ret": weighted_ret, "L_tv": tv, "L_budget": budget,
            "L_total": total, "cos_target_clean": cos_clean, "cos_target_bg": cos_bg,
            "relative_gap": cos_bg - cos_clean, "non_target_feature_drift": ret,
            "target_feature_drift": torch.stack(target_drift).mean(),
            "selected_proto_id": torch.tensor(float(proto_ids[0]), device=self.device),
            "selected_proto_ids": ",".join(str(int(i)) for i in proto_ids),
            "context_proto_similarity": torch.stack(proto_similarity).mean(),
            "context_feature_norm": torch.stack(context_norm).mean(),
            "max_abs_delta": delta.abs().max(), "mean_abs_delta": delta.abs().mean(),
            "saturation_ratio": (delta.abs() >= self.eps - 1e-6).float().mean(),
            "support_area_ratio": (support_t > 0).float().mean(),
            "outside_support_max": (delta * (support_t == 0)).abs().max(),
        }
        metrics.update({
            "L_BG_full": bg_full, "L_rel_full": rel_full,
            "cos_target_bg_full": cos_bg_full, "cos_target_clean_full": cos_clean_full,
            "L_BG_sel": bg, "L_rel_sel": rel, "cos_target_bg_sel": cos_bg,
            "cos_target_clean_sel": cos_clean, "L_orth_worst": orth,
            "L_orth_worst_tsfl_channels": orth,
            "L_orth_worst_full_channels": orth_full,
            "L_orth_reference_energy": reference_orth,
            "L_orth_objective": orth_objective,
            "orth_objective_reference_energy": poisoned.new_tensor(float(self.orth_objective == 'reference_energy')),
            "weighted_L_orth": self.lambda_orth * orth_objective,
            "target_displacement_norm": displacement.norm(dim=-1).mean(),
            "target_displacement_tsfl_norm": selected_displacement.norm(dim=-1).mean(),
            "orth_coordinate_channels": poisoned.new_tensor(float(self.channels.numel())),
            "mean_non_target_feature_drift": ret,
        })
        for j, c in enumerate(sorted(self.bases)):
            metrics["L_orth_"+CLASSES[c]] = conflicts[j]
            metrics["feature_drift_"+CLASSES[c]] = (torch.stack(class_drifts[c]).mean()
                if class_drifts[c] else poisoned.sum()*0.)
            metrics["feature_count_"+CLASSES[c]] = poisoned.new_tensor(len(class_drifts[c]))
        for j in range(self.worst_k):
            c = worst[j]
            metrics[f"worst_class_{j+1}"] = poisoned.new_tensor(c)
            metrics[f"worst_Lorth_{j+1}"] = conflicts[sorted(self.bases).index(c)]
        for key in ("num_selected_channels", "selected_channel_fraction", "mean_S_target_selected",
                    "mean_S_nt_selected", "mean_S_target_unselected", "mean_S_nt_unselected",
                    "specificity_selected_mean", "channel_selection_overlap_with_previous_run"):
            source_key = "num_channels_selected" if key == "num_selected_channels" else key
            value = len(self.channels)/len(self.tsfl["S_target"]) if key == "selected_channel_fraction" else self.summary[source_key]
            metrics[key] = poisoned.new_tensor(value)
        return total, metrics


class B2SemanticTrainer(B2AdaptiveTrainer, SemanticFrozenTrainer):
    def __init__(self, cfg, method_cfg, device, frozen_reference, adaptive_model):
        super().__init__(cfg, method_cfg, device, frozen_reference, adaptive_model)
        settings = method_cfg["semantic"]
        artifacts = Path(settings["artifact_dir"])
        self.tsfl = torch.load(artifacts/"target_specific_channels.pt", weights_only=True)
        sub = torch.load(artifacts/"non_target_subspaces.pt", weights_only=True)
        with open(artifacts/"tsfl_summary.json") as f:
            self.summary = json.load(f)
        identity = self.tsfl["identity"]
        if identity != sub["identity"] or identity["checkpoint_sha256"] != sha256(cfg["surrogate"]["ckpt"]):
            raise ValueError("Semantic coordinates checkpoint identity mismatch")
        if identity["feature_layer"] != self.feature_layer or identity["validation_used"]:
            raise ValueError("Invalid semantic layer or calibration split")
        for key, val in identity["settings"].items():
            if settings[key] != val:
                raise ValueError("Semantic coordinate setting changed: "+key)
        if identity["data_root"] != cfg["data"]["dataset_root"] or identity["train_images"] != cfg["data"]["train_images"] or identity["train_labels"] != cfg["data"]["train_labels"]:
            raise ValueError("Calibration dataset mismatch")
        self.channels = self.tsfl["indices"].to(device)
        self.bases = {int(c): item["basis"].to(device) for c, item in sub["classes"].items()}
        if sorted(self.bases) != [c for c in range(self.num_classes) if c != self.target_class_id]:
            raise ValueError("Incomplete non-target class subspaces")
        self.orth_bases = project_bases_to_channels(self.bases, self.channels)
        self.lambda_orth = float(settings["lambda_orth"])
        self.orth_objective = settings.get('orth_objective', 'angular')
        if self.orth_objective not in ('angular', 'reference_energy'):
            raise ValueError('Unknown orthogonality objective')
        self.overlap_weight = float(settings["overlap_orth_weight"])
        self.worst_k = int(settings["orth_worst_k"])
        self.orth_eps = float(settings.get("orth_eps", 1e-12))
        self.grad_interval = int(settings.get("gradient_interval", 50))
        if not 1 <= self.worst_k <= len(self.bases) or self.grad_interval < 1:
            raise ValueError("Invalid semantic diagnostics parameters")
        self._semantic_calls = 0

    def _batch_loss(self, items, prototypes):
        # B2's adaptive inner update and selection logic are executed without editing its file.
        result = super()._batch_loss(items, prototypes)
        if result is None:
            return None
        total, metrics = result
        active = self._semantic_calls % self.grad_interval == 0 or self.max_steps > 0
        self._semantic_calls += 1
        names = ["BG_sel", "rel_sel", "ret", "orth", "adapt", "total"]
        grads = {}
        if active:
            terms = dict(self._semantic_terms, adapt=metrics["L_adaptive"], total=total)
            for key in names:
                grads[key] = torch.autograd.grad(terms[key], self._last_poison, retain_graph=True)[0]
            grads["attack"] = grads["BG_sel"] + grads["rel_sel"]
            raw_grad = torch.autograd.grad(self._orth_raw, self._last_poison, retain_graph=True)[0]
        metrics["gradient_diagnostics_active"] = total.new_tensor(float(active))
        for key in names + ["attack"]:
            metrics["grad_"+key+"_norm"] = grads[key].norm() if active else total.new_zeros(())
        metrics["grad_orth_raw_norm"] = raw_grad.norm() if active else total.new_zeros(())
        for key in ["attack", "adapt", "ret"]:
            metrics["cos_grad_"+key+"_orth"] = ((grads[key]*grads["orth"]).sum() /
                (grads[key].norm()*grads["orth"].norm()).clamp_min(1e-12)) if active else total.new_zeros(())
        return total, metrics

    def train_universal(self, train_img_dir, train_label_dir, global_params_path,
                        diagnostics_csv_path, diagnostics_json_path, seed):
        result = super().train_universal(train_img_dir, train_label_dir, global_params_path,
                                         diagnostics_csv_path, diagnostics_json_path, seed)
        # Parent averages numeric EOT diagnostics; rebuild class IDs from averaged conflicts.
        with open(diagnostics_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        for row in rows:
            worst = sorted(self.bases, key=lambda c: (-float(row["L_orth_"+CLASSES[c]]), c))[:self.worst_k]
            for j,c in enumerate(worst):
                row[f"worst_class_{j+1}"] = c
                row[f"worst_Lorth_{j+1}"] = float(row["L_orth_"+CLASSES[c]])
        with open(diagnostics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        with open(diagnostics_json_path, encoding="utf-8") as f:
            diagnostic = json.load(f)
        for j in range(self.worst_k):
            for key in (f"worst_class_{j+1}", f"worst_Lorth_{j+1}"):
                diagnostic["latest"][key] = rows[-1][key]
        with open(diagnostics_json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic,f,indent=2)
        return result
