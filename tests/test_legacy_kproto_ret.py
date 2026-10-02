"""Contract checks for the isolated legacy_kproto_ret method."""

import os
import tempfile
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ue_framework.config import load_config
from ue_framework.methods.alce_acgt import build_local_context_mask
from ue_framework.methods.legacy_kproto_ret import (
    LegacyKProtoTrainer, _local_context_mask, deterministic_kmeans, kproto_losses,
    non_target_retention_loss, select_prototype,
)


class LossContracts(unittest.TestCase):
    def test_context_mask_matches_legacy_pooling(self):
        for size in ((32, 37), (81, 64)):
            mask = torch.zeros((1, 1, *size))
            mask[:, :, 4:18, 8:21] = 1
            mask[:, :, -2:, -3:] = 1
            for inner, outer in ((2, 5), (4, 9)):
                expected = build_local_context_mask(mask, inner, outer)
                self.assertTrue(torch.equal(_local_context_mask(mask, inner, outer), expected))

    def test_prototype_selection_and_determinism(self):
        bank = torch.tensor([[1., 0.], [0., 1.]])
        indices, chosen, score = select_prototype(torch.tensor([[0., 3.]]), bank)
        self.assertEqual(indices.tolist(), [1])
        self.assertTrue(torch.equal(chosen, bank[1:]))
        self.assertAlmostEqual(float(score.item()), 1.0)
        vectors = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]])
        self.assertTrue(torch.equal(deterministic_kmeans(vectors, 2, 7), deterministic_kmeans(vectors, 2, 7)))

    def test_bg_direction_relative_hinge_and_retention(self):
        bg = torch.tensor([[1., 0.]])
        clean = torch.tensor([[0., 1.]])
        poisoned = bg.clone().requires_grad_()
        nt_clean = torch.tensor([[2., 3.]])
        nt_poison = nt_clean.clone().requires_grad_()
        l_bg, l_rel, l_ret, _, _ = kproto_losses(poisoned, clean, bg, nt_poison, nt_clean, 0.2)
        self.assertAlmostEqual(float(l_bg), 0.)
        self.assertAlmostEqual(float(l_rel), 0.)
        self.assertAlmostEqual(float(l_ret), 0.)
        self.assertGreater(float(kproto_losses(clean, bg, bg, nt_poison, nt_clean, 0.2)[1]), 0.)

    def test_no_non_target_is_finite_zero(self):
        p = torch.tensor([[0.8, 0.2]], requires_grad=True)
        z = p.new_zeros((0, 2))
        losses = kproto_losses(p, torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), z, z, 0.2)
        self.assertEqual(float(losses[2]), 0.)
        self.assertTrue(torch.isfinite(losses[2]))

    def test_retention_gradient_only_on_poisoned_feature(self):
        p = torch.tensor([[1., 0.]], requires_grad=True)
        clean = torch.tensor([[0., 1.]], requires_grad=True)
        nt_p = torch.tensor([[2., 4.]], requires_grad=True)
        nt_c = torch.tensor([[1., 3.]], requires_grad=True)
        ret = kproto_losses(p, clean, clean.detach(), nt_p, nt_c, 0.2)[2]
        ret.backward()
        self.assertGreater(float(nt_p.grad.norm()), 0.)
        self.assertIsNone(nt_c.grad)
        self.assertIsNone(p.grad)

    def test_retention_means_over_objects_across_images(self):
        one_object = torch.tensor([[2., 0.]], requires_grad=True)
        three_objects = torch.zeros((3, 2), requires_grad=True)
        clean = torch.zeros((4, 2))
        ret = non_target_retention_loss(torch.cat([one_object, three_objects]), clean)
        self.assertAlmostEqual(float(ret), 1.0)
        ret.backward()
        self.assertAlmostEqual(float(one_object.grad[0, 0]), 1.0)

    def test_configured_jnd_floor_reaches_legacy_schedule(self):
        checkpoint = os.environ.get("KPROTO_CKPT")
        config_path = os.environ.get("KPROTO_CONFIG")
        if not checkpoint or not config_path:
            self.skipTest("Set KPROTO_CKPT and KPROTO_CONFIG for the server integration test")
        from ultralytics import YOLO
        cfg = load_config(config_path)
        cfg["surrogate"]["ckpt"] = checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(checkpoint).model.to(device)
        trainer = LegacyKProtoTrainer(cfg, cfg["methods"]["legacy_kproto_ret"], device, model)
        image = torch.linspace(0.1, 0.9, 64 * 64, device=device).reshape(1, 1, 64, 64).repeat(1, 3, 1, 1)
        first = trainer._jnd_gain(image, current_floor=0.4)
        trainer.jnd_floor += 0.1
        changed = trainer._jnd_gain(image, current_floor=0.4)
        self.assertGreater(float((changed - first).abs().max()), 0.01)
        trainer.jnd_floor -= 0.1
        late = trainer._jnd_gain(image, current_floor=0.5)
        self.assertGreater(float((late - first).abs().max()), 0.01)


class FrozenModelAndSupport(unittest.TestCase):
    def test_strict_prebaked_support_fails_closed(self):
        trainer = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
        trainer.target_class_id = 14
        trainer.use_prebaked_instance_mask = True
        trainer.strict_instance_mask = True
        with tempfile.TemporaryDirectory() as root:
            trainer.instance_mask_dir = root
            image_path = os.path.join(root, "sample.jpg")
            target = [{"cls": 14, "bbox": (0.5, 0.5, 0.5, 0.5)}]
            with self.assertRaises(FileNotFoundError):
                trainer._build_support((32, 32, 3), target, image_path=image_path)
            mask_path = os.path.join(root, "sample.png")
            with open(mask_path, "wb") as handle:
                handle.write(b"not a PNG")
            with self.assertRaises(ValueError):
                trainer._build_support((32, 32, 3), target, image_path=image_path)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 1
            cv2.imwrite(mask_path, mask)
            support, _, source = trainer._build_support((32, 32, 3), target, image_path=image_path)
            self.assertEqual(source, "prebaked")
            self.assertEqual(int(support.sum()), 256)
            overlap = target + [{"cls": 6, "bbox": (0.5, 0.5, 1.0, 1.0)}]
            support, _, source = trainer._build_support((32, 32, 3), overlap, image_path=image_path)
            self.assertEqual(source, "prebaked_excluded")
            self.assertEqual(int(support.sum()), 0)
            cv2.imwrite(mask_path, np.zeros_like(mask))
            with self.assertRaises(ValueError):
                trainer._build_support((32, 32, 3), target, image_path=image_path)
            mask[8:24, 8:24] = 255
            cv2.imwrite(mask_path, mask)
            with self.assertRaises(ValueError):
                trainer._build_support((32, 32, 3), target, image_path=image_path)

    def test_one_step_freeze_gradient_support_and_budget(self):
        checkpoint = os.environ.get("KPROTO_CKPT")
        config_path = os.environ.get("KPROTO_CONFIG")
        if not checkpoint or not config_path:
            self.skipTest("Set KPROTO_CKPT and KPROTO_CONFIG for the server integration test")
        from ultralytics import YOLO
        cfg = load_config(config_path)
        cfg["surrogate"]["ckpt"] = checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(checkpoint).model.to(device)
        trainer = LegacyKProtoTrainer(cfg, cfg["methods"]["legacy_kproto_ret"], device, model)
        before = [p.detach().clone() for p in model.parameters()]
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        optimizer = torch.optim.Adam([trainer.fourier_coeff], lr=0.05)
        self.assertEqual([p for group in optimizer.param_groups for p in group["params"]], [trainer.fourier_coeff])
        image = torch.full((1, 3, 64, 64), 0.5, device=device)
        support = torch.zeros((1, 1, 64, 64), device=device)
        support[:, :, 16:48, 16:48] = 1
        trainer._epoch = 0
        raw, delta, poisoned, _, _ = trainer._compose_delta_batched(
            image, support, torch.ones_like(support), trainer.coords,
            trainer.fourier_coeff[:len(trainer.coords)], trainer.suppress_small, 0
        )
        self.assertEqual(float(delta[support.expand_as(delta) == 0].abs().max()), 0.)
        self.assertLessEqual(float(delta.abs().max()), trainer.eps + 1e-6)
        feat = trainer._feature(poisoned, grad=True)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        reference = F.normalize(torch.arange(pooled.shape[1], device=device).float()[None] + 1, dim=1)
        loss = 1.0 - F.cosine_similarity(pooled, reference).mean()
        loss.backward()
        self.assertIsNotNone(trainer.fourier_coeff.grad)
        self.assertGreater(float(trainer.fourier_coeff.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        optimizer.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
