"""Contract checks for full target boxes and half-strength overlap."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer
from ue_framework.methods.tausb_universal import TAUSBUniversalTrainer


class BBoxHalfOverlapTest(unittest.TestCase):
    def test_support_covers_target_and_halves_only_non_target_overlap(self):
        method = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
        method.support_mode = "bbox_half_overlap"
        method.target_class_id = 14
        annotations = [
            {"cls": 14, "bbox": [0.5, 0.5, 0.75, 0.75]},
            {"cls": 3, "bbox": [0.75, 0.5, 0.5, 0.5]},
        ]
        support, ring, source = method._build_support((8, 8, 3), annotations)
        self.assertEqual(source, "bbox_half_overlap")
        self.assertEqual(float(support[3, 2]), 1.0)
        self.assertEqual(float(support[3, 5]), 0.5)
        self.assertEqual(float(support[0, 0]), 0.0)
        self.assertFalse(ring.any())
        fully_overlapped = [
            {"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]},
            {"cls": 3, "bbox": [0.5, 0.5, 0.5, 0.5]},
        ]
        support, _, source = method._build_support((8, 8, 3), fully_overlapped)
        self.assertEqual(source, "bbox_half_overlap")
        self.assertTrue(np.all(support[support > 0] == 0.5))

    def test_saved_uint8_changes_every_target_pixel_and_obeys_budgets(self):
        method = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
        method.support_mode = "bbox_half_overlap"
        method.eps = 16 / 255
        image = torch.tensor([[[[0., 1., 1 / 255, 0.]],
                               [[1., 0., 254 / 255, 1.]],
                               [[0.5, 0.5, 0.5, 0.5]]]])
        support = torch.tensor([[[[1., 0.5, 0.5, 0.]]]])
        raw = torch.zeros_like(image)
        base = torch.zeros_like(image, requires_grad=True)

        def inherited(_self, img, binary, ring, *args):
            self.assertTrue(torch.equal(binary, (support > 0).float()))
            return raw, base, img, binary, torch.ones_like(binary)

        with patch.object(TAUSBUniversalTrainer, "_compose_delta_batched", inherited):
            _, delta, adv, _, _ = method._compose_delta_batched(
                image, support, torch.zeros_like(support), [], None, None)
        before = np.rint(image.numpy() * 255).astype(np.uint8)
        after = np.rint(adv.detach().numpy() * 255).astype(np.uint8)
        changed = np.any(before != after, axis=1)
        np.testing.assert_array_equal(changed[0, 0], np.array([True, True, True, False]))
        self.assertLessEqual(float(delta[:, :, :, 1:3].abs().max()), 8 / 255 + 1e-7)
        self.assertEqual(float(delta[:, :, :, 3:].abs().max()), 0.0)
        delta.sum().backward()
        self.assertIsNotNone(base.grad)
        self.assertGreater(float(base.grad.abs().sum()), 0.0)

    def test_actual_fourier_composition_reaches_every_target_pixel(self):
        method = LegacyKProtoTrainer.__new__(LegacyKProtoTrainer)
        method.support_mode = "bbox_half_overlap"
        method.eps = 16 / 255
        method.device = torch.device("cpu")
        method.imgsz = 32
        method.tanh_temp = 1.0
        method.freq_amp_buffer = 1.0
        method.lambda_freq = 1.0
        method.jnd_floor = 0.4
        method.jnd_ceiling = 1.0
        method.is_universal_training = False
        pixels = torch.arange(0, 8 * 8 * 3, dtype=torch.int64).reshape(1, 3, 8, 8)
        image = (pixels.remainder(256).float() / 255).detach()
        support = torch.zeros((1, 1, 8, 8))
        support[:, :, 1:7, 1:5] = 1.0
        support[:, :, 3:6, 3:5] = 0.5
        coeff = torch.tensor([[1.0, 0.5, -0.5], [0.2, 1.0, 0.7]], requires_grad=True)
        _, delta, adv, _, _ = method._compose_delta_batched(
            image, support, torch.zeros_like(support), [(1, 1), (2, 3)], coeff, torch.zeros(1)
        )
        before = np.rint(image.numpy() * 255).astype(np.uint8)
        after = np.rint(adv.detach().numpy() * 255).astype(np.uint8)
        changed = np.any(before != after, axis=1)
        np.testing.assert_array_equal(changed, support[:, 0].numpy() > 0)
        self.assertLessEqual(float(delta[:, :, 3:6, 3:5].abs().max()), 8 / 255 + 1e-7)
        weights = torch.arange(1, adv.numel() + 1, dtype=adv.dtype).reshape_as(adv)
        (adv * weights).sum().backward()
        self.assertGreater(float(coeff.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
