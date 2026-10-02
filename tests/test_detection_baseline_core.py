"""Protocol invariants independent of trained-model behavior."""

import unittest

import numpy as np
import torch

from ue_framework.baselines_det.core import (
    EPS, PersonOnlyBCE, decoded_pixel_audit, letterbox, person_batch, poison_image,
    project_delta, support_mask, support_tensor,
)


class DetectionBaselineCoreTests(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            {"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]},
            {"cls": 11, "bbox": [0.625, 0.5, 0.25, 0.5]},
        ]

    def test_half_overlap_and_outside_zero(self):
        support = support_mask(8, 8, self.annotations)
        self.assertEqual(float(support[3, 2]), 1.0)
        self.assertEqual(float(support[3, 5]), 0.5)
        self.assertEqual(float(support[0, 0]), 0.0)
        clean = torch.full((1, 3, 8, 8), 0.5)
        mask = support_tensor(clean, self.annotations)
        delta = project_delta(torch.full_like(clean, EPS * 3), mask)
        self.assertAlmostEqual(float(delta[0, 0, 3, 2]), EPS, places=7)
        self.assertAlmostEqual(float(delta[0, 0, 3, 5]), EPS / 2, places=7)
        self.assertEqual(float(delta[0, 0, 0, 0]), 0.0)
        self.assertTrue(torch.equal(poison_image(clean, torch.zeros_like(clean), mask), clean))

    def test_decoded_png_budget_and_label_agnostic_geometry(self):
        clean = np.full((8, 8, 3), 100, dtype=np.uint8)
        poison = clean.copy()
        support = support_mask(8, 8, self.annotations)
        poison[support == 1] += 8
        poison[support == 0.5] += 4
        stats = decoded_pixel_audit(clean, poison, self.annotations)
        self.assertEqual(stats["max_abs_delta_byte"],
                         {"target_only": 8, "overlap": 4, "outside": 0})
        poison[0, 0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "spatial budget"):
            decoded_pixel_audit(clean, poison, self.annotations)

    def test_person_only_loss_boxes_and_letterbox(self):
        image = torch.zeros(1, 3, 80, 160)
        padded, geometry = letterbox(image, 640)
        self.assertEqual(tuple(padded.shape), (1, 3, 640, 640))
        self.assertEqual((geometry.new_height, geometry.new_width, geometry.pad_top),
                         (320, 640, 160))
        batch = person_batch(image, self.annotations, 640)
        self.assertEqual(tuple(batch["bboxes"].shape), (1, 4))
        self.assertEqual(batch["cls"].tolist(), [[14.0]])
        self.assertAlmostEqual(float(batch["bboxes"][0, 0]), 0.5)
        self.assertAlmostEqual(float(batch["bboxes"][0, 1]), 0.5)
        self.assertAlmostEqual(float(batch["bboxes"][0, 2]), 0.5)
        self.assertAlmostEqual(float(batch["bboxes"][0, 3]), 0.25)

    def test_classification_gradient_reaches_person_only(self):
        logits = torch.zeros(1, 2, 20, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 0, 14] = 1.0
        result = PersonOnlyBCE()(logits, targets)
        result.sum().backward()
        self.assertGreater(float(logits.grad[..., 14].abs().sum()), 0)
        self.assertEqual(float(logits.grad[..., :14].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[..., 15:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
