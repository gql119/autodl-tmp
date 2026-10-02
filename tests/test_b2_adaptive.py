"""Geometry and detection-evidence contracts for the B2 clean query."""

import unittest

import numpy as np
import torch

from ue_framework.methods.adaptive_learner import AdaptiveLearner
from scripts.run_isolated_victim import _summarize_clean_box_metrics


class AdaptiveQueryContracts(unittest.TestCase):
    def setUp(self):
        self.learner = AdaptiveLearner.__new__(AdaptiveLearner)
        self.learner.imgsz = 640
        self.learner.target_class_id = 14

    def test_original_labels_are_mapped_through_letterbox(self):
        image = np.zeros((100, 200, 3), dtype=np.float32)
        anns = [{"cls": 14, "bbox": (0.5, 0.5, 0.5, 0.5)}]
        batch = self.learner._batch(torch.zeros(1, 3, 100, 200), [(image, anns, None)])
        self.assertEqual(tuple(batch["img"].shape), (1, 3, 640, 640))
        torch.testing.assert_close(
            batch["bboxes"][0], torch.tensor([0.5, 0.5, 0.5, 0.25])
        )
        self.assertEqual(batch["cls"].tolist(), [[14.0]])
        self.assertEqual(batch["batch_idx"].tolist(), [0])

    def test_target_evidence_uses_both_score_and_localization(self):
        batch = {
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.25]]),
            "cls": torch.tensor([[14.0]]),
            "batch_idx": torch.tensor([0]),
        }
        prediction = torch.zeros(1, 24, 2)
        prediction[0, :4, 0] = torch.tensor([330.0, 320.0, 320.0, 160.0])
        prediction[0, 18, 0] = 0.9
        prediction[0, :4, 1] = torch.tensor([50.0, 50.0, 40.0, 40.0])
        prediction[0, 18, 1] = 0.9
        score = prediction.clone().requires_grad_()
        evidence = self.learner._target_evidence(score, batch)
        evidence.backward()
        self.assertGreater(float(evidence), 0.0)
        self.assertGreater(float(score.grad[0, 18, 0]), 0.0)
        self.assertNotEqual(float(score.grad[0, 0, 0]), 0.0)
        farther = prediction.clone()
        farther[0, 0, 0] = 420.0
        self.assertLess(float(self.learner._target_evidence(farther, batch)), float(evidence))


class CleanValidationMetricContracts(unittest.TestCase):
    def test_person_recall_is_read_from_class_aligned_metric(self):
        class Box:
            ap_class_index = np.arange(20)
            ap50 = np.linspace(0.2, 0.9, 20)
            r = np.linspace(0.1, 0.8, 20)
            map50 = float(ap50.mean())

        result = _summarize_clean_box_metrics(Box())
        self.assertAlmostEqual(result["Recall_target"], float(Box.r[14]))
        self.assertAlmostEqual(result["mAP50_target"], float(Box.ap50[14]))
        self.assertEqual(len(result["Recall_per_class"]), 20)
        Box.r = Box.r[:-1]
        with self.assertRaisesRegex(RuntimeError, "AP50/Recall"):
            _summarize_clean_box_metrics(Box())


if __name__ == "__main__":
    unittest.main()
