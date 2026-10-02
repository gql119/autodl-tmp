import numpy as np
import unittest

from scripts.run_isolated_victim import (
    _accumulate_budget,
    _budget_image_stats,
    _finish_budget_accumulator,
    _new_budget_accumulator,
)


def _fixture():
    clean = np.full((12, 12, 3), 120, dtype=np.uint8)
    annotations = [
        {"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]},
        {"cls": 1, "bbox": [0.65, 0.5, 0.3, 0.5]},
    ]
    poison = clean.copy()
    target_mask, other_mask = _masks(annotations, clean.shape)
    overlap = target_mask & other_mask
    target_non_overlap = target_mask & ~other_mask
    poison[target_non_overlap, 0] -= 8
    poison[overlap, 1] -= 4
    return clean, poison, annotations, target_mask, other_mask


def _masks(annotations, shape):
    from scripts.run_isolated_victim import _bbox_masks

    return _bbox_masks(shape, annotations)


class PerturbationBudgetAuditTests(unittest.TestCase):
    def test_e8_budget_gate_and_reported_stats(self):
        clean, poison, annotations, target, other = _fixture()
        delta, per_pixel, regions, limits, maxima = _budget_image_stats(
            clean, poison, annotations, 8 / 255, 0.5
        )
        self.assertEqual(limits, {"target_non_overlap": 8, "overlap": 4})
        self.assertEqual(maxima["global"], 8)
        self.assertEqual(maxima["target_non_overlap"], 8)
        self.assertEqual(maxima["overlap"], 4)
        self.assertEqual(maxima["outside_target"], 0)
        acc = _new_budget_accumulator()
        _accumulate_budget(acc, delta, per_pixel, regions, limits)
        stats = _finish_budget_accumulator(acc)
        self.assertEqual(stats["max_abs_delta_byte"]["global"], 8)
        self.assertEqual(stats["saturation_ratio"]["target_non_overlap"], 1.0)
        self.assertEqual(stats["saturation_ratio"]["overlap"], 1.0)
        self.assertEqual(stats["perturbed_pixel_ratio"], np.count_nonzero(target) / target.size)
        self.assertAlmostEqual(stats["mean_abs_delta"]["target_non_overlap"], 8 / (3 * 255))
        self.assertAlmostEqual(stats["mean_abs_delta"]["overlap"], 4 / (3 * 255))


    def test_e8_budget_gate_rejects_overlap_and_outside_violations(self):
        clean, poison, annotations, target, other = _fixture()
        overlap = target & other
        bad_overlap = poison.copy()
        bad_overlap[overlap, 2] -= 5
        with self.assertRaisesRegex(RuntimeError, "overlap decoded-pixel budget"):
            _budget_image_stats(clean, bad_overlap, annotations, 8 / 255, 0.5)

        bad_outside = poison.copy()
        bad_outside[0, 0, 0] -= 1
        with self.assertRaisesRegex(RuntimeError, "Global decoded-pixel budget|outside target"):
            _budget_image_stats(clean, bad_outside, annotations, 8 / 255, 0.5)


if __name__ == "__main__":
    unittest.main()
