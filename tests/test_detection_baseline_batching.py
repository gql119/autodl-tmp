"""Small contract tests for batched sample-wise detector inputs."""

import unittest

import torch

from ue_framework.baselines_det.core import flip_annotations, person_batch
from ue_framework.baselines_det.production import (
    _batch_dict, entf_push_batch, make_item, new_counters,
)


class BatchedDetectorInputTests(unittest.TestCase):
    def test_batch_geometry_matches_single_image_builder_and_tracks_indices(self):
        images = [torch.rand(1, 3, 48, 64), torch.rand(1, 3, 32, 48)]
        annotations = [
            [{"cls": 14, "bbox": [0.4, 0.5, 0.3, 0.5]},
             {"cls": 1, "bbox": [0.45, 0.5, 0.1, 0.1]}],
            [{"cls": 14, "bbox": [0.6, 0.5, 0.4, 0.5]}],
        ]
        flips = [False, True]
        packed = _batch_dict(images, annotations, flips, imgsz=64)
        self.assertEqual(tuple(packed["img"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(packed["cls"].shape), (2, 1))
        self.assertEqual(packed["batch_idx"].tolist(), [0.0, 1.0])
        expected0 = person_batch(images[0], annotations[0], imgsz=64)
        expected1 = person_batch(images[1].flip(-1), flip_annotations(annotations[1]), imgsz=64)
        self.assertTrue(torch.allclose(packed["img"][0:1], expected0["img"]))
        self.assertTrue(torch.allclose(packed["img"][1:2], expected1["img"]))
        self.assertTrue(torch.allclose(packed["bboxes"][0:1], expected0["bboxes"]))
        self.assertTrue(torch.allclose(packed["bboxes"][1:2], expected1["bboxes"]))

    def test_item_support_is_sample_specific_and_zeroes_outside_person(self):
        image = torch.full((1, 3, 12, 16), 0.5)
        anns = [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}]
        item = make_item(image, anns, "sample")
        self.assertEqual(tuple(item.raw_delta.shape), tuple(image.shape))
        self.assertGreater(float(item.support.sum()), 0.0)
        self.assertEqual(float(item.support[0, 0, 0, 0]), 0.0)

    def test_counter_schema_is_explicit(self):
        counters = new_counters()
        self.assertEqual(set(counters), {
            "forward_calls", "backward_calls", "optimizer_steps", "pgd_steps",
            "noise_steps", "eot_samples", "stability_samples", "push_steps",
        })
        self.assertTrue(all(value == 0 for value in counters.values()))

    def test_entf_second_block_continues_existing_perturbation(self):
        image = torch.full((1, 3, 12, 16), 0.5)
        item = make_item(image, [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}], "sample")

        def extractor(images, _annotations):
            return [image.mean().reshape(1, 1) for image in images]

        counters = new_counters()
        first = entf_push_batch(extractor, [item], torch.zeros(1), steps=2,
                                seed=123, counters=counters)
        continued = entf_push_batch(extractor, [item], torch.zeros(1), steps=0,
                                    seed=456, initial_deltas=first, counters=counters)
        self.assertTrue(torch.equal(first[0], continued[0]))
        self.assertEqual(counters["push_steps"], 2)


if __name__ == "__main__":
    unittest.main()
