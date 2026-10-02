"""Targeted tests for the REM/SEM computation-graph distinction."""

import unittest
from unittest.mock import patch

import torch

from ue_framework.baselines_det import rem, sem


ANNOTATIONS = [{"cls": 14, "bbox": [0.5, 0.5, 0.5, 0.5]}]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def toy_loss(model, image, _annotations, _imgsz):
    value = image.square().mean() * model.weight
    return value, {"toy": value.detach()}


class DistinctAlgorithmsTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.full((1, 3, 8, 8), 0.5)
        self.delta = torch.zeros_like(self.image)

    def test_rem_calls_inner_adversarial_ascent_for_each_defender_sample(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        calls = []

        def fake_ascent(_model, base, _annotations, _support, **_kwargs):
            calls.append(base.shape)
            return torch.zeros_like(base)

        with patch.object(rem, "target_detection_loss", toy_loss), \
             patch.object(rem, "adversarial_ascent", fake_ascent), \
             patch.object(rem, "adversarial_surrogate_step", return_value=0.5):
            result = rem.rem_update(
                model, self.image, ANNOTATIONS, self.delta, optimizer,
                rem.REMSchedule(defender_steps=2, eot_samples=3,
                                adversarial_steps=2), imgsz=32,
            )
        self.assertEqual(len(calls), 6)
        self.assertEqual(result["inner_adversarial_calls"], 7)
        self.assertTrue(all(value > 0 for value in result["perturbation_gradient_norms"]))

    def test_sem_defender_samples_uniform_without_inner_pgds(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        calls = []

        def counted_loss(_model, image, _annotations, _imgsz):
            calls.append(image.detach().clone())
            return toy_loss(_model, image, _annotations, _imgsz)

        with patch.object(sem, "target_detection_loss", counted_loss), \
             patch.object(sem, "adversarial_surrogate_step", return_value=0.5):
            result = sem.sem_update(
                model, self.image, ANNOTATIONS, self.delta, optimizer,
                sem.SEMSchedule(defender_steps=2, stability_samples=3,
                                surrogate_adversarial_steps=2), imgsz=32,
            )
        self.assertEqual(len(calls), 6)
        self.assertEqual(result["defender_inner_adversarial_calls"], 0)
        self.assertEqual(result["stability_draws"], 6)
        self.assertTrue(all(value > 0 for value in result["perturbation_gradient_norms"]))


if __name__ == "__main__":
    unittest.main()
