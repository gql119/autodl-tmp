import unittest
import torch
from ue_framework.methods.orth_reference_design import reference_energy_orth
from ue_framework.methods.semantic_coordinates import normalized_orth


class ReferenceOrthTests(unittest.TestCase):
    def test_amplitude_scaling_and_gradient(self):
        bases = {1: torch.tensor([[1.], [0.]], dtype=torch.float64)}
        clean = torch.tensor([[2., 1.]], dtype=torch.float64, requires_grad=True)
        weight = torch.ones(1, dtype=torch.float64)
        values, norms = [], []
        angular, angular_norms = [], []
        for scale in (1., 16.):
            delta = torch.tensor([[scale, scale]], dtype=torch.float64, requires_grad=True)
            loss, _, _ = reference_energy_orth(delta, clean, bases, weight, worst_k=1)
            grads = torch.autograd.grad(loss, (delta, clean), allow_unused=True)
            self.assertIsNone(grads[1])
            values.append(loss.detach()); norms.append(grads[0].norm())
            angle, _, _ = normalized_orth(delta, bases, weight, worst_k=1)
            angular.append(angle.detach())
            angular_norms.append(torch.autograd.grad(angle, delta)[0].norm())
        torch.testing.assert_close(values[1], values[0] * 256.)
        torch.testing.assert_close(norms[1], norms[0] * 16.)
        torch.testing.assert_close(angular[0], angular[1])
        torch.testing.assert_close(angular_norms[0], angular_norms[1] * 16.)

    def test_weighted_ranking_and_zero_reference_rejection(self):
        delta = torch.tensor([[1., 0.], [0., 1.]])
        bases = {3: torch.tensor([[1.], [0.]]), 8: torch.tensor([[0.], [1.]])}
        loss, _, worst = reference_energy_orth(delta, torch.ones_like(delta), bases,
                                              torch.tensor([1., 4.]), worst_k=1)
        self.assertEqual(worst, [8])
        torch.testing.assert_close(loss, torch.tensor(1.))
        with self.assertRaises(ValueError):
            reference_energy_orth(delta, torch.zeros_like(delta), bases,
                                  torch.ones(2), worst_k=1)


if __name__ == '__main__':
    unittest.main()
