from __future__ import annotations

import unittest

import torch

from aomp_opsd.lookahead import temporary_sgd_step, trainable_parameters


class AOMPOPSDLookaheadTests(unittest.TestCase):
    def test_temporary_lookahead_restores_parameters(self):
        model = torch.nn.Linear(3, 2, bias=False)
        before = [param.detach().clone() for param in model.parameters()]
        grads = [torch.ones_like(param) for param in trainable_parameters(model)]
        with temporary_sgd_step(model, grads, step_size=0.1) as stats:
            during = [param.detach().clone() for param in model.parameters()]
            self.assertFalse(torch.allclose(before[0], during[0]))
            self.assertGreater(stats.update_norm, 0.0)
            self.assertEqual(stats.updated_tensors, 1)
        after = [param.detach().clone() for param in model.parameters()]
        self.assertTrue(torch.allclose(before[0], after[0]))


if __name__ == "__main__":
    unittest.main()
