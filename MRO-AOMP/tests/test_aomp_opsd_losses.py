from __future__ import annotations

import unittest

import torch

from aomp_opsd.losses import (
    OutcomeCalibratedLossConfig,
    compute_group_advantages,
    compute_outcome_calibrated_opsd_loss,
    normalize_token_weights,
)


class AOMPOPSDLossTests(unittest.TestCase):
    def test_compute_group_advantages(self):
        rewards = torch.tensor([1.0, 0.0, 0.25, 0.25])
        group_ids = torch.tensor([0, 0, 1, 1])
        advantages = compute_group_advantages(rewards, group_ids)
        self.assertTrue(torch.allclose(advantages[:2], torch.tensor([1.0, -1.0]), atol=1e-5))
        self.assertTrue(torch.allclose(advantages[2:], torch.zeros(2), atol=1e-5))

    def test_token_weight_normalization_with_masks(self):
        weights = torch.tensor([[2.0, 2.0, 10.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        normalized = normalize_token_weights(weights, mask, mode="sequence_sum")
        self.assertTrue(torch.allclose(normalized[0], torch.tensor([0.5, 0.5, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(normalized[1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6))

    def _toy_logits(self):
        student_logits = torch.tensor(
            [[[0.0, 1.0, -1.0], [0.5, 0.0, -0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]], dtype=torch.float32)
        sampled = torch.tensor([[1, 0]])
        mask = torch.tensor([[1, 1]], dtype=torch.bool)
        return student_logits, teacher_logits, sampled, mask

    def test_positive_advantages_activate_positive_path(self):
        student_logits, teacher_logits, sampled, mask = self._toy_logits()
        loss, diagnostics = compute_outcome_calibrated_opsd_loss(
            student_logits,
            teacher_logits,
            sampled,
            mask,
            advantages=torch.tensor([1.0]),
            rewards=torch.tensor([1.0]),
            config=OutcomeCalibratedLossConfig(uniform_token_weights=True),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(diagnostics["loss_positive"]), 0.0)
        self.assertEqual(float(diagnostics["negative_fraction"]), 0.0)

    def test_negative_advantages_activate_negative_path(self):
        student_logits, teacher_logits, sampled, mask = self._toy_logits()
        loss, diagnostics = compute_outcome_calibrated_opsd_loss(
            student_logits,
            teacher_logits,
            sampled,
            mask,
            advantages=torch.tensor([-1.0]),
            rewards=torch.tensor([0.0]),
            config=OutcomeCalibratedLossConfig(uniform_token_weights=True),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(diagnostics["loss_negative"]), 0.0)
        self.assertEqual(float(diagnostics["positive_fraction"]), 0.0)

    def test_loss_is_finite_with_empty_responses(self):
        student_logits = torch.zeros((1, 2, 3), requires_grad=True)
        teacher_logits = torch.zeros((1, 2, 3))
        sampled = torch.zeros((1, 2), dtype=torch.long)
        mask = torch.zeros((1, 2), dtype=torch.bool)
        loss, diagnostics = compute_outcome_calibrated_opsd_loss(
            student_logits,
            teacher_logits,
            sampled,
            mask,
            advantages=torch.tensor([0.0]),
            rewards=torch.tensor([0.0]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(diagnostics["token_weight_max"]), 0.0)


if __name__ == "__main__":
    unittest.main()
