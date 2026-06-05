"""Outcome-audited AOMP-OPSD experimental method."""

from .losses import (
    compute_group_advantages,
    compute_outcome_calibrated_opsd_loss,
    compute_prefix_reliability,
    compute_student_entropy,
    compute_teacher_entropy,
    compute_token_kl,
    normalize_token_weights,
)

__all__ = [
    "compute_group_advantages",
    "compute_outcome_calibrated_opsd_loss",
    "compute_prefix_reliability",
    "compute_student_entropy",
    "compute_teacher_entropy",
    "compute_token_kl",
    "normalize_token_weights",
]
