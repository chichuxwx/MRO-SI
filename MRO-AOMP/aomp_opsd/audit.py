from __future__ import annotations

from dataclasses import dataclass

import torch

from .mrosi_compat import batch_verify_answer


@dataclass(frozen=True)
class AuditResult:
    rewards: torch.Tensor
    source: str


def compute_sequence_audit_rewards(
    completions: list[str],
    reference_solutions: list[str],
    audit_source: str = "auto",
    device: torch.device | None = None,
) -> AuditResult:
    """Compute sequence-level audit scores.

    The first runnable backend uses the existing MRO-SI boxed-answer verifier.
    Reward model, unit-test, LLM-judge, and human-preference backends can plug
    into this function later without changing the token loss.
    """

    source = (audit_source or "auto").lower()
    if source not in {"auto", "verifier", "exact_match"}:
        # TODO: add reward_model, unit_test, and calibrated judge backends here.
        source = "verifier"

    correct = batch_verify_answer(completions, reference_solutions)
    rewards = torch.tensor([1.0 if value else 0.0 for value in correct], dtype=torch.float32, device=device)
    return AuditResult(rewards=rewards, source=source)
