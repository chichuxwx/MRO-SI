from __future__ import annotations

from contextlib import contextmanager

from aomp_opsd.mrosi_compat import MROSIDataCollator


@contextmanager
def temporary_padding_side(tokenizer, padding_side: str):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


class AOMPOPSDDataCollator(MROSIDataCollator):
    """MRO-SI collator with left-padded student prompts for HF generation.

    The teacher channel stays byte-for-byte aligned with the original MRO-SI
    right-padding behavior. Only the student rollout prompt padding changes, so
    decoder-only generation sees the true last prompt token on every row.
    """

    def _tokenize_prompts(self, prompts: list[str], prefix: str):
        padding_side = "left" if prefix == "student_prompts" else "right"
        with temporary_padding_side(self.tokenizer, padding_side):
            return super()._tokenize_prompts(prompts, prefix)
