from __future__ import annotations

import re


try:
    from math_verify import parse, verify

    _MATH_VERIFY_AVAILABLE = True
except Exception:
    parse = None
    verify = None
    _MATH_VERIFY_AVAILABLE = False


def extract_boxed_answer(text: str) -> str | None:
    """Return the last answer inside a LaTeX \\boxed{...} expression."""
    text = text or ""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    left_brace = text.find("{", idx)
    if left_brace < 0:
        return None

    depth = 0
    for pos in range(left_brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[left_brace + 1 : pos].strip()
    return None


def _normalize_answer(text: str) -> str:
    text = str(text or "")
    text = text.replace("$", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\s+", "", text)
    return text.lower().strip()


def grade_answer(predicted: str | None, ground_truth: str | None) -> bool:
    """Grade a predicted math answer against a ground-truth answer."""
    if predicted is None or ground_truth is None:
        return False

    if _MATH_VERIFY_AVAILABLE:
        try:
            pred = predicted if "$" in predicted else f"${predicted}$"
            gt = ground_truth if "$" in ground_truth else f"${ground_truth}$"
            return bool(verify(parse(gt, fallback_mode="no_fallback"), parse(pred, fallback_mode="no_fallback")))
        except Exception:
            pass

    return _normalize_answer(predicted) == _normalize_answer(ground_truth)


def batch_verify_answer(completions: list[str], reference_solutions: list[str]) -> list[bool]:
    """Verify each completion against the corresponding reference solution."""
    results = []
    for completion, reference in zip(completions, reference_solutions):
        predicted = extract_boxed_answer(completion)
        target = extract_boxed_answer(reference) or reference
        results.append(grade_answer(predicted, target))
    return results
