from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .verifier_utils import extract_boxed_answer


LEAKAGE_TERMS = (
    "reference solution",
    "reference answer",
    "ground truth",
    "gold answer",
    "verifier",
    "hidden answer",
    "official solution",
)

MASK_PLACEHOLDER = "[OMITTED]"

PROMPT_ECHO_TERMS = (
    "the json must",
    "return strict json",
    "do not use markdown",
    "must not have any other keys",
    "for example,",
    "write one json object",
    "json schema",
    "masked_derivation:",
    "omitted_result:",
    "3-8 concise sentences",
    "where the requested result belongs",
    "type description only",
    "1-3 short non-answer consistency checks",
)

FINAL_ANSWER_TERMS = (
    "final answer",
    "the answer is",
    "answer is",
    "final result is",
    "result is",
)


@dataclass
class MaskedDerivationCalibration:
    passed: bool
    reason: str
    masked_derivation: str = ""
    omitted_result: str = ""
    checks: list[str] | None = None


def build_privileged_masked_derivation_prompt(problem: str, reference_solution: str) -> str:
    return (
        "Convert the worked math solution into one private route hint for another solver.\n"
        "The route hint may use the method and equations, but it must hide the requested result.\n"
        "Reply with exactly one JSON object and no surrounding text.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Worked solution:\n{reference_solution}\n\n"
        "JSON schema:\n"
        '{"masked_derivation":"3-8 concise sentences ending with [OMITTED] where the requested result belongs",'
        '"omitted_result":"type description only","checks":["1-3 short non-answer consistency checks"]}\n\n'
        "Hard rules:\n"
        "- The first output character must be {.\n"
        "- Include [OMITTED] in masked_derivation.\n"
        "- Do not reveal or restate the requested result, boxed answer, choice label, or proof conclusion.\n"
        "- Do not mention JSON rules, prompts, hidden/reference/official solutions, scoring, or verifiers.\n"
        "- Do not use markdown or code fences."
    )


def _try_load_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _balanced_json_objects(text: str) -> list[str]:
    objects = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def _parse_jsonish(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    candidates = []
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    )
    candidates.extend(_balanced_json_objects(text))
    candidates.append(text)

    for candidate in candidates:
        data = _try_load_json(candidate)
        if data is not None and "masked_derivation" in data:
            return data
    return None


def parse_masked_derivation(raw_text: str) -> dict[str, Any] | None:
    parsed = _parse_jsonish(raw_text)
    if parsed is not None:
        return parsed
    text = (raw_text or "").strip()
    if not text:
        return None
    return {"masked_derivation": text, "omitted_result": "", "checks": []}


def _contains_leakage(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in LEAKAGE_TERMS)


def _contains_prompt_echo(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in PROMPT_ECHO_TERMS)


def _contains_final_answer_wording(text: str) -> bool:
    lowered = (text or "").lower()
    placeholder = re.escape(MASK_PLACEHOLDER.lower())
    lowered = re.sub(
        rf"(?:final answer|the answer is|answer is|final result is|result is)\s*(?:[:=])?\s*{placeholder}",
        "",
        lowered,
    )
    return any(term in lowered for term in FINAL_ANSWER_TERMS)


def _reference_answer(reference_solution: str, reference_answer: str | None = None) -> str:
    answer = (reference_answer or "").strip()
    if not answer:
        answer = extract_boxed_answer(reference_solution) or ""
    answer = answer.strip()
    answer = answer.removeprefix("\\boxed{").removesuffix("}")
    return answer.strip()


def _contains_answer_copy(
    combined: str,
    omitted_result: str,
    reference_solution: str,
    reference_answer: str | None = None,
) -> bool:
    answer = _reference_answer(reference_solution, reference_answer)
    if not answer:
        return False

    answer_compact = re.sub(r"\s+", "", answer)
    combined_compact = re.sub(r"\s+", "", combined)
    omitted_compact = re.sub(r"\s+", "", omitted_result)
    if answer_compact and answer_compact == omitted_compact:
        return True

    if not answer_compact:
        return False

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer_compact):
        answer_pattern = re.escape(answer_compact)
        numeric_patterns = [
            rf"{answer_pattern}\[OMITTED\]",
            rf"{answer_pattern}(?:plums?|seconds?|ways?|cases?|arrangements?|cubes?|routes?|choices?)",
        ]
        if any(re.search(pattern, combined_compact, flags=re.IGNORECASE) for pattern in numeric_patterns):
            return True
        # Numeric final answers are easy to leak accidentally; a standalone
        # occurrence is usually more harmful than losing one noisy hint.
        if len(answer_compact) >= 2 and re.search(rf"(?<![\w.]){answer_pattern}(?![\w.])", combined):
            return True

    final_context = (
        r"(?:answer|final|omitted|results?|values?|therefore|thus|hence|gives|obtain|equals|total|count|number)"
    )
    escaped = re.escape(answer_compact)
    patterns = [
        rf"{final_context}[^\n.;:]{{0,32}}{escaped}",
        rf"{escaped}[^\n.;:]{{0,32}}{final_context}",
        rf"\\boxed\s*\{{\s*{escaped}\s*\}}",
    ]
    lowered = combined.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
        return True

    # Long symbolic answers are unlikely to appear as harmless intermediate values.
    return len(answer_compact) >= 8 and answer_compact in combined_compact


def _redact_answer_text(text: str, answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return text

    redacted = text
    boxed_pattern = re.compile(rf"\\boxed\s*\{{\s*{re.escape(answer)}\s*\}}", flags=re.IGNORECASE)
    redacted = boxed_pattern.sub(MASK_PLACEHOLDER, redacted)

    # Redact exact answer occurrences with boundaries. This fallback favors not
    # leaking the requested result over preserving every intermediate constant.
    escaped = re.escape(answer)
    redacted = re.sub(rf"(?<![\w.\[]){escaped}(?![\w.\]])", MASK_PLACEHOLDER, redacted)
    return redacted


def fallback_masked_derivation_from_reference(
    reference_solution: str,
    reference_answer: str | None = None,
    max_chars: int = 1800,
) -> str:
    """Create a conservative masked route from the worked solution if generation fails."""
    answer = _reference_answer(reference_solution, reference_answer)
    text = str(reference_solution or "").strip()
    if not text:
        return ""

    text = re.sub(r"<\|begin_of_thought\|>|<\|end_of_thought\|>", "", text)
    text = re.sub(r"<\|begin_of_solution\|>|<\|end_of_solution\|>", "", text)
    text = re.sub(r"#+\s*(?:conclusion|final answer)\s*:?", "Requested result:", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)final answer", "requested result", text)
    text = _redact_answer_text(text, answer)
    text = re.sub(r"\\boxed\s*\{[^{}]*\}", MASK_PLACEHOLDER, text)
    text = re.sub(
        rf"(?i)\bthe answer is\s*{re.escape(MASK_PLACEHOLDER)}",
        f"the requested quantity becomes {MASK_PLACEHOLDER}",
        text,
    )
    text = re.sub(
        rf"(?i)\banswer is\s*{re.escape(MASK_PLACEHOLDER)}",
        f"requested quantity becomes {MASK_PLACEHOLDER}",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()

    if MASK_PLACEHOLDER not in text:
        text = text.rstrip(". ") + f" Therefore the requested quantity becomes {MASK_PLACEHOLDER}."

    if len(text) > max_chars:
        placeholder_idx = text.find(MASK_PLACEHOLDER)
        if placeholder_idx >= 0:
            start = max(0, placeholder_idx - max_chars + 200)
            text = text[start : placeholder_idx + 200]
            if start > 0:
                text = "..." + text
        text = text[:max_chars].rstrip()
        if MASK_PLACEHOLDER not in text:
            text = text.rstrip(". ") + f" The requested quantity becomes {MASK_PLACEHOLDER}."
    return text


def calibrate_masked_derivation(
    raw_text: str,
    reference_solution: str,
    reference_answer: str | None = None,
    min_chars: int = 120,
    max_chars: int = 5000,
) -> MaskedDerivationCalibration:
    parsed = parse_masked_derivation(raw_text)
    if parsed is None:
        return MaskedDerivationCalibration(False, "parse_failed")

    masked = str(parsed.get("masked_derivation", "")).strip()
    omitted = str(parsed.get("omitted_result", "")).strip()
    checks_raw = parsed.get("checks", [])
    checks = [str(item).strip() for item in checks_raw] if isinstance(checks_raw, list) else []
    combined = "\n".join([masked, omitted, "\n".join(checks)])

    if len(masked) < min_chars:
        return MaskedDerivationCalibration(False, "too_short", masked, omitted, checks)
    if len(masked) > max_chars:
        return MaskedDerivationCalibration(False, "too_long", masked, omitted, checks)
    if _contains_prompt_echo(masked):
        return MaskedDerivationCalibration(False, "prompt_echo", masked, omitted, checks)
    if "\\boxed" in combined or _contains_final_answer_wording(masked):
        return MaskedDerivationCalibration(False, "answer_not_omitted", masked, omitted, checks)
    if MASK_PLACEHOLDER not in masked:
        return MaskedDerivationCalibration(False, "missing_omission_marker", masked, omitted, checks)
    if _contains_leakage(combined):
        return MaskedDerivationCalibration(False, "meta_leakage", masked, omitted, checks)

    if _contains_answer_copy(combined, omitted, reference_solution, reference_answer):
        return MaskedDerivationCalibration(False, "answer_copy", masked, omitted, checks)

    return MaskedDerivationCalibration(True, "passed", masked, omitted, checks)
