"""Pure, prespecified diagnosis of failed whole-response arithmetic answers.

This module reads no evidence and changes no task outcomes. Callers authenticate
the completed-response cohort and supply its expected ordered integer answers.
Wrapper recovery is descriptive, not a replacement success label.
"""

from __future__ import annotations

import re

ASCII_WHITESPACE = " \t\r\n\v\f"
CATEGORIES = (
    "parseable_wrong_values",
    "wrong_count",
    "wrapper_correct_values",
    "wrapper_wrong_values",
    "unresolved",
)
WRONG_COUNT_STYLES = ("bare", "bracketed", "fenced")
_INTEGER = re.compile(r"[+-]?[0-9]+", flags=re.ASCII)
_OPENING_FENCES = ("```", "```csv", "```json")


class EvidenceConsistencyError(ValueError):
    """A supposedly failed response already satisfies the original exact oracle."""


def _integer_list(text: str) -> list[str] | None:
    normalized = []
    for raw in text.split(","):
        field = raw.strip(ASCII_WHITESPACE)
        if _INTEGER.fullmatch(field) is None:
            return None
        negative = field[0] == "-"
        digits = field[1:] if field[0] in "+-" else field
        digits = digits.lstrip("0") or "0"
        normalized.append("-" + digits if negative and digits != "0" else digits)
    return normalized


def _single_wrapper(text: str) -> tuple[str, str] | None:
    whole = text.strip(ASCII_WHITESPACE)
    if whole.startswith("[") and whole.endswith("]"):
        body = whole[1:-1]
        if any(marker in body for marker in ("[", "]", "`")):
            return None
        return body, "bracketed"

    first_lf = whole.find("\n")
    if first_lf < 0:
        return None
    opening = whole[:first_lf]
    if opening.endswith("\r"):
        opening = opening[:-1]
    if opening not in _OPENING_FENCES:
        return None
    remainder = whole[first_lf + 1 :]
    if remainder == "```":
        body = ""  # Empty two-line fence is recognized, but is not a numeric list.
    elif remainder.endswith("\n```"):
        body = remainder[:-4]
        if body.endswith("\r"):
            body = body[:-1]
    else:
        return None
    if any(marker in body for marker in ("[", "]", "`")):
        return None
    return body, "fenced"


def classify_response(text: str, answers: list[int]) -> dict:
    """Return one category and a style only for parseable wrong-count lists.

    Baseline acceptance is ASCII integer CSV with numeric normalization, including
    leading zeros and signed zero. A baseline exact success is a cohort error.
    Only one whole-response bracket or plain/csv/json Markdown fence is removed.
    """
    if type(text) is not str:
        raise ValueError("response_text_type")
    if type(answers) is not list or not answers or any(type(x) is not int for x in answers):
        raise ValueError("expected_integer_answers")
    expected = [str(answer) for answer in answers]
    parsed = _integer_list(text)
    if parsed is not None:
        if len(parsed) != len(expected):
            return {"category": "wrong_count", "wrong_count_style": "bare"}
        if parsed == expected:
            raise EvidenceConsistencyError("baseline_response_already_exact_success")
        return {"category": "parseable_wrong_values", "wrong_count_style": None}

    wrapper = _single_wrapper(text)
    if wrapper is None:
        return {"category": "unresolved", "wrong_count_style": None}
    body, style = wrapper
    parsed = _integer_list(body)
    if parsed is None:
        return {"category": "unresolved", "wrong_count_style": None}
    if len(parsed) != len(expected):
        return {"category": "wrong_count", "wrong_count_style": style}
    category = "wrapper_correct_values" if parsed == expected else "wrapper_wrong_values"
    return {"category": category, "wrong_count_style": None}


def aggregate_diagnoses(diagnoses: list[dict]) -> dict:
    """Aggregate all supplied records without texts, identities or new scores.

    The caller enforces its fixed cohort size (five for A190). Empty input yields
    explicit zero counts; no case is filtered or inferred from another field.
    """
    if type(diagnoses) is not list:
        raise ValueError("diagnoses_list_type")
    categories = dict.fromkeys(CATEGORIES, 0)
    styles = dict.fromkeys(WRONG_COUNT_STYLES, 0)
    for diagnosis in diagnoses:
        if type(diagnosis) is not dict or set(diagnosis) != {"category", "wrong_count_style"}:
            raise ValueError("diagnosis_fields")
        category, style = diagnosis["category"], diagnosis["wrong_count_style"]
        if type(category) is not str or category not in CATEGORIES:
            raise ValueError("diagnosis_category")
        if category == "wrong_count":
            if type(style) is not str or style not in WRONG_COUNT_STYLES:
                raise ValueError("wrong_count_style")
            styles[style] += 1
        elif style is not None:
            raise ValueError("unexpected_wrong_count_style")
        categories[category] += 1
    return {
        "category_counts": categories,
        "wrong_count_style_counts": styles,
        "completed_responses": len(diagnoses),
    }
