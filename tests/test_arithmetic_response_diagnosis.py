"""Invented whole-response cases only; no native evidence or runtime imports."""

import pytest

from lexical_prompt_study.arithmetic_response_diagnosis import (
    EvidenceConsistencyError,
    aggregate_diagnoses,
    classify_response,
)


@pytest.mark.parametrize("text", ["3,-4", "\t+0003 , -0004\r\n", "\v003\f, -004 "])
def test_baseline_success_is_an_evidence_consistency_error(text):
    with pytest.raises(EvidenceConsistencyError, match="already_exact_success"):
        classify_response(text, [3, -4])


def test_signed_zero_and_long_leading_zeros_keep_baseline_acceptance():
    with pytest.raises(EvidenceConsistencyError):
        classify_response("-000,+000", [0, 0])
    with pytest.raises(EvidenceConsistencyError):
        classify_response("0" * 6000 + "3,-4", [3, -4])
    assert classify_response("9" * 6000 + ",-4", [3, -4])["category"] == "parseable_wrong_values"


@pytest.mark.parametrize(
    ("text", "category", "style"),
    [
        ("4,-4", "parseable_wrong_values", None),
        ("-4,3", "parseable_wrong_values", None),
        ("3", "wrong_count", "bare"),
        ("3,-4,0", "wrong_count", "bare"),
        ("[3]", "wrong_count", "bracketed"),
        ("```csv\n3\n```", "wrong_count", "fenced"),
        ("[ +003 , -0004 ]", "wrapper_correct_values", None),
        ("[3,4]", "wrapper_wrong_values", None),
        ("```\n3,-4\n```", "wrapper_correct_values", None),
        ("```csv\r\n3,-4\r\n```", "wrapper_correct_values", None),
        (" \t```json\n+03,-04\n```\r\n", "wrapper_correct_values", None),
        ("```json\r\n3,-4\n```", "wrapper_correct_values", None),
        ("```json\n3,4\n```", "wrapper_wrong_values", None),
        ("```\n```", "unresolved", None),
        ("```csv\r\n```", "unresolved", None),
        ("```\n\n```", "unresolved", None),
        ("[]", "unresolved", None),
        ("", "unresolved", None),
    ],
)
def test_exclusive_categories_and_whole_wrappers(text, category, style):
    assert classify_response(text, [3, -4]) == {"category": category, "wrong_count_style": style}


@pytest.mark.parametrize(
    "text",
    [
        "3,,",
        "3,-4,",
        "3.0,-4",
        "3e0,-4",
        "３,-4",
        "3,−4",
        "3,-4\u00a0",
        "Answers: 3,-4",
        "3,-4 done",
        "3;-4",
        "(3,-4)",
        "{3,-4}",
        '"3,-4"',
        "[[3,-4]]",
        "```json\n[3,-4]\n```",
        "[```\n3,-4\n```]",
        "```text\n3,-4\n```",
        "```CSV\n3,-4\n```",
        "```csv \n3,-4\n```",
        "``` 3,-4 ```",
        "```\r3,-4\r```",
        "````\n3,-4\n````",
        "~~~csv\n3,-4\n~~~",
        "before\n```\n3,-4\n```",
        "```\n3,-4\n```\nafter",
        "```\n3,-4\n ```",
        "```\n3,-4```",
        "```\n3,-4\n```\n```\n3,-4\n```",
    ],
)
def test_no_prose_search_nested_removal_or_alternate_grammar(text):
    assert classify_response(text, [3, -4]) == {"category": "unresolved", "wrong_count_style": None}


def test_wrong_count_precedes_value_judgment():
    for text, style in (("999", "bare"), ("[999]", "bracketed"), ("```\n999\n```", "fenced")):
        assert classify_response(text, [3, -4]) == {
            "category": "wrong_count",
            "wrong_count_style": style,
        }


@pytest.mark.parametrize(
    ("text", "answers"),
    [(None, [1]), (b"1", [1]), ("1", (1,)), ("1", []), ("1", [True]), ("1", [1.0]), ("1", ["1"])],
)
def test_invalid_api_inputs_raise_value_error(text, answers):
    with pytest.raises(ValueError):
        classify_response(text, answers)


def test_all_categories_retained_and_only_aggregate_keys_emitted():
    inputs = ["4,-4", "[3]", "[3,-4]", "```\n3,4\n```", "Answer: 3,-4"]
    diagnoses = [classify_response(text, [3, -4]) for text in inputs]
    aggregate = aggregate_diagnoses(diagnoses)
    assert aggregate == {
        "category_counts": {
            "parseable_wrong_values": 1,
            "wrong_count": 1,
            "wrapper_correct_values": 1,
            "wrapper_wrong_values": 1,
            "unresolved": 1,
        },
        "wrong_count_style_counts": {"bare": 0, "bracketed": 1, "fenced": 0},
        "completed_responses": 5,
    }
    diagnoses[0]["category"] = "unresolved"
    assert aggregate["category_counts"]["parseable_wrong_values"] == 1


def test_aggregator_is_not_a_cohort_selector():
    empty = aggregate_diagnoses([])
    assert empty["completed_responses"] == 0
    assert not any(empty["category_counts"].values())
    cases = [classify_response(text, [3, -4]) for text in ("1", "[1]", "```\n1\n```", "?")]
    output = aggregate_diagnoses(cases)
    assert output["completed_responses"] == 4
    assert output["wrong_count_style_counts"] == {"bare": 1, "bracketed": 1, "fenced": 1}
    assert sum(output["category_counts"].values()) == 4


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"category": "unresolved"},
        {"category": "unresolved", "wrong_count_style": None, "text": "private"},
        {"category": "success", "wrong_count_style": None},
        {"category": True, "wrong_count_style": None},
        {"category": "wrong_count", "wrong_count_style": None},
        {"category": "wrong_count", "wrong_count_style": True},
        {"category": "wrong_count", "wrong_count_style": "other"},
        {"category": "unresolved", "wrong_count_style": "bare"},
    ],
)
def test_aggregate_rejects_bad_record_or_leaked_payload(record):
    with pytest.raises(ValueError):
        aggregate_diagnoses([record])


def test_aggregate_requires_list():
    with pytest.raises(ValueError):
        aggregate_diagnoses(())
