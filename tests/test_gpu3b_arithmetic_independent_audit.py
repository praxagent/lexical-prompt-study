"""Invented controls for the independent endpoint and fixed-cohort audit."""

import pytest

from lexical_prompt_study import gpu3b_arithmetic_independent_audit as audit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" +0003 , -0 ", True),
        ("3, 0", True),
        ("3,0,", False),
        ("3,0 extra", False),
        ("3,\u00a00", False),
        ("\u0663,0", False),
        ("3.0,0", False),
        ("3;0", False),
    ],
)
def test_exact_ascii_whole_content(text, expected):
    assert audit.whole_content(text, [3, 0]) is expected


@pytest.mark.parametrize(("text", "expected"), [("-000", "0"), ("--3", None)])
def test_sign_normalization(text, expected):
    assert audit.integer_text(text) == expected


@pytest.mark.parametrize(
    ("answers", "state", "tokens", "text", "prefix", "key", "expected"),
    [
        ([3, 0], "completed", [1, 1, 1, 99], "3,0", None, "label", 1),
        ([3, 0], "completed", [1] * 64, "3,0", "3,0", "label", 0),
        ([3, 0], "infrastructure_failed", [1, 99], None, None, "label", None),
        ([3, 0], "completed", [1, 1, 1, 99], "3,0", None, "prefix_reached", False),
        ([3, 0], "completed", [1, 1, 1, 1, 99], "3,0", "3,0", "prefix_reached", True),
        ([3, 0], "infrastructure_failed", [1] * 3, None, None, "prefix_reached", None),
        ([3, 0], "infrastructure_failed", [1] * 4, None, "3", "completed_fields", 0),
        ([3, 0], "infrastructure_failed", [1] * 4, None, "3,", "completed_fields", 1),
        ([3, 0], "infrastructure_failed", [1] * 4, None, "3,", "known_error", False),
        ([3, 0], "infrastructure_failed", [1] * 4, None, "7,", "known_error", True),
        ([3, 0], "infrastructure_failed", [1] * 4, None, "3,0,", "known_error", True),
        (
            [3, 0, 4, 5],
            "infrastructure_failed",
            [1] * 4,
            None,
            "3,0,",
            "unflagged_remaining_ge2",
            True,
        ),
    ],
)
def test_endpoint_and_prefix_boundaries(answers, state, tokens, text, prefix, key, expected):
    actual = audit.rebuild_score(answers, state, tokens, text, prefix, [99])[key]
    assert audit.canonical(actual) == audit.canonical(expected)


def test_unattempted_is_unknown_in_every_diagnostic():
    assert audit.rebuild_score([3, 0], "unattempted", [], None, None, [99]) == {
        key: "missing" if key == "outcome_status" else None for key in audit.FIELDS
    }


@pytest.mark.parametrize("tokens", [[1] * 63, [1] * 65, [99, 1, 99], [True, 99]])
def test_invalid_completed_tokens_rejected(tokens):
    with pytest.raises(AssertionError):
        audit.rebuild_score([3, 0], "completed", tokens, "3,0", None, [99])


def rows():
    return [
        {"core_index": core, "item_count": count, "magnitude": magnitude}
        for core in range(8)
        for count in (2, 4)
        for magnitude in ("one_digit", "two_digit")
    ]


def test_all_unknown_contrast_keeps_sixteen_pairs():
    assert audit.contrast(rows(), [None] * 32, "item_count", 2, 4) == {
        "point": None,
        "lower": -1.0,
        "upper": 1.0,
        "planned_outcomes": 32,
        "resolved_outcomes": 0,
        "planned_pairs": 16,
        "resolved_pairs": 0,
    }


@pytest.mark.parametrize("polarity", [1, -1])
def test_complete_directional_contrast(polarity):
    labels = [int(row["item_count"] == (2 if polarity == 1 else 4)) for row in rows()]
    assert audit.contrast(rows(), labels, "item_count", 2, 4)["point"] == polarity


def test_known_zero_contrast_is_not_missing():
    assert audit.contrast(rows(), [0] * 32, "magnitude", "one_digit", "two_digit")["point"] == 0


def test_one_missing_label_preserves_sharp_bound():
    labels = [int(row["item_count"] == 2) for row in rows()]
    labels[0] = None
    result = audit.contrast(rows(), labels, "item_count", 2, 4)
    assert {
        key: result[key]
        for key in ("point", "lower", "upper", "resolved_outcomes", "resolved_pairs")
    } == {
        "point": None,
        "lower": 15 / 16,
        "upper": 1.0,
        "resolved_outcomes": 31,
        "resolved_pairs": 15,
    }


def test_boolean_is_not_a_binary_label():
    with pytest.raises(AssertionError):
        audit.contrast(rows(), [True] * 32, "item_count", 2, 4)
    with pytest.raises(AssertionError):
        audit.rate([False])


def test_rate_bound_uses_all_planned_labels():
    assert audit.rate([1, 0, None, None]) == {
        "point": None,
        "lower": 0.25,
        "upper": 0.75,
        "planned_outcomes": 4,
        "resolved_outcomes": 2,
    }
