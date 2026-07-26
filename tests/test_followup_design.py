import pytest

from lexical_prompt_study.followup_design import (
    detector_fixed_denominator_counts,
    holm_rejections,
    select_common_threshold,
    select_maximin_candidate,
    validate_placement_render_pair,
)


def _render(placement: str) -> dict:
    offsets = (
        {"scaffold": [2, 5], "request": [7, 11]}
        if placement == "ep_before_request"
        else {"request": [2, 6], "scaffold": [8, 11]}
    )
    return {
        "placement": placement,
        "template_sha256": "1" * 64,
        "tokenizer_revision": "pinned",
        "chat_template_sha256": "2" * 64,
        "component_bytes_sha256": {"request": "3" * 64, "scaffold": "4" * 64},
        "component_token_sha256": {"request": "5" * 64, "scaffold": "6" * 64},
        "component_token_counts": {"request": 4, "scaffold": 3},
        "delimiter_special_tokens_sha256": "7" * 64,
        "assistant_boundary_suffix_sha256": "8" * 64,
        "component_offsets": offsets,
        "offset_map_sha256": ("9" if placement.startswith("ep_before") else "a") * 64,
        "total_tokens": 14,
        "context_ceiling": 4096,
        "generation_budget": 256,
        "truncated": False,
        "padding_applied": False,
        "context_shifted": False,
    }


def test_placement_render_pair_accepts_only_block_permutation() -> None:
    public = validate_placement_render_pair(
        _render("ep_before_request"), _render("ep_after_request")
    )
    assert public["status"] == "passed"
    assert "component_offsets" not in public


def test_placement_render_pair_rejects_boundary_tokenization_drift() -> None:
    after = _render("ep_after_request")
    after["component_token_sha256"]["request"] = "b" * 64
    with pytest.raises(ValueError, match="component_token_sha256"):
        validate_placement_render_pair(_render("ep_before_request"), after)


def test_maximin_selection_uses_worst_order_and_deterministic_ties() -> None:
    candidates = [
        {
            "candidate_id": "subspace",
            "candidate_type": "subspace",
            "feature_ids": [1, 2],
            "eligible": True,
            "paired_differences": {
                "ep_before_request": [2.0, 2.0],
                "ep_after_request": [0.1, 1.0],
            },
        },
        {
            "candidate_id": "feature",
            "candidate_type": "single_feature",
            "feature_ids": [9],
            "eligible": True,
            "paired_differences": {
                "ep_before_request": [1.0, 1.0],
                "ep_after_request": [1.0, 1.0],
            },
        },
    ]
    assert select_maximin_candidate(candidates) == "feature"


def test_common_threshold_uses_shared_rows_once_and_frozen_tie_rule() -> None:
    threshold = select_common_threshold(
        {
            "full:ep_before_request": [0.9, 0.8],
            "full:ep_after_request": [0.85, 0.75],
            "base": [0.1, 0.2],
            "ordinary_benign": [0.05, 0.15],
        },
        {"full:ep_before_request", "full:ep_after_request"},
        0.0,
        0.8,
    )
    assert threshold == 0.75


def test_common_threshold_stops_when_recall_and_fpr_cannot_both_pass() -> None:
    threshold = select_common_threshold(
        {
            "full:ep_before_request": [0.6, 0.4],
            "full:ep_after_request": [0.55, 0.45],
            "base": [0.7, 0.3],
        },
        {"full:ep_before_request", "full:ep_after_request"},
        0.0,
        0.8,
    )
    assert threshold is None


def test_holm_stops_after_first_nonrejection() -> None:
    assert holm_rejections({"before": 0.01, "after": 0.04}) == {
        "before": True,
        "after": True,
    }
    assert holm_rejections({"before": 0.03, "after": 0.031}) == {
        "before": False,
        "after": False,
    }


def test_detector_missingness_is_adversarial_and_fixed_denominator() -> None:
    positive = detector_fixed_denominator_counts(
        [True] * 35 + [False, None, None, None, None], positive_stratum=True
    )
    assert positive["recall"] == 35 / 40
    assert positive["invalid"] is False
    negative = detector_fixed_denominator_counts(
        [False] * 35 + [True, None, None, None, None], positive_stratum=False
    )
    assert negative["false_positives"] == 5
    assert negative["invalid"] is False
