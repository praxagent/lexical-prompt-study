from __future__ import annotations

import json

import numpy as np
import pytest

from lexical_prompt_study import incremental_threshold_v2 as v2
from lexical_prompt_study.jlens_incremental_analysis import _folds, _select_threshold


def separating_fixture():
    labels = np.tile([True, False], 10)
    cores = [f"core-{index}" for index in range(10) for _ in range(2)]
    hashes = [f"{index:064x}" for index in range(10) for _ in range(2)]
    return labels.astype(float)[:, None], labels, _folds(cores, hashes, 5)


def test_real_synthetic_nested_logistic_handles_refit_rounding_gap():
    matrix, labels, folds = separating_fixture()
    result = v2.nested_candidate(matrix, labels, ~labels, folds, np.ones(len(labels), bool))
    np.testing.assert_array_equal(result["trips"], labels)
    for selection in result["fold_calibrations"].values():
        assert selection["lower_score"] < selection["threshold"] < selection["upper_score"]
        assert selection["safe_negative_trips"] == 0
        assert selection["positive_trips"] == selection["positive_count"] == 8
    # The legacy selector remains a distinct implementation; no monkeypatch or
    # automatic migration redirects existing callers to the prospective helper.
    assert _select_threshold is not v2.select_threshold


def test_held_out_labels_cannot_change_own_fold_cutoff_or_predictions():
    matrix, labels, folds = separating_fixture()
    available = np.ones(len(labels), bool)
    baseline = v2.nested_candidate(matrix, labels, ~labels, folds, available)
    changed_labels = labels.copy()
    changed_labels[folds == 0] = ~changed_labels[folds == 0]
    changed = v2.nested_candidate(matrix, changed_labels, ~changed_labels, folds, available)
    assert changed["fold_calibrations"]["0"] == baseline["fold_calibrations"]["0"]
    np.testing.assert_array_equal(
        changed["predictions"][folds == 0], baseline["predictions"][folds == 0]
    )
    np.testing.assert_array_equal(changed["trips"][folds == 0], baseline["trips"][folds == 0])


def test_one_ulp_lower_new_positive_remains_above_midgap_cutoff():
    selected = v2.select_threshold(np.array([0.9, 0.1]), np.array([True, False]),
                                   np.array([False, True]))
    assert selected["threshold"] == 0.5
    np.testing.assert_array_equal(
        v2.apply_threshold(np.array([np.nextafter(0.9, 0.0), 0.1]), selected["threshold"]),
        [True, False],
    )
    # This is a new-score semantic change, not historical reproduction.
    legacy = _select_threshold(np.array([0.9, 0.1]), np.array([True, False]),
                               np.array([False, True]))
    assert legacy["threshold"] == 0.9


@pytest.mark.parametrize("safe_count,expected_capture", [(49, 1), (50, 2), (51, 2)])
def test_exact_two_percent_budget_with_a_safe_positive_score_tie(safe_count, expected_capture):
    scores = np.array([0.9, 0.8, 0.8, *([0.1] * (safe_count - 1))])
    labels = np.arange(len(scores)) < 2
    safe = ~labels
    selected = v2.select_threshold(scores, labels, safe)
    trips = v2.apply_threshold(scores, selected["threshold"])
    assert int((trips & labels).sum()) == expected_capture
    assert int((trips & safe).sum()) == expected_capture - 1
    assert int((trips & safe).sum()) * 50 <= safe_count
    assert trips[1] == trips[2]  # Equal scores cannot be split by their labels.


def test_no_trip_is_explicit_and_does_not_trip_unseen_probability_one():
    selected = v2.select_threshold(np.array([0.5, 0.5]), np.array([True, False]),
                                   np.array([False, True]))
    assert selected["threshold"] is None
    assert selected["boundary_kind"] == "no_trip"
    assert not v2.apply_threshold(np.array([0.0, 0.5, 1.0, -np.inf]), None).any()
    assert json.loads(json.dumps(selected, allow_nan=False))["threshold"] is None


def test_adjacent_representable_scores_keep_the_partition_without_epsilon():
    lower = 0.5
    upper = np.nextafter(lower, 1.0)
    scores = np.array([upper, lower])
    selected = v2.select_threshold(scores, np.array([True, False]), np.array([False, True]))
    assert selected["threshold"] == upper
    np.testing.assert_array_equal(v2.apply_threshold(scores, selected["threshold"]), [True, False])


def test_zero_one_and_missing_scores_keep_explicit_coverage():
    scores = np.array([1.0, -np.inf, 0.0, -np.inf])
    labels = np.array([True, True, False, False])
    selected = v2.select_threshold(scores, labels, ~labels)
    assert selected["threshold"] == 0.5
    assert selected["positive_count"] == selected["safe_negative_count"] == 2
    assert selected["available_positive_count"] == selected["available_safe_negative_count"] == 1
    assert selected["successful_harmful_capture_fraction"] == 0.5
    np.testing.assert_array_equal(v2.apply_threshold(scores, selected["threshold"]),
                                  [True, False, False, False])


def test_all_finite_partition_does_not_trip_unavailable_safe_rows():
    scores = np.array([0.0, -np.inf])
    selected = v2.select_threshold(scores, np.array([True, False]), np.array([False, True]))
    assert selected["boundary_kind"] == "all_finite"
    assert selected["threshold"] == 0.0
    assert selected["available_safe_negative_count"] == 0
    np.testing.assert_array_equal(v2.apply_threshold(scores, selected["threshold"]), [True, False])


def test_midgap_can_change_unseen_scores_and_does_not_guarantee_test_fpr():
    selected = v2.select_threshold(np.array([0.9, 0.1]), np.array([True, False]),
                                   np.array([False, True]))
    # A new negative in the old observed gap can trip. No training-budget
    # guarantee is silently claimed for held-out data or distribution shift.
    assert v2.apply_threshold(np.array([0.6]), selected["threshold"])[0]


def test_calibration_partitions_match_legacy_and_are_permutation_invariant():
    rng = np.random.default_rng(20260914)
    for _ in range(100):
        scores = rng.choice(np.array([-np.inf, 0.0, 0.1, 0.5, 0.8, 1.0]), size=102)
        scores[0] = 0.5
        labels = rng.random(len(scores)) < 0.4
        labels[:2] = [True, False]
        safe = ~labels
        old = _select_threshold(scores, labels, safe)
        new = v2.select_threshold(scores, labels, safe)
        expected = np.isfinite(scores) & (scores >= old["threshold"])
        np.testing.assert_array_equal(v2.apply_threshold(scores, new["threshold"]), expected)
        order = rng.permutation(len(scores))
        assert v2.select_threshold(scores[order], labels[order], safe[order]) == new


@pytest.mark.parametrize("scores", [
    [np.nan, 0.5], [np.inf, 0.5], [-0.1, 0.5], [1.1, 0.5], [-np.inf, -np.inf],
    [[0.1, 0.5]], [],
])
def test_invalid_calibration_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        v2.select_threshold(np.asarray(scores), np.array([True, False]), np.array([False, True]))


@pytest.mark.parametrize("labels,safe", [
    ([True, False], [True, True]), ([False, False], [True, True]),
    ([True, False], [False, False]), ([True], [False, True]), ([1, 0], [False, True]),
])
def test_invalid_endpoint_masks_fail_closed(labels, safe):
    with pytest.raises(ValueError):
        v2.select_threshold(np.array([0.9, 0.1]), np.asarray(labels), np.asarray(safe))


@pytest.mark.parametrize("threshold", [np.inf, -np.inf, np.nan, -0.1, 1.1, True])
def test_invalid_application_cutoffs_fail_closed(threshold):
    with pytest.raises(ValueError):
        v2.apply_threshold(np.array([0.5]), threshold)


@pytest.mark.parametrize("folds", [np.array([0, 1]), np.array([0.0, 1.0]), np.array([-1, 0])])
def test_nested_requires_multiple_valid_inner_folds(folds):
    with pytest.raises(ValueError, match="outer folds"):
        v2.nested_candidate(np.array([[0.0], [1.0]]), np.array([False, True]),
                            np.array([True, False]), folds, np.array([True, True]))
