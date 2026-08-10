from __future__ import annotations

import numpy as np

from lexical_prompt_study.weaponization_analysis import (
    NEGATIVES,
    empirical_midranks,
    exact_binomial_lower,
    exact_binomial_upper,
    fit_ridge_logistic,
    select_one_key_threshold,
    select_two_key_thresholds,
)


def _synthetic_gate_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([1] * 100 + [0] * 600)
    strata = np.asarray(
        ["positive_attack_harmful"] * 100
        + [stratum for stratum in NEGATIVES for _ in range(100)],
        dtype=object,
    )
    structure = np.asarray([1.0] * 95 + [0.0] * 5 + [0.0] * 600)
    weaponization = np.asarray([1.0] * 92 + [0.0] * 8 + [0.0] * 600)
    return structure, weaponization, labels, strata


def test_empirical_midranks_average_ties() -> None:
    observed = empirical_midranks([1.0, 1.0, 3.0, 4.0])
    assert np.allclose(observed, [0.375, 0.375, 0.75, 1.0])


def test_ridge_logistic_is_deterministic_and_has_expected_shape() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    first = fit_ridge_logistic(features, labels)
    second = fit_ridge_logistic(features, labels)
    assert np.array_equal(first["coefficients"], second["coefficients"])
    assert first["coefficients"].shape == (2,)
    assert first["probability"][0] < first["probability"][-1]


def test_two_key_threshold_gate_selects_eligible_pair() -> None:
    structure, weaponization, labels, strata = _synthetic_gate_rows()
    result = select_two_key_thresholds(structure, weaponization, labels, strata)
    assert result["eligible"] is True
    assert result["metrics"]["recall"] == 0.92
    assert all(
        row["false_positive_count"] == 0
        for row in result["metrics"]["critical_negatives"].values()
    )


def test_one_key_threshold_stops_when_one_negative_stratum_matches_positive() -> None:
    structure, _, labels, strata = _synthetic_gate_rows()
    structure[100:200] = 1.0
    result = select_one_key_threshold(structure, labels, strata)
    assert result["eligible"] is False


def test_exact_binomial_bounds_cover_observed_proportions() -> None:
    assert 0.0 < exact_binomial_lower(90, 100) < 0.9
    assert 0.02 < exact_binomial_upper(2, 100) < 0.1
    assert exact_binomial_lower(0, 100) == 0.0
    assert exact_binomial_upper(100, 100) == 1.0
