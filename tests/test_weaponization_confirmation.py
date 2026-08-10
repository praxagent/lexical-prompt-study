from __future__ import annotations

import numpy as np

from lexical_prompt_study.weaponization_analysis import NEGATIVES
from lexical_prompt_study.weaponization_confirmation import (
    _confirmation_metrics,
    apply_logistic_head,
    extend_empirical_percentile,
)


def test_empirical_reference_extension_matches_calibration_ties() -> None:
    reference = [1.0, 1.0, 3.0, 4.0]
    assert extend_empirical_percentile(1.0, reference) == 0.375
    assert extend_empirical_percentile(3.0, reference) == 0.75
    assert extend_empirical_percentile(10.0, reference) == 1.0
    assert extend_empirical_percentile(-1.0, reference) == 0.125


def test_frozen_logistic_head_application() -> None:
    head = {
        "mean": [0.0] * 31,
        "scale": [1.0] * 31,
        "coefficients": [0.0] + [1.0] + [0.0] * 30,
    }
    assert apply_logistic_head([1.0] + [0.0] * 30, head) > 0.5
    assert apply_logistic_head([-1.0] + [0.0] * 30, head) < 0.5


def test_confirmation_gate_requires_zero_false_positives_per_stratum() -> None:
    labels = np.asarray([1] * 300 + [0] * 1800)
    strata = np.asarray(
        ["positive_attack_harmful"] * 300 + [stratum for stratum in NEGATIVES for _ in range(300)],
        dtype=object,
    )
    predictions = np.asarray([True] * 275 + [False] * 25 + [False] * 1800)
    passed = _confirmation_metrics(predictions, labels, strata)
    assert passed["placement_gate_passed"] is True
    predictions[300] = True
    failed = _confirmation_metrics(predictions, labels, strata)
    assert failed["placement_gate_passed"] is False
    assert failed["critical_negatives"][NEGATIVES[0]]["false_positive_count"] == 1
