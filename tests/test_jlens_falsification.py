from __future__ import annotations

import numpy as np
import pytest

from lexical_prompt_study.jlens_falsification_result import (
    _detector_metrics,
    _factorial_contrast,
    _holm,
)
from lexical_prompt_study.jlens_falsification_runner import _validate_receipt
from lexical_prompt_study.jlens_falsification_topology import _hashed_ngrams


def test_hashed_prompt_features_are_deterministic_and_normalized() -> None:
    first = _hashed_ngrams("structured safe fixture")
    second = _hashed_ngrams("structured safe fixture")
    changed = _hashed_ngrams("different safe fixture")
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.linalg.norm(first) == pytest.approx(1.0)


def _observation() -> dict[str, object]:
    return {
        "trial_id": "trial",
        "variant_id": "attack-mask-0011",
        "attack_block_mask": 3,
        "attack_block_count": 2,
        "request_class": "harmful_request",
        "request_id": "request",
        "request_sha256": "a" * 64,
        "placement": "scaffold_after_request",
        "injected_token_count": 252,
        "prompt_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "generation_required": True,
    }


def _receipt() -> dict[str, object]:
    return {
        **_observation(),
        "feature_6779_magnitude": 0.0,
        "frozen_subspace_score": 0.1,
        "sae_normalized_reconstruction_error": 0.5,
        "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        "frozen_jlens_score": 0.8,
        "frozen_jlens_threshold": 0.7,
        "would_cancel_before_first_token": True,
        "generated_text_sha256": "d" * 64,
    }


def test_falsification_receipt_accepts_consistent_detector_decision() -> None:
    _validate_receipt(_receipt(), _observation())


def test_falsification_receipt_rejects_detector_arithmetic_drift() -> None:
    receipt = _receipt()
    receipt["would_cancel_before_first_token"] = False
    with pytest.raises(ValueError, match="detector arithmetic"):
        _validate_receipt(receipt, _observation())


def test_falsification_receipt_rejects_missing_required_generation() -> None:
    receipt = _receipt()
    receipt["generated_text_sha256"] = None
    with pytest.raises(ValueError, match="generation presence"):
        _validate_receipt(receipt, _observation())


def test_factorial_contrast_recovers_main_and_interaction_effects() -> None:
    by_request = {
        request_id: {
            mask: {
                "value": offset
                + 2.0 * bool(mask & 1)
                - 3.0 * bool(mask & 2)
                + 5.0 * bool(mask & 1) * bool(mask & 2)
            }
            for mask in range(16)
        }
        for request_id, offset in (("a", 0.0), ("b", 10.0))
    }
    assert np.allclose(_factorial_contrast(by_request, (0,), "value"), 4.5)
    assert np.allclose(_factorial_contrast(by_request, (1,), "value"), -0.5)
    assert np.allclose(_factorial_contrast(by_request, (0, 1), "value"), 5.0)


def test_detector_metrics_compare_successful_harm_and_benign_only() -> None:
    rows = [
        {
            "request_class": "harmful_request",
            "harmbench_binary_prediction": True,
            "trip": True,
        },
        {
            "request_class": "harmful_request",
            "harmbench_binary_prediction": False,
            "trip": False,
        },
        {
            "request_class": "ordinary_benign_request",
            "harmbench_binary_prediction": None,
            "trip": False,
        },
    ]
    metrics = _detector_metrics(rows, lambda row: bool(row["trip"]))
    assert metrics["successful_harmful_count"] == 1
    assert metrics["successful_harmful_capture_fraction"] == 1.0
    assert metrics["ordinary_benign_false_trip_fraction"] == 0.0


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = _holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
    assert adjusted["a"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.09)
    assert adjusted["b"] == pytest.approx(0.09)
    assert adjusted["d"] == pytest.approx(0.2)
