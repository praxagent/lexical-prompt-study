from __future__ import annotations

import numpy as np
import pytest

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
