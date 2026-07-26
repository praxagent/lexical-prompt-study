from __future__ import annotations

import numpy as np
import pytest

from lexical_prompt_study.interventions import (
    deterministic_isotropic_direction,
    direction_sha256,
    intervention_delta,
    stable_intervention_trial_id,
    unit_direction,
    validate_intervention_receipt,
)


def _receipt() -> dict:
    return {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-llama33-70b-v1",
        "intervention_plan_sha256": "a" * 64,
        "public_plan_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "run_id": "gate4-fixture",
        "phase": "confirmatory_attack",
        "trial_id": "fixture-trial",
        "attempt": 1,
        "behavior_id": "SAFE-001",
        "category": "safe",
        "split": "confirmatory",
        "arm": "full",
        "turn": 2,
        "condition_id": "primary_positive",
        "direction_kind": "primary",
        "feature_id": 10146,
        "isotropic_seed": None,
        "direction_sha256": "d" * 64,
        "requested_sign": 1,
        "requested_alpha": 0.01,
        "calibration_rho": None,
        "calibration_reference_norm": None,
        "layer": 50,
        "hook": "residual_post",
        "positions": "current_final_token_on_every_decoding_step",
        "prompt_sha256": "e" * 64,
        "prompt_token_ids_sha256": "f" * 64,
        "parent_generation_receipt_sha256": "1" * 64,
        "generated_token_ids": [42],
        "generated_text_sha256": "2" * 64,
        "restricted_text_path": "restricted/fixture.json",
        "restricted_artifact_sha256": "3" * 64,
        "finish_reason": "eos",
        "generated_token_count": 1,
        "refusal_detected": False,
        "truncated": False,
        "evaluator_score": None,
        "evaluator_parse_ok": False,
        "intervention_steps": [
            {
                "generated_token_index": 0,
                "requested_delta_norm": 0.01,
                "realized_delta_norm": 0.01,
                "pre_residual_norm": 1.0,
                "post_residual_norm": 1.00005,
                "requested_realized_relative_error": 0.0,
                "delta_to_pre_residual_norm": 0.01,
                "clipped": False,
            }
        ],
        "model_revision": "model",
        "tokenizer_revision": "tokenizer",
        "sae_sha256": "4" * 64,
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": None,
        "software": {},
        "runtime": {},
        "error": None,
    }


def test_isotropic_direction_is_deterministic_and_normalized() -> None:
    first = deterministic_isotropic_direction(2026072501, 32)
    second = deterministic_isotropic_direction(2026072501, 32)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, rel=1e-6)
    assert direction_sha256(first) == direction_sha256(second)


def test_direction_rejects_zero_and_dimension_drift() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        unit_direction(np.zeros(4))
    with pytest.raises(ValueError, match="dimension mismatch"):
        unit_direction(np.ones(4), expected_dimension=5)


def test_intervention_delta_records_realized_norm_without_clipping() -> None:
    residual = np.array([3.0, 4.0], dtype=np.float32)
    post, diagnostics = intervention_delta(
        residual,
        np.array([1.0, 0.0]),
        sign=-1,
        alpha=0.1,
    )
    assert np.array_equal(post, np.array([2.9, 4.0], dtype=np.float32))
    assert diagnostics["realized_delta_norm"] == pytest.approx(0.1)
    assert diagnostics["delta_to_pre_residual_norm"] == pytest.approx(0.02)
    assert diagnostics["clipped"] is False


def test_intervention_delta_fails_residual_budget() -> None:
    with pytest.raises(ValueError, match="residual norm budget"):
        intervention_delta(
            np.ones(2),
            np.array([1.0, 0.0]),
            sign=1,
            alpha=0.1,
        )


def test_intervention_receipt_validates_strict_topology() -> None:
    assert validate_intervention_receipt(_receipt()).feature_id == 10146
    payload = _receipt()
    payload["intervention_steps"][0]["clipped"] = True
    with pytest.raises(ValueError, match="clipping is forbidden"):
        validate_intervention_receipt(payload)


def test_intervention_receipt_rejects_identity_and_count_drift() -> None:
    payload = _receipt()
    payload["direction_kind"] = "isotropic"
    with pytest.raises(ValueError, match="isotropic intervention identity mismatch"):
        validate_intervention_receipt(payload)
    payload = _receipt()
    payload["generated_token_count"] = 2
    with pytest.raises(ValueError, match="generated token count mismatch"):
        validate_intervention_receipt(payload)


def test_intervention_trial_id_is_stable_and_condition_specific() -> None:
    first = stable_intervention_trial_id("study", "run", "behavior", "positive")
    assert first == stable_intervention_trial_id(
        "study", "run", "behavior", "positive"
    )
    assert first != stable_intervention_trial_id(
        "study", "run", "behavior", "negative"
    )
