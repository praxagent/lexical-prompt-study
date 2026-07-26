from __future__ import annotations

import numpy as np
import pytest

from lexical_prompt_study.mechanisms import (
    PRIMARY_POSITION,
    RANDOM_TRANSPORT_SEED,
    deterministic_transport_seed,
    probe_margin,
    probe_margin_from_embedding_moments,
    sae_feature_diagnostics,
    select_sae_candidates,
    validate_mechanism_receipt,
    validate_transport_metadata,
)


def test_probe_margin_uses_vocabulary_z_scores() -> None:
    logits = np.array([0.0, 1.0, 2.0, 3.0])
    result = probe_margin(logits, [3], [0])
    assert result["vocabulary_logit_mean"] == 1.5
    assert result["refusal_minus_compliance_margin"] == pytest.approx(
        3 / np.std(logits)
    )


def test_probe_margin_rejects_overlapping_probes() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        probe_margin([0.0, 1.0, 2.0], [1], [1, 2])


def test_embedding_moment_shortcut_matches_materialized_vocabulary() -> None:
    rng = np.random.default_rng(20260725)
    hidden = rng.normal(size=7)
    unembedding = rng.normal(size=(31, 7))
    epsilon = 1e-5
    normalized = hidden / np.sqrt(np.mean(np.square(hidden)) + epsilon)
    logits = unembedding @ normalized
    direct = probe_margin(logits, [1, 4, 9], [3, 8])
    shortcut = probe_margin_from_embedding_moments(
        hidden,
        unembedding,
        [1, 4, 9],
        [3, 8],
        rms_norm_epsilon=epsilon,
    )
    assert shortcut == pytest.approx(direct, abs=1e-12)


def test_random_transport_seed_is_stable_and_layer_specific() -> None:
    first = deterministic_transport_seed(RANDOM_TRANSPORT_SEED, 12)
    assert first == deterministic_transport_seed(RANDOM_TRANSPORT_SEED, 12)
    assert first != deterministic_transport_seed(RANDOM_TRANSPORT_SEED, 13)


def test_random_transport_must_be_norm_matched() -> None:
    with pytest.raises(ValueError, match="norm mismatch"):
        validate_transport_metadata(
            transport="random_gaussian",
            layer=12,
            fitted_frobenius_norm=10.0,
            realized_frobenius_norm=11.0,
            random_seed=1,
        )


def test_sae_selection_is_discovery_paired_and_deterministic() -> None:
    full = np.array(
        [
            [3.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )
    sham = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ]
    )
    diagnostics = sae_feature_diagnostics(full, sham, [1.0, 2.0, 3.0])
    selected = select_sae_candidates(diagnostics)
    assert [item.feature_id for item in selected] == [0, 1]


def test_zero_decoder_norm_feature_is_retained_but_not_selectable() -> None:
    full = np.array([[2.0, 2.0], [3.0, 3.0]])
    sham = np.array([[0.0, 0.0], [1.0, 1.0]])
    diagnostics = sae_feature_diagnostics(full, sham, [0.0, 1.0])
    assert diagnostics[0].decoder_norm == 0.0
    assert [item.feature_id for item in select_sae_candidates(diagnostics)] == [1]


def test_mechanism_receipt_fails_closed() -> None:
    receipt = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-llama33-70b-v1",
        "public_plan_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "run_id": "gate3-discovery-v1",
        "split": "discovery",
        "observation_id": "safe-observation",
        "behavior_id": "JBB-H-001",
        "category": "safe",
        "arm": "full",
        "turn": 2,
        "position": PRIMARY_POSITION,
        "position_token_index": None,
        "position_available": True,
        "missing_position_reason": None,
        "prompt_token_ids_sha256": "e" * 64,
        "prefix_token_ids_sha256": "f" * 64,
        "transport": "identity",
        "layer": 1,
        "refusal_probe_token_ids": [1],
        "compliance_probe_token_ids": [2],
        "margin": {
            "vocabulary_logit_mean": 0.0,
            "vocabulary_logit_std": 1.0,
            "refusal_probe_mean_z": 1.0,
            "compliance_probe_mean_z": -1.0,
            "refusal_minus_compliance_margin": 2.0,
        },
        "model_revision": "m",
        "tokenizer_revision": "t",
        "lens_sha256": "c" * 64,
        "sae_sha256": "d" * 64,
        "runtime": {},
    }
    validate_mechanism_receipt(receipt)
    receipt["turn"] = 1
    with pytest.raises(ValueError, match="turn"):
        validate_mechanism_receipt(receipt)
