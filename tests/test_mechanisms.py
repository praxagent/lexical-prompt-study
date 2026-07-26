from __future__ import annotations

import numpy as np
import pytest

from lexical_prompt_study.mechanisms import (
    PRIMARY_POSITION,
    RANDOM_TRANSPORT_SEED,
    deterministic_transport_seed,
    probe_margin,
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


def test_mechanism_receipt_fails_closed() -> None:
    receipt = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-llama33-70b-v1",
        "public_plan_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "split": "discovery",
        "behavior_id": "JBB-H-001",
        "arm": "full",
        "turn": 2,
        "position": PRIMARY_POSITION,
        "position_token_index": None,
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
    }
    validate_mechanism_receipt(receipt)
    receipt["turn"] = 1
    with pytest.raises(ValueError, match="turn 2"):
        validate_mechanism_receipt(receipt)
