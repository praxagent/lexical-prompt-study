from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lexical_prompt_study.followup_mechanism_analysis import (
    Candidate,
    candidate_scores,
    discover_candidates,
    fit_common_dense_projection,
    rank_calibration_candidates,
    stable_bootstrap_seed,
    standardized_paired_effect,
    validate_state_payload,
    verify_source_probe_plan,
)


PLACEMENTS = ("ep_before_request", "ep_after_request")


def _synthetic_sae() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    sham = {
        placement: np.zeros((4, 5), dtype=np.float64) for placement in PLACEMENTS
    }
    full = {
        "ep_before_request": np.asarray(
            [
                [0, 1, 4, 2, 0],
                [0, 2, 4, 2, 0],
                [0, 1, 4, 2, 0],
                [0, 2, 4, 2, 0],
            ],
            dtype=np.float64,
        ),
        "ep_after_request": np.asarray(
            [
                [0, 1, 3, 2, 0],
                [0, 2, 3, 2, 0],
                [0, 1, 3, 2, 0],
                [0, 2, 3, 2, 0],
            ],
            dtype=np.float64,
        ),
    }
    # Feature 3 has a positive but constant delta. Its RMS is non-zero, so it
    # remains eligible. Features 0 and 4 are true zero-RMS coordinates.
    norms = np.ones(5, dtype=np.float64)
    return full, sham, norms


def test_discovery_candidates_use_common_maximin_ranking_and_worst_order_scale() -> None:
    full, sham, norms = _synthetic_sae()
    rows, single, subspace = discover_candidates(
        full_by_placement=full,
        sham_by_placement=sham,
        decoder_norms=norms,
        maximum_subspace_features=3,
    )

    assert single == Candidate("single_feature", (2,), (1.0,))
    assert subspace.feature_ids == (1, 2, 3)
    assert np.linalg.norm(subspace.weights) == pytest.approx(1.0)
    assert rows[0]["feature_id"] == 2
    assert {row["feature_id"] for row in rows} == {1, 2, 3}
    assert 0 not in {row["feature_id"] for row in rows}
    assert 4 not in {row["feature_id"] for row in rows}

    # Feature 2's worst-order RMS is larger than feature 1's, so inverse-scale
    # weighting must give feature 2 less weight.
    weight_by_id = dict(zip(subspace.feature_ids, subspace.weights, strict=True))
    assert weight_by_id[2] < weight_by_id[1]


def test_candidate_scoring_and_zero_rms_ineligibility() -> None:
    activations = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    candidate = Candidate("linear_subspace", (0, 2), (0.5, 0.5))
    assert candidate_scores(activations, candidate).tolist() == [2.0, 5.0]
    assert standardized_paired_effect(np.zeros(4)) == (0.0, 0.0, 0.0)

    selected, rows = rank_calibration_candidates(
        {
            candidate.candidate_id: {
                "candidate": candidate,
                "ep_before_request": np.zeros(4),
                "ep_after_request": np.ones(4),
            }
        }
    )
    assert selected is None
    assert rows[0]["eligible"] is False


def test_calibration_tie_prefers_single_before_subspace() -> None:
    single = Candidate("single_feature", (7,), (1.0,))
    subspace = Candidate("linear_subspace", (7,), (1.0,))
    payload = {}
    for candidate in (subspace, single):
        payload[candidate.candidate_id] = {
            "candidate": candidate,
            "ep_before_request": np.ones(4),
            "ep_after_request": np.ones(4),
        }
    selected, rows = rank_calibration_candidates(payload)
    assert selected == single
    assert rows[0]["kind"] == "single_feature"


def test_dense_projection_gives_orderings_equal_weight() -> None:
    full = {
        "ep_before_request": np.asarray([[2.0, 0.0], [2.0, 0.0]]),
        "ep_after_request": np.asarray([[0.0, 3.0], [0.0, 3.0]]),
    }
    sham = {
        placement: np.zeros((2, 2), dtype=np.float64) for placement in PLACEMENTS
    }
    direction = fit_common_dense_projection(full, sham)
    assert direction == pytest.approx(np.asarray([2**-0.5, 2**-0.5]))


def test_bootstrap_seed_is_stable_and_stratum_specific() -> None:
    before = stable_bootstrap_seed(
        base_seed=20260801,
        partition="discovery",
        placement="ep_before_request",
        layer=3,
        transport="jacobian_lens",
        statistic="full_minus_structural_sham",
    )
    assert before == stable_bootstrap_seed(
        base_seed=20260801,
        partition="discovery",
        placement="ep_before_request",
        layer=3,
        transport="jacobian_lens",
        statistic="full_minus_structural_sham",
    )
    after = stable_bootstrap_seed(
        base_seed=20260801,
        partition="discovery",
        placement="ep_after_request",
        layer=3,
        transport="jacobian_lens",
        statistic="full_minus_structural_sham",
    )
    assert before != after


def test_state_payload_provenance_fails_closed() -> None:
    receipt = SimpleNamespace(
        trial_id="safe-trial",
        plan_sha256="1" * 64,
        private_plan_sha256="2" * 64,
        source_commit="3" * 40,
        run_id="safe-run",
        model_revision="4" * 40,
        prompt_token_ids_sha256="5" * 64,
        lens_sha256="6" * 64,
        sae_sha256="7" * 64,
    )
    provenance = {
        "trial_id": receipt.trial_id,
        "public_plan_sha256": receipt.plan_sha256,
        "private_plan_sha256": receipt.private_plan_sha256,
        "source_commit": receipt.source_commit,
        "run_id": receipt.run_id,
        "model_revision": receipt.model_revision,
        "prompt_token_ids_sha256": receipt.prompt_token_ids_sha256,
        "lens_sha256": receipt.lens_sha256,
        "sae_sha256": receipt.sae_sha256,
    }
    payload = {
        "provenance": provenance,
        "states": {
            layer: torch.zeros(4096, dtype=torch.bfloat16)
            for layer in range(31)
        },
    }
    states, layers = validate_state_payload(
        state_payload=payload,
        receipt=receipt,
        expected_layers=tuple(range(31)),
    )
    assert len(states) == 31
    assert layers == tuple(range(31))

    payload["provenance"] = provenance | {"run_id": "drift"}
    with pytest.raises(ValueError, match="provenance"):
        validate_state_payload(
            state_payload=payload,
            receipt=receipt,
            expected_layers=tuple(range(31)),
        )


def test_source_probe_plan_is_hash_and_row_bound(tmp_path) -> None:
    source = {
        "probes": {
            "primary_refusal": [
                {"token_id": 1, "text_sha256": "a" * 64, "seen_in_prompt": False}
            ],
            "primary_compliance": [
                {"token_id": 2, "text_sha256": "b" * 64, "seen_in_prompt": False}
            ],
        }
    }
    path = tmp_path / "source-plan.json"
    path.write_text(json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n")
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    mechanism_probe = {
        "source_plan_sha256": source_sha,
        "refusal": [{"token_id": 1, "text_sha256": "a" * 64}],
        "compliance": [{"token_id": 2, "text_sha256": "b" * 64}],
    }
    assert (
        verify_source_probe_plan(
            source_probe_plan_path=path,
            mechanism_probe=mechanism_probe,
        )
        == source_sha
    )

    mechanism_probe["refusal"][0]["token_id"] = 3
    with pytest.raises(ValueError, match="refusal"):
        verify_source_probe_plan(
            source_probe_plan_path=path,
            mechanism_probe=mechanism_probe,
        )
