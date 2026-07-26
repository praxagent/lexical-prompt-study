from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.artifacts import EVALUATOR_REVISION
from lexical_prompt_study.followup_behavior_analysis import (
    _holm_adjusted,
    _sign_flip_p_value,
    analyze_followup_behavior_calibration,
    analyze_followup_behavior_discovery,
)
from lexical_prompt_study.hashing import sha256_file


PLAN_PATH = Path(__file__).parents[1] / "plans" / "followup_v2.public.json"


def _generation_receipt(
    *,
    trial_id: str,
    behavior_id: str,
    arm: str,
    placement: str | None,
    partition: str = "discovery",
) -> dict:
    shared = arm == "base"
    if partition == "calibration":
        plan_sha256 = (
            "5243beefaf643d69a06a3049661415edece8d88e2be168dd12b5a2bbf7c33003"
        )
        private_plan_sha256 = (
            "b191e600e83408b53c913841dfeac2fd3cc8491117c511b0cc0351e66039e8cc"
        )
        source_commit = "6ce5433cbaa8aaa49cc672fba9230cb37023b95e"
        run_id = "g2-calibration-a029-20260726"
    else:
        plan_sha256 = (
            "c16a227bbc641ff16d202b4dca5ee5670682e98e1dad7a53868b2ad0901cbdff"
        )
        private_plan_sha256 = (
            "3f96a1860bf47f5543f96002585c1f8afc00bca53ea88216c03b2a6598d2a128"
        )
        source_commit = "e126e5e7cd887d01a303d48d750e42a2ebcf37a8"
        run_id = "g2-discovery-a025-20260726"
    return {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-followup-v2",
        "plan_sha256": plan_sha256,
        "private_plan_sha256": private_plan_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "trial_id": trial_id,
        "partition": partition,
        "behavior_id": behavior_id,
        "category": "safe",
        "arm": arm,
        "placement": placement,
        "stratum": arm if shared else f"{arm}:{placement}",
        "shared_reference": shared,
        "request_sha256": "4" * 64,
        "prompt_sha256": "5" * 64,
        "prompt_token_ids_sha256": "6" * 64,
        "render_validation_sha256": None if shared else "7" * 64,
        "generated_text_sha256": "8" * 64,
        "generated_token_ids_sha256": "9" * 64,
        "generated_token_count": 1,
        "finish_reason": "eos",
        "truncated": False,
        "refusal_detected": arm != "full",
        "divider_detected": False,
        "post_divider_present": False,
        "restricted_artifact_path": "/private/restricted.json",
        "restricted_artifact_sha256": "a" * 64,
        "state_bundle_path": "/private/state.pt",
        "state_bundle_sha256": "b" * 64,
        "capture_layers": [19],
        "state_shape": [1, 4096],
        "state_dtype": "torch.bfloat16",
        "generation_elapsed_seconds": 0.1,
        "capture_elapsed_seconds": 0.1,
        "peak_memory_bytes": None,
        "model_revision": "c" * 40,
        "tokenizer_revision": "c" * 40,
        "lens_sha256": "d" * 64,
        "sae_sha256": "e" * 64,
        "software": {},
    }


def test_followup_discovery_analysis_keeps_orderings_separate(tmp_path: Path) -> None:
    generation_trials = tmp_path / "generation" / "receipts" / "trials"
    score_trials = tmp_path / "scores" / "trials"
    generation_trials.mkdir(parents=True)
    score_trials.mkdir(parents=True)
    placements = ("ep_before_request", "ep_after_request")
    for behavior_index in range(20):
        behavior_id = f"SAFE-{behavior_index:02d}"
        conditions = [("base", None)]
        conditions.extend(
            (arm, placement)
            for arm in ("inert_length", "structural_sham", "full")
            for placement in placements
        )
        for arm, placement in conditions:
            trial_id = f"{behavior_id}-{arm}-{placement or 'shared'}"
            generation_path = generation_trials / f"{trial_id}.json"
            generation_path.write_text(
                json.dumps(
                    _generation_receipt(
                        trial_id=trial_id,
                        behavior_id=behavior_id,
                        arm=arm,
                        placement=placement,
                    )
                )
            )
            score = 0.8 if arm == "full" else 0.1
            (score_trials / f"{trial_id}.json").write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "generation_receipt_sha256": sha256_file(generation_path),
                        "evaluator_revision": EVALUATOR_REVISION,
                        "yes_probability": score,
                        "no_probability": 1.0 - score,
                        "binary_prediction": score >= 0.5,
                        "parse_ok": True,
                    }
                )
            )

    result = analyze_followup_behavior_discovery(
        public_plan_path=PLAN_PATH,
        generation_root=tmp_path / "generation",
        score_root=tmp_path / "scores",
        output_path=tmp_path / "result.json",
    )

    assert result["n_generation_receipts"] == 140
    assert result["n_score_receipts"] == 140
    assert result["pooled_estimate_reported"] is False
    assert set(result["ordering_results"]) == set(placements)
    assert result["both_orderings_continuation_gate_passed"] is True
    for placement in placements:
        contrast = result["ordering_results"][placement][
            "full_minus_structural_sham"
        ]
        assert contrast["estimate"] == pytest.approx(0.7)
        assert contrast["continuation_gate_passed"] is True


def test_followup_discovery_analysis_fails_closed_on_missing_score(
    tmp_path: Path,
) -> None:
    (tmp_path / "generation" / "receipts" / "trials").mkdir(parents=True)
    (tmp_path / "scores" / "trials").mkdir(parents=True)

    try:
        analyze_followup_behavior_discovery(
            public_plan_path=PLAN_PATH,
            generation_root=tmp_path / "generation",
            score_root=tmp_path / "scores",
            output_path=tmp_path / "result.json",
        )
    except ValueError as error:
        assert "requires 140 generations and scores" in str(error)
    else:
        raise AssertionError("missing topology was accepted")


def test_sign_flip_and_holm_rules_are_deterministic() -> None:
    import numpy as np

    values = np.full(20, 0.7)
    assert _sign_flip_p_value(values, 65536, 20260729) < 0.001
    adjusted = _holm_adjusted(
        {"ep_before_request": 0.01, "ep_after_request": 0.04}
    )
    assert adjusted == {
        "ep_before_request": 0.02,
        "ep_after_request": 0.04,
    }


def test_followup_calibration_analysis_applies_holm_without_pooling(
    tmp_path: Path,
) -> None:
    generation_trials = tmp_path / "generation" / "receipts" / "trials"
    score_trials = tmp_path / "scores" / "trials"
    generation_trials.mkdir(parents=True)
    score_trials.mkdir(parents=True)
    placements = ("ep_before_request", "ep_after_request")
    for behavior_index in range(20):
        behavior_id = f"SAFE-C-{behavior_index:02d}"
        conditions = [("base", None)]
        conditions.extend(
            (arm, placement)
            for arm in ("inert_length", "structural_sham", "full")
            for placement in placements
        )
        for arm, placement in conditions:
            trial_id = f"{behavior_id}-{arm}-{placement or 'shared'}"
            generation_path = generation_trials / f"{trial_id}.json"
            generation_path.write_text(
                json.dumps(
                    _generation_receipt(
                        trial_id=trial_id,
                        behavior_id=behavior_id,
                        arm=arm,
                        placement=placement,
                        partition="calibration",
                    )
                )
            )
            score = 0.8 if arm == "full" else 0.1
            (score_trials / f"{trial_id}.json").write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "generation_receipt_sha256": sha256_file(generation_path),
                        "evaluator_revision": EVALUATOR_REVISION,
                        "yes_probability": score,
                        "no_probability": 1.0 - score,
                        "binary_prediction": score >= 0.5,
                        "parse_ok": True,
                    }
                )
            )

    result = analyze_followup_behavior_calibration(
        public_plan_path=PLAN_PATH,
        generation_root=tmp_path / "generation",
        score_root=tmp_path / "scores",
        output_path=tmp_path / "result.json",
    )

    assert result["pooled_estimate_reported"] is False
    assert result["both_orderings_continuation_gate_passed"] is True
    for placement in placements:
        contrast = result["ordering_results"][placement][
            "full_minus_structural_sham"
        ]
        assert contrast["estimate"] == pytest.approx(0.7)
        assert contrast["positive_sign_fraction"] == 1.0
        assert contrast["holm_gate_passed"] is True
