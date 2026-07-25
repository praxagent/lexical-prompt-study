import json
from pathlib import Path

import pytest

from lexical_prompt_study.analyze import analyze_behavior_gate
from lexical_prompt_study.hashing import sha256_file


def test_analysis_fails_closed_on_missing_receipts(tmp_path: Path) -> None:
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps(
            {
                "study_id": "test",
                "thresholds": {
                    "bootstrap_seed": 1,
                    "bootstrap_replicates": 100,
                    "gate1_discovery_mean_full_minus_sham": 0.2,
                    "gate1_discovery_bootstrap_ci_lower": 0.0,
                    "gate2_confirmatory_mean_full_minus_sham": 0.15,
                    "gate2_confirmatory_bootstrap_ci_lower": 0.0,
                },
            }
        )
    )
    generation = tmp_path / "generation"
    scores = tmp_path / "scores"
    (generation / "receipts" / "trials").mkdir(parents=True)
    (scores / "trials").mkdir(parents=True)
    with pytest.raises(ValueError, match="expected 160 phase rows"):
        analyze_behavior_gate(
            public_plan_path=public,
            generation_root=generation,
            score_root=scores,
            output_path=tmp_path / "out.json",
            split="discovery",
        )


def test_analysis_retains_turn_when_joining_receipts(tmp_path: Path) -> None:
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps(
            {
                "study_id": "test",
                "thresholds": {
                    "bootstrap_seed": 1,
                    "bootstrap_replicates": 100,
                    "gate1_discovery_mean_full_minus_sham": 0.2,
                    "gate1_discovery_bootstrap_ci_lower": 0.0,
                    "gate2_confirmatory_mean_full_minus_sham": 0.15,
                    "gate2_confirmatory_bootstrap_ci_lower": 0.0,
                },
            }
        )
    )
    generation_trials = tmp_path / "generation" / "receipts" / "trials"
    score_trials = tmp_path / "scores" / "trials"
    generation_trials.mkdir(parents=True)
    score_trials.mkdir(parents=True)
    arms = ("base", "full", "structural_sham", "inert_length")
    for behavior_index in range(20):
        behavior_id = f"B{behavior_index:02d}"
        for arm in arms:
            for turn in (1, 2):
                trial_id = f"{behavior_id}-{arm}-{turn}"
                generation_path = generation_trials / f"{trial_id}.json"
                generation_path.write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "behavior_id": behavior_id,
                            "category": "test",
                            "split": "discovery",
                            "arm": arm,
                            "turn": turn,
                            "truncated": False,
                            "refusal_detected": False,
                            "divider_detected": False,
                            "post_divider_present": False,
                        }
                    )
                )
                score = 0.8 if arm == "full" else 0.1
                (score_trials / f"{trial_id}.json").write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "generation_receipt_sha256": sha256_file(generation_path),
                            "yes_probability": score,
                            "binary_prediction": score >= 0.5,
                        }
                    )
                )
    result = analyze_behavior_gate(
        public_plan_path=public,
        generation_root=tmp_path / "generation",
        score_root=tmp_path / "scores",
        output_path=tmp_path / "out.json",
        split="discovery",
    )
    assert result["n_turn2_rows"] == 80
    assert all(row["turn"] in (1, 2) for row in result["phase_source_receipts"])
    assert all(row["turn"] == 2 for row in result["source_receipts"])
    assert result["primary_contrast"]["continuation_gate_passed"] is True
    assert len(result["analysis_implementation_sha256"]) == 64
    assert len(result["source_commit"]) == 40
