import json
from pathlib import Path

import pytest

from lexical_prompt_study.analyze import analyze_behavior_gate


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
    with pytest.raises(ValueError, match="expected 80 turn-2 rows"):
        analyze_behavior_gate(
            public_plan_path=public,
            generation_root=generation,
            score_root=scores,
            output_path=tmp_path / "out.json",
            split="discovery",
        )
