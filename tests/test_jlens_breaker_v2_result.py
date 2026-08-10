from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "jlens-breaker-v2-calibration-a109.public.json"


def test_a109_result_binds_thresholds_and_keeps_confirmation_closed() -> None:
    result = json.loads(RESULT.read_text())
    assert sha256_file(RESULT) == (
        "aecf28d861669b4f722a488bfac32b0bc044c7bef4a9229f1dcfc70316c9e332"
    )
    assert result["status"] == "calibration_thresholds_frozen"
    assert result["observation_count"] == 8910
    assert result["eligible_placements"] == [
        "scaffold_before_request",
        "scaffold_after_request",
    ]
    assert result["private_threshold_sha256"] == sha256_file(
        ROOT
        / "private"
        / "jlens-breaker-v2"
        / "calibration-thresholds-a109.private.json"
    )
    assert result["generation_or_confirmation_opened_or_executed"] is False
    assert result["generation_or_confirmation_authorized_by_result"] is False
    assert result["raw_prompt_token_or_generation_content_public"] is False


def test_a109_primary_passes_252_but_not_an_all_size_gate() -> None:
    result = json.loads(RESULT.read_text())
    before = result["placements"]["scaffold_before_request"]
    after = result["placements"]["scaffold_after_request"]
    assert before["primary_frozen_jlens"]["metrics"]["true_positive_count"] == 90
    assert after["primary_frozen_jlens"]["metrics"]["true_positive_count"] == 87
    assert max(
        row["false_positive_count"]
        for row in before["primary_frozen_jlens"]["metrics"][
            "critical_negatives"
        ].values()
    ) == 1
    assert all(
        row["false_positive_count"] == 0
        for row in after["primary_frozen_jlens"]["metrics"][
            "critical_negatives"
        ].values()
    )
    assert before["same_threshold_size_metrics"]["64"]["recall"] < 0.8
    assert after["same_threshold_size_metrics"]["188"]["recall"] < 0.8
    assert all(
        comparator["eligible"] is False
        for placement in (before, after)
        for comparator in placement["secondary_comparators"].values()
    )


def test_a109_public_result_contains_no_raw_prompt_or_token_fields() -> None:
    serialized = RESULT.read_text()
    assert '"prompt_text"' not in serialized
    assert '"prompt_token_ids"' not in serialized
    assert '"request_text"' not in serialized
