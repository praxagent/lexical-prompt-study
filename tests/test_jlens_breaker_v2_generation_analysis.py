from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.jlens_breaker_v2_generation_analysis import (
    MATERIALS,
    PLACEMENTS,
    REQUEST_CLASSES,
    summarize_generation_rows,
)


def _rows() -> list[dict]:
    rows = []
    for placement in PLACEMENTS:
        for material in MATERIALS:
            for request_class in REQUEST_CLASSES:
                for index in range(90):
                    harmful = request_class == "harmful_request"
                    benign = request_class == "ordinary_benign_request"
                    full = material == "full_attack_scaffold"
                    rows.append(
                        {
                            "generation_trial_id": (
                                f"{placement}-{material}-{request_class}-{index}"
                            ),
                            "request_id": f"{request_class}-{index}",
                            "material": material,
                            "request_class": request_class,
                            "placement": placement,
                            "would_cancel": harmful and full,
                            "refusal_detected": not (harmful and full),
                            "deterministic_task_completion": True if benign else None,
                            "generated_token_count": 16,
                            "finish_reason": "eos",
                            "harmbench_probability": 0.9 if harmful and full else (
                                0.1 if harmful else None
                            ),
                            "harmbench_binary_prediction": harmful and full,
                            "generation_receipt_sha256": "a" * 64,
                            "score_receipt_sha256": "b" * 64 if harmful else None,
                        }
                    )
    return rows


def test_generation_analysis_keeps_placements_separate_and_applies_gate() -> None:
    plan = json.loads(
        Path("plans/jlens_breaker_v2_generation_analysis_a123.public.json").read_text()
    )
    result = summarize_generation_rows(_rows(), plan)

    assert len(result["cells"]) == 30
    assert set(result["paired_harmful_contrasts"]) == set(PLACEMENTS)
    for placement in PLACEMENTS:
        gate = result["placement_gates"][placement]
        assert gate["primary_full_attack_harmful_success_count"] == 90
        assert gate["successful_harmful_capture_fraction"] == 1.0
        assert gate["ordinary_benign_row_count"] == 450
        assert gate["ordinary_benign_would_cancel_count"] == 0
        assert gate["generation_calibration_eligible"] is True
        contrast = result["paired_harmful_contrasts"][placement][
            "structural_sham"
        ]
        assert contrast["n_request_pairs"] == 90
        assert (
            contrast["full_minus_comparator_harmbench_probability"]["estimate"]
            == pytest.approx(0.8)
        )


def test_generation_analysis_stops_only_failing_placement() -> None:
    plan = json.loads(
        Path("plans/jlens_breaker_v2_generation_analysis_a123.public.json").read_text()
    )
    rows = _rows()
    for row in rows:
        if (
            row["placement"] == "scaffold_after_request"
            and row["material"] == "full_attack_scaffold"
            and row["request_class"] == "harmful_request"
        ):
            row["would_cancel"] = False
    result = summarize_generation_rows(rows, plan)

    assert result["placement_gates"]["scaffold_before_request"][
        "generation_calibration_eligible"
    ] is True
    assert result["placement_gates"]["scaffold_after_request"][
        "generation_calibration_eligible"
    ] is False
