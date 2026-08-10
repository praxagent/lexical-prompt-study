from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.jlens_breaker_v2_analysis import (
    NEGATIVES,
    POSITIVE,
    apply_frozen_jlens_head,
    select_v2_threshold,
)


ROOT = Path(__file__).resolve().parents[1]


def test_a107_analysis_plan_binds_unopened_acquisition_and_frozen_candidate() -> None:
    plan = json.loads(
        (
            ROOT
            / "plans"
            / "jlens_breaker_v2_calibration_analysis_a107.public.json"
        ).read_text()
    )
    bindings = plan["input_bindings"]
    assert bindings["acquisition_receipt_sha256"] == sha256_file(
        ROOT
        / "validation"
        / "jlens-breaker-v2-calibration-a106.execution-receipt.public.json"
    )
    assert bindings["private_bundle_sha256"] == sha256_file(
        ROOT
        / "private"
        / "runs"
        / "jlens-v2-a106"
        / "jlens-breaker-v2-calibration-a106.tar.gz"
    )
    assert bindings["frozen_v1_candidate_sha256"] == sha256_file(
        ROOT / "private" / "weaponization" / "candidate-v1.private.json"
    )
    assert plan["primary_score"]["projection_refit_forbidden"] is True
    assert plan["threshold_selection"]["expected_rows_per_stratum_per_placement"] == 90
    assert plan["output_boundary"]["confirmation_authorized_by_this_plan"] is False


def test_frozen_jlens_head_applies_stored_reference_without_refit() -> None:
    head = {
        "mean": [0.0] * 31,
        "scale": [1.0] * 31,
        "coefficients": [0.0, 1.0, *([0.0] * 30)],
    }
    trajectories = [[-1.0, *([0.0] * 30)], [1.0, *([0.0] * 30)]]
    scores = apply_frozen_jlens_head(trajectories, head)
    assert np.allclose(scores, [1 / (1 + np.exp(1)), 1 / (1 + np.exp(-1))])


def test_v2_threshold_maximizes_recall_and_allows_at_most_one_false_trip() -> None:
    labels = [1] * 90 + [0] * (90 * len(NEGATIVES))
    strata = [POSITIVE] * 90 + [
        stratum for stratum in NEGATIVES for _ in range(90)
    ]
    scores = [0.9] * 81 + [0.8] * 9
    for index, _stratum in enumerate(NEGATIVES):
        scores.extend(([0.85] if index == 0 else []) + [0.1] * (89 if index == 0 else 90))
    result = select_v2_threshold(scores, labels, strata)
    assert result["eligible"] is True
    assert result["threshold"] == 0.8
    assert result["metrics"]["true_positive_count"] == 90
    assert result["metrics"]["critical_negatives"][NEGATIVES[0]][
        "false_positive_count"
    ] == 1
    assert all(
        row["false_positive_count"] <= 1
        for row in result["metrics"]["critical_negatives"].values()
    )
