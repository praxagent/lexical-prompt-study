from __future__ import annotations

import numpy as np

from lexical_prompt_study.analyze_mechanisms import (
    BOOTSTRAP_SEED,
    _bootstrap_interval,
    _matched_controls,
    _paired_deltas,
)


def test_paired_deltas_require_both_arms_and_available_position() -> None:
    rows = [
        {
            "behavior_id": "a",
            "arm": "full",
            "position_available": True,
            "margin": {"refusal_minus_compliance_margin": 2.0},
        },
        {
            "behavior_id": "a",
            "arm": "structural_sham",
            "position_available": True,
            "margin": {"refusal_minus_compliance_margin": 0.5},
        },
        {
            "behavior_id": "b",
            "arm": "full",
            "position_available": False,
            "margin": None,
        },
    ]
    behavior_ids, values = _paired_deltas(rows)
    assert behavior_ids == ["a"]
    assert values.tolist() == [1.5]


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert _bootstrap_interval(values, BOOTSTRAP_SEED) == _bootstrap_interval(
        values, BOOTSTRAP_SEED
    )


def test_matched_controls_exclude_selected_and_large_discovery_effects() -> None:
    diagnostics = [
        {
            "feature_id": 1,
            "decoder_norm": 1.0,
            "all_prevalence": 0.5,
            "paired_standardized_delta": 0.9,
        },
        {
            "feature_id": 2,
            "decoder_norm": 1.01,
            "all_prevalence": 0.5,
            "paired_standardized_delta": 0.1,
        },
        {
            "feature_id": 3,
            "decoder_norm": 0.99,
            "all_prevalence": 0.5,
            "paired_standardized_delta": -0.1,
        },
        {
            "feature_id": 4,
            "decoder_norm": 1.02,
            "all_prevalence": 0.5,
            "paired_standardized_delta": 0.0,
        },
        {
            "feature_id": 5,
            "decoder_norm": 1.0,
            "all_prevalence": 0.5,
            "paired_standardized_delta": 0.8,
        },
    ]
    controls = _matched_controls(diagnostics, [1], count=3)
    assert {item["feature_id"] for item in controls} == {2, 3, 4}
