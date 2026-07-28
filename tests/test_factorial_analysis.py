from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lexical_prompt_study.factorial_analysis import _primary_analysis_from_arrays


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PLAN = ROOT / "plans" / "factorial_8b_v1.analysis.json"


def test_primary_factorial_analysis_preserves_placement_and_detects_interaction() -> None:
    plan = json.loads(ANALYSIS_PLAN.read_text())
    plan["uncertainty"]["bootstrap_replicates"] = 100
    arrays = {}
    full_values = {
        "minimal_neutral_carrier": 0.10,
        "ordinary_benign_request": 0.15,
        "harmful_request": 0.40,
    }
    for request_class, full_value in full_values.items():
        value = np.zeros((20, 2, 3), dtype=np.float64)
        value[:, :, 2] = full_value
        arrays[request_class] = value
    result = _primary_analysis_from_arrays(arrays, plan)
    assert result["contrast_count"] == 22
    assert result["simultaneous_critical_value"] == 0
    assert result["placement_pooling"] is False
    for placement in plan["placement_order"]:
        decision = result["placement_decisions"][placement]
        assert decision["decision"] == "harmful_request_interaction_present"
        assert decision["harmful_request_interaction_present"] is True


def test_primary_factorial_analysis_does_not_call_null_interaction_independence() -> None:
    plan = json.loads(ANALYSIS_PLAN.read_text())
    plan["uncertainty"]["bootstrap_replicates"] = 100
    arrays = {
        request_class: np.zeros((20, 2, 3), dtype=np.float64)
        for request_class in plan["request_class_order"]
    }
    result = _primary_analysis_from_arrays(arrays, plan)
    for placement in plan["placement_order"]:
        decision = result["placement_decisions"][placement]
        assert decision["decision"] == "mixed_or_inconclusive"
        assert (
            decision["failure_to_show_interaction_is_independence_evidence"]
            is False
        )
