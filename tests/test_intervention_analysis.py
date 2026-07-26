from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lexical_prompt_study.intervention_analysis import (
    analyze_alpha_calibration,
    calibration_condition_id,
    planned_calibration_conditions,
)


ROOT = Path(__file__).parents[1]
PLAN = json.loads((ROOT / "plans/gate4_intervention_v1.public.json").read_text())


def _rows() -> list[dict]:
    rows = []
    for behavior_index in range(20):
        behavior_id = f"behavior-{behavior_index:02d}"
        rows.append(
            {
                "behavior_id": behavior_id,
                "condition_id": "zero",
                "evaluator_score": 0.6,
                "evaluator_parse_ok": True,
                "requested_alpha": 0.0,
                "maximum_requested_realized_relative_error": 0.0,
                "maximum_event_delta_to_pre_residual_norm": 0.0,
                "truncated": False,
                "error": None,
            }
        )
        for rho in PLAN["discovery_alpha_calibration"]["rho_ladder"]:
            effect = 0.05 if rho == 0.0025 else 0.25
            for sign, score in ((-1, 0.6 - effect), (1, 0.6 + effect)):
                rows.append(
                    {
                        "behavior_id": behavior_id,
                        "condition_id": calibration_condition_id(sign, rho),
                        "evaluator_score": score,
                        "evaluator_parse_ok": True,
                        "requested_alpha": rho * 100,
                        "maximum_requested_realized_relative_error": 0.0005,
                        "maximum_event_delta_to_pre_residual_norm": rho,
                        "truncated": False,
                        "error": None,
                    }
                )
    return rows


def test_calibration_condition_topology_is_frozen() -> None:
    conditions = planned_calibration_conditions(PLAN)
    assert conditions[0] == "zero"
    assert conditions[1] == "primary_negative_rho_0p0025"
    assert conditions[-1] == "primary_positive_rho_0p02"
    assert len(conditions) == 9


def test_alpha_selection_uses_smallest_eligible_rho() -> None:
    result = analyze_alpha_calibration(_rows(), PLAN)
    assert result["status"] == "selected"
    assert result["selection"]["rho"] == 0.005
    assert result["selection"]["restoring_sign"] == -1
    assert result["confirmatory_outcomes_opened"] is False


def test_alpha_selection_stops_when_no_dose_is_eligible() -> None:
    rows = _rows()
    for row in rows:
        if row["condition_id"] != "zero":
            row["evaluator_score"] = 0.6
    result = analyze_alpha_calibration(rows, PLAN)
    assert result["status"] == "stopped_no_eligible_alpha"
    assert result["selection"] is None


def test_alpha_selection_rejects_duplicate_or_missing_topology() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate calibration row"):
        analyze_alpha_calibration([*rows, copy.deepcopy(rows[0])], PLAN)
    with pytest.raises(ValueError, match="condition topology mismatch"):
        analyze_alpha_calibration(rows[:-1], PLAN)
