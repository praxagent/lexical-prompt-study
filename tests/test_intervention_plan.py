from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.intervention_plan import validate_intervention_plan


ROOT = Path(__file__).parents[1]
PLAN = ROOT / "plans/gate4_intervention_v1.public.json"
PUBLIC = ROOT / "plans/study_v1.public.json"
ANALYSIS = ROOT / "results/gate3.discovery.json"
PRIVATE = ROOT / "private/plans/study_v1.private.json"


def test_frozen_intervention_plan_validates_against_receipts() -> None:
    result = validate_intervention_plan(
        PLAN,
        PUBLIC,
        ANALYSIS,
        private_study_path=PRIVATE,
    )
    assert result["status"] == "valid"
    assert result["selected_feature_ids"] == [10146, 44802, 4057, 3907]
    assert result["matched_control_feature_ids"] == [26453, 9105, 40804]
    assert result["confirmatory_generations"] == 600
    assert result["utility_generations"] == 80


def test_intervention_plan_rejects_feature_drift(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text())
    payload["features"]["primary_feature_id"] = 0
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="candidate feature IDs drift"):
        validate_intervention_plan(drifted, PUBLIC, ANALYSIS)


def test_intervention_plan_rejects_topology_drift(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text())
    payload["confirmatory_attack_panel"]["expected_generations"] = 599
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="topology mismatch"):
        validate_intervention_plan(drifted, PUBLIC, ANALYSIS)
