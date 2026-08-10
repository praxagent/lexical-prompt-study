from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.jlens_breaker_v2_generation_topology import (
    EXPECTED_OBSERVATIONS,
    PRIMARY_SIZE,
    _detector_decisions,
)


ROOT = Path(__file__).resolve().parents[1]


def _head() -> dict:
    return {
        "eligible": True,
        "threshold": 0.5,
        "frozen_jlens_head": {
            "mean": [0.0] * 31,
            "scale": [1.0] * 31,
            "coefficients": [0.0] + [1.0] + [0.0] * 30,
        },
    }


def test_generation_detector_decisions_apply_placement_specific_heads() -> None:
    thresholds = {
        "placements": {
            "scaffold_before_request": _head(),
            "scaffold_after_request": _head(),
        }
    }
    receipt = {"jlens_refusal_minus_compliance_trajectory": [1.0] + [0.0] * 30}
    before = _detector_decisions(
        receipt=receipt,
        thresholds=thresholds,
        placement="scaffold_before_request",
    )
    assert list(before) == ["scaffold_before_request"]
    assert before["scaffold_before_request"]["would_cancel_before_first_token"]
    shared = _detector_decisions(
        receipt=receipt,
        thresholds=thresholds,
        placement=None,
    )
    assert set(shared) == {
        "scaffold_before_request",
        "scaffold_after_request",
    }


def test_generation_topology_constants_match_frozen_primary_design() -> None:
    assert PRIMARY_SIZE == 252
    assert EXPECTED_OBSERVATIONS == 2430


def test_public_generation_topology_receipt_is_nonreconstructive_if_present() -> None:
    path = ROOT / "validation" / "jlens-breaker-v2-generation-topology.a110.public.json"
    if not path.exists():
        return
    receipt = json.loads(path.read_text())
    assert receipt["observation_count"] == EXPECTED_OBSERVATIONS
    assert receipt["primary_size"] == PRIMARY_SIZE
    assert receipt["target_generation_authorized"] is False
    assert receipt["target_generation_outcome_exists"] is False
    assert receipt["enforcement_enabled"] is False
    assert receipt["sealed_confirmation_opened_or_executed"] is False
    serialized = json.dumps(receipt, sort_keys=True)
    assert '"prompt_text"' not in serialized
    assert '"prompt_token_ids"' not in serialized
    assert '"request_text"' not in serialized
