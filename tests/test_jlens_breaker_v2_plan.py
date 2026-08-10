from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "jlens_breaker_v2.public.json"


def test_v2_plan_preserves_failed_confirmation_boundary() -> None:
    plan = json.loads(PLAN.read_text())
    bindings = plan["predecessor_bindings"]
    assert bindings["v1_confirmation_result_sha256"] == sha256_file(
        ROOT / "results" / "weaponization-confirmation-v1.public.json"
    )
    assert "may never fit" in bindings["v1_confirmation_role"]
    assert "validate a v2" in bindings["v1_confirmation_role"]
    assert plan["candidate_freeze"]["projection_coefficients_may_change"] is False
    assert plan["candidate_freeze"]["standardization_references_may_change"] is False
    assert plan["candidate_freeze"]["sae_structure_gate_removed"] is True
    assert plan["candidate_freeze"]["v2_trainable_parameters"] == (
        "one scalar threshold per placement fitted on v2 calibration only"
    )


def test_v2_plan_keeps_size_placement_and_deployment_gates() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["factors"]["placement"] == [
        "scaffold_before_request",
        "scaffold_after_request",
    ]
    assert plan["factors"]["injected_token_count"] == [64, 128, 188, 252]
    assert plan["factors"]["sizes_never_pooled"] is True
    assert plan["factors"]["threshold_refit_by_size_forbidden"] is True
    assert plan["sealed_confirmation"]["one_shot"] is True
    assert plan["breaker_boundary"]["production_enforcement_authorized"] is False
    assert plan["restricted_data_boundary"][
        "agent_plaintext_or_reconstructive_token_inspection"
    ] is False
