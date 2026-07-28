from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "factorial-8b-secondary-dose.public.json"


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def test_secondary_dose_result_is_complete_bound_and_public_safe() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["source_commit"] == "4a95f23811851423ab7048195295d6d3dcc90d47"
    assert payload["analysis_plan_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_8b_v1.dose-analysis.json"
    )
    assert payload["execution_receipt_sha256"] == sha256_file(
        ROOT
        / "validation"
        / "factorial_secondary_dose_a065.execution-receipt.json"
    )
    assert payload["analysis_implementation_sha256"] == sha256_file(
        ROOT / "src" / "lexical_prompt_study" / "factorial_dose_analysis.py"
    )
    assert payload["source_receipts"]["new_partial_dose_count"] == 540
    assert payload["source_receipts"]["reused_canonical_count"] == 180
    assert payload["source_receipts"]["restricted_artifact_hashes_verified"] is True
    assert len(payload["cell_summaries"]) == 72
    assert all(
        metric["bootstrap_replicates"] == 10000
        and metric["interval_role"] == "pointwise_descriptive_not_simultaneous"
        and metric["formal_pass_fail_decision"] is False
        for metric in payload["metric_contrasts"].values()
    )
    assert payload["claim_limits"]["placement_pooling"] is False
    assert payload["claim_limits"]["size_pooling"] is False
    assert payload["claim_limits"]["monotonicity_test"] is False
    assert payload["claim_limits"]["threshold_fit"] is False
    assert payload["claim_limits"]["held_out_confirmation_opened"] is False
    assert not {
        "prompt_text",
        "prompt_token_ids",
        "generated_text",
        "generated_token_ids",
    }.intersection(_walk_keys(payload))
