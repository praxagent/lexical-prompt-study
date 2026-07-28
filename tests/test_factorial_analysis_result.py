from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "factorial-8b-canonical.public.json"


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def test_factorial_analysis_result_is_complete_bound_and_public_safe() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["source_commit"] == "c3650b7b393d83e97d68662dfd546e4f3f0bb1bb"
    assert payload["public_plan_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_8b_v1.public.json"
    )
    assert payload["analysis_plan_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_8b_v1.analysis.json"
    )
    assert payload["execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "factorial_8b_v1.execution-receipt.json"
    )
    assert payload["analysis_implementation_sha256"] == sha256_file(
        ROOT / "src" / "lexical_prompt_study" / "factorial_analysis.py"
    )
    assert payload["source_receipts"]["matrix_count"] == 420
    assert payload["source_receipts"]["sentinel_count"] == 2
    assert payload["source_receipts"]["restricted_artifact_hashes_verified"] is True
    assert payload["primary_feature_6779"]["bootstrap_replicates"] == 100000
    assert payload["primary_feature_6779"]["placement_pooling"] is False
    assert {
        row["decision"]
        for row in payload["primary_feature_6779"]["placement_decisions"].values()
    } == {"mixed_or_inconclusive"}
    assert payload["claim_limits"]["threshold_fit"] is False
    assert payload["claim_limits"]["held_out_confirmation_opened"] is False
    assert not {
        "prompt_text",
        "prompt_token_ids",
        "generated_text",
        "generated_token_ids",
    }.intersection(_walk_keys(payload))
