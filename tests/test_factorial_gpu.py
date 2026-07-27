from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.factorial_gpu import (
    REFUSAL_PARSER_VERSION,
    _legacy_rows,
    _parser_result,
    _probe_manifest,
)
from lexical_prompt_study.factorial_plan import load_factorial_plan
from lexical_prompt_study.hashing import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_PLAN = ROOT / "plans" / "factorial_8b_v1.public.json"
PROBE_PLAN = ROOT / "plans" / "study_v1.public.json"


def test_probe_manifest_matches_prospective_hash() -> None:
    factorial = load_factorial_plan(FACTORIAL_PLAN)
    probe = json.loads(PROBE_PLAN.read_text())
    assert sha256_bytes(canonical_json_bytes(_probe_manifest(probe))) == factorial[
        "core_readout_implementation"
    ]["jacobian_lens_probe_manifest_sha256"]


def test_generation_parser_is_descriptive_not_semantic_judge() -> None:
    parsed = _parser_result(
        "I cannot assist with that.",
        generated_token_count=5,
        finish_reason="eos",
        truncated=False,
    )
    assert parsed["parser_version"] == REFUSAL_PARSER_VERSION
    assert parsed["refusal_detected"] is True
    assert parsed["semantic_task_completion_judged"] is False


def test_legacy_canary_rows_bind_exact_hashes_without_plaintext_output() -> None:
    factorial = load_factorial_plan(FACTORIAL_PLAN)
    rows = []
    for frozen in factorial["assay_validity_gate"]["legacy_request_selection"]:
        request = f"safe fixture for {frozen['behavior_id']}"
        frozen["request_sha256"] = __import__("hashlib").sha256(
            request.encode()
        ).hexdigest()
        rows.append(
            {
                "behavior_id": frozen["behavior_id"],
                "request": request,
                "request_sha256": frozen["request_sha256"],
            }
        )
    selected = _legacy_rows(
        public_plan=factorial,
        legacy_private={"partitions": {"discovery": rows}},
    )
    assert [row["behavior_id"] for row in selected] == [
        "JBB-H-013",
        "JBB-H-076",
    ]
