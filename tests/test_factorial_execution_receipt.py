from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lexical_prompt_study.hashing import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "validation" / "factorial_8b_v1.execution-receipt.json"


def test_factorial_execution_receipt_binds_complete_unopened_topology() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["status"] == "canonical_generation_complete"
    assert payload["public_plan_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_8b_v1.public.json"
    )
    assert payload["assay_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "factorial_assay_a055.public.json"
    )
    assert payload["matrix"]["authorization_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_canonical_a056.authorization.json"
    )
    assert payload["descriptive_sentinels"]["authorization_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_sentinel_repair_a059.authorization.json"
    )
    components = {
        "matrix_receipt_count": payload["matrix"]["receipt_count"],
        "matrix_receipt_manifest_sha256": payload["matrix"][
            "receipt_manifest_sha256"
        ],
        "sentinel_receipt_count": payload["descriptive_sentinels"]["receipt_count"],
        "sentinel_receipt_manifest_sha256": payload["descriptive_sentinels"][
            "receipt_manifest_sha256"
        ],
    }
    assert hashlib.sha256(canonical_json_bytes(components)).hexdigest() == payload[
        "complete_topology"
    ]["component_binding_sha256"]
    assert payload["complete_topology"]["canonical_receipt_count"] == 422
    assert payload["outcome_state_at_binding"]["receipt_outcome_analysis_started"] is False
    assert payload["outcome_state_at_binding"]["held_out_confirmation_opened"] is False
