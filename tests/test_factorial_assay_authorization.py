from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.factorial_authorization import (
    validate_factorial_execution_authorization,
)
from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.models import FactorialAssayReceipt


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = ROOT / "plans" / "factorial_assay_a055.authorization.json"
CANONICAL_AUTHORIZATION = (
    ROOT / "plans" / "factorial_canonical_a056.authorization.json"
)
ASSAY_RECEIPT = ROOT / "validation" / "factorial_assay_a055.public.json"
SENTINEL_AUTHORIZATION = (
    ROOT / "plans" / "factorial_sentinel_repair_a058.authorization.json"
)


def test_a055_factorial_assay_authorization_is_exact_and_valid() -> None:
    payload = json.loads(AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="e5e6fe68b05ddbab0bf784d873f205edca4b3b3c",
        expected_stage="assay_canary",
    )
    assert payload["cost"]["maximum_compute_usd"] < 2
    assert payload["cost"]["conservative_post_run_ceiling_usd"] < 100


def test_a056_factorial_canonical_authorization_is_exact_and_valid() -> None:
    payload = json.loads(CANONICAL_AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="e5e6fe68b05ddbab0bf784d873f205edca4b3b3c",
        expected_stage="canonical_factorial",
    )
    receipt = FactorialAssayReceipt.model_validate_json(ASSAY_RECEIPT.read_text())
    assert receipt.status == "passed"
    assert (
        sha256_file(ASSAY_RECEIPT)
        == payload["bindings"]["assay_receipt_sha256"]
    )
    assert payload["cost"]["conservative_post_run_ceiling_usd"] < 100


def test_a058_sentinel_repair_authorization_is_exact_and_valid() -> None:
    payload = json.loads(SENTINEL_AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="13e19efb91e630da88b12b1f54ef1b62a37ef25b",
        expected_stage="descriptive_sentinel_repair",
    )
    assert payload["bindings"]["matrix_receipt_count"] == 420
    assert payload["cost"]["conservative_post_run_ceiling_usd"] < 100
