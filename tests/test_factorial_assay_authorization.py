from __future__ import annotations

import json
from pathlib import Path

import pytest

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
SENTINEL_RETRY_AUTHORIZATION = (
    ROOT / "plans" / "factorial_sentinel_repair_a059.authorization.json"
)
DOSE_AUTHORIZATION = (
    ROOT / "plans" / "factorial_secondary_dose_a060.authorization.json"
)
H200_DOSE_AUTHORIZATION = (
    ROOT / "plans" / "factorial_secondary_dose_a061.authorization.json"
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


def test_a059_sentinel_retry_authorization_is_exact_and_valid() -> None:
    payload = json.loads(SENTINEL_RETRY_AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="363d2d397ab0b98e48a17e41d376465954c3d7b3",
        expected_stage="descriptive_sentinel_repair",
    )
    assert payload["bindings"]["matrix_receipt_count"] == 420
    assert payload["cost"]["conservative_post_run_ceiling_usd"] < 100


def test_a060_secondary_dose_authorization_is_exact_and_valid() -> None:
    payload = json.loads(DOSE_AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="4b5bda248e99b34d4394365b19cc0cf666c295da",
        expected_stage="secondary_dose",
    )
    assert payload["scope"]["planned_conditions"] == 540
    assert payload["bindings"]["canonical_result_sha256"] == sha256_file(
        ROOT / "results" / "factorial-8b-canonical.public.json"
    )
    assert payload["bindings"]["canonical_execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "factorial_8b_v1.execution-receipt.json"
    )
    assert payload["cost"]["conservative_post_run_ceiling_usd"] > 100
    assert payload["cost"]["renewed_human_soft_gate_approval"] is True


def test_secondary_dose_authorization_rejects_unapproved_hardware() -> None:
    payload = json.loads(DOSE_AUTHORIZATION.read_text())
    payload["provider"]["gpu_type"] = "NVIDIA GeForce RTX 4090"
    with pytest.raises(ValueError, match="provider"):
        validate_factorial_execution_authorization(
            payload,
            expected_public_plan_sha256=(
                "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
            ),
            expected_private_plan_sha256=(
                "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
            ),
            expected_source_commit="4b5bda248e99b34d4394365b19cc0cf666c295da",
            expected_stage="secondary_dose",
        )


def test_a061_exact_h200_secondary_dose_authorization_is_valid() -> None:
    payload = json.loads(H200_DOSE_AUTHORIZATION.read_text())
    validate_factorial_execution_authorization(
        payload,
        expected_public_plan_sha256=(
            "8d0fdc4cd41d1ea79d0f1aebb4b642f7d0a072458c0c037a7f769c3a51c62375"
        ),
        expected_private_plan_sha256=(
            "055e27e7367d68fd64fd6109f1a0d3a3120e106a293c7adbf025410b908f1c3c"
        ),
        expected_source_commit="1e458fcf9ded89e68aee3e4949f2815848ff9cb7",
        expected_stage="secondary_dose",
    )
    assert payload["provider"]["gpu_type"] == "NVIDIA H200"
    assert payload["provider"]["minimum_gpu_memory_gib"] == 141
    assert payload["provider"]["fallback_allowed"] is False
    assert payload["cost"]["conservative_post_run_ceiling_usd"] < (
        json.loads(DOSE_AUTHORIZATION.read_text())["cost"][
            "conservative_post_run_ceiling_usd"
        ]
    )
