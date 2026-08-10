from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_a105_allocation_authorization_is_exact_and_below_hard_ceiling() -> None:
    authorization = json.loads(
        (
            ROOT
            / "plans"
            / "jlens_breaker_v2_calibration_a105.allocation-authorization.json"
        ).read_text()
    )
    assert authorization["current_account_pods"] == 0
    assert authorization["maximum_task_owned_pods"] == 1
    assert authorization["paid_model_execution_authorized"] is False
    assert authorization["requested_infrastructure"]["gpu_type_id"] == (
        "NVIDIA H100 80GB HBM3"
    )
    assert authorization["requested_infrastructure"]["network_volume_id"] == (
        "u85xfo0aue"
    )
    assert authorization["bindings"]["source_commit"] == (
        "087c5b22d952158c8a8a4849796ed803ad6ace2e"
    )
    assert authorization["bindings"]["public_plan_sha256"] == sha256_file(
        ROOT / "plans" / "jlens_breaker_v2.public.json"
    )
    assert authorization["bindings"]["private_topology_sha256"] == sha256_file(
        ROOT / "private" / "jlens-breaker-v2" / "calibration-topology.a103.private.json"
    )
    assert authorization["bindings"][
        "public_topology_receipt_sha256"
    ] == sha256_file(
        ROOT / "validation" / "jlens-breaker-v2-calibration-topology.a103.public.json"
    )
    budget = authorization["budget"]
    assert budget["maximum_new_compute_usd"] == (
        budget["maximum_wall_minutes"]
        / 60
        * authorization["requested_infrastructure"]["maximum_rate_usd_per_hour"]
    )
    assert budget["conservative_post_run_infrastructure_usd"] <= budget[
        "hard_ceiling_usd"
    ]


def test_a106_execution_authorization_binds_exact_pod_source_and_topology() -> None:
    authorization = json.loads(
        (
            ROOT / "plans" / "jlens_breaker_v2_calibration_a106.authorization.json"
        ).read_text()
    )
    allocation = json.loads(
        (
            ROOT
            / "validation"
            / "jlens-breaker-v2-a105.allocation-result.public.json"
        ).read_text()
    )
    assert authorization["status"] == "v2_prefill_authorized"
    assert authorization["paid_compute_authorized"] is True
    assert authorization["expected_observations"] == 8910
    assert authorization["infrastructure"]["task_pod_id"] == allocation["pod"]["id"]
    assert authorization["infrastructure"]["maximum_rate_usd_per_hour"] == (
        allocation["pod"]["cost_per_hour_usd"]
    )
    assert authorization["infrastructure"]["new_pod_creation_allowed"] is False
    assert authorization["bindings"]["source_commit"] == (
        "087c5b22d952158c8a8a4849796ed803ad6ace2e"
    )
    assert authorization["bindings"]["public_plan_sha256"] == sha256_file(
        ROOT / "plans" / "jlens_breaker_v2.public.json"
    )
    assert authorization["bindings"]["private_topology_sha256"] == sha256_file(
        ROOT / "private" / "jlens-breaker-v2" / "calibration-topology.a103.private.json"
    )
    assert authorization["conservative_postrun_infrastructure_ceiling_usd"] <= (
        authorization["hard_ceiling_usd"]
    )


def test_a106_execution_receipt_binds_complete_private_bundle_and_teardown() -> None:
    receipt = json.loads(
        (
            ROOT
            / "validation"
            / "jlens-breaker-v2-calibration-a106.execution-receipt.public.json"
        ).read_text()
    )
    acquisition = receipt["acquisition"]
    assert acquisition["completed_observations"] == acquisition[
        "expected_observations"
    ] == 8910
    assert acquisition["schema_validated_receipts"] == 8910
    assert acquisition["outcomes_analyzed"] is False
    assert acquisition["private_bundle_sha256"] == sha256_file(
        ROOT
        / "private"
        / "runs"
        / "jlens-v2-a106"
        / "jlens-breaker-v2-calibration-a106.tar.gz"
    )
    assert receipt["infrastructure"]["post_teardown_account_pods"] == 0
    assert receipt["sealed_confirmation_executed_or_opened"] is False
    assert receipt["infrastructure"]["post_run_infrastructure_ceiling_usd"] < (
        receipt["infrastructure"]["hard_ceiling_usd"]
    )
