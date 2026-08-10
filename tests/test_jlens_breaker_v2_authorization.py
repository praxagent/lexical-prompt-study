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
