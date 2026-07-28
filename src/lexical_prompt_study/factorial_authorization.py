from __future__ import annotations

import math
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_factorial_execution_authorization(
    plan: dict[str, Any],
    *,
    expected_public_plan_sha256: str,
    expected_private_plan_sha256: str,
    expected_source_commit: str,
    expected_stage: str,
) -> None:
    _require(plan["schema_version"] == "1.0", "factorial authorization schema drift")
    _require(
        plan["study_id"] == "lexical-scaffold-8b-factorial-v1",
        "factorial authorization study drift",
    )
    _require(
        plan["stage"]
        in {
            "assay_canary",
            "canonical_factorial",
            "descriptive_sentinel_repair",
            "secondary_dose",
        },
        "factorial authorization stage drift",
    )
    _require(plan["stage"] == expected_stage, "wrong factorial authorization stage")
    bindings = plan["bindings"]
    _require(
        bindings["public_plan_sha256"] == expected_public_plan_sha256
        and bindings["private_plan_sha256"] == expected_private_plan_sha256
        and bindings["source_commit"] == expected_source_commit,
        "factorial authorization binding drift",
    )
    _require(
        len(bindings["source_commit"]) == 40,
        "factorial authorization source commit must be exact",
    )
    scope = plan["scope"]
    if expected_stage == "assay_canary":
        _require(
            plan["status"] == "prospective_factorial_assay_authorization"
            and scope["target_factorial_outcomes_authorized"] is False
            and scope["planned_conditions"] == 8
            and bindings["assay_receipt_sha256"] is None,
            "factorial assay authorization scope drift",
        )
    elif expected_stage == "canonical_factorial":
        _require(
            plan["status"] == "prospective_factorial_canonical_authorization"
            and scope["target_factorial_outcomes_authorized"] is True
            and scope["planned_conditions"] == 422
            and isinstance(bindings["assay_receipt_sha256"], str)
            and len(bindings["assay_receipt_sha256"]) == 64,
            "factorial canonical authorization scope drift",
        )
    elif expected_stage == "descriptive_sentinel_repair":
        _require(
            plan["status"]
            == "prospective_factorial_sentinel_repair_authorization"
            and scope["target_factorial_outcomes_authorized"] is False
            and scope["descriptive_sentinel_outcomes_authorized"] is True
            and scope["planned_conditions"] == 2
            and isinstance(bindings["assay_receipt_sha256"], str)
            and len(bindings["assay_receipt_sha256"]) == 64
            and bindings["matrix_receipt_count"] == 420
            and isinstance(bindings["matrix_receipt_manifest_sha256"], str)
            and len(bindings["matrix_receipt_manifest_sha256"]) == 64
            and isinstance(bindings["matrix_source_commit"], str)
            and len(bindings["matrix_source_commit"]) == 40
            and isinstance(bindings["matrix_run_id"], str),
            "factorial sentinel-repair authorization scope drift",
        )
    else:
        _require(
            plan["status"] == "prospective_factorial_dose_authorization"
            and scope["target_factorial_outcomes_authorized"] is True
            and 0 <= scope["planned_conditions"] <= 540
            and isinstance(bindings["assay_receipt_sha256"], str)
            and len(bindings["assay_receipt_sha256"]) == 64
            and isinstance(bindings["canonical_result_sha256"], str)
            and len(bindings["canonical_result_sha256"]) == 64
            and isinstance(bindings["canonical_execution_receipt_sha256"], str)
            and len(bindings["canonical_execution_receipt_sha256"]) == 64
            and bindings["matrix_receipt_count"] == 420
            and isinstance(bindings["matrix_receipt_manifest_sha256"], str)
            and len(bindings["matrix_receipt_manifest_sha256"]) == 64
            and isinstance(bindings["matrix_source_commit"], str)
            and len(bindings["matrix_source_commit"]) == 40
            and isinstance(bindings["matrix_run_id"], str)
            and isinstance(bindings["dose_observation_manifest_sha256"], str)
            and len(bindings["dose_observation_manifest_sha256"]) == 64,
            "factorial dose authorization scope drift",
        )
    _require(
        scope["placement_pooling"] is False
        and scope["size_pooling"] is False
        and scope["detector_threshold_fitting"] is False
        and scope["completed_receipt_overwrite"] is False,
        "factorial authorization scientific scope drift",
    )
    provider = plan["provider"]
    _require(
        provider["maximum_task_owned_pods"] == 1
        and provider["gpu_count"] == 1
        and provider["gpu_type"] == "NVIDIA B200"
        and provider["datacenter_id"] == "US-CA-2"
        and provider["secure_cloud"] is True
        and provider["persistent_volume_id"] == "u85xfo0aue"
        and provider["persistent_volume_mount"] == "/workspace"
        and provider["fallback_allowed"] is False,
        "factorial authorization provider drift",
    )
    cost = plan["cost"]
    expected_maximum = (
        float(cost["maximum_live_rate_usd_per_hour"])
        * float(cost["wall_time_minutes"])
        / 60
    )
    _require(
        math.isclose(
            float(cost["maximum_compute_usd"]),
            expected_maximum,
            rel_tol=0,
            abs_tol=1e-9,
        ),
        "factorial authorization cost arithmetic drift",
    )
    post_run = float(cost["conservative_pre_run_ceiling_usd"]) + float(
        cost["maximum_compute_usd"]
    )
    _require(
        math.isclose(
            float(cost["conservative_post_run_ceiling_usd"]),
            post_run,
            rel_tol=0,
            abs_tol=1e-9,
        )
        and post_run <= 200,
        "factorial authorization hard-ceiling drift",
    )
    if post_run > 100:
        _require(
            cost["renewed_human_soft_gate_approval"] is True,
            "factorial authorization crosses soft gate without renewed approval",
        )
    _require(
        int(cost["no_progress_stop_minutes"]) > 0
        and int(cost["no_progress_stop_minutes"]) < int(cost["wall_time_minutes"]),
        "factorial authorization stop-rule drift",
    )
