from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .jlens_breaker_v2_analysis import (
    PLACEMENTS,
    _load_bundle,
    apply_frozen_jlens_head,
)


PRIMARY_SIZE = 252
EXPECTED_OBSERVATIONS = 2430


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> str:
    raw = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == 0o600:
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    temporary.replace(path)
    return sha256_bytes(raw)


def _detector_decisions(
    *,
    receipt: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    placement: str | None,
) -> dict[str, Any]:
    selected = PLACEMENTS if placement is None else (placement,)
    decisions: dict[str, Any] = {}
    trajectory = [receipt["jlens_refusal_minus_compliance_trajectory"]]
    for candidate_placement in selected:
        frozen = thresholds["placements"][candidate_placement]
        if frozen.get("eligible") is not True:
            raise ValueError("generation topology includes an ineligible placement")
        score = float(
            apply_frozen_jlens_head(trajectory, frozen["frozen_jlens_head"])[0]
        )
        threshold = float(frozen["threshold"])
        decisions[candidate_placement] = {
            "frozen_jlens_score": score,
            "threshold": threshold,
            "would_cancel_before_first_token": score >= threshold,
        }
    return decisions


def build_jlens_breaker_v2_generation_topology(
    *,
    public_plan_path: Path,
    calibration_topology_path: Path,
    calibration_topology_receipt_path: Path,
    calibration_bundle_path: Path,
    calibration_execution_receipt_path: Path,
    calibration_result_path: Path,
    private_threshold_path: Path,
    private_output_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    plan = json.loads(public_plan_path.read_text())
    topology = json.loads(calibration_topology_path.read_text())
    topology_receipt = json.loads(calibration_topology_receipt_path.read_text())
    execution = json.loads(calibration_execution_receipt_path.read_text())
    result = json.loads(calibration_result_path.read_text())
    thresholds = json.loads(private_threshold_path.read_text())

    bindings = {
        "public_plan_sha256": sha256_file(public_plan_path),
        "calibration_topology_sha256": sha256_file(calibration_topology_path),
        "calibration_topology_receipt_sha256": sha256_file(
            calibration_topology_receipt_path
        ),
        "calibration_bundle_sha256": sha256_file(calibration_bundle_path),
        "calibration_execution_receipt_sha256": sha256_file(
            calibration_execution_receipt_path
        ),
        "calibration_result_sha256": sha256_file(calibration_result_path),
        "private_threshold_sha256": sha256_file(private_threshold_path),
    }
    if (
        plan.get("study_id") != "lexical-jlens-breaker-v2"
        or topology.get("status") != "v2_topology_frozen_no_target_outcomes"
        or topology.get("partition") != "detector_calibration"
        or topology.get("prefill_only") is not True
        or topology_receipt.get("private_topology_sha256")
        != bindings["calibration_topology_sha256"]
        or execution.get("bindings", {}).get("private_topology_sha256")
        != bindings["calibration_topology_sha256"]
        or execution.get("acquisition", {}).get("private_bundle_sha256")
        != bindings["calibration_bundle_sha256"]
        or result.get("status") != "calibration_thresholds_frozen"
        or result.get("private_threshold_sha256")
        != bindings["private_threshold_sha256"]
        or result.get("generation_or_confirmation_opened_or_executed") is not False
        or sorted(result.get("eligible_placements", [])) != sorted(PLACEMENTS)
    ):
        raise ValueError("generation topology calibration binding drift")

    bundle_summary, prefill_rows = _load_bundle(calibration_bundle_path)
    if (
        bundle_summary.get("private_topology_sha256")
        != bindings["calibration_topology_sha256"]
        or bundle_summary.get("receipt_manifest_sha256")
        != execution["acquisition"]["receipt_manifest_sha256"]
    ):
        raise ValueError("generation topology bundle provenance drift")
    prefill_by_id = {row["trial_id"]: row for row in prefill_rows}

    observations: list[dict[str, Any]] = []
    public_manifest: list[dict[str, Any]] = []
    for row in topology["observations"]:
        if row["material"] != "no_scaffold" and row["injected_token_count"] != PRIMARY_SIZE:
            continue
        prefill = prefill_by_id.get(row["trial_id"])
        if prefill is None:
            raise ValueError("generation topology lacks linked prefill receipt")
        decisions = _detector_decisions(
            receipt=prefill,
            thresholds=thresholds,
            placement=row["placement"],
        )
        generation_id = sha256_bytes(
            canonical_json_bytes(
                {
                    "study_id": plan["study_id"],
                    "stage": "generation_calibration",
                    "prefill_trial_id": row["trial_id"],
                    "max_new_tokens": int(topology["generation_budget"]),
                }
            )
        )[:24]
        public = {
            "generation_trial_id": generation_id,
            "prefill_trial_id": row["trial_id"],
            "request_class": row["request_class"],
            "request_id": row["request_id"],
            "request_sha256": row["request_sha256"],
            "material": row["material"],
            "material_sha256": row["material_sha256"],
            "wrapper_id": row["wrapper_id"],
            "placement": row["placement"],
            "injected_token_count": row["injected_token_count"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "prefill_receipt_sha256": sha256_bytes(
                canonical_json_bytes(prefill)
            ),
            "detector_decisions": decisions,
        }
        public_manifest.append(public)
        observations.append(
            {
                **public,
                "prompt_text": row["prompt_text"],
                "prompt_token_ids": row["prompt_token_ids"],
                "request_text": row["request_text"],
                "request_context": row.get("request_context"),
                "utility_judge": row.get("utility_judge"),
                "source_dataset": row.get("source_dataset"),
            }
        )

    if (
        len(observations) != EXPECTED_OBSERVATIONS
        or len({row["generation_trial_id"] for row in observations})
        != EXPECTED_OBSERVATIONS
    ):
        raise ValueError("generation topology observation count or identity drift")
    private_payload = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "stage": "generation_calibration",
        "status": "v2_generation_topology_frozen_no_generation_outcomes",
        "bindings": bindings,
        "primary_size": PRIMARY_SIZE,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": int(topology["generation_budget"]),
            "model_revision": plan["pinned_instrument"]["model_revision"],
        },
        "enforcement_enabled": False,
        "sealed_confirmation_opened_or_executed": False,
        "observations": observations,
    }
    private_sha256 = _atomic_json(private_output_path, private_payload, mode=0o600)
    receipt = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "stage": "generation_calibration",
        "status": private_payload["status"],
        "bindings": bindings,
        "private_generation_topology_sha256": private_sha256,
        "primary_size": PRIMARY_SIZE,
        "max_new_tokens": int(topology["generation_budget"]),
        "observation_count": len(observations),
        "request_class_counts": dict(
            Counter(row["request_class"] for row in public_manifest)
        ),
        "material_counts": dict(
            Counter(row["material"] for row in public_manifest)
        ),
        "placement_counts": dict(
            Counter(str(row["placement"]) for row in public_manifest)
        ),
        "detector_decision_counts": {
            placement: {
                "would_cancel": sum(
                    bool(row["detector_decisions"].get(placement, {}).get(
                        "would_cancel_before_first_token", False
                    ))
                    for row in public_manifest
                ),
                "evaluated": sum(
                    placement in row["detector_decisions"] for row in public_manifest
                ),
            }
            for placement in PLACEMENTS
        },
        "observation_manifest_sha256": sha256_bytes(
            canonical_json_bytes(public_manifest)
        ),
        "placements_reported_separately": True,
        "target_generation_authorized": False,
        "target_generation_outcome_exists": False,
        "enforcement_enabled": False,
        "sealed_confirmation_opened_or_executed": False,
        "raw_prompt_token_or_generation_content_public": False,
    }
    receipt_sha256 = _atomic_json(public_receipt_path, receipt, mode=0o644)
    return {
        "status": receipt["status"],
        "private_generation_topology_sha256": private_sha256,
        "public_receipt_sha256": receipt_sha256,
        "observation_count": len(observations),
        "target_generation_outcome_exists": False,
    }
