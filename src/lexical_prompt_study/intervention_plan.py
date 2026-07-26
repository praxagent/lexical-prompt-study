from __future__ import annotations

import json
import re
from pathlib import Path

from .hashing import sha256_file

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def validate_intervention_plan(
    plan_path: Path,
    public_study_path: Path,
    gate3_analysis_path: Path,
    *,
    private_study_path: Path | None = None,
) -> dict:
    plan = json.loads(plan_path.read_text())
    public = json.loads(public_study_path.read_text())
    analysis = json.loads(gate3_analysis_path.read_text())
    if plan["schema_version"] != "1.0":
        raise ValueError("unsupported intervention plan schema")
    if plan["status"] != "prospective-discovery-calibration-freeze":
        raise ValueError("intervention plan is not prospective")
    if plan["study_id"] != public["study_id"] or plan["study_id"] != analysis["study_id"]:
        raise ValueError("study ID mismatch")

    bindings = plan["source_bindings"]
    if _require_sha256(
        bindings["public_study_plan_sha256"], "public study plan hash"
    ) != sha256_file(public_study_path):
        raise ValueError("public study plan hash mismatch")
    if _require_sha256(
        bindings["gate3_analysis_sha256"], "Gate-3 analysis hash"
    ) != sha256_file(gate3_analysis_path):
        raise ValueError("Gate-3 analysis hash mismatch")
    if bindings["gate3_internal_artifact_sha256"] != analysis["gate3_artifact_sha256"]:
        raise ValueError("Gate-3 internal artifact hash mismatch")
    if bindings["sae_sha256"] != analysis["sae_sha256"]:
        raise ValueError("SAE hash mismatch")
    if private_study_path is not None:
        if _require_sha256(
            bindings["private_study_plan_sha256"], "private study plan hash"
        ) != sha256_file(private_study_path):
            raise ValueError("private study plan hash mismatch")

    features = plan["features"]
    selected = [int(value) for value in analysis["sae_discovery"]["selected_feature_ids"]]
    declared = [features["primary_feature_id"], *features["secondary_feature_ids"]]
    if declared != selected:
        raise ValueError("candidate feature IDs drift from Gate-3 selection")
    controls = [
        int(item["feature_id"]) for item in analysis["sae_discovery"]["matched_controls"]
    ]
    if features["matched_control_feature_ids"] != controls:
        raise ValueError("matched-control IDs drift from Gate-3 selection")
    if set(declared) & set(controls):
        raise ValueError("selected and control features overlap")

    split_counts: dict[str, int] = {}
    split_ids: dict[str, set[str]] = {}
    for behavior in public["behaviors"]:
        split = behavior["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        split_ids.setdefault(split, set()).add(behavior["behavior_id"])
    expected_counts = {"discovery": 20, "confirmatory": 40, "utility": 40}
    for split, count in expected_counts.items():
        if split_counts.get(split) != count:
            raise ValueError(f"{split} count mismatch")
    if split_ids["discovery"] & split_ids["confirmatory"]:
        raise ValueError("discovery and confirmatory behavior IDs overlap")

    calibration = plan["discovery_alpha_calibration"]
    rho = calibration["rho_ladder"]
    if rho != sorted(rho) or len(rho) != len(set(rho)) or any(value <= 0 for value in rho):
        raise ValueError("rho ladder must be unique, positive, and ascending")
    if calibration["behavior_count"] != split_counts["discovery"]:
        raise ValueError("discovery calibration topology mismatch")

    confirmatory = plan["confirmatory_attack_panel"]
    conditions = confirmatory["conditions_per_behavior"]
    if len(conditions) != len(set(conditions)):
        raise ValueError("duplicate confirmatory condition")
    if confirmatory["expected_generations"] != len(conditions) * split_counts["confirmatory"]:
        raise ValueError("confirmatory generation topology mismatch")
    if confirmatory["maximum_missing_trials"] != 0:
        raise ValueError("confirmatory plan must fail on missing trials")

    utility = plan["utility_panel"]
    utility_conditions = utility["conditions_per_behavior"]
    if utility["expected_generations"] != len(utility_conditions) * split_counts["utility"]:
        raise ValueError("utility generation topology mismatch")
    if not 0 <= utility["noninferiority_margin_absolute_refusal_rate_increase"] <= 1:
        raise ValueError("invalid utility noninferiority margin")

    receipt = plan["receipt_requirements"]
    required_runtime = set(receipt["required_runtime_fields"])
    mandatory = {
        "direction_sha256",
        "requested_sign",
        "requested_alpha",
        "realized_delta_norm_per_step",
        "pre_and_post_residual_norm_per_step",
        "layer",
        "hook",
        "token_position_per_step",
        "dtype",
        "clipping",
    }
    if not mandatory <= required_runtime:
        raise ValueError("intervention receipt runtime fields are incomplete")
    if plan["compute_policy"]["task_owned_pods"] != 1:
        raise ValueError("exactly one task-owned pod must be allowed")
    if plan["compute_policy"]["soft_gate_usd"] >= plan["compute_policy"]["hard_ceiling_usd"]:
        raise ValueError("soft gate must be below hard ceiling")

    return {
        "status": "valid",
        "plan_sha256": sha256_file(plan_path),
        "selected_feature_ids": declared,
        "matched_control_feature_ids": controls,
        "confirmatory_generations": confirmatory["expected_generations"],
        "utility_generations": utility["expected_generations"],
    }
