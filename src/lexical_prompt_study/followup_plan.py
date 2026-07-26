from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_STAGES = [
    "G0_local_integrity",
    "G0b_llama33_four_arm_sae_replay",
    "G1_one_unit_qualification",
    "G2_llama31_behavioral_viability",
    "G3_llama31_sparse_and_jlens_replication",
    "G4_coarse_causal_localization",
    "G5_component_causal_localization",
    "G6_detector_confirmation_and_shadow_mode",
    "G7_qwen_cross_family_pilot",
]

REQUIRED_ARMS = {"base", "inert_length", "structural_sham", "full"}
REQUIRED_PATCH_CONTROLS = {
    "full_into_full_identity",
    "sham_into_sham_identity",
    "no_op_hook",
    "same_site_magnitude_matched_seeded_random_delta",
    "irrelevant_layer",
    "irrelevant_token_position",
    "cross_behavior_category_and_length_matched_donor",
}
EXPECTED_PARTITION_COUNTS = {
    "discovery": 20,
    "calibration": 20,
    "confirmatory": 40,
    "adaptive_stress": 20,
    "utility_calibration": 50,
    "utility_confirmatory": 50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_followup_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_followup_plan(plan: dict[str, Any]) -> None:
    _require(plan["study_id"] == "lexical-scaffold-followup-v2", "wrong study namespace")
    _require(plan["outcome_status"] == "outcome-free", "plan must be outcome-free")
    _require(plan["stage_order"] == EXPECTED_STAGES, "stage order drift")

    attack = plan["attack_handling"]
    _require(attack["existing_artifact_only"] is True, "attack artifact must remain pinned")
    _require(
        attack["inspect_additional_attack_text"] is False,
        "additional attack text inspection is forbidden",
    )

    partitions = plan["partitions"]
    for name, count in EXPECTED_PARTITION_COUNTS.items():
        _require(partitions[name]["count"] == count, f"{name} count drift")
        _require(
            len(partitions[name]["sorted_id_list_sha256"]) == 64,
            f"{name} lacks exact ID-list hash",
        )

    replication = plan["replication"]
    _require(set(replication["arms"]) == REQUIRED_ARMS, "replication arm drift")
    _require(replication["feature_selection_partition"] == "discovery", "selection leakage")
    _require(replication["threshold_partition"] == "calibration", "threshold leakage")
    _require(replication["test_partition"] == "confirmatory", "test leakage")
    _require(replication["maximum_subspace_features"] <= 8, "subspace is not small")

    causal = plan["causal_localization"]
    _require(
        causal["primary_position"] == "turn2_assistant_boundary_before_first_generated_token",
        "primary patch position drift",
    )
    _require(
        causal["coarse_residual_post_layers"] == [0, 4, 8, 12, 16, 20, 24, 28, 31],
        "coarse layer grid drift",
    )
    _require(set(causal["controls"]) == REQUIRED_PATCH_CONTROLS, "patch controls incomplete")
    _require(
        causal["component_stage"]["maximum_confirmatory_components"] == 1,
        "confirmatory component multiplicity",
    )
    _require(
        causal["component_stage"]["head_localization"]
        == "deferred_pending_new_prospective_amendment",
        "head search must remain deferred",
    )
    replay = causal["private_replay_bundle"]
    _require(replay["retain_recipient_pre_patch_bf16"] is True, "recipient tensor not replayable")
    _require(replay["retain_realized_delta_bf16"] is True, "realized patch not replayable")

    qwen = plan["qwen_pilot"]
    _require(qwen["native_qwen_attack_claim"] is False, "Qwen native-attack overclaim")
    _require(qwen["sae_available"] is False, "Qwen SAE must not be implied")
    _require(qwen["no_silent_model_substitution"] is True, "Qwen substitution must fail closed")
    _require(qwen["qwen397b"] == "deferred_separate_authorization", "397B is not authorized")

    breaker = plan["circuit_breaker"]
    _require(breaker["requires_frozen_detector"] is True, "breaker precedes detector freeze")
    _require(breaker["mode"] == "shadow", "breaker must begin in shadow mode")
    _require(breaker["action"] == "cancel_request_not_server", "unsafe breaker action")
    _require(breaker["hide_internal_scores_and_thresholds"] is True, "detector leaks internals")

    raw = plan["raw_data_policy"]
    _require(not raw["public_raw_prompts"], "raw prompts cannot be public")
    _require(not raw["public_raw_generations"], "raw generations cannot be public")
    _require(not raw["public_reconstructive_token_ids"], "token IDs cannot be public")
    _require(not raw["public_replayable_tensors"], "replay tensors cannot be public")

    figures = plan["figure_contract"]
    _require(figures["programmatic_from_audited_receipts_only"] is True, "figure provenance gap")
    _require(set(figures["formats"]) == {"svg", "png", "pdf"}, "figure format drift")

    compute = plan["compute"]
    qualification = compute["qualification"]
    _require(compute["persistent_volume_id"] == "u85xfo0aue", "wrong task volume")
    _require(qualification["count"] == 1, "qualification must use exactly one GPU")
    _require(qualification["automatic_fallback"] is False, "unpriced GPU fallback forbidden")
    expected_max = (
        qualification["live_rate_usd_per_hour"] * qualification["wall_limit_minutes"] / 60
    )
    _require(
        abs(expected_max - qualification["max_compute_usd"]) < 1e-9,
        "qualification max cost arithmetic drift",
    )
    _require(
        compute["scientific_run_requires_measured_throughput"] is True,
        "scientific run lacks throughput gate",
    )
    _require(
        compute["scientific_run_requires_separate_exact_cost_statement"] is True,
        "scientific run lacks per-run cost gate",
    )
    _require(
        compute["prior_estimated_spend_usd"] + compute["incremental_hard_usd"]
        < compute["existing_cumulative_hard_ceiling_usd"],
        "campaign envelope exceeds cumulative hard ceiling",
    )
