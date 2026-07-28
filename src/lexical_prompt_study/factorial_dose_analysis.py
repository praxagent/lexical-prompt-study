from __future__ import annotations

from typing import Any


REQUEST_CLASSES = [
    "minimal_neutral_carrier",
    "ordinary_benign_request",
    "harmful_request",
]
MATERIALS = ["inert_length", "structural_sham", "full_scaffold"]
PLACEMENTS = ["ep_before_request", "ep_after_request"]
SIZE_IDS = ["blocks-001", "blocks-002", "blocks-003", "blocks-004"]
TOKEN_COUNTS = [64, 128, 188, 252]
CLAIM_LIMITS = {
    "secondary_descriptive_not_confirmatory",
    "fixed_model_sae_materials_requests_and_assistant_boundary_only",
    "no_harmful_request_detector_claim",
    "no_request_independence_claim",
    "no_monotonic_or_semantic_component_claim",
    "no_causal_mechanism_claim",
    "no_threshold_fit_or_deployment_claim",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_factorial_dose_analysis_plan(plan: dict[str, Any]) -> None:
    _require(plan["schema_version"] == "1.0", "dose-analysis schema drift")
    _require(
        plan["study_id"] == "lexical-scaffold-8b-factorial-v1"
        and plan["analysis_id"] == "factorial-8b-secondary-dose-v1"
        and plan["status"] == "prospective_before_secondary_dose_outcomes",
        "dose-analysis identity or prospective status drift",
    )
    bindings = plan["bindings"]
    for field in (
        "public_plan_sha256",
        "private_plan_sha256",
        "canonical_result_sha256",
        "canonical_execution_receipt_sha256",
        "canonical_matrix_receipt_manifest_sha256",
        "dose_authorization_sha256",
        "dose_observation_manifest_sha256",
    ):
        _require(_is_sha256(bindings[field]), f"dose-analysis {field} drift")
    _require(
        bindings["dose_execution_receipt_sha256"] is None,
        "dose outcome was bound before prospective plan freeze",
    )
    inputs = plan["inputs"]
    _require(
        inputs["new_partial_dose_receipts"] == 540
        and inputs["reused_canonical_receipts"] == 180
        and inputs["canonical_regeneration_forbidden"] is True
        and inputs["literal_sentinels_excluded"] is True
        and inputs["held_out_confirmation_excluded"] is True
        and inputs["raw_prompt_or_generation_text_opening_forbidden"] is True,
        "dose-analysis input boundary drift",
    )
    factors = plan["factors"]
    _require(
        factors["request_classes"] == REQUEST_CLASSES
        and factors["materials"] == MATERIALS
        and factors["placements"] == PLACEMENTS
        and factors["size_ids"] == SIZE_IDS
        and factors["injected_token_counts"] == TOKEN_COUNTS
        and factors["independent_unit"] == "prompt_family_id"
        and factors["expected_units_per_cell"] == 10
        and factors["placement_pooling"] is False
        and factors["size_pooling"] is False,
        "dose-analysis factor topology drift",
    )
    readouts = plan["readouts"]
    _require(
        readouts["primary"] == "feature_6779_magnitude"
        and readouts["strict_positive_prevalence"] == "descriptive_only"
        and readouts["secondary"]
        == [
            "frozen_subspace_score",
            "assistant_boundary_jlens_margin",
            "refusal_detected",
        ],
        "dose-analysis readout drift",
    )
    contrasts = plan["paired_contrasts_at_each_request_class_placement_and_size"]
    _require(
        contrasts
        == [
            {
                "name": "full_content_increment",
                "left": "full_scaffold",
                "right": "structural_sham",
            },
            {
                "name": "sham_over_inert_increment",
                "left": "structural_sham",
                "right": "inert_length",
            },
        ],
        "dose-analysis paired contrast drift",
    )
    uncertainty = plan["uncertainty"]
    _require(
        uncertainty["method"] == "paired_prompt_family_nonparametric_bootstrap"
        and uncertainty["replicates"] == 10000
        and uncertainty["master_seed"] == 20260728
        and uncertainty["interval"] == "two_sided_95_percent_percentile"
        and uncertainty["simultaneous_or_familywise_claim"] is False
        and uncertainty["p_values"] is False
        and uncertainty["formal_pass_fail_decision"] is False,
        "dose-analysis uncertainty drift",
    )
    shape = plan["dose_shape_policy"]
    _require(
        shape["plot_all_four_realized_token_counts"] is True
        and shape["monotonicity_test"] is False
        and shape["monotonicity_claim_forbidden"] is True
        and shape["linear_dose_response_claim_forbidden"] is True
        and shape["semantic_component_ablation_claim_forbidden"] is True,
        "dose-shape claim boundary drift",
    )
    missingness = plan["missingness"]
    _require(
        missingness["two_deterministic_attempts_per_new_observation"] is True
        and missingness["complete_case_substitution"] is False
        and missingness["imputation"] is False,
        "dose-analysis missingness drift",
    )
    _require(
        set(plan["claim_limits"]) == CLAIM_LIMITS,
        "dose-analysis claim limits drift",
    )
