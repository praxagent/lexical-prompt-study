from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_STAGES = [
    "F0_private_input_and_tokenization_freeze",
    "F1_synthetic_topology_and_receipt_qualification",
    "F2_canonical_factorial_and_size_dose_generation",
    "F3_behavioral_and_internal_readout_analysis",
    "F4_feature_interpretation",
    "F5_new_threshold_calibration",
    "F6_unopened_confirmation",
    "F7_fine_causal_localization",
    "F8_small_qwen_transfer",
    "F9_shadow_mode_only",
]
REQUEST_LEVELS = {
    "minimal_neutral_carrier",
    "ordinary_benign_request",
    "harmful_request",
}
SCAFFOLD_LEVELS = {
    "no_scaffold",
    "inert_length",
    "structural_sham",
    "full_scaffold",
}
PLACEMENT_LEVELS = {
    "scaffold_before_request",
    "scaffold_after_request",
}
DOSE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
FROZEN_SUBSPACE = [1980, 6779, 11954, 20449, 35705, 43596, 53185, 58843]


def load_factorial_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_factorial_plan(plan: dict[str, Any]) -> None:
    _require(plan["schema_version"] == "1.0", "factorial schema drift")
    _require(
        plan["study_id"] == "lexical-scaffold-8b-factorial-v1",
        "factorial study namespace drift",
    )
    _require(plan["amendment"] == "A051", "factorial amendment drift")
    _require(
        plan["status"] == "prospectively_frozen_public_design_no_new_target_outcomes",
        "factorial outcome boundary drift",
    )
    _require(plan["stage_order"] == EXPECTED_STAGES, "factorial stage order drift")

    authorization = plan["authorization"]
    _require(
        authorization["scientific_scope_approved"] is True
        and authorization["paid_compute_authorized_by_this_file"] is False
        and authorization[
            "compute_requires_separate_exact_source_rate_wall_time_and_spend_amendment"
        ]
        is True,
        "factorial compute authorization drift",
    )
    _require(
        authorization["soft_gate_usd"] == 100
        and authorization["hard_ceiling_usd"] == 200
        and authorization["single_task_owned_pod_maximum"] == 1
        and authorization["persistent_volume_id"] == "u85xfo0aue",
        "factorial resource gate drift",
    )

    artifacts = plan["pinned_artifacts"]
    _require(artifacts["selected_feature_id"] == 6779, "selected feature drift")
    _require(
        artifacts["sae_layer"] == 19
        and artifacts["frozen_subspace_feature_ids"] == FROZEN_SUBSPACE,
        "SAE candidate drift",
    )
    _require(
        len(artifacts["model_revision"]) == 40,
        "missing pinned artifact: model_revision",
    )
    for field in ("sae_sha256", "jacobian_lens_sha256"):
        _require(len(artifacts[field]) == 64, f"missing pinned artifact: {field}")

    handling = plan["attack_handling"]
    _require(
        handling["existing_restricted_scaffold_only"] is True
        and handling["agent_plaintext_inspection"] is False
        and handling["public_plaintext_or_reconstructive_tokens"] is False,
        "restricted attack boundary drift",
    )

    request = plan["request_factor"]
    _require(set(request["levels"]) == REQUEST_LEVELS, "request factor drift")
    _require(
        request["canonical_matrix_items_per_level"] == 20
        and request["independent_unit"] == "request_id",
        "request unit/count drift",
    )
    _require(
        request["minimal_neutral_carrier"]["varied_panel_required"] is True
        and request["minimal_neutral_carrier"][
            "descriptive_literal_sentinel_per_placement"
        ]
        == 1
        and request["minimal_neutral_carrier"]["literal_sentinel_role"]
        == "descriptive_n_equals_1_only",
        "neutral-carrier independence drift",
    )

    scaffold = plan["scaffold_factor"]
    _require(set(scaffold["levels"]) == SCAFFOLD_LEVELS, "scaffold factor drift")
    _require(
        set(scaffold["placement_levels"]) == PLACEMENT_LEVELS,
        "placement factor drift",
    )
    _require(
        set(scaffold["placement_applies_to"])
        == {"inert_length", "structural_sham", "full_scaffold"}
        and scaffold["shared_reference"] == "no_scaffold"
        and scaffold["shared_reference_not_duplicated_or_double_counted"] is True,
        "partially crossed topology drift",
    )

    size = plan["size_factor"]
    _require(size["dose_fractions"] == DOSE_FRACTIONS, "size levels drift")
    _require(
        size["canonical_size_fraction"] == 1.0
        and size["dose_panel_items_per_request_level"] == 10,
        "size panel drift",
    )
    _require(
        "complete restricted scaffold blocks" in size["block_rule"]
        and size["minimum_nonzero_block_count"] == 1,
        "size block-boundary rule drift",
    )
    matching = size["matching"]
    for field in (
        "exact_realized_injected_token_count_across_full_sham_and_inert",
        "same_request_bytes",
        "same_placement",
        "same_separator_and_special_token_sequences",
        "same_component_offsets_relative_to_request",
        "same_context_and_generation_budgets",
        "no_request_trimming",
        "no_padding_of_no_scaffold_reference",
    ):
        _require(matching[field] is True, f"size matching drift: {field}")
    _require(
        matching["mismatch_disposition"] == "stop_before_target_generation",
        "size mismatch is not fail closed",
    )
    size_analysis = size["analysis"]
    _require(
        size_analysis["all_sizes_reported_separately"] is True
        and size_analysis["placement_pooling_forbidden"] is True
        and size_analysis["size_pooling_forbidden"] is True
        and size_analysis["canonical_size_is_primary"] is True
        and size_analysis["monotonicity_claim_forbidden_without_new_prespecified_test"]
        is True,
        "size analysis drift",
    )

    estimands = plan["primary_estimands"]
    _require(
        estimands["strict_positive_prevalence_role"]
        == "descriptive_only_not_an_operational_threshold",
        "positive prevalence overread",
    )
    _require(
        estimands["paired_components_per_request_placement_and_size"]
        == {
            "length_effect": "inert_length - no_scaffold",
            "structural_effect": "structural_sham - inert_length",
            "full_content_increment": "full_scaffold - structural_sham",
        },
        "nested contrast drift",
    )
    separation = estimands["analysis_separation"]
    _require(
        separation["placements_separate"] is True
        and separation["request_levels_separate"] is True
        and separation["sizes_separate"] is True
        and separation["pooled_primary_estimate_forbidden"] is True,
        "factorial pooling drift",
    )
    _require(
        estimands["uncertainty"]["resampling_unit"] == "request_id",
        "wrong independent unit",
    )

    readouts = plan["readouts"]
    _require(
        "identity_transport_comparator" in readouts["every_canonical_cell"]
        and "frobenius_matched_seeded_random_transport_comparator"
        in readouts["every_canonical_cell"]
        and readouts["harmbench_on_benign_or_neutral"] is False,
        "readout/control drift",
    )
    _require(
        readouts["selected_feature_position_maps"]["generated_token_indices"]
        == [0, 1, 2, 4, 8, 16],
        "generated position drift",
    )

    interpretation = plan["feature_interpretation"]
    _require(
        interpretation["status"]
        == "exploratory_and_firewalled_from_candidate_or_threshold_changes"
        and interpretation["may_change_feature_id_or_threshold_rule"] is False,
        "interpretation leakage",
    )

    threshold = plan["threshold_program"]
    _require(
        threshold["candidate"] == "single_feature_6779"
        and threshold["candidate_reselection_forbidden"] is True
        and threshold["new_calibration_required"] is True
        and threshold[
            "reuse_of_feature_selection_or_prior_candidate_calibration_as_threshold_training"
        ]
        is False
        and threshold["common_threshold_across_placements"] is True,
        "threshold freeze drift",
    )
    _require(
        threshold["confirmation"][
            "existing_harmful_panel_remains_unopened_until_threshold_freeze"
        ]
        is True
        and threshold["confirmation"]["one_evaluation_only"] is True
        and threshold["confirmation"]["placements_reported_separately"] is True,
        "confirmation leakage",
    )

    causal = plan["fine_causal_program"]
    _require(
        causal["requires_separate_executable_pre_outcome_amendment"] is True
        and "preserving the original SAE reconstruction error"
        in causal["feature_edit"]
        and causal["placements_separate"] is True,
        "fine causal gate drift",
    )
    _require(
        "sae_reconstruction_error_preserved_noop" in causal["controls"],
        "SAE reconstruction control missing",
    )

    qwen = plan["qwen_transfer"]
    _require(
        qwen["small_checkpoint_only"] is True
        and qwen["sae_claim"] is False
        and qwen["qwen_397b_deferred"] is True,
        "Qwen scope drift",
    )

    deployment = plan["deployment"]
    _require(
        deployment["automatic_termination_not_authorized"] is True
        and deployment["first_mode"] == "shadow"
        and deployment["request_cancel_not_server_termination"] is True
        and deployment["large_disjoint_benign_shadow_panel_required"] is True,
        "deployment gate drift",
    )
    _require(
        set(deployment["cheap_input_baselines_required"])
        == {"exact_hash", "n_gram_or_fuzzy_lexical", "structural_parser"},
        "cheap detector baseline drift",
    )

    raw = plan["raw_data_policy"]
    for field in (
        "public_raw_prompts",
        "public_raw_generations",
        "public_reconstructive_token_ids",
        "public_replayable_tensors",
    ):
        _require(raw[field] is False, f"raw public policy drift: {field}")
    _require(
        raw["private_mode_0600_receipts_and_resumable_checkpoints"] is True,
        "private checkpoint policy drift",
    )
