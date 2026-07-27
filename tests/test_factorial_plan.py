from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from lexical_prompt_study.factorial_plan import (
    load_factorial_plan,
    validate_factorial_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "plans" / "factorial_8b_v1.public.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return load_factorial_plan(PLAN_PATH)


def test_factorial_plan_passes_strict_validation() -> None:
    validate_factorial_plan(_plan())


def test_factorial_plan_binds_predecessor_artifacts() -> None:
    plan = _plan()
    predecessor = plan["predecessor"]
    for path_field, hash_field in (
        ("followup_plan_path", "followup_plan_sha256"),
        ("g3_result_path", "g3_result_sha256"),
        ("g4_result_path", "g4_result_sha256"),
    ):
        assert _sha256(ROOT / predecessor[path_field]) == predecessor[hash_field]


def test_factorial_plan_cannot_authorize_compute() -> None:
    plan = _plan()
    plan["authorization"]["paid_compute_authorized_by_this_file"] = True
    with pytest.raises(ValueError, match="compute authorization"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_missing_request_level() -> None:
    plan = _plan()
    plan["request_factor"]["levels"].remove("minimal_neutral_carrier")
    with pytest.raises(ValueError, match="request factor"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_padded_base() -> None:
    plan = _plan()
    plan["size_factor"]["matching"]["no_padding_of_no_scaffold_reference"] = False
    with pytest.raises(ValueError, match="no_padding"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_unmatched_injection_size() -> None:
    plan = _plan()
    plan["size_factor"]["matching"][
        "exact_realized_injected_token_count_across_full_sham_and_inert"
    ] = False
    with pytest.raises(ValueError, match="exact_realized"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_midblock_truncation() -> None:
    plan = _plan()
    plan["size_factor"]["block_rule"] = "truncate at arbitrary token"
    with pytest.raises(ValueError, match="block-boundary"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_boundary_count_drift() -> None:
    plan = _plan()
    plan["size_factor"]["boundary_feasibility"][
        "selected_cumulative_token_counts"
    ][0] = 63
    with pytest.raises(ValueError, match="boundary-feasibility"):
        validate_factorial_plan(plan)


def test_factorial_plan_forbids_semantic_component_claim() -> None:
    plan = _plan()
    plan["size_factor"]["boundary_feasibility"][
        "semantic_component_ablation_claim_forbidden"
    ] = False
    with pytest.raises(ValueError, match="boundary-feasibility"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_size_pooling() -> None:
    plan = _plan()
    plan["size_factor"]["analysis"]["size_pooling_forbidden"] = False
    with pytest.raises(ValueError, match="size analysis"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_operational_zero_threshold() -> None:
    plan = _plan()
    plan["primary_estimands"]["strict_positive_prevalence_role"] = "detector"
    with pytest.raises(ValueError, match="prevalence"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_factorial_threshold_reuse() -> None:
    plan = _plan()
    plan["threshold_program"]["current_factorial_or_legacy_panels_may_train_threshold"] = (
        True
    )
    with pytest.raises(ValueError, match="threshold"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_confirmation_reopening() -> None:
    plan = _plan()
    plan["threshold_program"]["existing_harmful_confirmation_panel_remains_unopened"] = (
        False
    )
    with pytest.raises(ValueError, match="threshold"):
        validate_factorial_plan(plan)


def test_factorial_plan_requires_reconstruction_error_preservation() -> None:
    plan = _plan()
    plan["fine_causal_program"]["feature_edit"] = "replace residual with SAE decode"
    with pytest.raises(ValueError, match="causal"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_qwen_sae_claim() -> None:
    plan = _plan()
    plan["qwen_joint_shift_replication"]["sae_claim"] = True
    with pytest.raises(ValueError, match="Qwen"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_automatic_breaker() -> None:
    plan = _plan()
    plan["deployment"]["automatic_termination_not_authorized"] = False
    with pytest.raises(ValueError, match="deployment"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_generic_length_claim() -> None:
    plan = _plan()
    components = plan["primary_estimands"][
        "paired_components_per_request_placement_and_size"
    ]
    components["length_effect"] = components.pop("inert_injection_increment")
    with pytest.raises(ValueError, match="nested contrast"):
        validate_factorial_plan(plan)


def test_factorial_plan_requires_duplicate_dose_merge() -> None:
    plan = _plan()
    plan["size_factor"]["duplicate_realized_prefix_rule"] = (
        "count each nominal fraction independently"
    )
    with pytest.raises(ValueError, match="block-boundary"):
        validate_factorial_plan(plan)


def test_factorial_plan_requires_prompt_family_independence() -> None:
    plan = _plan()
    plan["request_factor"]["one_request_per_prompt_family_within_each_request_level"] = (
        False
    )
    with pytest.raises(ValueError, match="request unit"):
        validate_factorial_plan(plan)


def test_factorial_plan_requires_assay_gate() -> None:
    plan = _plan()
    plan["assay_validity_gate"]["must_pass_before_canonical_target_generation"] = False
    with pytest.raises(ValueError, match="assay validity"):
        validate_factorial_plan(plan)


def test_factorial_plan_binds_subspace_weights_and_jlens_summary() -> None:
    plan = _plan()
    plan["pinned_artifacts"]["frozen_subspace_weights"][0] = 0.0
    with pytest.raises(ValueError, match="SAE candidate"):
        validate_factorial_plan(plan)

    plan = _plan()
    plan["core_readout_implementation"]["jacobian_lens_source_layer"] = 29
    with pytest.raises(ValueError, match="core readout"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_enforcing_shadow_mode() -> None:
    plan = _plan()
    plan["deployment"]["shadow_records_decision_without_cancelling_request"] = False
    with pytest.raises(ValueError, match="deployment"):
        validate_factorial_plan(plan)


def test_factorial_plan_binds_adversarial_review() -> None:
    plan = _plan()
    plan["review_provenance"]["review_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="review binding"):
        validate_factorial_plan(plan)


def test_factorial_plan_freezes_staged_condition_counts() -> None:
    plan = _plan()
    plan["staged_bill_of_materials"]["stage_D_secondary_dose"][
        "additional_unique_conditions"
    ] = 541
    with pytest.raises(ValueError, match="bill of materials"):
        validate_factorial_plan(plan)


def test_factorial_mutations_do_not_change_source_fixture() -> None:
    original = _plan()
    mutated = copy.deepcopy(original)
    mutated["size_factor"]["dose_fractions"] = [1.0]
    assert original["size_factor"]["dose_fractions"] == [0.25, 0.5, 0.75, 1.0]
