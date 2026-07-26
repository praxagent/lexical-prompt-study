from pathlib import Path

import pytest

from lexical_prompt_study.followup_plan import (
    load_followup_plan,
    sha256_file,
    validate_followup_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "plans" / "followup_v2.public.json"


def test_followup_plan_passes_strict_validation() -> None:
    validate_followup_plan(load_followup_plan(PLAN_PATH))


def test_followup_plan_binds_unchanged_source_plan() -> None:
    plan = load_followup_plan(PLAN_PATH)
    source = ROOT / plan["partitions"]["source"]
    assert sha256_file(source) == plan["partitions"]["source_sha256"]


def test_followup_plan_binds_review_adjudication_and_replay_plan() -> None:
    plan = load_followup_plan(PLAN_PATH)
    review = plan["review_gate"]
    assert sha256_file(ROOT / review["adjudication"]) == review["adjudication_sha256"]
    replay = plan["llama33_four_arm_prerequisite"]
    assert sha256_file(ROOT / replay["plan"]) == replay["plan_sha256"]
    assert sha256_file(ROOT / replay["result"]) == replay["result_sha256"]


def test_followup_plan_rejects_outcome_contamination() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["outcome_status"] = "8b-outcomes-inspected"
    with pytest.raises(ValueError, match="outcome boundary"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_missing_patch_control() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["causal_localization"]["controls"].remove("no_op_hook")
    with pytest.raises(ValueError, match="patch controls incomplete"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_confirmation_leakage() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["replication"]["threshold_partition"] = "confirmatory"
    with pytest.raises(ValueError, match="threshold leakage"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_multiple_confirmatory_detectors() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["detectors"]["maximum_llama31_confirmatory_candidates"] = 2
    with pytest.raises(ValueError, match="multiplicity"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_missing_scaffold_ordering() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["levels"] = ["ep_before_request"]
    with pytest.raises(ValueError, match="placement levels"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_pooled_sae_readout() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["analysis"]["sae_readouts_separate"] = False
    with pytest.raises(ValueError, match="sae_readouts_separate"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_unequal_order_token_budgets() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["within_arm_matching"]["require_equal_prompt_token_count"] = False
    with pytest.raises(ValueError, match="require_equal_prompt_token_count"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_component_tokenization_drift_permission() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["within_arm_matching"][
        "require_equal_component_token_subsequences"
    ] = False
    with pytest.raises(ValueError, match="component_token_subsequences"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_ambiguous_behavioral_family() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["behavioral_family"]["discovery"]["status"] = "formal_test"
    with pytest.raises(ValueError, match="behavioral discovery"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_mutable_candidate_ties() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["placement_factor"]["candidate_rule"]["threshold_ties"] = []
    with pytest.raises(ValueError, match="threshold tie"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_pooled_or_missing_detector_ordering() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["detectors"]["positive_strata"] = ["full"]
    with pytest.raises(ValueError, match="confirmatory detector positive-strata"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_missing_order_specific_negative_stratum() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["detectors"]["negative_strata"].remove("inert_length:ep_after_request")
    with pytest.raises(ValueError, match="confirmatory detector negative-strata"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_non_boundary_confirmation() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["causal_localization"]["only_confirmatory_position"] = "generated_index_1"
    with pytest.raises(ValueError, match="non-boundary"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_causal_test_without_holm() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["causal_localization"]["confirmatory_rule"]["require_holm_rejection"] = False
    with pytest.raises(ValueError, match="Holm"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_detector_complete_case_missingness() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["detectors"]["confirmatory_success"]["missingness"][
        "unevaluable_negative"
    ] = "drop"
    with pytest.raises(ValueError, match="detector missingness"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_noninvalidating_positive_control_failure() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["causal_localization"]["safe_positive_control"]["failure_disposition"] = "continue"
    with pytest.raises(ValueError, match="positive-control"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_mutable_four_arm_retirement() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["llama33_four_arm_prerequisite"]["machine_enforced"] = False
    with pytest.raises(ValueError, match="four-arm"):
        validate_followup_plan(plan)


def test_followup_plan_rejects_unpriced_gpu_fallback() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["compute"]["qualification"]["automatic_fallback"] = True
    with pytest.raises(ValueError, match="fallback"):
        validate_followup_plan(plan)


def test_followup_plan_keeps_qwen397b_deferred() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["qwen_pilot"]["qwen397b"] = "included"
    with pytest.raises(ValueError, match="397B"):
        validate_followup_plan(plan)
