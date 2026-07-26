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


def test_factorial_plan_rejects_prior_calibration_reuse() -> None:
    plan = _plan()
    plan["threshold_program"][
        "reuse_of_feature_selection_or_prior_candidate_calibration_as_threshold_training"
    ] = True
    with pytest.raises(ValueError, match="threshold"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_confirmation_reopening() -> None:
    plan = _plan()
    plan["threshold_program"]["confirmation"][
        "existing_harmful_panel_remains_unopened_until_threshold_freeze"
    ] = False
    with pytest.raises(ValueError, match="confirmation"):
        validate_factorial_plan(plan)


def test_factorial_plan_requires_reconstruction_error_preservation() -> None:
    plan = _plan()
    plan["fine_causal_program"]["feature_edit"] = "replace residual with SAE decode"
    with pytest.raises(ValueError, match="causal"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_qwen_sae_claim() -> None:
    plan = _plan()
    plan["qwen_transfer"]["sae_claim"] = True
    with pytest.raises(ValueError, match="Qwen"):
        validate_factorial_plan(plan)


def test_factorial_plan_rejects_automatic_breaker() -> None:
    plan = _plan()
    plan["deployment"]["automatic_termination_not_authorized"] = False
    with pytest.raises(ValueError, match="deployment"):
        validate_factorial_plan(plan)


def test_factorial_mutations_do_not_change_source_fixture() -> None:
    original = _plan()
    mutated = copy.deepcopy(original)
    mutated["size_factor"]["dose_fractions"] = [1.0]
    assert original["size_factor"]["dose_fractions"] == [0.25, 0.5, 0.75, 1.0]
