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


def test_followup_plan_rejects_outcome_contamination() -> None:
    plan = load_followup_plan(PLAN_PATH)
    plan["outcome_status"] = "outcomes-inspected"
    with pytest.raises(ValueError, match="outcome-free"):
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
