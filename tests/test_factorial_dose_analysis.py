from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lexical_prompt_study.factorial_dose_analysis import (
    _percentile_interval,
    _validate_execution_receipt,
    validate_factorial_dose_analysis_plan,
)
from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "plans" / "factorial_8b_v1.dose-analysis.json"
EXECUTION_PATH = (
    ROOT / "validation" / "factorial_secondary_dose_a065.execution-receipt.json"
)


def test_secondary_dose_analysis_plan_is_prospective_and_exact() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    validate_factorial_dose_analysis_plan(plan)
    assert plan["bindings"]["public_plan_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_8b_v1.public.json"
    )
    assert plan["bindings"]["private_plan_sha256"] == sha256_file(
        ROOT / "private" / "plans" / "factorial_8b_v1.private.json"
    )
    assert plan["bindings"]["canonical_result_sha256"] == sha256_file(
        ROOT / "results" / "factorial-8b-canonical.public.json"
    )
    assert plan["bindings"]["canonical_execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "factorial_8b_v1.execution-receipt.json"
    )
    assert plan["bindings"]["dose_authorization_sha256"] == sha256_file(
        ROOT / "plans" / "factorial_secondary_dose_a065.authorization.json"
    )
    assert plan["bindings"]["dose_execution_receipt_sha256"] is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("factors", "placement_pooling"), True, "factor"),
        (("factors", "size_pooling"), True, "factor"),
        (("uncertainty", "p_values"), True, "uncertainty"),
        (("dose_shape_policy", "monotonicity_test"), True, "dose-shape"),
        (
            ("inputs", "held_out_confirmation_excluded"),
            False,
            "input boundary",
        ),
    ],
)
def test_secondary_dose_analysis_plan_rejects_scope_drift(
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    plan = json.loads(PLAN_PATH.read_text())
    mutated = copy.deepcopy(plan)
    mutated[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_factorial_dose_analysis_plan(mutated)


def test_secondary_dose_execution_receipt_is_bound_before_analysis() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    execution = json.loads(EXECUTION_PATH.read_text())
    _validate_execution_receipt(
        execution=execution,
        execution_path=EXECUTION_PATH,
        plan=plan,
        public_plan_path=ROOT / "plans" / "factorial_8b_v1.public.json",
        private_plan_path=ROOT / "private" / "plans" / "factorial_8b_v1.private.json",
        authorization_path=(
            ROOT / "plans" / "factorial_secondary_dose_a065.authorization.json"
        ),
        summary_path=(
            ROOT
            / "private"
            / "checkpoints"
            / "factorial-secondary-dose-a065"
            / "summary.json"
        ),
    )
    assert execution["analysis_boundaries_at_binding"][
        "receipt_outcome_analysis_started"
    ] is False
    assert execution["analysis_boundaries_at_binding"][
        "held_out_confirmation_opened"
    ] is False


def test_secondary_dose_bootstrap_stream_is_deterministic() -> None:
    values = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    first = _percentile_interval(values, replicates=250, seed=20260728)
    second = _percentile_interval(values, replicates=250, seed=20260728)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
