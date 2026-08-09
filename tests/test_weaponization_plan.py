from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lexical_prompt_study.weaponization_plan import (
    load_weaponization_plan,
    validate_weaponization_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "plans" / "weaponization_breaker_v1.public.json"


def _plan() -> dict:
    return load_weaponization_plan(PLAN_PATH)


def test_weaponization_plan_passes_and_binds_predecessors() -> None:
    validate_weaponization_plan(_plan(), root=ROOT)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("authorization", "paid_compute_authorized_by_this_file"), True, "authorization"),
        (("restricted_data_boundary", "agent_plaintext_inspection"), True, "restricted-data"),
        (("factors", "placements_never_pooled"), False, "matching"),
        (("factors", "sizes_never_pooled"), False, "matching"),
        (
            ("constructs", "harmless_scaffold_panel", "minimum_semantic_families"),
            2,
            "harmless scaffold",
        ),
        (
            ("partitions", "detector_confirmation", "critical_negatives_per_stratum_per_placement"),
            299,
            "confirmation partition",
        ),
        (
            ("detector_gates", "confirmation", "maximum_false_positives_each_300_item_critical_negative_stratum"),
            1,
            "confirmation gate",
        ),
        (("breaker", "production_deployment_authorized"), True, "breaker safety"),
        (("breaker", "maximum_benign_cancellations"), 1, "breaker safety"),
        (
            ("fine_causal_localization", "discovery_pass", "both_directions_required"),
            False,
            "causal-localization",
        ),
        (("size_robustness", "threshold_refit_by_size_forbidden"), False, "size-robustness"),
    ],
)
def test_weaponization_plan_rejects_safety_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    plan = copy.deepcopy(_plan())
    target = plan
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_weaponization_plan(plan)


def test_weaponization_plan_rejects_predecessor_drift() -> None:
    plan = _plan()
    plan["predecessors"]["factorial_result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor"):
        validate_weaponization_plan(plan, root=ROOT)
