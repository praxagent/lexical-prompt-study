from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.followup_design import PLACEMENTS
from lexical_prompt_study.followup_patch import (
    NEGATIVE_CONTROL_CONDITIONS,
    PRIMARY,
    RECIPROCAL,
    NoOpResidualHook,
    PatchUnit,
    ResidualStatePatch,
    analyze_coarse_patch_rows,
    magnitude_matched_random_deltas,
    select_cross_behavior_donors,
)


ROOT = Path(__file__).parents[1]
PLAN = json.loads((ROOT / "plans/followup_v2.public.json").read_text())


def test_cross_behavior_donor_is_category_and_length_matched() -> None:
    units = [
        PatchUnit("a", "cat-1", 100),
        PatchUnit("b", "cat-1", 103),
        PatchUnit("c", "cat-1", 130),
        PatchUnit("d", "cat-2", 101),
        PatchUnit("e", "cat-2", 102),
    ]
    donors = select_cross_behavior_donors(units)
    assert donors["a"] == "b"
    assert donors["b"] == "a"
    assert donors["d"] == "e"
    assert all(recipient != donor for recipient, donor in donors.items())


def test_residual_state_patch_runs_once_on_prefill() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    replacement = torch.tensor(
        [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]]
    )
    hook = ResidualStatePatch(torch, replacement=replacement)
    modified = hook(None, (), hidden)
    assert torch.equal(modified[:, :-1], hidden[:, :-1])
    assert torch.equal(modified[:, -1], replacement)
    assert hook.applied is True
    assert hook.replay is not None
    decode = torch.zeros(2, 1, 4)
    assert hook(None, (), decode) is decode


def test_noop_hook_clones_without_changing_prefill() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.randn(2, 3, 4)
    hook = NoOpResidualHook(torch, batch_size=2)
    modified = hook(None, (), hidden)
    assert modified is not hidden
    assert torch.equal(modified, hidden)
    assert torch.count_nonzero(hook.replay["realized_delta"]) == 0


def test_random_deltas_match_each_reference_norm() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    random = magnitude_matched_random_deltas(
        torch,
        reference,
        seeds=[1, 2],
    )
    assert torch.allclose(random.norm(dim=1), reference.norm(dim=1), rtol=1e-6)
    assert not torch.equal(random, reference)


def _analysis_rows(*, make_primary_eligible: bool) -> list[dict]:
    rows = []
    conditions = PLAN["causal_localization"]["execution"]["condition_kinds"]
    layers = PLAN["causal_localization"]["coarse_residual_post_layers"]
    for placement in PLACEMENTS:
        for layer in layers:
            for condition in conditions:
                for index in range(20):
                    effect = 0.0
                    if layer == 8 and make_primary_eligible:
                        if condition == PRIMARY:
                            effect = -0.2
                        elif condition == RECIPROCAL:
                            effect = 0.2
                    if condition in NEGATIVE_CONTROL_CONDITIONS and index % 2:
                        effect = 0.01
                    if condition in NEGATIVE_CONTROL_CONDITIONS and index % 2 == 0:
                        effect = -0.01
                    rows.append(
                        {
                            "partition": "discovery",
                            "placement": placement,
                            "candidate_layer": layer,
                            "condition": condition,
                            "behavior_id": f"b{index:02d}",
                            "baseline_score": 0.5,
                            "patched_score": 0.5 + effect,
                        }
                    )
    return rows


def test_coarse_patch_analysis_selects_common_layer_without_pooling() -> None:
    result = analyze_coarse_patch_rows(
        _analysis_rows(make_primary_eligible=True),
        plan=PLAN,
        partition="discovery",
    )
    assert result["selected_common_layer"] == 8
    assert result["pooled_placement_estimate_reported"] is False
    assert set(result["ordering_results"]) == set(PLACEMENTS)


def test_coarse_patch_analysis_stops_when_no_layer_is_common() -> None:
    result = analyze_coarse_patch_rows(
        _analysis_rows(make_primary_eligible=False),
        plan=PLAN,
        partition="discovery",
    )
    assert result["status"] == "stopped_no_eligible_layer"
    assert result["selected_common_layer"] is None
