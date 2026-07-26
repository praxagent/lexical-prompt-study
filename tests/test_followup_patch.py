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
from lexical_prompt_study.followup_patch_runner import (
    load_frozen_safe_positive_control,
    projected_discovery_maximum_tokens,
    validate_patch_run_authorization,
)
from lexical_prompt_study.hashing import sha256_file, write_json_atomic


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
    layers = PLAN["causal_localization"]["instrument_strength_calibration"][
        "target_candidate_layers"
    ]
    for placement in PLACEMENTS:
        for layer in layers:
            for condition in conditions:
                for index in range(20):
                    effect = 0.0
                    if layer == 20 and make_primary_eligible:
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
    assert result["selected_common_layer"] == 20
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


def test_patch_qualification_requires_exact_prospective_bindings() -> None:
    private_sha = "b" * 64
    scientific_sha = "a" * 64
    source_commit = "c" * 40
    plan = {
        "compute": {
            "scientific_runs": {
                "g4_patch_qualification": {
                    "status": "safe_only_authorized_target_closed",
                    "runner_source_commit": source_commit,
                    "partition": "discovery",
                    "qualification_only": True,
                    "run_id": "safe-run",
                    "target_generation_authorized": False,
                    "input_binding": {
                        "patch_scientific_plan_sha256": scientific_sha,
                        "patch_private_plan_sha256": private_sha,
                        "patch_private_scientific_plan_sha256": scientific_sha,
                    },
                }
            }
        }
    }
    private = {"public_plan_sha256": scientific_sha}
    authorization = validate_patch_run_authorization(
        plan=plan,
        patch_private_plan=private,
        patch_private_plan_sha256=private_sha,
        source_commit=source_commit,
        partition="discovery",
        qualification_only=True,
        run_id="safe-run",
    )
    assert authorization["target_generation_authorized"] is False
    private["public_plan_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="authorization binding drift"):
        validate_patch_run_authorization(
            plan=plan,
            patch_private_plan=private,
            patch_private_plan_sha256=private_sha,
            source_commit=source_commit,
            partition="discovery",
            qualification_only=True,
            run_id="safe-run",
        )


def test_patch_target_cannot_use_safe_only_authorization() -> None:
    with pytest.raises(ValueError, match="g4_patch_discovery is not prospectively"):
        validate_patch_run_authorization(
            plan={"compute": {"scientific_runs": {}}},
            patch_private_plan={"public_plan_sha256": "a" * 64},
            patch_private_plan_sha256="b" * 64,
            source_commit="c" * 40,
            partition="discovery",
            qualification_only=False,
            run_id="target-run",
        )


def test_patch_throughput_qualification_resolves_by_exact_run_id() -> None:
    private_sha = "b" * 64
    private_input_sha = "a" * 64
    source_commit = "c" * 40
    run_id = "throughput-run"
    plan = {
        "compute": {
            "scientific_runs": {
                "g4_patch_qualification": {
                    "status": "safe_only_authorized_target_closed",
                    "qualification_only": True,
                    "run_id": "prior-safe-run",
                },
                "g4_patch_throughput_qualification": {
                    "status": "throughput_only_authorized_target_closed",
                    "runner_source_commit": source_commit,
                    "partition": "discovery",
                    "qualification_only": True,
                    "run_id": run_id,
                    "target_generation_authorized": False,
                    "input_binding": {
                        "patch_private_plan_sha256": private_sha,
                        "patch_private_input_plan_sha256": private_input_sha,
                    },
                },
            }
        }
    }
    authorization = validate_patch_run_authorization(
        plan=plan,
        patch_private_plan={"public_plan_sha256": private_input_sha},
        patch_private_plan_sha256=private_sha,
        source_commit=source_commit,
        partition="discovery",
        qualification_only=True,
        run_id=run_id,
    )
    assert authorization["status"] == "throughput_only_authorized_target_closed"


def test_frozen_safe_control_validates_layer_specific_instrument(
    tmp_path: Path,
) -> None:
    plan = json.loads(json.dumps(PLAN))
    instrument = plan["causal_localization"]["instrument_strength_calibration"]
    eligible = instrument["target_candidate_layers"]
    result = {
        "run_id": instrument["source_run_id"],
        "source_commit": instrument["source_commit"],
        "public_plan_sha256": instrument["source_public_plan_sha256"],
        "patch_private_plan_sha256": instrument["source_private_plan_sha256"],
        "pair_count": instrument["source_pair_count"],
        "raw_prompts_or_token_ids_public": False,
        "layers": [
            {
                "layer": layer,
                "gate_passed": layer in eligible,
                "identity_and_noop_passed": True,
            }
            for layer in plan["causal_localization"]["coarse_residual_post_layers"]
        ],
    }
    path = tmp_path / "safe.json"
    write_json_atomic(path, result)
    instrument["source_result_sha256"] = sha256_file(path)
    loaded = load_frozen_safe_positive_control(path=path, plan=plan)
    assert loaded["pair_count"] == 20
    result["layers"][0]["identity_and_noop_passed"] = False
    write_json_atomic(path, result)
    instrument["source_result_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="layer partition drift"):
        load_frozen_safe_positive_control(path=path, plan=plan)


def test_target_projection_uses_only_instrument_passing_layers() -> None:
    assert projected_discovery_maximum_tokens(PLAN) == 1_843_200
