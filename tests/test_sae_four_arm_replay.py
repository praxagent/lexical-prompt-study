import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lexical_prompt_study.sae_four_arm_replay import (
    ARMS,
    compute_four_arm_replay,
    validate_replay_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "plans" / "gate3_sae_four_arm_replay_v1.public.json").read_text())


def _fixture():
    observations = []
    states = []
    arm_offset = {"base": 0.0, "inert_length": 0.2, "structural_sham": 0.0, "full": 1.0}
    for behavior_index in range(20):
        for arm in ARMS:
            observations.append(
                {
                    "observation_id": f"{behavior_index}-{arm}",
                    "behavior_id": f"b{behavior_index:02d}",
                    "arm": arm,
                    "position": "assistant_boundary",
                    "position_available": True,
                }
            )
            states.append([arm_offset[arm] + behavior_index / 100, 0.0])
    return np.asarray(states), observations


def test_replay_plan_passes_strict_validation() -> None:
    validate_replay_plan(PLAN)


def test_replay_plan_rejects_new_forward_pass() -> None:
    plan = copy.deepcopy(PLAN)
    plan["compute"]["requires_new_70b_forward_pass"] = True
    with pytest.raises(ValueError, match="forward pass"):
        validate_replay_plan(plan)


def test_four_arm_replay_preserves_pairing_and_summarizes_prevalence() -> None:
    states, observations = _fixture()
    public, private = compute_four_arm_replay(
        layer_states=states,
        observations=observations,
        encoder_rows=np.asarray([[1.0, 0.0]], dtype=np.float32),
        encoder_bias=np.asarray([-0.5], dtype=np.float32),
        feature_ids=[10146],
        bootstrap_seed=20260726,
        bootstrap_replicates=100,
    )
    feature = public["features"][0]
    assert feature["arms"]["base"]["positive_count"] == 0
    assert feature["arms"]["inert_length"]["positive_count"] == 0
    assert feature["arms"]["structural_sham"]["positive_count"] == 0
    assert feature["arms"]["full"]["positive_count"] == 20
    assert len(private["rows"]) == 80


def test_four_arm_replay_rejects_unaligned_behavior_ids() -> None:
    states, observations = _fixture()
    observations[-1]["behavior_id"] = "wrong"
    with pytest.raises(ValueError, match="alignment"):
        compute_four_arm_replay(
            layer_states=states,
            observations=observations,
            encoder_rows=np.asarray([[1.0, 0.0]], dtype=np.float32),
            encoder_bias=np.asarray([-0.5], dtype=np.float32),
            feature_ids=[10146],
            bootstrap_seed=20260726,
            bootstrap_replicates=10,
        )
