from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lexical_prompt_study.intervention_runner import (
    ResidualPostIntervention,
    _reference_norm,
    calibration_conditions,
)


ROOT = Path(__file__).parents[1]
PLAN = json.loads((ROOT / "plans/gate4_intervention_v1.public.json").read_text())


def test_production_calibration_conditions_match_frozen_topology() -> None:
    conditions = calibration_conditions(PLAN)
    assert len(conditions) == 9
    assert conditions[0].condition_id == "zero"
    assert conditions[0].sign == 0
    assert conditions[-1].condition_id == "primary_positive_rho_0p02"
    assert conditions[-1].rho == 0.02


def test_qualification_calibration_conditions_are_bounded() -> None:
    conditions = calibration_conditions(PLAN, max_rhos=1)
    assert [item.condition_id for item in conditions] == [
        "zero",
        "primary_negative_rho_0p0025",
        "primary_positive_rho_0p0025",
    ]
    with pytest.raises(ValueError, match="positive"):
        calibration_conditions(PLAN, max_rhos=0)


def test_reference_norm_uses_all_zero_intervention_events() -> None:
    receipts = [
        {
            "intervention_steps": [
                {"pre_residual_norm": 2.0},
                {"pre_residual_norm": 4.0},
            ]
        },
        {
            "intervention_steps": [
                {"pre_residual_norm": 6.0},
                {"pre_residual_norm": 8.0},
            ]
        },
    ]
    assert _reference_norm(receipts) == 5.0


def test_reference_norm_rejects_empty_or_invalid_events() -> None:
    with pytest.raises(ValueError, match="invalid"):
        _reference_norm([])
    with pytest.raises(ValueError, match="invalid"):
        _reference_norm([{"intervention_steps": [{"pre_residual_norm": 0.0}]}])


def test_residual_hook_modifies_only_current_final_token() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
        dtype=torch.float32,
    )
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hook = ResidualPostIntervention(torch, direction, sign=-1, alpha=0.1)
    modified, sentinel = hook(None, (), (hidden, "cache"))
    assert sentinel == "cache"
    assert torch.equal(modified[:, 0, :], hidden[:, 0, :])
    assert modified[0, -1, 0].item() == pytest.approx(3.9)
    assert torch.equal(modified[0, -1, 1:], hidden[0, -1, 1:])
    assert len(hook.steps) == 1
    assert hook.steps[0]["generated_token_index"] == 0
    assert hook.steps[0]["clipped"] is False


def test_bfloat16_hook_meets_frozen_effective_norm_tolerance() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(20260725)
    hidden = torch.randn(
        1,
        1,
        8192,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    direction = torch.randn(8192, generator=generator, dtype=torch.float32)
    direction = direction / direction.norm()
    pre_norm = float(hidden.float().norm())
    observed_errors = []
    for rho in (0.0025, 0.005, 0.01, 0.02):
        hook = ResidualPostIntervention(
            torch,
            direction,
            sign=1,
            alpha=rho * pre_norm,
        )
        hook(None, (), hidden)
        observed_errors.append(
            hook.steps[0]["requested_realized_relative_error"]
        )
    assert max(observed_errors) <= 0.1
    assert max(observed_errors) > 0.001
    assert np.all(np.isfinite(observed_errors))
