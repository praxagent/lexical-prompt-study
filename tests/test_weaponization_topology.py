from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.weaponization_topology import _normalize_prompt_token_ids


ROOT = Path(__file__).resolve().parents[1]


def _receipt(name: str) -> dict:
    return json.loads((ROOT / "validation" / name).read_text())


def test_weaponization_calibration_topology_is_prefill_only_and_blocked() -> None:
    receipt = _receipt("weaponization-calibration-topology.reviewed.public.json")
    assert receipt["partition"] == "detector_calibration"
    assert receipt["observation_count"] == 1900
    assert receipt["stratum_counts"]["positive_attack_harmful"] == 200
    assert receipt["stratum_counts"]["negative_harmless_harmful"] == 200
    assert receipt["stratum_counts"]["shared_no_scaffold_harmful"] == 100
    assert receipt["prefill_only"] is True
    assert receipt["human_semantic_review_approved"] is True
    assert receipt["target_generation_authorized"] is False
    assert receipt["target_outcome_exists"] is False


def test_weaponization_confirmation_topology_has_300_families_per_placement() -> None:
    receipt = _receipt("weaponization-confirmation-topology.reviewed.public.json")
    assert receipt["partition"] == "detector_confirmation"
    assert receipt["observation_count"] == 5700
    for stratum in (
        "positive_attack_harmful",
        "negative_attack_benign",
        "negative_attack_neutral",
        "negative_harmless_harmful",
        "negative_harmless_benign",
        "negative_structural_sham_harmful",
        "negative_inert_harmful",
    ):
        assert receipt["stratum_counts"][stratum] == 600
    assert receipt["raw_prompt_or_token_ids_public"] is False
    assert receipt["target_outcome_exists"] is False


def test_weaponization_topology_receipts_contain_no_raw_fields() -> None:
    for name in (
        "weaponization-calibration-topology.reviewed.public.json",
        "weaponization-confirmation-topology.reviewed.public.json",
    ):
        payload = _receipt(name)
        serialized = json.dumps(payload, sort_keys=True)
        assert '"prompt_text"' not in serialized
        assert '"prompt_token_ids"' not in serialized
        assert '"request"' not in serialized


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ([11, 12], [11, 12]),
        ([[11, 12]], [11, 12]),
        ({"input_ids": [11, 12], "attention_mask": [1, 1]}, [11, 12]),
        ({"input_ids": [[11, 12]], "attention_mask": [[1, 1]]}, [11, 12]),
    ),
)
def test_normalize_prompt_token_ids_accepts_supported_unbatched_shapes(
    raw: object, expected: list[int]
) -> None:
    assert _normalize_prompt_token_ids(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ({"attention_mask": [1]}, ["input_ids"], [], [[1], [2]], [True]),
)
def test_normalize_prompt_token_ids_rejects_ambiguous_or_noninteger_shapes(
    raw: object,
) -> None:
    with pytest.raises(ValueError):
        _normalize_prompt_token_ids(raw)
