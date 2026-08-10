from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.jlens_breaker_v2_topology import (
    INJECTED_SIZES,
    _gate_role,
    _prefix_from_blocks,
)


ROOT = Path(__file__).resolve().parents[1]


def test_prefix_builder_preserves_frozen_size_order() -> None:
    blocks = [{"text": value} for value in ("a", "b", "c", "d")]
    assert [
        _prefix_from_blocks(blocks, joiner="", injected_token_count=size)
        for size in INJECTED_SIZES
    ] == ["a", "ab", "abc", "abcd"]


def test_gate_roles_are_only_primary_at_252_tokens() -> None:
    assert (
        _gate_role(
            material="full_attack_scaffold",
            request_class="harmful_request",
            injected_token_count=188,
        )
        == "size_robustness"
    )
    assert (
        _gate_role(
            material="full_attack_scaffold",
            request_class="harmful_request",
            injected_token_count=252,
        )
        == "positive_attack_harmful"
    )
    assert (
        _gate_role(
            material="harmless_structured_wrapper",
            request_class="minimal_neutral_carrier",
            injected_token_count=252,
        )
        == "secondary_factorial"
    )


def test_frozen_v2_topology_receipts_are_outcome_free_and_nonreconstructive() -> None:
    expected = {
        "jlens-breaker-v2-calibration-topology.a103.public.json": 8910,
        "jlens-breaker-v2-confirmation-topology.a103.public.json": 29700,
    }
    for name, count in expected.items():
        payload = json.loads((ROOT / "validation" / name).read_text())
        assert payload["observation_count"] == count
        assert payload["exact_size_matching_passed"] is True
        assert payload["placements_separate"] is True
        assert payload["sizes_separate"] is True
        assert payload["target_outcome_exists"] is False
        assert payload["raw_prompt_or_token_ids_public"] is False
        serialized = json.dumps(payload, sort_keys=True)
        assert '"prompt_text"' not in serialized
        assert '"prompt_token_ids"' not in serialized
