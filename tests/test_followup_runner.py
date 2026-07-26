from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.followup_runner import (
    PLACEMENTS,
    PrivateCheckpointStore,
    _assert_no_raw_public_fields,
    _placement_stage_name,
    build_placement_render_pair,
)


class SafePlacementTokenizer:
    chat_template = "safe-template-v1"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = "[USER]" + messages[0]["content"] + "[ASSISTANT]"
        return [ord(character) for character in text] if tokenize else text


def test_realized_placement_pair_is_exact_block_permutation() -> None:
    public, private = build_placement_render_pair(
        tokenizer=SafePlacementTokenizer(),
        request="safe request",
        scaffold="safe instructions",
        tokenizer_revision="pinned",
        context_ceiling=4096,
        generation_budget=16,
    )
    assert public["status"] == "passed"
    assert public["placements"] == list(PLACEMENTS)
    assert (
        private["ep_before_request"]["render_receipt"]["total_tokens"]
        == private["ep_after_request"]["render_receipt"]["total_tokens"]
    )
    assert "rendered_text" not in public
    assert "component_offsets" not in public


def test_placement_render_fails_before_generation_on_context_drift() -> None:
    with pytest.raises(ValueError, match="context ceiling"):
        build_placement_render_pair(
            tokenizer=SafePlacementTokenizer(),
            request="safe request",
            scaffold="safe instructions",
            tokenizer_revision="pinned",
            context_ceiling=2,
            generation_budget=16,
        )


def test_private_checkpoint_resume_and_provenance_are_fail_closed(tmp_path: Path) -> None:
    provenance = {
        "plan_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "run_id": "safe-run",
    }
    first = PrivateCheckpointStore(tmp_path, provenance)
    digest = first.write("generation-ep-before-request", {"safe": [1, 2, 3]})
    assert len(digest) == 64

    resumed = PrivateCheckpointStore(tmp_path, provenance)
    assert resumed.load("generation-ep-before-request") == {"safe": [1, 2, 3]}
    with pytest.raises(ValueError, match="overwrite completed"):
        resumed.write("generation-ep-before-request", {"safe": [4]})
    assert resumed.load("generation-ep-before-request") == {"safe": [1, 2, 3]}

    drifted = PrivateCheckpointStore(
        tmp_path,
        {**provenance, "source_commit": "3" * 40},
    )
    with pytest.raises(ValueError, match="provenance drift"):
        drifted.load("generation-ep-before-request")


def test_placement_stage_names_are_store_safe_and_unique() -> None:
    names = [_placement_stage_name(placement) for placement in PLACEMENTS]
    assert names == [
        "generation-ep-before-request",
        "generation-ep-after-request",
    ]
    assert len(names) == len(set(names))
    with pytest.raises(ValueError, match="unknown placement"):
        _placement_stage_name("other")


def test_private_checkpoint_hash_detects_interrupted_or_tampered_stage(
    tmp_path: Path,
) -> None:
    store = PrivateCheckpointStore(
        tmp_path,
        {"plan_sha256": "1" * 64, "source_commit": "2" * 40},
    )
    store.write("render", {"safe": True})
    stage = tmp_path / "stages" / "render.json"
    stage.write_text(json.dumps({"safe": False}))
    with pytest.raises(ValueError, match="hash drift"):
        store.load("render")


def test_public_receipt_guard_rejects_raw_or_replayable_fields() -> None:
    _assert_no_raw_public_fields({"hash": "a" * 64, "rows": [{"count": 1}]})
    with pytest.raises(ValueError, match="raw fields"):
        _assert_no_raw_public_fields({"nested": {"generated_token_ids": [1, 2]}})
