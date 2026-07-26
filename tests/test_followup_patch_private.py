from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lexical_prompt_study.followup_patch_private import (
    SAFE_ANSWER_WORDS,
    build_followup_patch_private_plan,
)


ROOT = Path(__file__).parents[1]


class SafeTokenizer:
    _commit_hash = "0e9e39f249a16976918f6564b8830bc894c89659"

    def __init__(self) -> None:
        self.answer_ids = {
            answer: 1000 + index for index, answer in enumerate(SAFE_ANSWER_WORDS)
        }

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [self.answer_ids[text]]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        content = messages[0]["content"]
        answer = next(answer for answer in SAFE_ANSWER_WORDS if answer in content)
        return [1, self.answer_ids[answer], 2]


def test_build_patch_private_plan_freezes_twenty_distinct_safe_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = SafeTokenizer()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: tokenizer
            )
        ),
    )
    out = tmp_path / "patch.private.json"
    result = build_followup_patch_private_plan(
        public_plan_path=ROOT / "plans/followup_v2.public.json",
        tokenizer_path="/safe/pinned/tokenizer",
        output_path=out,
    )
    payload = json.loads(out.read_text())
    assert result["pair_count"] == 20
    assert payload["pair_count"] == 20
    assert len(payload["qualification_prompts"]) == 20
    assert out.stat().st_mode & 0o777 == 0o600
    answer_ids = [
        side["answer_token_id"]
        for pair in payload["pairs"]
        for side in (pair["recipient"], pair["donor"])
    ]
    assert len(answer_ids) == len(set(answer_ids)) == 40
