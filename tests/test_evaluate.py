from __future__ import annotations

import pytest

from lexical_prompt_study.evaluate import _single_token_id


class FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]


def test_classifier_label_must_be_exactly_one_token() -> None:
    tokenizer = FakeTokenizer({"Yes": [8241], "No": [3782, 13]})

    assert _single_token_id(tokenizer, "Yes") == 8241
    with pytest.raises(ValueError, match="not exactly one token"):
        _single_token_id(tokenizer, "No")
