from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study import evaluate
from lexical_prompt_study.artifacts import EVALUATOR_REVISION
from lexical_prompt_study.evaluate import (
    _behavior_rows,
    _behavior_text,
    _restricted_path,
    _single_token_id,
    score_behavior_receipts,
)
from lexical_prompt_study.hashing import sha256_file


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


def test_complete_scoring_resume_does_not_load_model(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"harmful_behaviors": []}))
    generation_path = tmp_path / "generation" / "receipts" / "trials" / "trial.json"
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text(json.dumps({"trial_id": "trial"}))
    score_path = tmp_path / "scores" / "trials" / "trial.json"
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        json.dumps(
            {
                "generation_receipt_sha256": sha256_file(generation_path),
                "evaluator_revision": EVALUATOR_REVISION,
                "scoring_implementation_sha256": sha256_file(Path(evaluate.__file__)),
            }
        )
    )

    summary = score_behavior_receipts(
        private_plan_path=private,
        generation_root=tmp_path / "generation",
        evaluator_path="must-not-be-loaded",
        output_root=tmp_path / "scores",
    )

    assert summary["written_this_call"] == 0
    assert summary["model_loaded_this_call"] is False


def test_followup_scoring_resolves_partition_rows_and_restricted_path(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    raw_path = tmp_path / "restricted.json"
    receipts = {
        receipt_path: {
            "partition": "discovery",
            "restricted_artifact_path": str(raw_path),
        }
    }
    private = {
        "partitions": {
            "discovery": [
                {
                    "behavior_id": "SAFE-001",
                    "behavior": "safe behavior",
                }
            ]
        }
    }

    assert _behavior_rows(private, receipts) == private["partitions"]["discovery"]
    assert _restricted_path(receipts[receipt_path], receipt_path) == raw_path
    assert _behavior_text({"request": "safe request"}) == "safe request"
    assert _behavior_text({"behavior": "safe behavior"}) == "safe behavior"


def test_followup_scoring_rejects_mixed_partitions(tmp_path: Path) -> None:
    receipts = {
        tmp_path / "a.json": {"partition": "discovery"},
        tmp_path / "b.json": {"partition": "calibration"},
    }
    with pytest.raises(ValueError, match="exactly one partition"):
        _behavior_rows({"partitions": {}}, receipts)
