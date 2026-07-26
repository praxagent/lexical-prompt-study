from __future__ import annotations

from pathlib import Path

from lexical_prompt_study.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
import pytest

from lexical_prompt_study.mechanism_runner import (
    ARMS,
    _validate_layer_checkpoint,
    build_observations,
)
from lexical_prompt_study.models import TrialReceipt
from lexical_prompt_study.receipts import stable_trial_id


class SafeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        ids = [128000]
        for message in messages:
            ids.extend([len(message["role"]), len(message["content"])])
        ids.append(128006)
        return ids


def _write_generation(
    root: Path,
    *,
    public_sha: str,
    study_id: str,
    behavior_id: str,
    arm: str,
    turn: int,
    prompt_ids: list[int],
    generated_text: str,
    generated_ids: list[int],
) -> None:
    trial_id = stable_trial_id(study_id, behavior_id, arm, turn, 0)
    raw_path = root / "restricted" / f"{trial_id}.json"
    raw_hash = write_json_atomic(
        raw_path,
        {
            "behavior_id": behavior_id,
            "arm": arm,
            "turn": turn,
            "generated_text": generated_text,
            "generated_token_ids": generated_ids,
        },
    )
    receipt = TrialReceipt(
        study_id=study_id,
        plan_sha256=public_sha,
        run_id="safe-fixture",
        trial_id=trial_id,
        attempt=1,
        behavior_id=behavior_id,
        category="safe",
        split="discovery",
        arm=arm,
        turn=turn,
        seed=0,
        prompt_sha256="0" * 64,
        prompt_token_ids_sha256=sha256_bytes(canonical_json_bytes(prompt_ids)),
        generated_token_ids=generated_ids,
        generated_text_sha256=sha256_text(generated_text),
        restricted_text_path=str(raw_path),
        restricted_artifact_sha256=raw_hash,
        finish_reason="eos",
        generated_token_count=len(generated_ids),
        refusal_detected=False,
        divider_detected=False,
        post_divider_present=False,
        truncated=False,
        evaluator_score=None,
        evaluator_parse_ok=False,
        elapsed_seconds=0.1,
        peak_memory_bytes=None,
        model_revision="safe",
        tokenizer_revision="safe",
        source_commit="1" * 40,
        software={},
        runtime={},
    )
    write_json_atomic(
        root / "receipts" / "trials" / f"{trial_id}.json",
        receipt.model_dump(mode="json"),
    )


def test_observation_builder_reconstructs_turn2_and_marks_missing(tmp_path: Path) -> None:
    public_path = tmp_path / "public.json"
    write_json_atomic(public_path, {"study_id": "safe-study"})
    public_sha = sha256_file(public_path)
    private_path = tmp_path / "private.json"
    behavior = {
        "behavior_id": "SAFE-001",
        "category": "safe",
        "split": "discovery",
        "rendered_arms": {arm: {"text": f"safe prompt {arm}"} for arm in ARMS},
    }
    write_json_atomic(
        private_path,
        {
            "public_plan_sha256": public_sha,
            "followup": "safe followup",
            "harmful_behaviors": [behavior],
        },
    )
    tokenizer = SafeTokenizer()
    generation_root = tmp_path / "generation"
    for arm in ARMS:
        response = f"safe response {arm}"
        prompt1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": behavior["rendered_arms"][arm]["text"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        _write_generation(
            generation_root,
            public_sha=public_sha,
            study_id="safe-study",
            behavior_id="SAFE-001",
            arm=arm,
            turn=1,
            prompt_ids=prompt1,
            generated_text=response,
            generated_ids=[10, 11],
        )
        prompt2 = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": behavior["rendered_arms"][arm]["text"]},
                {"role": "assistant", "content": response},
                {"role": "user", "content": "safe followup"},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        _write_generation(
            generation_root,
            public_sha=public_sha,
            study_id="safe-study",
            behavior_id="SAFE-001",
            arm=arm,
            turn=2,
            prompt_ids=prompt2,
            generated_text="safe final",
            generated_ids=[20, 21],
        )

    observations = build_observations(
        tokenizer=tokenizer,
        private_plan_path=private_path,
        public_plan_path=public_path,
        generation_root=generation_root,
        split="discovery",
    )
    assert len(observations) == 28
    assert sum(item.position_available for item in observations) == 12
    generated_one = next(
        item
        for item in observations
        if item.arm == "full" and item.position_token_index == 1
    )
    assert generated_one.prefix_token_ids[-2:] == (20, 21)
    missing = next(
        item
        for item in observations
        if item.arm == "full" and item.position_token_index == 2
    )
    assert not missing.position_available
    assert missing.prefix_token_ids == ()
    assert "safe response" not in str(missing.public_metadata())

    with pytest.raises(ValueError, match="receipt count mismatch"):
        _validate_layer_checkpoint(
            {
                "layer": 0,
                "public_plan_sha256": public_sha,
                "source_commit": "1" * 40,
                "run_id": "gate3-safe",
                "capture_sha256": "2" * 64,
                "receipts": [],
            },
            layer=0,
            public_plan_sha256=public_sha,
            source_commit="1" * 40,
            run_id="gate3-safe",
            capture_sha256="2" * 64,
            observations=observations,
        )
