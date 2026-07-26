from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lexical_prompt_study.followup_generation import (
    FollowupReceiptStore,
    PlannedObservation,
    build_partition_observations,
    stable_followup_trial_id,
    _validate_followup_trial_topology,
)
from lexical_prompt_study.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from lexical_prompt_study.models import FollowupTrialReceipt


class SafeWordTokenizer:
    chat_template = "safe-chat-template"

    def _tokens(self, text: str) -> list[str]:
        for marker in (
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
        ):
            text = text.replace(marker, f" {marker} ")
        return text.split()

    def _id(self, token: str) -> int:
        return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [self._id(token) for token in self._tokens(text)]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = f"<chat-user> {messages[0]['content']} <chat-assistant>"
        return self.encode(text, add_special_tokens=False) if tokenize else text


def _public_plan() -> dict:
    return {
        "study_id": "lexical-scaffold-followup-v2",
        "replication": {
            "decoding": {
                "context_ceiling": 8192,
                "max_new_tokens": 1024,
            }
        },
    }


def test_harmful_partition_has_one_shared_and_six_order_specific_rows() -> None:
    request = "safe request about checksum verification"
    private = {
        "partitions": {
            "discovery": [
                {
                    "behavior_id": "SAFE-H-001",
                    "category": "safe",
                    "request": request,
                    "request_sha256": sha256_text(request),
                }
            ]
        }
    }
    observations, validations = build_partition_observations(
        tokenizer=SafeWordTokenizer(),
        private_plan=private,
        public_plan=_public_plan(),
        partition="discovery",
        tokenizer_revision="pinned",
    )
    assert len(observations) == 7
    assert len(validations) == 3
    assert [item.stratum for item in observations].count("base") == 1
    assert sum(item.shared_reference for item in observations) == 1
    assert {
        item.placement for item in observations if not item.shared_reference
    } == {"ep_before_request", "ep_after_request"}
    for arm in ("full", "structural_sham", "inert_length"):
        pair = [item for item in observations if item.arm == arm]
        assert len(pair) == 2
        assert len(pair[0].prompt_token_ids) == len(pair[1].prompt_token_ids)


def test_utility_partition_deduplicates_ordinary_benign() -> None:
    request = "safe benign checksum request"
    private = {
        "partitions": {
            "utility_calibration": [
                {
                    "behavior_id": "SAFE-B-001",
                    "category": "safe",
                    "request": request,
                    "request_sha256": sha256_text(request),
                }
            ]
        }
    }
    observations, validations = build_partition_observations(
        tokenizer=SafeWordTokenizer(),
        private_plan=private,
        public_plan=_public_plan(),
        partition="utility_calibration",
        tokenizer_revision="pinned",
    )
    assert len(observations) == 3
    assert len(validations) == 1
    assert [item.arm for item in observations].count("ordinary_benign") == 1
    assert [item.arm for item in observations].count("structured_benign") == 2


def test_followup_trial_id_includes_placement_and_partition() -> None:
    before = stable_followup_trial_id(
        "study", "behavior", "full", "ep_before_request", "discovery"
    )
    after = stable_followup_trial_id(
        "study", "behavior", "full", "ep_after_request", "discovery"
    )
    calibration = stable_followup_trial_id(
        "study", "behavior", "full", "ep_before_request", "calibration"
    )
    assert len({before, after, calibration}) == 3


def test_receipt_resume_validates_private_artifacts_and_state_provenance(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    observation = PlannedObservation(
        behavior_id="SAFE-H-001",
        category="safe",
        partition="discovery",
        request="safe request",
        request_sha256=sha256_text("safe request"),
        arm="base",
        placement=None,
        shared_reference=True,
        prompt_text="safe rendered prompt",
        prompt_token_ids=(1, 2, 3),
        render_validation=None,
    )
    study_id = "lexical-scaffold-followup-v2"
    trial_id = observation.trial_id(study_id)
    plan_sha = "1" * 64
    private_sha = "2" * 64
    source_commit = "3" * 40
    run_id = "safe-run"
    model_revision = "4" * 40
    lens_sha = "5" * 64
    sae_sha = "6" * 64
    raw = {
        "trial_id": trial_id,
        "behavior_id": observation.behavior_id,
        "partition": observation.partition,
        "arm": observation.arm,
        "placement": observation.placement,
        "request": observation.request,
        "prompt_text": observation.prompt_text,
        "prompt_token_ids": list(observation.prompt_token_ids),
        "generated_token_ids": [7, 8],
        "generated_text": "safe output",
    }
    restricted = tmp_path / "restricted.json"
    restricted.write_bytes(canonical_json_bytes(raw))
    states = tmp_path / "states.pt"
    prompt_ids_sha = sha256_bytes(
        canonical_json_bytes(list(observation.prompt_token_ids))
    )
    torch.save(
        {
            "provenance": {
                "trial_id": trial_id,
                "public_plan_sha256": plan_sha,
                "private_plan_sha256": private_sha,
                "source_commit": source_commit,
                "run_id": run_id,
                "prompt_token_ids_sha256": prompt_ids_sha,
                "model_revision": model_revision,
                "lens_sha256": lens_sha,
                "sae_sha256": sae_sha,
            },
            "states": {19: torch.zeros(4096, dtype=torch.bfloat16)},
        },
        states,
    )
    receipt = FollowupTrialReceipt(
        study_id=study_id,
        plan_sha256=plan_sha,
        private_plan_sha256=private_sha,
        source_commit=source_commit,
        run_id=run_id,
        trial_id=trial_id,
        partition="discovery",
        behavior_id=observation.behavior_id,
        category="safe",
        arm="base",
        placement=None,
        stratum="base",
        shared_reference=True,
        request_sha256=observation.request_sha256,
        prompt_sha256=sha256_text(observation.prompt_text),
        prompt_token_ids_sha256=prompt_ids_sha,
        render_validation_sha256=None,
        generated_text_sha256=sha256_text(raw["generated_text"]),
        generated_token_ids_sha256=sha256_bytes(
            canonical_json_bytes(raw["generated_token_ids"])
        ),
        generated_token_count=2,
        finish_reason="eos",
        truncated=False,
        refusal_detected=False,
        divider_detected=False,
        post_divider_present=False,
        restricted_artifact_path=str(restricted),
        restricted_artifact_sha256=sha256_file(restricted),
        state_bundle_path=str(states),
        state_bundle_sha256=sha256_file(states),
        capture_layers=[19],
        state_shape=[1, 4096],
        state_dtype="torch.bfloat16",
        generation_elapsed_seconds=0.1,
        capture_elapsed_seconds=0.1,
        peak_memory_bytes=None,
        model_revision=model_revision,
        tokenizer_revision=model_revision,
        lens_sha256=lens_sha,
        sae_sha256=sae_sha,
        software={},
    )
    store = FollowupReceiptStore(tmp_path / "receipts")
    store.write(receipt.model_dump(mode="json"))
    assert store.load_validated(
        observation=observation,
        study_id=study_id,
        plan_sha256=plan_sha,
        private_plan_sha256=private_sha,
        source_commit=source_commit,
        run_id=run_id,
    )["trial_id"] == trial_id
    raw["generated_text"] = "tampered"
    restricted.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(ValueError, match="restricted artifact hash"):
        store.load_validated(
            observation=observation,
            study_id=study_id,
            plan_sha256=plan_sha,
            private_plan_sha256=private_sha,
            source_commit=source_commit,
            run_id=run_id,
        )


def test_trial_topology_rejects_duplicated_shared_reference() -> None:
    receipt = FollowupTrialReceipt.model_construct(
        arm="base",
        shared_reference=False,
        placement="ep_before_request",
        render_validation_sha256="1" * 64,
        stratum="base:ep_before_request",
        capture_layers=[19],
        state_shape=[1, 4096],
    )
    with pytest.raises(ValueError, match="shared-reference"):
        _validate_followup_trial_topology(receipt)
