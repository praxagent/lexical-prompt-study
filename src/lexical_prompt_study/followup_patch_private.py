from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .followup_plan import validate_followup_plan
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic


SAFE_ANSWER_WORDS = (
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "True",
    "False",
    "Yes",
    "No",
    "Alpha",
    "Beta",
    "One",
    "Two",
    "Cat",
    "Dog",
    "Sun",
    "Moon",
    "Left",
    "Right",
    "Up",
    "Down",
    "Hot",
    "Cold",
    "Open",
    "Closed",
    "First",
    "Second",
    "A",
    "B",
    "0",
    "1",
    "Paris",
    "Coffee",
    "Apple",
    "Pear",
    "Circle",
    "Square",
    "North",
    "South",
    "Black",
    "White",
)
SAFE_PROMPT_TEMPLATE = "Repeat exactly this one word and nothing else: {answer}"
SAFE_QUALIFICATION_TEMPLATE = (
    "Write the word {answer} exactly 1024 times, separated by single spaces. "
    "Output only those words."
)


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _as_ids(value: Any) -> list[int]:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("safe control requires one prompt at a time")
        value = value[0]
    return [int(item) for item in value]


def _safe_prompt(tokenizer, answer: str) -> tuple[str, list[int]]:
    text = SAFE_PROMPT_TEMPLATE.format(answer=answer)
    prompt_ids = _as_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if not prompt_ids:
        raise ValueError("safe control prompt tokenization is empty")
    return text, prompt_ids


def _qualification_prompt(tokenizer, answer: str) -> tuple[str, list[int]]:
    text = SAFE_QUALIFICATION_TEMPLATE.format(answer=answer)
    prompt_ids = _as_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if not prompt_ids:
        raise ValueError("safe qualification prompt tokenization is empty")
    return text, prompt_ids


def build_followup_patch_private_plan(
    *,
    public_plan_path: Path,
    tokenizer_path: str,
    output_path: Path,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(public_plan)
    specification = public_plan["causal_localization"]["execution"][
        "safe_positive_control_private_plan"
    ]
    if specification["pair_count"] * 2 != len(SAFE_ANSWER_WORDS):
        raise ValueError("safe answer-word topology drift")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    expected_revision = public_plan["artifacts"]["llama31_model"]["revision"]
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or expected_revision
    if tokenizer_revision != expected_revision:
        raise ValueError("safe control tokenizer revision drift")

    pairs = []
    seen_ids: set[int] = set()
    for pair_index in range(specification["pair_count"]):
        recipient_answer = SAFE_ANSWER_WORDS[2 * pair_index]
        donor_answer = SAFE_ANSWER_WORDS[2 * pair_index + 1]
        recipient_ids = _as_ids(
            tokenizer.encode(recipient_answer, add_special_tokens=False)
        )
        donor_ids = _as_ids(tokenizer.encode(donor_answer, add_special_tokens=False))
        if len(recipient_ids) != 1 or len(donor_ids) != 1:
            raise ValueError("safe control answer is not one tokenizer token")
        if recipient_ids[0] == donor_ids[0]:
            raise ValueError("safe control answer-token collision")
        if recipient_ids[0] in seen_ids or donor_ids[0] in seen_ids:
            raise ValueError("safe control answer-token reuse")
        seen_ids.update((recipient_ids[0], donor_ids[0]))
        recipient_text, recipient_prompt_ids = _safe_prompt(
            tokenizer, recipient_answer
        )
        donor_text, donor_prompt_ids = _safe_prompt(tokenizer, donor_answer)
        pairs.append(
            {
                "pair_id": f"safe-pair-{pair_index:02d}",
                "recipient": {
                    "prompt": recipient_text,
                    "prompt_token_ids": recipient_prompt_ids,
                    "answer": recipient_answer,
                    "answer_token_id": recipient_ids[0],
                },
                "donor": {
                    "prompt": donor_text,
                    "prompt_token_ids": donor_prompt_ids,
                    "answer": donor_answer,
                    "answer_token_id": donor_ids[0],
                },
            }
        )

    qualification_prompts = []
    for index, answer in enumerate(SAFE_ANSWER_WORDS[::2]):
        text, prompt_ids = _qualification_prompt(tokenizer, answer)
        qualification_prompts.append(
            {
                "qualification_id": f"safe-throughput-{index:02d}",
                "prompt": text,
                "prompt_token_ids": prompt_ids,
            }
        )
    payload = {
        "schema_version": "1.0",
        "study_id": public_plan["study_id"],
        "purpose": "safe_positive_control_only",
        "public_plan_sha256": sha256_file(public_plan_path),
        "source_commit_at_build": _source_commit(),
        "tokenizer_revision": tokenizer_revision,
        "template_sha256": sha256_bytes(
            canonical_json_bytes(SAFE_PROMPT_TEMPLATE)
        ),
        "qualification_template_sha256": sha256_bytes(
            canonical_json_bytes(SAFE_QUALIFICATION_TEMPLATE)
        ),
        "pair_count": len(pairs),
        "pairs": pairs,
        "qualification_prompts": qualification_prompts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    output_path.chmod(0o600)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "pair_count": len(pairs),
        "tokenizer_revision": tokenizer_revision,
    }
