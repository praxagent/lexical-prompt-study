from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from .hashing import sha256_file, sha256_text
from .models import TrialReceipt
from .receipts import ReceiptStore, stable_trial_id
from .scaffolds import FOLLOWUP, render_arms


def run_synthetic(root: Path, n: int = 8, stop_after: int | None = None) -> dict:
    store = ReceiptStore(root)
    plan_hash = sha256_text("synthetic-plan-v1")
    expected: list[str] = []
    written = 0
    for index in range(n):
        behavior = f"synthetic-{index:03d}"
        for arm in ("base", "full", "structural_sham", "inert_length"):
            trial_id = stable_trial_id("synthetic", behavior, arm, 2, 0)
            expected.append(trial_id)
            if trial_id in store.completed_ids():
                continue
            text = f"safe synthetic output {index} {arm}"
            receipt = TrialReceipt(
                study_id="synthetic",
                plan_sha256=plan_hash,
                run_id="gate0",
                trial_id=trial_id,
                attempt=1,
                behavior_id=behavior,
                category="synthetic",
                split="engineering",
                arm=arm,
                turn=2,
                seed=0,
                prompt_sha256=sha256_text(f"safe prompt {index} {arm}"),
                prompt_token_ids_sha256=sha256_text(f"tokens {index} {arm}"),
                generated_token_ids=[index, len(arm)],
                generated_text_sha256=sha256_text(text),
                restricted_text_path=None,
                finish_reason="eos",
                generated_token_count=2,
                refusal_detected=arm in {"structural_sham", "inert_length"},
                divider_detected=arm != "base",
                post_divider_present=arm != "base",
                truncated=False,
                evaluator_score=1.0 if arm == "full" else 0.0,
                evaluator_parse_ok=True,
                elapsed_seconds=0.001,
                peak_memory_bytes=0,
                model_revision="synthetic",
                tokenizer_revision="synthetic",
                source_commit="synthetic",
                software={"python": "synthetic"},
                runtime={"device": "cpu"},
            )
            store.write(receipt)
            written += 1
            if stop_after is not None and written >= stop_after:
                return {"status": "interrupted", "expected": expected, "written": written}
    all_expected = [
        stable_trial_id("synthetic", f"synthetic-{index:03d}", arm, 2, 0)
        for index in range(n)
        for arm in ("base", "full", "structural_sham", "inert_length")
    ]
    store.validate_expected(all_expected)
    summary = {"status": "complete", "expected": len(all_expected), "written_this_call": written}
    (root / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    return summary


def build_engineering_fixture(public_plan: Path, output: Path, tokenizer_id: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    goal = "Explain how to sort a short list of harmless integers in Python."
    arms = render_arms(goal, tokenizer)
    payload = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-engineering-fixture",
        "public_plan_sha256": sha256_file(public_plan),
        "source_commit_at_build": "engineering-fixture",
        "source_files": {},
        "followup": FOLLOWUP,
        "harmful_behaviors": [
            {
                "behavior_id": "SAFE-ENGINEERING-000",
                "category": "synthetic",
                "split": "discovery",
                "goal": goal,
                "target": "A safe sorting explanation.",
                "behavior": "safe engineering fixture",
                "source": "local synthetic",
                "goal_sha256": sha256_text(goal),
                "rendered_arms": {
                    name: {
                        "text": arm.text,
                        "text_sha256": arm.sha256,
                        "token_ids": arm.token_ids,
                    }
                    for name, arm in arms.items()
                },
            }
        ],
        "utility_behaviors": [],
        "length_audit": [
            {
                "behavior_id": "SAFE-ENGINEERING-000",
                **{name: len(arms[name].token_ids) for name in arms},
            }
        ],
    }
    from .hashing import write_json_atomic

    digest = write_json_atomic(output, payload)
    return {"path": str(output), "sha256": digest, "tokenizer": tokenizer_id}
