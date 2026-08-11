from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lexical_prompt_study.artifacts import EVALUATOR_REVISION
from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.jlens_breaker_v2_generation import (
    JLensBreakerV2GenerationReceipt,
)
from lexical_prompt_study.jlens_breaker_v2_scoring import (
    HarmBenchExecutor,
    JLensBreakerV2HarmBenchReceipt,
    run_jlens_breaker_v2_harmbench_loop,
)


def _score_payload() -> dict:
    return {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-breaker-v2",
        "status": "complete",
        "stage": "generation_calibration_harmbench_scoring",
        "scoring_run_id": "score-test",
        "generation_trial_id": "trial-0",
        "request_id": "request-0",
        "request_sha256": "a" * 64,
        "material": "full_attack_scaffold",
        "placement": "scaffold_before_request",
        "generation_receipt_sha256": "b" * 64,
        "generated_text_sha256": "c" * 64,
        "evaluator_repository": "cais/HarmBench-Llama-2-13b-cls",
        "evaluator_revision": EVALUATOR_REVISION,
        "scoring_implementation_sha256": "d" * 64,
        "evaluator_prompt_sha256": "e" * 64,
        "yes_token_id": 1,
        "no_token_id": 2,
        "yes_logit": 2.0,
        "no_logit": 1.0,
        "yes_probability": 0.75,
        "no_probability": 0.25,
        "binary_prediction": True,
        "parse_ok": True,
        "batch_size": 4,
        "batch_elapsed_seconds": 0.1,
        "source_commit": "1" * 40,
        "software": {},
    }


def test_v2_harmbench_receipt_rejects_raw_content_and_probability_drift() -> None:
    payload = _score_payload()
    payload["generation"] = "must not appear"
    with pytest.raises(ValidationError, match="generation"):
        JLensBreakerV2HarmBenchReceipt.model_validate(payload)

    payload = _score_payload()
    payload["yes_probability"] = 0.8
    with pytest.raises(ValidationError, match="sum to one"):
        JLensBreakerV2HarmBenchReceipt.model_validate(payload)


def _generation_receipt(index: int) -> JLensBreakerV2GenerationReceipt:
    decision = {
        "frozen_jlens_score": 0.7,
        "threshold": 0.6,
        "would_cancel_before_first_token": True,
    }
    return JLensBreakerV2GenerationReceipt.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "lexical-jlens-breaker-v2",
            "status": "complete",
            "stage": "generation_calibration",
            "run_id": "generation-test",
            "source_commit": "2" * 40,
            "public_plan_sha256": "3" * 64,
            "private_generation_topology_sha256": "4" * 64,
            "generation_trial_id": f"trial-{index:04d}",
            "prefill_trial_id": f"prefill-{index:04d}",
            "request_class": "harmful_request",
            "request_id": f"request-{index:04d}",
            "request_sha256": "5" * 64,
            "material": "full_attack_scaffold",
            "placement": "scaffold_before_request",
            "injected_token_count": 252,
            "prompt_sha256": "6" * 64,
            "prompt_token_ids_sha256": "7" * 64,
            "detector_decisions": {"scaffold_before_request": decision},
            "attempt": 1,
            "generated_text_sha256": "8" * 64,
            "generated_token_ids_sha256": "9" * 64,
            "generated_token_count": 1,
            "finish_reason": "eos",
            "refusal_detected": False,
            "deterministic_task_completion": None,
            "restricted_artifact_path": f"/private/trial-{index:04d}.json",
            "restricted_artifact_sha256": "a" * 64,
            "generation_elapsed_ms": 1.0,
            "peak_gpu_memory_bytes": 0,
            "model_revision": "b" * 40,
            "software": {},
        }
    )


def test_v2_harmbench_loop_is_810_row_resumable_and_no_load_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lexical_prompt_study.jlens_breaker_v2_scoring as scoring

    monkeypatch.setattr(scoring, "_source_commit", lambda: "1" * 40)
    plan = tmp_path / "plan.json"
    topology = tmp_path / "topology.json"
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    summary_path = generation_root / "summary.json"
    authorization = tmp_path / "authorization.json"
    plan.write_text("{}")
    topology.write_text("{}")
    summary_path.write_text(
        json.dumps(
            {
                "source_commit": "2" * 40,
                "public_plan_sha256": sha256_file(plan),
                "private_generation_topology_sha256": sha256_file(topology),
            }
        )
    )

    items = []
    generation_receipts = generation_root / "receipts"
    generation_receipts.mkdir()
    for index in range(810):
        path = generation_receipts / f"trial-{index:04d}.json"
        path.write_text("{}")
        items.append(
            {
                "generation_path": path,
                "generation_receipt": _generation_receipt(index),
                "behavior": "restricted behavior",
                "generation": "restricted generation",
            }
        )
    monkeypatch.setattr(
        scoring,
        "_load_generation_inputs",
        lambda **_: (
            items,
            {
                "source_commit": "2" * 40,
                "public_plan_sha256": sha256_file(plan),
                "private_generation_topology_sha256": sha256_file(topology),
            },
            "c" * 64,
        ),
    )
    implementation_sha256 = sha256_file(Path(scoring.__file__))
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "lexical-jlens-breaker-v2",
                "status": "v2_generation_calibration_harmbench_scoring_authorized",
                "paid_compute_authorized": True,
                "scoring_run_id": "score-test",
                "expected_generation_receipts": 2430,
                "expected_harmful_score_receipts": 810,
                "single_task_owned_scoring_pod_maximum": 1,
                "hard_ceiling_usd": 200,
                "maximum_new_compute_usd": 1,
                "conservative_postrun_infrastructure_ceiling_usd": 150,
                "sealed_confirmation_opened_or_executed": False,
                "raw_generation_or_classifier_input_public": False,
                "bindings": {
                    "scoring_source_commit": "1" * 40,
                    "scoring_implementation_sha256": implementation_sha256,
                    "public_plan_sha256": sha256_file(plan),
                    "private_generation_topology_sha256": sha256_file(topology),
                    "generation_summary_sha256": sha256_file(summary_path),
                    "generation_receipt_manifest_sha256": "c" * 64,
                    "generation_source_commit": "2" * 40,
                    "evaluator_revision": EVALUATOR_REVISION,
                },
            }
        )
    )

    calls = 0

    def execute(rows: list[dict[str, str]]) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "generation_trial_id": row["generation_trial_id"],
                "evaluator_prompt_sha256": "d" * 64,
                "yes_token_id": 1,
                "no_token_id": 2,
                "yes_logit": 2.0,
                "no_logit": 1.0,
                "yes_probability": 0.75,
                "no_probability": 0.25,
                "binary_prediction": True,
                "parse_ok": True,
                "batch_size": len(rows),
                "batch_elapsed_seconds": 0.1,
                "software": {},
            }
            for row in rows
        ]

    first = run_jlens_breaker_v2_harmbench_loop(
        public_plan_path=plan,
        private_topology_path=topology,
        generation_root=generation_root,
        authorization_path=authorization,
        output_root=tmp_path / "scores",
        scoring_run_id="score-test",
        batch_size=64,
        execute_batch=execute,
    )
    assert first["written_this_invocation"] == 810
    assert first["harmful_score_receipt_count"] == 810
    assert calls == 13
    receipt = json.loads(
        (tmp_path / "scores/trials/trial-0000.json").read_text()
    )
    assert "generation" not in receipt
    assert "behavior" not in receipt

    def must_not_execute(_: list[dict[str, str]]) -> list[dict]:
        raise AssertionError("completed resume must not execute classifier")

    second = run_jlens_breaker_v2_harmbench_loop(
        public_plan_path=plan,
        private_topology_path=topology,
        generation_root=generation_root,
        authorization_path=authorization,
        output_root=tmp_path / "scores",
        scoring_run_id="score-test",
        batch_size=64,
        execute_batch=must_not_execute,
    )
    assert second["written_this_invocation"] == 0
    assert second["resumed_score_receipt_count"] == 810
    assert second["model_loaded_this_invocation"] is False


def test_harmbench_executor_is_lazy() -> None:
    executor = HarmBenchExecutor("must-not-be-loaded")
    assert executor.model_loaded is False
