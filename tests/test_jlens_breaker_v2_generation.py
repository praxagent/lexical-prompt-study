from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lexical_prompt_study.hashing import sha256_file, sha256_text
from lexical_prompt_study.jlens_breaker_v2_generation import (
    JLensBreakerV2GenerationReceipt,
    deterministic_utility_completion,
    run_jlens_breaker_v2_generation_loop,
)


def test_deterministic_utility_judges_are_strict() -> None:
    assert deterministic_utility_completion("42", {"kind": "integer", "expected": 42})
    assert not deterministic_utility_completion(
        "the answer is 42", {"kind": "integer", "expected": 42}
    )
    assert deterministic_utility_completion(
        "  BRIDGE 7  ", {"kind": "exact_text", "expected": "BRIDGE 7"}
    )
    assert deterministic_utility_completion(
        "amber, garden, lantern",
        {"kind": "comma_list", "expected": ["amber", "garden", "lantern"]},
    )
    assert deterministic_utility_completion(
        '{"item":"amber","count":0}',
        {"kind": "json", "expected": {"count": 0, "item": "amber"}},
    )


def _decision() -> dict:
    return {
        "scaffold_before_request": {
            "frozen_jlens_score": 0.9,
            "threshold": 0.6,
            "would_cancel_before_first_token": True,
        }
    }


def _receipt() -> dict:
    return {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-breaker-v2",
        "status": "complete",
        "stage": "generation_calibration",
        "run_id": "synthetic-generation",
        "source_commit": "1" * 40,
        "public_plan_sha256": "2" * 64,
        "private_generation_topology_sha256": "3" * 64,
        "generation_trial_id": "trial-0",
        "prefill_trial_id": "prefill-0",
        "request_class": "ordinary_benign_request",
        "request_id": "safe-0",
        "request_sha256": "4" * 64,
        "material": "full_attack_scaffold",
        "placement": "scaffold_before_request",
        "injected_token_count": 252,
        "prompt_sha256": "5" * 64,
        "prompt_token_ids_sha256": "6" * 64,
        "detector_decisions": _decision(),
        "attempt": 1,
        "generated_text_sha256": "7" * 64,
        "generated_token_ids_sha256": "8" * 64,
        "generated_token_count": 1,
        "finish_reason": "eos",
        "refusal_detected": False,
        "deterministic_task_completion": True,
        "restricted_artifact_path": "/private/safe.json",
        "restricted_artifact_sha256": "9" * 64,
        "generation_elapsed_ms": 1.0,
        "peak_gpu_memory_bytes": 0,
        "model_revision": "a" * 40,
        "software": {},
    }


def test_generation_receipt_rejects_raw_fields_and_decision_drift() -> None:
    raw = _receipt()
    raw["generated_text"] = "must stay private"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        JLensBreakerV2GenerationReceipt.model_validate(raw)
    raw = _receipt()
    raw["detector_decisions"]["scaffold_before_request"][
        "would_cancel_before_first_token"
    ] = False
    with pytest.raises(ValidationError, match="arithmetic drift"):
        JLensBreakerV2GenerationReceipt.model_validate(raw)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"study_id": "lexical-jlens-breaker-v2"}))
    observation = {
        "generation_trial_id": "trial-0",
        "prefill_trial_id": "prefill-0",
        "request_class": "ordinary_benign_request",
        "request_id": "safe-0",
        "request_sha256": sha256_text("safe request"),
        "material": "full_attack_scaffold",
        "placement": "scaffold_before_request",
        "injected_token_count": 252,
        "prompt_sha256": sha256_text("safe prompt"),
        "prompt_token_ids_sha256": "6" * 64,
        "prompt_token_ids": [1, 2, 3],
        "prompt_text": "safe prompt",
        "detector_decisions": _decision(),
        "utility_judge": {"kind": "integer", "expected": 42},
    }
    topology = tmp_path / "topology.json"
    topology.write_text(
        json.dumps(
            {
                "study_id": "lexical-jlens-breaker-v2",
                "status": "v2_generation_topology_frozen_no_generation_outcomes",
                "stage": "generation_calibration",
                "primary_size": 252,
                "decoding": {
                    "max_new_tokens": 128,
                    "do_sample": False,
                    "model_revision": "a" * 40,
                },
                "enforcement_enabled": False,
                "sealed_confirmation_opened_or_executed": False,
                "bindings": {"public_plan_sha256": sha256_file(plan)},
                "observations": [observation],
            }
        )
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "lexical-jlens-breaker-v2",
                "status": "v2_generation_calibration_authorized",
                "paid_compute_authorized": True,
                "run_id": "synthetic-generation",
                "expected_observations": 1,
                "single_task_owned_pod_maximum": 1,
                "hard_ceiling_usd": 200,
                "maximum_new_compute_usd": 1,
                "conservative_postrun_infrastructure_ceiling_usd": 150,
                "enforcement_enabled": False,
                "sealed_confirmation_opened_or_executed": False,
                "approved_harmless_library_sha256": (
                    "46d5697068e3de551cd4a1ef06d2112248e4e66c4b613365d72f61340a8e0276"
                ),
                "bindings": {
                    "public_plan_sha256": sha256_file(plan),
                    "private_generation_topology_sha256": sha256_file(topology),
                    "source_commit": "1" * 40,
                },
            }
        )
    )
    return plan, topology, authorization


def test_generation_loop_resumes_and_verifies_restricted_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lexical_prompt_study.jlens_breaker_v2_generation as generation

    monkeypatch.setattr(generation, "_source_commit", lambda: "1" * 40)
    plan, topology, authorization = _write_fixture(tmp_path)
    output = tmp_path / "run"
    calls = 0

    def execute(rows: list[dict], attempt: int) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "generation_trial_id": rows[0]["generation_trial_id"],
                "attempt": attempt,
                "generated_text": "42",
                "generated_token_ids": [42],
                "finish_reason": "eos",
                "generation_elapsed_ms": 1.0,
                "peak_gpu_memory_bytes": 0,
                "software": {},
            }
        ]

    first = run_jlens_breaker_v2_generation_loop(
        public_plan_path=plan,
        private_topology_path=topology,
        authorization_path=authorization,
        output_root=output,
        run_id="synthetic-generation",
        batch_size=1,
        execute_batch=execute,
    )
    assert first["written_this_invocation"] == 1
    assert calls == 1
    receipt = json.loads((output / "receipts" / "trial-0.json").read_text())
    assert receipt["deterministic_task_completion"] is True
    assert "generated_text" not in receipt

    second = run_jlens_breaker_v2_generation_loop(
        public_plan_path=plan,
        private_topology_path=topology,
        authorization_path=authorization,
        output_root=output,
        run_id="synthetic-generation",
        batch_size=1,
        execute_batch=execute,
    )
    assert second["written_this_invocation"] == 0
    assert calls == 1

    (output / "restricted" / "trial-0.json").write_text("{}")
    with pytest.raises(ValueError, match="restricted generation hash drift"):
        run_jlens_breaker_v2_generation_loop(
            public_plan_path=plan,
            private_topology_path=topology,
            authorization_path=authorization,
            output_root=output,
            run_id="synthetic-generation",
            batch_size=1,
            execute_batch=execute,
        )
