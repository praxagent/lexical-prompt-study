from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.jlens_breaker_v2_runner import (
    JLensBreakerV2PrefillReceipt,
    JLensBreakerV2ReceiptStore,
    run_jlens_breaker_v2_prefill_loop,
)


def _receipt(trial_id: str, *, material: str = "no_scaffold") -> dict:
    injected = material != "no_scaffold"
    return {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-breaker-v2",
        "status": "complete",
        "run_id": "synthetic-v2-safe-run",
        "source_commit": "1" * 40,
        "public_plan_sha256": "2" * 64,
        "private_topology_sha256": "3" * 64,
        "partition": "detector_calibration",
        "trial_id": trial_id,
        "gate_role": "synthetic_safe",
        "request_class": "ordinary_benign_request",
        "request_id": f"request-{trial_id}",
        "request_sha256": "4" * 64,
        "material": material,
        "material_sha256": "5" * 64 if injected else None,
        "wrapper_id": f"wrapper-{trial_id}"
        if material == "harmless_structured_wrapper"
        else None,
        "placement": "scaffold_before_request" if injected else None,
        "injected_token_count": 64 if injected else 0,
        "prompt_sha256": "6" * 64,
        "prompt_token_ids_sha256": "7" * 64,
        "attempt": 1,
        "feature_6779_magnitude": 0.0,
        "frozen_subspace_score": 0.0,
        "sae_normalized_reconstruction_error": 0.2,
        "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        "restricted_exact_match": False,
        "structural_metrics": {"line_count": 1.0},
        "prefill_latency_ms": 1.0,
        "detector_readout_latency_ms": 0.1,
        "peak_gpu_memory_bytes": 0,
    }


def _provenance() -> dict[str, str]:
    return {
        "run_id": "synthetic-v2-safe-run",
        "source_commit": "1" * 40,
        "public_plan_sha256": "2" * 64,
        "private_topology_sha256": "3" * 64,
        "partition": "detector_calibration",
    }


def test_v2_receipt_rejects_raw_fields_and_invalid_sizes() -> None:
    raw = _receipt("trial-0")
    raw["prompt_text"] = "must never enter a receipt"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        JLensBreakerV2PrefillReceipt.model_validate(raw)
    invalid = _receipt("trial-1", material="full_attack_scaffold")
    invalid["injected_token_count"] = 100
    with pytest.raises(ValidationError, match="injected receipt topology drift"):
        JLensBreakerV2PrefillReceipt.model_validate(invalid)


def test_v2_store_is_atomic_idempotent_and_provenance_bound(tmp_path: Path) -> None:
    store = JLensBreakerV2ReceiptStore(tmp_path / "run")
    payload = _receipt("trial-0")
    first_hash = store.write(payload)
    assert store.write(payload) == first_hash
    assert store.load("trial-0", provenance=_provenance()) is not None
    drifted = _provenance()
    drifted["run_id"] = "different"
    with pytest.raises(ValueError, match="provenance drift"):
        store.load("trial-0", provenance=drifted)


def _executor_result(trial_id: str, attempt: int) -> dict:
    return {
        "trial_id": trial_id,
        "attempt": attempt,
        "feature_6779_magnitude": 0.0,
        "frozen_subspace_score": 0.0,
        "sae_normalized_reconstruction_error": 0.2,
        "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        "restricted_exact_match": False,
        "structural_metrics": {"line_count": 1.0},
        "prefill_latency_ms": 1.0,
        "detector_readout_latency_ms": 0.1,
        "peak_gpu_memory_bytes": 0,
    }


def test_v2_loop_resumes_after_forced_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lexical_prompt_study.jlens_breaker_v2_runner as runner

    monkeypatch.setattr(runner, "_source_commit", lambda: "1" * 40)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"study_id": "lexical-jlens-breaker-v2"}))
    observations = [
        {
            "trial_id": f"trial-{index}",
            "gate_role": "synthetic_safe",
            "request_class": "ordinary_benign_request",
            "request_id": f"request-{index}",
            "request_sha256": "4" * 64,
            "material": "no_scaffold",
            "material_sha256": None,
            "wrapper_id": None,
            "placement": None,
            "injected_token_count": 0,
            "prompt_sha256": "6" * 64,
            "prompt_token_ids_sha256": "7" * 64,
            "prompt_text": f"safe synthetic prompt {index}",
            "prompt_token_ids": [index + 1],
        }
        for index in range(2)
    ]
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(
        json.dumps(
            {
                "study_id": "lexical-jlens-breaker-v2",
                "status": "v2_topology_frozen_no_target_outcomes",
                "prefill_only": True,
                "partition": "detector_calibration",
                "input_sha256": {"plan": sha256_file(plan_path)},
                "observations": observations,
            }
        )
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "lexical-jlens-breaker-v2",
                "status": "v2_prefill_authorized",
                "paid_compute_authorized": True,
                "approved_harmless_library_sha256": (
                    "46d5697068e3de551cd4a1ef06d2112248e4e66c4b613365d72f61340a8e0276"
                ),
                "partition": "detector_calibration",
                "run_id": "synthetic-v2-safe-run",
                "expected_observations": 2,
                "single_task_owned_pod_maximum": 1,
                "hard_ceiling_usd": 200,
                "maximum_new_compute_usd": 1.0,
                "conservative_postrun_infrastructure_ceiling_usd": 150.0,
                "bindings": {
                    "public_plan_sha256": sha256_file(plan_path),
                    "private_topology_sha256": sha256_file(topology_path),
                    "source_commit": "1" * 40,
                },
            }
        )
    )
    output_root = tmp_path / "run"

    def interrupted(batch: list[dict], attempt: int) -> list[dict]:
        if batch[0]["trial_id"] == "trial-1":
            raise RuntimeError("forced interruption")
        return [_executor_result(batch[0]["trial_id"], attempt)]

    with pytest.raises(RuntimeError, match="forced interruption"):
        run_jlens_breaker_v2_prefill_loop(
            public_plan_path=plan_path,
            private_topology_path=topology_path,
            authorization_path=authorization_path,
            output_root=output_root,
            run_id="synthetic-v2-safe-run",
            batch_size=1,
            execute_batch=interrupted,
        )
    assert len(list((output_root / "receipts").glob("*.json"))) == 1

    resumed_trials: list[str] = []

    def resumed(batch: list[dict], attempt: int) -> list[dict]:
        resumed_trials.append(batch[0]["trial_id"])
        return [_executor_result(batch[0]["trial_id"], attempt)]

    summary = run_jlens_breaker_v2_prefill_loop(
        public_plan_path=plan_path,
        private_topology_path=topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id="synthetic-v2-safe-run",
        batch_size=1,
        execute_batch=resumed,
    )
    assert resumed_trials == ["trial-1"]
    assert summary["written_this_invocation"] == 1
    assert summary["resumed_receipt_count"] == 1

    def must_not_run(_batch: list[dict], _attempt: int) -> list[dict]:
        raise AssertionError("completed receipts must not execute again")

    complete = run_jlens_breaker_v2_prefill_loop(
        public_plan_path=plan_path,
        private_topology_path=topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id="synthetic-v2-safe-run",
        batch_size=1,
        execute_batch=must_not_run,
    )
    assert complete["written_this_invocation"] == 0
    assert complete["resumed_receipt_count"] == 2
