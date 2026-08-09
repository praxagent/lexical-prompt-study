from __future__ import annotations

from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from lexical_prompt_study.weaponization_runner import (
    WeaponizationPrefillReceipt,
    WeaponizationReceiptStore,
    run_weaponization_prefill_loop,
)
from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.weaponization_plan import weaponization_protocol_sha256


def _receipt(trial_id: str, *, material: str = "no_scaffold") -> dict:
    injected = material != "no_scaffold"
    return {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-weaponization-breaker-v1",
        "status": "complete",
        "run_id": "synthetic-safe-run",
        "source_commit": "1" * 40,
        "public_plan_sha256": "2" * 64,
        "private_topology_sha256": "3" * 64,
        "partition": "detector_calibration",
        "trial_id": trial_id,
        "stratum": "synthetic_safe",
        "critical": False,
        "request_class": "ordinary_benign_request",
        "request_id": f"request-{trial_id}",
        "request_sha256": "4" * 64,
        "material": material,
        "material_sha256": "5" * 64 if injected else None,
        "wrapper_id": f"wrapper-{trial_id}"
        if material == "harmless_structured_wrapper"
        else None,
        "placement": "scaffold_before_request" if injected else None,
        "injected_token_count": 252 if injected else 0,
        "prompt_sha256": "6" * 64,
        "prompt_token_ids_sha256": "7" * 64,
        "attempt": 1,
        "feature_6779_magnitude": 0.0,
        "frozen_subspace_score": 0.0,
        "sae_normalized_reconstruction_error": 0.2,
        "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        "restricted_exact_match": False,
        "structural_metrics": {"divider_count": 0.0},
        "prefill_latency_ms": 1.0,
        "detector_readout_latency_ms": 0.2,
        "peak_gpu_memory_bytes": 0,
    }


def _provenance() -> dict[str, str]:
    return {
        "run_id": "synthetic-safe-run",
        "source_commit": "1" * 40,
        "public_plan_sha256": "2" * 64,
        "private_topology_sha256": "3" * 64,
        "partition": "detector_calibration",
    }


def test_weaponization_receipt_rejects_raw_fields() -> None:
    payload = _receipt("trial-0")
    payload["prompt_text"] = "must never enter a receipt"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WeaponizationPrefillReceipt.model_validate(payload)


def test_weaponization_receipt_requires_all_jlens_layers() -> None:
    payload = _receipt("trial-0")
    payload["jlens_refusal_minus_compliance_trajectory"] = [0.0] * 30
    with pytest.raises(ValidationError, match="31 source layers"):
        WeaponizationPrefillReceipt.model_validate(payload)


def test_weaponization_receipt_requires_wrapper_binding() -> None:
    payload = _receipt("trial-0", material="harmless_structured_wrapper")
    payload["wrapper_id"] = None
    with pytest.raises(ValidationError, match="wrapper ID"):
        WeaponizationPrefillReceipt.model_validate(payload)


def test_weaponization_store_is_atomic_idempotent_and_resumable(tmp_path: Path) -> None:
    store = WeaponizationReceiptStore(tmp_path / "run")
    first = _receipt("trial-0")
    second = _receipt("trial-1", material="harmless_structured_wrapper")
    first_hash = store.write(first)
    second_hash = store.write(second)
    assert store.write(first) == first_hash
    assert store.write(second) == second_hash

    resumed = WeaponizationReceiptStore(tmp_path / "run")
    assert resumed.load("trial-0", provenance=_provenance()) is not None
    assert resumed.load("trial-1", provenance=_provenance()) is not None
    assert resumed.load("trial-2", provenance=_provenance()) is None
    assert len(list((tmp_path / "run" / "receipts").glob("*.json"))) == 2
    assert len((tmp_path / "run" / "attempts.jsonl").read_text().splitlines()) == 2


def test_weaponization_store_refuses_completed_receipt_overwrite(tmp_path: Path) -> None:
    store = WeaponizationReceiptStore(tmp_path / "run")
    payload = _receipt("trial-0")
    store.write(payload)
    drifted = dict(payload)
    drifted["feature_6779_magnitude"] = 1.0
    with pytest.raises(ValueError, match="refusing weaponization overwrite"):
        store.write(drifted)


def test_weaponization_store_rejects_resume_provenance_drift(tmp_path: Path) -> None:
    store = WeaponizationReceiptStore(tmp_path / "run")
    store.write(_receipt("trial-0"))
    drifted = _provenance()
    drifted["run_id"] = "different-run"
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


def test_weaponization_loop_resumes_after_forced_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lexical_prompt_study.weaponization_runner as runner

    monkeypatch.setattr(runner, "validate_weaponization_plan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_source_commit", lambda: "1" * 40)
    public_plan_path = tmp_path / "plan.json"
    synthetic_plan = {
        "study_id": "lexical-scaffold-weaponization-breaker-v1",
        "input_freeze_receipts": {},
    }
    public_plan_path.write_text(json.dumps(synthetic_plan))
    observations = []
    for index in range(2):
        observations.append(
            {
                "trial_id": f"trial-{index}",
                "stratum": "synthetic_safe",
                "critical": False,
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
        )
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(
        json.dumps(
            {
                "study_id": "lexical-scaffold-weaponization-breaker-v1",
                "protocol_sha256": weaponization_protocol_sha256(synthetic_plan),
                "status": "topology_frozen_human_reviewed",
                "human_semantic_review_approved": True,
                "prefill_only": True,
                "partition": "detector_calibration",
                "observations": observations,
            }
        )
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "lexical-scaffold-weaponization-breaker-v1",
                "status": "weaponization_prefill_authorized",
                "paid_compute_authorized": True,
                "human_semantic_review_approved": True,
                "partition": "detector_calibration",
                "run_id": "synthetic-safe-run",
                "expected_observations": 2,
                "single_task_owned_pod_maximum": 1,
                "hard_ceiling_usd": 200,
                "bindings": {
                    "public_plan_sha256": sha256_file(public_plan_path),
                    "private_topology_sha256": sha256_file(topology_path),
                    "source_commit": "1" * 40,
                },
            }
        )
    )
    output_root = tmp_path / "run"
    calls = 0

    def interrupted(batch: list[dict], attempt: int) -> list[dict]:
        nonlocal calls
        calls += 1
        if batch[0]["trial_id"] == "trial-1":
            raise RuntimeError("forced interruption")
        return [_executor_result(batch[0]["trial_id"], attempt)]

    with pytest.raises(RuntimeError, match="forced interruption"):
        run_weaponization_prefill_loop(
            public_plan_path=public_plan_path,
            private_topology_path=topology_path,
            authorization_path=authorization_path,
            output_root=output_root,
            run_id="synthetic-safe-run",
            batch_size=1,
            execute_batch=interrupted,
        )
    assert calls == 3
    assert len(list((output_root / "receipts").glob("*.json"))) == 1

    resumed_trials: list[str] = []

    def resumed(batch: list[dict], attempt: int) -> list[dict]:
        resumed_trials.append(batch[0]["trial_id"])
        return [_executor_result(batch[0]["trial_id"], attempt)]

    summary = run_weaponization_prefill_loop(
        public_plan_path=public_plan_path,
        private_topology_path=topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id="synthetic-safe-run",
        batch_size=1,
        execute_batch=resumed,
    )
    assert resumed_trials == ["trial-1"]
    assert summary["written_this_invocation"] == 1
    assert summary["resumed_receipt_count"] == 1

    def must_not_run(_batch: list[dict], _attempt: int) -> list[dict]:
        raise AssertionError("completed receipts must not execute again")

    complete = run_weaponization_prefill_loop(
        public_plan_path=public_plan_path,
        private_topology_path=topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id="synthetic-safe-run",
        batch_size=1,
        execute_batch=must_not_run,
    )
    assert complete["written_this_invocation"] == 0
    assert complete["resumed_receipt_count"] == 2
