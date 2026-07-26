from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lexical_prompt_study.factorial_receipts import (
    FactorialReceiptStore,
    validate_factorial_trial_receipt,
)
from lexical_prompt_study.hashing import write_json_atomic
from lexical_prompt_study.models import FactorialTrialReceipt


def _receipt(tmp_path: Path, *, injected: bool = False) -> dict:
    restricted = tmp_path / ("injected.private.json" if injected else "base.private.json")
    restricted_sha256 = write_json_atomic(
        restricted,
        {
            "generated_text": "safe synthetic output",
            "generated_token_ids": [1, 2, 3],
        },
    )
    return {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-8b-factorial-v1",
        "status": "complete",
        "public_plan_sha256": "1" * 64,
        "private_plan_sha256": "2" * 64,
        "assay_receipt_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "run_id": "safe-factorial",
        "trial_id": "trial-injected" if injected else "trial-base",
        "attempt": 1,
        "request_class": "harmful_request",
        "request_id": "safe-request",
        "prompt_family_id": "safe-family",
        "request_sha256": "5" * 64,
        "material": "full_scaffold" if injected else "no_scaffold",
        "placement": "ep_before_request" if injected else None,
        "size_id": "blocks-004" if injected else "natural-base",
        "injected_token_count": 19 if injected else 0,
        "shared_reference": not injected,
        "render_group_sha256": "6" * 64 if injected else None,
        "prompt_sha256": "7" * 64,
        "prompt_token_ids_sha256": "8" * 64,
        "generated_text_sha256": "9" * 64,
        "generated_token_ids_sha256": "a" * 64,
        "generated_token_count": 3,
        "finish_reason": "eos",
        "truncated": False,
        "feature_6779_magnitude": 0.25,
        "feature_6779_positive": True,
        "frozen_subspace_score": 0.5,
        "sae_reconstruction_relative_error": 0.1,
        "assistant_boundary_jlens_margin": -0.2,
        "refusal_detected": False,
        "task_completion": None,
        "utility_score": None,
        "phase_parser_result": {"status": "complete"},
        "restricted_artifact_path": str(restricted),
        "restricted_artifact_sha256": restricted_sha256,
        "generation_elapsed_seconds": 0.2,
        "readout_elapsed_seconds": 0.1,
        "peak_memory_bytes": 1024,
        "model_revision": "b" * 40,
        "tokenizer_revision": "c" * 40,
        "lens_sha256": "d" * 64,
        "sae_sha256": "e" * 64,
        "software": {"python": "test"},
    }


def test_factorial_trial_receipt_accepts_base_and_injected_topologies(
    tmp_path: Path,
) -> None:
    assert validate_factorial_trial_receipt(_receipt(tmp_path)).shared_reference is True
    assert (
        validate_factorial_trial_receipt(_receipt(tmp_path, injected=True)).placement
        == "ep_before_request"
    )


def test_factorial_trial_receipt_rejects_raw_public_token_ids(tmp_path: Path) -> None:
    payload = _receipt(tmp_path)
    payload["generated_token_ids"] = [1, 2, 3]
    with pytest.raises(ValidationError, match="Extra inputs"):
        FactorialTrialReceipt.model_validate(payload)


def test_factorial_trial_receipt_rejects_feature_prevalence_drift(
    tmp_path: Path,
) -> None:
    payload = _receipt(tmp_path)
    payload["feature_6779_positive"] = False
    with pytest.raises(ValueError, match="magnitude/prevalence"):
        validate_factorial_trial_receipt(payload)


def test_factorial_receipt_store_resumes_exact_bytes_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    payload = _receipt(tmp_path, injected=True)
    store = FactorialReceiptStore(tmp_path / "receipts")
    first_sha256 = store.write(payload)
    assert store.write(payload) == first_sha256
    changed = dict(payload)
    changed["feature_6779_magnitude"] = 0.3
    with pytest.raises(ValueError, match="refusing completed receipt overwrite"):
        store.write(changed)

    loaded = store.load_validated(
        payload["trial_id"],
        public_plan_sha256=payload["public_plan_sha256"],
        private_plan_sha256=payload["private_plan_sha256"],
        assay_receipt_sha256=payload["assay_receipt_sha256"],
        source_commit=payload["source_commit"],
        run_id=payload["run_id"],
    )
    assert loaded is not None
    assert loaded.feature_6779_magnitude == 0.25


def test_factorial_receipt_store_rejects_restricted_artifact_tamper(
    tmp_path: Path,
) -> None:
    payload = _receipt(tmp_path)
    restricted = Path(payload["restricted_artifact_path"])
    restricted.write_text("{}")
    with pytest.raises(ValueError, match="restricted factorial artifact hash"):
        validate_factorial_trial_receipt(payload)
