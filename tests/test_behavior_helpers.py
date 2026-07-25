from pathlib import Path

import pytest

from lexical_prompt_study.behavior import (
    REFUSAL_PATTERN,
    _load_completed_generation,
    _save_restricted,
)
from lexical_prompt_study.hashing import sha256_text
from lexical_prompt_study.models import TrialReceipt
from lexical_prompt_study.receipts import ReceiptStore
from lexical_prompt_study.synthetic import run_synthetic


def test_refusal_pattern() -> None:
    assert REFUSAL_PATTERN.search("I'm sorry, but I can't help with that.")
    assert not REFUSAL_PATTERN.search("Here is a harmless synthetic answer.")


def test_completed_generation_is_bound_to_restricted_artifact(tmp_path: Path) -> None:
    receipt_root = tmp_path / "receipts"
    run_synthetic(receipt_root, n=1)
    receipt_path = next((receipt_root / "trials").glob("*.json"))
    receipt = TrialReceipt.model_validate_json(receipt_path.read_text())
    raw_path = tmp_path / "restricted" / f"{receipt.trial_id}.json"
    payload = {
        "generated_text": "safe synthetic output",
        "generated_token_ids": receipt.generated_token_ids,
    }
    artifact_hash = _save_restricted(raw_path, payload)
    receipt = receipt.model_copy(
        update={
            "generated_text_sha256": sha256_text(payload["generated_text"]),
            "restricted_text_path": str(raw_path),
            "restricted_artifact_sha256": artifact_hash,
        }
    )
    store = ReceiptStore(receipt_root)
    store.write(receipt)

    assert (
        _load_completed_generation(
            store,
            receipt.trial_id,
            raw_path,
            expected_plan_sha256=receipt.plan_sha256,
            expected_run_id=receipt.run_id,
        )
        == payload
    )
    with pytest.raises(ValueError, match="plan hash drift"):
        _load_completed_generation(
            store,
            receipt.trial_id,
            raw_path,
            expected_plan_sha256="f" * 64,
            expected_run_id=receipt.run_id,
        )
    raw_path.write_text('{"generated_text":"drift","generated_token_ids":[]}\n')
    with pytest.raises(ValueError, match="restricted artifact hash mismatch"):
        _load_completed_generation(
            store,
            receipt.trial_id,
            raw_path,
            expected_plan_sha256=receipt.plan_sha256,
            expected_run_id=receipt.run_id,
        )
