from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .factorial_authorization import validate_factorial_execution_authorization
from .factorial_plan import validate_factorial_plan
from .factorial_receipts import FactorialReceiptStore
from .hashing import canonical_json_bytes, sha256_file, sha256_text
from .models import FactorialAssayReceipt

ObservationExecutor = Callable[[dict[str, Any], int], dict[str, Any]]


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _canonical_observations(private_plan: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = [dose for dose in private_plan["doses"] if dose["canonical"]]
    if len(canonical) != 1:
        raise ValueError("factorial private plan lacks one canonical dose")
    canonical_size_id = canonical[0]["size_id"]
    rows = [
        row
        for row in private_plan["observations"]
        if row["material"] == "no_scaffold" or row["size_id"] == canonical_size_id
    ]
    if len(rows) != 422:
        raise ValueError("canonical factorial observation topology drift")
    return rows


def _failure_path(output_root: Path, trial_id: str) -> Path:
    return output_root / "failures" / f"{trial_id}.json"


def _load_failure(
    output_root: Path,
    trial_id: str,
    *,
    public_plan_sha256: str,
    private_plan_sha256: str,
    assay_receipt_sha256: str,
    source_commit: str,
    run_id: str,
) -> dict[str, Any] | None:
    path = _failure_path(output_root, trial_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = {
        "trial_id": trial_id,
        "public_plan_sha256": public_plan_sha256,
        "private_plan_sha256": private_plan_sha256,
        "assay_receipt_sha256": assay_receipt_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "attempts_exhausted": 2,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{trial_id}: failure receipt provenance drift for {field}")
    return payload


def _write_failure(
    output_root: Path,
    *,
    trial_id: str,
    public_plan_sha256: str,
    private_plan_sha256: str,
    assay_receipt_sha256: str,
    source_commit: str,
    run_id: str,
    error: Exception,
) -> None:
    path = _failure_path(output_root, trial_id)
    payload = {
        "schema_version": "1.0",
        "status": "missing_after_two_deterministic_attempts",
        "trial_id": trial_id,
        "public_plan_sha256": public_plan_sha256,
        "private_plan_sha256": private_plan_sha256,
        "assay_receipt_sha256": assay_receipt_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "attempts_exhausted": 2,
        "error_type": type(error).__name__,
        "error_message_sha256": sha256_text(str(error)),
    }
    encoded = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_factorial_canonical(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    assay_receipt_path: Path,
    authorization_path: Path,
    output_root: Path,
    run_id: str,
    execute_observation: ObservationExecutor,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_factorial_plan(public_plan)
    private_plan = json.loads(private_plan_path.read_text())
    public_sha256 = sha256_file(public_plan_path)
    private_sha256 = sha256_file(private_plan_path)
    if (
        private_plan["study_id"] != public_plan["study_id"]
        or private_plan["public_plan_sha256"] != public_sha256
    ):
        raise ValueError("factorial private/public plan binding drift")
    assay = FactorialAssayReceipt.model_validate_json(assay_receipt_path.read_text())
    assay_sha256 = sha256_file(assay_receipt_path)
    if (
        assay.status != "passed"
        or assay.public_plan_sha256 != public_sha256
        or assay.private_plan_sha256 != private_sha256
        or assay.target_factorial_outcome_generated is not False
    ):
        raise ValueError("factorial assay gate has not passed for exact inputs")
    source_commit = _source_commit()
    authorization = json.loads(authorization_path.read_text())
    validate_factorial_execution_authorization(
        authorization,
        expected_public_plan_sha256=public_sha256,
        expected_private_plan_sha256=private_sha256,
        expected_source_commit=source_commit,
        expected_stage="canonical_factorial",
    )
    if authorization["bindings"]["assay_receipt_sha256"] != assay_sha256:
        raise ValueError("factorial authorization assay binding drift")
    if authorization["run_id"] != run_id:
        raise ValueError("factorial authorization run-ID drift")

    observations = _canonical_observations(private_plan)
    store = FactorialReceiptStore(output_root / "receipts")
    completed = 0
    written = 0
    missing = 0
    for observation in observations:
        trial_id = observation["trial_id"]
        existing = store.load_validated(
            trial_id,
            public_plan_sha256=public_sha256,
            private_plan_sha256=private_sha256,
            assay_receipt_sha256=assay_sha256,
            source_commit=source_commit,
            run_id=run_id,
        )
        if existing is not None:
            completed += 1
            continue
        failure = _load_failure(
            output_root,
            trial_id,
            public_plan_sha256=public_sha256,
            private_plan_sha256=private_sha256,
            assay_receipt_sha256=assay_sha256,
            source_commit=source_commit,
            run_id=run_id,
        )
        if failure is not None:
            missing += 1
            continue
        result = None
        attempt = 0
        for attempt in (1, 2):
            try:
                result = execute_observation(observation, attempt)
                break
            except Exception as error:
                if attempt == 2:
                    _write_failure(
                        output_root,
                        trial_id=trial_id,
                        public_plan_sha256=public_sha256,
                        private_plan_sha256=private_sha256,
                        assay_receipt_sha256=assay_sha256,
                        source_commit=source_commit,
                        run_id=run_id,
                        error=error,
                    )
                    missing += 1
        if result is None:
            continue
        immutable = {
            "schema_version": "1.0",
            "study_id": public_plan["study_id"],
            "status": "complete",
            "public_plan_sha256": public_sha256,
            "private_plan_sha256": private_sha256,
            "assay_receipt_sha256": assay_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "trial_id": trial_id,
            "attempt": attempt,
            "request_class": observation["request_class"],
            "request_id": observation["request_id"],
            "prompt_family_id": observation["prompt_family_id"],
            "request_sha256": observation["request_sha256"],
            "material": observation["material"],
            "placement": observation["placement"],
            "size_id": observation["size_id"],
            "injected_token_count": observation["injected_token_count"],
            "shared_reference": observation["shared_reference"],
            "render_group_sha256": observation.get("render_group_sha256"),
            "prompt_sha256": observation["prompt_sha256"],
            "prompt_token_ids_sha256": observation["prompt_token_ids_sha256"],
        }
        overlap = set(immutable).intersection(result)
        if overlap:
            raise ValueError(
                f"{trial_id}: executor attempted to override provenance {sorted(overlap)}"
            )
        store.write({**immutable, **result})
        written += 1
    final_count = completed + written
    if final_count + missing != len(observations):
        raise ValueError("factorial canonical resolved-unit count drift")
    return {
        "schema_version": "1.0",
        "status": "complete" if missing == 0 else "incomplete",
        "stage": "canonical_factorial",
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_sha256,
        "private_plan_sha256": private_sha256,
        "assay_receipt_sha256": assay_sha256,
        "planned_observations": len(observations),
        "preexisting_receipts": completed,
        "receipts_written_this_call": written,
        "final_receipt_count": final_count,
        "missing_after_two_attempts": missing,
        "detector_threshold_fit": False,
        "placement_pooling": False,
        "size_pooling": False,
    }
