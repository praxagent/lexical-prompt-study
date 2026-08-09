from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .weaponization_plan import (
    validate_weaponization_plan,
    weaponization_protocol_sha256,
)


class WeaponizationPrefillReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-weaponization-breaker-v1"]
    status: Literal["complete"] = "complete"
    run_id: str
    source_commit: str
    public_plan_sha256: str
    private_topology_sha256: str
    partition: Literal["detector_calibration", "detector_confirmation"]
    trial_id: str
    stratum: str
    critical: bool
    request_class: str
    request_id: str
    request_sha256: str
    material: str
    material_sha256: str | None
    wrapper_id: str | None
    placement: str | None
    injected_token_count: int = Field(ge=0)
    prompt_sha256: str
    prompt_token_ids_sha256: str
    attempt: int = Field(ge=1, le=2)
    feature_6779_magnitude: float
    frozen_subspace_score: float
    sae_normalized_reconstruction_error: float = Field(ge=0)
    jlens_refusal_minus_compliance_trajectory: list[float]
    restricted_exact_match: bool
    structural_metrics: dict[str, float]
    prefill_latency_ms: float = Field(ge=0)
    detector_readout_latency_ms: float = Field(ge=0)
    peak_gpu_memory_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_topology_and_finiteness(self) -> WeaponizationPrefillReceipt:
        for field in (
            "source_commit",
            "public_plan_sha256",
            "private_topology_sha256",
            "request_sha256",
            "prompt_sha256",
            "prompt_token_ids_sha256",
        ):
            value = getattr(self, field)
            if len(value) not in (40, 64) or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid hexadecimal provenance field: {field}")
        if len(self.jlens_refusal_minus_compliance_trajectory) != 31:
            raise ValueError("weaponization J-lens trajectory must contain 31 source layers")
        floats = [
            self.feature_6779_magnitude,
            self.frozen_subspace_score,
            self.sae_normalized_reconstruction_error,
            self.prefill_latency_ms,
            self.detector_readout_latency_ms,
            *self.jlens_refusal_minus_compliance_trajectory,
            *self.structural_metrics.values(),
        ]
        if any(not math.isfinite(value) for value in floats):
            raise ValueError("weaponization receipt contains a non-finite value")
        if self.material == "no_scaffold":
            if (
                self.placement is not None
                or self.material_sha256 is not None
                or self.wrapper_id is not None
                or self.injected_token_count != 0
            ):
                raise ValueError("no-scaffold weaponization receipt topology drift")
        elif (
            self.placement
            not in {"scaffold_before_request", "scaffold_after_request"}
            or self.material_sha256 is None
            or self.injected_token_count != 252
        ):
            raise ValueError("injected weaponization receipt topology drift")
        if self.material == "harmless_structured_wrapper" and self.wrapper_id is None:
            raise ValueError("harmless receipt must bind a wrapper ID")
        return self


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class WeaponizationReceiptStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.receipt_root = root / "receipts"
        self.attempt_log = root / "attempts.jsonl"
        self.receipt_root.mkdir(parents=True, exist_ok=True)
        self.receipt_root.chmod(0o700)

    def _path(self, trial_id: str) -> Path:
        return self.receipt_root / f"{trial_id}.json"

    def load(
        self,
        trial_id: str,
        *,
        provenance: Mapping[str, str],
    ) -> WeaponizationPrefillReceipt | None:
        path = self._path(trial_id)
        if not path.exists():
            return None
        receipt = WeaponizationPrefillReceipt.model_validate_json(path.read_text())
        for field, expected in provenance.items():
            if getattr(receipt, field) != expected:
                raise ValueError(f"{trial_id}: weaponization receipt provenance drift: {field}")
        return receipt

    def write(self, receipt: WeaponizationPrefillReceipt | Mapping[str, Any]) -> str:
        validated = (
            receipt
            if isinstance(receipt, WeaponizationPrefillReceipt)
            else WeaponizationPrefillReceipt.model_validate(receipt)
        )
        payload = canonical_json_bytes(validated.model_dump(mode="json"))
        path = self._path(validated.trial_id)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"{validated.trial_id}: refusing weaponization overwrite")
            return sha256_bytes(payload)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
        with self.attempt_log.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.attempt_log.chmod(0o600)
        return sha256_bytes(payload)


BatchExecutor = Callable[[Sequence[dict[str, Any]], int], Sequence[Mapping[str, Any]]]


def _validate_authorization(
    authorization: Mapping[str, Any],
    *,
    public_plan_sha256: str,
    private_topology_sha256: str,
    partition: str,
    run_id: str,
    source_commit: str,
    expected_observations: int,
) -> None:
    if (
        authorization.get("schema_version") != "1.0"
        or authorization.get("study_id")
        != "lexical-scaffold-weaponization-breaker-v1"
        or authorization.get("status") != "weaponization_prefill_authorized"
        or authorization.get("paid_compute_authorized") is not True
        or authorization.get("human_semantic_review_approved") is not True
        or authorization.get("partition") != partition
        or authorization.get("run_id") != run_id
        or authorization.get("expected_observations") != expected_observations
    ):
        raise ValueError("weaponization execution authorization boundary drift")
    bindings = authorization.get("bindings", {})
    expected = {
        "public_plan_sha256": public_plan_sha256,
        "private_topology_sha256": private_topology_sha256,
        "source_commit": source_commit,
    }
    if any(bindings.get(field) != value for field, value in expected.items()):
        raise ValueError("weaponization execution authorization binding drift")
    if authorization.get("single_task_owned_pod_maximum") != 1:
        raise ValueError("weaponization authorization permits duplicate pods")
    if authorization.get("hard_ceiling_usd") != 200:
        raise ValueError("weaponization authorization hard ceiling drift")


def run_weaponization_prefill_loop(
    *,
    public_plan_path: Path,
    private_topology_path: Path,
    authorization_path: Path,
    output_root: Path,
    run_id: str,
    batch_size: int,
    execute_batch: BatchExecutor,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("weaponization batch size must be positive")
    public_plan = json.loads(public_plan_path.read_text())
    validate_weaponization_plan(public_plan, root=public_plan_path.resolve().parents[1])
    topology = json.loads(private_topology_path.read_text())
    public_plan_sha256 = sha256_file(public_plan_path)
    private_topology_sha256 = sha256_file(private_topology_path)
    source_commit = _source_commit()
    if (
        topology["study_id"] != public_plan["study_id"]
        or topology["protocol_sha256"] != weaponization_protocol_sha256(public_plan)
        or topology["status"] != "topology_frozen_human_reviewed"
        or topology["human_semantic_review_approved"] is not True
        or topology["prefill_only"] is not True
    ):
        raise ValueError("weaponization private topology is not execution eligible")
    partition = topology["partition"]
    observations = sorted(topology["observations"], key=lambda row: row["trial_id"])
    authorization = json.loads(authorization_path.read_text())
    _validate_authorization(
        authorization,
        public_plan_sha256=public_plan_sha256,
        private_topology_sha256=private_topology_sha256,
        partition=partition,
        run_id=run_id,
        source_commit=source_commit,
        expected_observations=len(observations),
    )
    store = WeaponizationReceiptStore(output_root)
    provenance = {
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_plan_sha256,
        "private_topology_sha256": private_topology_sha256,
        "partition": partition,
    }
    missing = [
        row
        for row in observations
        if store.load(row["trial_id"], provenance=provenance) is None
    ]
    written = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        results: Sequence[Mapping[str, Any]] | None = None
        error: Exception | None = None
        for attempt in (1, 2):
            try:
                results = execute_batch(batch, attempt)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - deterministic retry boundary
                error = exc
        if results is None:
            assert error is not None
            raise error
        by_trial = {str(result["trial_id"]): result for result in results}
        if set(by_trial) != {str(row["trial_id"]) for row in batch}:
            raise ValueError("weaponization batch executor topology drift")
        for row in batch:
            result = dict(by_trial[row["trial_id"]])
            result.update(
                {
                    "schema_version": "1.0",
                    "study_id": public_plan["study_id"],
                    "status": "complete",
                    **provenance,
                    "trial_id": row["trial_id"],
                    "stratum": row["stratum"],
                    "critical": row["critical"],
                    "request_class": row["request_class"],
                    "request_id": row["request_id"],
                    "request_sha256": row["request_sha256"],
                    "material": row["material"],
                    "material_sha256": row["material_sha256"],
                    "wrapper_id": row["wrapper_id"],
                    "placement": row["placement"],
                    "injected_token_count": row["injected_token_count"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                    "attempt": int(result.get("attempt", 1)),
                }
            )
            store.write(result)
            written += 1
    receipt_paths = sorted(store.receipt_root.glob("*.json"))
    if len(receipt_paths) != len(observations):
        raise ValueError("weaponization run incomplete after receipt loop")
    manifest = [
        {"trial_id": path.stem, "receipt_sha256": sha256_file(path)}
        for path in receipt_paths
    ]
    summary = {
        "schema_version": "1.0",
        "study_id": public_plan["study_id"],
        "status": "complete",
        **provenance,
        "observation_count": len(observations),
        "written_this_invocation": written,
        "resumed_receipt_count": len(observations) - written,
        "receipt_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "raw_prompt_or_token_ids_public": False,
    }
    summary_path = output_root / "summary.json"
    encoded = canonical_json_bytes(summary)
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(summary_path)
    return summary
