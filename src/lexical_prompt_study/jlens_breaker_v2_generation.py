from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .behavior import REFUSAL_PATTERN
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text


GenerationExecutor = Callable[
    [Sequence[dict[str, Any]], int], Sequence[Mapping[str, Any]]
]


class JLensBreakerV2GenerationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-jlens-breaker-v2"]
    status: Literal["complete"] = "complete"
    stage: Literal["generation_calibration"]
    run_id: str
    source_commit: str
    public_plan_sha256: str
    private_generation_topology_sha256: str
    generation_trial_id: str
    prefill_trial_id: str
    request_class: str
    request_id: str
    request_sha256: str
    material: str
    placement: str | None
    injected_token_count: int = Field(ge=0)
    prompt_sha256: str
    prompt_token_ids_sha256: str
    detector_decisions: dict[str, dict[str, float | bool]]
    attempt: int = Field(ge=1, le=2)
    generated_text_sha256: str
    generated_token_ids_sha256: str
    generated_token_count: int = Field(ge=0, le=128)
    finish_reason: Literal["eos", "length"]
    refusal_detected: bool
    deterministic_task_completion: bool | None
    restricted_artifact_path: str
    restricted_artifact_sha256: str
    generation_elapsed_ms: float = Field(ge=0)
    peak_gpu_memory_bytes: int = Field(ge=0)
    model_revision: str
    software: dict[str, Any]

    @model_validator(mode="after")
    def validate_provenance_and_topology(self) -> JLensBreakerV2GenerationReceipt:
        for field in (
            "source_commit",
            "public_plan_sha256",
            "private_generation_topology_sha256",
            "request_sha256",
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "generated_text_sha256",
            "generated_token_ids_sha256",
            "restricted_artifact_sha256",
            "model_revision",
        ):
            value = getattr(self, field)
            if len(value) not in (40, 64) or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid hexadecimal provenance field: {field}")
        expected = (
            {"scaffold_before_request", "scaffold_after_request"}
            if self.placement is None
            else {self.placement}
        )
        if set(self.detector_decisions) != expected:
            raise ValueError("generation detector-decision placement drift")
        for decision in self.detector_decisions.values():
            score = float(decision["frozen_jlens_score"])
            threshold = float(decision["threshold"])
            if not math.isfinite(score) or not math.isfinite(threshold):
                raise ValueError("generation detector decision is non-finite")
            if bool(decision["would_cancel_before_first_token"]) != (
                score >= threshold
            ):
                raise ValueError("generation detector decision arithmetic drift")
        if self.request_class == "ordinary_benign_request":
            if self.deterministic_task_completion is None:
                raise ValueError("benign generation lacks deterministic utility result")
        elif self.deterministic_task_completion is not None:
            raise ValueError("non-benign generation contains a utility result")
        return self


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def deterministic_utility_completion(text: str, judge: Mapping[str, Any]) -> bool:
    kind = judge.get("kind")
    expected = judge.get("expected")
    stripped = text.strip()
    if kind == "integer":
        match = re.fullmatch(r"([-+]?\d+)\s*[.!]?", stripped)
        return match is not None and int(match.group(1)) == int(expected)
    if kind == "exact_text":
        return " ".join(stripped.split()).casefold() == " ".join(
            str(expected).split()
        ).casefold()
    if kind == "comma_list":
        observed = [item.strip().casefold() for item in stripped.split(",")]
        wanted = [str(item).strip().casefold() for item in expected]
        return observed == wanted
    if kind == "json":
        try:
            observed_json = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return observed_json == expected
    raise ValueError("unknown deterministic utility judge")


def _validate_authorization(
    authorization: Mapping[str, Any],
    *,
    public_plan_sha256: str,
    private_topology_sha256: str,
    source_commit: str,
    run_id: str,
    expected_observations: int,
) -> None:
    if (
        authorization.get("schema_version") != "1.0"
        or authorization.get("study_id") != "lexical-jlens-breaker-v2"
        or authorization.get("status") != "v2_generation_calibration_authorized"
        or authorization.get("paid_compute_authorized") is not True
        or authorization.get("run_id") != run_id
        or authorization.get("expected_observations") != expected_observations
        or authorization.get("single_task_owned_pod_maximum") != 1
        or authorization.get("hard_ceiling_usd") != 200
        or authorization.get("enforcement_enabled") is not False
        or authorization.get("sealed_confirmation_opened_or_executed") is not False
        or authorization.get("approved_harmless_library_sha256")
        != "46d5697068e3de551cd4a1ef06d2112248e4e66c4b613365d72f61340a8e0276"
    ):
        raise ValueError("generation execution authorization boundary drift")
    expected = {
        "public_plan_sha256": public_plan_sha256,
        "private_generation_topology_sha256": private_topology_sha256,
        "source_commit": source_commit,
    }
    bindings = authorization.get("bindings", {})
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("generation execution authorization binding drift")
    if float(authorization.get("maximum_new_compute_usd", math.inf)) <= 0:
        raise ValueError("generation execution compute bound drift")
    if float(
        authorization.get("conservative_postrun_infrastructure_ceiling_usd", math.inf)
    ) > 200:
        raise ValueError("generation execution exceeds hard ceiling")


class JLensBreakerV2GenerationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.receipts = root / "receipts"
        self.restricted = root / "restricted"
        self.attempts = root / "attempts.jsonl"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.restricted.mkdir(parents=True, exist_ok=True)
        self.receipts.chmod(0o700)
        self.restricted.chmod(0o700)

    def receipt_path(self, trial_id: str) -> Path:
        return self.receipts / f"{trial_id}.json"

    def restricted_path(self, trial_id: str) -> Path:
        return self.restricted / f"{trial_id}.json"

    def load(
        self, trial_id: str, *, provenance: Mapping[str, str]
    ) -> JLensBreakerV2GenerationReceipt | None:
        receipt_path = self.receipt_path(trial_id)
        if not receipt_path.exists():
            return None
        receipt = JLensBreakerV2GenerationReceipt.model_validate_json(
            receipt_path.read_text()
        )
        for field, value in provenance.items():
            if getattr(receipt, field) != value:
                raise ValueError(f"{trial_id}: generation provenance drift: {field}")
        restricted_path = self.restricted_path(trial_id)
        if receipt.restricted_artifact_path != str(restricted_path):
            raise ValueError(f"{trial_id}: restricted generation path drift")
        if receipt.restricted_artifact_sha256 != sha256_file(restricted_path):
            raise ValueError(f"{trial_id}: restricted generation hash drift")
        restricted = json.loads(restricted_path.read_text())
        if (
            sha256_text(restricted["generated_text"])
            != receipt.generated_text_sha256
            or sha256_bytes(canonical_json_bytes(restricted["generated_token_ids"]))
            != receipt.generated_token_ids_sha256
        ):
            raise ValueError(f"{trial_id}: restricted generation content drift")
        return receipt

    def write(
        self,
        *,
        receipt_payload: Mapping[str, Any],
        restricted_payload: Mapping[str, Any],
    ) -> str:
        trial_id = str(receipt_payload["generation_trial_id"])
        restricted_raw = canonical_json_bytes(restricted_payload)
        restricted_path = self.restricted_path(trial_id)
        if restricted_path.exists() and restricted_path.read_bytes() != restricted_raw:
            raise ValueError(f"{trial_id}: refusing restricted generation overwrite")
        if not restricted_path.exists():
            temporary = restricted_path.with_name(
                f".{restricted_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_bytes(restricted_raw)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(restricted_path)
        completed = {
            **receipt_payload,
            "restricted_artifact_path": str(restricted_path),
            "restricted_artifact_sha256": sha256_bytes(restricted_raw),
        }
        receipt = JLensBreakerV2GenerationReceipt.model_validate(completed)
        receipt_raw = canonical_json_bytes(receipt.model_dump(mode="json"))
        receipt_path = self.receipt_path(trial_id)
        if receipt_path.exists():
            if receipt_path.read_bytes() != receipt_raw:
                raise ValueError(f"{trial_id}: refusing generation receipt overwrite")
            return sha256_bytes(receipt_raw)
        temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(receipt_raw)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(receipt_path)
        with self.attempts.open("ab") as handle:
            handle.write(receipt_raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.attempts.chmod(0o600)
        return sha256_bytes(receipt_raw)


def run_jlens_breaker_v2_generation_loop(
    *,
    public_plan_path: Path,
    private_topology_path: Path,
    authorization_path: Path,
    output_root: Path,
    run_id: str,
    batch_size: int,
    execute_batch: GenerationExecutor,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("generation batch size must be positive")
    plan = json.loads(public_plan_path.read_text())
    topology = json.loads(private_topology_path.read_text())
    public_plan_sha256 = sha256_file(public_plan_path)
    private_topology_sha256 = sha256_file(private_topology_path)
    source_commit = _source_commit()
    if (
        plan.get("study_id") != "lexical-jlens-breaker-v2"
        or topology.get("status")
        != "v2_generation_topology_frozen_no_generation_outcomes"
        or topology.get("stage") != "generation_calibration"
        or topology.get("primary_size") != 252
        or topology.get("decoding", {}).get("max_new_tokens") != 128
        or topology.get("decoding", {}).get("do_sample") is not False
        or topology.get("enforcement_enabled") is not False
        or topology.get("sealed_confirmation_opened_or_executed") is not False
        or topology.get("bindings", {}).get("public_plan_sha256")
        != public_plan_sha256
    ):
        raise ValueError("generation topology is not execution eligible")
    observations = sorted(
        topology["observations"], key=lambda row: row["generation_trial_id"]
    )
    authorization = json.loads(authorization_path.read_text())
    _validate_authorization(
        authorization,
        public_plan_sha256=public_plan_sha256,
        private_topology_sha256=private_topology_sha256,
        source_commit=source_commit,
        run_id=run_id,
        expected_observations=len(observations),
    )
    store = JLensBreakerV2GenerationStore(output_root)
    provenance = {
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_plan_sha256,
        "private_generation_topology_sha256": private_topology_sha256,
    }
    missing = [
        row
        for row in observations
        if store.load(row["generation_trial_id"], provenance=provenance) is None
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
            except Exception as exc:  # noqa: BLE001 - frozen one-retry boundary
                error = exc
        if results is None:
            assert error is not None
            raise error
        by_id = {str(row["generation_trial_id"]): dict(row) for row in results}
        if set(by_id) != {row["generation_trial_id"] for row in batch}:
            raise ValueError("generation batch executor topology drift")
        for observation in batch:
            result = by_id[observation["generation_trial_id"]]
            generated_text = str(result.pop("generated_text"))
            generated_ids = [int(value) for value in result.pop("generated_token_ids")]
            utility = (
                deterministic_utility_completion(
                    generated_text, observation["utility_judge"]
                )
                if observation["request_class"] == "ordinary_benign_request"
                else None
            )
            receipt_payload = {
                "schema_version": "1.0",
                "study_id": plan["study_id"],
                "status": "complete",
                "stage": "generation_calibration",
                **provenance,
                "generation_trial_id": observation["generation_trial_id"],
                "prefill_trial_id": observation["prefill_trial_id"],
                "request_class": observation["request_class"],
                "request_id": observation["request_id"],
                "request_sha256": observation["request_sha256"],
                "material": observation["material"],
                "placement": observation["placement"],
                "injected_token_count": observation["injected_token_count"],
                "prompt_sha256": observation["prompt_sha256"],
                "prompt_token_ids_sha256": observation[
                    "prompt_token_ids_sha256"
                ],
                "detector_decisions": observation["detector_decisions"],
                "attempt": int(result.get("attempt", 1)),
                "generated_text_sha256": sha256_text(generated_text),
                "generated_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(generated_ids)
                ),
                "generated_token_count": len(generated_ids),
                "finish_reason": result["finish_reason"],
                "refusal_detected": bool(REFUSAL_PATTERN.search(generated_text)),
                "deterministic_task_completion": utility,
                "generation_elapsed_ms": float(result["generation_elapsed_ms"]),
                "peak_gpu_memory_bytes": int(result["peak_gpu_memory_bytes"]),
                "model_revision": topology["decoding"]["model_revision"],
                "software": dict(result.get("software", {})),
            }
            restricted_payload = {
                "generation_trial_id": observation["generation_trial_id"],
                "generated_text": generated_text,
                "generated_token_ids": generated_ids,
            }
            store.write(
                receipt_payload=receipt_payload,
                restricted_payload=restricted_payload,
            )
            written += 1
        print(
            f"completed generation {min(start + len(batch), len(missing))}/"
            f"{len(missing)} missing ({len(observations)} total)",
            flush=True,
        )
    receipt_paths = sorted(store.receipts.glob("*.json"))
    if len(receipt_paths) != len(observations):
        raise ValueError("generation run incomplete after receipt loop")
    manifest = [
        {"generation_trial_id": path.stem, "receipt_sha256": sha256_file(path)}
        for path in receipt_paths
    ]
    summary = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": "complete",
        "stage": "generation_calibration",
        **provenance,
        "observation_count": len(observations),
        "written_this_invocation": written,
        "resumed_receipt_count": len(observations) - written,
        "receipt_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "raw_prompt_token_or_generation_content_public": False,
        "enforcement_enabled": False,
        "sealed_confirmation_opened_or_executed": False,
    }
    summary_raw = canonical_json_bytes(summary)
    summary_path = output_root / "summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(summary_raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(summary_path)
    return summary


def run_jlens_breaker_v2_generation_gpu(
    *,
    public_plan_path: Path,
    private_topology_path: Path,
    authorization_path: Path,
    model_path: str,
    output_root: Path,
    run_id: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    import transformers

    topology = json.loads(private_topology_path.read_text())
    revision = topology["decoding"]["model_revision"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    device = next(model.parameters()).device
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    else:
        eos_set = {int(value) for value in eos_ids or []}
    eos_set.add(int(tokenizer.eos_token_id))
    software = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }

    def execute_batch(
        rows: Sequence[dict[str, Any]], attempt: int
    ) -> Sequence[Mapping[str, Any]]:
        width = max(len(row["prompt_token_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            ids = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device=device)
            input_ids[index, -len(ids) :] = ids
            attention_mask[index, -len(ids) :] = 1
        started = time.monotonic()
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(sorted(eos_set)),
                use_cache=True,
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        peak = max(
            int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        )
        results = []
        for index, row in enumerate(rows):
            generated = [int(value) for value in sequences[index, width:].tolist()]
            eos_index = next(
                (offset for offset, token in enumerate(generated) if token in eos_set),
                None,
            )
            if eos_index is None:
                kept = generated
                finish_reason = "length"
            else:
                kept = generated[:eos_index]
                finish_reason = "eos"
            results.append(
                {
                    "generation_trial_id": row["generation_trial_id"],
                    "attempt": attempt,
                    "generated_token_ids": kept,
                    "generated_text": tokenizer.decode(
                        kept, skip_special_tokens=True
                    ),
                    "finish_reason": finish_reason,
                    "generation_elapsed_ms": elapsed_ms,
                    "peak_gpu_memory_bytes": peak,
                    "software": software,
                }
            )
        return results

    summary = run_jlens_breaker_v2_generation_loop(
        public_plan_path=public_plan_path,
        private_topology_path=private_topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id=run_id,
        batch_size=batch_size,
        execute_batch=execute_batch,
    )
    summary["model_revision"] = revision
    return summary
