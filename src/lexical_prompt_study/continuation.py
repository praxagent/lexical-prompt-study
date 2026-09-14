"""Outcome-blind A142 continuation selection and crash-safe token checkpoints.

This module never decodes tokens and has no network or model dependencies.  Token
lists belong only in restricted artifacts.  The plan is private (row identifiers
are not public); ``public_plan_summary`` is the aggregate-only projection.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file


A142_SUMMARY_SHA256 = "bc0038ba514598d9c836181f3ccbb41c0437e6b433855e2b0fbcc299757c38fc"
A142_MANIFEST_SHA256 = "7beac6a624ad80a7ad35f538817646d2afba2b1113b87de82868f8af21bd314b"
A142_TOPOLOGY_SHA256 = "52aabffe282219c9e39fce2f24890a606c2db22044d3e8be38712ba1837ff4bc"
A142_PLAN_SHA256 = "699e5168034109ad21768db5c37ebaab643e64faca9550a9af687538d8ac07da"
A142_SOURCE_COMMIT = "8507cc8e57d987b4c759e0897232f03d44b55b16"
CHECKPOINTS = (128, 256, 512, 1024)
STUDY_ID = "lexical-a142-censored-continuation-v1"
INTENTS = frozenset(
    ("unsafe_direct", "safe_classify_exact", "safe_refuse_exact", "safe_acknowledge_exact")
)
FAMILIES = frozenset(
    ("attack_block_mask", "structural_sham", "harmless_structured_wrapper", "no_scaffold")
)
PLACEMENTS = frozenset((None, "scaffold_before_request", "scaffold_after_request"))
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FIELDS = (
    "trial_id",
    "request_core_id",
    "request_core_sha256",
    "intent_frame",
    "safe_intent",
    "variant_family",
    "attack_block_count",
    "attack_block_mask",
    "placement",
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "generated_token_ids_sha256",
    "generated_text_sha256",
    "restricted_artifact_sha256",
    "generated_token_count",
    "finish_reason",
)


def token_hash(token_ids: Sequence[int]) -> str:
    """Hash the exact canonical token list, matching the original acquisition."""
    return sha256_bytes(canonical_json_bytes(_tokens(token_ids)))


def _tokens(value: Sequence[int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise ValueError("token sequence must contain non-negative integers")
    return list(value)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("invalid private row identifier")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("invalid sha256 binding")
    return value


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return row["intent_frame"], row["placement"] or "no_scaffold", row["attack_block_count"]


def _row_order(row: Mapping[str, Any]) -> str:
    # Use only pre-generation identities.  The receipt hash also incorporates
    # outcomes, so it binds provenance but does not determine pilot selection.
    return sha256_bytes(
        canonical_json_bytes(
            [
                STUDY_ID,
                row["trial_id"],
                row["request_core_sha256"],
                row["prompt_token_ids_sha256"],
            ]
        )
    )


def select_pilot(rows: Sequence[Mapping[str, Any]], size: int = 64) -> list[str]:
    """One hash-selected row per available stratum, then round robin.

    A small first pass reserves one slot per family and available placement.
    This prevents the much larger attack-mask family from excluding the rarer
    unchanged controls or observing them in only one ordering.
    Selection is descriptive and balanced, not a population-weighted estimate.
    """
    if type(size) is not int or not 1 <= size <= 64:
        raise ValueError("pilot size must be between 1 and 64")
    ordered = sorted(rows, key=_row_order)
    if len({row["trial_id"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate pilot input rows")
    families = sorted({(row["variant_family"], row["placement"] or "no_scaffold") for row in rows})
    if size < len(families):
        raise ValueError("pilot too small to cover available family placements")
    selected: list[str] = []
    for family, placement in families:
        selected.append(
            next(
                row["trial_id"]
                for row in ordered
                if row["variant_family"] == family
                and (row["placement"] or "no_scaffold") == placement
            )
        )
    selected_set = set(selected)
    strata: dict[tuple[str, str, int], deque[Mapping[str, Any]]] = defaultdict(deque)
    for row in ordered:
        if row["trial_id"] not in selected_set:
            strata[_stratum(row)].append(row)
    # First cover strata absent from the family reservation, then keep counts
    # as even as available stratum sizes permit.
    counts = Counter(_stratum(row) for row in ordered if row["trial_id"] in selected_set)
    while len(selected) < min(size, len(ordered)):
        available = [key for key, queue in strata.items() if queue]
        if not available:
            break
        key = min(available, key=lambda item: (counts[item], item))
        selected.append(strata[key].popleft()["trial_id"])
        counts[key] += 1
    return selected


def _project_receipt(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    row = {key: receipt[key] for key in _ROW_FIELDS}
    _identifier(row["trial_id"])
    _identifier(row["request_core_id"])
    for key, value in row.items():
        if key.endswith("sha256"):
            _hash(value)
    if (
        row["intent_frame"] not in INTENTS
        or row["variant_family"] not in FAMILIES
        or row["placement"] not in PLACEMENTS
        or type(row["attack_block_count"]) is not int
        or not 0 <= row["attack_block_count"] <= 4
        or type(row["safe_intent"]) is not bool
        or row["safe_intent"] != (row["intent_frame"] != "unsafe_direct")
        or type(row["generated_token_count"]) is not int
        or not 0 <= row["generated_token_count"] <= 128
        or row["finish_reason"] not in ("eos", "length")
    ):
        raise ValueError("original numeric receipt metadata drift")
    mask = row["attack_block_mask"]
    if mask is not None and (type(mask) is not int or not 0 <= mask <= 15):
        raise ValueError("original block mask metadata drift")
    if row["finish_reason"] == "length" and row["generated_token_count"] != 128:
        raise ValueError("capped original receipt is not exactly 128 tokens")
    row["original_receipt_sha256"] = _hash(digest)
    return row


def build_continuation_plan(
    acquisition_root: Path,
    *,
    pilot_size: int = 64,
    expected_summary_sha256: str = A142_SUMMARY_SHA256,
    expected_manifest_sha256: str = A142_MANIFEST_SHA256,
    expected_observations: int = 8880,
    expected_capped: int = 6037,
) -> dict[str, Any]:
    """Build a private plan from hash-verified numeric receipts, without scores.

    The explicit expected arguments support synthetic tests; production calls
    must retain the pinned A142 defaults.  No topology or restricted artifacts
    are opened here.
    """
    summary_path = acquisition_root / "summary.json"
    if sha256_file(summary_path) != _hash(expected_summary_sha256):
        raise ValueError("original acquisition summary hash mismatch")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "acquisition_complete"
        or summary.get("source_commit") != A142_SOURCE_COMMIT
        or summary.get("plan_sha256") != A142_PLAN_SHA256
        or summary.get("private_topology_sha256") != A142_TOPOLOGY_SHA256
        or summary.get("observation_count") != expected_observations
        or summary.get("generation_count") != expected_observations
        or summary.get("receipt_manifest_sha256") != expected_manifest_sha256
        or summary.get("generation_checkpoints") != [0, 1, 4, 8]
        or summary.get("enforcement_enabled") is not False
        or summary.get("unopened_v2_confirmation_opened") is not False
    ):
        raise ValueError("original acquisition summary provenance drift")
    paths = sorted((acquisition_root / "receipts").glob("*.json"))
    if len(paths) != expected_observations:
        raise ValueError("original acquisition receipt count mismatch")
    manifest = [{"trial_id": path.stem, "sha256": sha256_file(path)} for path in paths]
    if sha256_bytes(canonical_json_bytes(manifest)) != _hash(expected_manifest_sha256):
        raise ValueError("original acquisition receipt manifest mismatch")
    rows = []
    for path, item in zip(paths, manifest, strict=True):
        receipt = json.loads(path.read_text())
        if (
            receipt.get("trial_id") != path.stem
            or receipt.get("source_commit") != summary["source_commit"]
            or receipt.get("plan_sha256") != summary["plan_sha256"]
            or receipt.get("private_topology_sha256") != summary["private_topology_sha256"]
            or receipt.get("run_id") != summary["run_id"]
            or receipt.get("status") != "complete"
        ):
            raise ValueError("original acquisition row provenance drift")
        rows.append(_project_receipt(receipt, item["sha256"]))
    capped = [row for row in rows if row["finish_reason"] == "length"]
    if len(capped) != expected_capped:
        raise ValueError("original capped cohort count mismatch")
    return {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "outcome_blind_continuation_plan_frozen",
        "interpretation": "conditional_continuation_of_fixed_saved_prefix_not_fresh_uninterrupted_generation",
        "bindings": {
            "original_summary_sha256": expected_summary_sha256,
            "original_receipt_manifest_sha256": expected_manifest_sha256,
            "original_source_commit": summary["source_commit"],
            "original_plan_sha256": summary["plan_sha256"],
            "original_topology_sha256": summary["private_topology_sha256"],
        },
        "selection_rule": "all_prior_length_capped_128_token_rows_regardless_of_outcome",
        "pilot_rule": "fixed_sha256_family_placement_coverage_then_frame_placement_blockcount_balancing",
        "generation_checkpoints": list(CHECKPOINTS),
        "original_observation_count": len(rows),
        "continuation_row_count": len(capped),
        "eos_carryforward_count": len(rows) - len(capped),
        "original_receipts": manifest,
        "rows": capped,
        "pilot_trial_ids": select_pilot(capped, pilot_size),
        "raw_prompt_token_generation_or_row_level_content_public": False,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
    }


def public_plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return only protocol constants, provenance hashes and aggregate counts."""
    selected = set(plan["pilot_trial_ids"])
    pilot = [row for row in plan["rows"] if row["trial_id"] in selected]
    return {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "private_plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        "bindings": dict(plan["bindings"]),
        "generation_checkpoints": list(CHECKPOINTS),
        "original_observation_count": plan["original_observation_count"],
        "continuation_row_count": plan["continuation_row_count"],
        "eos_carryforward_count": plan["eos_carryforward_count"],
        "pilot_count": len(pilot),
        "pilot_intent_counts": dict(sorted(Counter(row["intent_frame"] for row in pilot).items())),
        "pilot_placement_counts": dict(
            sorted(Counter(row["placement"] or "no_scaffold" for row in pilot).items())
        ),
        "pilot_family_counts": dict(
            sorted(Counter(row["variant_family"] for row in pilot).items())
        ),
        "outcome_blind_selection": True,
        "raw_prompt_token_generation_or_row_level_content_public": False,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
    }


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create a private JSON record; never overwrite differing data."""
    raw = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.is_symlink():
        raise ValueError("immutable record cannot be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError("immutable record already exists with different content") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(raw)


@contextlib.contextmanager
def trial_lock(directory: Path) -> Iterator[None]:
    """OS-released advisory lock; a crashed PID cannot leave a stale live lock."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    lock_path = directory / ".continuation.lock"
    if lock_path.is_symlink():
        raise ValueError("continuation lock cannot be a symbolic link")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("continuation trial is already running") from None
        yield
    finally:
        os.close(descriptor)


_READOUT_SCALARS = frozenset(
    (
        "feature_6779_magnitude",
        "frozen_subspace_score",
        "sae_normalized_reconstruction_error",
        "prefill_latency_ms",
        "detector_readout_latency_ms",
        "peak_gpu_memory_bytes",
    )
)


def _safe_readout(value: Mapping[str, Any] | None, prefix: Sequence[int]) -> dict[str, Any] | None:
    if value is None:
        return None
    allowed = _READOUT_SCALARS | {
        "prefix_token_count",
        "prefix_token_ids_sha256",
        "jlens_refusal_minus_compliance_trajectory",
        "residual_artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError("readout contains unexpected control-plane fields")
    if value.get("prefix_token_count") != len(prefix) or value.get(
        "prefix_token_ids_sha256"
    ) != token_hash(prefix):
        raise ValueError("readout prefix binding mismatch")
    for key in _READOUT_SCALARS & set(value):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError("readout scalar is not finite numeric data")
    trajectory = value.get("jlens_refusal_minus_compliance_trajectory")
    if trajectory is not None and (
        not isinstance(trajectory, (list, tuple))
        or len(trajectory) != 31
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in trajectory)
    ):
        raise ValueError("readout trajectory is not a finite 31-layer vector")
    if "residual_artifact_sha256" in value:
        _hash(value["residual_artifact_sha256"])
    return dict(value)


def _receipt_from_artifact(
    artifact: Mapping[str, Any],
    digest: str,
    readout: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "continuation_checkpoint_complete",
        "trial_id": artifact["trial_id"],
        "bindings": dict(artifact["bindings"]),
        "requested_horizon": artifact["requested_horizon"],
        "generated_token_count": len(artifact["generated_token_ids"]),
        "generated_token_ids_sha256": token_hash(artifact["generated_token_ids"]),
        "restricted_artifact_sha256": digest,
        "finish_reason": "eos" if artifact["eos"] else "length",
        "right_censored": not artifact["eos"],
        "readout": readout,
        "classifier_status": "not_scored",
    }


def resume_trial(
    *,
    row: Mapping[str, Any],
    original_token_ids: Sequence[int],
    output_root: Path,
    generator: Callable[[list[int], int], Mapping[str, Any]],
    deadline_monotonic: float,
    plan_sha256: str,
    runtime_sha256: str,
    readout: Callable[[list[int]], Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    checkpoints: Sequence[int] = CHECKPOINTS,
) -> dict[str, Any]:
    """Resume one capped row from exact IDs with immutable per-horizon records.

    ``generator(prefix, max_new_tokens)`` must return ONLY new token IDs and an
    EOS boolean.  It can cache model state privately.  The deadline prevents
    starting additional chunks; an independently enforced compute deadline is
    still required because this pure function cannot preempt model calls.
    """
    if tuple(checkpoints) != CHECKPOINTS:
        raise ValueError("frozen continuation horizons changed")
    if not math.isfinite(deadline_monotonic):
        raise ValueError("finite continuation deadline required")
    trial_id = _identifier(row["trial_id"])
    original = _tokens(original_token_ids)
    if (
        row.get("finish_reason") != "length"
        or row.get("generated_token_count") != 128
        or len(original) != 128
        or token_hash(original) != row.get("generated_token_ids_sha256")
    ):
        raise ValueError("original exact capped token prefix mismatch")
    bindings = {
        "original_receipt_sha256": _hash(row["original_receipt_sha256"]),
        "original_token_ids_sha256": token_hash(original),
        "original_restricted_artifact_sha256": _hash(row["restricted_artifact_sha256"]),
        "plan_sha256": _hash(plan_sha256),
        "runtime_sha256": _hash(runtime_sha256),
        "readout_required": readout is not None,
    }
    directory = output_root / trial_id
    with trial_lock(directory):
        prefix = original
        last_receipt: dict[str, Any] | None = None
        for horizon in checkpoints:
            token_path = directory / "restricted" / f"tokens-{horizon:04d}.json"
            receipt_path = directory / "receipts" / f"checkpoint-{horizon:04d}.json"
            if token_path.is_symlink() or receipt_path.is_symlink():
                raise ValueError("checkpoint cannot be a symbolic link")
            if receipt_path.exists() and not token_path.exists():
                raise ValueError("checkpoint restricted artifact missing")
            if token_path.exists():
                artifact = json.loads(token_path.read_text())
                expected_keys = {
                    "schema_version",
                    "study_id",
                    "trial_id",
                    "bindings",
                    "requested_horizon",
                    "generated_token_ids",
                    "eos",
                }
                if (
                    set(artifact) != expected_keys
                    or artifact["schema_version"] != "1.0"
                    or artifact["study_id"] != STUDY_ID
                    or artifact["trial_id"] != trial_id
                    or artifact["bindings"] != bindings
                    or artifact["requested_horizon"] != horizon
                    or type(artifact["eos"]) is not bool
                ):
                    raise ValueError("continuation artifact provenance drift")
                saved = _tokens(artifact["generated_token_ids"])
                if saved[: len(prefix)] != prefix or not len(prefix) <= len(saved) <= horizon:
                    raise ValueError("continuation saved prefix drift")
                if (not artifact["eos"] and len(saved) != horizon) or (
                    horizon == 128 and artifact["eos"]
                ):
                    raise ValueError("continuation saved length drift")
                if receipt_path.exists():
                    existing = json.loads(receipt_path.read_text())
                    numeric = _safe_readout(existing.get("readout"), saved)
                    if (numeric is not None) != (readout is not None):
                        raise ValueError("continuation checkpoint instrumentation drift")
                else:
                    if readout and clock() >= deadline_monotonic:
                        return {
                            "status": "paused_deadline",
                            "trial_id": trial_id,
                            "last_checkpoint": last_receipt,
                            "new_chunk_started": False,
                        }
                    numeric = _safe_readout(readout(list(saved)) if readout else None, saved)
                    if readout is not None and numeric is None:
                        raise ValueError("continuation checkpoint readout missing")
                expected_receipt = _receipt_from_artifact(
                    artifact, sha256_file(token_path), numeric
                )
                # A crash between writing tokens and writing the receipt is
                # recovered without calling the model again.
                write_immutable_json(receipt_path, expected_receipt)
                prefix = saved
                last_receipt = expected_receipt
            else:
                if clock() >= deadline_monotonic:
                    return {
                        "status": "paused_deadline",
                        "trial_id": trial_id,
                        "last_checkpoint": last_receipt,
                        "new_chunk_started": False,
                    }
                eos = False
                if horizon > 128:
                    generated = generator(list(prefix), horizon - len(prefix))
                    if not isinstance(generated, Mapping) or set(generated) != {
                        "delta_token_ids",
                        "eos",
                    }:
                        raise ValueError("generator returned unexpected fields")
                    delta = _tokens(generated["delta_token_ids"])
                    if type(generated["eos"]) is not bool:
                        raise ValueError("generator EOS must be boolean")
                    eos = generated["eos"]
                    if len(delta) > horizon - len(prefix) or (
                        not eos and len(delta) != horizon - len(prefix)
                    ):
                        raise ValueError("generator token count disagrees with completion state")
                    prefix = [*prefix, *delta]
                artifact = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "trial_id": trial_id,
                    "bindings": bindings,
                    "requested_horizon": horizon,
                    "generated_token_ids": prefix,
                    "eos": eos,
                }
                digest = write_immutable_json(token_path, artifact)
                # Commit exact tokens before potentially expensive diagnostic
                # readouts.  Readout failure can require replaying diagnostics,
                # but must not require regenerating the completed chunk.
                if readout and clock() >= deadline_monotonic:
                    return {
                        "status": "paused_deadline",
                        "trial_id": trial_id,
                        "last_checkpoint": last_receipt,
                        "new_chunk_started": False,
                    }
                numeric = _safe_readout(readout(list(prefix)) if readout else None, prefix)
                if readout is not None and numeric is None:
                    raise ValueError("continuation checkpoint readout missing")
                last_receipt = _receipt_from_artifact(artifact, digest, numeric)
                write_immutable_json(receipt_path, last_receipt)
            if last_receipt["finish_reason"] == "eos":
                return {
                    "status": "complete_eos",
                    "trial_id": trial_id,
                    "last_checkpoint": last_receipt,
                }
        return {
            "status": "complete_censored_at_ceiling",
            "trial_id": trial_id,
            "last_checkpoint": last_receipt,
        }
