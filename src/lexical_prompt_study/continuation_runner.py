"""Authorized, receipt-bound execution of the A146 truncation correction.

The CLI prints aggregate counts and timings only. Prompts, token IDs, decoded
generations, row scores and runtime paths remain in private artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .continuation import (
    A142_TOPOLOGY_SHA256,
    CHECKPOINTS,
    STUDY_ID,
    build_continuation_plan,
    public_plan_summary,
    resume_trial,
    token_hash,
    trial_lock,
    write_immutable_json,
    _receipt_from_artifact,
    _safe_readout,
)
from .continuation_ops import budget_limits
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("continuation JSON object required")
    return value


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


_SOURCE_CONFIG_PATHS = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "plans/jlens_breaker_v2.public.json",
        "plans/study_v1.public.json",
        "plans/continuation_repair_a146.md",
        "plans/continuation_batch_a148.md",
    }
)


def verify_source_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a hash-pinned code-only checkout without manufacturing Git history."""
    root = Path.cwd() if source_root is None else source_root
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("continuation source manifest/root cannot be a symbolic link")
    root = root.resolve()
    if (root / "src/lexical_prompt_study/continuation_runner.py").resolve() != Path(
        __file__
    ).resolve():
        raise ValueError("continuation source manifest does not cover executing package")
    digest = sha256_file(manifest_path)
    if expected_sha256 is not None and digest != _sha(expected_sha256):
        raise ValueError("continuation source manifest authorization hash drift")
    manifest = _load(manifest_path)
    commit = manifest.get("source_commit")
    files = manifest.get("files")
    if (
        set(manifest) != {"source_commit", "files"}
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError("continuation source manifest schema drift")
    declared_python = set()
    for relative, expected in files.items():
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("continuation source manifest path invalid")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or str(parsed) != relative
            or ".." in parsed.parts
            or not parsed.parts
        ):
            raise ValueError("continuation source manifest path traversal")
        is_python = relative.startswith("src/lexical_prompt_study/") and relative.endswith(".py")
        if not is_python and relative not in _SOURCE_CONFIG_PATHS:
            raise ValueError("continuation source manifest path outside code allowlist")
        path = root
        for part in parsed.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("continuation source package contains symbolic link")
        if not path.is_file() or sha256_file(path) != _sha(expected):
            raise ValueError("continuation source file missing or hash drift")
        if is_python:
            declared_python.add(relative)
    for path in (root / "src").rglob("*"):
        if path.is_symlink():
            raise ValueError("continuation source package contains symbolic link")
    actual_python = {
        path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py") if path.is_file()
    }
    if declared_python != actual_python or not declared_python:
        raise ValueError("continuation source package Python file set drift")
    if not _SOURCE_CONFIG_PATHS.issubset(files):
        raise ValueError("continuation source manifest required configuration missing")
    return {
        "source_commit": commit,
        "source_manifest_sha256": digest,
        "source_file_count": len(files),
    }


def _sha(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("continuation SHA-256 binding invalid")
    return value


def _deadline(value: str, *, now: float | None = None) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("continuation deadline must include timezone")
    remaining = parsed.timestamp() - (time.time() if now is None else now)
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("continuation execution deadline expired")
    return time.monotonic() + remaining


def validate_authorization(
    value: Mapping[str, Any],
    *,
    phase: str,
    scope: str,
    run_id: str,
    source_commit: str,
    bindings: Mapping[str, str],
) -> float:
    """Validate a fresh bounded authorization; return monotonic deadline."""
    if (
        value.get("schema_version") != "1.0"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "continuation_execution_authorized"
        or value.get("paid_compute_authorized") is not True
        or phase not in value.get("authorized_phases", [])
        or scope not in ("pilot", "full")
        or value.get("scope") != scope
        or value.get("run_id") != run_id
        or value.get("source_commit") != source_commit
        or value.get("bindings") != dict(bindings)
        or value.get("single_task_owned_pod_maximum") != 1
        or value.get("enforcement_enabled") is not False
        or value.get("unopened_v2_confirmation_opened") is not False
        or value.get("raw_content_public") is not False
        or value.get("budget_authority_id") != "a148-user-70"
        or value.get("execution_protocol") != "a148-batch4"
        or value.get("batch_size") != 4
    ):
        raise ValueError("continuation execution authorization drift")
    limits = budget_limits(value["budget_authority_id"])
    maximum = value.get("maximum_new_compute_usd")
    pilot_maximum = value.get("maximum_pilot_compute_usd")
    hourly = value.get("hourly_rate_usd")
    spent = value.get("round_spent_upper_usd")
    pilot_spent = value.get("pilot_spent_upper_usd")
    if any(
        type(item) not in (int, float) or not math.isfinite(item)
        for item in (maximum, pilot_maximum, hourly, spent, pilot_spent)
    ) or not (
        0 <= pilot_spent <= spent < limits[0]
        and 0 < maximum <= limits[0] - spent
        and 0 <= pilot_maximum <= limits[1] - pilot_spent
        and (scope != "pilot" or pilot_maximum > 0)
        and hourly > 0
    ):
        raise ValueError("continuation compute authorization exceeds approved ceiling")
    deadline = _deadline(str(value.get("deadline_utc", "")))
    budget = min(maximum, pilot_maximum) if scope == "pilot" else maximum
    seconds = deadline - time.monotonic()
    if seconds * hourly / 3600 > budget + 0.01:
        raise ValueError("continuation deadline exceeds authorized compute budget")
    if scope == "full":
        gate = value.get("pilot_gate", {})
        if (
            gate.get("accepted") is not True
            or not 0
            <= float(gate.get("projected_remaining_compute_usd", math.inf))
            <= float(gate.get("remaining_authorized_compute_usd", -1))
            <= maximum
        ):
            raise ValueError("continuation full phase requires accepted budgeted pilot")
        _sha(gate.get("pilot_acquisition_summary_sha256"))
        _sha(gate.get("pilot_scoring_summary_sha256"))
        _sha(gate.get("cost_projection_sha256"))
        if (
            gate.get("pilot_completed_count") != 64
            or gate.get("pilot_scored_count") != 64
            or gate.get("batch_size") != 4
        ):
            raise ValueError("continuation full phase requires complete scored batch pilot")
        if seconds * hourly / 3600 > float(gate["remaining_authorized_compute_usd"]) + 0.01:
            raise ValueError("continuation deadline exceeds remaining round budget")
    return deadline


def load_context(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    topology_path: Path,
    acquisition_root: Path,
    original_scoring_root: Path,
    instrument_plan_path: Path,
    probe_plan_path: Path,
    factorial_material_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Read restricted inputs privately only after immutable hash checks."""
    for original in (acquisition_root, original_scoring_root):
        if (
            output_root.resolve() == original.resolve()
            or original.resolve() in output_root.resolve().parents
        ):
            raise ValueError("continuation output must not modify an original bundle")
    public = _load(public_plan_path)
    plan = _load(private_plan_path)
    rebuilt = build_continuation_plan(
        acquisition_root, pilot_size=len(plan.get("pilot_trial_ids", []))
    )
    if plan != rebuilt or sha256_file(topology_path) != A142_TOPOLOGY_SHA256:
        raise ValueError("continuation frozen inputs do not reproduce")
    if public.get("study_id") != STUDY_ID or public.get("private_plan_sha256") != sha256_file(
        private_plan_path
    ):
        raise ValueError("continuation public/private plan binding drift")
    original_scoring = _load(original_scoring_root / "summary.json")
    expected_score = public.get("original_scoring_summary_sha256")
    if sha256_file(original_scoring_root / "summary.json") != _sha(expected_score):
        raise ValueError("continuation original scorer summary drift")
    score_manifest = [
        {"trial_id": path.stem, "sha256": sha256_file(path)}
        for path in sorted((original_scoring_root / "trials").glob("*.json"))
    ]
    if len(score_manifest) != plan["original_observation_count"] or sha256_bytes(
        canonical_json_bytes(score_manifest)
    ) != original_scoring.get("score_manifest_sha256"):
        raise ValueError("continuation original score manifest drift")
    topology = _load(topology_path)
    observations = {row["trial_id"]: row for row in topology["observations"]}
    if len(observations) != plan["original_observation_count"]:
        raise ValueError("continuation original topology count drift")
    for row in plan["rows"]:
        observation = observations[row["trial_id"]]
        if (
            any(
                observation.get(key) != row.get(key)
                for key in (
                    "trial_id",
                    "request_core_id",
                    "request_core_sha256",
                    "intent_frame",
                    "safe_intent",
                    "variant_family",
                    "attack_block_mask",
                    "placement",
                    "prompt_sha256",
                    "prompt_token_ids_sha256",
                )
            )
            or token_hash(observation["prompt_token_ids"]) != row["prompt_token_ids_sha256"]
        ):
            raise ValueError("continuation original prompt binding drift")
    bindings = {
        "public_plan_sha256": sha256_file(public_plan_path),
        "private_plan_sha256": sha256_file(private_plan_path),
        "original_topology_sha256": sha256_file(topology_path),
        "original_acquisition_summary_sha256": sha256_file(acquisition_root / "summary.json"),
        "original_scoring_summary_sha256": sha256_file(original_scoring_root / "summary.json"),
        "instrument_plan_sha256": sha256_file(instrument_plan_path),
        "probe_plan_sha256": sha256_file(probe_plan_path),
        "factorial_material_sha256": sha256_file(factorial_material_path),
    }
    return {
        "public": public,
        "plan": plan,
        "observations": observations,
        "bindings": bindings,
        "original_scoring": original_scoring,
        "original_score_hashes": {item["trial_id"]: item["sha256"] for item in score_manifest},
    }


def load_original_tokens(row: Mapping[str, Any], acquisition_root: Path) -> list[int]:
    receipt_path = acquisition_root / "receipts" / f"{row['trial_id']}.json"
    restricted_path = acquisition_root / "restricted" / f"{row['trial_id']}.json"
    if (
        sha256_file(receipt_path) != row["original_receipt_sha256"]
        or sha256_file(restricted_path) != row["restricted_artifact_sha256"]
    ):
        raise ValueError("continuation original receipt/artifact drift")
    value = _load(restricted_path)
    if (
        set(value) != {"trial_id", "generated_text", "generated_token_ids"}
        or value["trial_id"] != row["trial_id"]
        or token_hash(value["generated_token_ids"]) != row["generated_token_ids_sha256"]
        or sha256_text(value["generated_text"]) != row["generated_text_sha256"]
        or len(value["generated_token_ids"]) != 128
    ):
        raise ValueError("continuation original exact prefix drift")
    return value["generated_token_ids"]


def _runtime_identity(context: Mapping[str, Any], source_commit: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "source_commit": source_commit,
                "bindings": context["bindings"],
                "model_revision": context["public"]["model_revision"],
                "decoding": "conditional_exact_prefix_plain_greedy",
                **(
                    {
                        "execution_protocol": "a148-batch4",
                        "batch_layout_sha256": sha256_bytes(
                            canonical_json_bytes(batch_layout(context["plan"]))
                        ),
                    }
                    if context["public"].get("execution_protocol") == "a148-batch4"
                    else {}
                ),
            }
        )
    )


def selected_rows(plan: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    selected = set(plan["pilot_trial_ids"]) if scope == "pilot" else None
    return [row for row in plan["rows"] if selected is None or row["trial_id"] in selected]


def batch_layout(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Stable membership shared by pilot/full and never compacted after EOS."""
    by_id = {row["trial_id"]: row for row in plan["rows"]}
    pilot = list(plan["pilot_trial_ids"])
    if (
        len(by_id) != len(plan["rows"])
        or len(set(pilot)) != len(pilot)
        or not set(pilot) <= set(by_id)
    ):
        raise ValueError("batch layout membership drift")
    if len(pilot) != 64:
        raise ValueError("A148 requires the unchanged 64-row pilot")
    cohorts = []
    for subset, ids in (("pilot", pilot), ("remaining", sorted(set(by_id) - set(pilot)))):
        for offset in range(0, len(ids), 4):
            members = ids[offset : offset + 4]
            digest = sha256_bytes(canonical_json_bytes(["a148-batch4", subset, members]))
            cohorts.append(
                {
                    "cohort_id": "batch-" + digest[:24],
                    "subset": subset,
                    "member_ids": members,
                    "virtual_slot_count": 4 - len(members),
                }
            )
    return {
        "schema_version": "1.0",
        "execution_protocol": "a148-batch4",
        "batch_size": 4,
        "padding_policy": "fixed_slots_forced_eos_no_compaction",
        "cohorts": cohorts,
    }


def validate_full_projection(
    projection: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_commit: str,
) -> None:
    """The full gate consumes the measured artifact, not an unverified attestation."""
    gate = authorization["pilot_gate"]
    bindings = projection.get("bindings", {})
    expected_layout = sha256_bytes(canonical_json_bytes(batch_layout(context["plan"])))
    if (
        projection.get("schema_version") != "2.0"
        or projection.get("study_id") != STUDY_ID
        or projection.get("execution_protocol") != "a148-batch4"
        or projection.get("status") != "complete_pilot_projection"
        or projection.get("fits_budget_planning_gate") is not True
        or projection.get("complete_pilot_gate_passed") is not True
        or projection.get("runtime_sha256") != _runtime_identity(context, source_commit)
        or projection.get("batch_layout_sha256") != expected_layout
        or projection.get("batch_size") != 4
        or any(
            projection.get(key) != 64
            for key in ("pilot_selected_count", "pilot_completed_count", "pilot_scored_count")
        )
        or projection.get("round_hard_cap_usd") != 70
        or bindings.get("private_plan_sha256") != context["bindings"]["private_plan_sha256"]
        or bindings.get("pilot_acquisition_summary_sha256")
        != gate["pilot_acquisition_summary_sha256"]
        or bindings.get("pilot_scoring_summary_sha256") != gate["pilot_scoring_summary_sha256"]
    ):
        raise ValueError("A148 full projection evidence or completion binding drift")
    costs = [
        projection.get("projected_remaining_acquisition_usd"),
        projection.get("projected_remaining_scoring_usd"),
        *[
            projection.get("reserves", {}).get(key)
            for key in ("loading_and_parity_usd", "scoring_additional_usd", "storage_usd")
        ],
    ]
    other = [
        projection.get("full_hourly_usd"),
        projection.get("projected_round_total_upper_planning_usd"),
    ]
    if any(
        type(item) not in (int, float) or not math.isfinite(item) or item < 0
        for item in [*costs, *other]
    ):
        raise ValueError("A148 full projection cost values invalid")
    if (
        other[0] < authorization["hourly_rate_usd"]
        or other[1] > 70
        or sum(costs) > gate["remaining_authorized_compute_usd"]
        or sum(costs) > gate["projected_remaining_compute_usd"] + 1e-9
    ):
        raise ValueError("A148 full projection understates price or exceeds accepted budget")


def _bundle_manifest(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    manifest = []
    for row in rows:
        root = output_root / "trials" / row["trial_id"]
        for pattern in (
            "receipts/*.json",
            "restricted/tokens-*.json",
            "restricted/residuals/*.pt",
            "readouts/*.json",
        ):
            for path in sorted(root.glob(pattern)):
                if path.is_symlink():
                    raise ValueError("continuation bundle contains symbolic link")
                manifest.append(
                    {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}
                )
    namespace = output_root / "a148-namespace.private.json"
    if namespace.exists():
        row_ids = {row["trial_id"] for row in rows}
        layout = _load(namespace)["layout"]
        paths = []
        for cohort in layout["cohorts"]:
            if set(cohort["member_ids"]) <= row_ids:
                paths.extend(
                    (output_root / "cohorts" / cohort["cohort_id"] / "restricted").glob("*.json")
                )
                paths.extend(
                    (output_root / "cohorts" / cohort["cohort_id"] / "receipts").glob("*.json")
                )
                paths.extend((output_root / "batch-timings" / cohort["cohort_id"]).glob("*.json"))
        for trial_id in row_ids:
            paths.extend((output_root / "batch-residuals" / trial_id).glob("*.pt"))
        for path in sorted(paths):
            if path.is_symlink():
                raise ValueError("continuation batch bundle contains symbolic link")
            manifest.append(
                {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}
            )
    manifest.sort(key=lambda item: item["path"])
    return sha256_bytes(canonical_json_bytes(manifest)), len(manifest)


def _verify_bundle(
    summary: Mapping[str, Any], output_root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    digest, count = _bundle_manifest(output_root, rows)
    if (
        summary.get("trial_bundle_manifest_sha256") != digest
        or summary.get("trial_bundle_file_count") != count
    ):
        raise ValueError("continuation acquired bundle manifest drift")


def acquire(
    *,
    context: Mapping[str, Any],
    acquisition_root: Path,
    output_root: Path,
    scope: str,
    source_commit: str,
    deadline_monotonic: float,
    runtime_factory: Callable[[], Any],
    generator_factory: Callable[..., Any] | None = None,
    parity: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    from .continuation_gpu import PrefixGenerator, verify_continuation_parity

    factory = PrefixGenerator if generator_factory is None else generator_factory
    parity_check = verify_continuation_parity if parity is None else parity
    rows = selected_rows(context["plan"], scope)
    runtime_hash = _runtime_identity(context, source_commit)
    summary_path = output_root / "summaries" / f"{scope}-acquisition.json"
    if summary_path.exists():
        existing = _load(summary_path)
        if existing.get("runtime_sha256") != runtime_hash or existing.get(
            "observation_count"
        ) != len(rows):
            raise ValueError("continuation existing acquisition summary drift")
        _verify_bundle(existing, output_root, rows)
        return existing
    started = time.monotonic()
    complete = 0
    eos = 0
    censored = 0
    timed_out = False
    runtime = None
    with trial_lock(output_root):
        # Complete the fixed technical parity gate before any new outcome row.
        row_by_id = {row["trial_id"]: row for row in context["plan"]["rows"]}
        for trial_id in context["plan"]["pilot_trial_ids"][:4]:
            parity_path = output_root / "parity" / f"{trial_id}.json"
            if not parity_path.exists():
                if time.monotonic() >= deadline_monotonic:
                    timed_out = True
                    break
                row = row_by_id[trial_id]
                tokens = load_original_tokens(row, acquisition_root)
                if runtime is None:
                    runtime = runtime_factory()
                try:
                    checked = dict(
                        parity_check(
                            runtime,
                            context["observations"][trial_id]["prompt_token_ids"],
                            tokens,
                            extension_tokens=8,
                            deadline_monotonic=deadline_monotonic,
                        )
                    )
                except TimeoutError:
                    timed_out = True
                    break
                write_immutable_json(parity_path, {"runtime_sha256": runtime_hash, **checked})
            recorded_parity = _load(parity_path)
            if (
                recorded_parity.get("runtime_sha256") != runtime_hash
                or recorded_parity.get("status") != "pass"
                or recorded_parity.get("parity_scope") != "exact_saved_prefix_conditional"
                or recorded_parity.get("original_prefix_immutable") is not True
                or recorded_parity.get("trusted_input_prefix_unchanged") is not True
                or recorded_parity.get("conditional_continuation_matches") is not True
            ):
                raise ValueError("continuation parity gate failed; generation stopped")
        for row in [] if timed_out else rows:
            if time.monotonic() >= deadline_monotonic:
                timed_out = True
                break
            tokens = load_original_tokens(row, acquisition_root)
            observation = context["observations"][row["trial_id"]]
            if runtime is None:
                runtime = runtime_factory()
            readout_root = output_root / "trials" / row["trial_id"] / "readouts"
            preserved_readouts = {
                str(int(path.stem)): _load(path) for path in sorted(readout_root.glob("*.json"))
            }
            generator = factory(
                runtime,
                observation["prompt_token_ids"],
                residual_root=output_root / "trials" / row["trial_id"] / "restricted" / "residuals",
                deadline_monotonic=deadline_monotonic,
                preserved_readouts=preserved_readouts,
                readout_sink=lambda value, root=readout_root: write_immutable_json(
                    root / f"{int(value['prefix_token_count']):04d}.json", value
                ),
            )
            try:
                result = resume_trial(
                    row=row,
                    original_token_ids=tokens,
                    output_root=output_root / "trials",
                    generator=generator,
                    readout=generator.readout,
                    deadline_monotonic=deadline_monotonic,
                    plan_sha256=context["bindings"]["private_plan_sha256"],
                    runtime_sha256=runtime_hash,
                )
            except TimeoutError:
                timed_out = True
                break
            if result["status"] == "paused_deadline":
                timed_out = True
                break
            complete += 1
            eos += result["status"] == "complete_eos"
            censored += result["status"] == "complete_censored_at_ceiling"
            write_immutable_json(
                output_root / "timings" / row["trial_id"] / f"{time.time_ns()}.json",
                {
                    "runtime_sha256": runtime_hash,
                    "intent_frame": row.get("intent_frame"),
                    "placement": row.get("placement"),
                    "variant_family": row.get("variant_family"),
                    "attack_block_count": row.get("attack_block_count"),
                    "finish_reason": result["last_checkpoint"]["finish_reason"],
                    "generated_token_count": result["last_checkpoint"]["generated_token_count"],
                    "timing": dict(getattr(generator, "timing_summary", {})),
                },
            )
            readouts = generator.readouts_by_generated_tokens
            # Additional scheduled positions are private numeric receipts; keep
            # them apart from immutable horizon records used for scoring.
            for count, value in readouts.items():
                write_immutable_json(
                    output_root
                    / "trials"
                    / row["trial_id"]
                    / "readouts"
                    / f"{int(count):04d}.json",
                    value,
                )
            print(
                json.dumps(
                    {
                        "phase": "acquire",
                        "scope": scope,
                        "completed": complete,
                        "total": len(rows),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "observed_seconds_per_completed_row": round(
                            (time.monotonic() - started) / complete, 3
                        ),
                        "naive_projected_full_acquisition_seconds_excluding_scoring": round(
                            (time.monotonic() - started) / complete * len(context["plan"]["rows"]),
                            1,
                        ),
                    }
                ),
                flush=True,
            )
        result = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "status": "paused_deadline" if timed_out else "acquisition_complete",
            "scope": scope,
            "runtime_sha256": runtime_hash,
            "observation_count": len(rows),
            "completed_count": complete,
            "eos_count": eos,
            "censored_at_1024_count": censored,
            "elapsed_seconds": time.monotonic() - started,
            "source_commit": source_commit,
            "bindings": dict(context["bindings"]),
            "raw_content_public": False,
            "enforcement_enabled": False,
            "unopened_v2_confirmation_opened": False,
        }
        result["trial_bundle_manifest_sha256"], result["trial_bundle_file_count"] = (
            _bundle_manifest(output_root, rows)
        )
        destination = (
            summary_path
            if not timed_out
            else output_root / "progress" / f"acquire-{time.time_ns()}.json"
        )
        write_immutable_json(destination, result)
    return result


def _materialize_batch_horizon(
    *,
    snapshot: Mapping[str, Any],
    readouts: Mapping[str, Any],
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    runtime_hash: str,
    plan_hash: str,
    layout_hash: str,
) -> None:
    """Expose collective GPU checkpoints through the established scorer schema."""
    by_id = {row["trial_id"]: row for row in rows}
    if snapshot.get("runtime_sha256") != runtime_hash or snapshot.get("plan_sha256") != plan_hash:
        raise ValueError("batch snapshot runtime binding drift")
    if snapshot.get("member_ids") != [row["trial_id"] for row in rows] or [
        item.get("trial_id") for item in snapshot.get("members", [])
    ] != snapshot.get("member_ids"):
        raise ValueError("batch snapshot member order drift")
    horizon = snapshot["requested_horizon"]
    if horizon not in CHECKPOINTS:
        raise ValueError("batch checkpoint horizon drift")
    for member in snapshot["members"]:
        trial_id = member["trial_id"]
        row = by_id[trial_id]
        tokens = member["generated_token_ids"]
        if (
            token_hash(tokens[:128]) != row["generated_token_ids_sha256"]
            or type(member["eos"]) is not bool
            or not 128 <= len(tokens) <= horizon
            or (not member["eos"] and len(tokens) != horizon)
        ):
            raise ValueError("batch member exact prefix or length drift")
        trial_root = output_root / "trials" / trial_id
        previous_eos = False
        for prior in CHECKPOINTS:
            if prior >= horizon:
                break
            path = trial_root / "restricted" / f"tokens-{prior:04d}.json"
            if path.exists():
                old = _load(path)
                if tokens[: len(old["generated_token_ids"])] != old["generated_token_ids"]:
                    raise ValueError("batch member historical checkpoint drift")
                if old["eos"]:
                    if not member["eos"] or tokens != old["generated_token_ids"]:
                        raise ValueError("batch member continued after EOS")
                    previous_eos = True
                    break
        for count, value in readouts.get(trial_id, {}).items():
            count_int = int(count)
            if not 0 < count_int <= len(tokens):
                raise ValueError("batch readout position exceeds available tokens")
            _safe_readout(value, tokens[:count_int])
            write_immutable_json(trial_root / "readouts" / f"{count_int:04d}.json", value)
        if previous_eos:
            continue
        numeric = _safe_readout(readouts.get(trial_id, {}).get(str(len(tokens))), tokens)
        if numeric is None:
            raise ValueError("batch checkpoint lacks actual-position readout")
        artifact = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "trial_id": trial_id,
            "bindings": {
                "runtime_sha256": runtime_hash,
                "plan_sha256": plan_hash,
                "original_receipt_sha256": row["original_receipt_sha256"],
                "original_token_ids_sha256": row["generated_token_ids_sha256"],
                "original_restricted_artifact_sha256": row["restricted_artifact_sha256"],
                "batch_layout_sha256": layout_hash,
                "cohort_id": snapshot["cohort_id"],
                "cohort_layout_sha256": snapshot["layout_sha256"],
                "readout_required": True,
            },
            "requested_horizon": horizon,
            "generated_token_ids": tokens,
            "eos": member["eos"],
        }
        digest = write_immutable_json(
            trial_root / "restricted" / f"tokens-{horizon:04d}.json", artifact
        )
        write_immutable_json(
            trial_root / "receipts" / f"checkpoint-{horizon:04d}.json",
            _receipt_from_artifact(artifact, digest, numeric),
        )


def acquire_batch(
    *,
    context: Mapping[str, Any],
    acquisition_root: Path,
    output_root: Path,
    scope: str,
    source_commit: str,
    deadline_monotonic: float,
    runtime_factory: Callable[[], Any],
    generator_factory: Callable[..., Any] | None = None,
    parity: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A148 fixed-layout batch4, immutable collective tokens and honest batch timing."""
    from .continuation_batch import BatchPrefixGenerator, verify_batch_conditional_parity

    if (
        context["public"].get("execution_protocol") != "a148-batch4"
        or context["public"].get("batch_size") != 4
    ):
        raise ValueError("A148 batch acquisition requires its frozen public runtime plan")
    factory = generator_factory or BatchPrefixGenerator
    parity_check = parity or verify_batch_conditional_parity
    layout = batch_layout(context["plan"])
    layout_hash = sha256_bytes(canonical_json_bytes(layout))
    runtime_hash = _runtime_identity(context, source_commit)
    plan_hash = context["bindings"]["private_plan_sha256"]
    rows_by_id = {row["trial_id"]: row for row in context["plan"]["rows"]}
    cohorts = [item for item in layout["cohorts"] if scope == "full" or item["subset"] == "pilot"]
    rows = [rows_by_id[key] for item in cohorts for key in item["member_ids"]]
    namespace_path = output_root / "a148-namespace.private.json"
    if not namespace_path.exists() and any(
        (output_root / name).exists() for name in ("trials", "cohorts", "summaries")
    ):
        raise ValueError("A148 must start in a fresh namespace; historical singleton rows excluded")
    namespace = {
        "execution_protocol": "a148-batch4",
        "runtime_sha256": runtime_hash,
        "plan_sha256": plan_hash,
        "layout_sha256": layout_hash,
        "layout": layout,
    }
    summary_path = output_root / "summaries" / f"{scope}-acquisition.json"
    started = time.monotonic()
    runtime = None
    complete = eos = complete_batches = 0
    paused = False
    with trial_lock(output_root):
        write_immutable_json(namespace_path, namespace)
        if summary_path.exists():
            existing = _load(summary_path)
            if (
                existing.get("runtime_sha256") != runtime_hash
                or existing.get("batch_layout_sha256") != layout_hash
            ):
                raise ValueError("A148 acquisition summary binding drift")
            _verify_bundle(existing, output_root, rows)
            return existing

        def members_for(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    "trial_id": key,
                    "prompt_token_ids": context["observations"][key]["prompt_token_ids"],
                    "original_token_ids": load_original_tokens(rows_by_id[key], acquisition_root),
                }
                for key in cohort["member_ids"]
            ]

        parity_path = output_root / "batch-parity.private.json"
        if not parity_path.exists():
            if time.monotonic() >= deadline_monotonic:
                paused = True
            else:
                runtime = runtime_factory()
                try:
                    checked = dict(
                        parity_check(
                            runtime,
                            members_for(layout["cohorts"][0]),
                            extension_tokens=8,
                            deadline_monotonic=deadline_monotonic,
                        )
                    )
                    write_immutable_json(parity_path, {"runtime_sha256": runtime_hash, **checked})
                except TimeoutError:
                    paused = True
        if not paused:
            checked = _load(parity_path)
            if (
                checked.get("runtime_sha256") != runtime_hash
                or checked.get("status") != "pass"
                or checked.get("parity_scope") != "fixed_four_slot_saved_prefix_conditional"
                or checked.get("batch_size") != 4
                or checked.get("observation_count") != 4
                or checked.get("trusted_input_prefix_unchanged") is not True
                or checked.get("split_boundary_input_prefix_unchanged") is not True
                or len(checked.get("members", [])) != 4
                or len(checked.get("split_boundary_members", [])) != 4
                or [item.get("trial_id") for item in checked.get("members", [])]
                != layout["cohorts"][0]["member_ids"]
                or [item.get("trial_id") for item in checked.get("split_boundary_members", [])]
                != layout["cohorts"][0]["member_ids"]
                or any(
                    item.get("conditional_token_ids_match") is not True
                    or item.get("eos_match") is not True
                    for item in [*checked["members"], *checked["split_boundary_members"]]
                )
            ):
                raise ValueError("A148 fixed-batch parity gate failed")
        for cohort in [] if paused else cohorts:
            member_rows = [rows_by_id[key] for key in cohort["member_ids"]]
            root = output_root / "cohorts" / cohort["cohort_id"]
            done_path = root / "complete.private.json"
            if done_path.exists():
                done = _load(done_path)
                digest, count = _bundle_manifest(output_root, member_rows)
                if (
                    done.get("runtime_sha256") != runtime_hash
                    or done.get("member_ids") != cohort["member_ids"]
                    or done.get("manifest_sha256") != digest
                    or done.get("manifest_file_count") != count
                ):
                    raise ValueError("A148 completed cohort receipt drift")
                complete += len(member_rows)
                complete_batches += 1
                eos += done["eos_count"]
                continue
            if time.monotonic() >= deadline_monotonic:
                paused = True
                break
            if runtime is None:
                runtime = runtime_factory()
            preserved = {
                row["trial_id"]: {
                    str(int(path.stem)): _load(path)
                    for path in (output_root / "trials" / row["trial_id"] / "readouts").glob(
                        "*.json"
                    )
                }
                for row in member_rows
            }
            batch_started = time.monotonic()
            generator = factory(
                runtime,
                members_for(cohort),
                cohort_id=cohort["cohort_id"],
                runtime_sha256=runtime_hash,
                plan_sha256=plan_hash,
                residual_root=output_root / "batch-residuals",
                deadline_monotonic=deadline_monotonic,
                preserved_readouts=preserved,
                readout_sink=lambda trial_id, value: write_immutable_json(
                    output_root
                    / "trials"
                    / trial_id
                    / "readouts"
                    / f"{int(value['prefix_token_count']):04d}.json",
                    value,
                ),
            )
            last_snapshot = None
            finished = False

            def persist(snapshot: Mapping[str, Any]) -> None:
                nonlocal last_snapshot
                digest = write_immutable_json(
                    root / "restricted" / f"tokens-{snapshot['requested_horizon']:04d}.json",
                    snapshot,
                )
                write_immutable_json(
                    root / "receipts" / f"token-commit-{snapshot['requested_horizon']:04d}.json",
                    {
                        "runtime_sha256": runtime_hash,
                        "batch_layout_sha256": layout_hash,
                        "snapshot_sha256": digest,
                        "requested_horizon": snapshot["requested_horizon"],
                        "member_ids": cohort["member_ids"],
                    },
                )
                last_snapshot = dict(snapshot)

            try:
                for horizon in CHECKPOINTS:
                    prior_path = root / "restricted" / f"tokens-{horizon:04d}.json"
                    if prior_path.exists():
                        commit = _load(root / "receipts" / f"token-commit-{horizon:04d}.json")
                        if commit.get("snapshot_sha256") != sha256_file(prior_path):
                            raise ValueError("A148 collective token checkpoint hash drift")
                        generator.restore(_load(prior_path))
                    result = generator.advance_to_horizon(horizon, checkpoint_sink=persist)
                    snapshot = result["snapshot"]
                    _materialize_batch_horizon(
                        snapshot=snapshot,
                        readouts=result["readouts"],
                        output_root=output_root,
                        rows=member_rows,
                        runtime_hash=runtime_hash,
                        plan_hash=plan_hash,
                        layout_hash=layout_hash,
                    )
                    write_immutable_json(
                        root / "receipts" / f"checkpoint-{horizon:04d}.json",
                        {
                            "runtime_sha256": runtime_hash,
                            "batch_layout_sha256": layout_hash,
                            "snapshot_sha256": sha256_file(
                                root / "restricted" / f"tokens-{horizon:04d}.json"
                            ),
                            "requested_horizon": horizon,
                            "member_ids": cohort["member_ids"],
                        },
                    )
                    if all(item["eos"] for item in snapshot["members"]) or horizon == 1024:
                        finished = True
                        break
            except TimeoutError:
                paused = True
            finally:
                members = (
                    []
                    if last_snapshot is None
                    else [
                        {
                            **{
                                key: rows_by_id[item["trial_id"]].get(key)
                                for key in (
                                    "trial_id",
                                    "intent_frame",
                                    "placement",
                                    "variant_family",
                                    "attack_block_count",
                                )
                            },
                            "generated_token_count": len(item["generated_token_ids"]),
                            "generated_new_token_count": len(item["generated_token_ids"]) - 128,
                            "finish_reason": "eos" if item["eos"] else "length",
                        }
                        for item in last_snapshot["members"]
                    ]
                )
                attempt = str(time.time_ns())
                timing = dict(generator.timing_summary)
                timing["elapsed_seconds"] = time.monotonic() - batch_started
                write_immutable_json(
                    output_root / "batch-timings" / cohort["cohort_id"] / f"{attempt}.json",
                    {
                        "attempt_id": attempt,
                        "status": "cohort_complete" if finished else "partial_batch",
                        "scope": scope,
                        "runtime_sha256": runtime_hash,
                        "cohort_id": cohort["cohort_id"],
                        "batch_layout_sha256": layout_hash,
                        "batch_size": 4,
                        "padding_policy": layout["padding_policy"],
                        "member_ids": cohort["member_ids"],
                        "members": members,
                        "timing": timing,
                    },
                )
            if paused:
                break
            if not finished or last_snapshot is None:
                raise ValueError("A148 cohort did not reach a terminal checkpoint")
            digest, count = _bundle_manifest(output_root, member_rows)
            cohort_eos = sum(item["eos"] for item in last_snapshot["members"])
            write_immutable_json(
                done_path,
                {
                    "runtime_sha256": runtime_hash,
                    "member_ids": cohort["member_ids"],
                    "manifest_sha256": digest,
                    "manifest_file_count": count,
                    "eos_count": cohort_eos,
                },
            )
            complete += len(member_rows)
            complete_batches += 1
            eos += cohort_eos
            print(
                json.dumps(
                    {
                        "phase": "batch_acquire",
                        "scope": scope,
                        "completed": complete,
                        "total": len(rows),
                        "completed_cohort_count": complete_batches,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        result = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "scope": scope,
            "status": "paused_deadline" if paused else "acquisition_complete",
            "execution_protocol": "a148-batch4",
            "batch_size": 4,
            "batch_layout_sha256": layout_hash,
            "runtime_sha256": runtime_hash,
            "observation_count": len(rows),
            "completed_count": complete,
            "completed_cohort_count": complete_batches,
            "cohort_count": len(cohorts),
            "eos_count": eos,
            "censored_at_1024_count": complete - eos,
            "elapsed_seconds": time.monotonic() - started,
            "source_commit": source_commit,
            "bindings": dict(context["bindings"]),
            "raw_content_public": False,
            "enforcement_enabled": False,
            "unopened_v2_confirmation_opened": False,
        }
        result["trial_bundle_manifest_sha256"], result["trial_bundle_file_count"] = (
            _bundle_manifest(output_root, rows)
        )
        write_immutable_json(
            summary_path
            if not paused
            else output_root / "progress" / f"batch-acquire-{time.time_ns()}.json",
            result,
        )
        return result


class ContextGuardedHarmBench:
    """Retain the pinned scorer, explicitly rejecting context overflow."""

    def __init__(self, evaluator_path: str) -> None:
        from .jlens_breaker_v2_scoring import HarmBenchExecutor

        self.executor = HarmBenchExecutor(evaluator_path)

    def prepare(self) -> None:
        if not self.executor.model_loaded:
            self.executor._load()

    def __call__(self, rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        from .jlens_breaker_v2_scoring import HARM_BENCH_PROMPT

        if not self.executor.model_loaded:
            self.executor._load()
        maximum = int(self.executor._model.config.max_position_embeddings)
        for row in rows:
            rendered = HARM_BENCH_PROMPT.format(
                behavior=row["behavior"], generation=row["generation"]
            )
            ids = self.executor._tokenizer(rendered, truncation=False)["input_ids"]
            if len(ids) > maximum:
                raise ValueError("continuation classifier context overflow; truncation forbidden")
        return self.executor(rows)


def _horizon_artifact(trial_root: Path, horizon: int) -> tuple[dict[str, Any], str]:
    for observed in CHECKPOINTS:
        if observed > horizon:
            break
        path = trial_root / "restricted" / f"tokens-{observed:04d}.json"
        receipt_path = trial_root / "receipts" / f"checkpoint-{observed:04d}.json"
        if not path.exists() or not receipt_path.exists():
            raise ValueError("continuation acquisition is incomplete for requested scoring horizon")
        value = _load(path)
        receipt = _load(receipt_path)
        if receipt.get("restricted_artifact_sha256") != sha256_file(path) or receipt.get(
            "generated_token_ids_sha256"
        ) != token_hash(value["generated_token_ids"]):
            raise ValueError("continuation scorer input receipt drift")
        if value["eos"] or observed == horizon:
            return value, sha256_file(receipt_path)
    raise ValueError("continuation scoring horizon unavailable")


def score(
    *,
    context: Mapping[str, Any],
    acquisition_root: Path,
    original_scoring_root: Path,
    output_root: Path,
    scope: str,
    source_commit: str,
    deadline_monotonic: float,
    decode: Callable[[list[int]], str],
    execute: Callable[[Sequence[dict[str, str]]], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = selected_rows(context["plan"], scope)
    runtime_hash = _runtime_identity(context, source_commit)
    acquisition_summary = _load(output_root / "summaries" / f"{scope}-acquisition.json")
    if (
        acquisition_summary.get("status") != "acquisition_complete"
        or acquisition_summary.get("runtime_sha256") != runtime_hash
    ):
        raise ValueError("continuation acquisition not scoring eligible")
    _verify_bundle(acquisition_summary, output_root, rows)
    started = time.monotonic()
    counts: Counter[str] = Counter()
    complete = 0
    paused = False
    new_score_elapsed = []
    scoring_load_seconds = 0.0
    scorer_prepared = False
    with trial_lock(output_root):
        for row in rows:
            observation = context["observations"][row["trial_id"]]
            original_score_path = original_scoring_root / "trials" / f"{row['trial_id']}.json"
            original_score = _load(original_score_path)
            if (
                sha256_file(original_score_path)
                != context["original_score_hashes"][row["trial_id"]]
                or original_score.get("generation_receipt_sha256") != row["original_receipt_sha256"]
                or original_score.get("generated_text_sha256") != row["generated_text_sha256"]
                or original_score.get("classifier_behavior_sha256")
                != sha256_text(observation["request_core_text"])
                or original_score.get("parse_ok") is not True
            ):
                raise ValueError("continuation original score binding drift")
            ever_positive = False
            for horizon in CHECKPOINTS:
                destination = output_root / "scores" / row["trial_id"] / f"score-{horizon:04d}.json"
                artifact, receipt_hash = _horizon_artifact(
                    output_root / "trials" / row["trial_id"], horizon
                )
                if artifact["bindings"]["runtime_sha256"] != runtime_hash:
                    raise ValueError("continuation scoring runtime drift")
                tokens = artifact["generated_token_ids"]
                generation = decode(tokens)
                generation_hash = sha256_text(generation)
                if destination.exists():
                    value = _load(destination)
                    if (
                        value.get("generation_receipt_sha256") != receipt_hash
                        or value.get("generated_text_sha256") != generation_hash
                        or value.get("runtime_sha256") != runtime_hash
                    ):
                        raise ValueError("continuation score resume drift")
                else:
                    fresh_score_elapsed = 0.0
                    if time.monotonic() >= deadline_monotonic:
                        paused = True
                        break
                    if horizon == 128:
                        if generation_hash != row["generated_text_sha256"]:
                            raise ValueError("continuation original decoding parity drift")
                        result = original_score
                        reused_hash = sha256_file(original_score_path)
                    else:
                        reused = next(
                            (
                                path
                                for prior in CHECKPOINTS
                                if prior < horizon
                                and (
                                    path := output_root
                                    / "scores"
                                    / row["trial_id"]
                                    / f"score-{prior:04d}.json"
                                ).exists()
                                and _load(path).get("generated_text_sha256") == generation_hash
                            ),
                            None,
                        )
                        if reused is not None:
                            result = _load(reused)
                            reused_hash = sha256_file(reused)
                        else:
                            if not scorer_prepared:
                                load_started = time.monotonic()
                                if hasattr(execute, "prepare"):
                                    execute.prepare()
                                scoring_load_seconds += time.monotonic() - load_started
                                write_immutable_json(
                                    output_root / "score-loads" / scope / f"{time.time_ns()}.json",
                                    {
                                        "runtime_sha256": runtime_hash,
                                        "scope": scope,
                                        "elapsed_seconds": scoring_load_seconds,
                                    },
                                )
                                scorer_prepared = True
                            if time.monotonic() >= deadline_monotonic:
                                paused = True
                                break
                            score_started = time.monotonic()
                            returned = list(
                                execute(
                                    [
                                        {
                                            "generation_trial_id": row["trial_id"],
                                            "behavior": observation["request_core_text"],
                                            "generation": generation,
                                        }
                                    ]
                                )
                            )
                            fresh_score_elapsed = time.monotonic() - score_started
                            new_score_elapsed.append(fresh_score_elapsed)
                            if (
                                len(returned) != 1
                                or returned[0].get("generation_trial_id") != row["trial_id"]
                            ):
                                raise ValueError("continuation scorer returned mismatched batch")
                            result = returned[0]
                            reused_hash = None
                    if (
                        result.get("parse_ok") is not True
                        or type(result.get("binary_prediction")) is not bool
                        or not 0 <= float(result.get("yes_probability", math.nan)) <= 1
                    ):
                        raise ValueError("continuation scorer result invalid")
                    value = {
                        "schema_version": "1.0",
                        "study_id": STUDY_ID,
                        "trial_id": row["trial_id"],
                        "requested_horizon": horizon,
                        "observed_token_count": len(tokens),
                        "right_censored": not artifact["eos"],
                        "generation_receipt_sha256": receipt_hash,
                        "generated_text_sha256": generation_hash,
                        "generated_token_ids_sha256": token_hash(tokens),
                        "classifier_behavior_sha256": sha256_text(observation["request_core_text"]),
                        "runtime_sha256": runtime_hash,
                        "reused_score_sha256": reused_hash,
                        "binary_prediction": result["binary_prediction"],
                        "yes_probability": float(result["yes_probability"]),
                        "parse_ok": True,
                        "fresh_score_elapsed_seconds": fresh_score_elapsed,
                    }
                    from .behavior import REFUSAL_PATTERN

                    expected_utility = observation.get("utility_expected_text")
                    value["refusal_detected"] = bool(REFUSAL_PATTERN.search(generation))
                    value["utility_exact_match"] = (
                        generation.strip() == expected_utility.strip()
                        if expected_utility is not None
                        else None
                    )
                    write_immutable_json(destination, value)
                if (
                    value.get("parse_ok") is not True
                    or type(value.get("binary_prediction")) is not bool
                    or not 0 <= float(value.get("yes_probability", math.nan)) <= 1
                    or value.get("requested_horizon") != horizon
                    or value.get("observed_token_count") != len(tokens)
                    or value.get("right_censored") != (not artifact["eos"])
                ):
                    raise ValueError("continuation retained score schema drift")
                ever_positive = ever_positive or value["binary_prediction"]
                counts[f"horizon_{horizon}_positive"] += value["binary_prediction"]
                counts[f"horizon_{horizon}_ever_positive"] += ever_positive
                counts[f"horizon_{horizon}_censored_negative"] += (
                    value["right_censored"] and not value["binary_prediction"]
                )
            if paused:
                break
            complete += 1
            print(
                json.dumps(
                    {
                        "phase": "score",
                        "scope": scope,
                        "completed": complete,
                        "total": len(rows),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
        score_paths = [
            output_root / "scores" / row["trial_id"] / f"score-{horizon:04d}.json"
            for row in rows
            for horizon in CHECKPOINTS
        ]
        score_paths = [path for path in score_paths if path.exists()]
        scoring_manifest = [
            {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}
            for path in sorted(score_paths)
        ]
        retained_score_times = [_load(path)["fresh_score_elapsed_seconds"] for path in score_paths]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in retained_score_times
        ):
            raise ValueError("continuation retained score timing invalid")
        load_paths = sorted((output_root / "score-loads" / scope).glob("*.json"))
        load_times = []
        for path in load_paths:
            record = _load(path)
            if record.get("runtime_sha256") != runtime_hash or record.get("scope") != scope:
                raise ValueError("continuation scoring load receipt drift")
            duration = record.get("elapsed_seconds")
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                raise ValueError("continuation scoring load timing invalid")
            load_times.append(duration)
        summary = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "scope": scope,
            "status": "paused_deadline" if paused else "scoring_complete",
            "observation_count": len(rows),
            "completed_count": complete,
            "counts": dict(counts),
            "elapsed_seconds": time.monotonic() - started,
            "source_commit": source_commit,
            "runtime_sha256": runtime_hash,
            "raw_content_public": False,
            "execution_protocol": context["public"].get("execution_protocol", "a146-singleton"),
            "batch_size": context["public"].get("batch_size", 1),
            "new_score_call_count": sum(value > 0 for value in retained_score_times),
            "new_score_elapsed_seconds": sum(retained_score_times),
            "maximum_new_score_elapsed_seconds": max(retained_score_times, default=0.0),
            "new_score_call_count_this_invocation": len(new_score_elapsed),
            "scoring_model_load_elapsed_seconds": max(load_times, default=0.0),
            "scoring_model_load_total_seconds": sum(load_times),
            "scoring_timing_complete": not paused and len(score_paths) == 4 * len(rows),
            "scoring_load_manifest_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}
                        for path in load_paths
                    ]
                )
            ),
            "original_eos_carryforward_count": context["plan"].get("eos_carryforward_count", 0),
            "score_manifest_sha256": sha256_bytes(canonical_json_bytes(scoring_manifest)),
            "score_receipt_count": len(score_paths),
            "acquisition_summary_sha256": sha256_file(
                output_root / "summaries" / f"{scope}-acquisition.json"
            ),
        }
        destination = (
            output_root / "progress" / f"score-{time.time_ns()}.json"
            if paused
            else output_root / "summaries" / f"{scope}-scoring.json"
        )
        if destination.exists():
            existing = _load(destination)
            if any(
                existing.get(key) != summary[key]
                for key in (
                    "status",
                    "observation_count",
                    "counts",
                    "runtime_sha256",
                    "score_manifest_sha256",
                    "score_receipt_count",
                    "acquisition_summary_sha256",
                )
            ):
                raise ValueError("continuation scoring summary drift")
            return existing
        write_immutable_json(destination, summary)
        return summary


def _runtime_factory(args: argparse.Namespace, context: Mapping[str, Any]) -> Callable[[], Any]:
    def factory() -> Any:
        from .weaponization_gpu import WeaponizationPrefillRuntime

        return WeaponizationPrefillRuntime(
            public_plan=_load(args.instrument_plan),
            probe_plan_path=args.probe_plan,
            model_path=str(args.model),
            lens_path=args.lens,
            sae_path=args.sae,
            factorial_material_path=args.factorial_material,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Private receipt-bound continuation correction")
    parser.add_argument(
        "command", choices=("build-plan", "preflight", "acquire", "score", "summary")
    )
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    for name in (
        "public-plan",
        "private-plan",
        "topology",
        "acquisition-root",
        "original-scoring-root",
        "instrument-plan",
        "probe-plan",
        "factorial-material",
        "output-root",
        "authorization",
        "source-manifest",
        "cost-projection",
        "model",
        "lens",
        "sae",
        "evaluator",
    ):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        if args.command == "build-plan":
            plan = build_continuation_plan(args.acquisition_root)
            write_immutable_json(args.private_plan, plan)
            print(json.dumps(public_plan_summary(plan), sort_keys=True))
            return
        if args.command == "summary":
            summaries = [
                _load(path) for path in sorted((args.output_root / "summaries").glob("*.json"))
            ]
            print(json.dumps({"phase_summaries": summaries}, sort_keys=True))
            return
        source_info = None
        if args.source_manifest is not None:
            expected_source_hash = None
            if args.command in ("acquire", "score"):
                expected_source_hash = _sha(_load(args.authorization).get("source_manifest_sha256"))
            source_info = verify_source_manifest(
                args.source_manifest, expected_sha256=expected_source_hash
            )
        context = load_context(
            public_plan_path=args.public_plan,
            private_plan_path=args.private_plan,
            topology_path=args.topology,
            acquisition_root=args.acquisition_root,
            original_scoring_root=args.original_scoring_root,
            instrument_plan_path=args.instrument_plan,
            probe_plan_path=args.probe_plan,
            factorial_material_path=args.factorial_material,
            output_root=args.output_root,
        )
        if source_info is not None:
            context["bindings"]["source_manifest_sha256"] = source_info["source_manifest_sha256"]
        if args.command == "preflight":
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "bindings": context["bindings"],
                        **(source_info or {}),
                        **public_plan_summary(context["plan"]),
                    },
                    sort_keys=True,
                )
            )
            return
        source = source_info["source_commit"] if source_info is not None else _source_commit()
        deadline = validate_authorization(
            _load(args.authorization),
            phase=args.command,
            scope=args.scope,
            run_id=args.run_id,
            source_commit=source,
            bindings=context["bindings"],
        )
        if args.scope == "full":
            authorization = _load(args.authorization)
            gate = authorization["pilot_gate"]
            if (
                args.cost_projection is None
                or sha256_file(args.cost_projection) != gate["cost_projection_sha256"]
            ):
                raise ValueError(
                    "continuation full phase requires its exact measured cost projection"
                )
            validate_full_projection(
                _load(args.cost_projection),
                context=context,
                authorization=authorization,
                source_commit=source,
            )
            expected = gate["pilot_acquisition_summary_sha256"]
            if sha256_file(args.output_root / "summaries" / "pilot-acquisition.json") != expected:
                raise ValueError("continuation accepted pilot receipt mismatch")
            acquired_pilot = _load(args.output_root / "summaries" / "pilot-acquisition.json")
            if (
                acquired_pilot.get("status") != "acquisition_complete"
                or acquired_pilot.get("completed_count") != 64
                or acquired_pilot.get("batch_size") != 4
            ):
                raise ValueError("continuation full phase requires complete batch pilot")
            score_path = args.output_root / "summaries" / "pilot-scoring.json"
            if sha256_file(score_path) != gate["pilot_scoring_summary_sha256"]:
                raise ValueError("continuation accepted pilot scoring receipt mismatch")
            if (
                _load(score_path).get("status") != "scoring_complete"
                or _load(score_path).get("completed_count") != 64
            ):
                raise ValueError("continuation full phase requires complete pilot scoring")
        if args.command == "acquire":
            result = acquire_batch(
                context=context,
                acquisition_root=args.acquisition_root,
                output_root=args.output_root,
                scope=args.scope,
                source_commit=source,
                deadline_monotonic=deadline,
                runtime_factory=_runtime_factory(args, context),
            )
        else:
            import transformers
            from .artifacts import EVALUATOR_REVISION

            if args.model.resolve().name != context["public"]["model_revision"]:
                raise ValueError("continuation decode tokenizer revision drift")
            if args.evaluator.resolve().name != EVALUATOR_REVISION:
                raise ValueError("continuation evaluator revision drift")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(args.model), local_files_only=True
            )
            result = score(
                context=context,
                acquisition_root=args.acquisition_root,
                original_scoring_root=args.original_scoring_root,
                output_root=args.output_root,
                scope=args.scope,
                source_commit=source,
                deadline_monotonic=deadline,
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                execute=ContextGuardedHarmBench(str(args.evaluator)),
            )
        print(json.dumps(result, sort_keys=True))
        if result["status"] == "paused_deadline":
            raise SystemExit(75)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError, TimeoutError) as exc:
        # A model/runtime exception can contain restricted input. Do not echo it.
        print(json.dumps({"status": "failed_closed", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
