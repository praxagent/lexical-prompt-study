"""A175 fixed post hoc, stdlib-only structure audit. No raw reads at import.

Only a bound, reviewed audit execution may inspect the already-verified parent
responses. Features never replace the parent's primary scores or termination.
"""
from __future__ import annotations

from collections import Counter
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import time
import uuid

from . import output_structure_audit as previous

A172_SHA = "4a5628bfd6bc835fb476e717d01197b4294c855ad7dfaf6798814f7245740c02"

PARENT_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a174/run-001")
AUDIT_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a175/run-001")
PARENT_ARTIFACT_SHA256 = {
    "FREEZE.json": "2165509b7aae096dd17e02d32b7aec3b1a7de95a01f44cd76008028e10d38d4a",
    "VERIFICATION.json": "9a2ca3fe840bc6ca4a8a20e53ed70486fdbbd2b0af5695f0cbcc92f16c5df364",
    "inputs/plan.json": "5a5a7afc3b17eea4c0d3ad387552a10a5a75afb68a9053b1903fcaa3515c94f4",
    "execution/analysis.json": "c4072d3ad6a38741fefd012d6aea851cc457b33d3b4f26c13115be1d05c9f7f9",
    "execution/records.json": "dcae744d2c14bd3f0bcc41c8dd28b0dbfa5ff0ead608974f0e0540a4c47ce9b4",
    "execution/run.json": "09e4e176c7717cf0939a2ff9c1d401fae5429aef2625078f35ea5bf99e15a485",
    "execution/execution-finished.json": "b9d6c7134c96fdaef25148e749db3742448f7f3e4ee8fe32aa23276da142bb5a",
}
INSTRUCTION_CORES = CLOSING_CONSTRAINTS = ("legacy", "a171_standard")
OBJECT_CATEGORIES, FEATURES, OVERLAPS = previous.OBJECT_CATEGORIES, previous.FEATURES, previous.OVERLAPS
inspect_output = previous.inspect_output
_validate_features, _group, _strata = previous._validate_features, previous._group, previous._strata
MAX_SECONDS, KILL_GRACE_SECONDS = 600, 60
_PROCESS_CONSUMED = False


def require(condition, code):
    if not condition:
        raise ValueError("a175_" + code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _same(left, right):
    return canonical(left) == canonical(right)


def file_sha(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def is_hash(value, length=64):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("a175_duplicate_json_key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("a175_nonstandard_json_constant")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def _plain(path, directory=False):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "plain_absolute_path")
    require(path.is_dir() if directory else path.is_file() and stat.S_ISREG(path.stat().st_mode), "plain_path_type")


def _bound_bytes(path, sha256):
    _plain(path)
    require(is_hash(sha256), "artifact_hash_schema")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha256, "artifact_hash_drift")
    return raw


def _read(path):
    _plain(path)
    return _json(path.read_bytes())


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _inventory(base, expected):
    _plain(base, directory=True)
    require(type(expected) is dict and bool(expected), "inventory_schema")
    for relative, sha256 in expected.items():
        require(type(relative) is str and relative and str(Path(relative)) == relative
                and not Path(relative).is_absolute() and not {".", ".."}.intersection(Path(relative).parts)
                and "\\" not in relative and is_hash(sha256), "inventory_entry")
    actual = set()
    for path in base.rglob("*"):
        require(not path.is_symlink(), "inventory_symlink")
        if path.is_dir():
            continue
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "inventory_special_file")
        actual.add(path.relative_to(base).as_posix())
    require(actual == set(expected), "exact_inventory")
    for relative, sha256 in expected.items():
        require(file_sha(base / relative) == sha256, "inventory_hash_drift")


def compile_plan(parent_bindings, protocol_sha256, tests_sha256):
    """Bind only supplied verification metadata; never open any parent file."""
    require(file_sha(Path(previous.__file__)) == A172_SHA, "pinned_detector_source")
    require(type(parent_bindings) is dict and set(parent_bindings) == {"root", "artifact_sha256", "result_sha256"}, "parent_binding_schema")
    require(parent_bindings["root"] == str(PARENT_ROOT) and parent_bindings["artifact_sha256"] == PARENT_ARTIFACT_SHA256,
            "fixed_parent_artifacts")
    hashes = parent_bindings["result_sha256"]
    require(type(hashes) is dict and len(hashes) == 32 and all(is_hash(k, 24) and is_hash(v) for k, v in hashes.items()),
            "complete_result_hash_manifest")
    require(is_hash(protocol_sha256) and is_hash(tests_sha256), "plan_hash_bindings")
    return {"schema_version": "a175-plan-v1", "posthoc": True, "planned_outputs": 32, "model_calls_allowed": 0,
            "parent_bindings": _json(canonical(parent_bindings)),
            "bindings": {"source_sha256": file_sha(Path(__file__)), "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256, "a172_helper_sha256": A172_SHA},
            "detectors": {"fence": "literal_ascii_triple_backtick_or_triple_tilde_anywhere",
                "answer_objects": "strict_raw_decode_at_every_literal_open_brace_exact_answer_string_object",
                "repetition": "overlapping_four_token_gram_frequency_at_least_four_remove_only_final_frozen_eos"},
            "object_categories": list(OBJECT_CATEGORIES), "overlap_feature_order": list(FEATURES),
            "maximum_seconds": MAX_SECONDS, "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0,
            "resume_allowed": False, "primary_scores_changed": False}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_schema")
    expected = compile_plan(plan["parent_bindings"], plan["bindings"]["protocol_sha256"], plan["bindings"]["tests_sha256"])
    require(canonical(plan) == canonical(expected), "fixed_plan_drift")


def summarize(records):
    require(type(records) is list and all(r["instruction_core"] in INSTRUCTION_CORES
            and r["closing_constraint"] in CLOSING_CONSTRAINTS and type(r["original_capped"]) is bool for r in records), "summary_metadata")
    groups = {}
    for core in INSTRUCTION_CORES:
        groups[core] = {}
        for constraint in CLOSING_CONSTRAINTS:
            subset = [r for r in records if r["instruction_core"] == core and r["closing_constraint"] == constraint]
            groups[core][constraint] = {"all": _group(subset), "by_cap_status": _strata(subset)}
    return {"overall": _group(records), "by_cap_status": _strata(records), "by_condition": groups}


def _audit_inputs(plan_path, plan_sha256, protocol_path, freeze_path, freeze_sha256):
    require(freeze_path == AUDIT_ROOT / "FREEZE.json" and plan_path == AUDIT_ROOT / "inputs/plan.json"
            and protocol_path == AUDIT_ROOT / "frozen/plans/crossover_output_audit_a175.md", "fixed_audit_paths")
    plan = _json(_bound_bytes(plan_path, plan_sha256))
    validate_plan(plan)
    freeze = _json(_bound_bytes(freeze_path, freeze_sha256))
    require(freeze.get("schema_version") == "a175-freeze-v1" and is_hash(freeze.get("source_commit"), 40)
            and freeze.get("pushed_commit") == freeze["source_commit"], "reviewed_pushed_freeze")
    _inventory(AUDIT_ROOT / "frozen", freeze["source_files_sha256"])
    _inventory(AUDIT_ROOT / "inputs", freeze["input_sha256"])
    source_key, protocol_key, tests_key = "src/lexical_prompt_study/crossover_output_audit.py", "plans/crossover_output_audit_a175.md", "tests/test_crossover_output_audit.py"
    require(freeze["input_sha256"].get("plan.json") == plan_sha256 == freeze.get("plan_sha256"), "frozen_plan")
    for field, key in (("source_sha256", source_key), ("protocol_sha256", protocol_key), ("tests_sha256", tests_key)):
        require(freeze.get(field) == plan["bindings"][field] == freeze["source_files_sha256"].get(key), "frozen_scientific_binding")
    require(freeze.get("a172_helper_sha256") == plan["bindings"]["a172_helper_sha256"] == A172_SHA
            == freeze["source_files_sha256"].get("src/lexical_prompt_study/output_structure_audit.py"), "frozen_detector_binding")
    require(file_sha(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_hash")
    require(freeze.get("parent_verification_sha256") == PARENT_ARTIFACT_SHA256["VERIFICATION.json"]
            and type(freeze.get("target_model_calls")) is int and freeze["target_model_calls"] == 0, "audit_scope")
    review = _json(_bound_bytes(AUDIT_ROOT / "REVIEW.json", freeze["review_sha256"]))
    require(review.get("status") == "pass", "review_pass")
    return plan, freeze


def _parent_metadata(plan):
    """Verify metadata before any response-result bytes are opened."""
    bindings = plan["parent_bindings"]
    raw = {name: _bound_bytes(PARENT_ROOT / name, sha256) for name, sha256 in bindings["artifact_sha256"].items()}
    values = {name: _json(value) for name, value in raw.items()}
    freeze, verification = values["FREEZE.json"], values["VERIFICATION.json"]
    _inventory(PARENT_ROOT / "frozen", freeze["source_files_sha256"])
    _inventory(PARENT_ROOT / "inputs", freeze["input_sha256"])
    for name, key in (("REVIEW.json", "review_sha256"), ("NATIVE-AUDIT.json", "native_audit_sha256")):
        _bound_bytes(PARENT_ROOT / name, freeze[key])
    coverage = {"attempted": 32, "completed": 32, "infrastructure_failed": 0, "interrupted": 0, "unattempted": 0, "missing": 0}
    require(verification.get("schema_version") == "a174-independent-verification-v1" and verification.get("status") == "pass"
            and _same(verification.get("coverage"), coverage) and verification.get("terminal_status") == "finished_schedule"
            and verification.get("independent_native_constructions") == 32 and verification.get("independent_completed_response_scores") == 32
            and verification.get("independent_all_planned_rows_and_contrasts_replayed") is True
            and verification.get("model_forward_reexecuted") is False, "parent_verified_complete")
    require(verification["source_files"] == len(freeze["source_files_sha256"])
            and verification["input_files"] == len(freeze["input_sha256"])
            and verification["model_files"] == len(freeze["model_files_sha256"]), "parent_verification_counts")
    require(verification["verifier_sha256"] == freeze["source_files_sha256"]["verification/independent_a174.py"], "parent_verifier_binding")
    for field, path in (("freeze_sha256", "FREEZE.json"), ("analysis_sha256", "execution/analysis.json"),
                        ("records_sha256", "execution/records.json"), ("run_sha256", "execution/run.json"),
                        ("execution_finished_sha256", "execution/execution-finished.json")):
        require(verification[field] == bindings["artifact_sha256"][path], "parent_verification_artifact")
    require(verification["result_sha256"] == bindings["result_sha256"], "parent_result_manifest")
    parent_plan, records = values["inputs/plan.json"], values["execution/records.json"]
    analysis, header, finished = values["execution/analysis.json"], values["execution/run.json"], values["execution/execution-finished.json"]
    require(parent_plan.get("schema_version") == "a174-plan-v1" and len(parent_plan["trials"]) == 32
            and len(records) == 32 and {t["trial_id"] for t in parent_plan["trials"]} == set(bindings["result_sha256"]), "parent_cohort")
    require(freeze["input_sha256"]["plan.json"] == bindings["artifact_sha256"]["inputs/plan.json"] == header["plan_sha256"]
            and header["config_sha256"] == freeze["input_sha256"]["runtime-config.json"], "parent_plan_config_binding")
    require(header.get("schema_version") == "a174-run-v1" and finished.get("schema_version") == "a174-execution-finished-v1"
            and finished["status"] == "finished_schedule" and finished["run_sha256"] == object_sha(header)
            and finished["interruption"] is None, "parent_terminal_binding")
    require(_same(analysis["coverage"], coverage) and analysis["status"] == "complete" and analysis["execution_status"] == "finished_schedule"
            and analysis["numeric_records_sha256"] == object_sha(records) and analysis["plan_object_sha256"] == object_sha(parent_plan)
            and analysis["provenance"]["run_sha256"] == object_sha(header), "parent_analysis_binding")
    require(type(header["eos_token_ids"]) is list and header["eos_token_ids"]
            and all(type(t) is int and t >= 0 for t in header["eos_token_ids"]), "parent_eos_schema")
    for index, (trial, record) in enumerate(zip(parent_plan["trials"], records)):
        require(trial["sequence_index"] == record["sequence_index"] == index
                and all(trial[k] == record[k] for k in ("trial_id", "instruction_core", "closing_constraint", "selector"))
                and trial["instruction_core"] in INSTRUCTION_CORES and trial["closing_constraint"] in CLOSING_CONSTRAINTS
                and type(record["capped"]) is bool and type(record["fence_marker"]) is bool and record["attempted"] is True
                and record["category"] not in ("infrastructure", "missing"), "parent_row_binding")
    require(Counter((t["instruction_core"], t["closing_constraint"]) for t in parent_plan["trials"]) ==
            Counter({(c, k): 8 for c in INSTRUCTION_CORES for k in CLOSING_CONSTRAINTS}), "parent_four_groups")
    require(sum(record["capped"] for record in records) == 8, "parent_cap_strata")
    folders = PARENT_ROOT / "execution/trials"
    _plain(folders, directory=True)
    require({p.name for p in folders.iterdir()} == set(bindings["result_sha256"]), "parent_result_inventory")
    for trial in parent_plan["trials"]:
        folder = folders / trial["trial_id"]
        _plain(folder, directory=True)
        require({p.name for p in folder.iterdir()} == {"attempt.json", "result.json"}, "parent_trial_inventory")
    return parent_plan["trials"], records, header


def _result_bytes(plan, trials, before_read=None):
    """All 32 byte hashes pass before any response JSON is parsed."""
    result = {}
    for trial in trials:
        if before_read is not None:
            before_read(trial)
        result[trial["trial_id"]] = _bound_bytes(PARENT_ROOT / "execution/trials" / trial["trial_id"] / "result.json",
                                                plan["parent_bindings"]["result_sha256"][trial["trial_id"]])
    return result


def _parent_result(raw, trial, record, header):
    result = _json(raw)
    require(result.get("schema_version") == "a174-result-v1" and result["status"] == "completed" and result["error"] is None,
            "parent_result_completed")
    attempt = _read(PARENT_ROOT / "execution/trials" / trial["trial_id"] / "attempt.json")
    require(result["attempt_sha256"] == object_sha(attempt) and attempt["run_sha256"] == object_sha(header)
            and attempt["trial_id"] == trial["trial_id"] and attempt["trial_sha256"] == object_sha(trial), "parent_attempt_binding")
    score = result["score"]
    require(type(score) is dict and type(score["response_text"]) is str
            and all(_same(record[key], value) for key, value in score.items() if key != "response_text"), "parent_saved_score_binding")
    tokens = result["generated_token_ids"]
    require(type(tokens) is list and 1 <= len(tokens) <= 64 and all(type(t) is int and t >= 0 for t in tokens)
            and score["generated_token_count"] == len(tokens), "parent_saved_tokens")
    return score["response_text"], tokens


def _record(trial, parent_record, features):
    return {"trial_id": trial["trial_id"], "sequence_index": trial["sequence_index"], "instruction_core": trial["instruction_core"],
            "closing_constraint": trial["closing_constraint"], "original_category": parent_record["category"],
            "original_capped": parent_record["capped"], "features": features}


def _attempt(run, trial, plan):
    return {"schema_version": "a175-attempt-v1", "run_sha256": object_sha(run), "trial_id": trial["trial_id"],
            "sequence_index": trial["sequence_index"], "parent_result_sha256": plan["parent_bindings"]["result_sha256"][trial["trial_id"]]}


def _run_header(plan, plan_sha256, freeze_sha256, pid):
    return {"schema_version": "a175-run-v1", "pid": pid, "plan_sha256": plan_sha256, "freeze_sha256": freeze_sha256,
            "source_sha256": plan["bindings"]["source_sha256"], "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "a172_helper_sha256": A172_SHA, "parent_bindings_sha256": object_sha(plan["parent_bindings"]),
            "planned_outputs": 32, "target_model_calls": 0, "posthoc": True}


def _aggregate(plan, run, records, attempted, processed, status):
    complete = processed == 32 and status == "finished_schedule"
    return {"schema_version": "a175-analysis-v1", "status": "complete" if complete else "incomplete", "execution_status": status,
            "posthoc": True, "planned_outputs": 32, "target_model_calls": 0,
            "coverage": {"attempted": attempted, "processed": processed, "unattempted": 32 - attempted, "missing": 32 - processed},
            "coverage_definitions": {"attempted": "parent_result_hash_read_attempts", "processed": "structurally_audited_outputs"},
            "summaries": summarize(records) if complete else None,
            "numeric_records_sha256": object_sha(records) if records is not None else None,
            "provenance": {"run_sha256": object_sha(run), "plan_sha256": run["plan_sha256"], "freeze_sha256": run["freeze_sha256"],
                           **plan["bindings"], "parent_artifact_sha256": plan["parent_bindings"]["artifact_sha256"]},
            "claim_boundaries": {"primary_scores_changed": False, "accuracy_rescue": False, "confirmatory": False,
                "internal_mechanism_identified": False, "detector_expansion": False, "complete_case_summary": False}}


class A175Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        super().__init__()


class A175Deadline(A175Interrupted):
    pass


def _interrupt(signum, _frame):
    raise A175Interrupted(signum)


def _deadline(signum, _frame):
    raise A175Deadline(signum)


@contextlib.contextmanager
def bounded_signals():
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "existing_timer")
    handlers = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    try:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, _interrupt)
        signal.signal(signal.SIGALRM, _deadline)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in handlers.items():
            signal.signal(number, handler)


def _output_root(path):
    require(path == AUDIT_ROOT / "execution" and path.is_absolute()
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_audit_output")
    return path


def run_plan(*, plan_path, plan_sha256, protocol_path, freeze_path, freeze_sha256, output_root):
    """One frozen two-phase audit; errors stop, with no retries or imputation."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    root = _output_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ("one-shot.json", "run.json", "outputs")), "consumed_audit")
        plan, _freeze = _audit_inputs(plan_path, plan_sha256, protocol_path, freeze_path, freeze_sha256)
        run = _run_header(plan, plan_sha256, freeze_sha256, os.getpid())
        _write_new(root / "one-shot.json", {"schema_version": "a175-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
        _PROCESS_CONSUMED = True
        _write_new(root / "run.json", run)
        started, status, interruption, attempted, processed = time.monotonic(), "failed", None, 0, 0
        trials, parent_records, records, error = None, None, None, None
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(bounded_signals())
                parent_lock = stack.enter_context((PARENT_ROOT / "execution/.lock").open("rb"))
                fcntl.flock(parent_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                trials, parent_records, header = _parent_metadata(plan)
                records = [_record(t, r, None) for t, r in zip(trials, parent_records)]

                def before_read(trial):
                    nonlocal attempted
                    _write_new(root / "outputs" / trial["trial_id"] / "attempt.json", _attempt(run, trial, plan))
                    attempted += 1

                raw = _result_bytes(plan, trials, before_read)
                for index, (trial, parent_record) in enumerate(zip(trials, parent_records)):
                    text, tokens = _parent_result(raw[trial["trial_id"]], trial, parent_record, header)
                    features = inspect_output(text, tokens, trial, header["eos_token_ids"])
                    _validate_features(features)
                    require(features["fence"] is parent_record["fence_marker"], "parent_fence_marker_reconciliation")
                    record = _record(trial, parent_record, features)
                    _write_new(root / "outputs" / trial["trial_id"] / "result.json", {"schema_version": "a175-result-v1",
                        "attempt_sha256": object_sha(_attempt(run, trial, plan)), "record": record})
                    records[index] = record
                    processed += 1
                status = "finished_schedule"
            except (A175Interrupted, KeyboardInterrupt) as exc:
                status = "deadline" if isinstance(exc, A175Deadline) else "interrupted"
                number = exc.signum if isinstance(exc, A175Interrupted) else None
                interruption = {"signal_number": number, "signal_name": signal.Signals(number).name if number is not None else None,
                                "pid": os.getpid(), "exception_type": type(exc).__name__}
                error = {"exception_type": type(exc).__name__, "code": "a175_audit_interrupted"}
            except Exception as exc:
                error = {"exception_type": type(exc).__name__, "code": "a175_audit_failed"}
            finally:
                # A signal can arrive between a durable link and the Python
                # counter update. Reconcile only saved numeric receipts here;
                # this does not interpret another parent response.
                if trials is not None:
                    attempted = sum((root / "outputs" / t["trial_id"] / "attempt.json").is_file() for t in trials)
                    processed = 0
                    for index, trial in enumerate(trials):
                        path = root / "outputs" / trial["trial_id"] / "result.json"
                        if path.is_file():
                            saved = _read(path)
                            records[index] = saved["record"]
                            processed += 1
                if error is not None:
                    _write_new(root / "error.json", error)
                _write_new(root / "execution-finished.json", {"schema_version": "a175-execution-finished-v1", "run_sha256": object_sha(run),
                    "status": status, "elapsed_seconds": time.monotonic() - started, "interruption": interruption,
                    "attempted": attempted, "processed": processed, "target_model_calls": 0})
        aggregate = _aggregate(plan, run, records, attempted, processed, status)
        if records is not None:
            _write_new(root / "records.json", records)
        _write_new(root / "analysis.json", aggregate)
        return aggregate


def export_run(*, plan_path, plan_sha256, protocol_path, freeze_path, freeze_sha256, output_root):
    """Replay only already-recorded features. Never fill an unaudited slot."""
    root = _output_root(output_root)
    with (root / ".lock").open("rb") as lock, (PARENT_ROOT / "execution/.lock").open("rb") as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, _freeze = _audit_inputs(plan_path, plan_sha256, protocol_path, freeze_path, freeze_sha256)
        run = _read(root / "run.json")
        require(type(run.get("pid")) is int and run["pid"] > 0
                and _same(run, _run_header(plan, plan_sha256, freeze_sha256, run["pid"])), "run_binding")
        require(_same(_read(root / "one-shot.json"), {"schema_version": "a175-one-shot-v1", "pid": run["pid"], "plan_sha256": plan_sha256}), "one_shot_binding")
        trials, parent_records, header = _parent_metadata(plan)
        raw = _result_bytes(plan, trials)
        folders = root / "outputs"
        if folders.exists():
            _plain(folders, directory=True)
            require(all(p.name in plan["parent_bindings"]["result_sha256"] and p.is_dir() and not p.is_symlink()
                        for p in folders.iterdir()), "unknown_audit_output")
        attempted, processed, attempt_gap, result_gap, records = 0, 0, False, False, []
        for trial, parent_record in zip(trials, parent_records):
            folder = folders / trial["trial_id"]
            features = None
            if not folder.exists():
                attempt_gap = True
            else:
                require(not attempt_gap and {p.name for p in folder.iterdir()} <= {"attempt.json", "result.json"}, "audit_attempt_sequence")
                attempt = _read(folder / "attempt.json")
                require(_same(attempt, _attempt(run, trial, plan)), "audit_attempt_binding")
                attempted += 1
                if not (folder / "result.json").exists():
                    result_gap = True
                else:
                    require(not result_gap, "audit_result_sequence")
                    text, tokens = _parent_result(raw[trial["trial_id"]], trial, parent_record, header)
                    features = inspect_output(text, tokens, trial, header["eos_token_ids"])
                    require(features["fence"] is parent_record["fence_marker"], "parent_fence_marker_reconciliation")
                    result = _read(folder / "result.json")
                    require(_same(result, {"schema_version": "a175-result-v1", "attempt_sha256": object_sha(attempt),
                            "record": _record(trial, parent_record, features)}), "audit_feature_replay")
                    processed += 1
            records.append(_record(trial, parent_record, features))
        require(processed == 0 or attempted == 32, "all_hashes_before_features")
        finished = _read(root / "execution-finished.json")
        require(type(finished) is dict and set(finished) == {"schema_version", "run_sha256", "status", "elapsed_seconds", "interruption", "attempted", "processed", "target_model_calls"}
                and finished["schema_version"] == "a175-execution-finished-v1" and finished["run_sha256"] == object_sha(run)
                and type(finished["attempted"]) is int and type(finished["processed"]) is int
                and finished["attempted"] == attempted and finished["processed"] == processed
                and type(finished["elapsed_seconds"]) in (float, int) and math.isfinite(finished["elapsed_seconds"]) and finished["elapsed_seconds"] >= 0
                and type(finished["target_model_calls"]) is int and finished["target_model_calls"] == 0
                and finished["status"] in ("finished_schedule", "failed", "interrupted", "deadline"), "finished_receipt")
        if finished["status"] == "finished_schedule":
            require(attempted == processed == 32 and finished["interruption"] is None, "finished_coverage")
        elif finished["status"] in ("interrupted", "deadline"):
            value = finished["interruption"]
            require(type(value) is dict and set(value) == {"signal_number", "signal_name", "pid", "exception_type"}
                    and type(value["pid"]) is int and value["pid"] == run["pid"] and value["exception_type"] in ("A175Interrupted", "A175Deadline", "KeyboardInterrupt"), "signal_receipt")
            allowed = {"A175Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP), "A175Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            number = value["signal_number"]
            require((number is None or type(number) is int) and number in allowed[value["exception_type"]] and value["signal_name"] == (signal.Signals(number).name if number is not None else None)
                    and (finished["status"] == "deadline") == (value["exception_type"] == "A175Deadline"), "actual_signal_receipt")
        else:
            require(finished["interruption"] is None, "failed_signal_receipt")
        # A guard failure before metadata validation has no feature rows. Keep
        # that original failure representation; do not manufacture an export.
        if not (root / "records.json").exists() and processed == attempted == 0:
            records = None
        aggregate = _aggregate(plan, run, records, attempted, processed, finished["status"])
        for name, value in (("records.json", records), ("analysis.json", aggregate)):
            if (root / name).exists():
                require(_same(_read(root / name), value), "stored_export_drift")
        return {"records": records, "aggregate": aggregate}


run = run_plan
export = export_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "export"), default="run")
    for name in ("plan", "protocol", "freeze", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "freeze-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    kwargs = {"plan_path": args.plan, "plan_sha256": args.plan_sha256, "protocol_path": args.protocol,
              "freeze_path": args.freeze, "freeze_sha256": args.freeze_sha256, "output_root": args.output_root}
    try:
        aggregate = run_plan(**kwargs) if args.mode == "run" else export_run(**kwargs)["aggregate"]
        # No row identifiers, hashes, matched text, tokens or raw exceptions.
        print(json.dumps({key: aggregate[key] for key in ("status", "execution_status", "coverage", "target_model_calls")}, sort_keys=True))
        return 0 if aggregate["status"] == "complete" else 1
    except (Exception, A175Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a175_failed", "error_type": type(exc).__name__, "target_model_calls": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
