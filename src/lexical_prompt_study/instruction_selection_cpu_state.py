"""A166 fixed 24-call CPU state diagnostic; no model calls at import.

Run in one fresh process under an external 1,800-second deadline. Private
receipt replay precedes every aggregate. No retry, held-out use or extension.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

from . import instruction_selection_cpu_development as parent

native, tasks = parent.native, parent.tasks
REFERENCE_SHA = parent.REFERENCE_SHA
MIN_AVAILABLE_BYTES = 48 * 1024**3
MAX_SECONDS = 1800
PHASES = ("pre", "inert", "post")
CATEGORIES = ("exact", "other_selector", "other_answer", "format")
REUSED_SOURCE_HASHES = {
    **parent.REUSED_SOURCE_HASHES,
    "__init__.py": "38df51bcc0f0179554da19316a656daaa4ba07dd998a704399b29e3085bce948",
    "instruction_selection_cpu_development.py": "631685329939f1af9a5cc82219bfd2324ef702a21709368b5ec07866dead175d",
    "instruction_selection_cpu_development_analysis.py": "d00aa2f48dd1c153fd0189d55d0d3087df73b131d08ba270c5bdc397d592e2de",
}
_PROCESS_CONSUMED = False


class A166Interrupted(BaseException):
    """SIGTERM bypasses the inherited per-trial Exception recovery."""


class A166Deadline(BaseException):
    """A process-wide backstop, including preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a166_" + code)


def object_sha(value):
    return tasks.sha(tasks.canonical(value))


def validate_sources():
    root = Path(__file__).parent
    for name, expected in REUSED_SOURCE_HASHES.items():
        require(native.engine.file_digest(root / name) == expected, "reused_source_drift")
    return {**REUSED_SOURCE_HASHES, Path(__file__).name: native.engine.file_digest(Path(__file__))}


def _validate_parent(plan, header):
    validate_sources()
    parent.validate_plan(plan)
    require(type(header) is dict and header.get("schema_version") == "a165-run-v1"
            and header.get("plan_sha256") == object_sha(plan)
            and tasks.is_hash(header.get("config_sha256")), "parent_run_binding")
    require(header.get("source_hashes") == parent.validate_sources()
            and header.get("cpu_reference_script_sha256") == REFERENCE_SHA
            and header.get("reference_protocol_sha256") == parent.REFERENCE_PROTOCOL_SHA
            and header.get("reference_amendment_sha256") == parent.REFERENCE_AMENDMENT_SHA
            and header.get("heldout_allowed") is False and header.get("retries") == 0
            and header.get("exploratory_development_only") is True, "parent_instrument_binding")
    settings = header.get("settings")
    require(type(settings) is dict and set(settings) == {"device", "dtype", "quantization",
            "attention_implementation", "cpu_threads", "seed", "max_new_tokens"}
            and {k: v for k, v in settings.items() if k != "seed"} == {
                "device": "cpu", "dtype": "float32", "quantization": None,
                "attention_implementation": "sdpa", "cpu_threads": 4, "max_new_tokens": 64}
            and type(settings["seed"]) is int, "parent_cpu_settings")


def _build_plan(a165_plan, a165_run_header, protocol_sha256):
    require(tasks.is_hash(protocol_sha256), "protocol_hash")
    _validate_parent(a165_plan, a165_run_header)
    base = a165_plan["base_plan"]
    worlds = list(dict.fromkeys(row["world_id"] for row in base["trials"]))[:4]
    selected = [row for row in base["trials"] if row["world_id"] in worlds]
    require(len(selected) == 8 and len(worlds) == 4, "fixed_selection")
    lookup = {(row["world_id"], row["selector"], row["scaffold_kind"], row["placement"]): row
              for row in a165_plan["trials"]}
    bindings = {"protocol_sha256": protocol_sha256, "a166_source_sha256": native.engine.file_digest(Path(__file__)),
                "a165_plan_sha256": object_sha(a165_plan), "a165_run_sha256": object_sha(a165_run_header),
                "config_sha256": a165_run_header["config_sha256"], "cpu_reference_script_sha256": REFERENCE_SHA}
    trials = []
    for index, original in enumerate(selected):
        for phase in PHASES:
            kind, placement = ("inert", "before") if phase == "inert" else ("none", "none")
            source = lookup[original["world_id"], original["selector"], kind, placement]
            row = copy.deepcopy(source)
            row.update(diagnostic_phase=phase, triplet_index=index, source_a165_trial_id=source["trial_id"])
            del row["trial_id"]
            row["trial_id"] = object_sha({"schema_version": "a166-trial-identity-v1", "bindings": bindings,
                                          "trial": row})[:24]
            trials.append(row)
    return {"schema_version": "a166-plan-v1", "partition": "development", "stage": "cpu_state_diagnostic",
            "world_count": 4, "triplet_count": 8, "trial_count": 24, "trials": trials,
            "a165_plan": copy.deepcopy(a165_plan), "a165_run_header": copy.deepcopy(a165_run_header),
            "bindings": bindings, "likelihood_collected": False, "heldout_allowed": False}


def compile_plan(a165_plan, *, a165_run_header, protocol_sha256):
    """Compile only first-four-world pre/inert-before/post triplets; read no outcomes."""
    return _build_plan(a165_plan, a165_run_header, protocol_sha256)


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    expected = _build_plan(plan["a165_plan"], plan["a165_run_header"], plan["bindings"]["protocol_sha256"])
    require(tasks.canonical(plan) == tasks.canonical(expected), "fixed_plan_drift")
    identifiers = [row["trial_id"] for row in plan["trials"]]
    require(len(set(identifiers)) == 24 and not set(identifiers) & {
        row["trial_id"] for row in plan["a165_plan"]["trials"]}, "new_trial_identities")


def _interrupt(_signum, _frame):
    raise A166Interrupted()


def _deadline(_signum, _frame):
    raise A166Deadline()


@contextlib.contextmanager
def bounded_signals():
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "existing_timer")
    handlers = {number: signal.getsignal(number) for number in
                (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
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


def _private_root(path):
    require(path.is_absolute() and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(not path.is_relative_to(Path(__file__).resolve().parents[2]), "private_output_outside_source")
    return path


def _read(path):
    require(path.is_file() and not path.is_symlink(), "receipt_file")
    raw = path.read_bytes()
    value = json.loads(raw)
    require(raw == tasks.canonical(value), "canonical_receipt_bytes")
    return value


def _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader):
    plan = native._read_bound(plan_path, plan_sha256)
    config = native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    require(config_sha256 == plan["bindings"]["config_sha256"], "same_parent_config_required")
    require(config["attention_implementation"] == "sdpa" and config["cpu_threads"] == 4
            and config["seed"] == plan["a165_run_header"]["settings"]["seed"], "same_cpu_settings")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    sources = validate_sources()
    tokenizer = tokenizer_loader(config)
    eos = native.pinned_eos_ids(config)
    require(eos == plan["a165_run_header"]["eos_token_ids"], "parent_eos_drift")
    prepared = {row["trial_id"]: native.prepare_generation(tokenizer, row, config) for row in plan["trials"]}
    for offset in range(0, 24, 3):
        pre, _, post = plan["trials"][offset:offset + 3]
        require(prepared[pre["trial_id"]] == prepared[post["trial_id"]], "baseline_native_identity")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    """Native-token preparation only, no model; returned prompts/tokens must stay private."""
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256,
        protocol_path, reference_script, tokenizer_loader)
    return {"schema_version": "a166-native-freeze-v1", "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_hashes": sources,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "planned_cells": 24, "native_prepared_sha256": object_sha(prepared),
            "eos_token_ids": eos, "native_inputs": prepared}


def _attempt(header, trial, prepared, index):
    return {"schema_version": "a166-attempt-v1", "run_sha256": object_sha(header),
            "trial_sha256": object_sha(trial), "prepared": prepared[trial["trial_id"]],
            "eos_token_ids": header["eos_token_ids"], "sequence_index": index}


def _validate_result(result, attempt, trial, tokenizer):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "record", "private",
            "elapsed_seconds"} and result["schema_version"] == "a166-result-v1"
            and result["record"].get("schema_version") == "a166-cell-v1", "result_schema")
    require(type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"])
            and result["elapsed_seconds"] >= 0, "result_elapsed")
    require(type(result["private"]) is dict and set(result["private"]) == {
        "generated_token_ids", "response_text", "error"}, "private_result_schema")
    record = result["record"]
    require(type(record.get("score")) is dict and set(record["score"]) == {"category", "strict_correct"},
            "strict_score_schema")
    if record.get("generation_status") == "completed":
        require(type(record["score"]["strict_correct"]) is bool, "strict_score_boolean")
    wire = {**result, "schema_version": "a164-result-v1",
            "record": {**result["record"], "schema_version": "a164-cell-v1"}}
    native.validate_result(wire, attempt, trial, object_sha(attempt), tokenizer)


def _arm(trials, results, attempted):
    rows = [results[t["trial_id"]]["record"] for t in trials if t["trial_id"] in results]
    complete = [r for r in rows if r["generation_status"] == "completed"]
    correct = sum(r["score"]["strict_correct"] for r in complete)
    return {"planned": len(trials), "attempted": sum(t["trial_id"] in attempted for t in trials),
            "recorded": len(rows), "completed": len(complete), "missing": len(trials) - len(rows),
            "infrastructure_failed": len(rows) - len(complete), "strict_correct": correct,
            "strict_failed": len(complete) - correct, "capped": sum(r["finish_reason"] == "length" for r in complete),
            "strict_accuracy_bounds_pp": [100 * correct / len(trials),
                                          100 * (correct + len(trials) - len(complete)) / len(trials)],
            "parser_categories": {category: sum(r["score"]["category"] == category for r in complete)
                                  for category in CATEGORIES}}


def _aggregate(plan, results, attempted):
    arms = {phase: _arm([t for t in plan["trials"] if t["diagnostic_phase"] == phase], results, attempted)
            for phase in PHASES}
    improved = deteriorated = unchanged = paired = token_identical = 0
    low = high = 0
    token_receipts = []
    for offset in range(0, 24, 3):
        pre, _, post = plan["trials"][offset:offset + 3]
        values, token_hashes = [], []
        for trial in (pre, post):
            result = results.get(trial["trial_id"])
            complete = result is not None and result["record"]["generation_status"] == "completed"
            values.append(int(result["record"]["score"]["strict_correct"]) if complete else None)
            token_hashes.append(object_sha(result["private"]["generated_token_ids"]) if complete else None)
        old, new = values
        low += (new if new is not None else 0) - (old if old is not None else 1)
        high += (new if new is not None else 1) - (old if old is not None else 0)
        if None not in values:
            paired += 1
            improved += new > old
            deteriorated += new < old
            unchanged += new == old
            token_identical += token_hashes[0] == token_hashes[1]
        token_receipts.append({"triplet_index": offset // 3, "pre_tokens_sha256": token_hashes[0],
                               "post_tokens_sha256": token_hashes[1]})
    total = _arm(plan["trials"], results, attempted)
    incomplete = total["completed"] != 24
    pre_failed = arms["pre"]["completed"] != 8 or arms["pre"]["strict_correct"] != 8
    reproduced = arms["inert"]["completed"] == 8 and arms["inert"]["strict_failed"] == 8
    reasons = []
    if incomplete:
        reasons.append("missing_or_infrastructure_failure")
    if pre_failed:
        reasons.append("pre_baseline_competence_not_established")
    if not reproduced:
        reasons.append("inert_failure_not_fully_reproduced")
    if paired == 8 and token_identical != 8:
        reasons.append("baseline_token_identity_not_reproduced")
    if deteriorated:
        decision = "investigate_state"
    elif not reasons and arms["post"]["strict_correct"] == 8:
        decision = "carries_less_persistent_state_concern"
    else:
        decision = "inconclusive"
    return {"schema_version": "a166-analysis-v1", "status": "incomplete" if incomplete else "complete",
            "planned_worlds": 4, "planned_triplets": 8, "planned_cells": 24, "coverage": total, "arms": arms,
            "unattempted_cells": 24 - len(attempted), "interrupted_attempts": len(attempted) - len(results),
            "pre_baseline_controls": {"planned": 8, "complete": arms["pre"]["completed"] == 8,
                "all_strict_correct": arms["pre"]["strict_correct"] == 8,
                "competence_established": not pre_failed, "execution_gate": False},
            "pre_post": {"planned_pairs": 8, "completed_pairs": paired, "unresolved_pairs": 8 - paired,
                "strict_improvement_pairs": improved, "strict_deterioration_pairs": deteriorated,
                "strict_unchanged_pairs": unchanged, "token_identical_pairs": token_identical,
                "strict_difference_post_minus_pre_pp": 100 * low / 8 if paired == 8 else None,
                "strict_difference_bounds_pp": [100 * low / 8, 100 * high / 8]},
            "decision": decision, "observed_deterioration_requires_investigation": deteriorated > 0,
            "inconclusive_reasons": reasons, "all_inert_strict_failures_reproduced": reproduced,
            "plan_object_sha256": object_sha(plan), "numeric_records_sha256": object_sha([
                results[t["trial_id"]]["record"] for t in plan["trials"] if t["trial_id"] in results]),
            "private_token_comparison_sha256": object_sha(token_receipts),
            "claim_boundaries": {"descriptive_posthoc_development_diagnostic": True, "heldout_allowed": False,
                "confidence_intervals": False, "population_inference": False, "state_mechanism_proven": False,
                "persistent_state_absence_proven": False, "quantization_only_causal_claim": False,
                "missing_outcomes_imputed": False, "raw_text_ids_or_token_hashes_exported": False,
                "automatic_extension": False}}, token_receipts


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256,
            protocol_path, reference_script):
    header = _read(root / "run.json")
    expected_fixed = {"schema_version": "a166-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "a165_plan_sha256": plan["bindings"]["a165_plan_sha256"],
        "a165_run_sha256": plan["bindings"]["a165_run_sha256"], "protocol_sha256": native.engine.file_digest(protocol_path),
        "reference_script_sha256": REFERENCE_SHA, "reference_script_path": str(reference_script.absolute()),
        "native_prepared_sha256": object_sha(prepared), "eos_token_ids": eos,
        "model_manifest_sha256": object_sha(config["model_files_sha256"]), "settings": {
            **plan["a165_run_header"]["settings"], "do_sample": False, "num_beams": 1},
        "planned_cells": 24, "maximum_seconds": MAX_SECONDS, "retries": 0, "heldout_allowed": False}
    require(set(header) == {*expected_fixed, "pid", "available_memory_bytes"}
            and all(header[k] == v for k, v in expected_fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0
            and type(header["available_memory_bytes"]) is int and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES,
            "run_lineage")
    require(_read(root / "one-shot.json") == {"schema_version": "a166-one-shot-v1", "pid": header["pid"],
            "plan_sha256": plan_sha256}, "one_shot_binding")
    require(_read(root / "native-inputs.json") == prepared, "native_input_replay")
    expected_ids = {t["trial_id"] for t in plan["trials"]}
    trial_root = root / "trials"
    if trial_root.exists():
        require(not trial_root.is_symlink() and trial_root.is_dir() and all(
            p.name in expected_ids and p.is_dir() and not p.is_symlink() for p in trial_root.iterdir()), "unknown_trial")
    results, attempted = {}, set()
    sequence_closed = False
    for index, trial in enumerate(plan["trials"]):
        folder = trial_root / trial["trial_id"]
        if not folder.exists():
            sequence_closed = True
            continue
        require({p.name for p in folder.iterdir()} <= {"attempt.json", "result.json"}, "unknown_trial_artifact")
        require(not sequence_closed and (folder / "attempt.json").is_file(), "attempt_sequence")
        attempt = _read(folder / "attempt.json")
        require(attempt == _attempt(header, trial, prepared, index), "attempt_native_replay")
        attempted.add(trial["trial_id"])
        if not (folder / "result.json").exists():
            sequence_closed = True
            continue
        result = _read(folder / "result.json")
        _validate_result(result, attempt, trial, tokenizer)
        results[trial["trial_id"]] = result
    records = [results[t["trial_id"]]["record"] for t in plan["trials"] if t["trial_id"] in results]
    aggregate, token_receipts = _aggregate(plan, results, attempted)
    aggregate["provenance"] = {"run_sha256": object_sha(header), "plan_sha256": plan_sha256,
        "config_sha256": config_sha256, "source_hashes": sources, "protocol_sha256": header["protocol_sha256"],
        "parent_run_sha256": header["a165_run_sha256"]}
    for name, expected in (("records.json", records), ("private-token-comparison.json", token_receipts),
                           ("analysis.json", aggregate)):
        if (root / name).exists():
            require(_read(root / name) == expected, "stored_export_drift")
    return {"records": records, "aggregate": aggregate, "private_token_comparison": token_receipts}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    """Re-score immutable private receipts without loading a model; no partial-row claims."""
    root = _private_root(output_root)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, config, sources, tokenizer, eos, prepared = _inputs(plan_path, plan_sha256, config_path,
            config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256,
                       protocol_path, reference_script)


def run_plan(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
             output_root, reference_loader=parent.load_reference, tokenizer_loader=native.load_tokenizer):
    """One fresh CPU model/process, exactly the fixed order, never a retry or resume."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    root = _private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with bounded_signals(), (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ("one-shot.json", "run.json", "trials")), "consumed_run")
        plan, config, sources, tokenizer, eos, prepared = _inputs(plan_path, plan_sha256, config_path,
            config_sha256, protocol_path, reference_script, tokenizer_loader)
        require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"],
                "original_model_manifest")
        reference = reference_loader(reference_script)
        available = reference.require_memory()
        require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
        _PROCESS_CONSUMED = True
        native.engine.write_private(root / "one-shot.json", {"schema_version": "a166-one-shot-v1",
            "pid": os.getpid(), "plan_sha256": plan_sha256})
        header = {"schema_version": "a166-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "a165_plan_sha256": plan["bindings"]["a165_plan_sha256"],
            "a165_run_sha256": plan["bindings"]["a165_run_sha256"], "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "reference_script_sha256": REFERENCE_SHA, "reference_script_path": str(reference_script.absolute()),
            "native_prepared_sha256": object_sha(prepared), "eos_token_ids": eos, "pid": os.getpid(),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "available_memory_bytes": available,
            "settings": {**plan["a165_run_header"]["settings"], "do_sample": False, "num_beams": 1},
            "planned_cells": 24, "maximum_seconds": MAX_SECONDS, "retries": 0, "heldout_allowed": False}
        native.engine.write_private(root / "run.json", header)
        native.engine.write_private(root / "native-inputs.json", prepared)
        started, status, runtime = time.monotonic(), "failed", None
        try:
            runtime = reference.load_cpu_reference(config, native.engine)
            require(runtime.eos_ids == eos and tasks.sha(runtime.tokenizer.get_chat_template().encode())
                    == config["chat_template_sha256"], "runtime_native_binding")
            for index, trial in enumerate(plan["trials"]):
                attempt = _attempt(header, trial, prepared, index)
                folder = root / "trials" / trial["trial_id"]
                native.engine.write_private(folder / "attempt.json", attempt)
                tick = time.monotonic()
                record, private = native.evaluate_trial(runtime, trial, attempt["prepared"])
                record["schema_version"] = "a166-cell-v1"
                result = {"schema_version": "a166-result-v1", "attempt_sha256": object_sha(attempt),
                          "record": record, "private": private, "elapsed_seconds": time.monotonic() - tick}
                _validate_result(result, attempt, trial, tokenizer)
                native.engine.write_private(folder / "result.json", result)
            status = "finished_schedule"
        except BaseException as exc:
            status = "deadline" if isinstance(exc, A166Deadline) else (
                "interrupted" if isinstance(exc, (A166Interrupted, KeyboardInterrupt)) else "failed")
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a166_execution_failed"))
        finally:
            runtime = None
            native.engine.write_private(root / "execution-finished.json", {"schema_version": "a166-execution-finished-v1",
                "status": status, "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header)})
        # All exported scores and token comparisons are reconstructed from disk.
        exported = _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256,
                           protocol_path, reference_script)
        native.engine.write_private(root / "records.json", exported["records"])
        native.engine.write_private(root / "private-token-comparison.json", exported["private_token_comparison"])
        native.engine.write_private(root / "analysis.json", exported["aggregate"])
        return exported["aggregate"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "export"), default="run")
    for name in ("plan", "config", "protocol", "reference-script", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "config-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    kwargs = {"plan_path": args.plan, "plan_sha256": args.plan_sha256, "config_path": args.config,
        "config_sha256": args.config_sha256, "protocol_path": args.protocol, "reference_script": args.reference_script,
        "output_root": args.output_root}
    try:
        root = _private_root(args.output_root)
        if args.mode == "run":
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (root / "execution.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    aggregate = run_plan(**kwargs)
        else:
            # Tokenizer libraries may write progress; export also stays private.
            with (root / "replay.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    aggregate = export_run(**kwargs)["aggregate"]
        print(json.dumps(aggregate, sort_keys=True))
        return 0 if aggregate["status"] == "complete" else 1
    except BaseException as exc:
        print(json.dumps({"status": "a166_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
