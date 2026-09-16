"""A167 fixed matched-count prose substitution diagnostic; no model call at import.

The sole prose passage is caller supplied and prospectively reviewed. This
instrument neither creates material nor reads previous study responses.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time

from . import instruction_selection_cpu_state as state

native, tasks, parent = state.native, state.tasks, state.parent
REFERENCE_SHA = state.REFERENCE_SHA
STATE_SHA = "1f51ed42e6a36ba27b34aeeff5810e579eed0055068a4b884639e49069a90f04"
MAX_SECONDS = 3600
PHASES = ("baseline", "inert", "prose")
ORDERS = ("inert_then_prose", "prose_then_inert")
MIN_AVAILABLE_BYTES = state.MIN_AVAILABLE_BYTES
object_sha = state.object_sha
_PROCESS_CONSUMED = False


class A167Interrupted(BaseException):
    """Bypass inherited per-trial Exception recovery."""


class A167Deadline(BaseException):
    """Bound preflight, loading and execution in the same process."""


def require(condition, code):
    tasks.require(condition, "a167_" + code)


def validate_sources():
    require(native.engine.file_digest(Path(state.__file__)) == STATE_SHA, "a166_source_drift")
    return {**state.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def validate_prose(prose):
    """Mechanical exclusions only; absence of imperatives requires root review."""
    require(type(prose) is str and prose.strip() and len(prose.encode()) <= 1_000_000, "prose_string")
    require(not any(symbol in prose.lower() for symbol in tasks.SYMBOLS), "prose_answer_symbol")
    require(not any(char in prose for char in '{}[]<>`"')
            and not re.search(r"(?im)^\s*(system|user|assistant|developer|tool)\s*:", prose)
            and not any(ord(char) < 32 and char not in "\n\t\r" for char in prose), "prose_structural_syntax")


def _build_plan(a165_plan, a165_run_header, prose, protocol_sha256):
    validate_sources()
    state._validate_parent(a165_plan, a165_run_header)
    validate_prose(prose)
    require(tasks.is_hash(protocol_sha256), "protocol_hash")
    base = a165_plan["base_plan"]
    worlds = list(dict.fromkeys(t["world_id"] for t in base["trials"]))[:4]
    selected = [t for t in base["trials"] if t["world_id"] in worlds]
    require(len(worlds) == 4 and len(selected) == 8, "fixed_selection")
    selectors = {world: [t["selector"] for t in selected if t["world_id"] == world] for world in worlds}
    require(all(len(s) == len(set(s)) == 2 for s in selectors.values()), "both_original_selectors")
    lookup = {(t["world_id"], t["selector"], t["scaffold_kind"], t["placement"]): t for t in a165_plan["trials"]}
    bindings = {"protocol_sha256": protocol_sha256, "a167_source_sha256": native.engine.file_digest(Path(__file__)),
        "a166_source_sha256": STATE_SHA, "a165_plan_sha256": object_sha(a165_plan),
        "a165_run_sha256": object_sha(a165_run_header), "config_sha256": a165_run_header["config_sha256"],
        "prose_material_sha256": tasks.sha(prose.encode()), "cpu_reference_script_sha256": REFERENCE_SHA}
    trials = []
    for original in selected:
        world_index = worlds.index(original["world_id"])
        selector_index = selectors[original["world_id"]].index(original["selector"])
        for placement_index, placement in enumerate(("before", "after")):
            triplet = len(trials) // 3
            order = ORDERS[(world_index + selector_index + placement_index) % 2]
            for phase in ("baseline", *order.split("_then_")):
                source = lookup[original["world_id"], original["selector"],
                                "none" if phase == "baseline" else "inert",
                                "none" if phase == "baseline" else placement]
                row = copy.deepcopy(source)
                row.update(diagnostic_phase=phase, triplet_index=triplet, paired_placement=placement,
                           material_order=order, world_index=world_index, selector_index=selector_index,
                           placement_index=placement_index, source_a165_trial_id=source["trial_id"])
                if phase == "prose":
                    row["scaffold_kind"] = "prose"
                    user = original["messages"][1]["content"]
                    row["messages"][1]["content"] = prose + "\n\n" + user if placement == "before" else user + "\n\n" + prose
                del row["trial_id"]
                row["trial_id"] = object_sha({"schema_version": "a167-trial-identity-v1", "bindings": bindings,
                                              "trial": row})[:24]
                trials.append(row)
    return {"schema_version": "a167-plan-v1", "partition": "development", "stage": "prose_substitution_diagnostic",
        "world_count": 4, "triplet_count": 16, "trial_count": 48, "trials": trials, "bindings": bindings,
        "a165_plan": copy.deepcopy(a165_plan), "a165_run_header": copy.deepcopy(a165_run_header),
        "prose_material": prose, "prose_material_receipt": {"sha256": tasks.sha(prose.encode()), "bytes": len(prose.encode())},
        "likelihood_collected": False, "heldout_allowed": False}


def compile_plan(a165_plan, a165_run_header, prose, protocol_sha256):
    """Compile one supplied passage, fixed worlds/order and no outcome-based choices."""
    return _build_plan(a165_plan, a165_run_header, prose, protocol_sha256)


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    expected = _build_plan(plan["a165_plan"], plan["a165_run_header"], plan["prose_material"],
                           plan["bindings"]["protocol_sha256"])
    require(tasks.canonical(plan) == tasks.canonical(expected), "fixed_plan_drift")
    identities = [t["trial_id"] for t in plan["trials"]]
    require(len(set(identities)) == 48 and not set(identities) & {t["trial_id"] for t in plan["a165_plan"]["trials"]},
            "new_trial_identities")


def _interrupt(_signum, _frame):
    raise A167Interrupted()


def _deadline(_signum, _frame):
    raise A167Deadline()


@contextlib.contextmanager
def bounded_signals():
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "existing_timer")
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
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
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(not path.is_relative_to(Path(__file__).resolve().parents[2]), "private_output_outside_source")
    return path


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
    prepared = {t["trial_id"]: native.prepare_generation(tokenizer, t, config) for t in plan["trials"]}
    originals = {t["trial_id"]: t for t in plan["a165_plan"]["trials"]}
    matched = []
    for offset in range(0, 48, 3):
        group = {t["diagnostic_phase"]: t for t in plan["trials"][offset:offset + 3]}
        for phase in ("baseline", "inert"):
            trial = group[phase]
            original = originals[trial["source_a165_trial_id"]]
            require(trial["messages"] == original["messages"] and prepared[trial["trial_id"]]
                    == native.prepare_generation(tokenizer, original, config), "original_native_identity")
        counts = {phase: len(prepared[t["trial_id"]]["prompt_token_ids"]) for phase, t in group.items()}
        require(counts["prose"] == counts["inert"], "exact_material_prompt_token_match")
        matched.append({"triplet_index": offset // 3, "paired_placement": group["baseline"]["paired_placement"],
                        "prompt_token_counts": counts})
    return plan, config, sources, tokenizer, eos, prepared, matched


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    """Token-only preflight, no model; all returned message/token arrays remain private."""
    plan, _, sources, _, eos, prepared, matched = _inputs(plan_path, plan_sha256, config_path, config_sha256,
        protocol_path, reference_script, tokenizer_loader)
    return {"schema_version": "a167-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "reference_script_sha256": REFERENCE_SHA, "prose_material_receipt": plan["prose_material_receipt"],
        "planned_cells": 48, "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared,
        "eos_token_ids": eos, "matched_count_receipts": matched, "matched_counts_sha256": object_sha(matched)}


def _attempt(header, trial, prepared, index):
    return {"schema_version": "a167-attempt-v1", "run_sha256": object_sha(header),
        "trial_sha256": object_sha(trial), "prepared": prepared[trial["trial_id"]],
        "eos_token_ids": header["eos_token_ids"], "sequence_index": index}


def _validate_result(result, attempt, trial, tokenizer):
    require(type(result) is dict and result.get("schema_version") == "a167-result-v1"
            and type(result.get("record")) is dict and result["record"].get("schema_version") == "a167-cell-v1",
            "result_schema")
    # Explicit adapter; durable receipts always retain their distinct A167 schemas.
    wire = {**result, "schema_version": "a166-result-v1",
            "record": {**result["record"], "schema_version": "a166-cell-v1"}}
    state._validate_result(wire, attempt, trial, tokenizer)


def _binary(trial, results, outcome):
    result = results.get(trial["trial_id"])
    if result is None or result["record"]["generation_status"] != "completed":
        return 0, 1
    row = result["record"]
    value = (row["score"]["strict_correct"] if outcome == "strict_correct" else
             row["finish_reason"] == "length" if outcome == "capped" else row["score"]["category"] == "format")
    return int(value), int(value)


def _contrast(plan, results, *, outcome="strict_correct", placement=None, order=None):
    by_world = {}
    pairs = identified_pairs = 0
    for offset in range(0, 48, 3):
        group = {t["diagnostic_phase"]: t for t in plan["trials"][offset:offset + 3]}
        meta = group["baseline"]
        if placement is not None and meta["paired_placement"] != placement or order is not None and meta["material_order"] != order:
            continue
        plus, minus = _binary(group["prose"], results, outcome), _binary(group["inert"], results, outcome)
        bound = plus[0] - minus[1], plus[1] - minus[0]
        by_world.setdefault(meta["world_index"], []).append(bound)
        pairs += 1
        identified_pairs += bound[0] == bound[1]
    require(len(by_world) == 4 and len({len(values) for values in by_world.values()}) == 1, "contrast_world_balance")
    means = [(sum(v[0] for v in values) / len(values), sum(v[1] for v in values) / len(values))
             for values in by_world.values()]
    lower, upper = sum(v[0] for v in means) / 4, sum(v[1] for v in means) / 4
    return {"planned_worlds": 4, "planned_material_pairs": pairs, "identified_material_pairs": identified_pairs,
        "identified_worlds": sum(v[0] == v[1] for v in means), "outcome": outcome,
        "direction": "prose_minus_inert", "world_mean_difference_pp": 100 * lower if lower == upper else None,
        "worst_best_bounds_pp": [100 * lower, 100 * upper], "worlds_are_fixed_descriptive_units": True}


def _aggregate(plan, results, attempted):
    arms = {phase: state._arm([t for t in plan["trials"] if t["diagnostic_phase"] == phase], results, attempted)
            for phase in PHASES}
    coverage = state._arm(plan["trials"], results, attempted)
    complete = coverage["completed"] == 48
    baseline_ok = arms["baseline"]["completed"] == 16 and arms["baseline"]["strict_correct"] == 16
    inert, prose = arms["inert"]["strict_correct"], arms["prose"]["strict_correct"]
    pattern = ("descriptive_prose_substitution_recovery" if inert <= 1 and prose >= 15 else
               "both_materials_floor" if inert <= 1 and prose <= 1 else "mixed") if complete else None
    decision = pattern if complete and baseline_ok else "inconclusive"
    return {"schema_version": "a167-analysis-v1", "status": "complete" if complete else "incomplete",
        "planned_worlds": 4, "planned_triplets": 16, "planned_cells": 48, "coverage": coverage, "arms": arms,
        "unattempted_cells": 48 - len(attempted), "interrupted_attempts": len(attempted) - len(results),
        "baseline_controls": {"planned": 16, "complete": arms["baseline"]["completed"] == 16,
            "all_strict_correct": arms["baseline"]["strict_correct"] == 16,
            "competence_established": baseline_ok, "execution_gate": False},
        "primary_prose_minus_inert": _contrast(plan, results),
        "secondary_placement_contrasts": {p: _contrast(plan, results, placement=p) for p in ("before", "after")},
        "secondary_order_contrasts": {order: _contrast(plan, results, order=order) for order in ORDERS},
        "secondary_format_and_cap_contrasts": {outcome: _contrast(plan, results, outcome=outcome) for outcome in ("format", "capped")},
        "by_placement": {p: {phase: state._arm([t for t in plan["trials"] if t["paired_placement"] == p
            and t["diagnostic_phase"] == phase], results, attempted) for phase in PHASES} for p in ("before", "after")},
        "by_material_order": {order: {phase: state._arm([t for t in plan["trials"] if t["material_order"] == order
            and t["diagnostic_phase"] == phase], results, attempted) for phase in PHASES} for order in ORDERS},
        "decision": decision, "descriptive_material_pattern": pattern,
        "interpretation_guards": {"complete_scheduled_coverage": complete, "all_baselines_strict_correct": baseline_ok,
            "recovery_inert_correct_maximum": 1, "recovery_prose_correct_minimum": 15,
            "floor_correct_maximum_each_material": 1, "thresholds_are_statistical_tests": False},
        "plan_object_sha256": object_sha(plan), "numeric_records_sha256": object_sha([
            results[t["trial_id"]]["record"] for t in plan["trials"] if t["trial_id"] in results]),
        "counting_note": "Format is the parser category including capped responses, so format and cap counts can overlap.",
        "estimand": "This single supplied prose passage versus this original inert material at exactly matched native prompt token count.",
        "claim_boundaries": {"descriptive_posthoc_development_diagnostic": True, "heldout_allowed": False,
            "confidence_intervals": False, "population_inference": False, "world_independence_assumed": False,
            "semantics_alone_isolated": False, "structure_alone_isolated": False, "length_causality_identified": False,
            "material_order_eliminates_all_carryover": False, "quantization_only_causal_claim": False,
            "missing_outcomes_imputed": False, "raw_text_ids_or_per_item_token_hashes_exported": False,
            "automatic_calls_beyond_fixed_schedule": False}}


def _fixed_header(plan, config, sources, eos, prepared, matched, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a167-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "a165_plan_sha256": plan["bindings"]["a165_plan_sha256"],
        "a165_run_sha256": plan["bindings"]["a165_run_sha256"], "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "prose_material_receipt": plan["prose_material_receipt"], "reference_script_sha256": REFERENCE_SHA,
        "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
        "matched_counts_sha256": object_sha(matched), "eos_token_ids": eos,
        "model_manifest_sha256": object_sha(config["model_files_sha256"]), "settings": {
            **plan["a165_run_header"]["settings"], "do_sample": False, "num_beams": 1},
        "planned_cells": 48, "maximum_seconds": MAX_SECONDS, "retries": 0, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, matched, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, matched, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(header[k] == v for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(state._read(root / "one-shot.json") == {"schema_version": "a167-one-shot-v1", "pid": header["pid"],
        "plan_sha256": plan_sha256}, "one_shot_binding")
    require(state._read(root / "native-inputs.json") == prepared and state._read(root / "matched-counts.json") == matched,
            "native_input_replay")
    expected = {t["trial_id"] for t in plan["trials"]}
    trial_root = root / "trials"
    if trial_root.exists():
        require(not trial_root.is_symlink() and trial_root.is_dir() and all(p.name in expected and p.is_dir()
                and not p.is_symlink() for p in trial_root.iterdir()), "unknown_trial")
    results, attempted = {}, set()
    closed = False
    for index, trial in enumerate(plan["trials"]):
        folder = trial_root / trial["trial_id"]
        if not folder.exists():
            closed = True
            continue
        require({p.name for p in folder.iterdir()} <= {"attempt.json", "result.json"}, "unknown_trial_artifact")
        require(not closed and (folder / "attempt.json").is_file(), "attempt_sequence")
        attempt = state._read(folder / "attempt.json")
        require(attempt == _attempt(header, trial, prepared, index), "attempt_native_replay")
        attempted.add(trial["trial_id"])
        if not (folder / "result.json").exists():
            closed = True
            continue
        result = state._read(folder / "result.json")
        _validate_result(result, attempt, trial, tokenizer)
        results[trial["trial_id"]] = result
    records = [results[t["trial_id"]]["record"] for t in plan["trials"] if t["trial_id"] in results]
    aggregate = _aggregate(plan, results, attempted)
    aggregate["provenance"] = {"run_sha256": object_sha(header), "plan_sha256": plan_sha256,
        "config_sha256": config_sha256, "source_hashes": sources, "protocol_sha256": header["protocol_sha256"],
        "parent_run_sha256": header["a165_run_sha256"], "prose_material_receipt": header["prose_material_receipt"],
        "native_prepared_sha256": header["native_prepared_sha256"], "matched_counts_sha256": header["matched_counts_sha256"]}
    for name, expected_value in (("records.json", records), ("analysis.json", aggregate)):
        if (root / name).exists():
            require(state._read(root / name) == expected_value, "stored_export_drift")
    return {"records": records, "aggregate": aggregate}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    """Native/decoded-token/scorer replay only; never load the model."""
    root = _private_root(output_root)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, *inputs, plan_sha256, config_sha256, reference_script)


def run_plan(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
             output_root, reference_loader=parent.load_reference, tokenizer_loader=native.load_tokenizer):
    """Exactly one prospective 48-call schedule, with no retries or adaptive extension."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    root = _private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with bounded_signals(), (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ("one-shot.json", "run.json", "trials")), "consumed_run")
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        plan, config, sources, tokenizer, eos, prepared, matched = inputs
        require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"], "original_model_manifest")
        reference = reference_loader(reference_script)
        available = reference.require_memory()
        require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
        _PROCESS_CONSUMED = True
        native.engine.write_private(root / "one-shot.json", {"schema_version": "a167-one-shot-v1",
            "pid": os.getpid(), "plan_sha256": plan_sha256})
        header = {**_fixed_header(plan, config, sources, eos, prepared, matched, plan_sha256, config_sha256, reference_script),
                  "pid": os.getpid(), "available_memory_bytes": available}
        native.engine.write_private(root / "run.json", header)
        native.engine.write_private(root / "native-inputs.json", prepared)
        native.engine.write_private(root / "matched-counts.json", matched)
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
                record["schema_version"] = "a167-cell-v1"
                result = {"schema_version": "a167-result-v1", "attempt_sha256": object_sha(attempt), "record": record,
                          "private": private, "elapsed_seconds": time.monotonic() - tick}
                _validate_result(result, attempt, trial, tokenizer)
                native.engine.write_private(folder / "result.json", result)
            status = "finished_schedule"
        except BaseException as exc:
            status = "deadline" if isinstance(exc, A167Deadline) else (
                "interrupted" if isinstance(exc, (A167Interrupted, KeyboardInterrupt)) else "failed")
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a167_execution_failed"))
        finally:
            runtime = None
            native.engine.write_private(root / "execution-finished.json", {"schema_version": "a167-execution-finished-v1",
                "status": status, "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header)})
        exported = _replay(root, *inputs, plan_sha256, config_sha256, reference_script)
        native.engine.write_private(root / "records.json", exported["records"])
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
            with (root / "replay.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    aggregate = export_run(**kwargs)["aggregate"]
        print(json.dumps(aggregate, sort_keys=True))
        return 0 if aggregate["status"] == "complete" else 1
    except BaseException as exc:
        print(json.dumps({"status": "a167_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
