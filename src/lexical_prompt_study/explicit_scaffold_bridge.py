"""A178 explicit-instruction scaffold bridge; no model calls at import.

All 56 cells, one primary contrast and five secondary contrasts
are fixed before outcomes. Raw generation evidence remains private; stdout is coverage only.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
from itertools import product
import math
import os
from pathlib import Path
import signal
import time

from . import mapped_value_rewrite_path as previous

state, native, tasks, parent = previous.state, previous.native, previous.tasks, previous.parent
object_sha = state.object_sha
A177_SHA = "7a10f0e4c361f35b0c56b92c6921bd89ef7cec2f7b8deefdb99912dacaeb394f"
CONFIG_SHA = "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745"
REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
PRIVATE_RUNS_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a178")
MAX_SECONDS, KILL_GRACE_SECONDS = 7200, 60
MAX_NEW_TOKENS = 64
MIN_AVAILABLE_BYTES = 48 * 1024**3
SELECTORS = ("A", "B")
CATEGORIES = previous.CATEGORIES
WORLDS = (("s84", "s91", "AB"), ("s88", "s85", "BA"),
          ("s87", "s90", "BA"), ("s89", "s86", "AB"))
CONDITIONS = ("clean", "full_before", "full_after", "sham_before", "sham_after", "inert_before", "inert_after")
CONDITION_METADATA = {"clean": ("none", "none"), **{
    f"{kind}_{placement}": (kind, placement) for kind in ("full", "sham", "inert") for placement in ("before", "after")}}
BASE_SCHEDULE = tuple(product(CONDITIONS, SELECTORS))
MATERIAL_SOURCE_SHA = "ee899a47ceca1b53e90787c5279174412e8d01e5872db9bba44a486921eba345"
MATERIAL_RECEIPTS = {
    "full": {"sha256": "150cfb4e3b6fffab221543a8434a3b093859f28f44ee11510c169cafeb6822f6", "bytes": 901},
    "sham": {"sha256": "3f3819e8b468a35cc62e60e19ba2a4fbd6d30432479d2e6f645ce7673e355cc9", "bytes": 1019},
    "inert": {"sha256": "402a805911f0ad6e6708ca7447bc47c2149183433cbde15df62310cbacb021ed", "bytes": 1275},
}
prepare_trial = previous.prepare_trial
generate_tokens = previous.generate_tokens
score_tokens = previous.score_tokens
_validate_tokens = previous._validate_tokens

_PROCESS_CONSUMED = False


class A178Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        self.signal_name = signal.Signals(signum).name
        super().__init__()


class A178Deadline(A178Interrupted):
    """Internal deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a178_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    require(native.engine.file_digest(Path(previous.__file__)) == A177_SHA, "a177_helper_source_drift")
    require(parent.REFERENCE_SHA == REFERENCE_SHA, "reference_binding")
    return {**previous.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def system_instruction(selector):
    require(selector in SELECTORS, "system_selector")
    return previous.system_instruction("of_explicit_label", selector)


def _validate_materials(materials):
    require(type(materials) is dict and set(materials) == {"full", "sham", "inert"}, "material_keys")
    for kind, receipt in MATERIAL_RECEIPTS.items():
        value = materials[kind]
        require(type(value) is str, "material_type")
        raw = value.encode("utf-8")
        require(len(raw) == receipt["bytes"] and tasks.sha(raw) == receipt["sha256"], "material_bytes_binding")


def _pair_definition(positive, negative, count):
    return {"positive_conditions": list(positive), "negative_conditions": list(negative),
            "planned_outcomes": 2 * count, "planned_pairs": count, "cell_coefficient_magnitude": 1 / count}


def compile_plan(config_sha256, protocol_sha256, tests_sha256, materials):
    """Compile supplied, byte-bound private materials; never read a parent data file."""
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256) and tasks.is_hash(tests_sha256), "plan_bindings")
    _validate_materials(materials)
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256,
                "source_sha256": sources[Path(__file__).name], "reference_script_sha256": REFERENCE_SHA,
                "a177_helper_sha256": A177_SHA, "material_source_sha256": MATERIAL_SOURCE_SHA,
                "materials_object_sha256": object_sha(materials)}
    trials = []
    for world_index, (value_a, value_b, presentation) in enumerate(WORLDS):
        rotation = 7 * (world_index // 2)
        schedule = BASE_SCHEDULE[rotation:] + BASE_SCHEDULE[:rotation]
        if world_index % 2:
            schedule = tuple(reversed(schedule))
        mapping = {"A": value_a, "B": value_b}
        payload = json.dumps({key: mapping[key] for key in presentation}, separators=(",", ":"))
        for condition, selector in schedule:
            kind, placement = CONDITION_METADATA[condition]
            if placement == "none":
                user = payload
            elif placement == "before":
                user = materials[kind] + "\n\n" + payload
            else:
                user = payload + "\n\n" + materials[kind]
            trial = {"world_index": world_index, "world_id": f"w{world_index:02d}", "presentation_order": presentation,
                     "condition": condition, "scaffold_kind": kind, "placement": placement, "selector": selector,
                     "selected_answer": mapping[selector], "unselected_answer": mapping["B" if selector == "A" else "A"],
                     "sequence_index": len(trials),
                     "messages": [{"role": "system", "content": system_instruction(selector)}, {"role": "user", "content": user}]}
            trial["trial_id"] = object_sha({"schema_version": "a178-cell-identity-v1", "bindings": bindings, "trial": trial})[:24]
            trials.append(trial)
    require(len(trials) == 56 and len({t["trial_id"] for t in trials}) == 56, "fixed_matrix")
    secondary = {"inert_minus_clean": {
        "positive_conditions": ["inert_before", "inert_after"], "negative_conditions": ["clean"],
        "planned_outcomes": 24, "planned_triples": 8, "inert_coefficient": 1 / 16, "clean_coefficient": -1 / 8}}
    for placement in ("before", "after"):
        secondary["full_minus_sham_" + placement] = _pair_definition(("full_" + placement,), ("sham_" + placement,), 8)
        secondary["inert_minus_clean_" + placement] = _pair_definition(("inert_" + placement,), ("clean",), 8)
    return {"schema_version": "a178-plan-v1", "partition": "fresh_public_worlds_with_frozen_private_materials", "bindings": bindings,
            "materials": dict(materials), "trials": trials, "world_count": 4, "planned_cells": 56, "maximum_generation_calls": 56,
            "max_new_tokens": MAX_NEW_TOKENS, "heldout_allowed": False, "retries": 0,
            "resume_allowed": False, "baseline_gate": False, "likelihood_collected": False,
            "prose_allowed": False, "clarification_reminder_allowed": False, "mapping_heading_allowed": False,
            "study_prefix_allowed": False, "conditions": list(CONDITIONS),
            "material_receipts": {kind: dict(receipt) for kind, receipt in MATERIAL_RECEIPTS.items()},
            "primary_contrast_definitions": {"full_minus_sham": _pair_definition(
                ("full_before", "full_after"), ("sham_before", "sham_after"), 16)},
            "secondary_contrast_definitions": secondary,
            "fence_marker_definition": "literal_ascii_triple_backtick_or_triple_tilde_anywhere"}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    bindings = plan["bindings"]
    require(_same(plan, compile_plan(bindings["config_sha256"], bindings["protocol_sha256"], bindings["tests_sha256"], plan["materials"])), "fixed_plan_drift")


def _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader):
    plan, config = native._read_bound(plan_path, plan_sha256), native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    require(config_sha256 == CONFIG_SHA == plan["bindings"]["config_sha256"] and config["cpu_threads"] == 4
            and config["attention_implementation"] == "sdpa" and config["max_new_tokens"] == MAX_NEW_TOKENS, "original_config")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    model_config_path = Path(config["model_path"]) / "config.json"
    require(native.engine.file_digest(model_config_path) == config["model_files_sha256"]["config.json"], "model_config_binding")
    model_config = json.loads(model_config_path.read_bytes())
    context_limit = model_config.get("max_position_embeddings")
    vocab_size = model_config.get("vocab_size")
    require(type(context_limit) is int and context_limit > 0, "model_context_limit")
    require(type(vocab_size) is int and vocab_size > 0, "model_vocabulary")
    sources, tokenizer, eos = validate_sources(), tokenizer_loader(config), native.pinned_eos_ids(config)
    require(all(e < vocab_size for e in eos), "model_eos_vocabulary")
    prepared = {t["trial_id"]: prepare_trial(tokenizer, t, config, eos) for t in plan["trials"]}
    for value in prepared.values():
        require(len(value["prompt_token_ids"]) + MAX_NEW_TOKENS <= context_limit, "generation_context_fit")
        require(all(t < vocab_size for t in value["prompt_token_ids"] + value["assistant_probe_token_ids"]), "native_token_vocabulary")
        value["model_context_limit"] = context_limit
        value["model_vocab_size"] = vocab_size
    require(len({object_sha(p["prompt_token_ids"]) for p in prepared.values()}) == 56, "unique_native_cells")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path,
                                                reference_script, tokenizer_loader)
    return {"schema_version": "a178-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "tests_sha256": plan["bindings"]["tests_sha256"],
            "source_hashes": sources, "reference_script_sha256": REFERENCE_SHA, "planned_cells": 56,
            "maximum_generation_calls": 56, "eos_token_ids": eos,
            "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _private_root(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(path.is_relative_to(PRIVATE_RUNS_ROOT) and path != PRIVATE_RUNS_ROOT, "a178_private_run_root")
    require(not path.is_relative_to(Path(__file__).absolute().parents[2]), "private_outside_source")
    return path


def _attempt(header, trial, prepared):
    return {"schema_version": "a178-attempt-v1", "run_sha256": object_sha(header), "trial_sha256": object_sha(trial),
            "trial_id": trial["trial_id"], "sequence_index": trial["sequence_index"], "prepared": prepared[trial["trial_id"]]}


def _startup_attempt(pid, plan_sha256, config_sha256):
    return {"schema_version": "a178-startup-attempt-v1", "pid": pid, "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_sha256": native.engine.file_digest(Path(__file__))}


def _interruption(exc):
    return {"signal_number": exc.signum if isinstance(exc, A178Interrupted) else None,
            "signal_name": exc.signal_name if isinstance(exc, A178Interrupted) else None,
            "pid": exc.pid if isinstance(exc, A178Interrupted) else os.getpid(), "exception_type": type(exc).__name__}


def _validate_result(result, attempt, trial, tokenizer, eos):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "generated_token_ids",
            "score", "error", "elapsed_seconds"} and result["schema_version"] == "a178-result-v1"
            and result["attempt_sha256"] == object_sha(attempt) and type(result["elapsed_seconds"]) in (int, float)
            and math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0, "result_binding")
    if result["status"] == "completed":
        _validate_tokens(result["generated_token_ids"])
        require(all(t < attempt["prepared"]["model_vocab_size"] for t in result["generated_token_ids"]), "result_token_vocabulary")
        require(result["error"] is None and _same(result["score"], score_tokens(tokenizer, result["generated_token_ids"], trial, eos)),
                "result_score_replay")
    else:
        require(result["status"] == "infrastructure_failed" and result["generated_token_ids"] is None and result["score"] is None
                and type(result["error"]) is dict and set(result["error"]) == {"exception_type", "frames", "code"}, "failed_result_schema")


def _record(trial, result, attempted):
    metadata = {key: trial[key] for key in ("trial_id", "sequence_index", "world_index", "world_id", "presentation_order",
                                           "condition", "scaffold_kind", "placement", "selector")}
    if result is not None and result["status"] == "completed":
        score = {key: value for key, value in result["score"].items() if key != "response_text"}
    else:
        infrastructure = result is not None
        score = {"category": "infrastructure" if infrastructure else "missing", "strict_correct": None, "selected_label": None,
                 "format_valid": None, "eos_valid": None, "capped": None, "fence_marker": None, "attempted": attempted,
                 "reason": "recoverable_evaluation_failure" if infrastructure else "interrupted_attempt" if attempted else "unattempted",
                 "generated_token_count": None}
    return {"schema_version": "a178-cell-v1", **metadata, **score}


def _signed_bounds(terms):
    known = [(value, coefficient) for value, coefficient in terms if value is not None]
    fixed = math.fsum(coefficient * int(value) for value, coefficient in known)
    unresolved = [coefficient for value, coefficient in terms if value is None]
    return {"point": fixed if not unresolved else None,
            "lower": math.fsum([fixed, *(min(0., coefficient) for coefficient in unresolved)]),
            "upper": math.fsum([fixed, *(max(0., coefficient) for coefficient in unresolved)]),
            "planned_outcomes": len(terms), "resolved_outcomes": len(known)}


def _paired_bound(records, positive, negative, planned_pairs, pair_fields):
    """Fixed signed paired mean over the required condition cohort only."""
    positive, negative = set(positive), set(negative)
    require(positive and negative and not positive.intersection(negative)
            and positive.union(negative) <= set(CONDITIONS), "comparison_conditions")
    selected = [row for row in records if row["condition"] in positive.union(negative)]
    require(len(selected) == 2 * planned_pairs, "comparison_cohort")
    pairs, terms = {}, []
    for row in selected:
        value = row["strict_correct"]
        require(value is None or type(value) is bool, "comparison_outcome_schema")
        sign = 1 if row["condition"] in positive else -1
        values = pairs.setdefault(tuple(row[key] for key in pair_fields), {})
        require(sign not in values, "comparison_duplicate_cell")
        values[sign] = value
        terms.append((value, sign / planned_pairs))
    require(len(pairs) == planned_pairs and all(set(values) == {-1, 1} for values in pairs.values()), "comparison_pair_cells")
    return {**_signed_bounds(terms), "planned_pairs": planned_pairs,
            "resolved_pairs": sum(all(value is not None for value in values.values()) for values in pairs.values())}


def _pair_bound(records, positive, negative):
    return _paired_bound(records, (positive,), (negative,), 8, ("world_index", "selector"))


def _full_minus_sham_bound(records):
    return _paired_bound(records, ("full_before", "full_after"), ("sham_before", "sham_after"),
                         16, ("world_index", "selector", "placement"))


def _inert_minus_clean_bound(records):
    """Combine each shared clean coefficient before calculating sharp bounds."""
    required = {"clean", "inert_before", "inert_after"}
    selected = [row for row in records if row["condition"] in required]
    require(len(selected) == 24, "inert_comparison_cohort")
    triples, terms = {}, []
    for row in selected:
        value = row["strict_correct"]
        require(value is None or type(value) is bool, "inert_comparison_outcome_schema")
        values = triples.setdefault((row["world_index"], row["selector"]), {})
        require(row["condition"] not in values, "inert_comparison_duplicate_cell")
        values[row["condition"]] = value
        terms.append((value, -1 / 8 if row["condition"] == "clean" else 1 / 16))
    require(len(triples) == 8 and all(set(values) == required for values in triples.values()), "inert_comparison_triple_cells")
    return {**_signed_bounds(terms), "planned_triples": 8,
            "resolved_triples": sum(all(value is not None for value in values.values()) for values in triples.values())}


def _contrasts(records):
    return {"full_minus_sham": _full_minus_sham_bound(records)}


def _secondary_contrasts(records):
    values = {"inert_minus_clean": _inert_minus_clean_bound(records)}
    for placement in ("before", "after"):
        values["full_minus_sham_" + placement] = _pair_bound(records, "full_" + placement, "sham_" + placement)
        values["inert_minus_clean_" + placement] = _pair_bound(records, "inert_" + placement, "clean")
    return values


def _counts(records):
    known = [r for r in records if r["strict_correct"] is not None]
    successes = sum(r["strict_correct"] is True for r in records)
    unknown = len(records) - len(known)
    return {"planned_cells": len(records), "resolved_cells": len(known),
            "categories": {category: sum(r["category"] == category for r in records) for category in CATEGORIES},
            "strict_accuracy": {"point": successes / len(records) if not unknown else None,
                "lower": successes / len(records), "upper": (successes + unknown) / len(records),
                "planned_outcomes": len(records), "resolved_outcomes": len(known)},
            "indicators": {key: {"true": sum(r[key] is True for r in records), "false": sum(r[key] is False for r in records),
                                  "missing": sum(r[key] is None for r in records)} for key in ("format_valid", "eos_valid", "capped", "fence_marker")}}


def _aggregate(plan, records, results, attempted):
    completed = sum(r["status"] == "completed" for r in results.values())
    return {"schema_version": "a178-analysis-v1", "status": "complete" if completed == 56 else "incomplete",
            "planned_worlds": 4, "planned_cells": 56, "maximum_generation_calls": 56,
            "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                         "interrupted": len(attempted) - len(results), "unattempted": 56 - len(attempted),
                         "missing": 56 - len(results)},
            "overall": _counts(records),
            "by_condition": {condition: _counts([row for row in records if row["condition"] == condition])
                             for condition in CONDITIONS},
            "contrasts": {"strict_accuracy": _contrasts(records)},
            "secondary_contrasts": {"strict_accuracy": _secondary_contrasts(records)},
            "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
            "claim_boundaries": {"fixed_synthetic_worlds_only": True, "confidence_intervals": False,
                "population_inference": False, "internal_mechanism_identified": False, "a169_reopened": False,
                "missing_imputed": False, "complete_case_analysis": False, "heldout_allowed": False,
                "within_run_adaptation": False, "likelihood_collected": False, "interaction_estimands": False, "inert_and_placement_contrasts_are_secondary": True,
                "cohort_local_missingness": True, "selection_mechanism_identified": False,
                "population_equivalence": False, "floor_contrast_is_equivalence": False,
                "fence_is_accuracy_rescue": False, "isolated_phrase_or_token_mechanism": False}}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a178-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
            "settings": {"device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
                         "cpu_threads": 4, "seed": config["seed"], "max_new_tokens": MAX_NEW_TOKENS,
                         "do_sample": False, "num_beams": 1, "use_cache": True, "logits_to_keep": 1},
            "planned_cells": 56, "maximum_generation_calls": 56, "maximum_seconds": MAX_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a178-one-shot-v1", "pid": header["pid"],
            "plan_sha256": plan_sha256}), "one_shot_binding")
    require(_same(state._read(root / "startup-attempt.json"), _startup_attempt(header["pid"], plan_sha256, config_sha256))
            and not (root / "startup-failure.json").exists(), "startup_lineage")
    require(_same(state._read(root / "native-inputs.json"), prepared), "native_input_replay")
    folders, expected = root / "trials", {t["trial_id"] for t in plan["trials"]}
    if folders.exists():
        require(not folders.is_symlink() and folders.is_dir() and all(p.name in expected and p.is_dir()
                and not p.is_symlink() for p in folders.iterdir()), "unknown_trial")
    results, attempted, closed = {}, set(), False
    for trial in plan["trials"]:
        folder = folders / trial["trial_id"]
        if not folder.exists():
            closed = True
            continue
        require(not closed and {p.name for p in folder.iterdir()} <= {"attempt.json", "result.json"}
                and (folder / "attempt.json").is_file(), "attempt_sequence")
        attempt = state._read(folder / "attempt.json")
        require(_same(attempt, _attempt(header, trial, prepared)), "attempt_native_replay")
        attempted.add(trial["trial_id"])
        if not (folder / "result.json").exists():
            closed = True
            continue
        result = state._read(folder / "result.json")
        _validate_result(result, attempt, trial, tokenizer, eos)
        results[trial["trial_id"]] = result
    finished = None
    if (root / "execution-finished.json").exists():
        finished = state._read(root / "execution-finished.json")
        require(type(finished) is dict and set(finished) == {"schema_version", "status", "elapsed_seconds", "run_sha256", "interruption"}
                and finished["schema_version"] == "a178-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (float, int) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed"), "finished_receipt")
        interruption = finished["interruption"]
        if finished["status"] in ("interrupted", "deadline"):
            require(type(interruption) is dict and set(interruption) == {"signal_number", "signal_name", "pid", "exception_type"}
                    and type(interruption["pid"]) is int and interruption["pid"] == header["pid"] and interruption["exception_type"] in
                    ("A178Interrupted", "A178Deadline", "KeyboardInterrupt"), "interruption_receipt")
            allowed = {"A178Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       "A178Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            number = interruption["signal_number"]
            require((number is None or type(number) is int) and number in allowed[interruption["exception_type"]] and interruption["signal_name"] ==
                    (signal.Signals(number).name if number is not None else None)
                    and (finished["status"] == "deadline") == (interruption["exception_type"] == "A178Deadline"), "actual_signal_receipt")
        else:
            require(interruption is None, "unexpected_interruption")
        if finished["status"] == "finished_schedule":
            require(len(results) == 56, "finished_schedule_coverage")
    records = [_record(t, results.get(t["trial_id"]), t["trial_id"] in attempted) for t in plan["trials"]]
    aggregate = _aggregate(plan, records, results, attempted)
    aggregate["execution_status"] = finished["status"] if finished else "receipt_missing"
    aggregate["provenance"] = {key: header[key] for key in ("plan_sha256", "config_sha256", "source_hashes", "tests_sha256",
        "protocol_sha256", "native_prepared_sha256", "model_manifest_sha256")}
    aggregate["provenance"]["run_sha256"] = object_sha(header)
    for name, value in (("records.json", records), ("analysis.json", aggregate)):
        if (root / name).exists():
            require(_same(state._read(root / name), value), "stored_export_drift")
    return {"records": records, "aggregate": aggregate}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    root = _private_root(output_root)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, *inputs, plan_sha256, config_sha256, reference_script)


def _interrupt(signum, _frame):
    raise A178Interrupted(signum)


def _deadline(signum, _frame):
    raise A178Deadline(signum)


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


def run_plan(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
             output_root, reference_loader=parent.load_reference, tokenizer_loader=native.load_tokenizer,
             generation_runner=generate_tokens):
    """One fresh model, all 56 scheduled cells, no outcome gate or retry."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    root = _private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ("startup-attempt.json", "startup-failure.json", "one-shot.json", "run.json", "trials")), "consumed_run")
        startup = _startup_attempt(os.getpid(), plan_sha256, config_sha256)
        native.engine.write_private(root / "startup-attempt.json", startup)
        _PROCESS_CONSUMED = True
        header, status, interruption, runtime, error = None, "failed", None, None, None
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(bounded_signals())
                inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
                plan, config, sources, tokenizer, eos, prepared = inputs
                require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"], "original_model_manifest")
                reference = reference_loader(reference_script)
                available = reference.require_memory()
                require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
                native.engine.write_private(root / "one-shot.json", {"schema_version": "a178-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
                pending_header = {**_fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script),
                                  "pid": os.getpid(), "available_memory_bytes": available}
                native.engine.write_private(root / "run.json", pending_header)
                native.engine.write_private(root / "native-inputs.json", prepared)
                header = pending_header
                runtime = reference.load_cpu_reference(config, native.engine)
                require(runtime.eos_ids == eos and str(runtime.device) == "cpu"
                        and tasks.sha(runtime.tokenizer.get_chat_template().encode()) == config["chat_template_sha256"], "runtime_native_binding")
                for trial in plan["trials"]:
                    attempt = _attempt(header, trial, prepared)
                    folder = root / "trials" / trial["trial_id"]
                    native.engine.write_private(folder / "attempt.json", attempt)
                    tick = time.monotonic()
                    result = {"schema_version": "a178-result-v1", "attempt_sha256": object_sha(attempt), "status": "infrastructure_failed",
                              "generated_token_ids": None, "score": None, "error": None, "elapsed_seconds": 0.}
                    try:
                        tokens = generation_runner(runtime, prepared[trial["trial_id"]])
                        result.update(status="completed", generated_token_ids=tokens, score=score_tokens(runtime.tokenizer, tokens, trial, eos))
                        _validate_result(result, attempt, trial, tokenizer, eos)
                    except Exception as exc:
                        result.update(status="infrastructure_failed", generated_token_ids=None, score=None,
                                      error=native.engine.safe_error(exc, "a178_generation_failed"))
                    result["elapsed_seconds"] = time.monotonic() - tick
                    _validate_result(result, attempt, trial, tokenizer, eos)
                    native.engine.write_private(folder / "result.json", result)
                status = "finished_schedule"
            except (A178Interrupted, KeyboardInterrupt) as exc:
                error = exc
                status = "deadline" if isinstance(exc, A178Deadline) else "interrupted"
                interruption = _interruption(exc)
            except Exception as exc:
                error = exc
            finally:
                runtime = None
                if header is None:
                    native.engine.write_private(root / "startup-failure.json", {"schema_version": "a178-startup-failure-v1",
                        "startup_attempt_sha256": object_sha(startup), "status": status, "elapsed_seconds": time.monotonic() - started,
                        "interruption": interruption, "target_model_calls": 0,
                        "error": native.engine.safe_error(error, "a178_startup_failed") if error is not None else None})
                else:
                    if error is not None:
                        native.engine.write_private(root / "error.json", native.engine.safe_error(error, "a178_execution_failed"))
                    native.engine.write_private(root / "execution-finished.json", {"schema_version": "a178-execution-finished-v1", "status": status,
                        "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header), "interruption": interruption})
        if header is None:
            if error is not None:
                raise error
            raise RuntimeError("a178_startup_failed_without_exception")
        exported = _replay(root, *inputs, plan_sha256, config_sha256, reference_script)
        native.engine.write_private(root / "records.json", exported["records"])
        native.engine.write_private(root / "analysis.json", exported["aggregate"])
        return exported["aggregate"]


run = run_plan
export = export_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "export"), default="run")
    for name in ("plan", "config", "protocol", "reference-script", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "config-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    kwargs = {"plan_path": args.plan, "plan_sha256": args.plan_sha256, "config_path": args.config,
              "config_sha256": args.config_sha256, "protocol_path": args.protocol, "reference_script": args.reference_script,
              "output_root": args.output_root}
    try:
        root = _private_root(args.output_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / ("execution.log" if args.mode == "run" else "replay.log")).open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                aggregate = run_plan(**kwargs) if args.mode == "run" else export_run(**kwargs)["aggregate"]
        print(json.dumps({key: aggregate[key] for key in ("status", "execution_status", "coverage")}, sort_keys=True, allow_nan=False))
        return 0 if aggregate["status"] == "complete" and aggregate["execution_status"] == "finished_schedule" else 1
    except (Exception, A178Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a178_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
