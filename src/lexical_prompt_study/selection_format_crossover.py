"""A176 selection/format crossover; no model calls at import.

All 48 public cells, two primary contrasts and three secondary contrasts
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

from . import closing_constraint_crossover as previous

state, native, tasks, parent = previous.state, previous.native, previous.tasks, previous.parent
object_sha = state.object_sha
A174_SHA = "3de139bdd52be83e850cb9f942ef1f728d7efb96aabd65c58a9903dc5cc7d13b"
CONFIG_SHA = "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745"
REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
PRIVATE_RUNS_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a176")
MAX_SECONDS, KILL_GRACE_SECONDS = 3600, 60
MAX_NEW_TOKENS = 64
MIN_AVAILABLE_BYTES = 48 * 1024**3
SELECTION_BLOCKS = FORMAT_REQUESTS = ("legacy", "a171_standard")
SELECTORS = ("A", "B")
AXES = ("selection_block", "format_request")
LEVELS = (SELECTION_BLOCKS, FORMAT_REQUESTS)
CATEGORIES = previous.CATEGORIES
WORLDS = (("s60", "s67", "AB"), ("s64", "s61", "BA"),
          ("s63", "s66", "BA"), ("s65", "s62", "AB"))
CONDITIONS = (
    ("normalized_legacy_legacy", "normalized", "legacy", "legacy"),
    ("normalized_legacy_standard", "normalized", "legacy", "a171_standard"),
    ("normalized_standard_legacy", "normalized", "a171_standard", "legacy"),
    ("normalized_standard_standard", "normalized", "a171_standard", "a171_standard"),
    ("original_legacy", "original", "legacy", "legacy"),
    ("original_standard", "original", "a171_standard", "a171_standard"),
)
CONDITION_NAMES = tuple(value[0] for value in CONDITIONS)
NORMALIZED_CONDITIONS = CONDITION_NAMES[:4]
NORMALIZATION_CONDITIONS = {"legacy": ("normalized_legacy_legacy", "original_legacy"),
                            "a171_standard": ("normalized_standard_standard", "original_standard")}
BASE_SCHEDULE = tuple(product(CONDITION_NAMES, SELECTORS))
SELECTION_TEXT = {
    "legacy": ("The authoritative query selector is {selector}. The user supplies two explicit "
               "answer values labeled A and B. Copy the value with the authoritative label."),
    "a171_standard": "Select the entry labeled {selector} from the user's mapping.",
}
NORMALIZED_FORMAT_TEXT = {
    "legacy": ('Output exactly one JSON object with the single field "answer" containing the mapped value '
               'of the entry labeled {selector} as a string.'),
    "a171_standard": ('Return the mapped value of the entry labeled {selector} as exactly one JSON object '
                      'with the single key "answer" and a string value.'),
}
ORIGINAL_FORMAT_TEXT = {
    "legacy": 'Output exactly one JSON object with the single field "answer" containing that value as a string.',
    "a171_standard": ("Return that entry's mapped value as exactly one JSON object with the single key "
                      '"answer" and a string value.'),
}
CLOSING_TEXT = "Do not transform the value or include other fields, explanations, or Markdown fences."
prepare_trial = previous.prepare_trial
generate_tokens = previous.generate_tokens
score_tokens = previous.score_tokens
_validate_tokens = previous._validate_tokens

_PROCESS_CONSUMED = False


class A176Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        self.signal_name = signal.Signals(signum).name
        super().__init__()


class A176Deadline(A176Interrupted):
    """Internal deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a176_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    require(native.engine.file_digest(Path(previous.__file__)) == A174_SHA, "a174_helper_source_drift")
    require(parent.REFERENCE_SHA == REFERENCE_SHA, "reference_binding")
    return {**previous.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def system_instruction(construction, selection_block, format_request, selector):
    require(construction in ("normalized", "original") and selection_block in SELECTION_BLOCKS
            and format_request in FORMAT_REQUESTS and selector in SELECTORS, "system_factor_levels")
    require(construction == "normalized" or selection_block == format_request, "original_diagonal_only")
    formats = NORMALIZED_FORMAT_TEXT if construction == "normalized" else ORIGINAL_FORMAT_TEXT
    return " ".join((SELECTION_TEXT[selection_block].format(selector=selector),
                     formats[format_request].format(selector=selector), CLOSING_TEXT))


def compile_plan(config_sha256, protocol_sha256, tests_sha256):
    """Compile the fixed public study; never read parent experimental data."""
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256) and tasks.is_hash(tests_sha256), "plan_bindings")
    for level in SELECTION_BLOCKS:
        for selector in SELECTORS:
            require(system_instruction("original", level, level, selector) == previous.system_instruction(level, "legacy", selector),
                    "exact_original_control_reconstruction")
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256,
                "source_sha256": sources[Path(__file__).name], "reference_script_sha256": REFERENCE_SHA,
                "a174_helper_sha256": A174_SHA}
    specifications = {name: (construction, selection, request) for name, construction, selection, request in CONDITIONS}
    trials = []
    for world_index, (value_a, value_b, presentation) in enumerate(WORLDS):
        rotation = 6 * (world_index // 2)
        schedule = BASE_SCHEDULE[rotation:] + BASE_SCHEDULE[:rotation]
        if world_index % 2:
            schedule = tuple(reversed(schedule))
        mapping = {"A": value_a, "B": value_b}
        user = json.dumps({key: mapping[key] for key in presentation}, separators=(",", ":"))
        for condition, selector in schedule:
            construction, selection, request = specifications[condition]
            system = system_instruction(construction, selection, request, selector)
            trial = {"world_index": world_index, "world_id": f"w{world_index:02d}", "presentation_order": presentation,
                     "construction": construction, "selection_block": selection, "format_request": request,
                     "condition": condition, "selector": selector,
                     "selected_answer": mapping[selector], "unselected_answer": mapping["B" if selector == "A" else "A"],
                     "sequence_index": len(trials),
                     "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            trial["trial_id"] = object_sha({"schema_version": "a176-cell-identity-v1", "bindings": bindings, "trial": trial})[:24]
            trials.append(trial)
    require(len(trials) == 48 and len({t["trial_id"] for t in trials}) == 48, "fixed_matrix")
    primary = {axis: {"positive": levels[1], "negative": levels[0], "conditions": list(NORMALIZED_CONDITIONS),
                     "planned_outcomes": 32, "planned_pairs": 16, "cell_coefficient_magnitude": 1 / 16}
               for axis, levels in zip(AXES, LEVELS)}
    secondary = {"selection_by_format_interaction": {"conditions": list(NORMALIZED_CONDITIONS),
        "positive_cells": [["legacy", "legacy"], ["a171_standard", "a171_standard"]],
        "negative_cells": [["legacy", "a171_standard"], ["a171_standard", "legacy"]],
        "planned_outcomes": 32, "planned_quartets": 8, "cell_coefficient_magnitude": 1 / 8}}
    for level, (positive, negative) in NORMALIZATION_CONDITIONS.items():
        secondary["normalization_" + level] = {"conditions": [positive, negative], "positive": positive, "negative": negative,
            "planned_outcomes": 16, "planned_pairs": 8, "cell_coefficient_magnitude": 1 / 8}
    return {"schema_version": "a176-plan-v1", "partition": "fresh_public_synthetic", "bindings": bindings,
            "trials": trials, "world_count": 4, "planned_cells": 48, "maximum_generation_calls": 48,
            "max_new_tokens": MAX_NEW_TOKENS, "heldout_allowed": False, "retries": 0,
            "resume_allowed": False, "baseline_gate": False, "likelihood_collected": False,
            "prose_allowed": False, "clarification_reminder_allowed": False, "mapping_heading_allowed": False,
            "study_prefix_allowed": False, "factor_levels": {axis: list(levels) for axis, levels in zip(AXES, LEVELS)},
            "conditions": [{"condition": name, "construction": construction, "selection_block": selection,
                            "format_request": request} for name, construction, selection, request in CONDITIONS],
            "primary_contrast_definitions": primary, "secondary_contrast_definitions": secondary,
            "controls_in_primary_estimands": False,
            "fence_marker_definition": "literal_ascii_triple_backtick_or_triple_tilde_anywhere"}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    bindings = plan["bindings"]
    require(_same(plan, compile_plan(bindings["config_sha256"], bindings["protocol_sha256"], bindings["tests_sha256"])), "fixed_plan_drift")


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
    require(len({object_sha(p["prompt_token_ids"]) for p in prepared.values()}) == 48, "unique_native_cells")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path,
                                                reference_script, tokenizer_loader)
    return {"schema_version": "a176-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "tests_sha256": plan["bindings"]["tests_sha256"],
            "source_hashes": sources, "reference_script_sha256": REFERENCE_SHA, "planned_cells": 48,
            "maximum_generation_calls": 48, "eos_token_ids": eos,
            "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _private_root(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(path.is_relative_to(PRIVATE_RUNS_ROOT) and path != PRIVATE_RUNS_ROOT, "a176_private_run_root")
    require(not path.is_relative_to(Path(__file__).absolute().parents[2]), "private_outside_source")
    return path


def _attempt(header, trial, prepared):
    return {"schema_version": "a176-attempt-v1", "run_sha256": object_sha(header), "trial_sha256": object_sha(trial),
            "trial_id": trial["trial_id"], "sequence_index": trial["sequence_index"], "prepared": prepared[trial["trial_id"]]}


def _startup_attempt(pid, plan_sha256, config_sha256):
    return {"schema_version": "a176-startup-attempt-v1", "pid": pid, "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_sha256": native.engine.file_digest(Path(__file__))}


def _interruption(exc):
    return {"signal_number": exc.signum if isinstance(exc, A176Interrupted) else None,
            "signal_name": exc.signal_name if isinstance(exc, A176Interrupted) else None,
            "pid": exc.pid if isinstance(exc, A176Interrupted) else os.getpid(), "exception_type": type(exc).__name__}


def _validate_result(result, attempt, trial, tokenizer, eos):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "generated_token_ids",
            "score", "error", "elapsed_seconds"} and result["schema_version"] == "a176-result-v1"
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
                                           "construction", "selection_block", "format_request", "condition", "selector")}
    if result is not None and result["status"] == "completed":
        score = {key: value for key, value in result["score"].items() if key != "response_text"}
    else:
        infrastructure = result is not None
        score = {"category": "infrastructure" if infrastructure else "missing", "strict_correct": None, "selected_label": None,
                 "format_valid": None, "eos_valid": None, "capped": None, "fence_marker": None, "attempted": attempted,
                 "reason": "recoverable_evaluation_failure" if infrastructure else "interrupted_attempt" if attempted else "unattempted",
                 "generated_token_count": None}
    return {"schema_version": "a176-cell-v1", **metadata, **score}


def _signed_bounds(terms):
    known = [(value, coefficient) for value, coefficient in terms if value is not None]
    fixed = math.fsum(coefficient * int(value) for value, coefficient in known)
    unresolved = [coefficient for value, coefficient in terms if value is None]
    return {"point": fixed if not unresolved else None,
            "lower": math.fsum([fixed, *(min(0., coefficient) for coefficient in unresolved)]),
            "upper": math.fsum([fixed, *(max(0., coefficient) for coefficient in unresolved)]),
            "planned_outcomes": len(terms), "resolved_outcomes": len(known)}


def _normalized(records):
    selected = [row for row in records if row["condition"] in NORMALIZED_CONDITIONS]
    require(len(selected) == 32 and all(row["construction"] == "normalized" for row in selected), "normalized_cohort")
    return selected


def _binary_bound(records, field, axis):
    """Primary normalized-only main effect, with sixteen equal matched pairs."""
    require(axis in AXES, "contrast_axis")
    selected = _normalized(records)
    levels = LEVELS[AXES.index(axis)]
    pairs, terms = {}, []
    for row in selected:
        require(row[axis] in levels and (row[field] is None or type(row[field]) is bool), "contrast_outcome_schema")
        key = (row["world_index"], row["selector"], *(row[other] for other in AXES if other != axis))
        values = pairs.setdefault(key, {})
        require(row[axis] not in values, "contrast_duplicate_level")
        values[row[axis]] = row[field]
        terms.append((row[field], (1 if row[axis] == levels[1] else -1) / 16))
    require(len(pairs) == 16 and all(set(values) == set(levels) for values in pairs.values()), "paired_contrast_cells")
    return {**_signed_bounds(terms), "planned_pairs": 16,
            "resolved_pairs": sum(all(value is not None for value in values.values()) for values in pairs.values())}


def _contrasts(records):
    return {axis: _binary_bound(records, "strict_correct", axis) for axis in AXES}


def _interaction_bound(records):
    """Secondary normalized-only difference of differences, eight quartets."""
    selected = _normalized(records)
    quartets, terms = {}, []
    expected = set(product(SELECTION_BLOCKS, FORMAT_REQUESTS))
    for row in selected:
        cell = (row["selection_block"], row["format_request"])
        value = row["strict_correct"]
        require(cell in expected and (value is None or type(value) is bool), "interaction_outcome_schema")
        values = quartets.setdefault((row["world_index"], row["selector"]), {})
        require(cell not in values, "interaction_duplicate_cell")
        values[cell] = value
        terms.append((value, (1 if cell[0] == cell[1] else -1) / 8))
    require(len(quartets) == 8 and all(set(values) == expected for values in quartets.values()), "complete_interaction_quartets")
    return {**_signed_bounds(terms), "planned_quartets": 8,
            "resolved_quartets": sum(all(value is not None for value in values.values()) for values in quartets.values())}


def _normalization_bound(records, level):
    """Secondary normalized diagonal minus original, using only its eight pairs."""
    require(level in NORMALIZATION_CONDITIONS, "normalization_level")
    positive, negative = NORMALIZATION_CONDITIONS[level]
    selected = [row for row in records if row["condition"] in (positive, negative)]
    require(len(selected) == 16, "normalization_cohort")
    pairs, terms = {}, []
    for row in selected:
        value = row["strict_correct"]
        require(value is None or type(value) is bool, "normalization_outcome_schema")
        values = pairs.setdefault((row["world_index"], row["selector"]), {})
        require(row["condition"] not in values, "normalization_duplicate_cell")
        values[row["condition"]] = value
        terms.append((value, (1 if row["condition"] == positive else -1) / 8))
    require(len(pairs) == 8 and all(set(values) == {positive, negative} for values in pairs.values()), "normalization_pair_cells")
    return {**_signed_bounds(terms), "planned_pairs": 8,
            "resolved_pairs": sum(all(value is not None for value in values.values()) for values in pairs.values())}


def _secondary_contrasts(records):
    return {"selection_by_format_interaction": _interaction_bound(records),
            **{"normalization_" + level: _normalization_bound(records, level) for level in SELECTION_BLOCKS}}


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
    return {"schema_version": "a176-analysis-v1", "status": "complete" if completed == 48 else "incomplete",
            "planned_worlds": 4, "planned_cells": 48, "maximum_generation_calls": 48,
            "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                         "interrupted": len(attempted) - len(results), "unattempted": 48 - len(attempted),
                         "missing": 48 - len(results)},
            "overall": _counts(records),
            "by_condition": {condition: _counts([row for row in records if row["condition"] == condition])
                             for condition in CONDITION_NAMES},
            "contrasts": {"strict_accuracy": _contrasts(records)},
            "secondary_contrasts": {"strict_accuracy": _secondary_contrasts(records)},
            "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
            "claim_boundaries": {"fixed_synthetic_worlds_only": True, "confidence_intervals": False,
                "population_inference": False, "internal_mechanism_identified": False, "a169_reopened": False,
                "missing_imputed": False, "complete_case_analysis": False, "heldout_allowed": False,
                "within_run_adaptation": False, "likelihood_collected": False, "unplanned_interaction_estimands": False, "interaction_is_secondary": True,
                "normalization_contrasts_are_secondary": True, "controls_gate_primary_cohort": False,
                "fence_is_accuracy_rescue": False, "isolated_phrase_or_token_mechanism": False}}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a176-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
            "settings": {"device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
                         "cpu_threads": 4, "seed": config["seed"], "max_new_tokens": MAX_NEW_TOKENS,
                         "do_sample": False, "num_beams": 1, "use_cache": True, "logits_to_keep": 1},
            "planned_cells": 48, "maximum_generation_calls": 48, "maximum_seconds": MAX_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a176-one-shot-v1", "pid": header["pid"],
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
                and finished["schema_version"] == "a176-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (float, int) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed"), "finished_receipt")
        interruption = finished["interruption"]
        if finished["status"] in ("interrupted", "deadline"):
            require(type(interruption) is dict and set(interruption) == {"signal_number", "signal_name", "pid", "exception_type"}
                    and type(interruption["pid"]) is int and interruption["pid"] == header["pid"] and interruption["exception_type"] in
                    ("A176Interrupted", "A176Deadline", "KeyboardInterrupt"), "interruption_receipt")
            allowed = {"A176Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       "A176Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            number = interruption["signal_number"]
            require((number is None or type(number) is int) and number in allowed[interruption["exception_type"]] and interruption["signal_name"] ==
                    (signal.Signals(number).name if number is not None else None)
                    and (finished["status"] == "deadline") == (interruption["exception_type"] == "A176Deadline"), "actual_signal_receipt")
        else:
            require(interruption is None, "unexpected_interruption")
        if finished["status"] == "finished_schedule":
            require(len(results) == 48, "finished_schedule_coverage")
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
    raise A176Interrupted(signum)


def _deadline(signum, _frame):
    raise A176Deadline(signum)


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
    """One fresh model, all 48 scheduled cells, no outcome gate or retry."""
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
                native.engine.write_private(root / "one-shot.json", {"schema_version": "a176-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
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
                    result = {"schema_version": "a176-result-v1", "attempt_sha256": object_sha(attempt), "status": "infrastructure_failed",
                              "generated_token_ids": None, "score": None, "error": None, "elapsed_seconds": 0.}
                    try:
                        tokens = generation_runner(runtime, prepared[trial["trial_id"]])
                        result.update(status="completed", generated_token_ids=tokens, score=score_tokens(runtime.tokenizer, tokens, trial, eos))
                        _validate_result(result, attempt, trial, tokenizer, eos)
                    except Exception as exc:
                        result.update(status="infrastructure_failed", generated_token_ids=None, score=None,
                                      error=native.engine.safe_error(exc, "a176_generation_failed"))
                    result["elapsed_seconds"] = time.monotonic() - tick
                    _validate_result(result, attempt, trial, tokenizer, eos)
                    native.engine.write_private(folder / "result.json", result)
                status = "finished_schedule"
            except (A176Interrupted, KeyboardInterrupt) as exc:
                error = exc
                status = "deadline" if isinstance(exc, A176Deadline) else "interrupted"
                interruption = _interruption(exc)
            except Exception as exc:
                error = exc
            finally:
                runtime = None
                if header is None:
                    native.engine.write_private(root / "startup-failure.json", {"schema_version": "a176-startup-failure-v1",
                        "startup_attempt_sha256": object_sha(startup), "status": status, "elapsed_seconds": time.monotonic() - started,
                        "interruption": interruption, "target_model_calls": 0,
                        "error": native.engine.safe_error(error, "a176_startup_failed") if error is not None else None})
                else:
                    if error is not None:
                        native.engine.write_private(root / "error.json", native.engine.safe_error(error, "a176_execution_failed"))
                    native.engine.write_private(root / "execution-finished.json", {"schema_version": "a176-execution-finished-v1", "status": status,
                        "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header), "interruption": interruption})
        if header is None:
            if error is not None:
                raise error
            raise RuntimeError("a176_startup_failed_without_exception")
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
    except (Exception, A176Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a176_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
