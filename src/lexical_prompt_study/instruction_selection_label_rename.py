"""A168 fixed label-renaming diagnostic; no model calls at import.

Caller-supplied A167 inputs are bound and replayed; prior responses are never read.
The primary comparison includes all sixteen prospectively fixed prose pairs.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from . import instruction_selection_prose_control as prose

state, native, tasks, parent = prose.state, prose.native, prose.tasks, prose.parent
REFERENCE_SHA = prose.REFERENCE_SHA
PROSE_SHA = "12de497f8977bf51edf09c284c8a18d882d187fca7dd050578540518fff6cb43"
MAX_SECONDS = 2400
PHASES = ("original", "renamed")
ORDERS = ("original_then_renamed", "renamed_then_original")
DISPLAY_LABELS = {"original": {"A": "A", "B": "B"}, "renamed": {"A": "C", "B": "D"}}
RESPONSE_CLASSES = ("correct_value", "other_given_value", "selected_display_label", "other_display_label",
                    "legacy_selected_label", "legacy_other_label", "other_allowed_symbol", "other_string", "format_or_cap")
MIN_AVAILABLE_BYTES = state.MIN_AVAILABLE_BYTES
object_sha = state.object_sha
_PROCESS_CONSUMED = False


class A168Interrupted(BaseException):
    """Bypass inherited per-trial Exception recovery."""


class A168Deadline(BaseException):
    """Bound preflight, loading and execution in the same process."""


def require(condition, code):
    tasks.require(condition, "a168_" + code)


def validate_sources():
    require(native.engine.file_digest(Path(prose.__file__)) == PROSE_SHA, "a167_source_drift")
    return {**prose.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def _validate_parent(plan, header):
    validate_sources()
    prose.validate_plan(plan)
    expected = {"schema_version": "a167-run-v1", "plan_sha256": object_sha(plan),
        "config_sha256": plan["bindings"]["config_sha256"], "source_hashes": prose.validate_sources(),
        "a165_plan_sha256": plan["bindings"]["a165_plan_sha256"],
        "a165_run_sha256": plan["bindings"]["a165_run_sha256"],
        "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "prose_material_receipt": plan["prose_material_receipt"], "reference_script_sha256": REFERENCE_SHA,
        "eos_token_ids": plan["a165_run_header"]["eos_token_ids"], "settings": {
            **plan["a165_run_header"]["settings"], "do_sample": False, "num_beams": 1},
        "planned_cells": 48, "maximum_seconds": prose.MAX_SECONDS, "retries": 0, "heldout_allowed": False}
    variable = {"reference_script_path", "native_prepared_sha256", "matched_counts_sha256",
                "model_manifest_sha256", "pid", "available_memory_bytes"}
    require(type(header) is dict and set(header) == set(expected) | variable
            and all(tasks.canonical(header[k]) == tasks.canonical(v) for k, v in expected.items()), "a167_header_binding")
    require(all(tasks.is_hash(header[k]) for k in ("native_prepared_sha256", "matched_counts_sha256", "model_manifest_sha256"))
            and type(header["reference_script_path"]) is str and Path(header["reference_script_path"]).is_absolute()
            and type(header["pid"]) is int and header["pid"] > 0
            and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "a167_header_metadata")


def _rename_messages(trial, material):
    """Rename only known rendered label slots, never arbitrary prose characters."""
    selector = trial["selector"]
    world = tasks.world_from_payload(trial["world"])
    original_task = tasks.render_world(world, trial["query_order"])
    original_system = tasks.system_instruction(selector)
    placement = trial["paired_placement"]
    original_user = (material + "\n\n" + original_task if placement == "before"
                     else original_task + "\n\n" + material)
    require(trial["messages"] == [{"role": "system", "content": original_system},
                                  {"role": "user", "content": original_user}], "original_renderer_identity")
    renamed_system = original_system.replace(f"selector is {selector}.",
        f"selector is {DISPLAY_LABELS['renamed'][selector]}.", 1).replace("labeled A and B.", "labeled C and D.", 1)
    renamed_task = json.dumps({DISPLAY_LABELS["renamed"][key]: tasks.oracle(world, key)
                              for key in trial["query_order"]}, separators=(",", ":"))
    renamed_user = (material + "\n\n" + renamed_task if placement == "before"
                    else renamed_task + "\n\n" + material)
    return [{"role": "system", "content": renamed_system}, {"role": "user", "content": renamed_user}]


def _build_plan(a167_plan, a167_run_header, protocol_sha256):
    _validate_parent(a167_plan, a167_run_header)
    require(tasks.is_hash(protocol_sha256), "protocol_hash")
    selected = [t for t in a167_plan["trials"] if t["diagnostic_phase"] == "prose"]
    require(len(selected) == 16 and {t["world_index"] for t in selected} == set(range(4)), "all_original_prose_trials")
    bindings = {"protocol_sha256": protocol_sha256, "a168_source_sha256": native.engine.file_digest(Path(__file__)),
        "a167_source_sha256": PROSE_SHA, "a167_plan_sha256": object_sha(a167_plan),
        "a167_run_sha256": object_sha(a167_run_header), "config_sha256": a167_run_header["config_sha256"],
        "cpu_reference_script_sha256": REFERENCE_SHA}
    trials = []
    for pair_index, original in enumerate(selected):
        order = ORDERS[(original["world_index"] + original["selector_index"] + original["placement_index"]) % 2]
        renamed_messages = _rename_messages(original, a167_plan["prose_material"])
        for phase in order.split("_then_"):
            row = copy.deepcopy(original)
            row.pop("triplet_index")
            row.pop("material_order")
            row.pop("source_a165_trial_id")
            row.pop("trial_id")
            row.update(diagnostic_phase=phase, pair_index=pair_index, label_order=order,
                       source_a167_trial_id=original["trial_id"], displayed_label_mapping=DISPLAY_LABELS[phase].copy())
            if phase == "renamed":
                row["messages"] = copy.deepcopy(renamed_messages)
            row["trial_id"] = object_sha({"schema_version": "a168-trial-identity-v1", "bindings": bindings, "trial": row})[:24]
            trials.append(row)
    return {"schema_version": "a168-plan-v1", "partition": "development", "stage": "label_renaming_diagnostic",
        "world_count": 4, "pair_count": 16, "trial_count": 32, "trials": trials, "bindings": bindings,
        "a167_plan": copy.deepcopy(a167_plan), "a167_run_header": copy.deepcopy(a167_run_header),
        "displayed_label_mappings": copy.deepcopy(DISPLAY_LABELS),
        "likelihood_collected": False, "heldout_allowed": False}


def compile_plan(a167_plan, a167_run_header, protocol_sha256):
    """Compile all sixteen original prose trials; consume no A167 responses."""
    return _build_plan(a167_plan, a167_run_header, protocol_sha256)


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    expected = _build_plan(plan["a167_plan"], plan["a167_run_header"], plan["bindings"]["protocol_sha256"])
    require(tasks.canonical(plan) == tasks.canonical(expected), "fixed_plan_drift")
    identities = [t["trial_id"] for t in plan["trials"]]
    require(len(set(identities)) == 32 and not set(identities) & {t["trial_id"] for t in plan["a167_plan"]["trials"]},
            "new_trial_identities")


def _interrupt(_signum, _frame):
    raise A168Interrupted()


def _deadline(_signum, _frame):
    raise A168Deadline()


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
    header = plan["a167_run_header"]
    require(config["attention_implementation"] == "sdpa" and config["cpu_threads"] == 4
            and config["seed"] == header["settings"]["seed"], "same_cpu_settings")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    sources = validate_sources()
    tokenizer = tokenizer_loader(config)
    eos = native.pinned_eos_ids(config)
    require(eos == header["eos_token_ids"], "parent_eos_drift")
    # Reconstruct the complete bound A167 native freeze with its original identities.
    old = plan["a167_plan"]
    original_prepared = {t["trial_id"]: native.prepare_generation(tokenizer, t, config) for t in old["trials"]}
    original_matched = []
    for offset in range(0, 48, 3):
        group = {t["diagnostic_phase"]: t for t in old["trials"][offset:offset + 3]}
        counts = {phase: len(original_prepared[t["trial_id"]]["prompt_token_ids"]) for phase, t in group.items()}
        require(counts["prose"] == counts["inert"], "parent_exact_token_match")
        original_matched.append({"triplet_index": offset // 3, "paired_placement": group["baseline"]["paired_placement"],
                                 "prompt_token_counts": counts})
    expected_header = prose._fixed_header(old, config, prose.validate_sources(), eos, original_prepared,
        original_matched, object_sha(old), config_sha256, Path(header["reference_script_path"]))
    require(all(tasks.canonical(header[k]) == tasks.canonical(v) for k, v in expected_header.items()), "a167_native_header_replay")
    prepared = {t["trial_id"]: native.prepare_generation(tokenizer, t, config) for t in plan["trials"]}
    originals = {t["trial_id"]: t for t in old["trials"]}
    matched = []
    for offset in range(0, 32, 2):
        group = {t["diagnostic_phase"]: t for t in plan["trials"][offset:offset + 2]}
        original = group["original"]
        source = originals[original["source_a167_trial_id"]]
        require(original["messages"] == source["messages"] and prepared[original["trial_id"]]
                == original_prepared[source["trial_id"]], "original_native_identity")
        counts = {phase: len(prepared[t["trial_id"]]["prompt_token_ids"]) for phase, t in group.items()}
        require(counts["original"] == counts["renamed"], "exact_renamed_prompt_token_match")
        matched.append({"pair_index": offset // 2, "paired_placement": original["paired_placement"],
                        "prompt_token_counts": counts})
    return plan, config, sources, tokenizer, eos, prepared, matched


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    """Token-only preflight; returned message/token arrays remain private."""
    plan, _, sources, _, eos, prepared, matched = _inputs(plan_path, plan_sha256, config_path, config_sha256,
        protocol_path, reference_script, tokenizer_loader)
    return {"schema_version": "a168-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "reference_script_sha256": REFERENCE_SHA, "a167_run_sha256": plan["bindings"]["a167_run_sha256"],
        "planned_cells": 32, "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared,
        "eos_token_ids": eos, "matched_count_receipts": matched, "matched_counts_sha256": object_sha(matched)}


def _attempt(header, trial, prepared, index):
    return {"schema_version": "a168-attempt-v1", "run_sha256": object_sha(header),
        "trial_sha256": object_sha(trial), "prepared": prepared[trial["trial_id"]],
        "eos_token_ids": header["eos_token_ids"], "sequence_index": index}


def _validate_result(result, attempt, trial, tokenizer):
    require(type(result) is dict and result.get("schema_version") == "a168-result-v1"
            and type(result.get("record")) is dict and result["record"].get("schema_version") == "a168-cell-v1",
            "result_schema")
    # Explicit adapter; durable receipts always retain their distinct A168 schemas.
    wire = {**result, "schema_version": "a166-result-v1",
            "record": {**result["record"], "schema_version": "a166-cell-v1"}}
    state._validate_result(wire, attempt, trial, tokenizer)


def _response_class(trial, results):
    """Called only after token decoding, EOS and strict-score receipt validation."""
    result = results.get(trial["trial_id"])
    if result is None or result["record"]["generation_status"] != "completed":
        return None
    record = result["record"]
    if record["finish_reason"] != "eos" or record["score"]["category"] == "format":
        return "format_or_cap"
    value = json.loads(result["private"]["response_text"])["answer"]
    if value == trial["selected_answer"]:
        return "correct_value"
    if value == trial["unselected_answer"]:
        return "other_given_value"
    selector = trial["selector"]
    other = "B" if selector == "A" else "A"
    mapping = trial["displayed_label_mapping"]
    if value == mapping[selector]:
        return "selected_display_label"
    if value == mapping[other]:
        return "other_display_label"
    if trial["diagnostic_phase"] == "renamed":
        if value == selector:
            return "legacy_selected_label"
        if value == other:
            return "legacy_other_label"
    return "other_allowed_symbol" if value in tasks.SYMBOLS else "other_string"


def _binary(trial, results, outcome):
    result = results.get(trial["trial_id"])
    if result is None or result["record"]["generation_status"] != "completed":
        return 0, 1
    row = result["record"]
    require(type(row["score"]["strict_correct"]) is bool, "strict_score_boolean")
    value = (row["score"]["strict_correct"] if outcome == "strict_correct" else
             row["finish_reason"] == "length" if outcome == "capped" else row["score"]["category"] == "format")
    return int(value), int(value)


def _contrast(plan, results, *, outcome="strict_correct", placement=None, order=None):
    by_world = {}
    pairs = identified_pairs = 0
    for offset in range(0, 32, 2):
        group = {t["diagnostic_phase"]: t for t in plan["trials"][offset:offset + 2]}
        meta = group["original"]
        if ((placement is not None and meta["paired_placement"] != placement)
                or (order is not None and meta["label_order"] != order)):
            continue
        plus, minus = _binary(group["renamed"], results, outcome), _binary(group["original"], results, outcome)
        bound = plus[0] - minus[1], plus[1] - minus[0]
        by_world.setdefault(meta["world_index"], []).append(bound)
        pairs += 1
        identified_pairs += bound[0] == bound[1]
    require(len(by_world) == 4 and len({len(values) for values in by_world.values()}) == 1, "contrast_world_balance")
    means = [(sum(v[0] for v in by_world[index]) / len(by_world[index]),
              sum(v[1] for v in by_world[index]) / len(by_world[index])) for index in range(4)]
    lower, upper = sum(v[0] for v in means) / 4, sum(v[1] for v in means) / 4
    return {"planned_worlds": 4, "planned_pairs": pairs, "identified_pairs": identified_pairs,
        "identified_worlds": sum(v[0] == v[1] for v in means), "outcome": outcome,
        "direction": "renamed_minus_original", "world_mean_difference_pp": 100 * lower if lower == upper else None,
        "worst_best_bounds_pp": [100 * lower, 100 * upper], "fixed_world_means": [
            {"planned_pairs": len(by_world[index]), "difference_pp": 100 * lo if lo == hi else None,
             "worst_best_bounds_pp": [100 * lo, 100 * hi]} for index, (lo, hi) in enumerate(means)],
        "worlds_are_fixed_descriptive_units": True}


def _category_summary(trials, results):
    values = [_response_class(t, results) for t in trials]
    counts = {category: values.count(category) for category in RESPONSE_CLASSES}
    missing = values.count(None)
    return {"planned": len(trials), "resolved": len(trials) - missing, "unresolved": missing,
            "counts": counts, "count_bounds": {key: [value, value + sum(
                category is None and (not key.startswith("legacy_") or trial["diagnostic_phase"] == "renamed")
                for trial, category in zip(trials, values, strict=True))] for key, value in counts.items()}}


def _transfer(plan, results):
    known, unknown = [], []
    for offset in range(0, 32, 2):
        group = {t["diagnostic_phase"]: t for t in plan["trials"][offset:offset + 2]}
        category = _response_class(group["original"], results)
        if category == "selected_display_label":
            known.append(group["renamed"])
        elif category is None:
            unknown.append(group["renamed"])
    counterpart = _category_summary(known, results)
    uncertain = _category_summary(unknown, results)
    k, u, missing = len(known), len(unknown), counterpart["unresolved"]
    known_rates = {category: [100 * count / k, 100 * (count + missing) / k] if k else None
                   for category, count in counterpart["counts"].items()}
    # Conservative over all possible memberships and unresolved binary categories.
    # Unknown originals may or may not enter the subset; no historical errors select it.
    all_rates = {category: [100 * count / (k + u), 100 * (count + missing + u) / (k + u)]
                 if k else ([0., 100.] if u else None) for category, count in counterpart["counts"].items()}
    return {"definition": "Contemporaneous original response is valid JSON plus EOS containing its selected displayed label.",
        "known_members_K": k, "unknown_membership_pairs": u, "membership_count_bounds": [k, k + u],
        "known_nonmembers": 16 - k - u, "status": ("descriptive" if k else
            "inconclusive_membership" if u else "uninformative_K_zero"),
        "subset_may_be_empty": k == 0, "known_member_counterparts": counterpart,
        "known_member_category_rate_bounds_pp": known_rates, "unknown_membership_counterparts": uncertain,
        "possible_full_subset_category_rate_bounds_pp": all_rates,
        "possible_full_subset_bounds_are_conservative": True, "primary_pairs_retained": 16}


def _aggregate(plan, results, attempted):
    arms = {phase: state._arm([t for t in plan["trials"] if t["diagnostic_phase"] == phase], results, attempted)
            for phase in PHASES}
    coverage = state._arm(plan["trials"], results, attempted)
    complete = coverage["completed"] == 32
    return {"schema_version": "a168-analysis-v1", "status": "complete" if complete else "incomplete",
        "planned_worlds": 4, "planned_pairs": 16, "planned_cells": 32, "coverage": coverage, "arms": arms,
        "unattempted_cells": 32 - len(attempted), "interrupted_attempts": len(attempted) - len(results),
        "primary_renamed_minus_original": _contrast(plan, results),
        "secondary_placement_contrasts": {p: _contrast(plan, results, placement=p) for p in ("before", "after")},
        "secondary_order_contrasts": {order: _contrast(plan, results, order=order) for order in ORDERS},
        "secondary_format_and_cap_contrasts": {outcome: _contrast(plan, results, outcome=outcome) for outcome in ("format", "capped")},
        "response_classes": {phase: _category_summary([t for t in plan["trials"] if t["diagnostic_phase"] == phase], results)
                             for phase in PHASES}, "selected_label_error_transfer": _transfer(plan, results),
        "plan_object_sha256": object_sha(plan), "numeric_records_sha256": object_sha([
            results[t["trial_id"]]["record"] for t in plan["trials"] if t["trial_id"] in results]),
        "counting_note": "Response classes are exclusive; the separate parser-format and cap counts can overlap.",
        "estimand": "This displayed-label renaming in all sixteen fixed original prose trials at exactly matched native prompt count.",
        "claim_boundaries": {"descriptive_development_diagnostic": True, "heldout_allowed": False,
            "confidence_intervals": False, "population_inference": False, "world_independence_assumed": False,
            "role_versus_letter_mechanism_proven": False, "label_order_eliminates_all_carryover": False,
            "missing_outcomes_imputed": False, "raw_text_ids_or_per_item_token_hashes_exported": False,
            "historical_error_subset_used": False, "automatic_calls_beyond_fixed_schedule": False}}


def _fixed_header(plan, config, sources, eos, prepared, matched, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a168-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "a167_plan_sha256": plan["bindings"]["a167_plan_sha256"],
        "a167_run_sha256": plan["bindings"]["a167_run_sha256"], "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "displayed_label_mappings": plan["displayed_label_mappings"], "reference_script_sha256": REFERENCE_SHA,
        "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
        "matched_counts_sha256": object_sha(matched), "eos_token_ids": eos,
        "model_manifest_sha256": object_sha(config["model_files_sha256"]), "settings": plan["a167_run_header"]["settings"],
        "planned_cells": 32, "maximum_seconds": MAX_SECONDS, "retries": 0, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, matched, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, matched, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(tasks.canonical(header[k]) == tasks.canonical(v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(tasks.canonical(state._read(root / "one-shot.json")) == tasks.canonical(
        {"schema_version": "a168-one-shot-v1", "pid": header["pid"], "plan_sha256": plan_sha256}), "one_shot_binding")
    require(tasks.canonical(state._read(root / "native-inputs.json")) == tasks.canonical(prepared)
            and tasks.canonical(state._read(root / "matched-counts.json")) == tasks.canonical(matched),
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
        require(tasks.canonical(attempt) == tasks.canonical(_attempt(header, trial, prepared, index)), "attempt_native_replay")
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
        "parent_run_sha256": header["a167_run_sha256"], "displayed_label_mappings": header["displayed_label_mappings"],
        "native_prepared_sha256": header["native_prepared_sha256"], "matched_counts_sha256": header["matched_counts_sha256"]}
    for name, expected_value in (("records.json", records), ("analysis.json", aggregate)):
        if (root / name).exists():
            require(tasks.canonical(state._read(root / name)) == tasks.canonical(expected_value), "stored_export_drift")
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
    """Exactly one prospective 32-call schedule, with no retries or adaptive extension."""
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
        native.engine.write_private(root / "one-shot.json", {"schema_version": "a168-one-shot-v1",
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
                record["schema_version"] = "a168-cell-v1"
                result = {"schema_version": "a168-result-v1", "attempt_sha256": object_sha(attempt), "record": record,
                          "private": private, "elapsed_seconds": time.monotonic() - tick}
                _validate_result(result, attempt, trial, tokenizer)
                native.engine.write_private(folder / "result.json", result)
            status = "finished_schedule"
        except BaseException as exc:
            status = "deadline" if isinstance(exc, A168Deadline) else (
                "interrupted" if isinstance(exc, (A168Interrupted, KeyboardInterrupt)) else "failed")
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a168_execution_failed"))
        finally:
            runtime = None
            native.engine.write_private(root / "execution-finished.json", {"schema_version": "a168-execution-finished-v1",
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
        print(json.dumps({"status": "a168_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
