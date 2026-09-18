"""A179 fixed clean canonical-path numerical and functional qualification.

No model calls at import. All native arrays and token readouts remain private.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

from . import mapped_value_rewrite_path as previous
from . import instruction_selection_answer_path as a169

state, native, tasks, parent = previous.state, previous.native, previous.tasks, previous.parent
object_sha = state.object_sha
A177_SHA = "7a10f0e4c361f35b0c56b92c6921bd89ef7cec2f7b8deefdb99912dacaeb394f"
A169_SHA = "1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e"
CONFIG_SHA = "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745"
REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
PRIVATE_RUNS_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a179")
MAX_SECONDS, KILL_GRACE_SECONDS = 3600, 60
MIN_AVAILABLE_BYTES = 48 * 1024**3
PREFIX_COUNT = 3
NUMERICAL_TOLERANCE_NATS = 1e-4
FUNCTIONAL_SEPARATION_NATS = 1e-3
JSON_PREFIX = '{"answer":"'
VALUE_PAIRS = (("s96", "s97"), ("s98", "s99"))
CANDIDATES = ("value_1", "value_2")
SELECTOR_ORDERS = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"))
FUNCTIONAL_STATUSES = ("unresolved", "pass", "worst_reversed", "worst_tie", "threshold_boundary", "positive_low_separation")
FUNCTIONAL_FLAGS = ("functional_passed", "mean_tie", "worst_tie", "mean_reversed", "worst_reversed",
                    "positive_low_separation", "threshold_boundary")
teacher_force = a169.teacher_force
_PROCESS_CONSUMED = False


class A179Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        self.signal_name = signal.Signals(signum).name
        super().__init__()


class A179Deadline(A179Interrupted):
    """Internal deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a179_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    require(native.engine.file_digest(Path(previous.__file__)) == A177_SHA, "a177_source_drift")
    require(native.engine.file_digest(Path(a169.__file__)) == A169_SHA, "a169_source_drift")
    require(parent.REFERENCE_SHA == a169.REFERENCE_SHA == REFERENCE_SHA, "reference_binding")
    return {**previous.validate_sources(), **a169.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def compile_plan(config_sha256, protocol_sha256, tests_sha256):
    """Compile all fresh public contexts and every forward before outcomes."""
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256) and tasks.is_hash(tests_sha256), "plan_bindings")
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256,
                "source_sha256": sources[Path(__file__).name], "a177_helper_sha256": A177_SHA,
                "a169_helper_sha256": A169_SHA, "reference_script_sha256": REFERENCE_SHA}
    contexts, evaluations = [], []
    for world_index, selectors in enumerate(SELECTOR_ORDERS):
        pair_index, swapped = world_index // 2, bool(world_index % 2)
        values = dict(zip(CANDIDATES, VALUE_PAIRS[pair_index]))
        displayed = tuple(reversed(VALUE_PAIRS[pair_index])) if swapped else VALUE_PAIRS[pair_index]
        mapping = dict(zip(("A", "B"), displayed))
        user = json.dumps(mapping, separators=(",", ":"))
        order = ("value_1", "value_2", "value_2", "value_1") if not swapped else ("value_2", "value_1", "value_1", "value_2")
        for selector in selectors:
            selected = next(name for name, value in values.items() if value == mapping[selector])
            other = next(name for name in CANDIDATES if name != selected)
            context = {"world_index": world_index, "world_id": f"w{world_index:02d}", "pair_index": pair_index,
                "assignment_swapped": swapped, "presentation_order": "AB", "selector": selector,
                "selected_candidate": selected, "other_candidate": other,
                "selected_answer": mapping[selector], "unselected_answer": mapping["B" if selector == "A" else "A"],
                "context_index": len(contexts),
                "messages": [{"role": "system", "content": previous.system_instruction("of_explicit_label", selector)},
                             {"role": "user", "content": user}],
                "canonical_candidates": {name: json.dumps({"answer": value}, separators=(",", ":")) for name, value in values.items()}}
            context["context_id"] = object_sha({"schema_version": "a179-context-identity-v1", "bindings": bindings, "context": context})[:24]
            contexts.append(context)
            repeats = {name: 0 for name in CANDIDATES}
            for candidate in order:
                evaluation = {"context_index": context["context_index"], "context_id": context["context_id"],
                    "candidate": candidate, "repeat_index": repeats[candidate], "sequence_index": len(evaluations)}
                repeats[candidate] += 1
                evaluation["evaluation_id"] = object_sha({"schema_version": "a179-evaluation-identity-v1", "bindings": bindings,
                                                          "evaluation": evaluation})[:24]
                evaluations.append(evaluation)
    require(len(contexts) == len({c["context_id"] for c in contexts}) == 8
            and len(evaluations) == len({e["evaluation_id"] for e in evaluations}) == 32, "fixed_matrix")
    return {"schema_version": "a179-plan-v1", "partition": "fresh_public_synthetic", "bindings": bindings,
            "contexts": contexts, "evaluations": evaluations, "planned_value_pairs": 2, "planned_world_assignments": 4,
            "planned_contexts": 8, "planned_forward_calls": 32, "prefix_tokens": PREFIX_COUNT,
            "numerical_tolerance_nats": NUMERICAL_TOLERANCE_NATS, "functional_separation_nats": FUNCTIONAL_SEPARATION_NATS,
            "suffix_includes_eos": True, "equal_candidate_lengths_required": True, "padding_allowed": False,
            "retries": 0, "resume_allowed": False, "heldout_allowed": False, "interim_gate": False,
            "generation_allowed": False, "scaffold_allowed": False, "warmup_allowed": False}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    b = plan["bindings"]
    require(_same(plan, compile_plan(b["config_sha256"], b["protocol_sha256"], b["tests_sha256"])), "fixed_plan_drift")


def prepare_context(tokenizer, context, config, eos_ids):
    template_sha = tasks.sha(tokenizer.get_chat_template().encode())
    require(template_sha == config["chat_template_sha256"], "native_template_drift")
    kwargs = {"add_generation_prompt": True, "date_string": native.engine.TEMPLATE_DATE}
    rendered = tokenizer.apply_chat_template(context["messages"], tokenize=False, **kwargs)
    prompt = native.engine._ids(tokenizer.apply_chat_template(context["messages"], tokenize=True, return_dict=False, **kwargs))
    require(prompt == native.engine._ids(tokenizer.encode(rendered, add_special_tokens=False))
            and all(message["content"] in rendered for message in context["messages"]), "native_rendering")
    require(0 < len(prompt) <= config["max_prompt_tokens"], "prompt_context_limit")
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos in eos_ids, "intended_eos")
    paths = {}
    for candidate in CANDIDATES:
        response = context["canonical_candidates"][candidate]
        full = native.engine._ids(tokenizer.encode(rendered + response, add_special_tokens=False))
        require(full[:len(prompt)] == prompt and len(full) > len(prompt), "native_prompt_prefix_invariance")
        continuation = full[len(prompt):]
        require(a169._decode(tokenizer, continuation) == response, "canonical_continuation_roundtrip")
        require(not any(token in eos_ids for token in continuation), "premature_eos")
        closed_messages = context["messages"] + [{"role": "assistant", "content": response}]
        closed_kwargs = {"add_generation_prompt": False, "date_string": native.engine.TEMPLATE_DATE}
        closed_text = tokenizer.apply_chat_template(closed_messages, tokenize=False, **closed_kwargs)
        closed = native.engine._ids(tokenizer.apply_chat_template(closed_messages, tokenize=True, return_dict=False, **closed_kwargs))
        require(closed == native.engine._ids(tokenizer.encode(closed_text, add_special_tokens=False))
                and closed == full + [eos], "native_assistant_eos_audit")
        require(len(closed) <= config["max_prompt_tokens"] + config["max_new_tokens"], "candidate_context_limit")
        paths[candidate] = continuation + [eos]
    lengths = {name: len(tokens) for name, tokens in paths.items()}
    require(len(set(lengths.values())) == 1 and min(lengths.values()) > PREFIX_COUNT, "equal_candidate_lengths")
    prefix = paths[CANDIDATES[0]][:PREFIX_COUNT]
    require(all(tokens[:PREFIX_COUNT] == prefix for tokens in paths.values()), "three_shared_prefix_tokens")
    text = a169._decode(tokenizer, prefix)
    require(bool(text) and JSON_PREFIX.startswith(text), "pure_json_prefix")
    require(native.engine._ids(tokenizer.encode(rendered + text, add_special_tokens=False)) == prompt + prefix,
            "shared_prefix_joint_roundtrip")
    return {"rendered_text": rendered, "prompt_token_ids": prompt, "chat_template_sha256": template_sha,
            "shared_json_prefix_token_ids": prefix, "shared_json_prefix_text": text,
            "candidate_token_ids": paths, "candidate_lengths": lengths, "intended_eos_token_id": eos}


def _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader):
    plan, config = native._read_bound(plan_path, plan_sha256), native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    require(config_sha256 == CONFIG_SHA == plan["bindings"]["config_sha256"] and config["cpu_threads"] == 4
            and config["attention_implementation"] == "sdpa", "original_config")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    model_config_path = Path(config["model_path"]) / "config.json"
    require(native.engine.file_digest(model_config_path) == config["model_files_sha256"]["config.json"], "model_config_binding")
    metadata = json.loads(model_config_path.read_bytes())
    limit, vocab = metadata.get("max_position_embeddings"), metadata.get("vocab_size")
    require(type(limit) is int and limit > 0 and type(vocab) is int and vocab > 0, "model_dimensions")
    sources, tokenizer, eos = validate_sources(), tokenizer_loader(config), native.pinned_eos_ids(config)
    require(all(type(e) is int and 0 <= e < vocab for e in eos), "model_eos_vocabulary")
    prepared = {c["context_id"]: prepare_context(tokenizer, c, config, eos) for c in plan["contexts"]}
    for value in prepared.values():
        for target in value["candidate_token_ids"].values():
            require(len(value["prompt_token_ids"]) + len(target) <= limit, "model_context_fit")
            require(all(type(t) is int and 0 <= t < vocab for t in value["prompt_token_ids"] + target), "native_token_vocabulary")
        value["model_context_limit"], value["model_vocab_size"] = limit, vocab
    require(len({object_sha(v["prompt_token_ids"]) for v in prepared.values()}) == 8, "unique_native_contexts")
    require(len({object_sha(v["shared_json_prefix_token_ids"]) for v in prepared.values()}) == 1, "fixed_common_prefix_boundary")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path,
                                                reference_script, tokenizer_loader)
    return {"schema_version": "a179-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "tests_sha256": plan["bindings"]["tests_sha256"],
            "source_hashes": sources, "reference_script_sha256": REFERENCE_SHA, "planned_contexts": 8,
            "planned_forward_calls": 32, "eos_token_ids": eos, "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _attempt(header, evaluation, prepared):
    value = prepared[evaluation["context_id"]]
    return {"schema_version": "a179-attempt-v1", "run_sha256": object_sha(header), "evaluation": evaluation,
        "prepared_sha256": object_sha(value), "prompt_token_ids": value["prompt_token_ids"],
        "target_token_ids": value["candidate_token_ids"][evaluation["candidate"]],
        "json_prefix_token_count": PREFIX_COUNT, "intended_eos_token_id": value["intended_eos_token_id"]}


def _context_record(context, prepared, evaluations, results):
    measurements = {name: [None, None] for name in CANDIDATES}
    for evaluation in evaluations:
        result = results.get(evaluation["evaluation_id"])
        if result is not None and result["status"] == "completed":
            measurements[evaluation["candidate"]][evaluation["repeat_index"]] = result["measurement"]
    available = [m for values in measurements.values() for m in values if m is not None]
    complete = len(available) == 4
    prefix_range = max(m["scores"]["json_prefix_logprob"] for m in available) - min(m["scores"]["json_prefix_logprob"] for m in available) if complete else None
    prefix_guard = prefix_range <= NUMERICAL_TOLERANCE_NATS if complete else None
    drift = {name: abs(values[1]["scores"]["answer_continuation_logprob"] - values[0]["scores"]["answer_continuation_logprob"])
             if all(value is not None for value in values) else None for name, values in measurements.items()}
    repeat_guard = {name: value <= NUMERICAL_TOLERANCE_NATS if value is not None else None for name, value in drift.items()}
    guards = [prefix_guard, *repeat_guard.values()]
    numerical = False if any(value is False for value in guards) else True if complete and all(value is True for value in guards) else None
    mean, worst, functional_status = None, None, "unresolved"
    flags = dict.fromkeys(FUNCTIONAL_FLAGS)
    if numerical is True:
        selected = [m["scores"]["answer_continuation_logprob"] for m in measurements[context["selected_candidate"]]]
        other = [m["scores"]["answer_continuation_logprob"] for m in measurements[context["other_candidate"]]]
        mean = math.fsum(selected) / 2 - math.fsum(other) / 2
        worst = min(selected) - max(other)
        flags = {"functional_passed": worst > FUNCTIONAL_SEPARATION_NATS, "mean_tie": mean == 0., "worst_tie": worst == 0.,
                 "mean_reversed": mean < 0., "worst_reversed": worst < 0.,
                 "positive_low_separation": 0. < worst <= FUNCTIONAL_SEPARATION_NATS,
                 "threshold_boundary": worst == FUNCTIONAL_SEPARATION_NATS}
        functional_status = ("pass" if flags["functional_passed"] else "worst_reversed" if worst < 0. else "worst_tie" if worst == 0.
                             else "threshold_boundary" if flags["threshold_boundary"] else "positive_low_separation")
    metadata = {key: context[key] for key in ("context_index", "context_id", "world_index", "world_id", "pair_index",
                "assignment_swapped", "selector", "selected_candidate", "other_candidate")}
    return {"schema_version": "a179-context-qualification-v1", **metadata,
            "native_prompt_length": len(prepared["prompt_token_ids"]), "candidate_target_lengths": prepared["candidate_lengths"],
            "planned_evaluations": 4, "completed_evaluations": len(available), "complete": complete,
            "measurements": measurements, "prefix_sum_range_nats": prefix_range, "prefix_guard_passed": prefix_guard,
            "continuation_repeat_drift_nats": drift, "repeat_guard_passed": repeat_guard, "numerical_valid": numerical,
            "mean_margin_nats": mean, "worst_margin_nats": worst,
            "mean_margin_bounds_nats": {"lower": mean, "upper": mean}, "worst_margin_bounds_nats": {"lower": worst, "upper": worst},
            "margins_unbounded": numerical is not True, **flags, "functional_status": functional_status}


def _records(plan, prepared, results):
    return [_context_record(c, prepared[c["context_id"]],
            [e for e in plan["evaluations"] if e["context_id"] == c["context_id"]], results) for c in plan["contexts"]]


def _flag_counts(values):
    return {"true": sum(v is True for v in values), "false": sum(v is False for v in values), "unknown": sum(v is None for v in values)}


def _aggregate(plan, records, results, attempted):
    completed = sum(result["status"] == "completed" for result in results.values())
    verdicts = [row[key] for row in records for key in ("numerical_valid", "functional_passed")]
    qualified = False if any(v is False for v in verdicts) else True if len(records) == 8 and all(v is True for v in verdicts) else None
    return {"schema_version": "a179-analysis-v1", "status": "complete" if completed == 32 else "incomplete",
            "planned_value_pairs": 2, "planned_world_assignments": 4, "planned_contexts": 8, "planned_forward_calls": 32,
            "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                         "interrupted": len(attempted) - len(results), "unattempted": 32 - len(attempted), "missing": 32 - completed},
            "complete_contexts": sum(r["complete"] for r in records), "qualified": qualified,
            "numerical_valid": _flag_counts([r["numerical_valid"] for r in records]),
            "functional_flags": {key: _flag_counts([r[key] for r in records]) for key in FUNCTIONAL_FLAGS},
            "functional_statuses": {key: sum(r["functional_status"] == key for r in records) for key in FUNCTIONAL_STATUSES},
            "numerical_tolerance_nats": NUMERICAL_TOLERANCE_NATS, "functional_separation_nats": FUNCTIONAL_SEPARATION_NATS,
            "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
            "claim_boundaries": {"a169_reopened": False, "canonical_path_conditional_preference_only": True,
                "suffix_includes_eos": True, "numeric_tolerance_is_error_bound": False, "generation_accuracy": False,
                "scaffold_effect": False, "selection_mechanism": False, "patching_qualified": False,
                "confidence_intervals": False, "population_inference": False, "missing_imputed": False,
                "heldout_allowed": False, "adaptive_tolerance": False}}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a179-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
            "settings": {"device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
                         "cpu_threads": 4, "seed": config["seed"], "use_cache": False,
                         "logits_to_keep": "target_count_plus_one", "padding": False},
            "planned_contexts": 8, "planned_forward_calls": 32, "maximum_seconds": MAX_SECONDS,
            "prefix_tokens": PREFIX_COUNT, "numerical_tolerance_nats": NUMERICAL_TOLERANCE_NATS,
            "functional_separation_nats": FUNCTIONAL_SEPARATION_NATS, "suffix_includes_eos": True,
            "normalization_arithmetic": "float64_logsumexp_of_float32_logits",
            "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False, "heldout_allowed": False}

def _private_root(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(path.is_relative_to(PRIVATE_RUNS_ROOT) and path != PRIVATE_RUNS_ROOT, "a179_private_run_root")
    require(not path.is_relative_to(Path(__file__).absolute().parents[2]), "private_outside_source")
    return path


def _startup_attempt(pid, plan_sha256, config_sha256):
    return {"schema_version": "a179-startup-attempt-v1", "pid": pid, "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_sha256": native.engine.file_digest(Path(__file__))}


def _interruption(exc):
    return {"signal_number": exc.signum if isinstance(exc, A179Interrupted) else None,
            "signal_name": exc.signal_name if isinstance(exc, A179Interrupted) else None,
            "pid": exc.pid if isinstance(exc, A179Interrupted) else os.getpid(), "exception_type": type(exc).__name__}


def _validate_result(result, attempt):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "measurement", "error", "elapsed_seconds"}
            and result["schema_version"] == "a179-result-v1" and result["attempt_sha256"] == object_sha(attempt)
            and type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"])
            and result["elapsed_seconds"] >= 0, "result_binding")
    if result["status"] == "completed":
        value = result["measurement"]
        require(result["error"] is None and type(value) is dict and set(value) == {"chosen_logits", "log_normalizers", "token_logprobs", "scores"}, "measurement_schema")
        require(all(type(value[key]) is list and len(value[key]) == len(attempt["target_token_ids"])
                    and all(type(v) in (float, int) and math.isfinite(v) for v in value[key])
                    for key in ("chosen_logits", "log_normalizers", "token_logprobs")), "score_length_or_finiteness")
        require(value["token_logprobs"] == [a - b for a, b in zip(value["chosen_logits"], value["log_normalizers"])],
                "normalization_recomputation")
        require(_same(value["scores"], a169._scores(value["token_logprobs"], attempt["json_prefix_token_count"])), "score_recomputation")
    else:
        require(result["status"] == "infrastructure_failed" and result["measurement"] is None
                and type(result["error"]) is dict and set(result["error"]) == {"exception_type", "frames", "code"}, "failed_result_schema")


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a179-one-shot-v1", "pid": header["pid"],
            "plan_sha256": plan_sha256}), "one_shot_binding")
    require(_same(state._read(root / "startup-attempt.json"), _startup_attempt(header["pid"], plan_sha256, config_sha256))
            and not (root / "startup-failure.json").exists(), "startup_lineage")
    require(_same(state._read(root / "native-inputs.json"), prepared), "native_input_replay")
    folders, expected = root / "evaluations", {t["evaluation_id"] for t in plan["evaluations"]}
    if folders.exists():
        require(not folders.is_symlink() and folders.is_dir() and all(p.name in expected and p.is_dir()
                and not p.is_symlink() for p in folders.iterdir()), "unknown_trial")
    results, attempted, closed = {}, set(), False
    for evaluation in plan["evaluations"]:
        folder = folders / evaluation["evaluation_id"]
        if not folder.exists():
            closed = True
            continue
        require(not closed and {p.name for p in folder.iterdir()} <= {"attempt.json", "result.json"}
                and (folder / "attempt.json").is_file(), "attempt_sequence")
        attempt = state._read(folder / "attempt.json")
        require(_same(attempt, _attempt(header, evaluation, prepared)), "attempt_native_replay")
        attempted.add(evaluation["evaluation_id"])
        if not (folder / "result.json").exists():
            closed = True
            continue
        result = state._read(folder / "result.json")
        _validate_result(result, attempt)
        results[evaluation["evaluation_id"]] = result
    finished = None
    if (root / "execution-finished.json").exists():
        finished = state._read(root / "execution-finished.json")
        require(type(finished) is dict and set(finished) == {"schema_version", "status", "elapsed_seconds", "run_sha256", "interruption"}
                and finished["schema_version"] == "a179-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (float, int) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed"), "finished_receipt")
        interruption = finished["interruption"]
        if finished["status"] in ("interrupted", "deadline"):
            require(type(interruption) is dict and set(interruption) == {"signal_number", "signal_name", "pid", "exception_type"}
                    and type(interruption["pid"]) is int and interruption["pid"] == header["pid"] and interruption["exception_type"] in
                    ("A179Interrupted", "A179Deadline", "KeyboardInterrupt"), "interruption_receipt")
            allowed = {"A179Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       "A179Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            number = interruption["signal_number"]
            require((number is None or type(number) is int) and number in allowed[interruption["exception_type"]] and interruption["signal_name"] ==
                    (signal.Signals(number).name if number is not None else None)
                    and (finished["status"] == "deadline") == (interruption["exception_type"] == "A179Deadline"), "actual_signal_receipt")
        else:
            require(interruption is None, "unexpected_interruption")
        if finished["status"] == "finished_schedule":
            require(len(results) == 32, "finished_schedule_coverage")
    records = _records(plan, prepared, results)
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
    raise A179Interrupted(signum)


def _deadline(signum, _frame):
    raise A179Deadline(signum)


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
             forward_scorer=teacher_force):
    """One fresh model, all 32 scheduled forwards, no outcome gate or retry."""
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
        require(not any((root / name).exists() for name in ("startup-attempt.json", "startup-failure.json", "one-shot.json", "run.json", "evaluations")), "consumed_run")
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
                native.engine.write_private(root / "one-shot.json", {"schema_version": "a179-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
                pending_header = {**_fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script),
                                  "pid": os.getpid(), "available_memory_bytes": available}
                native.engine.write_private(root / "run.json", pending_header)
                native.engine.write_private(root / "native-inputs.json", prepared)
                header = pending_header
                runtime = reference.load_cpu_reference(config, native.engine)
                require(runtime.eos_ids == eos and str(runtime.device) == "cpu"
                        and tasks.sha(runtime.tokenizer.get_chat_template().encode()) == config["chat_template_sha256"], "runtime_native_binding")
                for evaluation in plan["evaluations"]:
                    attempt = _attempt(header, evaluation, prepared)
                    folder = root / "evaluations" / evaluation["evaluation_id"]
                    native.engine.write_private(folder / "attempt.json", attempt)
                    tick = time.monotonic()
                    result = {"schema_version": "a179-result-v1", "attempt_sha256": object_sha(attempt), "status": "infrastructure_failed",
                              "measurement": None, "error": None, "elapsed_seconds": 0.}
                    try:
                        result["measurement"] = forward_scorer(runtime.model, runtime.torch, attempt["prompt_token_ids"],
                            attempt["target_token_ids"], PREFIX_COUNT, runtime.device)
                        result["status"] = "completed"
                        _validate_result(result, attempt)
                    except Exception as exc:
                        result.update(status="infrastructure_failed", measurement=None,
                                      error=native.engine.safe_error(exc, "a179_measurement_failed"))
                    result["elapsed_seconds"] = time.monotonic() - tick
                    _validate_result(result, attempt)
                    native.engine.write_private(folder / "result.json", result)
                status = "finished_schedule"
            except (A179Interrupted, KeyboardInterrupt) as exc:
                error = exc
                status = "deadline" if isinstance(exc, A179Deadline) else "interrupted"
                interruption = _interruption(exc)
            except Exception as exc:
                error = exc
            finally:
                runtime = None
                if header is None:
                    native.engine.write_private(root / "startup-failure.json", {"schema_version": "a179-startup-failure-v1",
                        "startup_attempt_sha256": object_sha(startup), "status": status, "elapsed_seconds": time.monotonic() - started,
                        "interruption": interruption, "target_model_calls": 0,
                        "error": native.engine.safe_error(error, "a179_startup_failed") if error is not None else None})
                else:
                    if error is not None:
                        native.engine.write_private(root / "error.json", native.engine.safe_error(error, "a179_execution_failed"))
                    native.engine.write_private(root / "execution-finished.json", {"schema_version": "a179-execution-finished-v1", "status": status,
                        "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header), "interruption": interruption})
        if header is None:
            if error is not None:
                raise error
            raise RuntimeError("a179_startup_failed_without_exception")
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
    except (Exception, A179Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a179_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
