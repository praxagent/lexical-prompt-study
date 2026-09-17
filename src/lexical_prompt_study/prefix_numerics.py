"""A170 bounded synthetic prefix-numerics study; no model calls at import.

This is a separate method qualification, not a repeat or repair of A169. Only
the three identical canonical JSON-prefix tokens are scored. Native arrays and
all token-level measurements stay in private input/receipt return objects.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import signal
import time

from . import instruction_selection_answer_path as a169

native, state, tasks, parent = a169.native, a169.state, a169.tasks, a169.parent
object_sha = state.object_sha
A169_SHA = "1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e"
REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
CONFIG_SHA = "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745"
MAX_SECONDS, KILL_GRACE_SECONDS = 1800, 60
MIN_AVAILABLE_BYTES = 48 * 1024**3
PREFIX_COUNT = 3
PREFIX_ABSOLUTE_TOLERANCE_NATS = 1e-4
REPETITIONS = (0, 24, 72)
VALUES = ("s00", "s01", "A", "B")
FULL_METHODS = ("variable_full", "fixed_full")
SYSTEM = "Return the JSON object requested by the user."
JSON_PREFIX = '{"answer":"'
_PROCESS_CONSUMED = False


class A170Interrupted(BaseException):
    """A terminal signal, with its actual number and receiving PID preserved."""

    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        super().__init__()


class A170Deadline(A170Interrupted):
    """The internal wall deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a170_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    require(native.engine.file_digest(Path(a169.__file__)) == A169_SHA, "a169_source_drift")
    require(a169.REFERENCE_SHA == REFERENCE_SHA, "reference_binding_drift")
    return {**a169.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def compile_plan(config_sha256, protocol_sha256):
    """Build public fixed stimuli and exactly 30 measurements; no parent reads."""
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256), "plan_bindings")
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256,
                "a170_source_sha256": sources[Path(__file__).name], "a169_source_sha256": A169_SHA,
                "reference_script_sha256": REFERENCE_SHA}
    contexts, evaluations = [], []
    for index, repeats in enumerate(REPETITIONS):
        # Strip only the repeated filler. At zero repetitions two newlines remain.
        user = ("The field is quiet. " * repeats).rstrip() + '\n\nReturn {"answer":"s00"}.'
        context = {"context_index": index, "filler_repetitions": repeats,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                   "canonical_candidates": {value: json.dumps({"answer": value}, ensure_ascii=True,
                                              separators=(",", ":")) for value in VALUES}}
        context["context_id"] = object_sha({"bindings": bindings, "context": context})[:24]
        contexts.append(context)
        schedule = [("prefix_pre", None)] + [(method, value) for method in FULL_METHODS for value in VALUES]
        schedule += [("prefix_post", None)]
        for method, value in schedule:
            evaluation = {"context_index": index, "context_id": context["context_id"], "method": method,
                          "candidate": value, "sequence_index": len(evaluations)}
            evaluation["evaluation_id"] = object_sha({"bindings": bindings, "evaluation": evaluation})[:24]
            evaluations.append(evaluation)
    return {"schema_version": "a170-plan-v1", "partition": "public_synthetic", "bindings": bindings,
            "contexts": contexts, "evaluations": evaluations, "planned_contexts": 3, "planned_forward_calls": 30,
            "prefix_tokens": PREFIX_COUNT, "candidate_values": list(VALUES), "heldout_allowed": False,
            "retries": 0, "resume_allowed": False, "answer_scores_collected": False}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    expected = compile_plan(plan["bindings"]["config_sha256"], plan["bindings"]["protocol_sha256"])
    require(_same(plan, expected), "fixed_plan_drift")


def prepare_context(tokenizer, context, config, eos_ids):
    """Audit joint token boundaries, native assistant EOS, length pairs and pad."""
    template_sha = tasks.sha(tokenizer.get_chat_template().encode())
    require(template_sha == config["chat_template_sha256"], "native_template_drift")
    kwargs = {"add_generation_prompt": True, "date_string": native.engine.TEMPLATE_DATE}
    rendered = tokenizer.apply_chat_template(context["messages"], tokenize=False, **kwargs)
    prompt = native.engine._ids(tokenizer.apply_chat_template(context["messages"], tokenize=True,
                                                             return_dict=False, **kwargs))
    require(prompt == native.engine._ids(tokenizer.encode(rendered, add_special_tokens=False))
            and all(message["content"] in rendered for message in context["messages"]), "native_rendering")
    require(len(prompt) <= config["max_prompt_tokens"], "prompt_context_limit")
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos in eos_ids, "intended_eos")
    pad = getattr(tokenizer, "pad_token_id", None)
    pad_source = "tokenizer_pad_token_id" if pad is not None else "native_eos_fallback"
    pad = eos if pad is None else pad
    require(type(pad) is int and pad >= 0, "pad_token")
    paths = {}
    for value in VALUES:
        response = context["canonical_candidates"][value]
        full = native.engine._ids(tokenizer.encode(rendered + response, add_special_tokens=False))
        require(full[:len(prompt)] == prompt and len(full) > len(prompt), "native_prompt_prefix_invariance")
        continuation = full[len(prompt):]
        require(a169._decode(tokenizer, continuation) == response, "canonical_continuation_roundtrip")
        require(not any(token in eos_ids for token in continuation), "premature_eos")
        closed_messages = context["messages"] + [{"role": "assistant", "content": response}]
        closed_kwargs = {"add_generation_prompt": False, "date_string": native.engine.TEMPLATE_DATE}
        closed_text = tokenizer.apply_chat_template(closed_messages, tokenize=False, **closed_kwargs)
        closed = native.engine._ids(tokenizer.apply_chat_template(closed_messages, tokenize=True,
                                                                  return_dict=False, **closed_kwargs))
        require(closed == native.engine._ids(tokenizer.encode(closed_text, add_special_tokens=False))
                and closed == full + [eos], "native_assistant_eos_audit")
        require(len(closed) <= config["max_prompt_tokens"] + config["max_new_tokens"], "candidate_context_limit")
        paths[value] = continuation + [eos]
    common = list(paths[VALUES[0]])
    for tokens in paths.values():
        stop = next((i for i, (left, right) in enumerate(zip(common, tokens)) if left != right),
                    min(len(common), len(tokens)))
        common = common[:stop]
    text = a169._decode(tokenizer, common) if common else ""
    require(len(common) == PREFIX_COUNT and bool(text) and JSON_PREFIX.startswith(text), "three_token_json_prefix")
    require(native.engine._ids(tokenizer.encode(rendered + text, add_special_tokens=False)) == prompt + common,
            "shared_prefix_joint_roundtrip")
    # Audit the minimal prefix-only input independently, including its last boundary.
    minimal_text = a169._decode(tokenizer, common[:-1])
    require(native.engine._ids(tokenizer.encode(rendered + minimal_text, add_special_tokens=False)) == prompt + common[:-1],
            "minimal_prefix_joint_roundtrip")
    lengths = {value: len(tokens) for value, tokens in paths.items()}
    require(lengths["s00"] == lengths["s01"] and lengths["A"] == lengths["B"]
            and lengths["s00"] != lengths["A"] and min(lengths.values()) > PREFIX_COUNT, "two_equal_length_pairs")
    return {"rendered_text": rendered, "prompt_token_ids": prompt, "chat_template_sha256": template_sha,
            "shared_json_prefix_token_ids": common, "shared_json_prefix_text": text,
            "candidate_token_ids": paths, "candidate_lengths": lengths, "longest_target_count": max(lengths.values()),
            "intended_eos_token_id": eos, "padding_token_id": pad, "padding_token_source": pad_source}


def _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader):
    plan = native._read_bound(plan_path, plan_sha256)
    config = native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    require(config_sha256 == CONFIG_SHA == plan["bindings"]["config_sha256"]
            and config["attention_implementation"] == "sdpa" and config["cpu_threads"] == 4, "original_config")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    sources, tokenizer, eos = validate_sources(), tokenizer_loader(config), native.pinned_eos_ids(config)
    prepared = {c["context_id"]: prepare_context(tokenizer, c, config, eos) for c in plan["contexts"]}
    require(len({len(p["prompt_token_ids"]) for p in prepared.values()}) == 3, "three_native_prompt_lengths")
    require(len({object_sha(p["shared_json_prefix_token_ids"]) for p in prepared.values()}) == 1,
            "fixed_common_prefix_boundary")
    require(len({object_sha(p["candidate_lengths"]) for p in prepared.values()}) == 1, "fixed_candidate_lengths")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    """Token-only preflight. No parent plans, previous outcomes or model loading."""
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256,
                                                protocol_path, reference_script, tokenizer_loader)
    return {"schema_version": "a170-native-freeze-v1", "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_hashes": sources,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "planned_contexts": 3, "planned_forward_calls": 30, "eos_token_ids": eos,
            "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _forward_spec(prepared, evaluation):
    prompt, prefix = prepared["prompt_token_ids"], prepared["shared_json_prefix_token_ids"]
    method, candidate = evaluation["method"], evaluation["candidate"]
    if method in ("prefix_pre", "prefix_post"):
        require(candidate is None, "reference_has_no_candidate")
        ids, mask, keep, target_count, pad_count = prompt + prefix[:-1], [1] * (len(prompt) + PREFIX_COUNT - 1), PREFIX_COUNT, None, 0
    else:
        require(method in FULL_METHODS and candidate in VALUES, "full_method")
        targets = prepared["candidate_token_ids"][candidate]
        require(targets[:PREFIX_COUNT] == prefix and targets[-1] == prepared["intended_eos_token_id"], "forward_native_targets")
        target_count = len(targets)
        physical_targets = prepared["longest_target_count"] if method == "fixed_full" else target_count
        pad_count = physical_targets - target_count
        ids = prompt + targets + [prepared["padding_token_id"]] * pad_count
        mask = [1] * (len(prompt) + target_count) + [0] * pad_count
        keep = physical_targets + 1
    start = len(ids) - keep
    require(start == len(prompt) - 1 and len(prefix) == PREFIX_COUNT, "predictor_alignment")
    return {"input_token_ids": ids, "attention_mask": mask, "logits_to_keep": keep,
            "prefix_target_token_ids": prefix, "physical_sequence_length": len(ids),
            "native_prompt_length": len(prompt), "candidate_target_count": target_count, "right_pad_count": pad_count,
            "predictor_positions": list(range(start, start + PREFIX_COUNT))}


def measure_prefix(model, torch, spec, device="cpu"):
    """Exactly one fresh full forward and float64 readout of only prefix rows."""
    require(str(device) == "cpu", "cpu_only")
    require(len(spec["input_token_ids"]) == len(spec["attention_mask"]) == spec["physical_sequence_length"]
            and len(spec["prefix_target_token_ids"]) == PREFIX_COUNT, "forward_shape")
    ids = torch.tensor([spec["input_token_ids"]], dtype=torch.long, device=device)
    mask = torch.tensor([spec["attention_mask"]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=spec["logits_to_keep"]).logits
        require(logits.ndim == 3 and tuple(logits.shape[:2]) == (1, spec["logits_to_keep"])
                and logits.dtype == torch.float32, "fp32_shifted_logit_shape")
        used = logits[0, :PREFIX_COUNT]
        require(bool(torch.isfinite(used).all().item()), "finite_prefix_logits")
        targets = torch.tensor(spec["prefix_target_token_ids"], dtype=torch.long, device=used.device)
        chosen = used.gather(1, targets[:, None])[:, 0].double().tolist()
        normalizers = torch.logsumexp(used.double(), dim=-1).tolist()
        logprobs = [left - right for left, right in zip(chosen, normalizers)]
    result = {"chosen_logits": chosen, "log_normalizers": normalizers, "token_logprobs": logprobs,
              "json_prefix_logprob": math.fsum(logprobs)}
    _validate_measurement(result)
    return result


def _validate_measurement(value):
    require(type(value) is dict and set(value) == {"chosen_logits", "log_normalizers", "token_logprobs", "json_prefix_logprob"},
            "measurement_schema")
    require(all(type(value[key]) is list and len(value[key]) == PREFIX_COUNT
                and all(type(v) in (float, int) and math.isfinite(v) for v in value[key])
                for key in ("chosen_logits", "log_normalizers", "token_logprobs")), "score_length_or_finiteness")
    require(value["token_logprobs"] == [left - right for left, right in zip(value["chosen_logits"], value["log_normalizers"])]
            and all(v <= 0 for v in value["token_logprobs"]), "normalization_recomputation")
    require(type(value["json_prefix_logprob"]) in (float, int) and math.isfinite(value["json_prefix_logprob"])
            and _same(value["json_prefix_logprob"], math.fsum(value["token_logprobs"])), "prefix_sum_recomputation")


def _attempt(header, evaluation, prepared):
    value = prepared[evaluation["context_id"]]
    return {"schema_version": "a170-attempt-v1", "run_sha256": object_sha(header), "evaluation": evaluation,
            "prepared_sha256": object_sha(value), "forward_spec": _forward_spec(value, evaluation)}


def _validate_result(result, attempt):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "measurement", "error", "elapsed_seconds"}
            and result["schema_version"] == "a170-result-v1" and result["attempt_sha256"] == object_sha(attempt)
            and type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"])
            and result["elapsed_seconds"] >= 0, "result_binding")
    if result["status"] == "completed":
        require(result["error"] is None, "completed_error")
        _validate_measurement(result["measurement"])
    else:
        require(result["status"] == "infrastructure_failed" and result["measurement"] is None
                and type(result["error"]) is dict and set(result["error"]) == {"exception_type", "frames", "code"}, "failed_result_schema")


def _difference(left, right):
    """Signed left-minus-right deltas, with every unavailable value explicit."""
    present = left is not None and right is not None
    return {"json_prefix_logprob_nats": left["json_prefix_logprob"] - right["json_prefix_logprob"] if present else None,
            **{key: [a - b for a, b in zip(left[key], right[key])] if present else [None] * PREFIX_COUNT
               for key in ("chosen_logits", "log_normalizers", "token_logprobs")}}


def _group(measurements):
    complete = all(v is not None for v in measurements.values())
    spread = max(v["json_prefix_logprob"] for v in measurements.values()) - min(v["json_prefix_logprob"] for v in measurements.values()) if complete else None
    return {"planned_candidates": 4, "completed_candidates": sum(v is not None for v in measurements.values()),
            "prefix_sum_spread_nats": spread,
            "descriptive_guard_passed": spread <= PREFIX_ABSOLUTE_TOLERANCE_NATS if complete else None,
            "per_token_spreads": {key: [max(v[key][i] for v in measurements.values()) - min(v[key][i] for v in measurements.values())
                                        for i in range(PREFIX_COUNT)] if complete else [None] * PREFIX_COUNT
                                  for key in ("chosen_logits", "log_normalizers", "token_logprobs")},
            "pairwise_left_minus_right": {left + "_minus_" + right: _difference(measurements[left], measurements[right])
                                          for left, right in itertools.combinations(VALUES, 2)}}


def _records(plan, prepared, results):
    records = []
    for context in plan["contexts"]:
        measurements = {(e["method"], e["candidate"]): results.get(e["evaluation_id"], {}).get("measurement")
                        for e in plan["evaluations"] if e["context_id"] == context["context_id"]}
        pre, post = measurements["prefix_pre", None], measurements["prefix_post", None]
        methods = {method: {value: measurements[method, value] for value in VALUES} for method in FULL_METHODS}
        p = prepared[context["context_id"]]
        records.append({"schema_version": "a170-context-numerics-v1", "context_index": context["context_index"],
                        "context_id": context["context_id"], "filler_repetitions": context["filler_repetitions"],
                        "native_prompt_length": len(p["prompt_token_ids"]), "candidate_target_lengths": p["candidate_lengths"],
                        "measurements": {"prefix_pre": pre, "prefix_post": post, **methods},
                        "within_candidate_groups": {method: _group(values) for method, values in methods.items()},
                        "reference_post_minus_pre": _difference(post, pre),
                        "candidate_minus_reference_pre": {method: {value: _difference(measured, pre) for value, measured in values.items()}
                                                          for method, values in methods.items()},
                        "fixed_minus_variable": {value: _difference(methods["fixed_full"][value], methods["variable_full"][value]) for value in VALUES}})
    return records


def _aggregate(plan, records, results, attempted):
    completed = sum(r["status"] == "completed" for r in results.values())
    guards = [r["within_candidate_groups"][method]["descriptive_guard_passed"] for r in records for method in FULL_METHODS]
    return {"schema_version": "a170-analysis-v1", "status": "complete" if completed == 30 else "incomplete",
            "planned_contexts": 3, "planned_forward_calls": 30,
            "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                         "interrupted": len(attempted) - len(results), "unattempted": 30 - len(attempted)},
            "descriptive_prefix_guard": {"absolute_tolerance_nats": PREFIX_ABSOLUTE_TOLERANCE_NATS,
                "planned_four_candidate_groups": 6, "passed": sum(v is True for v in guards),
                "failed": sum(v is False for v in guards), "missing": sum(v is None for v in guards)},
            "prefix_reference_pairs_complete": sum(all(r["measurements"][k] is not None for k in ("prefix_pre", "prefix_post")) for r in records),
            "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
            "claim_boundaries": {"synthetic_method_qualification_only": True, "a169_reopened": False,
                "answer_scores_collected": False, "answer_ranks": False, "accuracy_analysis": False,
                "confidence_intervals": False, "internal_mechanism_identified": False, "adaptive_tolerance": False,
                "copied_prefix_spread_is_validation": False, "missing_scores_imputed": False, "heldout_allowed": False}}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a170-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "reference_script_sha256": REFERENCE_SHA, "reference_script_path": str(reference_script.absolute()),
            "native_prepared_sha256": object_sha(prepared), "model_manifest_sha256": object_sha(config["model_files_sha256"]),
            "eos_token_ids": eos, "settings": {"device": "cpu", "dtype": "float32", "quantization": None,
                "attention_implementation": "sdpa", "cpu_threads": 4, "seed": config["seed"], "use_cache": False,
                "planned_forward_calls": 30, "answer_scores_collected": False},
            "normalization_arithmetic": "float64_logsumexp_of_float32_logits", "prefix_tokens": PREFIX_COUNT,
            "prefix_absolute_tolerance_nats": PREFIX_ABSOLUTE_TOLERANCE_NATS,
            "maximum_seconds": MAX_SECONDS, "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a170-one-shot-v1", "pid": header["pid"],
            "plan_sha256": plan_sha256}), "one_shot_binding")
    require(_same(state._read(root / "native-inputs.json"), prepared), "native_input_replay")
    folders, expected = root / "evaluations", {e["evaluation_id"] for e in plan["evaluations"]}
    if folders.exists():
        require(not folders.is_symlink() and folders.is_dir() and all(p.name in expected and p.is_dir()
                and not p.is_symlink() for p in folders.iterdir()), "unknown_evaluation")
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
                and finished["schema_version"] == "a170-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (int, float) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed"), "finished_receipt")
        interruption = finished["interruption"]
        if finished["status"] in ("interrupted", "deadline"):
            require(type(interruption) is dict and set(interruption) == {"signal_number", "pid", "exception_type"}
                    and interruption["pid"] == header["pid"] and interruption["exception_type"] in
                    ("A170Interrupted", "A170Deadline", "KeyboardInterrupt"), "interruption_receipt")
            allowed = {"A170Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       "A170Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            require(interruption["signal_number"] in allowed[interruption["exception_type"]]
                    and (finished["status"] == "deadline") == (interruption["exception_type"] == "A170Deadline"), "actual_signal_receipt")
        else:
            require(interruption is None, "unexpected_interruption")
        if finished["status"] == "finished_schedule":
            require(len(results) == 30, "finished_schedule_coverage")
    records = _records(plan, prepared, results)
    aggregate = _aggregate(plan, records, results, attempted)
    aggregate["execution_status"] = finished["status"] if finished else "receipt_missing"
    aggregate["provenance"] = {key: header[key] for key in ("plan_sha256", "config_sha256", "source_hashes",
        "protocol_sha256", "native_prepared_sha256", "model_manifest_sha256")}
    aggregate["provenance"]["run_sha256"] = object_sha(header)
    for name, value in (("records.json", records), ("analysis.json", aggregate)):
        if (root / name).exists():
            require(_same(state._read(root / name), value), "stored_export_drift")
    return {"records": records, "aggregate": aggregate}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    """Replay positions, order, receipts and arithmetic under a shared lock."""
    root = a169.lineage._private_root(output_root)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, *inputs, plan_sha256, config_sha256, reference_script)


def _interrupt(signum, _frame):
    raise A170Interrupted(signum)


def _deadline(signum, _frame):
    raise A170Deadline(signum)


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
             prefix_scorer=measure_prefix):
    """One fresh model load and one non-adaptive 30-call schedule, never resumed."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    root = a169.lineage._private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with bounded_signals(), (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ("one-shot.json", "run.json", "evaluations")), "consumed_run")
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        plan, config, sources, tokenizer, eos, prepared = inputs
        require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"], "original_model_manifest")
        reference = reference_loader(reference_script)
        available = reference.require_memory()
        require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
        _PROCESS_CONSUMED = True
        native.engine.write_private(root / "one-shot.json", {"schema_version": "a170-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
        header = {**_fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script),
                  "pid": os.getpid(), "available_memory_bytes": available}
        native.engine.write_private(root / "run.json", header)
        native.engine.write_private(root / "native-inputs.json", prepared)
        status, interruption, runtime = "failed", None, None
        try:
            runtime = reference.load_cpu_reference(config, native.engine)
            require(runtime.eos_ids == eos and str(runtime.device) == "cpu"
                    and tasks.sha(runtime.tokenizer.get_chat_template().encode()) == config["chat_template_sha256"], "runtime_native_binding")
            for evaluation in plan["evaluations"]:
                attempt = _attempt(header, evaluation, prepared)
                folder = root / "evaluations" / evaluation["evaluation_id"]
                native.engine.write_private(folder / "attempt.json", attempt)
                tick = time.monotonic()
                result = {"schema_version": "a170-result-v1", "attempt_sha256": object_sha(attempt), "status": "infrastructure_failed",
                          "measurement": None, "error": None, "elapsed_seconds": 0.}
                try:
                    result["measurement"] = prefix_scorer(runtime.model, runtime.torch, attempt["forward_spec"], runtime.device)
                    result["status"] = "completed"
                    _validate_result(result, attempt)
                except Exception as exc:
                    result.update(status="infrastructure_failed", measurement=None,
                                  error=native.engine.safe_error(exc, "a170_measurement_failed"))
                result["elapsed_seconds"] = time.monotonic() - tick
                _validate_result(result, attempt)
                native.engine.write_private(folder / "result.json", result)
            status = "finished_schedule"
        except (A170Interrupted, KeyboardInterrupt) as exc:
            status = "deadline" if isinstance(exc, A170Deadline) else "interrupted"
            interruption = {"signal_number": exc.signum if isinstance(exc, A170Interrupted) else None,
                            "pid": exc.pid if isinstance(exc, A170Interrupted) else os.getpid(), "exception_type": type(exc).__name__}
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a170_execution_interrupted"))
        except Exception as exc:
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a170_execution_failed"))
        finally:
            runtime = None
            native.engine.write_private(root / "execution-finished.json", {"schema_version": "a170-execution-finished-v1",
                "status": status, "elapsed_seconds": time.monotonic() - started,
                "run_sha256": object_sha(header), "interruption": interruption})
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
        root = a169.lineage._private_root(args.output_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / ("execution.log" if args.mode == "run" else "replay.log")).open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                aggregate = run_plan(**kwargs) if args.mode == "run" else export_run(**kwargs)["aggregate"]
        print(json.dumps({key: aggregate[key] for key in ("status", "execution_status", "coverage")}, sort_keys=True, allow_nan=False))
        return 0 if aggregate["status"] == "complete" and aggregate["execution_status"] == "finished_schedule" else 1
    except (Exception, A170Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a170_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
