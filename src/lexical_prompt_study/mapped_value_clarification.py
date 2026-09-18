"""A171 fresh mapped-value clarification experiment; no model calls at import.

The 96 public synthetic cells are fixed prospectively. Generation evidence,
native arrays and row-level outcomes are private; stdout is coverage only.
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

from . import instruction_selection_cpu_state as state

native, tasks, parent = state.native, state.tasks, state.parent
object_sha = state.object_sha
STATE_SHA = "1f51ed42e6a36ba27b34aeeff5810e579eed0055068a4b884639e49069a90f04"
CONFIG_SHA = "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745"
REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
PRIVATE_RUNS_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a171")
MAX_SECONDS, KILL_GRACE_SECONDS = 7200, 60
MAX_NEW_TOKENS = 64
MIN_AVAILABLE_BYTES = 48 * 1024**3
CONTEXTS, WORDINGS, SELECTORS = ("none", "before", "after"), ("standard", "clarified"), ("A", "B")
CATEGORIES = ("correct", "selected_label", "other_label", "other_mapped_value", "other_or_format", "cap", "infrastructure", "missing")
WORLDS = (("a171_v47", "a171_v83", "AB"), ("a171_v26", "a171_v91", "BA"),
          ("a171_v68", "a171_v14", "BA"), ("a171_v35", "a171_v72", "AB"),
          ("a171_v89", "a171_v20", "AB"), ("a171_v56", "a171_v03", "BA"),
          ("a171_v11", "a171_v64", "BA"), ("a171_v97", "a171_v42", "AB"))
STANDARD = ('Select the entry labeled {selector} from the user\'s mapping. Return that entry\'s mapped value as '
            'exactly one JSON object with the single key "answer" and a string value. Do not return any other text.')
REMINDER = "The selector names a mapping entry; the answer is its mapped value, not the selector label."
PROSE = ("Beyond the window, a narrow garden path curved around a patch of low shrubs. Rain from the previous evening "
         "remained on the broad leaves, while the gravel had begun to dry. A wooden bench stood beneath the nearest tree. "
         "Its surface was smooth where visitors usually sat, and a few fallen leaves rested at one end. Farther along the "
         "path, small flowers grew beside a shallow stone basin. Water moved gently when a breeze crossed the garden. "
         "The branches above made shifting patterns of light on the ground. Near the wall, a climbing plant reached "
         "toward a sheltered corner. The air felt cool in the shade and warmer beside the open lawn. From time to time "
         "a bird landed on the fence, paused, and flew toward the neighboring trees. The scene changed slowly as the sun "
         "rose. Shadows shortened, the leaves became less damp, and the quiet path remained open between the plants.")
BASE_SCHEDULE = (("none", "standard", "A"), ("before", "clarified", "B"), ("after", "standard", "A"),
                 ("none", "clarified", "B"), ("before", "standard", "A"), ("after", "clarified", "B"),
                 ("none", "standard", "B"), ("before", "clarified", "A"), ("after", "standard", "B"),
                 ("none", "clarified", "A"), ("before", "standard", "B"), ("after", "clarified", "A"))
_PROCESS_CONSUMED = False


class A171Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        self.signal_name = signal.Signals(signum).name
        super().__init__()


class A171Deadline(A171Interrupted):
    """Internal deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a171_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    require(native.engine.file_digest(Path(state.__file__)) == STATE_SHA, "cpu_helper_source_drift")
    require(parent.REFERENCE_SHA == REFERENCE_SHA, "reference_binding")
    return {**state.validate_sources(), Path(__file__).name: native.engine.file_digest(Path(__file__))}


def compile_plan(config_sha256, protocol_sha256, tests_sha256):
    """Compile only fixed public worlds, wording and schedule; no parent reads."""
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256) and tasks.is_hash(tests_sha256), "plan_bindings")
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256,
                "source_sha256": sources[Path(__file__).name], "reference_script_sha256": REFERENCE_SHA}
    trials = []
    for world_index, (value_a, value_b, presentation) in enumerate(WORLDS):
        mapping = {"A": value_a, "B": value_b}
        core = "Mapping:\n" + json.dumps({key: mapping[key] for key in presentation}, separators=(",", ":"))
        rotation = 3 * (world_index // 2)
        schedule = BASE_SCHEDULE[rotation:] + BASE_SCHEDULE[:rotation]
        if world_index % 2:
            schedule = tuple(reversed(schedule))
        for context, wording, selector in schedule:
            user = core if context == "none" else PROSE + "\n\n" + core if context == "before" else core + "\n\n" + PROSE
            system = STANDARD.format(selector=selector) + (" " + REMINDER if wording == "clarified" else "")
            trial = {"world_index": world_index, "world_id": f"w{world_index:02d}", "presentation_order": presentation,
                     "context": context, "wording": wording, "selector": selector, "selected_answer": mapping[selector],
                     "unselected_answer": mapping["B" if selector == "A" else "A"], "sequence_index": len(trials),
                     "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            trial["trial_id"] = object_sha({"schema_version": "a171-cell-identity-v1", "bindings": bindings, "trial": trial})[:24]
            trials.append(trial)
    require(len(trials) == 96 and len({t["trial_id"] for t in trials}) == 96, "fixed_matrix")
    return {"schema_version": "a171-plan-v1", "partition": "fresh_public_synthetic", "bindings": bindings,
            "trials": trials, "world_count": 8, "planned_cells": 96, "maximum_generation_calls": 96,
            "max_new_tokens": MAX_NEW_TOKENS, "heldout_allowed": False, "retries": 0,
            "resume_allowed": False, "baseline_gate": False, "likelihood_collected": False}


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    bindings = plan["bindings"]
    require(_same(plan, compile_plan(bindings["config_sha256"], bindings["protocol_sha256"], bindings["tests_sha256"])), "fixed_plan_drift")


def _decode(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def prepare_trial(tokenizer, trial, config, eos_ids):
    template_sha = tasks.sha(tokenizer.get_chat_template().encode())
    require(template_sha == config["chat_template_sha256"], "native_template_drift")
    kwargs = {"add_generation_prompt": True, "date_string": native.engine.TEMPLATE_DATE}
    text = tokenizer.apply_chat_template(trial["messages"], tokenize=False, **kwargs)
    prompt = native.engine._ids(tokenizer.apply_chat_template(trial["messages"], tokenize=True, return_dict=False, **kwargs))
    require(prompt == native.engine._ids(tokenizer.encode(text, add_special_tokens=False))
            and all(m["content"] in text for m in trial["messages"]), "native_rendering")
    require(len(prompt) <= config["max_prompt_tokens"], "prompt_context_limit")
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos in eos_ids, "native_eos")
    # A token-only native assistant probe audits the generation boundary and
    # intended terminator. It is never a generation input or a scored path.
    probe = json.dumps({"answer": trial["selected_answer"]}, separators=(",", ":"))
    joint = native.engine._ids(tokenizer.encode(text + probe, add_special_tokens=False))
    require(joint[:len(prompt)] == prompt and len(joint) > len(prompt), "generation_prompt_boundary")
    suffix = joint[len(prompt):]
    require(_decode(tokenizer, suffix) == probe and not any(t in eos_ids for t in suffix), "native_probe_roundtrip")
    kwargs = {"add_generation_prompt": False, "date_string": native.engine.TEMPLATE_DATE}
    messages = trial["messages"] + [{"role": "assistant", "content": probe}]
    closed_text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    closed = native.engine._ids(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs))
    require(closed == joint + [eos] == native.engine._ids(tokenizer.encode(closed_text, add_special_tokens=False)), "native_assistant_eos")
    require(len(closed) <= config["max_prompt_tokens"] + MAX_NEW_TOKENS, "native_probe_context_limit")
    return {"rendered_text": text, "prompt_token_ids": prompt, "chat_template_sha256": template_sha,
            "intended_eos_token_id": eos, "assistant_probe_token_ids": suffix + [eos]}


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
    require(len({object_sha(p["prompt_token_ids"]) for p in prepared.values()}) == 96, "unique_native_cells")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path,
                                                reference_script, tokenizer_loader)
    return {"schema_version": "a171-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "tests_sha256": plan["bindings"]["tests_sha256"],
            "source_hashes": sources, "reference_script_sha256": REFERENCE_SHA, "planned_cells": 96,
            "maximum_generation_calls": 96, "eos_token_ids": eos,
            "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def generate_tokens(runtime, prepared):
    """One greedy generation with original CPU-reference settings, raw evidence.

Unlike the old wrapper, retain a valid-length malformed termination so it is an
observable behavioral failure, rather than converting it to infrastructure.
"""
    from transformers import GenerationConfig
    torch = runtime.torch
    require(str(runtime.device) == "cpu", "cpu_only")
    prompt = prepared["prompt_token_ids"]
    ids = torch.tensor([prompt], dtype=torch.long, device=runtime.device)
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        use_cache=True, eos_token_id=runtime.eos_ids, pad_token_id=runtime.eos_ids[0],
        bos_token_id=runtime.tokenizer.bos_token_id)
    with torch.inference_mode():
        generated = runtime.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           generation_config=config, logits_to_keep=1)
    require(generated.ndim == 2 and generated.shape[0] == 1, "generation_shape")
    full = generated[0].tolist()
    require(full[:len(prompt)] == prompt, "generation_preserves_prompt")
    tokens = full[len(prompt):]
    _validate_tokens(tokens)
    vocab_size = runtime.model.config.vocab_size
    require(type(vocab_size) is int and vocab_size > 0 and all(token < vocab_size for token in tokens), "generated_token_vocabulary")
    return tokens


def _validate_tokens(tokens):
    require(type(tokens) is list and 1 <= len(tokens) <= MAX_NEW_TOKENS
            and all(type(t) is int and t >= 0 for t in tokens), "generated_token_array")


def score_tokens(tokenizer, tokens, trial, eos_ids):
    """Strict protocol categories from raw tokens; no response-based rescue."""
    _validate_tokens(tokens)
    require(type(eos_ids) is list and eos_ids and all(type(e) is int and e >= 0 for e in eos_ids), "eos_ids")
    final_eos = tokens[-1] in eos_ids
    eos_valid = final_eos and not any(t in eos_ids for t in tokens[:-1])
    capped = len(tokens) == MAX_NEW_TOKENS and not final_eos
    text = _decode(tokenizer, tokens[:-1] if final_eos else tokens)
    answer, format_valid = None, False
    try:
        value = json.loads(text, object_pairs_hook=tasks._unique_object, parse_constant=tasks._reject_constant)
        format_valid = type(value) is dict and set(value) == {"answer"} and type(value["answer"]) is str
        if format_valid:
            answer = value["answer"]
    except (ValueError, TypeError, RecursionError):
        pass
    reason = None
    if capped:
        category, reason = "cap", "token_limit_without_terminal_eos"
    elif not eos_valid:
        category, reason = "other_or_format", "invalid_termination"
    elif not format_valid:
        category, reason = "other_or_format", "invalid_format"
    elif answer == trial["selected_answer"]:
        category = "correct"
    elif answer == trial["selector"]:
        category = "selected_label"
    elif answer == ("B" if trial["selector"] == "A" else "A"):
        category = "other_label"
    elif answer == trial["unselected_answer"]:
        category = "other_mapped_value"
    else:
        category, reason = "other_or_format", "unrecognized_answer"
    return {"category": category, "strict_correct": category == "correct", "selected_label": category == "selected_label",
            "format_valid": format_valid, "eos_valid": eos_valid, "capped": capped, "attempted": True,
            "reason": reason, "response_text": text, "generated_token_count": len(tokens)}


def _private_root(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(path.is_relative_to(PRIVATE_RUNS_ROOT) and path != PRIVATE_RUNS_ROOT, "a171_private_run_root")
    require(not path.is_relative_to(Path(__file__).absolute().parents[2]), "private_outside_source")
    return path


def _attempt(header, trial, prepared):
    return {"schema_version": "a171-attempt-v1", "run_sha256": object_sha(header), "trial_sha256": object_sha(trial),
            "trial_id": trial["trial_id"], "sequence_index": trial["sequence_index"], "prepared": prepared[trial["trial_id"]]}


def _startup_attempt(pid, plan_sha256, config_sha256):
    return {"schema_version": "a171-startup-attempt-v1", "pid": pid, "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_sha256": native.engine.file_digest(Path(__file__))}


def _interruption(exc):
    return {"signal_number": exc.signum if isinstance(exc, A171Interrupted) else None,
            "signal_name": exc.signal_name if isinstance(exc, A171Interrupted) else None,
            "pid": exc.pid if isinstance(exc, A171Interrupted) else os.getpid(), "exception_type": type(exc).__name__}


def _validate_result(result, attempt, trial, tokenizer, eos):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "generated_token_ids",
            "score", "error", "elapsed_seconds"} and result["schema_version"] == "a171-result-v1"
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
                                           "context", "wording", "selector")}
    if result is not None and result["status"] == "completed":
        score = {key: value for key, value in result["score"].items() if key != "response_text"}
    else:
        infrastructure = result is not None
        score = {"category": "infrastructure" if infrastructure else "missing", "strict_correct": None, "selected_label": None,
                 "format_valid": None, "eos_valid": None, "capped": None, "attempted": attempted,
                 "reason": "recoverable_evaluation_failure" if infrastructure else "interrupted_attempt" if attempted else "unattempted",
                 "generated_token_count": None}
    return {"schema_version": "a171-cell-v1", **metadata, **score}


def _binary_bound(records, field, coefficients):
    """Sharp linear bounds; repeated terms are combined before bounding."""
    terms = [(r, coefficients[r["trial_id"]]) for r in records if coefficients.get(r["trial_id"], 0) != 0]
    require(bool(terms) and all(math.isfinite(c) for _, c in terms), "contrast_coefficients")
    known = [(r, c) for r, c in terms if r[field] is not None]
    fixed = math.fsum(c * int(r[field]) for r, c in known)
    unresolved = [c for r, c in terms if r[field] is None]
    pairs = {}
    for row, _ in terms:
        pairs.setdefault((row["world_index"], row["selector"], row["context"]), []).append(row[field])
    require(all(len(values) == 2 for values in pairs.values()), "paired_contrast_cells")
    return {"point": fixed if not unresolved else None,
            "lower": math.fsum([fixed, *(min(0., c) for c in unresolved)]),
            "upper": math.fsum([fixed, *(max(0., c) for c in unresolved)]),
            "planned_outcomes": len(terms), "resolved_outcomes": len(known), "planned_pairs": len(pairs),
            "resolved_pairs": sum(all(v is not None for v in values) for values in pairs.values())}


def _contrast(records, field, context_weights):
    coefficients = {r["trial_id"]: context_weights.get(r["context"], 0.) * (1 if r["wording"] == "clarified" else -1) / 16
                    for r in records}
    return _binary_bound(records, field, coefficients)


def _contrasts(records, field):
    return {"wording_by_context": {context: _contrast(records, field, {context: 1.}) for context in CONTEXTS},
            "primary_prose_wording": _contrast(records, field, {"before": .5, "after": .5}),
            "prose_minus_none_interaction": _contrast(records, field, {"before": .5, "after": .5, "none": -1.}),
            "placement_minus_none_interaction": {context: _contrast(records, field, {context: 1., "none": -1.})
                                                for context in ("before", "after")}}


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
                                  "missing": sum(r[key] is None for r in records)} for key in ("format_valid", "eos_valid", "capped")}}


def _aggregate(plan, records, results, attempted):
    completed = sum(r["status"] == "completed" for r in results.values())
    return {"schema_version": "a171-analysis-v1", "status": "complete" if completed == 96 else "incomplete",
            "planned_worlds": 8, "planned_cells": 96, "maximum_generation_calls": 96,
            "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                         "interrupted": len(attempted) - len(results), "unattempted": 96 - len(attempted),
                         "missing": 96 - len(results)},
            "overall": _counts(records),
            "by_context_wording": {context: {wording: _counts([r for r in records if r["context"] == context and r["wording"] == wording])
                                              for wording in WORDINGS} for context in CONTEXTS},
            "contrasts": {"strict_accuracy": _contrasts(records, "strict_correct"), "selected_label": _contrasts(records, "selected_label")},
            "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
            "claim_boundaries": {"fixed_synthetic_worlds_only": True, "confidence_intervals": False,
                "population_inference": False, "internal_mechanism_identified": False, "a169_reopened": False,
                "missing_imputed": False, "complete_case_analysis": False, "heldout_allowed": False,
                "adaptive_followup": False, "likelihood_collected": False}}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a171-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
            "settings": {"device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
                         "cpu_threads": 4, "seed": config["seed"], "max_new_tokens": MAX_NEW_TOKENS,
                         "do_sample": False, "num_beams": 1, "use_cache": True, "logits_to_keep": 1},
            "planned_cells": 96, "maximum_generation_calls": 96, "maximum_seconds": MAX_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a171-one-shot-v1", "pid": header["pid"],
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
                and finished["schema_version"] == "a171-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (float, int) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed"), "finished_receipt")
        interruption = finished["interruption"]
        if finished["status"] in ("interrupted", "deadline"):
            require(type(interruption) is dict and set(interruption) == {"signal_number", "signal_name", "pid", "exception_type"}
                    and interruption["pid"] == header["pid"] and interruption["exception_type"] in
                    ("A171Interrupted", "A171Deadline", "KeyboardInterrupt"), "interruption_receipt")
            allowed = {"A171Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       "A171Deadline": (signal.SIGALRM,), "KeyboardInterrupt": (None,)}
            number = interruption["signal_number"]
            require(number in allowed[interruption["exception_type"]] and interruption["signal_name"] ==
                    (signal.Signals(number).name if number is not None else None)
                    and (finished["status"] == "deadline") == (interruption["exception_type"] == "A171Deadline"), "actual_signal_receipt")
        else:
            require(interruption is None, "unexpected_interruption")
        if finished["status"] == "finished_schedule":
            require(len(results) == 96, "finished_schedule_coverage")
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
    raise A171Interrupted(signum)


def _deadline(signum, _frame):
    raise A171Deadline(signum)


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
    """One fresh model, all 96 scheduled cells, no outcome gate or retry."""
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
        with bounded_signals():
            try:
                inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
                plan, config, sources, tokenizer, eos, prepared = inputs
                require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"], "original_model_manifest")
                reference = reference_loader(reference_script)
                available = reference.require_memory()
                require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
                native.engine.write_private(root / "one-shot.json", {"schema_version": "a171-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
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
                    result = {"schema_version": "a171-result-v1", "attempt_sha256": object_sha(attempt), "status": "infrastructure_failed",
                              "generated_token_ids": None, "score": None, "error": None, "elapsed_seconds": 0.}
                    try:
                        tokens = generation_runner(runtime, prepared[trial["trial_id"]])
                        result.update(status="completed", generated_token_ids=tokens, score=score_tokens(runtime.tokenizer, tokens, trial, eos))
                        _validate_result(result, attempt, trial, tokenizer, eos)
                    except Exception as exc:
                        result.update(status="infrastructure_failed", generated_token_ids=None, score=None,
                                      error=native.engine.safe_error(exc, "a171_generation_failed"))
                    result["elapsed_seconds"] = time.monotonic() - tick
                    _validate_result(result, attempt, trial, tokenizer, eos)
                    native.engine.write_private(folder / "result.json", result)
                status = "finished_schedule"
            except (A171Interrupted, KeyboardInterrupt) as exc:
                error = exc
                status = "deadline" if isinstance(exc, A171Deadline) else "interrupted"
                interruption = _interruption(exc)
            except Exception as exc:
                error = exc
            finally:
                runtime = None
                if header is None:
                    native.engine.write_private(root / "startup-failure.json", {"schema_version": "a171-startup-failure-v1",
                        "startup_attempt_sha256": object_sha(startup), "status": status, "elapsed_seconds": time.monotonic() - started,
                        "interruption": interruption, "target_model_calls": 0,
                        "error": native.engine.safe_error(error, "a171_startup_failed") if error is not None else None})
                else:
                    if error is not None:
                        native.engine.write_private(root / "error.json", native.engine.safe_error(error, "a171_execution_failed"))
                    native.engine.write_private(root / "execution-finished.json", {"schema_version": "a171-execution-finished-v1", "status": status,
                        "elapsed_seconds": time.monotonic() - started, "run_sha256": object_sha(header), "interruption": interruption})
        if header is None:
            if error is not None:
                raise error
            raise RuntimeError("a171_startup_failed_without_exception")
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
    except (Exception, A171Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "a171_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
