"""A182 fixed-site cross-world selector transport; private immutable evidence.

Importing this module performs no model, tokenizer, network or data access.
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
import struct
import sys
import time
import uuid

from . import conditional_path_qualification as previous
from . import instruction_selection_answer_path as a169
from . import prefix_residual_intervention as apparatus

state, native, tasks, parent = previous.state, previous.native, previous.tasks, previous.parent
object_sha = state.object_sha
A179_SHA = "af0b811da83166db5a410956c1b7160e7fde297458f07d4b531a30586f8790e8"
A169_SHA = "1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e"
APPARATUS_SHA = "1df9515417ecbc7395c23a92f128d4ef13a1969801b6c7e9fad35ada841c643b"
CONFIG_SHA = previous.CONFIG_SHA
REFERENCE_SHA = previous.REFERENCE_SHA
PRIVATE_RUNS_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/a182")
MAX_SECONDS, KILL_GRACE_SECONDS = 7200, 60
MIN_AVAILABLE_BYTES = 48 * 1024**3
PREFIX_COUNT, LAYER_INDEX, MODEL_LAYERS, HIDDEN_WIDTH = 3, 15, 32, 4096
NUMERICAL_TOLERANCE_NATS, FUNCTIONAL_SEPARATION_NATS = 1e-4, 1e-3
JSON_PREFIX = '{"answer":"'
RECIPIENT_PAIRS = (("s112", "s113"), ("s114", "s115"))
DONOR_PAIRS = (("s116", "s117"), ("s118", "s119"))
SELECTOR_ORDERS = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"))
ARMS = ("no_patch", "self", "match", "opposite")
RECIPIENT_CANDIDATES, DONOR_CANDIDATES = ("R1", "R2", "D1", "D2"), ("D1", "D2")
USER_DIRECTIVE = "Select the entry labeled {user_selector} from the mapping."
teacher_force = a169.teacher_force
_PROCESS_CONSUMED = False


class A182Interrupted(BaseException):
    def __init__(self, signum):
        self.signum, self.pid = int(signum), os.getpid()
        self.signal_name = signal.Signals(signum).name
        super().__init__()


class A182Deadline(A182Interrupted):
    """Internal deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a182_" + code)


def _same(left, right):
    return tasks.canonical(left) == tasks.canonical(right)


def validate_sources():
    for module, digest in ((previous, A179_SHA), (a169, A169_SHA), (apparatus, APPARATUS_SHA)):
        require(native.engine.file_digest(Path(module.__file__)) == digest, "helper_source_drift")
    return {**previous.validate_sources(), Path(apparatus.__file__).name: APPARATUS_SHA,
            Path(__file__).name: native.engine.file_digest(Path(__file__))}


def system_instruction(selector):
    require(selector in ("A", "B"), "selector")
    return previous.previous.system_instruction("of_explicit_label", selector)


def compile_plan(config_sha256, protocol_sha256, tests_sha256):
    sources = validate_sources()
    require(config_sha256 == CONFIG_SHA and tasks.is_hash(protocol_sha256) and tasks.is_hash(tests_sha256), "plan_bindings")
    bindings = {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256, "tests_sha256": tests_sha256,
                "source_sha256": sources[Path(__file__).name], "a179_helper_sha256": A179_SHA,
                "a169_helper_sha256": A169_SHA, "apparatus_sha256": APPARATUS_SHA, "reference_script_sha256": REFERENCE_SHA}
    contexts, evaluations, lookup, references = [], [], {}, {}
    for kind in ("donor", "recipient"):
        for world_index, selectors in enumerate(SELECTOR_ORDERS):
            pair_index, swapped = world_index // 2, bool(world_index % 2)
            values = dict(zip(RECIPIENT_CANDIDATES, RECIPIENT_PAIRS[pair_index] + DONOR_PAIRS[pair_index]))
            names = DONOR_CANDIDATES if kind == "donor" else RECIPIENT_CANDIDATES[:2]
            order = tuple(reversed(names)) if (swapped != (kind == "donor")) else names
            mapping = {label: values[name] for label, name in zip(("A", "B"), order)}
            for selector_index, selector in enumerate(selectors):
                other_label = "B" if selector == "A" else "A"
                selected = next(name for name in names if values[name] == mapping[selector])
                other = next(name for name in names if name != selected)
                payload = json.dumps(mapping, separators=(",", ":"))
                if kind == "recipient":
                    payload += "\n\n" + USER_DIRECTIVE.format(user_selector=other_label)
                candidates = DONOR_CANDIDATES if kind == "donor" else RECIPIENT_CANDIDATES
                context = {"kind": kind, "world_index": world_index, "world_id": f"w{world_index:02d}",
                    "pair_index": pair_index, "assignment_swapped": swapped != (kind == "donor"), "presentation_order": "AB",
                    "base_context_index": 2 * world_index + selector_index, "selector": selector,
                    "user_selector": None if kind == "donor" else other_label,
                    "selected_candidate": selected, "other_candidate": other,
                    "selected_answer": mapping[selector], "unselected_answer": mapping[other_label],
                    "context_index": len(contexts), "mapping": mapping,
                    "messages": [{"role": "system", "content": system_instruction(selector)}, {"role": "user", "content": payload}],
                    "canonical_candidates": {name: json.dumps({"answer": values[name]}, separators=(",", ":")) for name in candidates}}
                context["context_id"] = object_sha({"schema_version": "a182-context-identity-v1", "bindings": bindings, "context": context})[:24]
                contexts.append(context)
                lookup[kind, world_index, selector] = context["context_id"]
                patched = ARMS[1:]
                rotation = world_index % 3
                patched = patched[rotation:] + patched[:rotation]
                if selector_index:
                    patched = tuple(reversed(patched))
                arms = ("baseline",) if kind == "donor" else ("no_patch", *patched)
                unique = candidates if world_index % 2 == 0 else tuple(reversed(candidates))
                candidate_order = unique + tuple(reversed(unique))
                for arm in arms:
                    source_id = None
                    if arm in ARMS[1:]:
                        source_context = context["context_id"] if arm == "self" else lookup["donor", world_index, selector if arm == "match" else other_label]
                        source_id = references[source_context]
                    repeats = dict.fromkeys(candidates, 0)
                    for candidate in candidate_order:
                        evaluation = {"context_index": context["context_index"], "context_id": context["context_id"],
                            "arm": arm, "candidate": candidate, "repeat_index": repeats[candidate],
                            "sequence_index": len(evaluations), "source_evaluation_id": source_id}
                        repeats[candidate] += 1
                        evaluation["evaluation_id"] = object_sha({"schema_version": "a182-evaluation-identity-v1", "bindings": bindings,
                                                                  "evaluation": evaluation})[:24]
                        evaluations.append(evaluation)
                        if arm in ("baseline", "no_patch") and candidate == candidates[0] and evaluation["repeat_index"] == 0:
                            references[context["context_id"]] = evaluation["evaluation_id"]
    require(len(contexts) == 16 and len(evaluations) == 288 and len({e["evaluation_id"] for e in evaluations}) == 288, "fixed_matrix")
    return {"schema_version": "a182-plan-v1", "partition": "fresh_public_cross_world_selector_transport", "bindings": bindings,
        "contexts": contexts, "evaluations": evaluations, "planned_contexts": 16, "planned_native_candidate_paths": 48,
        "planned_scored_contexts": 40, "planned_candidate_paths": 144, "planned_forward_calls": 288,
        "planned_donor_value_pairs": 2, "planned_recipient_value_pairs": 2, "planned_world_assignments": 4,
        "prefix_tokens": PREFIX_COUNT, "layer_index": LAYER_INDEX, "model_layers": MODEL_LAYERS, "hidden_width": HIDDEN_WIDTH,
        "numerical_tolerance_nats": NUMERICAL_TOLERANCE_NATS, "functional_separation_nats": FUNCTIONAL_SEPARATION_NATS,
        "primary": {"name": "match_minus_opposite", "planned_pairs": 8, "planned_arm_margins": 16, "planned_measurements": 64,
                    "candidate_cohort": ["R1", "R2"], "cross_arm_prefix_guard": True},
        "source_references": references, "capture_equality": "finite_elementwise_exact_float32",
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
    for candidate in context["canonical_candidates"]:
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
    prefix = next(iter(paths.values()))[:PREFIX_COUNT]
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
    require(type(metadata.get("hidden_size")) is int and metadata["hidden_size"] == HIDDEN_WIDTH
            and type(metadata.get("num_hidden_layers")) is int and metadata["num_hidden_layers"] == MODEL_LAYERS, "model_site_dimensions")
    require(type(limit) is int and limit > 0 and type(vocab) is int and vocab > 0, "model_dimensions")
    sources, tokenizer, eos = validate_sources(), tokenizer_loader(config), native.pinned_eos_ids(config)
    require(all(type(e) is int and 0 <= e < vocab for e in eos), "model_eos_vocabulary")
    prepared = {c["context_id"]: prepare_context(tokenizer, c, config, eos) for c in plan["contexts"]}
    for value in prepared.values():
        for target in value["candidate_token_ids"].values():
            require(len(value["prompt_token_ids"]) + len(target) <= limit, "model_context_fit")
            require(all(type(t) is int and 0 <= t < vocab for t in value["prompt_token_ids"] + target), "native_token_vocabulary")
        value["model_context_limit"], value["model_vocab_size"] = limit, vocab
        value["hidden_width"], value["model_layers"] = HIDDEN_WIDTH, MODEL_LAYERS
        value["absolute_position"] = len(value["prompt_token_ids"]) + PREFIX_COUNT - 1
    special = tokenizer.all_special_ids
    require(type(special) in (list, tuple) and all(type(t) is int for t in special), "special_inventory")
    require(all(not set(special).intersection(native.engine._ids(tokenizer.encode(m["content"], add_special_tokens=False)))
                for c in plan["contexts"] for m in c["messages"]), "payload_special_tokens")
    require(len({object_sha(v["prompt_token_ids"]) for v in prepared.values()}) == 16, "unique_native_contexts")
    require(len({object_sha(v["shared_json_prefix_token_ids"]) for v in prepared.values()}) == 1, "fixed_common_prefix_boundary")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path,
                                                reference_script, tokenizer_loader)
    return {"schema_version": "a182-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "protocol_sha256": plan["bindings"]["protocol_sha256"], "tests_sha256": plan["bindings"]["tests_sha256"],
            "source_hashes": sources, "reference_script_sha256": REFERENCE_SHA, "planned_contexts": 16, "planned_native_candidate_paths": 48,
            "planned_forward_calls": 288, "eos_token_ids": eos, "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _tri(values):
    values = list(values)
    require(bool(values) and all(v is None or type(v) is bool for v in values), "tristate_values")
    return False if any(v is False for v in values) else True if all(v is True for v in values) else None


def _capture_metadata(attempt, raw):
    _capture_values(raw)
    return {"schema_version": "a182-capture-v1", "evaluation_id": attempt["evaluation"]["evaluation_id"],
            "attempt_sha256": object_sha(attempt), "layer_index": LAYER_INDEX,
            "absolute_position": len(attempt["prompt_token_ids"]) + PREFIX_COUNT - 1,
            "shape": [1, HIDDEN_WIDTH], "dtype": "float32", "byte_order": "little",
            "bytes": 4 * HIDDEN_WIDTH, "sha256": tasks.sha(raw)}


def _capture_values(raw):
    require(type(raw) is bytes and len(raw) == HIDDEN_WIDTH * 4, "capture_byte_shape")
    values = struct.unpack('<' + 'f' * HIDDEN_WIDTH, raw)
    require(all(math.isfinite(value) for value in values), "capture_finite")
    return values


def _baseline_evaluations(plan, context_id):
    return [e for e in plan["evaluations"] if e["context_id"] == context_id and e["arm"] in ("baseline", "no_patch")]


def _source_eligibility(plan, context_id, results, captures):
    evaluations = _baseline_evaluations(plan, context_id)
    require(len(evaluations) in (4, 8), "source_baseline_cohort")
    flags, values = [], []
    for evaluation in evaluations:
        key = evaluation["evaluation_id"]
        result = results.get(key)
        if result is None:
            flags.append(None)
        elif result["status"] != "completed":
            flags.append(False)
        else:
            require(key in captures and result["capture"] is not None, "completed_capture_missing")
            raw = captures[key]
            require(tasks.sha(raw) == result["capture"]["sha256"], "capture_digest")
            values.append(_capture_values(raw))
            flags.append(True)
    unequal = any(value != values[0] for value in values[1:])
    equality = False if unequal else True if len(values) == len(evaluations) else None
    return _tri([*flags, equality])


def _source_binding(plan, evaluation, results, captures):
    reference = evaluation["source_evaluation_id"]
    if reference is None:
        return None
    source = next(e for e in plan["evaluations"] if e["evaluation_id"] == reference)
    context_id = source["context_id"]
    baseline = _baseline_evaluations(plan, context_id)
    result = results.get(reference)
    return {"source_evaluation_id": reference, "source_context_id": context_id,
            "eligibility": _source_eligibility(plan, context_id, results, captures),
            "baseline_result_sha256": {e["evaluation_id"]: object_sha(results[e["evaluation_id"]])
                if e["evaluation_id"] in results else None for e in baseline},
            "capture_sha256": object_sha(result["capture"]) if result is not None and result["capture"] is not None else None}


def _attempt(header, evaluation, prepared, source=None):
    value = prepared[evaluation["context_id"]]
    return {"schema_version": "a182-attempt-v1", "run_sha256": object_sha(header), "evaluation": evaluation,
        "prepared_sha256": object_sha(value), "prompt_token_ids": value["prompt_token_ids"],
        "target_token_ids": value["candidate_token_ids"][evaluation["candidate"]],
        "json_prefix_token_count": PREFIX_COUNT, "intended_eos_token_id": value["intended_eos_token_id"], "source": source}


def _invocation(attempt):
    return {"schema_version": "a182-invocation-v1", "attempt_sha256": object_sha(attempt), "model_forward_entered": True}


def _measurements(plan, context_id, arm, results):
    context = next(c for c in plan["contexts"] if c["context_id"] == context_id)
    values = {name: [None, None] for name in context["canonical_candidates"]}
    for evaluation in plan["evaluations"]:
        if evaluation["context_id"] == context_id and evaluation["arm"] == arm:
            result = results.get(evaluation["evaluation_id"])
            if result is not None and result["status"] == "completed":
                values[evaluation["candidate"]][evaluation["repeat_index"]] = result["measurement"]
    return values


def _prefix_guard(measurements):
    complete = all(m is not None for m in measurements)
    spread = max(m["scores"]["json_prefix_logprob"] for m in measurements) - min(m["scores"]["json_prefix_logprob"] for m in measurements) if complete else None
    return {"planned_measurements": len(measurements), "completed_measurements": sum(m is not None for m in measurements),
            "range_nats": spread, "passed": spread <= NUMERICAL_TOLERANCE_NATS if complete else None}


def _cohort(values, candidates, selected, other):
    measurements = [m for name in candidates for m in values[name]]
    prefix = _prefix_guard(measurements)
    drift = {name: abs(pair[1]["scores"]["answer_continuation_logprob"] - pair[0]["scores"]["answer_continuation_logprob"])
             if all(m is not None for m in pair) else None for name in candidates for pair in [values[name]]}
    guards = {name: value <= NUMERICAL_TOLERANCE_NATS if value is not None else None for name, value in drift.items()}
    numerical = _tri([prefix["passed"], *guards.values()])
    means, mean, worst, best = dict.fromkeys(candidates), None, None, None
    if numerical is True:
        scores = {name: [m["scores"]["answer_continuation_logprob"] for m in values[name]] for name in candidates}
        means = {name: math.fsum(v) / 2 for name, v in scores.items()}
        mean = means[selected] - means[other]
        worst, best = min(scores[selected]) - max(scores[other]), max(scores[selected]) - min(scores[other])
    return {"candidates": list(candidates), "planned_measurements": len(measurements),
            "completed_measurements": prefix["completed_measurements"], "complete": prefix["completed_measurements"] == len(measurements),
            "prefix_guard": prefix, "repeat_drift_nats": drift, "repeat_guard_passed": guards, "numerical_valid": numerical,
            "candidate_mean_scores": means, "mean_margin_nats": mean, "worst_margin_nats": worst, "best_margin_nats": best,
            "mean_tie": mean == 0. if mean is not None else None, "mean_reversed": mean < 0. if mean is not None else None}


def _functional(cohort, separation):
    if cohort["numerical_valid"] is not True:
        separation = None
    return {"numerical_valid": cohort["numerical_valid"], "separation_nats": separation,
            "functional_passed": separation > FUNCTIONAL_SEPARATION_NATS if separation is not None else None,
            "threshold_boundary": separation == FUNCTIONAL_SEPARATION_NATS if separation is not None else None,
            "tie": separation == 0. if separation is not None else None,
            "reversed": separation < 0. if separation is not None else None,
            "qualified": _tri([cohort["numerical_valid"], separation > FUNCTIONAL_SEPARATION_NATS if separation is not None else None])}


def _records(plan, prepared, results, captures):
    donors, recipients = [], []
    for context in plan["contexts"]:
        cid = context["context_id"]
        row = {key: context[key] for key in ("context_id", "context_index", "kind", "world_index", "pair_index",
                                             "base_context_index", "selector", "selected_candidate", "other_candidate")}
        row["source_eligible"] = _source_eligibility(plan, cid, results, captures)
        selected, other = context["selected_candidate"], context["other_candidate"]
        if context["kind"] == "donor":
            values = _measurements(plan, cid, "baseline", results)
            cohort = _cohort(values, DONOR_CANDIDATES, selected, other)
            row.update(own_value=cohort, competence=_functional(cohort, cohort["worst_margin_nats"]))
            donors.append(row)
            continue
        values = {arm: _measurements(plan, cid, arm, results) for arm in ARMS}
        arms = {}
        for arm, measurements in values.items():
            own = _cohort(measurements, ("R1", "R2"), selected, other)
            four = _cohort(measurements, RECIPIENT_CANDIDATES, selected, other)
            expected = selected if arm == "match" else other if arm == "opposite" else None
            separation = None
            if expected is not None and four["numerical_valid"] is True:
                separation = min(m["scores"]["answer_continuation_logprob"] for m in measurements[expected]) - max(
                    m["scores"]["answer_continuation_logprob"] for name, pair in measurements.items() if name != expected for m in pair)
            arms[arm] = {"own_value": own, "four_path": four, "expected_recipient_candidate": expected,
                         "recipient_content": _functional(four, separation) if expected is not None else None}
        primary_prefix = _prefix_guard([m for arm in ("match", "opposite") for name in ("R1", "R2") for m in values[arm][name]])
        local_valid = _tri([arms[arm]["own_value"]["numerical_valid"] for arm in ("match", "opposite")] + [primary_prefix["passed"]])
        difference = arms["match"]["own_value"]["mean_margin_nats"] - arms["opposite"]["own_value"]["mean_margin_nats"] if local_valid is True else None
        cross_prefix = _prefix_guard([m for arm in ARMS for name in RECIPIENT_CANDIDATES for m in values[arm][name]])
        self_drift = {name: [abs(left["scores"]["answer_continuation_logprob"] - right["scores"]["answer_continuation_logprob"])
            if left is not None and right is not None else None for left, right in zip(values["self"][name], values["no_patch"][name])]
            for name in RECIPIENT_CANDIDATES}
        self_flags = [v <= NUMERICAL_TOLERANCE_NATS if v is not None else None for pair in self_drift.values() for v in pair]
        baseline = arms["no_patch"]["own_value"]
        row.update(arms=arms, conflict=_functional(baseline, -baseline["best_margin_nats"] if baseline["best_margin_nats"] is not None else None),
                   primary={"prefix_guard": primary_prefix, "numerical_valid": local_valid, "difference_nats": difference},
                   full_cross_arm_prefix_guard=cross_prefix,
                   self_no_patch={"drift_nats": self_drift, "passed": _tri(self_flags)})
        recipients.append(row)
    return {"schema_version": "a182-records-v1", "donors": donors, "recipients": recipients}


_flag_counts = previous._flag_counts


def _gate(rows):
    return {"planned": len(rows), "numerical_valid": _flag_counts([r["numerical_valid"] for r in rows]),
            "functional_passed": _flag_counts([r["functional_passed"] for r in rows]),
            "qualified": _tri([r["qualified"] for r in rows])}


def _aggregate(plan, records, results, attempted, invoked):
    donors, recipients = records["donors"], records["recipients"]
    require(len(donors) == len(recipients) == 8, "fixed_record_count")
    completed = sum(r["status"] == "completed" for r in results.values())
    differences = [r["primary"]["difference_nats"] for r in recipients]
    point = math.fsum(differences) / 8 if all(v is not None for v in differences) else None
    primary_valid = _tri([r["primary"]["numerical_valid"] for r in recipients])
    own = [r["arms"][arm]["own_value"] for r in recipients for arm in ("match", "opposite")]
    primary = {"planned_pairs": 8, "resolved_pairs": sum(v is not None for v in differences),
        "planned_arm_margins": 16, "resolved_arm_margins": sum(c["mean_margin_nats"] is not None for c in own),
        "planned_measurements": 64, "completed_measurements": sum(c["completed_measurements"] for c in own),
        "numerical_valid": primary_valid, "point": point, "lower": point, "upper": point, "unbounded": point is None}
    source = [r["source_eligible"] for r in donors + recipients]
    four = [r["arms"][arm]["four_path"]["numerical_valid"] for r in recipients for arm in ARMS]
    prefix = [r["full_cross_arm_prefix_guard"]["passed"] for r in recipients]
    identity = [r["self_no_patch"]["passed"] for r in recipients]
    apparatus_ok = _tri(source + four + prefix + identity)
    gates = {"donor_competence": _gate([r["competence"] for r in donors]),
             "recipient_conflict": _gate([r["conflict"] for r in recipients]),
             "recipient_content": _gate([r["arms"][arm]["recipient_content"] for r in recipients for arm in ("match", "opposite")])}
    positive = point > 0 if point is not None else None
    return {"schema_version": "a182-analysis-v1", "status": "complete" if completed == 288 else "incomplete",
        "planned_contexts": 16, "planned_native_candidate_paths": 48, "planned_scored_contexts": 40,
        "planned_candidate_paths": 144, "planned_forward_calls": 288,
        "coverage": {"attempted": len(attempted), "processed_slots": len(results), "forward_dispatch_entries": len(invoked),
            "completed": completed, "infrastructure_failed": sum(r["status"] == "infrastructure_failed" for r in results.values()),
            "dependency_unavailable": sum(r["status"] == "dependency_unavailable" for r in results.values()),
            "interrupted": len(attempted) - len(results), "unattempted": 288 - len(attempted), "missing": 288 - completed},
        "primary": primary, "apparatus": {"source_eligible": _flag_counts(source), "four_path_numerical_valid": _flag_counts(four),
            "full_cross_arm_prefix_guard": _flag_counts(prefix), "self_no_patch": _flag_counts(identity), "qualified": apparatus_ok},
        "functional_gates": gates, "qualified_transport": _tri([primary_valid, positive, apparatus_ok, *[g["qualified"] for g in gates.values()]]),
        "numeric_records_sha256": object_sha(records), "plan_object_sha256": object_sha(plan),
        "claim_boundaries": {"canonical_path_conditional_preference_only": True, "suffix_includes_eos": True,
            "saved_normalizers_not_independently_recomputed": True, "generation_accuracy": False, "scaffold_mediation": False,
            "abstract_selector_transport_established": False, "population_inference": False, "confidence_intervals": False,
            "complete_case_substitution": False, "missing_imputed": False, "adaptive_tolerance": False}}

def _validate_result(result, attempt):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "measurement", "capture", "error", "elapsed_seconds"}
            and result["schema_version"] == "a182-result-v1" and result["attempt_sha256"] == object_sha(attempt)
            and type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"])
            and result["elapsed_seconds"] >= 0, "result_binding")
    if attempt["source"] is not None and attempt["source"]["eligibility"] is not True:
        require(result["status"] == "dependency_unavailable", "blocked_source_status")
    if result["status"] == "completed":
        value = result["measurement"]
        require(result["error"] is None and type(value) is dict and set(value) == {"chosen_logits", "log_normalizers", "token_logprobs", "scores"}, "measurement_schema")
        require(all(type(value[key]) is list and len(value[key]) == len(attempt["target_token_ids"])
                    and all(type(v) in (float, int) and math.isfinite(v) for v in value[key])
                    for key in ("chosen_logits", "log_normalizers", "token_logprobs")), "score_length_or_finiteness")
        require(value["token_logprobs"] == [a - b for a, b in zip(value["chosen_logits"], value["log_normalizers"])], "normalization_recomputation")
        require(_same(value["scores"], a169._scores(value["token_logprobs"], PREFIX_COUNT)), "score_recomputation")
        baseline = attempt["evaluation"]["arm"] in ("baseline", "no_patch")
        require((type(result["capture"]) is dict) if baseline else result["capture"] is None, "capture_presence")
        require(attempt["source"] is None or attempt["source"]["eligibility"] is True, "completed_dependency")
    else:
        require(result["measurement"] is None and result["capture"] is None, "failed_measurement_absent")
        if result["status"] == "dependency_unavailable":
            require(result["error"] is None and attempt["source"] is not None
                    and attempt["source"]["eligibility"] is not True, "dependency_receipt")
        else:
            require(result["status"] == "infrastructure_failed" and type(result["error"]) is dict
                    and set(result["error"]) == {"exception_type", "frames", "code"}, "failed_result_schema")


def _write_capture(path, raw):
    """Immutable, owner-only binary publication; no partially written receipt."""
    _capture_values(raw)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not path.exists(), "capture_already_published")
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _hook_audit(model, torch, allowed=None):
    allowed = {} if allowed is None else allowed
    global_hooks = torch.nn.modules.module
    require(not global_hooks._global_forward_hooks and not global_hooks._global_forward_pre_hooks, "global_forward_hooks")
    for module in model.modules():
        expected_pre, expected_post = allowed.get(module, (set(), set()))
        require(set(module._forward_pre_hooks) == expected_pre and set(module._forward_hooks) == expected_post, "model_forward_hooks")


def _runtime_block(runtime):
    model, torch = runtime.model, runtime.torch
    require(isinstance(model, torch.nn.Module) and not model.training, "runtime_model")
    config = model.config
    require(type(config.hidden_size) is int and config.hidden_size == HIDDEN_WIDTH
            and type(config.num_hidden_layers) is int and config.num_hidden_layers == MODEL_LAYERS
            and config.model_type == "llama", "runtime_site_dimensions")
    require(len(model.model.layers) == MODEL_LAYERS and len({id(block) for block in model.model.layers}) == MODEL_LAYERS, "runtime_layers")
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()), "runtime_cpu_fp32")
    _hook_audit(model, torch)
    return model.model.layers[LAYER_INDEX]


def measure(runtime, attempt, invocation_callback, *, replacement=None, forward_scorer=teacher_force):
    """One dispatch entry and exactly one post-block hook; no extraction pass."""
    model, torch = runtime.model, runtime.torch
    block = _runtime_block(runtime)
    site = apparatus.PrefixResidualSite(len(attempt["prompt_token_ids"]), len(attempt["target_token_ids"]),
                PREFIX_COUNT, HIDDEN_WIDTH, len(attempt["prompt_token_ids"]) + PREFIX_COUNT - 1)
    tensor = None if replacement is None else torch.tensor([_capture_values(replacement)], dtype=torch.float32, device="cpu")
    entered = 0
    handle = None
    hook = apparatus.PrefixResidualIntervention(block, site, replacement=tensor)
    def dispatch(_module, _args):
        nonlocal entered
        entered += 1
        require(entered == 1, "model_dispatch_count")
        _hook_audit(model, torch, {model: ({handle.id}, set()), block: (set(), {hook._handle.id})})
        invocation_callback()
    try:
        handle = model.register_forward_pre_hook(dispatch)
        with hook:
            measurement = forward_scorer(model, torch, attempt["prompt_token_ids"], attempt["target_token_ids"], PREFIX_COUNT, runtime.device)
        require(entered == 1, "model_dispatch_count")
        # Values round-trip exactly through Python float and explicit LE float32.
        raw = struct.pack('<' + 'f' * HIDDEN_WIDTH, *hook.capture[0].tolist())
        _capture_values(raw)
        return measurement, raw
    finally:
        exceptional = sys.exc_info()[0] is not None
        if handle is not None:
            handle.remove()
        try:
            _hook_audit(model, torch)
        except BaseException:
            if not exceptional:
                raise
            # Preserve the actual signal/SystemExit/readout exception; failed
            # operations cannot publish a capture and the next audit rejects
            # any unexpected remaining hook before another forward.

def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a182-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
            "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
            "tests_sha256": plan["bindings"]["tests_sha256"], "reference_script_sha256": REFERENCE_SHA,
            "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
            "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
            "settings": {"device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
                         "cpu_threads": 4, "seed": config["seed"], "use_cache": False,
                         "logits_to_keep": "target_count_plus_one", "padding": False},
            "planned_contexts": 16, "planned_native_candidate_paths": 48, "planned_scored_contexts": 40,
            "layer_index": LAYER_INDEX, "hidden_width": HIDDEN_WIDTH, "planned_forward_calls": 288, "maximum_seconds": MAX_SECONDS,
            "prefix_tokens": PREFIX_COUNT, "numerical_tolerance_nats": NUMERICAL_TOLERANCE_NATS,
            "functional_separation_nats": FUNCTIONAL_SEPARATION_NATS, "suffix_includes_eos": True,
            "normalization_arithmetic": "float64_logsumexp_of_float32_logits",
            "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "resume_allowed": False, "heldout_allowed": False}

def _private_root(path):
    require(path.is_absolute() and path == Path(os.path.abspath(path))
            and all(not p.is_symlink() for p in (path, *path.parents)), "private_path")
    require(path.is_relative_to(PRIVATE_RUNS_ROOT) and path != PRIVATE_RUNS_ROOT, "a182_private_run_root")
    require(not path.is_relative_to(Path(__file__).absolute().parents[2]), "private_outside_source")
    return path


def _startup_attempt(pid, plan_sha256, config_sha256):
    return {"schema_version": "a182-startup-attempt-v1", "pid": pid, "plan_sha256": plan_sha256,
            "config_sha256": config_sha256, "source_sha256": native.engine.file_digest(Path(__file__))}


def _interruption(exc):
    return {"signal_number": exc.signum if isinstance(exc, A182Interrupted) else None,
            "signal_name": exc.signal_name if isinstance(exc, A182Interrupted) else None,
            "pid": exc.pid if isinstance(exc, A182Interrupted) else os.getpid(), "exception_type": type(exc).__name__}




def _interrupt(signum, _frame):
    raise A182Interrupted(signum)


def _deadline(signum, _frame):
    raise A182Deadline(signum)


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


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / 'run.json')
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, 'pid', 'available_memory_bytes'} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header['pid']) is int and header['pid'] > 0 and type(header['available_memory_bytes']) is int
            and header['available_memory_bytes'] >= MIN_AVAILABLE_BYTES, 'run_lineage')
    require(_same(state._read(root / 'one-shot.json'), {'schema_version': 'a182-one-shot-v1', 'pid': header['pid'],
            'plan_sha256': plan_sha256}), 'one_shot_binding')
    require(_same(state._read(root / 'startup-attempt.json'), _startup_attempt(header['pid'], plan_sha256, config_sha256))
            and not (root / 'startup-failure.json').exists(), 'startup_lineage')
    require(_same(state._read(root / 'native-inputs.json'), prepared), 'native_input_replay')
    folders, expected = root / 'evaluations', {e['evaluation_id'] for e in plan['evaluations']}
    if folders.exists():
        require(not folders.is_symlink() and folders.is_dir() and all(p.name in expected and p.is_dir()
                and not p.is_symlink() for p in folders.iterdir()), 'unknown_evaluation')
    results, captures, attempted, invoked, closed = {}, {}, set(), set(), False
    for evaluation in plan['evaluations']:
        key = evaluation['evaluation_id']
        folder = folders / key
        if not folder.exists():
            closed = True
            continue
        require(not closed and {p.name for p in folder.iterdir()} <= {'attempt.json', 'result.json', 'invocation.json', 'capture.fp32'}
                and (folder / 'attempt.json').is_file() and all(p.is_file() and not p.is_symlink() for p in folder.iterdir()), 'attempt_sequence')
        source = _source_binding(plan, evaluation, results, captures)
        attempt = state._read(folder / 'attempt.json')
        require(_same(attempt, _attempt(header, evaluation, prepared, source)), 'attempt_native_replay')
        attempted.add(key)
        if (folder / 'invocation.json').exists():
            require(_same(state._read(folder / 'invocation.json'), _invocation(attempt))
                    and (source is None or source['eligibility'] is True), 'invocation_binding')
            invoked.add(key)
        if not (folder / 'result.json').exists():
            # A published capture without a completed result is never reusable.
            if (folder / 'capture.fp32').exists():
                require(key in invoked and evaluation['arm'] in ('baseline', 'no_patch'), 'orphan_capture_provenance')
                _capture_values((folder / 'capture.fp32').read_bytes())
            closed = True
            continue
        result = state._read(folder / 'result.json')
        _validate_result(result, attempt)
        if result['status'] == 'completed':
            require(key in invoked, 'completed_dispatch_missing')
            if evaluation['arm'] in ('baseline', 'no_patch'):
                raw = (folder / 'capture.fp32').read_bytes()
                require(_same(result['capture'], _capture_metadata(attempt, raw)), 'capture_metadata_binding')
                captures[key] = raw
            else:
                require(not (folder / 'capture.fp32').exists(), 'unexpected_patch_capture')
        else:
            require(not (folder / 'capture.fp32').exists(), 'failed_capture_reuse')
            if result['status'] == 'dependency_unavailable':
                require(key not in invoked, 'blocked_dispatch')
        results[key] = result
    finished = None
    if (root / 'execution-finished.json').exists():
        finished = state._read(root / 'execution-finished.json')
        require(type(finished) is dict and set(finished) == {'schema_version', 'status', 'elapsed_seconds', 'run_sha256', 'interruption'}
                and finished['schema_version'] == 'a182-execution-finished-v1' and finished['run_sha256'] == object_sha(header)
                and type(finished['elapsed_seconds']) in (float, int) and math.isfinite(finished['elapsed_seconds'])
                and finished['elapsed_seconds'] >= 0 and finished['status'] in ('finished_schedule', 'interrupted', 'deadline', 'failed'), 'finished_receipt')
        interruption = finished['interruption']
        if finished['status'] in ('interrupted', 'deadline'):
            require(type(interruption) is dict and set(interruption) == {'signal_number', 'signal_name', 'pid', 'exception_type'}
                    and type(interruption['pid']) is int and interruption['pid'] == header['pid'] and interruption['exception_type'] in
                    ('A182Interrupted', 'A182Deadline', 'KeyboardInterrupt'), 'interruption_receipt')
            allowed = {'A182Interrupted': (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
                       'A182Deadline': (signal.SIGALRM,), 'KeyboardInterrupt': (None,)}
            number = interruption['signal_number']
            require((number is None or type(number) is int) and number in allowed[interruption['exception_type']] and interruption['signal_name'] ==
                    (signal.Signals(number).name if number is not None else None)
                    and (finished['status'] == 'deadline') == (interruption['exception_type'] == 'A182Deadline'), 'actual_signal_receipt')
        else:
            require(interruption is None, 'unexpected_interruption')
        if finished['status'] == 'finished_schedule':
            require(len(results) == 288, 'finished_schedule_coverage')
    records = _records(plan, prepared, results, captures)
    aggregate = _aggregate(plan, records, results, attempted, invoked)
    aggregate['execution_status'] = finished['status'] if finished else 'receipt_missing'
    aggregate['provenance'] = {key: header[key] for key in ('plan_sha256', 'config_sha256', 'source_hashes', 'tests_sha256',
        'protocol_sha256', 'native_prepared_sha256', 'model_manifest_sha256')}
    aggregate['provenance']['run_sha256'] = object_sha(header)
    for name, value in (('records.json', records), ('analysis.json', aggregate)):
        if (root / name).exists():
            require(_same(state._read(root / name), value), 'stored_export_drift')
    return {'records': records, 'aggregate': aggregate}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    root = _private_root(output_root)
    with (root / '.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, *inputs, plan_sha256, config_sha256, reference_script)


def run_plan(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
             output_root, reference_loader=parent.load_reference, tokenizer_loader=native.load_tokenizer,
             forward_scorer=teacher_force, measurement_runner=measure):
    """One fresh model; at most 288 dispatch entries, no extraction or retry."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, 'fresh_process_required')
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONOPTIMIZE='0')
    root = _private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / '.lock').open('a') as lock:
        os.chmod(root / '.lock', 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / name).exists() for name in ('startup-attempt.json', 'startup-failure.json', 'one-shot.json', 'run.json', 'evaluations')), 'consumed_run')
        startup = _startup_attempt(os.getpid(), plan_sha256, config_sha256)
        native.engine.write_private(root / 'startup-attempt.json', startup)
        _PROCESS_CONSUMED = True
        header, status, interruption, runtime, error = None, 'failed', None, None, None
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(bounded_signals())
                inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
                plan, config, sources, tokenizer, eos, prepared = inputs
                require(native.engine.snapshot_manifest(Path(config['model_path'])) == config['model_files_sha256'], 'original_model_manifest')
                reference = reference_loader(reference_script)
                available = reference.require_memory()
                require(type(available) is int and available >= MIN_AVAILABLE_BYTES, 'preload_memory_gate')
                native.engine.write_private(root / 'one-shot.json', {'schema_version': 'a182-one-shot-v1', 'pid': os.getpid(), 'plan_sha256': plan_sha256})
                pending_header = {**_fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script),
                                  'pid': os.getpid(), 'available_memory_bytes': available}
                native.engine.write_private(root / 'run.json', pending_header)
                native.engine.write_private(root / 'native-inputs.json', prepared)
                header = pending_header
                runtime = reference.load_cpu_reference(config, native.engine)
                require(runtime.eos_ids == eos and str(runtime.device) == 'cpu'
                        and tasks.sha(runtime.tokenizer.get_chat_template().encode()) == config['chat_template_sha256'], 'runtime_native_binding')
                results, captures = {}, {}
                for evaluation in plan['evaluations']:
                    source = _source_binding(plan, evaluation, results, captures)
                    attempt = _attempt(header, evaluation, prepared, source)
                    key = evaluation['evaluation_id']
                    folder = root / 'evaluations' / key
                    native.engine.write_private(folder / 'attempt.json', attempt)
                    tick = time.monotonic()
                    result = {'schema_version': 'a182-result-v1', 'attempt_sha256': object_sha(attempt), 'status': 'infrastructure_failed',
                              'measurement': None, 'capture': None, 'error': None, 'elapsed_seconds': 0.}
                    raw = None
                    if source is not None and source['eligibility'] is not True:
                        result['status'] = 'dependency_unavailable'
                    else:
                        entered = False
                        def dispatch_entry():
                            nonlocal entered
                            require(not entered, 'repeated_dispatch_entry')
                            native.engine.write_private(folder / 'invocation.json', _invocation(attempt))
                            entered = True
                        replacement = captures[source['source_evaluation_id']] if source is not None else None
                        try:
                            measurement, capture = measurement_runner(runtime, attempt, dispatch_entry,
                                                                      replacement=replacement, forward_scorer=forward_scorer)
                            require(entered, 'missing_dispatch_entry')
                            baseline = evaluation['arm'] in ('baseline', 'no_patch')
                            result.update(measurement=measurement, status='completed',
                                          capture=_capture_metadata(attempt, capture) if baseline else None)
                            _validate_result(result, attempt)
                            raw = capture if baseline else None
                        except Exception as exc:
                            result.update(status='infrastructure_failed', measurement=None, capture=None,
                                          error=native.engine.safe_error(exc, 'a182_measurement_failed'))
                    result['elapsed_seconds'] = time.monotonic() - tick
                    _validate_result(result, attempt)
                    if raw is not None:
                        _write_capture(folder / 'capture.fp32', raw)
                    native.engine.write_private(folder / 'result.json', result)
                    results[key] = result
                    if raw is not None:
                        captures[key] = raw
                status = 'finished_schedule'
            except (A182Interrupted, KeyboardInterrupt) as exc:
                error = exc
                status = 'deadline' if isinstance(exc, A182Deadline) else 'interrupted'
                interruption = _interruption(exc)
            except Exception as exc:
                error = exc
            finally:
                runtime = None
                if header is None:
                    native.engine.write_private(root / 'startup-failure.json', {'schema_version': 'a182-startup-failure-v1',
                        'startup_attempt_sha256': object_sha(startup), 'status': status, 'elapsed_seconds': time.monotonic() - started,
                        'interruption': interruption, 'target_model_calls': 0,
                        'error': native.engine.safe_error(error, 'a182_startup_failed') if error is not None else None})
                else:
                    if error is not None:
                        native.engine.write_private(root / 'error.json', native.engine.safe_error(error, 'a182_execution_failed'))
                    native.engine.write_private(root / 'execution-finished.json', {'schema_version': 'a182-execution-finished-v1', 'status': status,
                        'elapsed_seconds': time.monotonic() - started, 'run_sha256': object_sha(header), 'interruption': interruption})
        if header is None:
            if error is not None:
                raise error
            raise RuntimeError('a182_startup_failed_without_exception')
        exported = _replay(root, *inputs, plan_sha256, config_sha256, reference_script)
        native.engine.write_private(root / 'records.json', exported['records'])
        native.engine.write_private(root / 'analysis.json', exported['aggregate'])
        return exported['aggregate']


run = run_plan
export = export_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'export'), default='run')
    for name in ('plan', 'config', 'protocol', 'reference-script', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('plan-sha256', 'config-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    kwargs = {'plan_path': args.plan, 'plan_sha256': args.plan_sha256, 'config_path': args.config,
              'config_sha256': args.config_sha256, 'protocol_path': args.protocol, 'reference_script': args.reference_script,
              'output_root': args.output_root}
    try:
        root = _private_root(args.output_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / ('execution.log' if args.mode == 'run' else 'replay.log')).open('x', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                aggregate = run_plan(**kwargs) if args.mode == 'run' else export_run(**kwargs)['aggregate']
        print(json.dumps({key: aggregate[key] for key in ('status', 'execution_status', 'coverage')}, sort_keys=True, allow_nan=False))
        return 0 if aggregate['status'] == 'complete' and aggregate['execution_status'] == 'finished_schedule' else 1
    except (Exception, A182Interrupted, KeyboardInterrupt) as exc:
        print(json.dumps({'status': 'a182_failed', 'error_type': type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
