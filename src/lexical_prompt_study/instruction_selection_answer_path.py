"""A169 fixed teacher-forced answer-path assay; no model calls at import.

All inputs and token-level receipts are private return objects. The supplied
canonical JSON syntax is conditioning, not recovered free-generation competence.
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

from . import instruction_selection_label_rename as lineage

prose, state, native, tasks, parent = lineage.prose, lineage.state, lineage.native, lineage.tasks, lineage.parent
REFERENCE_SHA = lineage.REFERENCE_SHA
LINEAGE_SHA = "75ff1467f09ee14249e6a2ee112067126f141a10899672446c8fefa8ecf2fb77"
MAX_SECONDS = 7200
KILL_GRACE_SECONDS = 60
PREFIX_ABSOLUTE_TOLERANCE_NATS = 1e-4
MIN_AVAILABLE_BYTES = state.MIN_AVAILABLE_BYTES
KINDS = ("selected_value", "other_value", "selected_label", "other_label")
PHASES = ("baseline", "full", "sham", "inert", "prose")
MATERIAL_ORDERS = (("full", "sham", "prose", "inert"), ("sham", "inert", "full", "prose"),
                   ("inert", "prose", "sham", "full"), ("prose", "full", "inert", "sham"))
JSON_PREFIX = '{"answer":"'
object_sha = state.object_sha
_PROCESS_CONSUMED = False


class A169Interrupted(BaseException):
    """Signals bypass recoverable candidate errors."""


class A169Deadline(BaseException):
    """Wall deadline includes preflight and model loading."""


def require(condition, code):
    tasks.require(condition, "a169_" + code)


def validate_sources():
    require(native.engine.file_digest(Path(lineage.__file__)) == LINEAGE_SHA, "a168_source_drift")
    return {**lineage.validate_sources(), Path(lineage.__file__).name: LINEAGE_SHA,
            Path(__file__).name: native.engine.file_digest(Path(__file__))}


def _build_plan(a167_plan, a167_run_header, protocol_sha256):
    validate_sources()
    lineage._validate_parent(a167_plan, a167_run_header)
    require(tasks.is_hash(protocol_sha256), "protocol_hash")
    a165 = a167_plan["a165_plan"]
    original = [t for t in a167_plan["trials"] if t["diagnostic_phase"] == "baseline"
                and t["placement_index"] == 0]
    require(len(original) == 8 and {t["world_index"] for t in original} == set(range(4)), "fixed_worlds")
    lookup = {(t["world_id"], t["selector"], t["scaffold_kind"], t["placement"]): t for t in a165["trials"]}
    prose_lookup = {(t["world_id"], t["selector"], t["placement"]): t for t in a167_plan["trials"]
                    if t["diagnostic_phase"] == "prose"}
    bindings = {"protocol_sha256": protocol_sha256, "a169_source_sha256": native.engine.file_digest(Path(__file__)),
        "a167_plan_sha256": object_sha(a167_plan), "a167_run_sha256": object_sha(a167_run_header),
        "config_sha256": a167_run_header["config_sha256"], "reference_script_sha256": REFERENCE_SHA}
    trials = []
    # Build the exact parent contexts, then apply the prospective balanced schedule.
    for phase in PHASES:
        for base in original:
            for placement in (("none",) if phase == "baseline" else ("before", "after")):
                source = (prose_lookup[base["world_id"], base["selector"], placement] if phase == "prose" else
                          lookup[base["world_id"], base["selector"], "none" if phase == "baseline" else phase, placement])
                row = {key: copy.deepcopy(source[key]) for key in native.META if key != "scaffold"}
                row.update({key: copy.deepcopy(source[key]) for key in
                            ("messages", "world", "selected_answer", "unselected_answer", "scaffold_kind")})
                row.pop("trial_id")
                row.update(diagnostic_phase=phase, world_index=base["world_index"], selector_index=base["selector_index"],
                           source_study="a167" if phase == "prose" else "a165", source_trial_id=source["trial_id"])
                values = (row["selected_answer"], row["unselected_answer"], row["selector"],
                          "B" if row["selector"] == "A" else "A")
                require(len(set(values)) == 4, "distinct_candidates")
                row["canonical_candidates"] = {kind: json.dumps({"answer": value}, ensure_ascii=True, separators=(",", ":"))
                                               for kind, value in zip(KINDS, values)}
                row["trial_id"] = object_sha({"schema_version": "a169-context-identity-v1", "bindings": bindings, "trial": row})[:24]
                trials.append(row)
    def order_key(t):
        if t["diagnostic_phase"] == "baseline":
            return (0, t["world_index"], t["selector_index"], 0, 0)
        placement_index = ("before", "after").index(t["placement"])
        order = MATERIAL_ORDERS[(t["world_index"] + t["selector_index"] + placement_index) % 4]
        return (1, t["world_index"], t["selector_index"], placement_index, order.index(t["diagnostic_phase"]))
    trials.sort(key=order_key)
    require(len(trials) == len({object_sha(t["messages"]) for t in trials}) == 72, "unique_contexts")
    evaluations = [{"evaluation_id": object_sha({"schema_version": "a169-evaluation-identity-v1",
        "trial_id": trial["trial_id"], "candidate_kind": kind})[:24], "trial_id": trial["trial_id"],
        "candidate_kind": kind, "context_index": index, "sequence_index": index * 4 + kind_index}
        for index, trial in enumerate(trials) for kind_index, kind in enumerate(KINDS)]
    return {"schema_version": "a169-plan-v1", "partition": "development", "stage": "teacher_forced_answer_path",
        "world_count": 4, "trial_count": 72, "maximum_candidate_evaluations": 288, "baseline_candidate_evaluations": 32,
        "trials": trials, "evaluations": evaluations, "bindings": bindings, "a167_plan": copy.deepcopy(a167_plan),
        "a167_run_header": copy.deepcopy(a167_run_header), "free_generation_allowed": False, "heldout_allowed": False,
        "canonical_json_prefix": JSON_PREFIX, "eos_rule": "native_completed_assistant_single_token",
        "gate_rule": "all_8_selected_value_joint_logprob_strictly_above_all_3_alternatives"}


def compile_plan(a167_plan, a167_run_header, protocol_sha256):
    """Return the fixed 72 contexts/288 evaluations; never read parent responses."""
    return _build_plan(a167_plan, a167_run_header, protocol_sha256)


def validate_plan(plan):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    expected = _build_plan(plan["a167_plan"], plan["a167_run_header"], plan["bindings"]["protocol_sha256"])
    require(tasks.canonical(plan) == tasks.canonical(expected), "fixed_plan_drift")


def _same(left, right):
    """Canonical comparison preserves numeric and Boolean type distinctions."""
    return tasks.canonical(left) == tasks.canonical(right)


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def prepare_candidates(tokenizer, trial, prepared, config, eos_ids):
    """Joint tokenization and native assistant EOS audit, without a model call."""
    prompt, rendered = prepared["prompt_token_ids"], prepared["rendered_text"]
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos in eos_ids, "intended_eos")
    paths = {}
    for kind in KINDS:
        response = trial["canonical_candidates"][kind]
        full = native.engine._ids(tokenizer.encode(rendered + response, add_special_tokens=False))
        require(full[:len(prompt)] == prompt and len(full) > len(prompt), "native_prompt_prefix_invariance")
        continuation = full[len(prompt):]
        require(_decode(tokenizer, continuation) == response, "canonical_continuation_roundtrip")
        require(not any(token in eos_ids for token in continuation), "premature_eos")
        # Obtain the intended EOS from the complete native assistant rendering too.
        messages = trial["messages"] + [{"role": "assistant", "content": response}]
        kwargs = {"add_generation_prompt": False, "date_string": native.engine.TEMPLATE_DATE}
        closed_text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        closed = native.engine._ids(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs))
        require(closed == native.engine._ids(tokenizer.encode(closed_text, add_special_tokens=False))
                and closed == full + [eos], "native_assistant_eos_audit")
        require(len(closed) <= config["max_prompt_tokens"] + config["max_new_tokens"], "candidate_context_limit")
        paths[kind] = continuation + [eos]
    prefix = list(paths[KINDS[0]])
    for tokens in paths.values():
        prefix = prefix[:next((i for i, (a, b) in enumerate(zip(prefix, tokens)) if a != b), min(len(prefix), len(tokens)))]
    text = _decode(tokenizer, prefix) if prefix else ""
    require(bool(text) and JSON_PREFIX.startswith(text) and len(prefix) < min(map(len, paths.values())), "shared_json_syntax_prefix")
    require(tokenizer.encode(rendered + text, add_special_tokens=False) == prompt + prefix, "shared_prefix_joint_roundtrip")
    return {**prepared, "shared_json_prefix_token_ids": prefix, "shared_json_prefix_text": text,
            "intended_eos_token_id": eos, "candidate_token_ids": paths,
            "candidate_lengths": {kind: {"json_prefix_tokens": len(prefix), "answer_continuation_tokens": len(tokens) - len(prefix),
                                           "joint_tokens": len(tokens), "eos_tokens": 1} for kind, tokens in paths.items()}}


def _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader):
    plan = native._read_bound(plan_path, plan_sha256)
    config = native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    require(config_sha256 == plan["bindings"]["config_sha256"] and config["attention_implementation"] == "sdpa"
            and config["cpu_threads"] == 4 and config["seed"] == plan["a167_run_header"]["settings"]["seed"], "same_parent_config")
    require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "protocol_drift")
    require(native.engine.file_digest(reference_script) == REFERENCE_SHA, "reference_source_drift")
    sources, tokenizer, eos = validate_sources(), tokenizer_loader(config), native.pinned_eos_ids(config)
    old, header = plan["a167_plan"], plan["a167_run_header"]
    require(eos == header["eos_token_ids"], "parent_eos_drift")
    originals165 = {t["trial_id"]: t for t in old["a165_plan"]["trials"]}
    original_prepared = {t["trial_id"]: native.prepare_generation(tokenizer, t, config) for t in old["trials"]}
    matched = []
    for offset in range(0, 48, 3):
        group = {t["diagnostic_phase"]: t for t in old["trials"][offset:offset + 3]}
        for phase in ("baseline", "inert"):
            t = group[phase]
            source = originals165[t["source_a165_trial_id"]]
            require(t["messages"] == source["messages"] and original_prepared[t["trial_id"]]
                    == native.prepare_generation(tokenizer, source, config), "parent_a165_native_identity")
        counts = {phase: len(original_prepared[t["trial_id"]]["prompt_token_ids"]) for phase, t in group.items()}
        require(counts["prose"] == counts["inert"], "parent_material_token_match")
        matched.append({"triplet_index": offset // 3, "paired_placement": group["baseline"]["paired_placement"],
                        "prompt_token_counts": counts})
    fixed = prose._fixed_header(old, config, prose.validate_sources(), eos, original_prepared, matched, object_sha(old),
                                config_sha256, Path(header["reference_script_path"]))
    require(all(tasks.canonical(header[k]) == tasks.canonical(v) for k, v in fixed.items()), "parent_native_header_replay")
    originals167 = {t["trial_id"]: t for t in old["trials"]}
    prepared = {}
    for trial in plan["trials"]:
        source = (originals167 if trial["source_study"] == "a167" else originals165)[trial["source_trial_id"]]
        value = native.prepare_generation(tokenizer, trial, config)
        require(trial["messages"] == source["messages"] and value == native.prepare_generation(tokenizer, source, config),
                "all72_original_native_identity")
        prepared[trial["trial_id"]] = prepare_candidates(tokenizer, trial, value, config, eos)
    require(len({object_sha(v["prompt_token_ids"]) for v in prepared.values()}) == 72, "unique_native_contexts")
    boundaries = {object_sha(v["shared_json_prefix_token_ids"]) for v in prepared.values()}
    require(len(boundaries) == 1, "fixed_common_prefix_boundary")
    return plan, config, sources, tokenizer, eos, prepared


def prepare_inputs(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
                   tokenizer_loader=native.load_tokenizer):
    """Token-only preflight; raw strings and token arrays stay in returned objects."""
    plan, _, sources, _, eos, prepared = _inputs(plan_path, plan_sha256, config_path, config_sha256,
        protocol_path, reference_script, tokenizer_loader)
    return {"schema_version": "a169-native-freeze-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "reference_script_sha256": REFERENCE_SHA, "parent_run_sha256": plan["bindings"]["a167_run_sha256"],
        "planned_contexts": 72, "maximum_candidate_evaluations": 288, "eos_token_ids": eos,
        "native_prepared_sha256": object_sha(prepared), "native_inputs": prepared}


def _scores(token_logprobs, prefix_count):
    require(type(token_logprobs) is list and len(token_logprobs) > prefix_count > 0
            and all(type(v) in (float, int) and math.isfinite(v) and v <= 0 for v in token_logprobs), "finite_token_logprobs")
    prefix = math.fsum(token_logprobs[:prefix_count])
    continuation = math.fsum(token_logprobs[prefix_count:])
    joint = math.fsum((prefix, continuation))
    require(all(math.isfinite(v) for v in (prefix, continuation, joint)), "finite_sequence_logprobs")
    return {"json_prefix_logprob": prefix, "answer_continuation_logprob": continuation, "joint_logprob": joint,
            "eos_logprob": token_logprobs[-1], "json_prefix_tokens": prefix_count,
            "answer_continuation_tokens": len(token_logprobs) - prefix_count, "joint_tokens": len(token_logprobs), "eos_tokens": 1}


def teacher_force(model, torch, prompt_token_ids, target_token_ids, prefix_count, device="cpu"):
    """One fresh causal forward, scoring every JSON/EOS target with shifted logits.

    The last input EOS logit predicts a token outside the path and is discarded.
    Only the final target_count+1 logits are materialized, avoiding prompt logits.
    """
    require(bool(prompt_token_ids) and len(target_token_ids) > prefix_count > 0, "score_path_shape")
    ids = torch.tensor([prompt_token_ids + target_token_ids], dtype=torch.long, device=device)
    count = len(target_token_ids)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=count + 1).logits
        require(tuple(logits.shape[:2]) == (1, count + 1), "shifted_logit_shape")
        used = logits[0, :-1].float()
        require(bool(torch.isfinite(used).all().item()), "finite_logits")
        targets = torch.tensor(target_token_ids, dtype=torch.long, device=used.device)
        chosen = used.gather(1, targets[:, None])[:, 0].double()
        normalizers = torch.logsumexp(used.double(), dim=-1)
        chosen_logits, log_normalizers = chosen.tolist(), normalizers.tolist()
        token_logprobs = [a - b for a, b in zip(chosen_logits, log_normalizers)]
    return {"chosen_logits": chosen_logits, "log_normalizers": log_normalizers,
            "token_logprobs": token_logprobs, "scores": _scores(token_logprobs, prefix_count)}


def _attempt(header, evaluation, prepared):
    value = prepared[evaluation["trial_id"]]
    return {"schema_version": "a169-attempt-v1", "run_sha256": object_sha(header), "evaluation": evaluation,
        "prepared_sha256": object_sha(value), "prompt_token_ids": value["prompt_token_ids"],
        "target_token_ids": value["candidate_token_ids"][evaluation["candidate_kind"]],
        "json_prefix_token_count": len(value["shared_json_prefix_token_ids"]), "intended_eos_token_id": value["intended_eos_token_id"]}


def _validate_result(result, attempt):
    require(type(result) is dict and set(result) == {"schema_version", "attempt_sha256", "status", "measurement", "error", "elapsed_seconds"}
            and result["schema_version"] == "a169-result-v1" and result["attempt_sha256"] == object_sha(attempt)
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
        require(_same(value["scores"], _scores(value["token_logprobs"], attempt["json_prefix_token_count"])), "score_recomputation")
    else:
        require(result["status"] == "infrastructure_failed" and result["measurement"] is None
                and type(result["error"]) is dict and set(result["error"]) == {"exception_type", "frames", "code"}, "failed_result_schema")


def _context(plan, index, results):
    group = plan["evaluations"][4 * index:4 * index + 4]
    scores = {e["candidate_kind"]: (results[e["evaluation_id"]]["measurement"]["scores"]
              if e["evaluation_id"] in results and results[e["evaluation_id"]]["status"] == "completed" else None) for e in group}
    available = all(v is not None for v in scores.values())
    spread = max(v["json_prefix_logprob"] for v in scores.values()) - min(v["json_prefix_logprob"] for v in scores.values()) if available else None
    prefix_ok = spread <= PREFIX_ABSOLUTE_TOLERANCE_NATS if available else None
    complete = available and prefix_ok
    margins = {name: (scores["selected_value"][key] - max(scores[k][key] for k in KINDS[1:]) if complete else None)
               for name, key in (("continuation_margin_nats", "answer_continuation_logprob"), ("joint_margin_nats", "joint_logprob"))}
    return {"candidate_scores": scores, "complete": complete, "all_candidate_scores_available": available,
            "prefix_score_spread_nats": spread, "prefix_audit_passed": prefix_ok, **margins,
            "canonical_json_prefix_logprob_nats": scores["selected_value"]["json_prefix_logprob"] if complete else None,
            "selected_minus_other_value_nats": scores["selected_value"]["answer_continuation_logprob"] - scores["other_value"]["answer_continuation_logprob"] if complete else None,
            "selected_minus_selected_label_nats": scores["selected_value"]["answer_continuation_logprob"] - scores["selected_label"]["answer_continuation_logprob"] if complete else None,
            "selected_preferred": margins["continuation_margin_nats"] > 0 if complete else None,
            "joint_selected_preferred": margins["joint_margin_nats"] > 0 if complete else None}


def baseline_gate(plan, results):
    contexts = [_context(plan, index, results) for index in range(8)]
    return {"planned_contexts": 8, "planned_candidate_evaluations": 32,
        "completed_contexts": sum(c["complete"] for c in contexts),
        "strictly_preferred_contexts": sum(c["joint_selected_preferred"] is True for c in contexts),
        "gate_passed": all(c["joint_selected_preferred"] is True for c in contexts), "ties_fail": True}


def _contrast(plan, contexts, plus, minus, placement):
    lookup = {(t["world_index"], t["selector_index"], t["diagnostic_phase"], t["placement"]): contexts[i]
              for i, t in enumerate(plan["trials"])}
    world_means, identified = [], 0
    for world in range(4):
        differences = []
        for selector in range(2):
            left = lookup[world, selector, plus, placement]["continuation_margin_nats"]
            right = lookup[world, selector, minus, placement]["continuation_margin_nats"]
            differences.append(left - right if left is not None and right is not None else None)
        identified += sum(v is not None for v in differences)
        world_means.append(math.fsum(differences) / 2 if all(v is not None for v in differences) else None)
    complete = all(v is not None for v in world_means)
    return {"direction": plus + "_minus_" + minus, "placement": placement, "planned_worlds": 4, "planned_pairs": 8,
        "identified_pairs": identified, "missing_pairs": 8 - identified, "world_mean_difference_nats": world_means,
        "equal_world_mean_difference_nats": math.fsum(world_means) / 4 if complete else None,
        "continuous_missingness_bounds": None if complete else "unbounded", "missing_imputed": False}


def _pairwise_summary(plan, contexts, phase, placement):
    output = {}
    for key in ("selected_minus_other_value_nats", "selected_minus_selected_label_nats", "canonical_json_prefix_logprob_nats"):
        worlds = []
        identified = 0
        for world in range(4):
            values = [contexts[i][key] for i, t in enumerate(plan["trials"]) if t["world_index"] == world
                      and t["diagnostic_phase"] == phase and t["placement"] == placement]
            require(len(values) == 2, "pairwise_world_balance")
            identified += sum(v is not None for v in values)
            worlds.append(math.fsum(values) / 2 if all(v is not None for v in values) else None)
        complete = all(v is not None for v in worlds)
        output[key] = {"planned_worlds": 4, "planned_contexts": 8, "identified_contexts": identified,
            "world_mean_nats": worlds, "equal_world_mean_nats": math.fsum(worlds) / 4 if complete else None,
            "continuous_missingness_bounds": None if complete else "unbounded"}
    return output


def _aggregate(plan, results, attempted):
    contexts = [_context(plan, index, results) for index in range(72)]
    completed = sum(r["status"] == "completed" for r in results.values())
    gate = baseline_gate(plan, results)
    arms = {}
    for phase in PHASES:
        values = [contexts[i] for i, t in enumerate(plan["trials"]) if t["diagnostic_phase"] == phase]
        arms[phase] = {"planned_contexts": len(values), "complete_contexts": sum(v["complete"] for v in values),
            "selected_preferred": sum(v["selected_preferred"] is True for v in values),
            "selected_not_preferred": sum(v["selected_preferred"] is False for v in values),
            "selected_preference_missing": sum(v["selected_preferred"] is None for v in values)}
    return {"schema_version": "a169-analysis-v1", "status": "complete" if completed == 288 and all(c["complete"] for c in contexts) else
            "baseline_unqualified" if len(results) == 32 and not gate["gate_passed"] else "incomplete",
        "planned_worlds": 4, "planned_contexts": 72, "maximum_candidate_evaluations": 288,
        "coverage": {"attempted": len(attempted), "completed": completed, "infrastructure_failed": len(results) - completed,
                     "interrupted": len(attempted) - len(results), "unattempted": 288 - len(attempted)},
        "baseline_gate": gate, "arms": arms,
        "prefix_audit": {"absolute_tolerance_nats": PREFIX_ABSOLUTE_TOLERANCE_NATS,
            "failed_contexts": sum(c["prefix_audit_passed"] is False for c in contexts),
            "missing_contexts": sum(c["prefix_audit_passed"] is None for c in contexts)},
        "secondary_pairwise_margins": {phase: {p: _pairwise_summary(plan, contexts, phase, p)
            for p in (("none",) if phase == "baseline" else ("before", "after"))} for phase in PHASES},
        "primary_full_minus_sham": {p: _contrast(plan, contexts, "full", "sham", p) for p in ("before", "after")},
        "secondary_prose_minus_inert": {p: _contrast(plan, contexts, "prose", "inert", p) for p in ("before", "after")},
        "claim_boundaries": {"conditional_on_supplied_json_syntax": True, "rescued_free_generation": False,
            "direct_causal_evidence": False, "length_normalized": False, "accuracy_surrogate": False,
            "confidence_intervals": False, "population_inference": False, "heldout_allowed": False,
            "automatic_followup_or_task_repair": False, "missing_scores_imputed": False},
        "plan_object_sha256": object_sha(plan)}


def _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script):
    return {"schema_version": "a169-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "parent_run_sha256": plan["bindings"]["a167_run_sha256"],
        "protocol_sha256": plan["bindings"]["protocol_sha256"], "reference_script_sha256": REFERENCE_SHA,
        "reference_script_path": str(reference_script.absolute()), "native_prepared_sha256": object_sha(prepared),
        "model_manifest_sha256": object_sha(config["model_files_sha256"]), "eos_token_ids": eos,
        "settings": {**plan["a167_plan"]["a165_run_header"]["settings"], "teacher_forcing": True, "use_cache": False,
                     "maximum_candidate_evaluations": 288, "free_generation": False},
        "prefix_absolute_tolerance_nats": PREFIX_ABSOLUTE_TOLERANCE_NATS,
        "normalization_arithmetic": "float64_logsumexp_of_float32_logits",
        "maximum_seconds": MAX_SECONDS, "kill_grace_seconds": KILL_GRACE_SECONDS, "retries": 0, "heldout_allowed": False}


def _replay(root, plan, config, sources, tokenizer, eos, prepared, plan_sha256, config_sha256, reference_script):
    header = state._read(root / "run.json")
    fixed = _fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script)
    require(set(header) == {*fixed, "pid", "available_memory_bytes"} and all(_same(header[k], v) for k, v in fixed.items())
            and type(header["pid"]) is int and header["pid"] > 0 and type(header["available_memory_bytes"]) is int
            and header["available_memory_bytes"] >= MIN_AVAILABLE_BYTES, "run_lineage")
    require(_same(state._read(root / "one-shot.json"), {"schema_version": "a169-one-shot-v1", "pid": header["pid"],
            "plan_sha256": plan_sha256}), "one_shot_binding")
    require(_same(state._read(root / "native-inputs.json"), prepared), "native_input_replay")
    expected = {e["evaluation_id"] for e in plan["evaluations"]}
    folders = root / "evaluations"
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
        if evaluation["sequence_index"] >= 32:
            require(baseline_gate(plan, results)["gate_passed"], "baseline_gate_bypass")
        attempt = state._read(folder / "attempt.json")
        require(_same(attempt, _attempt(header, evaluation, prepared)), "attempt_native_replay")
        attempted.add(evaluation["evaluation_id"])
        if not (folder / "result.json").exists():
            closed = True
            continue
        result = state._read(folder / "result.json")
        _validate_result(result, attempt)
        results[evaluation["evaluation_id"]] = result
    if (root / "execution-finished.json").exists():
        finished = state._read(root / "execution-finished.json")
        require(type(finished) is dict and set(finished) == {"schema_version", "status", "elapsed_seconds", "run_sha256"}
                and finished["schema_version"] == "a169-execution-finished-v1" and finished["run_sha256"] == object_sha(header)
                and type(finished["elapsed_seconds"]) in (int, float) and math.isfinite(finished["elapsed_seconds"])
                and finished["elapsed_seconds"] >= 0 and finished["status"] in
                ("finished_schedule", "baseline_unqualified", "interrupted", "deadline", "failed"), "finished_receipt")
        if finished["status"] == "finished_schedule":
            require(len(results) == 288, "finished_schedule_coverage")
        elif finished["status"] == "baseline_unqualified":
            require(len(results) == len(attempted) == 32 and not baseline_gate(plan, results)["gate_passed"], "finished_gate_coverage")
        else:
            require(len(results) < 288, "finished_interruption_coverage")
    records = [{"schema_version": "a169-context-scores-v1", "trial_id": t["trial_id"], "world_index": t["world_index"],
                "selector_index": t["selector_index"], "phase": t["diagnostic_phase"], "placement": t["placement"],
                **_context(plan, i, results)} for i, t in enumerate(plan["trials"])]
    aggregate = _aggregate(plan, results, attempted)
    aggregate["numeric_records_sha256"] = object_sha(records)
    aggregate["provenance"] = {key: header[key] for key in ("plan_sha256", "config_sha256", "source_hashes",
        "parent_run_sha256", "protocol_sha256", "native_prepared_sha256", "model_manifest_sha256")}
    aggregate["provenance"]["run_sha256"] = object_sha(header)
    for name, value in (("records.json", records), ("analysis.json", aggregate)):
        if (root / name).exists():
            require(_same(state._read(root / name), value), "stored_export_drift")
    return {"records": records, "aggregate": aggregate}


def export_run(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
               output_root, tokenizer_loader=native.load_tokenizer):
    """Replay bound native paths and token-logprob arithmetic; no model is loaded."""
    root = lineage._private_root(output_root)
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        return _replay(root, *inputs, plan_sha256, config_sha256, reference_script)


def _interrupt(_signum, _frame):
    raise A169Interrupted()


def _deadline(_signum, _frame):
    raise A169Deadline()


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
             candidate_scorer=teacher_force):
    """One fresh load, baseline gate at evaluation 32, then at most 256 more calls."""
    global _PROCESS_CONSUMED
    require(not _PROCESS_CONSUMED, "fresh_process_required")
    root = lineage._private_root(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with bounded_signals(), (root / ".lock").open("a") as lock:
        os.chmod(root / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root / p).exists() for p in ("one-shot.json", "run.json", "evaluations")), "consumed_run")
        inputs = _inputs(plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script, tokenizer_loader)
        plan, config, sources, tokenizer, eos, prepared = inputs
        require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"], "original_model_manifest")
        reference = reference_loader(reference_script)
        available = reference.require_memory()
        require(type(available) is int and available >= MIN_AVAILABLE_BYTES, "preload_memory_gate")
        _PROCESS_CONSUMED = True
        native.engine.write_private(root / "one-shot.json", {"schema_version": "a169-one-shot-v1", "pid": os.getpid(), "plan_sha256": plan_sha256})
        header = {**_fixed_header(plan, config, sources, eos, prepared, plan_sha256, config_sha256, reference_script),
                  "pid": os.getpid(), "available_memory_bytes": available}
        native.engine.write_private(root / "run.json", header)
        native.engine.write_private(root / "native-inputs.json", prepared)
        started, status, runtime, results = time.monotonic(), "failed", None, {}
        try:
            runtime = reference.load_cpu_reference(config, native.engine)
            require(runtime.eos_ids == eos and tasks.sha(runtime.tokenizer.get_chat_template().encode())
                    == config["chat_template_sha256"], "runtime_native_binding")
            for evaluation in plan["evaluations"]:
                if evaluation["sequence_index"] == 32 and not baseline_gate(plan, results)["gate_passed"]:
                    status = "baseline_unqualified"
                    break
                attempt = _attempt(header, evaluation, prepared)
                folder = root / "evaluations" / evaluation["evaluation_id"]
                native.engine.write_private(folder / "attempt.json", attempt)
                tick = time.monotonic()
                result = {"schema_version": "a169-result-v1", "attempt_sha256": object_sha(attempt),
                          "status": "infrastructure_failed", "measurement": None, "error": None, "elapsed_seconds": 0.}
                try:
                    result["measurement"] = candidate_scorer(runtime.model, runtime.torch, attempt["prompt_token_ids"],
                        attempt["target_token_ids"], attempt["json_prefix_token_count"], runtime.device)
                    result["status"] = "completed"
                    _validate_result(result, attempt)
                except Exception as exc:
                    result.update(status="infrastructure_failed", measurement=None,
                                  error=native.engine.safe_error(exc, "a169_candidate_failed"))
                result["elapsed_seconds"] = time.monotonic() - tick
                _validate_result(result, attempt)
                native.engine.write_private(folder / "result.json", result)
                results[evaluation["evaluation_id"]] = result
            else:
                status = "finished_schedule"
        except BaseException as exc:
            status = "deadline" if isinstance(exc, A169Deadline) else (
                "interrupted" if isinstance(exc, (A169Interrupted, KeyboardInterrupt)) else "failed")
            native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a169_execution_failed"))
        finally:
            runtime = None
            native.engine.write_private(root / "execution-finished.json", {"schema_version": "a169-execution-finished-v1",
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
        root = lineage._private_root(args.output_root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (root / ("execution.log" if args.mode == "run" else "replay.log")).open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                aggregate = run_plan(**kwargs) if args.mode == "run" else export_run(**kwargs)["aggregate"]
        print(json.dumps(aggregate, sort_keys=True, allow_nan=False))
        return 0 if aggregate["status"] in ("complete", "baseline_unqualified") else 1
    except BaseException as exc:
        print(json.dumps({"status": "a169_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
