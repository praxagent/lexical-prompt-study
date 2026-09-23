"""A189 fixed generation-only arithmetic task-load acquisition.

Inject a qualified CUDA BF16 model/tokenizer; no asset loading or work on import.
The frozen outer worker owns authorization, resource limits, body profiling and
normal-return evidence. Native serialization is retained, not independently
retokenized during model-free replay. No residual capture, reference or fitting
is performed. Native BF16 logits and their exact FP32 widening are retained together.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import gpu3b_arithmetic_task_load as tasks
from . import landmark_evidence as e
from . import cuda_bf16_forward as primitive
from . import native_prefix_qualification as previous
from . import prefix_only_capture as controls

SCHEMA = "a189-bf16-arithmetic-acquisition-v1"
CAP, MAX_PROMPT, MAX_ENTRIES, PREFIX_TOKENS, ROWS = 64, 256, 2048, 4, 32
DATE = "23 Sep 2026"
A183_SHA = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
A184_SHA = "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69"
A185_SHA = "2542d54eaf839a253ae0b11c028a036ca0d6f697935dd456d0f4e3fc4d929526"
_write, _read = previous._write, previous._read


def require(ok, reason):
    if not ok:
        raise ValueError("a189_acquisition_" + reason)


def same(left, right, reason):
    require(e.canonical(left) == e.canonical(right), reason)


def source_sha256():
    return e.sha256(Path(__file__).read_bytes())


def source_bindings():
    pins = ((e, A183_SHA), (controls, A184_SHA), (previous, A185_SHA))
    for module, wanted in pins:
        require(e.sha256(Path(module.__file__).read_bytes()) == wanted, "dependency_source")
    return {
        "a183": A183_SHA,
        "a184": A184_SHA,
        "a185": A185_SHA,
        "cuda_bf16_forward": e.sha256(Path(primitive.__file__).read_bytes()),
        "acquisition": source_sha256(),
        "tasks": e.sha256(Path(tasks.__file__).read_bytes()),
    }


def compile_plan(
    *, provenance, geometry, eos_token_ids, runtime_binding_sha256, protocol_sha256, tests_sha256
):
    e._keys(provenance, "model_sha256 tokenizer_sha256 chat_template_sha256", "provenance")
    for value in (*provenance.values(), runtime_binding_sha256, protocol_sha256, tests_sha256):
        e._hash(value)
    e._keys(geometry, "vocab_size hidden_width model_layers context_limit", "geometry")
    for value in geometry.values():
        e._int(value, 1)
    e._ids(eos_token_ids, geometry["vocab_size"], nonempty=True)
    require(eos_token_ids == sorted(set(eos_token_ids)), "eos_order")
    rows = tasks.validate_roster(tasks.build_roster())
    require(len(rows) == ROWS, "roster_size")
    return {
        "schema_version": SCHEMA,
        "study_id": "a189_gpu3b_arithmetic_task_load",
        "rows": rows,
        "provenance": e._snapshot(provenance),
        "geometry": e._snapshot(geometry),
        "eos_token_ids": list(eos_token_ids),
        "decoder": dict(e.DECODER),
        "max_prompt_tokens": MAX_PROMPT,
        "chat_template_date": DATE,
        "max_entries": MAX_ENTRIES,
        "generation_policy": {
            "cap": CAP,
            "argmax": "first_maximum_index",
            "cache": False,
            "attention": "all_ones",
            "positions": "sequential",
            "logits_to_keep": 1,
            "output_hidden_states": False,
            "output_attentions": False,
            "native_logits_dtype": "bfloat16",
            "retained_logits": "native_bf16_and_exact_fp32",
            "device": "cuda:0",
            "sdpa_backend": "math",
            "cuda_synchronized_before_usable_readout": True,
        },
        "diagnostic": {"non_eos_prefix_tokens": PREFIX_TOKENS, "additional_forwards": 0},
        "technical_check": {
            "messages": tasks.technical_messages(),
            "forwards": tasks.TECHNICAL_FORWARDS,
            "maximum_total_entries": tasks.MAX_TARGET_ENTRIES,
            "native_dtype": "bfloat16",
            "storage_dtype": "float32",
            "readout": "exact_bf16_bytes_plus_bit_exact_fp32_widening",
        },
        "step_receipt_schema": primitive.SCHEMA,
        "bindings": {
            "sources": source_bindings(),
            "runtime_binding_sha256": runtime_binding_sha256,
            "protocol_sha256": protocol_sha256,
            "tests_sha256": tests_sha256,
        },
    }


def validate_plan(plan):
    require(type(plan) is dict, "plan_type")
    expected = compile_plan(
        provenance=plan["provenance"],
        geometry=plan["geometry"],
        eos_token_ids=plan["eos_token_ids"],
        **{
            k: plan["bindings"][k]
            for k in ("runtime_binding_sha256", "protocol_sha256", "tests_sha256")
        },
    )
    same(plan, expected, "fixed_plan")
    return e._snapshot(plan)


def _primitive_plan(plan):
    """Geometry-only compatibility object, never an A183 capture manifest."""
    g = plan["geometry"]
    return {
        "context_limit": g["context_limit"],
        "manifest": {
            "vocab_size": g["vocab_size"],
            "capture": {
                "hidden_width": g["hidden_width"],
                "model_layers": g["model_layers"],
                "layer_index": g["model_layers"] - 1,
            },
        },
    }


def prepare_inputs(plan, tokenizer):
    plan = validate_plan(plan)
    m = plan
    require(
        e.sha256(tokenizer.get_chat_template().encode()) == m["provenance"]["chat_template_sha256"],
        "template",
    )
    eos, special = tokenizer.eos_token_id, tokenizer.all_special_ids
    e._ids(special, m["geometry"]["vocab_size"], nonempty=True)
    require(
        type(eos) is int
        and eos in m["eos_token_ids"]
        and eos in special
        and len(set(special)) == len(special),
        "native_special_ids",
    )
    observations = []
    for row in plan["rows"]:
        messages = row["messages"]
        payload = [tokenizer.encode(item["content"], add_special_tokens=False) for item in messages]
        for values in payload:
            e._ids(values, m["geometry"]["vocab_size"], nonempty=True)
            require(not set(values).intersection(special), "payload_special_token")
        kwargs = {"add_generation_prompt": True, "date_string": DATE}
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        e._str(text)
        for values in (ids, encoded):
            e._ids(values, m["geometry"]["vocab_size"], nonempty=True)
        require(
            ids == encoded and all(item["content"] in text for item in messages), "native_rendering"
        )
        require(
            len(ids) <= MAX_PROMPT and len(ids) + CAP <= plan["geometry"]["context_limit"],
            "native_context",
        )
        joint = tokenizer.encode(text + "probe", add_special_tokens=False)
        e._ids(joint, m["geometry"]["vocab_size"], nonempty=True)
        require(
            joint[: len(ids)] == ids
            and len(joint) > len(ids)
            and not any(t in m["eos_token_ids"] for t in joint[len(ids) :]),
            "probe_boundary",
        )
        require(tokenizer.decode(joint[len(ids) :], **m["decoder"]) == "probe", "probe_decode")
        closed = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": "probe"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            date_string=DATE,
        )
        e._ids(closed, m["geometry"]["vocab_size"], nonempty=True)
        require(closed == joint + [eos], "native_closed_eos")
        observations.append(
            {
                "observation_id": row["observation_id"],
                "messages": messages,
                "rendered_prompt_text": text,
                "prompt_token_ids": ids,
                "prompt_ids_sha256": e.object_hash(ids),
                "prompt_utf8_sha256": e.sha256(text.encode()),
                "intended_eos_token_id": eos,
                "assistant_probe_token_ids": joint[len(ids) :] + [eos],
                "payload_token_ids": payload,
            }
        )
    return {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "special_token_ids": sorted(special),
        "observations": observations,
    }


def validate_prepared(plan, prepared):
    e._keys(prepared, "schema_version plan_sha256 special_token_ids observations", "prepared")
    require(
        prepared["schema_version"] == SCHEMA and prepared["plan_sha256"] == e.object_hash(plan),
        "prepared_binding",
    )
    special = prepared["special_token_ids"]
    e._ids(special, plan["geometry"]["vocab_size"], nonempty=True)
    require(special == sorted(set(special)), "prepared_special")
    require(
        type(prepared["observations"]) is list and len(prepared["observations"]) == 32,
        "prepared_count",
    )
    for row, obs in zip(plan["rows"], prepared["observations"], strict=True):
        e._keys(
            obs,
            "observation_id messages rendered_prompt_text prompt_token_ids prompt_ids_sha256 prompt_utf8_sha256 intended_eos_token_id assistant_probe_token_ids payload_token_ids",
            "observation",
        )
        e._str(obs["rendered_prompt_text"])
        e._ids(obs["prompt_token_ids"], plan["geometry"]["vocab_size"], nonempty=True)
        require(obs["observation_id"] == row["observation_id"], "observation_identity")
        same(obs["messages"], row["messages"], "messages")
        require(
            len(obs["prompt_token_ids"]) <= MAX_PROMPT
            and len(obs["prompt_token_ids"]) + CAP <= plan["geometry"]["context_limit"],
            "prepared_context",
        )
        require(
            obs["prompt_ids_sha256"] == e.object_hash(obs["prompt_token_ids"])
            and obs["prompt_utf8_sha256"] == e.sha256(obs["rendered_prompt_text"].encode()),
            "prepared_hash",
        )
        eos = obs["intended_eos_token_id"]
        require(
            type(eos) is int and eos in plan["eos_token_ids"] and eos in special,
            "prepared_eos",
        )
        probe = obs["assistant_probe_token_ids"]
        e._ids(probe, plan["geometry"]["vocab_size"], nonempty=True)
        require(
            len(probe) > 1
            and probe[-1] == eos
            and not any(t in plan["eos_token_ids"] for t in probe[:-1]),
            "prepared_probe",
        )
        require(
            type(obs["payload_token_ids"]) is list
            and len(obs["payload_token_ids"]) == len(row["messages"]),
            "prepared_payloads",
        )
        for values in obs["payload_token_ids"]:
            e._ids(values, plan["geometry"]["vocab_size"], nonempty=True)
            require(not set(values).intersection(special), "prepared_payload_special")
    return e._snapshot(prepared)


def _inventory(root):
    return primitive._inventory(root)


def _call_attempt(plan, run_sha, row, step, values):
    return {
        "schema_version": primitive.SCHEMA,
        "run_sha256": run_sha,
        "plan_sha256": e.object_hash(plan),
        "sequence_index": row["sequence_index"],
        "step_index": step,
        "input_token_ids": values,
        "input_ids_sha256": e.object_hash(values),
    }


def _generation_attempt(plan, row, observation, run_sha):
    return {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "row_sha256": e.object_hash(row),
        "observation_sha256": e.object_hash(observation),
        "policy": plan["generation_policy"],
    }


def _save_prefix(plan, tokenizer, root, ids):
    content = ids[:-1] if ids and ids[-1] in plan["eos_token_ids"] else ids
    available = len(content) >= PREFIX_TOKENS
    raw = None
    if available:
        decoded = tokenizer.decode(content[:PREFIX_TOKENS], **plan["decoder"])
        require(type(decoded) is str, "prefix_decode")
        raw = decoded.encode("utf-8")
        e._publish(root / "prefix.utf8", raw)
    value = {
        "schema_version": SCHEMA,
        "non_eos_tokens": PREFIX_TOKENS,
        "available": available,
        "token_ids": content[:PREFIX_TOKENS] if available else None,
        "decoded_utf8_sha256": e.sha256(raw) if raw is not None else None,
    }
    _write(root / "prefix.json", value)
    return value


def _checkpoint_prefix(plan, tokenizer, root, ids):
    value = _save_prefix(plan, tokenizer, root, ids)
    _write(
        root / "prefix-return.json",
        {
            "schema_version": SCHEMA,
            "event": "prefix_writer_returned_normally",
            "prefix_sha256": e.object_hash(value),
        },
    )


def _read_prefix(plan, root, ids, *, complete):
    content = ids[:-1] if ids and ids[-1] in plan["eos_token_ids"] else ids
    available = len(content) >= PREFIX_TOKENS
    metadata, payload = root / "prefix.json", root / "prefix.utf8"
    returned = root / "prefix-return.json"
    if not complete and (
        not metadata.exists() or not returned.exists() or available and not payload.exists()
    ):
        return None
    value = _read(metadata)
    same(
        _read(returned),
        {
            "schema_version": SCHEMA,
            "event": "prefix_writer_returned_normally",
            "prefix_sha256": e.object_hash(value),
        },
        "prefix_normal_return",
    )
    raw = e._read_file(payload) if available else None
    same(
        value,
        {
            "schema_version": SCHEMA,
            "non_eos_tokens": PREFIX_TOKENS,
            "available": available,
            "token_ids": content[:PREFIX_TOKENS] if available else None,
            "decoded_utf8_sha256": e.sha256(raw) if raw is not None else None,
        },
        "prefix_binding",
    )
    return raw.decode("utf-8") if raw is not None else None


def _generation(model, tokenizer, plan, row, observation, root, run_sha, counts):
    root = e._directory(root, create=True)
    attempt = _generation_attempt(plan, row, observation, run_sha)
    _write(root / "attempt.json", attempt)
    ids, entries, results = [], [], []
    for index in range(CAP):
        require(counts["generation"] < MAX_ENTRIES, "entry_budget_before_dispatch")
        call = _call_attempt(plan, run_sha, row, index, observation["prompt_token_ids"] + ids)
        result = primitive._forward(
            model, _primitive_plan(plan), root / f"step_{index:02d}", call, counts
        )
        require(counts["generation"] <= MAX_ENTRIES, "entry_budget_after_dispatch")
        ids.append(result["chosen_token_id"])
        entries.append(result["entry_sha256"])
        results.append(e.object_hash(result))
        if ids[-1] in plan["eos_token_ids"]:
            break
        if len(ids) == PREFIX_TOKENS:
            _checkpoint_prefix(plan, tokenizer, root, ids)
    if len(ids) < PREFIX_TOKENS or len(ids) == PREFIX_TOKENS and ids[-1] in plan["eos_token_ids"]:
        _checkpoint_prefix(plan, tokenizer, root, ids)
    dispatch = {
        "schema_version": SCHEMA,
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": entries,
        "step_result_sha256": results,
    }
    _write(root / "dispatch.json", dispatch)
    generation = {
        "status": "completed",
        "stop_reason": "eos" if ids[-1] in plan["eos_token_ids"] else "cap",
        "token_ids": ids,
        "observed_tokens": len(ids),
        "attempt_sha256": e.object_hash(attempt),
        "dispatch_sha256": e.object_hash(dispatch),
    }
    _write(root / "result.json", generation)
    content = ids[:-1] if generation["stop_reason"] == "eos" else ids
    text = tokenizer.decode(content, **plan["decoder"])
    require(type(text) is str, "response_decode")
    e._publish(root / "response.utf8", text.encode("utf-8"))
    score = tasks.score_generation(
        row, generation, text, _read_prefix(plan, root, ids, complete=True), plan["eos_token_ids"]
    )
    _write(root / "score.json", score)
    return {
        "generation_sha256": e.object_hash(generation),
        "response_sha256": e.sha256(text.encode()),
        "score_sha256": e.object_hash(score),
        "prefix_sha256": e.object_hash(_read(root / "prefix.json")),
    }


def _normal_return(run_sha, row, result):
    return {
        "schema_version": SCHEMA,
        "event": "generation_stage_returned_normally",
        "run_sha256": run_sha,
        "observation_id": row["observation_id"],
        **result,
    }


def _empty_record(row):
    return {
        **{
            key: row[key]
            for key in ("observation_id", "core_index", "item_count", "magnitude", "sequence_index")
        },
        "status": "unattempted",
        "generation_status": "missing",
        "outcome_status": "missing",
        "label": None,
        "terminal_eos": None,
        "capped": None,
        "censored": None,
        "prefix_reached": None,
        "completed_fields": None,
        "known_error": None,
        "remaining_fields": None,
        "unflagged_remaining_ge2": None,
        "result_sha256": None,
    }


def _record(row, score, *, status, generation_status, result_sha256=None):
    record = _empty_record(row)
    require(
        set(score)
        == set(record)
        - {
            "observation_id",
            "core_index",
            "item_count",
            "magnitude",
            "sequence_index",
            "status",
            "generation_status",
            "result_sha256",
        },
        "score_fields",
    )
    record.update(score)
    record.update(status=status, generation_status=generation_status, result_sha256=result_sha256)
    return record


def _partial_score(plan, row, root, ids):
    generation = {
        "status": "infrastructure_failed",
        "stop_reason": "infrastructure_failed",
        "token_ids": ids,
        "observed_tokens": len(ids),
    }
    return tasks.score_generation(
        row, generation, None, _read_prefix(plan, root, ids, complete=False), plan["eos_token_ids"]
    )


def _scan_steps(plan, row, observation, root, run_sha, *, complete):
    """Return usable chosen IDs, proved entries and whether that count is exact.

    Missing final result/failure evidence never turns a lower bound into an exact
    count. A bound primitive failure receipt reports actual dispatch-entry count;
    its optional saved logits/result remain unusable after that failed call.
    """
    if not root.exists():
        require(not complete, "missing_generation")
        return [], 0, True
    e._directory(root)
    attempt_path = root / "attempt.json"
    if not attempt_path.exists():
        require(
            not complete and {p.name for p in root.iterdir()} <= {".attempt.json.tmp"},
            "unpublished_generation_attempt",
        )
        return [], 0, True
    same(
        _read(attempt_path),
        _generation_attempt(plan, row, observation, run_sha),
        "generation_parent_attempt",
    )
    allowed = {
        "attempt.json",
        "prefix.json",
        "prefix-return.json",
        "prefix.utf8",
        "dispatch.json",
        "result.json",
        "response.utf8",
        "score.json",
    }
    names = sorted(p.name for p in root.iterdir() if p.name.startswith("step_"))
    require(
        {p.name for p in root.iterdir()}
        <= allowed | set(names) | {"." + name + ".tmp" for name in allowed},
        "partial_generation_inventory",
    )
    require(
        names == [f"step_{i:02d}" for i in range(len(names))] and len(names) <= CAP, "step_order"
    )
    ids, entries, exact = [], 0, True
    for index, name in enumerate(names):
        require(not ids or ids[-1] not in plan["eos_token_ids"], "no_step_after_eos")
        step = e._directory(root / name)
        files = {p.name for p in step.iterdir()}
        attempt = _call_attempt(plan, run_sha, row, index, observation["prompt_token_ids"] + ids)
        if not (step / "attempt.json").exists():
            require(
                not complete and index == len(names) - 1 and files <= {".attempt.json.tmp"},
                "unpublished_step_attempt",
            )
            continue
        same(_read(step / "attempt.json"), attempt, "step_attempt")
        present = (step / "entry.json").exists()
        if present:
            same(
                _read(step / "entry.json"),
                {
                    "schema_version": primitive.SCHEMA,
                    "event": "model_forward_pre_hook_entry",
                    "attempt_sha256": e.object_hash(attempt),
                    "entry_index": 0,
                },
                "step_entry",
            )
        if (step / "failure.json").exists():
            require(not complete and index == len(names) - 1, "only_final_failure")
            failure = _read(step / "failure.json")
            e._keys(
                failure,
                "schema_version attempt_sha256 entries model_call_returned cuda_synchronized cleanup_failure_type failure_type usable",
                "forward_failure",
            )
            require(
                failure["schema_version"] == primitive.SCHEMA
                and failure["usable"] is False
                and failure["attempt_sha256"] == e.object_hash(attempt)
                and type(failure["model_call_returned"]) is bool
                and type(failure["cuda_synchronized"]) is bool,
                "failed_forward_binding",
            )
            e._int(failure["entries"])
            require(failure["entries"] <= 1, "one_failed_dispatch")
            require(failure["entries"] >= int(present), "failed_entry_consistency")
            require(
                (not failure["model_call_returned"] or failure["entries"] == 1)
                and (not failure["cuda_synchronized"] or failure["model_call_returned"]),
                "returned_requires_entry",
            )
            e._str(failure["failure_type"])
            if failure["cleanup_failure_type"] is not None:
                e._str(failure["cleanup_failure_type"])
            allowed = {
                "attempt.json",
                "entry.json",
                "result.json",
                "logits.fp32",
                "logits.bf16",
                "failure.json",
            }
            require(
                files <= allowed | {"." + name + ".tmp" for name in allowed},
                "failed_step_inventory",
            )
            entries += failure["entries"]
        elif (step / "result.json").exists():
            result = primitive._replay_call(step, _primitive_plan(plan), attempt)
            ids.append(result["chosen_token_id"])
            entries += 1
        else:
            require(not complete and index == len(names) - 1, "only_final_incomplete")
            allowed = {"attempt.json", "entry.json", "logits.fp32", "logits.bf16"}
            require(
                files
                <= allowed
                | {"." + name + ".tmp" for name in allowed | {"result.json", "failure.json"}},
                "incomplete_step_inventory",
            )
            entries += int(present)
            exact = False
    require(entries <= CAP, "row_entry_budget")
    if complete:
        require(
            bool(ids)
            and len(ids) == len(names)
            and (ids[-1] in plan["eos_token_ids"] or len(ids) == CAP),
            "completed_stream",
        )
    return ids, entries, exact


def _replay_generation(plan, row, observation, root, run_sha):
    ids, entries, exact = _scan_steps(plan, row, observation, root, run_sha, complete=True)
    require(exact, "completed_entry_count")
    attempt = _generation_attempt(plan, row, observation, run_sha)
    dispatch = {
        "schema_version": SCHEMA,
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": [
            e.object_hash(_read(root / f"step_{i:02d}" / "entry.json")) for i in range(len(ids))
        ],
        "step_result_sha256": [
            e.object_hash(_read(root / f"step_{i:02d}" / "result.json")) for i in range(len(ids))
        ],
    }
    same(_read(root / "dispatch.json"), dispatch, "generation_dispatch")
    generation = {
        "status": "completed",
        "stop_reason": "eos" if ids[-1] in plan["eos_token_ids"] else "cap",
        "token_ids": ids,
        "observed_tokens": len(ids),
        "attempt_sha256": e.object_hash(attempt),
        "dispatch_sha256": e.object_hash(dispatch),
    }
    same(_read(root / "result.json"), generation, "generation_result")
    text = e._read_file(root / "response.utf8").decode("utf-8")
    prefix_text = _read_prefix(plan, root, ids, complete=True)
    score = tasks.score_generation(row, generation, text, prefix_text, plan["eos_token_ids"])
    same(_read(root / "score.json"), score, "score_replay")
    allowed = {
        "attempt.json",
        "dispatch.json",
        "result.json",
        "response.utf8",
        "score.json",
        "prefix.json",
        "prefix-return.json",
    }
    if len(ids) - int(ids[-1] in plan["eos_token_ids"]) >= PREFIX_TOKENS:
        allowed.add("prefix.utf8")
    require(
        {p.name for p in root.iterdir()} == allowed | {f"step_{i:02d}" for i in range(len(ids))},
        "generation_inventory",
    )
    produced = {
        "generation_sha256": e.object_hash(generation),
        "response_sha256": e.sha256(text.encode()),
        "score_sha256": e.object_hash(score),
        "prefix_sha256": e.object_hash(_read(root / "prefix.json")),
    }
    same(
        _read(root.parent / "generation-return.json"),
        _normal_return(run_sha, row, produced),
        "generation_normal_return",
    )
    return score, entries


def _replay_row(plan, row, observation, root, run_sha):
    require(
        {p.name for p in root.iterdir()} == {"generation", "generation-return.json", "result.json"},
        "row_inventory",
    )
    score, entries = _replay_generation(plan, row, observation, root / "generation", run_sha)
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "observation_id": row["observation_id"],
        "generation_return_sha256": e.object_hash(_read(root / "generation-return.json")),
        "score": score,
    }
    same(_read(root / "result.json"), result, "row_result")
    return _record(
        row,
        score,
        status="completed",
        generation_status="completed",
        result_sha256=e.object_hash(result),
    ), entries


def _summary(records, entries, exact, status):
    e._int(entries)
    require(entries <= MAX_ENTRIES, "aggregate_entry_ceiling")
    require(type(exact) is bool, "count_exactness")
    return {
        "planned_generations": ROWS,
        "completed_rows": sum(r["status"] == "completed" for r in records),
        "failed_rows": sum(r["status"] == "failed" for r in records),
        "unattempted_rows": sum(r["status"] == "unattempted" for r in records),
        "completed_generations": sum(r["generation_status"] == "completed" for r in records),
        "known_labels": sum(r["label"] is not None for r in records),
        "success_labels": sum(r["label"] == 1 for r in records),
        "generation_entries": entries if exact else None,
        "entry_lower_bound": entries,
        "entries_complete": exact,
        "total_entries": entries if exact else None,
        "schedule_finished": status == "finished",
    }


def _replay_prefix(plan, prepared, root, run_sha, accepted, active, phase):
    records = [_empty_record(row) for row in plan["rows"]]
    rows_root = root / "rows"
    expected = {f"{i:03d}" for i in range(accepted + int(active is not None))}
    require(
        rows_root.is_dir() and {p.name for p in rows_root.iterdir()} == expected, "schedule_prefix"
    )
    entries, exact = 0, True
    for index in range(accepted):
        records[index], count = _replay_row(
            plan,
            plan["rows"][index],
            prepared["observations"][index],
            rows_root / f"{index:03d}",
            run_sha,
        )
        entries += count
    if active is not None:
        require(
            active == accepted and phase in ("generation", "generation_return", "row_result"),
            "failed_position",
        )
        row, observation = plan["rows"][active], prepared["observations"][active]
        row_root = e._directory(rows_root / f"{active:03d}")
        allowed = {"generation", "generation-return.json", "result.json"}
        require(
            {p.name for p in row_root.iterdir()}
            <= allowed | {".generation-return.json.tmp", ".result.json.tmp"},
            "partial_row_inventory",
        )
        if phase == "row_result":
            score, count = _replay_generation(
                plan, row, observation, row_root / "generation", run_sha
            )
            records[active] = _record(row, score, status="failed", generation_status="completed")
        else:
            ids, count, exact = _scan_steps(
                plan, row, observation, row_root / "generation", run_sha, complete=False
            )
            score = _partial_score(plan, row, row_root / "generation", ids)
            records[active] = _record(
                row, score, status="failed", generation_status="infrastructure_failed"
            )
        entries += count
    return records, entries, exact


def run_acquisition(model, tokenizer, *, plan, prepared, runtime_binding, directory):
    """Run each fixed row once, and stop on every infrastructure failure."""
    if not __debug__:
        raise RuntimeError("a189_optimization_forbidden")
    plan = validate_plan(plan)
    prepared = validate_prepared(plan, prepared)
    require(
        type(runtime_binding) is dict
        and e.object_hash(runtime_binding) == plan["bindings"]["runtime_binding_sha256"],
        "runtime_binding",
    )
    same(prepare_inputs(plan, tokenizer), prepared, "native_replay")
    primitive.clean_model(model, _primitive_plan(plan))
    import torch

    require(
        torch.are_deterministic_algorithms_enabled() and torch.cuda.is_initialized(),
        "deterministic_cuda",
    )
    root = e._directory(directory, create=True)
    header = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "runtime_binding_sha256": e.object_hash(runtime_binding),
        "pid": os.getpid(),
        "sources": source_bindings(),
    }
    run_sha = e.object_hash(header)
    records = [_empty_record(row) for row in plan["rows"]]
    counts, active, accepted, phase = {"generation": 0}, None, 0, "startup"
    try:
        _write(root / "run.json", header)
        _write(root / "prepared.json", prepared)
        _write(root / "planned.json", records)
        e._directory(root / "rows", create=True)
        for index, (row, observation) in enumerate(
            zip(plan["rows"], prepared["observations"], strict=True)
        ):
            active, phase = index, "generation"
            row_root = e._directory(root / "rows" / f"{index:03d}", create=True)
            produced = _generation(
                model, tokenizer, plan, row, observation, row_root / "generation", run_sha, counts
            )
            phase = "generation_return"
            _write(row_root / "generation-return.json", _normal_return(run_sha, row, produced))
            score = _read(row_root / "generation/score.json")
            result = {
                "schema_version": SCHEMA,
                "status": "completed",
                "observation_id": row["observation_id"],
                "generation_return_sha256": e.object_hash(
                    _read(row_root / "generation-return.json")
                ),
                "score": score,
            }
            phase = "row_result"
            _write(row_root / "result.json", result)
            records[index] = _record(
                row,
                score,
                status="completed",
                generation_status="completed",
                result_sha256=e.object_hash(result),
            )
            accepted, active, phase = index + 1, None, "between_rows"
        summary = _summary(records, counts["generation"], True, "finished")
        analysis = tasks.aggregate_records(plan["rows"], records)
        terminal = {
            "schema_version": SCHEMA,
            "status": "finished",
            "run_sha256": run_sha,
            "records_sha256": e.object_hash(records),
            "analysis_sha256": e.object_hash(analysis),
            "summary": summary,
        }
        phase = "terminal_publication"
        _write(root / "records.json", records)
        _write(root / "analysis.json", analysis)
        _write(root / "terminal.json", terminal)
        return load_acquisition(root, plan)
    except BaseException as error:
        # Rebuild from durable receipts. Never promote the in-memory count if its
        # last dispatch lacks a completed/failure receipt, even if a count exists.
        try:
            replayed, count, exact = _replay_prefix(
                plan, prepared, root, run_sha, accepted, active, phase
            )
            require(count <= counts["generation"] <= MAX_ENTRIES, "observed_entry_budget")
            require(not exact or count == counts["generation"], "exact_failure_count")
            failure = {
                "schema_version": SCHEMA,
                "status": "failed",
                "run_sha256": run_sha,
                "failure_type": type(error).__name__,
                "active_index": active,
                "accepted_rows": accepted,
                "phase": phase,
                "observed_generation_entries": counts["generation"],
                "records": replayed,
                "summary": _summary(replayed, count, exact, "failed"),
                "analysis": tasks.aggregate_records(plan["rows"], replayed),
                "artifact_sha256": _inventory(root),
            }
            _write(root / "failure.json", failure)
        except BaseException as bookkeeping_error:
            try:
                _write(
                    root / "unverified-failure.json",
                    {
                        "schema_version": SCHEMA,
                        "status": "failed_unverified",
                        "run_sha256": run_sha,
                        "failure_type": type(error).__name__,
                        "bookkeeping_failure_type": type(bookkeeping_error).__name__,
                        "observed_generation_entries": counts["generation"],
                        "planned_generations": ROWS,
                        "records_unverified": records,
                    },
                )
            except BaseException:
                pass
        raise


def load_acquisition(directory, plan):
    """Replay once without model/tokenizer imports; interrupted counts stay honest."""
    plan = validate_plan(plan)
    root = e._directory(directory)
    require(not (root / "unverified-failure.json").exists(), "unverified_failure")
    header = _read(root / "run.json")
    e._keys(
        header,
        "schema_version plan_sha256 prepared_sha256 runtime_binding_sha256 pid sources",
        "run_fields",
    )
    e._int(header["pid"], 1)
    same(
        header,
        {
            "schema_version": SCHEMA,
            "plan_sha256": e.object_hash(plan),
            "prepared_sha256": header["prepared_sha256"],
            "runtime_binding_sha256": plan["bindings"]["runtime_binding_sha256"],
            "pid": header["pid"],
            "sources": source_bindings(),
        },
        "run_binding",
    )
    prepared = validate_prepared(plan, _read(root / "prepared.json"))
    require(header["prepared_sha256"] == e.object_hash(prepared), "prepared_binding")
    same(
        _read(root / "planned.json"), [_empty_record(row) for row in plan["rows"]], "planned_roster"
    )
    inventory = _inventory(root)
    failed = "failure.json" in inventory
    run_sha = e.object_hash(header)
    if failed:
        failure = _read(root / "failure.json")
        e._keys(
            failure,
            "schema_version status run_sha256 failure_type active_index accepted_rows phase observed_generation_entries records summary analysis artifact_sha256",
            "failure_fields",
        )
        require(
            failure["schema_version"] == SCHEMA
            and failure["status"] == "failed"
            and failure["run_sha256"] == run_sha,
            "failure_binding",
        )
        inventory.pop("failure.json")
        same(inventory, failure["artifact_sha256"], "failed_inventory")
        active, accepted, phase = (
            failure["active_index"],
            failure["accepted_rows"],
            failure["phase"],
        )
        e._int(accepted)
        require(accepted <= ROWS, "accepted_row_ceiling")
        require(active is None or type(active) is int and active == accepted < ROWS, "active_index")
        require(
            active is not None or phase in ("startup", "between_rows", "terminal_publication"),
            "failure_phase",
        )
        records, entries, exact = _replay_prefix(
            plan, prepared, root, run_sha, accepted, active, phase
        )
        observed = failure["observed_generation_entries"]
        e._int(observed, entries)
        require(observed <= MAX_ENTRIES, "observed_entry_ceiling")
        require(not exact or entries == observed, "failure_count_exact")
        same(failure["records"], records, "failed_records")
        summary = _summary(records, entries, exact, "failed")
        same(failure["summary"], summary, "failed_summary")
        analysis = tasks.aggregate_records(plan["rows"], records)
        same(failure["analysis"], analysis, "failed_analysis")
        return {
            "status": "failed",
            "records": records,
            "summary": summary,
            "analysis": analysis,
            "terminal_sha256": e.object_hash(failure),
        }
    require(
        {p.name for p in root.iterdir()}
        == {
            "run.json",
            "prepared.json",
            "planned.json",
            "rows",
            "records.json",
            "analysis.json",
            "terminal.json",
        },
        "root_inventory",
    )
    records, entries, exact = _replay_prefix(plan, prepared, root, run_sha, ROWS, None, "finished")
    summary = _summary(records, entries, exact, "finished")
    analysis = tasks.aggregate_records(plan["rows"], records)
    same(_read(root / "records.json"), records, "records_replay")
    same(_read(root / "analysis.json"), analysis, "analysis_replay")
    terminal = {
        "schema_version": SCHEMA,
        "status": "finished",
        "run_sha256": run_sha,
        "records_sha256": e.object_hash(records),
        "analysis_sha256": e.object_hash(analysis),
        "summary": summary,
    }
    same(_read(root / "terminal.json"), terminal, "terminal_replay")
    return {
        "status": "finished",
        "records": records,
        "summary": summary,
        "analysis": analysis,
        "terminal_sha256": e.object_hash(terminal),
    }
