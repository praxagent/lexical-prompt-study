"""A186 fixed matched-prefix acquisition; injected models, no asset loader.

The finite task roster and endpoint come from matched_prefix_tasks. Target
authorization, wall/RSS/storage supervision and sequential encoder loading are
the frozen caller's responsibility. Imports perform no model/tokenizer work.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import landmark_evidence as e
from . import matched_prefix_tasks as tasks
from . import native_prefix_qualification as previous
from . import prefix_only_capture as capture

SCHEMA = "a186-acquisition-v1"
A183_SHA = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
A184_SHA = "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69"
A185_SHA = "2542d54eaf839a253ae0b11c028a036ca0d6f697935dd456d0f4e3fc4d929526"
CAP, MAX_PROMPT, MAX_ENTRIES, LANDMARK = 64, 256, 7280, 8
DATE = "26 Jul 2024"
_raw, _values, _argmax = previous._raw, previous._values, previous._argmax
_write, _read = previous._write, previous._read


def require(ok, code):
    if not ok:
        raise ValueError("a186_acquisition_" + code)


def same(left, right, code):
    require(e.canonical(left) == e.canonical(right), code)


def source_sha256():
    return e.sha256(Path(__file__).read_bytes())


def source_bindings():
    for module, wanted in ((e, A183_SHA), (capture, A184_SHA), (previous, A185_SHA)):
        require(e.sha256(Path(module.__file__).read_bytes()) == wanted, "dependency_source")
    return {
        "a183": A183_SHA,
        "a184": A184_SHA,
        "a185": A185_SHA,
        "acquisition": source_sha256(),
        "tasks": e.sha256(Path(tasks.__file__).read_bytes()),
    }


def compile_plan(
    *,
    provenance,
    geometry,
    eos_token_ids,
    runtime_binding_sha256,
    protocol_sha256,
    tests_sha256,
    comparator_bindings,
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
    require(len(rows) == 112, "fixed_roster_count")
    sources = source_bindings()
    policy = {
        "cap": CAP,
        "argmax": "first_maximum_index",
        "cache": False,
        "attention": "all_ones",
        "positions": "sequential",
        "logits_to_keep": 1,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    endpoint = {
        "instrument_sha256": sources["tasks"],
        "target_sha256": e.object_hash(
            {
                "study": "a186",
                "native_eos_required": True,
                "ordered_ascii_integer_sums": 12,
                "sampled_cap": CAP,
                "protocol_sha256": protocol_sha256,
            }
        ),
        "horizon_tokens": CAP,
        "uncertainty_policy_sha256": e.object_hash(
            {"completed": "known_binary", "infrastructure": "missing_not_zero"}
        ),
    }
    manifest = {
        "schema_version": e.SCHEMA,
        "study_id": "a186_matched_prefix",
        "provenance": {**provenance, "capture_code_sha256": A184_SHA},
        "vocab_size": geometry["vocab_size"],
        "eos_token_ids": list(eos_token_ids),
        "generation": {
            "policy_sha256": e.object_hash(policy),
            "seed": 20260915,
            "max_new_tokens": CAP,
        },
        "landmarks": [LANDMARK],
        "count_convention": e.COUNT_CONVENTION,
        "decoder": dict(e.DECODER),
        "capture": {
            "layer_index": geometry["model_layers"] - 1,
            "model_layers": geometry["model_layers"],
            "hidden_width": geometry["hidden_width"],
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": {row["group_id"]: row["split"] for row in rows},
        "observations": [
            {
                **{key: row[key] for key in ("observation_id", "core_id", "group_id", "split")},
                "strata": {
                    "format": row["format"],
                    "task_strata_sha256": e.object_hash(row["strata"]),
                },
            }
            for row in rows
        ],
        "endpoint": endpoint,
        "comparators": e._snapshot(comparator_bindings),
    }
    e.validate_manifest(manifest)
    for descriptor in manifest["comparators"].values():
        require(
            descriptor["fit_budget"] == 1 and descriptor["calibration_budget"] == 0,
            "fixed_fit_budgets",
        )
    return {
        "schema_version": SCHEMA,
        "rows": rows,
        "manifest": manifest,
        "context_limit": geometry["context_limit"],
        "max_prompt_tokens": MAX_PROMPT,
        "max_entries": MAX_ENTRIES,
        "generation_policy": policy,
        "bindings": {
            "sources": sources,
            "runtime_binding_sha256": runtime_binding_sha256,
            "protocol_sha256": protocol_sha256,
            "tests_sha256": tests_sha256,
        },
    }


def validate_plan(plan):
    require(type(plan) is dict, "plan_type")
    m, b = plan["manifest"], plan["bindings"]
    expected = compile_plan(
        provenance={
            key: m["provenance"][key]
            for key in ("model_sha256", "tokenizer_sha256", "chat_template_sha256")
        },
        geometry={
            "vocab_size": m["vocab_size"],
            "hidden_width": m["capture"]["hidden_width"],
            "model_layers": m["capture"]["model_layers"],
            "context_limit": plan["context_limit"],
        },
        eos_token_ids=m["eos_token_ids"],
        runtime_binding_sha256=b["runtime_binding_sha256"],
        protocol_sha256=b["protocol_sha256"],
        tests_sha256=b["tests_sha256"],
        comparator_bindings=m["comparators"],
    )
    same(plan, expected, "fixed_plan")
    return e._snapshot(plan)


def prepare_inputs(plan, tokenizer):
    plan = validate_plan(plan)
    m = plan["manifest"]
    require(
        e.sha256(tokenizer.get_chat_template().encode()) == m["provenance"]["chat_template_sha256"],
        "template",
    )
    eos, special = tokenizer.eos_token_id, tokenizer.all_special_ids
    e._ids(special, m["vocab_size"], nonempty=True)
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
            e._ids(values, m["vocab_size"], nonempty=True)
            require(not set(values).intersection(special), "payload_special_token")
        kwargs = {"add_generation_prompt": True, "date_string": DATE}
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        e._str(text)
        for values in (ids, encoded):
            e._ids(values, m["vocab_size"], nonempty=True)
        require(
            ids == encoded and all(item["content"] in text for item in messages), "native_rendering"
        )
        require(
            len(ids) <= MAX_PROMPT and len(ids) + CAP <= plan["context_limit"], "native_context"
        )
        joint = tokenizer.encode(text + "probe", add_special_tokens=False)
        e._ids(joint, m["vocab_size"], nonempty=True)
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
        e._ids(closed, m["vocab_size"], nonempty=True)
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
    e._ids(special, plan["manifest"]["vocab_size"], nonempty=True)
    require(special == sorted(set(special)), "prepared_special")
    require(
        type(prepared["observations"]) is list and len(prepared["observations"]) == 112,
        "prepared_count",
    )
    for row, obs in zip(plan["rows"], prepared["observations"], strict=True):
        e._keys(
            obs,
            "observation_id messages rendered_prompt_text prompt_token_ids prompt_ids_sha256 prompt_utf8_sha256 intended_eos_token_id assistant_probe_token_ids payload_token_ids",
            "observation",
        )
        e._str(obs["rendered_prompt_text"])
        e._ids(obs["prompt_token_ids"], plan["manifest"]["vocab_size"], nonempty=True)
        require(obs["observation_id"] == row["observation_id"], "observation_identity")
        same(obs["messages"], row["messages"], "messages")
        require(
            len(obs["prompt_token_ids"]) <= MAX_PROMPT
            and len(obs["prompt_token_ids"]) + CAP <= plan["context_limit"],
            "prepared_context",
        )
        require(
            obs["prompt_ids_sha256"] == e.object_hash(obs["prompt_token_ids"])
            and obs["prompt_utf8_sha256"] == e.sha256(obs["rendered_prompt_text"].encode()),
            "prepared_hash",
        )
        eos = obs["intended_eos_token_id"]
        require(
            type(eos) is int and eos in plan["manifest"]["eos_token_ids"] and eos in special,
            "prepared_eos",
        )
        probe = obs["assistant_probe_token_ids"]
        e._ids(probe, plan["manifest"]["vocab_size"], nonempty=True)
        require(
            len(probe) > 1
            and probe[-1] == eos
            and not any(t in plan["manifest"]["eos_token_ids"] for t in probe[:-1]),
            "prepared_probe",
        )
        require(
            type(obs["payload_token_ids"]) is list
            and len(obs["payload_token_ids"]) == len(row["messages"]),
            "prepared_payloads",
        )
        for values in obs["payload_token_ids"]:
            e._ids(values, plan["manifest"]["vocab_size"], nonempty=True)
            require(not set(values).intersection(special), "prepared_payload_special")
    return e._snapshot(prepared)


def _inventory(root):
    found = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "artifact_symlink")
        if path.is_file():
            found[str(path.relative_to(root))] = e.sha256(e._read_file(path))
        else:
            require(path.is_dir(), "artifact_type")
    return found


def _call_attempt(plan, run_sha, index, step, values):
    return {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "plan_sha256": e.object_hash(plan),
        "sequence_index": index,
        "step_index": step,
        "input_token_ids": values,
        "input_ids_sha256": e.object_hash(values),
    }


def _forward(model, plan, directory, attempt, counts):
    import torch

    root = e._directory(directory, create=True)
    _write(root / "attempt.json", attempt)
    inputs = capture._tensor_inputs(torch, attempt["input_token_ids"])
    inputs.update(output_hidden_states=False, output_attentions=False)
    state = {"entries": 0, "model_call_returned": False, "cleanup_failure_type": None}
    handles, original = [], None

    def check(kwargs):
        require(
            kwargs.get("output_hidden_states") is False
            and kwargs.get("output_attentions") is False,
            "collector_flags",
        )
        capture._check_inputs(
            torch,
            {
                k: v
                for k, v in kwargs.items()
                if k not in ("output_hidden_states", "output_attentions")
            },
            attempt["input_token_ids"],
        )

    def entry(module, args, kwargs):
        state["entries"] += 1
        counts["generation"] += 1
        require(
            state["entries"] == 1
            and sum(counts.values()) <= MAX_ENTRIES
            and module is model
            and not args,
            "entry_ceiling",
        )
        capture._hook_audit(torch, model, handles)
        check(kwargs)
        require(
            torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cpu"),
            "inference_mode",
        )
        _write(
            root / "entry.json",
            {
                "schema_version": SCHEMA,
                "event": "model_forward_pre_hook_entry",
                "attempt_sha256": e.object_hash(attempt),
                "entry_index": 0,
            },
        )

    try:
        try:
            capture._hook_audit(torch, model)
            handles.append((model, "pre", model.register_forward_pre_hook(entry, with_kwargs=True)))
            with torch.inference_mode():
                output = model(**inputs)
            state["model_call_returned"] = True
            check(inputs)
            require(state["entries"] == 1, "one_entry")
            raw = _raw(torch, output.logits, (1, 1, plan["manifest"]["vocab_size"]))
        except BaseException as exc:
            original = exc
            raise
        finally:
            cleanup = None
            for module, _kind, handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as exc:
                    cleanup = cleanup or exc
                finally:
                    module._forward_pre_hooks.pop(handle.id, None)
                    module._forward_pre_hooks_with_kwargs.pop(handle.id, None)
            try:
                capture._hook_audit(torch, model)
            except BaseException as exc:
                cleanup = cleanup or exc
            if cleanup is not None:
                state["cleanup_failure_type"] = type(cleanup).__name__
                if original is None:
                    raise cleanup
        e._publish(root / "logits.fp32", raw)
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "attempt_sha256": e.object_hash(attempt),
            "entry_sha256": e.object_hash(_read(root / "entry.json")),
            "entries": 1,
            "model_call_returned": True,
            "logits_sha256": e.sha256(raw),
            "chosen_token_id": _argmax(raw, plan["manifest"]["vocab_size"]),
        }
        _write(root / "result.json", result)
        return result
    except BaseException as exc:
        try:
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "attempt_sha256": e.object_hash(attempt),
                    **state,
                    "failure_type": type(exc).__name__,
                    "usable": False,
                },
            )
        except BaseException:
            pass
        raise


def _generation(model, tokenizer, plan, row, observation, root, run_sha, counts):
    root = e._directory(root, create=True)
    attempt = {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "row_sha256": e.object_hash(row),
        "observation_sha256": e.object_hash(observation),
        "policy": plan["generation_policy"],
    }
    _write(root / "attempt.json", attempt)
    ids, entries, result_hashes = [], [], []
    for step in range(CAP):
        call = _call_attempt(
            plan, run_sha, row["sequence_index"], step, observation["prompt_token_ids"] + ids
        )
        result = _forward(model, plan, root / f"step_{step:02d}", call, counts)
        ids.append(result["chosen_token_id"])
        entries.append(result["entry_sha256"])
        result_hashes.append(e.object_hash(result))
        if ids[-1] in plan["manifest"]["eos_token_ids"]:
            break
    dispatch = {
        "schema_version": SCHEMA,
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": entries,
        "step_result_sha256": result_hashes,
    }
    _write(root / "dispatch.json", dispatch)
    generation = {
        "status": "completed",
        "stop_reason": "eos" if ids[-1] in plan["manifest"]["eos_token_ids"] else "cap",
        "token_ids": ids,
        "observed_tokens": len(ids),
        "attempt_sha256": e.object_hash(attempt),
        "dispatch_sha256": e.object_hash(dispatch),
    }
    _write(root / "result.json", generation)
    content = ids[:-1] if generation["stop_reason"] == "eos" else ids
    decoded = tokenizer.decode(content, **plan["manifest"]["decoder"])
    require(type(decoded) is str, "response_decode")
    e._publish(root / "response.utf8", decoded.encode())
    outcome = _outcome(plan, row, generation, decoded)
    _write(root / "outcome.json", outcome)
    available = len(content) >= LANDMARK
    prefix_text = (
        tokenizer.decode(content[:LANDMARK], **plan["manifest"]["decoder"]) if available else None
    )
    require(prefix_text is None or type(prefix_text) is str, "prefix_decode")
    if available:
        e._publish(root / "prefix.utf8", prefix_text.encode())
    prefix = {
        "landmark": LANDMARK,
        "status": "available" if available else "unreached",
        "reason": None if available else "early_eos",
        "token_ids": content[:LANDMARK] if available else None,
        "decoded_utf8_sha256": e.sha256(prefix_text.encode()) if available else None,
    }
    _write(root / "prefix.json", prefix)
    return {
        "generation_sha256": e.object_hash(generation),
        "response_sha256": e.sha256(decoded.encode()),
        "outcome_sha256": e.object_hash(outcome),
        "prefix_sha256": e.object_hash(prefix),
    }


def _outcome(plan, row, generation, decoded):
    scored = tasks.outcome_for_generation(
        row, generation, decoded, plan["manifest"]["eos_token_ids"]
    )
    require(
        scored["status"] == "known" and type(scored["label"]) is int and scored["label"] in (0, 1),
        "completed_endpoint",
    )
    return {
        **scored,
        "applicable": True,
        "disagreement": None,
        "endpoint": plan["manifest"]["endpoint"],
        "generation_sha256": e.object_hash(generation),
    }


def _generation_return(run_sha, row, value):
    return {
        "schema_version": SCHEMA,
        "event": "generation_stage_returned_normally",
        "run_sha256": run_sha,
        "observation_id": row["observation_id"],
        **value,
    }


def _empty_record(row):
    return {
        **{
            key: row[key]
            for key in ("observation_id", "core_index", "format", "split", "sequence_index")
        },
        "status": "unattempted",
        "generation_status": "missing",
        "capture_status": "missing",
        "prefix_available": None,
        "label": None,
        "result_sha256": None,
    }


def _generation_record(record, generation, prefix, outcome):
    record.update(
        generation_status=generation["status"],
        prefix_available=prefix["status"] == "available",
        label=outcome["label"],
        capture_status="missing" if prefix["status"] == "available" else "unavailable",
    )


def _summary(records, counts, status, capture_known):
    return {
        "planned_generations": 112,
        "planned_landmarks": 112,
        "completed_rows": sum(row["status"] == "completed" for row in records),
        "failed_rows": sum(row["status"] == "failed" for row in records),
        "unattempted_rows": sum(row["status"] == "unattempted" for row in records),
        "completed_generations": sum(row["generation_status"] == "completed" for row in records),
        "completed_captures": sum(row["capture_status"] == "completed" for row in records),
        "unavailable_captures": sum(row["capture_status"] == "unavailable" for row in records),
        "failed_captures": sum(row["capture_status"] == "infrastructure_failed" for row in records),
        "known_labels": sum(row["label"] is not None for row in records),
        "success_labels": sum(row["label"] == 1 for row in records),
        "entries": {**counts, "capture": counts["capture"] if capture_known else None},
        "entry_lower_bounds": dict(counts),
        "entries_complete": capture_known,
        "total_entries": sum(counts.values()) if capture_known else None,
        "schedule_finished": status == "finished",
    }


def _adapter_arguments(plan, observation, generation_root):
    obs = {key: observation[key] for key in ("observation_id", "messages", "prompt_token_ids")}
    obs["rendered_prompt_utf8"] = observation["rendered_prompt_text"].encode()
    prefix = _read(generation_root / "prefix.json")
    prefix.pop("decoded_utf8_sha256")
    prefix["decoded_utf8"] = (
        e._read_file(generation_root / "prefix.utf8") if prefix["status"] == "available" else None
    )
    return {
        "manifest": plan["manifest"],
        "observation": obs,
        "generation": _read(generation_root / "result.json"),
        "prefix": prefix,
        "outcome": _read(generation_root / "outcome.json"),
    }


def _capture_return(run_sha, row, result):
    return {
        "schema_version": SCHEMA,
        "event": "capture_prefix_returned_normally",
        "run_sha256": run_sha,
        "observation_id": row["observation_id"],
        "adapter_result_sha256": e.sha256(result.receipt_bytes),
        "bundle_sha256": result.bundle.sha256,
    }


def _failed_capture_count(root, arguments):
    """Return an exact declared count when bound; otherwise a proved lower bound."""
    manifest, observation, prefix = (
        arguments[key] for key in ("manifest", "observation", "prefix")
    )
    available = prefix["status"] == "available"
    values = observation["prompt_token_ids"] + (prefix["token_ids"] or [])
    initial = e.validate_landmark(
        **arguments,
        capture_metadata={
            "status": "missing" if available else "unavailable",
            "reason": "not_attempted" if available else "prefix_unavailable",
            "evidence": None,
        },
        capture_bytes=None,
    )

    def checked_attempt():
        attempt = _read(root / "attempt.json")
        capture._validate_attempt(attempt, manifest)
        expected = {
            "schema_version": capture.SCHEMA,
            "manifest_sha256": e.object_hash(manifest),
            "initial_bundle_sha256": initial.sha256,
            "adapter_source_sha256": A184_SHA,
            "evidence_source_sha256": A183_SHA,
            "observation_id": observation["observation_id"],
            "landmark": LANDMARK,
            "prefix_available": available,
            "input_ids_sha256": e.object_hash(values) if available else None,
            "input_length": len(values) if available else None,
            "generation_sha256": e.object_hash(arguments["generation"]),
            "provenance": manifest["provenance"],
        }
        same({key: attempt[key] for key in expected}, expected, "failed_current_attempt")
        return attempt

    try:
        failure, attempt = _read(root / "failure.json"), checked_attempt()
        require(
            failure["schema_version"] == capture.SCHEMA
            and failure["status"] == "failed"
            and failure["attempt_sha256"] == e.object_hash(attempt),
            "adapter_failure",
        )
        e._int(failure["dispatches"])
        return failure["dispatches"], True
    except BaseException:
        try:
            attempt, dispatch = checked_attempt(), _read(root / "dispatch.json")
            expected = {
                "schema_version": capture.SCHEMA,
                "attempt_sha256": e.object_hash(attempt),
                "event": "model_forward_pre_hook_entry",
                "input_ids_sha256": e.object_hash(values),
                "input_length": len(values),
                "dispatch_index": 0,
            }
            same(dispatch, expected, "adapter_dispatch")
            return 1, False
        except BaseException:
            return 0, False


def run_acquisition(model, tokenizer, *, plan, prepared, runtime_binding, directory):
    """Consume one new directory; hard failures stop and are rethrown unchanged.

    The caller must separately retain this function's normal return before
    encoding. A completed on-disk terminal cannot prove that return observation.
    """
    plan = validate_plan(plan)
    prepared = validate_prepared(plan, prepared)
    require(
        type(runtime_binding) is dict
        and e.object_hash(runtime_binding) == plan["bindings"]["runtime_binding_sha256"],
        "runtime_binding",
    )
    same(prepare_inputs(plan, tokenizer), prepared, "native_replay")
    previous._clean_model(model, plan)
    import torch

    require(
        torch.are_deterministic_algorithms_enabled() and not torch.cuda.is_initialized(),
        "deterministic_cpu",
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
    records = [_empty_record(row) for row in plan["rows"]]
    counts, capture_known = {"generation": 0, "capture": 0}, True
    active, phase = None, "startup"
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
                model,
                tokenizer,
                plan,
                row,
                observation,
                row_root / "generation",
                e.object_hash(header),
                counts,
            )
            phase = "generation_return"
            _write(
                row_root / "generation-return.json",
                _generation_return(e.object_hash(header), row, produced),
            )
            kwargs = _adapter_arguments(plan, observation, row_root / "generation")
            _generation_record(
                records[index], kwargs["generation"], kwargs["prefix"], kwargs["outcome"]
            )
            phase = "capture"
            if kwargs["prefix"]["status"] == "available":
                require(sum(counts.values()) < MAX_ENTRIES, "capture_ceiling")
            try:
                result = capture.capture_prefix(
                    model, blocks_path="model.layers", directory=row_root / "adapter", **kwargs
                )
            except BaseException:
                try:
                    n, known = _failed_capture_count(row_root / "adapter", kwargs)
                except BaseException:
                    # Failure bookkeeping cannot replace the original exception.
                    n, known = 0, False
                counts["capture"] += n
                capture_known = capture_known and known
                records[index]["capture_status"] = (
                    "infrastructure_failed"
                    if kwargs["prefix"]["status"] == "available"
                    else "unavailable"
                )
                raise
            counts["capture"] += result.receipt["dispatches"]
            phase = "capture_return"
            caller_return = _capture_return(e.object_hash(header), row, result)
            _write(row_root / "caller-return.json", caller_return)
            phase = "capture_validation"
            loaded = capture.load_capture(row_root / "adapter", plan["manifest"])
            require(
                loaded.bundle == result.bundle and loaded.receipt_bytes == result.receipt_bytes,
                "adapter_return_replay",
            )
            row_result = {
                "schema_version": SCHEMA,
                "status": "completed",
                "observation_id": row["observation_id"],
                "generation_return_sha256": e.object_hash(
                    _read(row_root / "generation-return.json")
                ),
                "caller_return_sha256": e.object_hash(caller_return),
                "adapter_result_sha256": e.sha256(result.receipt_bytes),
                "capture_status": result.receipt["status"],
            }
            phase = "row_result"
            _write(row_root / "result.json", row_result)
            records[index].update(
                status="completed",
                capture_status=row_result["capture_status"],
                result_sha256=e.object_hash(row_result),
            )
            active, phase = None, "between_rows"
        require(sum(counts.values()) <= MAX_ENTRIES, "entry_budget")
        terminal = {
            "schema_version": SCHEMA,
            "status": "finished",
            "run_sha256": e.object_hash(header),
            "records_sha256": e.object_hash(records),
            "summary": _summary(records, counts, "finished", capture_known),
            "failure_type": None,
        }
        phase = "terminal_publication"
        _write(root / "records.json", records)
        _write(root / "terminal.json", terminal)
        return load_acquisition(root, plan)
    except BaseException as exc:
        if active is not None:
            records[active]["status"] = "failed"
            records[active]["result_sha256"] = None
            if records[active]["generation_status"] == "missing":
                records[active]["generation_status"] = "infrastructure_failed"
            elif phase in ("capture", "capture_return", "capture_validation", "row_result"):
                records[active]["capture_status"] = (
                    "infrastructure_failed"
                    if records[active]["prefix_available"]
                    else "unavailable"
                )
        try:
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "run_sha256": e.object_hash(header),
                    "failure_type": type(exc).__name__,
                    "active_index": active,
                    "phase": phase,
                    "records": records,
                    "summary": _summary(records, counts, "failed", capture_known),
                    "artifact_sha256": _inventory(root),
                },
            )
        except BaseException:
            pass
        raise


def _replay_call(root, plan, attempt):
    same(_read(root / "attempt.json"), attempt, "call_attempt")
    expected_entry = {
        "schema_version": SCHEMA,
        "event": "model_forward_pre_hook_entry",
        "attempt_sha256": e.object_hash(attempt),
        "entry_index": 0,
    }
    same(_read(root / "entry.json"), expected_entry, "call_entry")
    require(
        {p.name for p in root.iterdir()}
        == {"attempt.json", "entry.json", "result.json", "logits.fp32"},
        "step_inventory",
    )
    raw = e._read_file(root / "logits.fp32")
    expected = {
        "schema_version": SCHEMA,
        "status": "completed",
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": e.object_hash(expected_entry),
        "entries": 1,
        "model_call_returned": True,
        "logits_sha256": e.sha256(raw),
        "chosen_token_id": _argmax(raw, plan["manifest"]["vocab_size"]),
    }
    same(_read(root / "result.json"), expected, "step_result")
    return expected


def _partial_generation_minimum(root, plan, row, observation, run_sha):
    """Count retained dispatch evidence without inventing missing completions."""
    if not root.exists():
        return 0
    e._directory(root)
    names = sorted(p.name for p in root.iterdir() if p.name.startswith("step_"))
    require(
        names == [f"step_{i:02d}" for i in range(len(names))] and len(names) <= CAP,
        "partial_step_order",
    )
    ids, total = [], 0
    for index, name in enumerate(names):
        step = e._directory(root / name)
        attempt = _call_attempt(
            plan, run_sha, row["sequence_index"], index, observation["prompt_token_ids"] + ids
        )
        same(_read(step / "attempt.json"), attempt, "partial_attempt")
        entry_exists = (step / "entry.json").exists()
        if entry_exists:
            same(
                _read(step / "entry.json"),
                {
                    "schema_version": SCHEMA,
                    "event": "model_forward_pre_hook_entry",
                    "attempt_sha256": e.object_hash(attempt),
                    "entry_index": 0,
                },
                "partial_entry",
            )
        if (step / "failure.json").exists():
            require(index == len(names) - 1, "failed_last_step")
            failure = _read(step / "failure.json")
            require(
                failure["schema_version"] == SCHEMA
                and failure["attempt_sha256"] == e.object_hash(attempt)
                and failure["usable"] is False
                and type(failure["model_call_returned"]) is bool,
                "partial_failure",
            )
            e._int(failure["entries"])
            require(int(entry_exists) <= failure["entries"], "partial_failed_entries")
            total += failure["entries"]
        elif (step / "result.json").exists():
            result = _replay_call(step, plan, attempt)
            total += 1
            ids.append(result["chosen_token_id"])
            require(
                ids[-1] not in plan["manifest"]["eos_token_ids"] or index == len(names) - 1,
                "partial_eos_end",
            )
        else:
            require(index == len(names) - 1, "incomplete_last_step")
            total += int(entry_exists)
    return total


def _replay_generation(plan, row, observation, root, run_sha):
    attempt = {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "row_sha256": e.object_hash(row),
        "observation_sha256": e.object_hash(observation),
        "policy": plan["generation_policy"],
    }
    same(_read(root / "attempt.json"), attempt, "generation_attempt")
    generation = _read(root / "result.json")
    content, _ = e._generation(plan["manifest"], generation)
    require(generation["status"] == "completed", "generation_complete")
    entries, results = [], []
    ids = generation["token_ids"]
    for j, token in enumerate(ids):
        call = _call_attempt(
            plan, run_sha, row["sequence_index"], j, observation["prompt_token_ids"] + ids[:j]
        )
        result = _replay_call(e._directory(root / f"step_{j:02d}"), plan, call)
        require(result["chosen_token_id"] == token, "chosen_id")
        entries.append(result["entry_sha256"])
        results.append(e.object_hash(result))
    dispatch = {
        "schema_version": SCHEMA,
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": entries,
        "step_result_sha256": results,
    }
    same(_read(root / "dispatch.json"), dispatch, "generation_dispatch")
    require(
        generation["attempt_sha256"] == e.object_hash(attempt)
        and generation["dispatch_sha256"] == e.object_hash(dispatch),
        "generation_binding",
    )
    decoded_raw = e._read_file(root / "response.utf8")
    outcome = _outcome(plan, row, generation, decoded_raw.decode("utf-8"))
    same(_read(root / "outcome.json"), outcome, "endpoint_replay")
    available = len(content) >= LANDMARK
    prefix_raw = e._read_file(root / "prefix.utf8") if available else None
    if prefix_raw is not None:
        prefix_raw.decode("utf-8")
    prefix = {
        "landmark": LANDMARK,
        "status": "available" if available else "unreached",
        "reason": None if available else "early_eos",
        "token_ids": content[:LANDMARK] if available else None,
        "decoded_utf8_sha256": e.sha256(prefix_raw) if available else None,
    }
    same(_read(root / "prefix.json"), prefix, "prefix_replay")
    expected_files = {
        "attempt.json",
        "dispatch.json",
        "result.json",
        "response.utf8",
        "outcome.json",
        "prefix.json",
    }
    require(
        {p.name for p in root.iterdir()}
        == expected_files
        | ({"prefix.utf8"} if available else set())
        | {f"step_{j:02d}" for j in range(len(ids))},
        "generation_inventory",
    )
    produced = {
        "generation_sha256": e.object_hash(generation),
        "response_sha256": e.sha256(decoded_raw),
        "outcome_sha256": e.object_hash(outcome),
        "prefix_sha256": e.object_hash(prefix),
    }
    same(
        _read(root.parent / "generation-return.json"),
        _generation_return(run_sha, row, produced),
        "generation_normal_return",
    )
    return generation, prefix, outcome


def _replay_row(plan, row, observation, root, run_sha):
    require(
        {p.name for p in root.iterdir()}
        == {"generation", "generation-return.json", "adapter", "caller-return.json", "result.json"},
        "row_inventory",
    )
    generation, prefix, outcome = _replay_generation(
        plan, row, observation, e._directory(root / "generation"), run_sha
    )
    adapter = capture.load_capture(root / "adapter", plan["manifest"])
    args = _adapter_arguments(plan, observation, root / "generation")
    expected_bundle = e.validate_landmark(
        **args,
        capture_metadata=adapter.bundle.metadata["capture"],
        capture_bytes=adapter.bundle.blobs.get("capture.fp32"),
    )
    require(expected_bundle == adapter.bundle, "adapter_same_boundary")
    caller = _capture_return(run_sha, row, adapter)
    same(_read(root / "caller-return.json"), caller, "adapter_normal_return")
    expected = {
        "schema_version": SCHEMA,
        "status": "completed",
        "observation_id": row["observation_id"],
        "generation_return_sha256": e.object_hash(_read(root / "generation-return.json")),
        "caller_return_sha256": e.object_hash(caller),
        "adapter_result_sha256": e.sha256(adapter.receipt_bytes),
        "capture_status": adapter.receipt["status"],
    }
    same(_read(root / "result.json"), expected, "row_result")
    record = _empty_record(row)
    _generation_record(record, generation, prefix, outcome)
    record.update(
        status="completed",
        capture_status=expected["capture_status"],
        result_sha256=e.object_hash(expected),
    )
    return record, len(generation["token_ids"]), adapter.receipt["dispatches"]


def load_acquisition(directory, plan):
    """Validate complete retained rows or a bounded hard-stop receipt, no inference.

    A failed root is never promoted to a finished acquisition or feature export.
    Partial counts include explicit declared lower bounds when receipt loss
    prevents exact adapter entry accounting.
    """
    plan = validate_plan(plan)
    root = e._directory(directory)
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
    require(header["prepared_sha256"] == e.object_hash(prepared), "native_prepared")
    records = [_empty_record(row) for row in plan["rows"]]
    same(_read(root / "planned.json"), records, "planned_cohort")
    failed = (root / "failure.json").exists() or (root / "failure.json").is_symlink()
    failure = _read(root / "failure.json") if failed else None
    if failed:
        e._keys(
            failure,
            "schema_version status run_sha256 failure_type active_index phase records summary artifact_sha256",
            "failure_fields",
        )
        require(
            failure["schema_version"] == SCHEMA
            and failure["status"] == "failed"
            and failure["run_sha256"] == e.object_hash(header),
            "failure_binding",
        )
        actual = _inventory(root)
        actual.pop("failure.json")
        same(actual, failure["artifact_sha256"], "failed_artifact_integrity")
        active = failure["active_index"]
        require(active is None or type(active) is int and 0 <= active < 112, "failed_active_index")
        accepted = (
            active
            if active is not None
            else sum(r["status"] == "completed" for r in failure["records"])
        )
        require(type(accepted) is int and 0 <= accepted <= 112, "accepted_rows")
    else:
        require(
            {p.name for p in root.iterdir()}
            == {
                "run.json",
                "prepared.json",
                "planned.json",
                "rows",
                "records.json",
                "terminal.json",
            },
            "root_inventory",
        )
        accepted = 112
        active = None
    expected_dirs = {f"{i:03d}" for i in range(accepted + int(active is not None))}
    require({p.name for p in (root / "rows").iterdir()} == expected_dirs, "schedule_prefix")
    counts = {"generation": 0, "capture": 0}
    for index in range(accepted):
        records[index], n, c = _replay_row(
            plan,
            plan["rows"][index],
            prepared["observations"][index],
            e._directory(root / "rows" / f"{index:03d}"),
            e.object_hash(header),
        )
        counts["generation"] += n
        counts["capture"] += c
    if failed:
        if active is not None:
            row_root = e._directory(root / "rows" / f"{active:03d}")
            phase = failure["phase"]
            require(
                phase
                in (
                    "generation",
                    "generation_return",
                    "capture",
                    "capture_return",
                    "capture_validation",
                    "row_result",
                ),
                "failed_phase",
            )
            if phase not in ("generation", "generation_return"):
                gen, prefix, outcome = _replay_generation(
                    plan,
                    plan["rows"][active],
                    prepared["observations"][active],
                    row_root / "generation",
                    e.object_hash(header),
                )
                _generation_record(records[active], gen, prefix, outcome)
                counts["generation"] += len(gen["token_ids"])
                try:
                    n, _known = _failed_capture_count(
                        row_root / "adapter",
                        _adapter_arguments(
                            plan, prepared["observations"][active], row_root / "generation"
                        ),
                    )
                except BaseException:
                    n = 0
                counts["capture"] += n
                records[active]["capture_status"] = (
                    "infrastructure_failed" if prefix["status"] == "available" else "unavailable"
                )
            else:
                records[active]["generation_status"] = "infrastructure_failed"
                counts["generation"] += _partial_generation_minimum(
                    row_root / "generation",
                    plan,
                    plan["rows"][active],
                    prepared["observations"][active],
                    e.object_hash(header),
                )
            records[active]["status"] = "failed"
        same(records, failure["records"], "failed_records")
        summary = failure["summary"]
        lower = summary["entry_lower_bounds"]
        e._keys(lower, "generation capture", "count_fields")
        for key, value in lower.items():
            e._int(value)
            require(value >= counts[key], "failed_count_lower_bound")
        require(type(summary["entries_complete"]) is bool, "known_count_flag")
        expected_summary = _summary(records, lower, "failed", summary["entries_complete"])
        same(summary, expected_summary, "failed_summary")
        return {
            "status": "failed",
            "summary": expected_summary,
            "records": records,
            "terminal_sha256": e.object_hash(failure),
        }
    require(sum(counts.values()) <= MAX_ENTRIES, "finished_entry_ceiling")
    summary = _summary(records, counts, "finished", True)
    same(_read(root / "records.json"), records, "records_replay")
    terminal = {
        "schema_version": SCHEMA,
        "status": "finished",
        "run_sha256": e.object_hash(header),
        "records_sha256": e.object_hash(records),
        "summary": summary,
        "failure_type": None,
    }
    same(_read(root / "terminal.json"), terminal, "terminal_replay")
    return {
        "status": "finished",
        "summary": summary,
        "records": records,
        "terminal_sha256": e.object_hash(terminal),
    }


def _feature_hash(content):
    """Hash only allowlisted feature content: bytes become lossless content pins."""

    def bind(value):
        if type(value) is bytes:
            return {"bytes_sha256": e.sha256(value), "byte_length": len(value)}
        if type(value) is dict:
            return {key: bind(item) for key, item in value.items()}
        if type(value) in (list, tuple):
            return [bind(item) for item in value]
        return value

    return e.object_hash(bind(content))


def feature_views(directory, plan):
    """Separate text-only packets, internal bytes and labels after complete replay.

    The caller must have observed run_acquisition return normally. Never feed the
    returned envelope or internal/label lists to the semantic encoder: feed only
    semantic_packets. A partial acquisition is retained but cannot be encoded.
    No full-generation hash, stop reason, future label or internal signal appears
    in a text packet or its feature hash.
    """
    plan = validate_plan(plan)
    loaded = load_acquisition(directory, plan)
    require(loaded["status"] == "finished", "feature_export_requires_finished")
    root = e._directory(directory)
    semantic, internal, labels = [], [], {}
    for index, row in enumerate(plan["rows"]):
        bundle = capture.load_capture(
            root / "rows" / f"{index:03d}" / "adapter", plan["manifest"]
        ).bundle
        common = None
        if bundle.metadata["prefix"]["status"] == "available":
            text = e.predictor_view(bundle, "text")["features"]
            check = tasks.prefix_check(text["prefix_utf8"].decode("utf-8"), row["expected_sums"])
            common = {
                "rendered_prompt_utf8": text["rendered_prompt_utf8"],
                "prompt_token_ids": list(text["prompt_token_ids"]),
                "prefix_utf8": text["prefix_utf8"],
                "prefix_token_ids": list(text["prefix_token_ids"]),
                "completed_fields": check["completed_fields"],
                "prefix_known_error": check["prefix_known_error"],
            }
        text_packet = {
            "schema_version": "a186-common-prefix-v1",
            "observation_id": row["observation_id"],
            "common": common,
        }
        semantic.append({**text_packet, "features_sha256": _feature_hash(text_packet)})
        raw = bundle.blobs.get("capture.fp32")
        vector_packet = {
            "schema_version": "a186-internal-prefix-v1",
            "observation_id": row["observation_id"],
            "residual_fp32le": raw,
            "residual_shape": [1, plan["manifest"]["capture"]["hidden_width"]]
            if raw is not None
            else None,
        }
        internal.append({**vector_packet, "features_sha256": _feature_hash(vector_packet)})
        labels[row["observation_id"]] = bundle.metadata["outcome"]["label"]
    return {"semantic_packets": semantic, "internal_packets": internal, "labels": labels}
