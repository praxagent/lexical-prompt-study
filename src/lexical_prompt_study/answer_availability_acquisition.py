"""A191 endpoint-only acquisition over a pinned CPU FP32 forward primitive.

Imports are model-free. The caller owns assets, process isolation, sampled
resources and independently profiled LM/decoder body counts. This coordinator
owns two technical dispatches and at most 1,024 scientific dispatches. No prefix
capture, encoder, fitting or old study runner is invoked.
"""

from pathlib import Path

from . import answer_availability_tasks as tasks
from . import landmark_evidence as e
from . import matched_prefix_acquisition as primitive
from . import native_prefix_qualification as previous
from . import prefix_only_capture as capture

SCHEMA = "a191-answer-availability-acquisition-v1"
CAP, ROWS, MAX_PROMPT, MAX_SCIENCE, MAX_ENTRIES = 64, 16, 256, 1024, 1026
DATE = "24 Sep 2026"
PINS = {
    "landmark_evidence": "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14",
    "prefix_only_capture": "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69",
    "native_prefix_qualification": "2542d54eaf839a253ae0b11c028a036ca0d6f697935dd456d0f4e3fc4d929526",
    "matched_prefix_acquisition": "7676fa455dc4ea1793808912562f16956bec6ec3adb34975cc157b97538ca7de",
}
_write, _read = previous._write, previous._read


def require(ok, code):
    if not ok:
        raise ValueError("a191_acquisition_" + code)


def same(a, b, code):
    require(e.canonical(a) == e.canonical(b), code)


def source_bindings():
    result = {}
    for module in (e, capture, previous, primitive):
        name = module.__name__.split(".")[-1]
        value = e.sha256(Path(module.__file__).read_bytes())
        require(value == PINS[name], "pinned_dependency")
        result[name] = value
    result["primitive_tasks"] = e.sha256(Path(primitive.tasks.__file__).read_bytes())
    result["tasks"] = e.sha256(Path(tasks.__file__).read_bytes())
    result["acquisition"] = e.sha256(Path(__file__).read_bytes())
    return result


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
    return {
        "schema_version": SCHEMA,
        "study_id": "a191_answer_availability",
        "rows": tasks.validate_roster(tasks.build_roster()),
        "provenance": e._snapshot(provenance),
        "geometry": e._snapshot(geometry),
        "eos_token_ids": list(eos_token_ids),
        "runtime_binding_sha256": runtime_binding_sha256,
        "protocol_sha256": protocol_sha256,
        "tests_sha256": tests_sha256,
        "sources": source_bindings(),
        "decoder": dict(e.DECODER),
        "chat_date": DATE,
        "max_prompt_tokens": MAX_PROMPT,
        "max_entries": MAX_ENTRIES,
        "max_scientific_entries": MAX_SCIENCE,
        "threads": 8,
        "interop_threads": 1,
        "generation_policy": {
            "cap": CAP,
            "argmax": "first_maximum_index",
            "cache": False,
            "attention": "all_ones",
            "positions": "sequential",
            "logits_to_keep": 1,
            "output_hidden_states": False,
            "output_attentions": False,
        },
        "step_receipt_schema": primitive.SCHEMA,
    }


def validate_plan(plan):
    require(type(plan) is dict, "plan_type")
    rebuilt = compile_plan(
        **{
            key: plan[key]
            for key in (
                "provenance",
                "geometry",
                "eos_token_ids",
                "runtime_binding_sha256",
                "protocol_sha256",
                "tests_sha256",
            )
        }
    )
    same(plan, rebuilt, "plan_reconstruction")
    return e._snapshot(plan)


def _primitive_plan(plan):
    g = plan["geometry"]
    return {
        "context_limit": g["context_limit"],
        "manifest": {
            "vocab_size": g["vocab_size"],
            "eos_token_ids": plan["eos_token_ids"],
            "capture": {
                "hidden_width": g["hidden_width"],
                "model_layers": g["model_layers"],
                "layer_index": g["model_layers"] - 1,
            },
        },
    }


def prepare_inputs(plan, tokenizer):
    plan = validate_plan(plan)
    observations = []
    for row in plan["rows"]:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, date_string=DATE
        )
        require(type(text) is str and text, "native_serialization")
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
        e._ids(values, plan["geometry"]["vocab_size"], nonempty=True)
        require(
            len(values) <= MAX_PROMPT and len(values) + CAP <= plan["geometry"]["context_limit"],
            "prompt_budget",
        )
        observations.append(
            {
                "row_sha256": e.object_hash(row),
                "prompt_utf8": text,
                "prompt_sha256": e.sha256(text.encode("utf-8")),
                "prompt_token_ids": values,
                "prompt_ids_sha256": e.object_hash(values),
            }
        )
    prepared = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "observations": observations,
    }
    return validate_prepared(plan, prepared)


def validate_prepared(plan, prepared):
    e._keys(prepared, "schema_version plan_sha256 observations", "prepared_fields")
    require(
        prepared["schema_version"] == SCHEMA and prepared["plan_sha256"] == e.object_hash(plan),
        "prepared_binding",
    )
    require(
        type(prepared["observations"]) is list and len(prepared["observations"]) == ROWS,
        "prepared_rows",
    )
    for row, obs in zip(plan["rows"], prepared["observations"], strict=True):
        e._keys(
            obs,
            "row_sha256 prompt_utf8 prompt_sha256 prompt_token_ids prompt_ids_sha256",
            "observation_fields",
        )
        require(type(obs["prompt_utf8"]) is str and obs["prompt_utf8"], "prompt_text")
        e._ids(obs["prompt_token_ids"], plan["geometry"]["vocab_size"], nonempty=True)
        require(
            len(obs["prompt_token_ids"]) <= MAX_PROMPT
            and len(obs["prompt_token_ids"]) + CAP <= plan["geometry"]["context_limit"],
            "prepared_context",
        )
        require(
            obs["row_sha256"] == e.object_hash(row)
            and obs["prompt_ids_sha256"] == e.object_hash(obs["prompt_token_ids"])
            and obs["prompt_sha256"] == e.sha256(obs["prompt_utf8"].encode("utf-8")),
            "prepared_hashes",
        )
    return e._snapshot(prepared)


def _preflight(model, plan):
    import torch

    require(
        torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 1, "eight_thread_regime"
    )
    require(
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "deterministic_regime",
    )
    require(not torch.is_autocast_enabled("cpu") and not model.training, "evaluation_regime")
    require(
        torch.backends.cuda.math_sdp_enabled()
        and not torch.backends.cuda.flash_sdp_enabled()
        and not torch.backends.cuda.mem_efficient_sdp_enabled()
        and not torch.backends.cuda.cudnn_sdp_enabled(),
        "math_attention_policy",
    )
    require(getattr(model.config, "_attn_implementation", None) == "sdpa", "sdpa_model")
    for attr, key in (
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_width"),
        ("num_hidden_layers", "model_layers"),
        ("max_position_embeddings", "context_limit"),
    ):
        same(getattr(model.config, attr), plan["geometry"][key], "model_geometry")
    for module in model.modules():
        require(not module.training, "all_modules_eval")
        require(
            not hasattr(module, "_orig_mod") and not hasattr(module, "_hf_hook"),
            "compiled_or_offload_module",
        )
    require(
        (
            getattr(model, "hf_device_map", None) in (None, {})
            or getattr(model, "hf_device_map", None) == {"": "cpu"}
        )
        and not getattr(model, "is_loaded_in_4bit", False)
        and not getattr(model, "is_loaded_in_8bit", False)
        and getattr(model, "quantization_method", None) is None,
        "unquantized_direct_model",
    )
    capture._hook_audit(torch, model)
    for value in (*model.parameters(), *model.buffers()):
        require(
            value.layout == torch.strided
            and not value.is_complex()
            and not value.is_quantized
            and value.device.type == "cpu"
            and (not value.is_floating_point() or value.dtype == torch.float32),
            "cpu_fp32_model",
        )


def _attempt(plan, run_sha, stage, index, step, values):
    return {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "plan_sha256": e.object_hash(plan),
        "stage": stage,
        "sequence_index": index,
        "step_index": step,
        "input_token_ids": values,
        "input_ids_sha256": e.object_hash(values),
    }


def _dispatch(model, plan, directory, attempt, counts):
    stage = attempt["stage"]
    require(stage in ("technical", "science"), "dispatch_stage")
    limit = 2 if stage == "technical" else MAX_SCIENCE
    require(
        counts[stage]["generation"] < limit
        and sum(c["generation"] for c in counts.values()) < MAX_ENTRIES,
        "entry_ceiling_before",
    )
    _preflight(model, plan)
    result = primitive._forward(model, _primitive_plan(plan), directory, attempt, counts[stage])
    require(
        counts[stage]["generation"] <= limit
        and sum(c["generation"] for c in counts.values()) <= MAX_ENTRIES,
        "entry_ceiling_after",
    )
    return result


def _record(row, *, status="unattempted", score=None):
    fields = ("observation_id", "core_index", "core_id", "group_id", "condition", "sequence_index")
    result = {key: row[key] for key in fields}
    result.update(
        status=status,
        generation_status="completed"
        if status == "completed"
        else "infrastructure_failed"
        if status == "failed"
        else "missing",
    )
    result.update(
        score
        or {
            "outcome_status": "missing",
            "label": None,
            "terminal_eos": None,
            "capped": None,
            "censored": None,
        }
    )
    return result


def _summary(records, counts, complete):
    tech, science = counts["technical"]["generation"], counts["science"]["generation"]
    return {
        "technical_entries": tech,
        "scientific_entries": science,
        "total_entries": tech + science,
        "entries_complete": complete,
        "planned_rows": ROWS,
        **{
            kind + "_rows": sum(r["status"] == kind for r in records)
            for kind in ("completed", "failed", "unattempted")
        },
    }


def _generation(model, tokenizer, plan, row, obs, directory, run_sha, counts):
    root = e._directory(directory, create=True)
    ids, step_hashes = [], []
    for step in range(CAP):
        attempt = _attempt(
            plan, run_sha, "science", row["sequence_index"], step, obs["prompt_token_ids"] + ids
        )
        result = _dispatch(model, plan, root / f"step_{step:02d}", attempt, counts)
        ids.append(result["chosen_token_id"])
        step_hashes.append(e.object_hash(result))
        if ids[-1] in plan["eos_token_ids"]:
            break
    generation = {
        "status": "completed",
        "stop_reason": "eos" if ids[-1] in plan["eos_token_ids"] else "cap",
        "token_ids": ids,
        "observed_tokens": len(ids),
    }
    content = ids[:-1] if generation["stop_reason"] == "eos" else ids
    text = tokenizer.decode(content, **plan["decoder"])
    require(type(text) is str, "decoded_text")
    score = tasks.score_generation(row, generation, text, plan["eos_token_ids"])
    e._publish(root / "response.utf8", text.encode("utf-8"))
    receipt = {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "row_sha256": e.object_hash(row),
        "generation": generation,
        "step_result_sha256": step_hashes,
        "response_sha256": e.sha256(text.encode("utf-8")),
        "score": score,
    }
    _write(root / "result.json", receipt)
    return receipt


def run_acquisition(model, tokenizer, *, plan, prepared, runtime_binding, directory, event=None):
    plan = validate_plan(plan)
    prepared = validate_prepared(plan, prepared)
    require(e.object_hash(runtime_binding) == plan["runtime_binding_sha256"], "runtime_binding")
    root = e._directory(directory, create=True)
    header = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "runtime_binding_sha256": e.object_hash(runtime_binding),
        "sources": source_bindings(),
    }
    _write(root / "run.json", header)
    _write(root / "prepared.json", prepared)
    run_sha = e.object_hash(header)
    counts = {"technical": {"generation": 0}, "science": {"generation": 0}}
    active, phase = None, "technical"
    records = [_record(row) for row in plan["rows"]]
    try:
        e._directory(root / "technical", create=True)
        if event is not None:
            event("technical")
        results = []
        for step in range(2):
            attempt = _attempt(
                plan, run_sha, "technical", 0, step, prepared["observations"][0]["prompt_token_ids"]
            )
            results.append(
                _dispatch(model, plan, root / "technical" / f"step_{step:02d}", attempt, counts)
            )
        require(
            results[0]["logits_sha256"] == results[1]["logits_sha256"]
            and results[0]["chosen_token_id"] == results[1]["chosen_token_id"],
            "technical_repeatability",
        )
        _write(
            root / "technical/result.json",
            {
                "schema_version": SCHEMA,
                "qualified": True,
                "call_result_sha256": [e.object_hash(x) for x in results],
            },
        )
        e._directory(root / "rows", create=True)
        phase = "science"
        if event is not None:
            event("science")
        for index, (row, obs) in enumerate(
            zip(plan["rows"], prepared["observations"], strict=True)
        ):
            active = index
            row_root = root / "rows" / f"{index:03d}"
            receipt = _generation(model, tokenizer, plan, row, obs, row_root, run_sha, counts)
            _write(
                row_root / "return.json",
                {"event": "generation_returned_normally", "result_sha256": e.object_hash(receipt)},
            )
            records[index] = _record(row, status="completed", score=receipt["score"])
            active = None
        phase = "terminal_publication"
        terminal = {
            "schema_version": SCHEMA,
            "status": "finished",
            "run_sha256": run_sha,
            "records": records,
            "summary": _summary(records, counts, True),
            "analysis": tasks.aggregate_records(plan["rows"], records),
            "artifact_sha256": primitive._inventory(root),
        }
        _write(root / "terminal.json", terminal)
        return load_acquisition(root, plan)
    except BaseException as exc:
        # Failure output is descriptive; the separate terminal verifier authenticates
        # any interrupted dispatch and preserves all unresolved endpoint slots.
        try:
            if active is not None:
                row_root = root / "rows" / f"{active:03d}"
                if (row_root / "return.json").exists():
                    completed = _read(row_root / "result.json")
                    same(
                        _read(row_root / "return.json"),
                        {
                            "event": "generation_returned_normally",
                            "result_sha256": e.object_hash(completed),
                        },
                        "interrupted_normal_return",
                    )
                    records[active] = _record(
                        plan["rows"][active], status="completed", score=completed["score"]
                    )
                else:
                    usable_ids, _, _, _ = _replay_steps(
                        row_root,
                        plan,
                        run_sha,
                        "science",
                        active,
                        prepared["observations"][active]["prompt_token_ids"],
                        completed=False,
                    )
                    partial_score = tasks.score_generation(
                        plan["rows"][active],
                        {
                            "status": "infrastructure_failed",
                            "token_ids": usable_ids,
                            "stop_reason": None,
                        },
                        None,
                        plan["eos_token_ids"],
                    )
                    records[active] = _record(
                        plan["rows"][active], status="failed", score=partial_score
                    )
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "run_sha256": run_sha,
                    "phase": phase,
                    "active_index": active,
                    "failure_type": type(exc).__name__,
                    "records": records,
                    "summary": _summary(records, counts, False),
                    "analysis": tasks.aggregate_records(plan["rows"], records),
                    "artifact_sha256": primitive._inventory(root),
                },
            )
        except BaseException:
            pass
        raise


def _replay_steps(root, plan, run_sha, stage, index, initial, *, completed):
    ids, hashes, count, exact = [], [], 0, True
    if not root.exists():
        require(not completed, "missing_completed_directory")
        return ids, hashes, count, exact
    e._directory(root)
    names = sorted(p.name for p in root.iterdir() if p.name.startswith("step_"))
    limit = 2 if stage == "technical" else CAP
    require(
        names == [f"step_{i:02d}" for i in range(len(names))] and len(names) <= limit, "step_order"
    )
    payloads = (
        {"result.json"} if stage == "technical" else {"result.json", "return.json", "response.utf8"}
    )
    require(
        {p.name for p in root.iterdir()}
        <= set(names) | payloads | {"." + x + ".tmp" for x in payloads},
        "generation_file_allowlist",
    )
    for step, name in enumerate(names):
        directory = root / name
        e._directory(directory)
        payloads = {"attempt.json", "entry.json", "result.json", "failure.json", "logits.fp32"}
        require(
            {p.name for p in directory.iterdir()}
            <= payloads | {"." + x + ".tmp" for x in payloads},
            "step_file_allowlist",
        )
        values = initial if stage == "technical" else initial + ids
        attempt = _attempt(plan, run_sha, stage, index, step, values)
        if not (directory / "attempt.json").exists():
            require(
                not completed
                and step == len(names) - 1
                and {p.name for p in directory.iterdir()} <= {".attempt.json.tmp"},
                "unpublished_attempt",
            )
            break
        same(_read(directory / "attempt.json"), attempt, "replay_attempt")
        if (directory / "entry.json").exists():
            same(
                _read(directory / "entry.json"),
                {
                    "schema_version": primitive.SCHEMA,
                    "event": "model_forward_pre_hook_entry",
                    "attempt_sha256": e.object_hash(attempt),
                    "entry_index": 0,
                },
                "partial_entry_binding",
            )
        if (directory / "failure.json").exists() or not (directory / "result.json").exists():
            require(not completed and step == len(names) - 1, "partial_last_only")
            observed = int((directory / "entry.json").exists())
            if (directory / "failure.json").exists():
                failure = _read(directory / "failure.json")
                e._keys(
                    failure,
                    "schema_version attempt_sha256 entries model_call_returned cleanup_failure_type failure_type usable",
                    "partial_failure_fields",
                )
                require(
                    failure["schema_version"] == primitive.SCHEMA
                    and failure["attempt_sha256"] == e.object_hash(attempt)
                    and failure["usable"] is False
                    and type(failure["entries"]) is int
                    and observed <= failure["entries"] <= 1
                    and type(failure["model_call_returned"]) is bool
                    and (not failure["model_call_returned"] or failure["entries"] == 1)
                    and (
                        failure["cleanup_failure_type"] is None
                        or type(failure["cleanup_failure_type"]) is str
                    )
                    and type(failure["failure_type"]) is str,
                    "partial_failure_binding",
                )
                observed = failure["entries"]
            elif observed == 0:
                exact = False
            count += observed
            break
        result = primitive._replay_call(directory, _primitive_plan(plan), attempt)
        count += 1
        hashes.append(e.object_hash(result))
        if stage == "science":
            ids.append(result["chosen_token_id"])
            require(ids[-1] not in plan["eos_token_ids"] or step == len(names) - 1, "eos_is_final")
    return ids, hashes, count, exact


def load_acquisition(directory, plan):
    """Producer replay; a later independent verifier remains a separate artifact."""
    plan = validate_plan(plan)
    root = e._directory(directory)
    payloads = {"run.json", "prepared.json", "terminal.json", "failure.json"}
    require(
        {p.name for p in root.iterdir()}
        <= {"technical", "rows"} | payloads | {"." + x + ".tmp" for x in payloads},
        "root_file_allowlist",
    )
    failed = (root / "failure.json").exists()
    name = "failure.json" if failed else "terminal.json"
    terminal = _read(root / name)
    require(
        terminal["schema_version"] == SCHEMA
        and terminal["status"] == ("failed" if failed else "finished"),
        "terminal_status",
    )
    inventory = primitive._inventory(root)
    inventory.pop(name)
    same(inventory, terminal["artifact_sha256"], "terminal_inventory")
    header = _read(root / "run.json")
    prepared = validate_prepared(plan, _read(root / "prepared.json"))
    same(
        header,
        {
            "schema_version": SCHEMA,
            "plan_sha256": e.object_hash(plan),
            "prepared_sha256": e.object_hash(prepared),
            "runtime_binding_sha256": plan["runtime_binding_sha256"],
            "sources": source_bindings(),
        },
        "run_header",
    )
    run_sha = e.object_hash(header)
    require(terminal["run_sha256"] == run_sha, "terminal_run")
    tech_completed = (root / "technical/result.json").exists()
    _, tech_hashes, tech_count, tech_exact = _replay_steps(
        root / "technical",
        plan,
        run_sha,
        "technical",
        0,
        prepared["observations"][0]["prompt_token_ids"],
        completed=tech_completed,
    )
    if tech_completed:
        require(tech_count == 2, "two_technical")
        a, b = (_read(root / "technical" / f"step_{i:02d}/result.json") for i in range(2))
        require(
            a["logits_sha256"] == b["logits_sha256"]
            and a["chosen_token_id"] == b["chosen_token_id"],
            "repeatability_replay",
        )
        same(
            _read(root / "technical/result.json"),
            {"schema_version": SCHEMA, "qualified": True, "call_result_sha256": tech_hashes},
            "technical_receipt",
        )
    require(failed or tech_completed, "technical_gate")
    records, science_count, science_exact = [], 0, True
    row_names = (
        sorted(p.name for p in (root / "rows").iterdir()) if (root / "rows").exists() else []
    )
    require(
        row_names == [f"{i:03d}" for i in range(len(row_names))] and len(row_names) <= ROWS,
        "attempted_row_prefix",
    )
    if failed:
        active = terminal["active_index"]
        require(
            active is None
            or type(active) is int
            and 0 <= active < ROWS
            and active in (len(row_names) - 1, len(row_names)),
            "final_active_row",
        )
    for index, (row, obs) in enumerate(zip(plan["rows"], prepared["observations"], strict=True)):
        row_root = root / "rows" / f"{index:03d}"
        completed = (row_root / "return.json").exists()
        if row_root.exists():
            require(tech_completed, "science_after_technical")
            require(
                completed or failed and terminal["active_index"] == index,
                "incomplete_final_active_only",
            )
        ids, step_hashes, entries, exact = _replay_steps(
            row_root, plan, run_sha, "science", index, obs["prompt_token_ids"], completed=completed
        )
        science_count += entries
        science_exact = science_exact and exact
        if completed:
            receipt = _read(row_root / "result.json")
            generation = {
                "status": "completed",
                "stop_reason": "eos" if ids and ids[-1] in plan["eos_token_ids"] else "cap",
                "token_ids": ids,
                "observed_tokens": len(ids),
            }
            raw = e._read_file(row_root / "response.utf8")
            score = tasks.score_generation(
                row, generation, raw.decode("utf-8"), plan["eos_token_ids"]
            )
            same(
                receipt,
                {
                    "schema_version": SCHEMA,
                    "run_sha256": run_sha,
                    "row_sha256": e.object_hash(row),
                    "generation": generation,
                    "step_result_sha256": step_hashes,
                    "response_sha256": e.sha256(raw),
                    "score": score,
                },
                "row_result",
            )
            same(
                _read(row_root / "return.json"),
                {"event": "generation_returned_normally", "result_sha256": e.object_hash(receipt)},
                "row_return",
            )
            records.append(_record(row, status="completed", score=score))
        else:
            require(failed, "missing_finished_row")
            is_active = terminal.get("active_index") == index
            score = tasks.score_generation(
                row,
                {
                    "status": "infrastructure_failed" if is_active else "unattempted",
                    "token_ids": ids,
                    "stop_reason": None,
                },
                None,
                plan["eos_token_ids"],
            )
            records.append(
                _record(row, status="failed" if is_active else "unattempted", score=score)
            )
    counts = {"technical": {"generation": tech_count}, "science": {"generation": science_count}}
    require(
        tech_count <= 2
        and science_count <= MAX_SCIENCE
        and tech_count + science_count <= MAX_ENTRIES,
        "replayed_budget",
    )
    same(records, terminal["records"], "record_replay")
    reconstructed = _summary(records, counts, not failed)
    reported = terminal["summary"]
    if not failed:
        same(reconstructed, reported, "summary_replay")
        summary = reconstructed
    else:
        require(
            type(reported) is dict and set(reported) == set(reconstructed), "failed_summary_fields"
        )
        for key in (
            "entries_complete",
            "planned_rows",
            "completed_rows",
            "failed_rows",
            "unattempted_rows",
        ):
            same(reported[key], reconstructed[key], "failed_summary_structure")
        for key, lower, exact, ceiling in (
            ("technical_entries", tech_count, tech_exact, 2),
            ("scientific_entries", science_count, science_exact, MAX_SCIENCE),
        ):
            require(
                type(reported[key]) is int
                and lower <= reported[key] <= min(ceiling, lower + int(not exact)),
                "failed_declared_dispatch_count",
            )
        require(
            type(reported["total_entries"]) is int
            and reported["total_entries"]
            == reported["technical_entries"] + reported["scientific_entries"]
            and reported["total_entries"] <= MAX_ENTRIES,
            "failed_declared_total",
        )
        # An unpublished final pre-hook receipt cannot establish whether that
        # dispatch entered. Keep the producer's declaration distinct from proof.
        summary = {
            **reconstructed,
            "technical_entries": tech_count if tech_exact else None,
            "scientific_entries": science_count if science_exact else None,
            "total_entries": tech_count + science_count if tech_exact and science_exact else None,
            "entries_complete": tech_exact and science_exact,
            "technical_entries_lower_bound": tech_count,
            "scientific_entries_lower_bound": science_count,
            "total_entries_lower_bound": tech_count + science_count,
        }
    analysis = tasks.aggregate_records(plan["rows"], records)
    same(analysis, terminal["analysis"], "analysis_replay")
    return {
        "status": terminal["status"],
        "records": records,
        "summary": summary,
        "reported_summary": reported,
        "analysis": analysis,
        "terminal_sha256": e.object_hash(terminal),
    }
