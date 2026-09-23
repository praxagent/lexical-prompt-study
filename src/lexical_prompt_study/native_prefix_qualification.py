"""A185 finite native integration core; injected models/tokenizers, no loading.

Two fixed harmless fixtures, at most 32 cache-free greedy steps and eight
capture/reference entries. This is technical qualification, not detector data
acquisition. Loader authentication, wall/RSS supervision and target authorization
belong to the separately frozen caller. Imports perform no model/tokenizer work.
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

from . import landmark_evidence as e
from . import prefix_only_capture as capture

SCHEMA = "a185-native-prefix-v1"
A183_SHA = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
A184_SHA = "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69"
CAP, MAX_PROMPT, MAX_ENTRIES = 16, 128, 40
LANDMARKS = (0, 8)
SYSTEM = "Follow the user's instruction."
USERS = (
    "Reply with the single word READY.",
    "Write the integers from 1 through 20 separated by single spaces.",
)
DATE = "26 Jul 2024"


def require(ok, code):
    if not ok:
        raise ValueError("a185_" + code)


def source_sha256():
    return e.sha256(Path(__file__).read_bytes())


def source_bindings():
    require(e.sha256(Path(e.__file__).read_bytes()) == A183_SHA, "a183_source")
    require(e.sha256(Path(capture.__file__).read_bytes()) == A184_SHA, "a184_source")
    return {"a183": A183_SHA, "a184": A184_SHA, "a185": source_sha256()}


def compile_plan(
    *,
    model_sha256,
    tokenizer_sha256,
    chat_template_sha256,
    vocab_size,
    hidden_width,
    model_layers,
    context_limit,
    eos_token_ids,
    protocol_sha256,
    tests_sha256,
    runtime_binding_sha256,
):
    for value in (
        model_sha256,
        tokenizer_sha256,
        chat_template_sha256,
        protocol_sha256,
        tests_sha256,
        runtime_binding_sha256,
    ):
        e._hash(value)
    for value in (vocab_size, hidden_width, model_layers, context_limit):
        e._int(value, 1)
    e._ids(eos_token_ids, vocab_size, nonempty=True)
    require(eos_token_ids == sorted(set(eos_token_ids)), "eos_order")
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

    def placeholder(name):
        return e.object_hash({"scope": "a185_apparatus_only_not_selected", "field": name})

    comparators = {
        name: {
            "model_sha256": placeholder(name + "_model"),
            "transform_sha256": placeholder(name + "_transform"),
            "train_manifest_sha256": placeholder("no_fit"),
            "calibration_manifest_sha256": placeholder("no_calibration"),
            "fit_group_ids": [],
            "calibration_group_ids": [],
            "fit_budget": 0,
            "calibration_budget": 0,
        }
        for name in e.COMPARATORS
    }
    manifest = {
        "schema_version": e.SCHEMA,
        "study_id": "a185_native_apparatus",
        "provenance": {
            "model_sha256": model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_template_sha256": chat_template_sha256,
            "capture_code_sha256": A184_SHA,
        },
        "vocab_size": vocab_size,
        "eos_token_ids": list(eos_token_ids),
        "generation": {
            "policy_sha256": e.object_hash(policy),
            "seed": 20260915,
            "max_new_tokens": CAP,
        },
        "landmarks": list(LANDMARKS),
        "count_convention": e.COUNT_CONVENTION,
        "decoder": dict(e.DECODER),
        "capture": {
            "layer_index": model_layers - 1,
            "model_layers": model_layers,
            "hidden_width": hidden_width,
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": {f"fixture_{i}": "evaluation" for i in range(2)},
        "observations": [
            {
                "observation_id": f"fixture_{i}",
                "core_id": f"fixture_{i}",
                "group_id": f"fixture_{i}",
                "split": "evaluation",
                "strata": {"scope": "apparatus_no_detector_evaluation"},
            }
            for i in range(2)
        ],
        "endpoint": {
            "instrument_sha256": placeholder("unmeasured_instrument"),
            "target_sha256": placeholder("unselected_target"),
            "horizon_tokens": CAP,
            "uncertainty_policy_sha256": placeholder("unselected_uncertainty"),
        },
        "comparators": comparators,
    }
    e.validate_manifest(manifest)
    return {
        "schema_version": SCHEMA,
        "bindings": {
            "sources": source_bindings(),
            "protocol_sha256": protocol_sha256,
            "tests_sha256": tests_sha256,
            "runtime_binding_sha256": runtime_binding_sha256,
        },
        "manifest": manifest,
        "context_limit": context_limit,
        "max_prompt_tokens": MAX_PROMPT,
        "max_entries": MAX_ENTRIES,
        "generation_policy": policy,
        "fixtures": [
            {
                "fixture_index": i,
                "observation_id": f"fixture_{i}",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            }
            for i, user in enumerate(USERS)
        ],
        "slots": [
            {"fixture_index": i, "landmark": t, "slot_id": f"fixture_{i}_t{t}"}
            for i in range(2)
            for t in LANDMARKS
        ],
    }


def validate_plan(plan):
    require(type(plan) is dict, "plan_type")
    m, b = plan["manifest"], plan["bindings"]
    expected = compile_plan(
        **{
            key: m["provenance"][key]
            for key in ("model_sha256", "tokenizer_sha256", "chat_template_sha256")
        },
        vocab_size=m["vocab_size"],
        hidden_width=m["capture"]["hidden_width"],
        model_layers=m["capture"]["model_layers"],
        context_limit=plan["context_limit"],
        eos_token_ids=m["eos_token_ids"],
        protocol_sha256=b["protocol_sha256"],
        tests_sha256=b["tests_sha256"],
        runtime_binding_sha256=b["runtime_binding_sha256"],
    )
    require(e.canonical(plan) == e.canonical(expected), "fixed_plan")
    return e._snapshot(plan)


def prepare_inputs(plan, tokenizer):
    plan = validate_plan(plan)
    manifest = plan["manifest"]
    require(
        e.sha256(tokenizer.get_chat_template().encode())
        == manifest["provenance"]["chat_template_sha256"],
        "template",
    )
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos in manifest["eos_token_ids"], "native_eos")
    special_ids = tokenizer.all_special_ids
    e._ids(special_ids, manifest["vocab_size"], nonempty=True)
    require(len(set(special_ids)) == len(special_ids) and eos in special_ids, "special_ids")
    special_ids = sorted(special_ids)
    observations = []
    for fixture in plan["fixtures"]:
        messages = fixture["messages"]
        payload_ids = [
            tokenizer.encode(message["content"], add_special_tokens=False) for message in messages
        ]
        for values in payload_ids:
            e._ids(values, manifest["vocab_size"], nonempty=True)
            require(not set(values).intersection(special_ids), "payload_special_token")
        kwargs = {"add_generation_prompt": True, "date_string": DATE}
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        e._str(text)
        e._ids(ids, manifest["vocab_size"], nonempty=True)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        e._ids(encoded, manifest["vocab_size"], nonempty=True)
        require(
            ids == encoded and all(m["content"] in text for m in messages),
            "native_rendering",
        )
        require(
            len(ids) <= MAX_PROMPT and len(ids) + CAP <= plan["context_limit"], "native_context"
        )
        joint = tokenizer.encode(text + "probe", add_special_tokens=False)
        e._ids(joint, manifest["vocab_size"], nonempty=True)
        require(
            joint[: len(ids)] == ids
            and len(joint) > len(ids)
            and not any(t in manifest["eos_token_ids"] for t in joint[len(ids) :]),
            "probe_boundary",
        )
        require(
            tokenizer.decode(joint[len(ids) :], **manifest["decoder"]) == "probe", "probe_decode"
        )
        closed = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": "probe"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            date_string=DATE,
        )
        e._ids(closed, manifest["vocab_size"], nonempty=True)
        require(closed == joint + [eos], "native_closed_eos")
        observations.append(
            {
                "observation_id": fixture["observation_id"],
                "messages": messages,
                "rendered_prompt_text": text,
                "prompt_token_ids": ids,
                "prompt_ids_sha256": e.object_hash(ids),
                "prompt_utf8_sha256": e.sha256(text.encode()),
                "intended_eos_token_id": eos,
                "assistant_probe_token_ids": joint[len(ids) :] + [eos],
                "payload_token_ids": payload_ids,
            }
        )
    return {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "special_token_ids": special_ids,
        "observations": observations,
    }


def _prepared(plan, prepared):
    e._keys(
        prepared, "schema_version plan_sha256 special_token_ids observations", "prepared_fields"
    )
    special_ids = prepared["special_token_ids"]
    e._ids(special_ids, plan["manifest"]["vocab_size"], nonempty=True)
    require(special_ids == sorted(set(special_ids)), "prepared_special_ids")
    require(
        prepared["schema_version"] == SCHEMA
        and prepared["plan_sha256"] == e.object_hash(plan)
        and type(prepared["observations"]) is list
        and len(prepared["observations"]) == 2,
        "prepared_binding",
    )
    for fixture, obs in zip(plan["fixtures"], prepared["observations"], strict=True):
        e._keys(
            obs,
            "observation_id messages rendered_prompt_text prompt_token_ids prompt_ids_sha256 prompt_utf8_sha256 intended_eos_token_id assistant_probe_token_ids payload_token_ids",
            "observation_fields",
        )
        e._ids(obs["prompt_token_ids"], plan["manifest"]["vocab_size"], nonempty=True)
        e._str(obs["rendered_prompt_text"])
        require(
            obs["observation_id"] == fixture["observation_id"]
            and e.canonical(obs["messages"]) == e.canonical(fixture["messages"]),
            "fixture_identity",
        )
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
        require(
            type(obs["intended_eos_token_id"]) is int
            and obs["intended_eos_token_id"] in plan["manifest"]["eos_token_ids"],
            "prepared_eos",
        )
        require(obs["intended_eos_token_id"] in special_ids, "prepared_special_eos")
        require(
            type(obs["payload_token_ids"]) is list and len(obs["payload_token_ids"]) == 2,
            "prepared_payloads",
        )
        for values in obs["payload_token_ids"]:
            e._ids(values, plan["manifest"]["vocab_size"], nonempty=True)
            require(not set(values).intersection(special_ids), "prepared_payload_special")
        e._ids(obs["assistant_probe_token_ids"], plan["manifest"]["vocab_size"], nonempty=True)
        require(
            obs["assistant_probe_token_ids"][-1] == obs["intended_eos_token_id"]
            and not any(
                t in plan["manifest"]["eos_token_ids"]
                for t in obs["assistant_probe_token_ids"][:-1]
            ),
            "prepared_probe",
        )
    return e._snapshot(prepared)


def _write(path, value):
    e._publish(path, e.canonical(value))


def _read(path):
    return e._read_json(e._read_file(path))


def _raw(torch, tensor, shape):
    require(
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tensor.layout == torch.strided
        and tuple(tensor.shape) == tuple(shape),
        "actual_tensor",
    )
    require(bool(torch.isfinite(tensor).all().item()), "nonfinite_tensor")
    return bytes(tensor.detach().clone().contiguous().view(torch.uint8).reshape(-1).tolist())


def _values(raw, width):
    require(type(raw) is bytes and len(raw) == 4 * width, "raw_shape")
    values = [x[0] for x in struct.iter_unpack("<f", raw)]
    require(all(math.isfinite(x) for x in values), "raw_nonfinite")
    return values


def _argmax(raw, width):
    values = _values(raw, width)
    return max(range(width), key=values.__getitem__)


def _clean_model(model, plan):
    import torch

    require(model.config.max_position_embeddings == plan["context_limit"], "model_context")
    require(
        model.config.output_hidden_states is False and model.config.output_attentions is False,
        "lazy_collectors",
    )
    require(
        getattr(model.config, "_attn_implementation", None) == "sdpa", "attention_implementation"
    )
    capture._model_preflight(torch, model, "model.layers", plan["manifest"], 1)
    require(model.get_submodule("model.norm") is not model, "norm_module")


def _forward(model, plan, directory, attempt, counts, *, reference=False):
    import torch

    directory = e._directory(directory, create=True)
    _write(directory / "attempt.json", attempt)
    inputs = capture._tensor_inputs(torch, attempt["input_token_ids"])
    inputs.update(output_hidden_states=False, output_attentions=False)
    state = {
        "entries": 0,
        "pre": 0,
        "post": 0,
        "pre_bytes": None,
        "post_bytes": None,
        "model_call_returned": False,
        "cleanup_failure_type": None,
    }
    handles, original = [], None
    phase = "reference" if reference else "generation"
    capture._hook_audit(torch, model)
    norm = model.get_submodule("model.norm")

    def entry(module, args, kwargs):
        state["entries"] += 1
        counts[phase] += 1
        require(
            state["entries"] == 1
            and sum(counts.values()) <= MAX_ENTRIES
            and module is model
            and not args,
            "entry_ceiling",
        )
        capture._hook_audit(torch, model, handles)
        require(
            kwargs.get("output_hidden_states") is False
            and kwargs.get("output_attentions") is False,
            "collector_kwargs",
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
        require(
            torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cpu"),
            "inference_mode",
        )
        _write(
            directory / "entry.json",
            {
                "schema_version": SCHEMA,
                "event": "model_forward_pre_hook_entry",
                "attempt_sha256": e.object_hash(attempt),
                "phase": phase,
                "entry_index": 0,
            },
        )

    def before(module, args):
        state["pre"] += 1
        require(module is norm and state["pre"] == 1 and len(args) == 1, "norm_pre")
        tensor = args[0]
        require(
            tuple(tensor.shape)
            == (1, len(attempt["input_token_ids"]), plan["manifest"]["capture"]["hidden_width"]),
            "norm_input_shape",
        )
        state["pre_bytes"] = _raw(
            torch, tensor[:, -1, :], (1, plan["manifest"]["capture"]["hidden_width"])
        )

    def after(module, args, output):
        state["post"] += 1
        require(module is norm and state["post"] == 1, "norm_post")
        require(
            tuple(output.shape)
            == (1, len(attempt["input_token_ids"]), plan["manifest"]["capture"]["hidden_width"]),
            "norm_output_shape",
        )
        state["post_bytes"] = _raw(
            torch, output[:, -1, :], (1, plan["manifest"]["capture"]["hidden_width"])
        )

    try:
        try:
            handles.append((model, "pre", model.register_forward_pre_hook(entry, with_kwargs=True)))
            if reference:
                handles.append((norm, "pre", norm.register_forward_pre_hook(before)))
                handles.append((norm, "post", norm.register_forward_hook(after)))
            with torch.inference_mode():
                output = model(**inputs)
            state["model_call_returned"] = True
            require(
                inputs["output_hidden_states"] is False and inputs["output_attentions"] is False,
                "returned_collector_flags",
            )
            capture._check_inputs(
                torch,
                {
                    k: v
                    for k, v in inputs.items()
                    if k not in ("output_hidden_states", "output_attentions")
                },
                attempt["input_token_ids"],
            )
            require(
                state["entries"] == 1 and (not reference or state["pre"] == state["post"] == 1),
                "incomplete_forward",
            )
            raw = _raw(torch, output.logits, (1, 1, plan["manifest"]["vocab_size"]))
        except BaseException as exc:
            original = exc
            raise
        finally:
            cleanup = None
            for module, kind, handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as exc:
                    cleanup = cleanup or exc
                finally:
                    for name in (
                        ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs")
                        if kind == "pre"
                        else (
                            "_forward_hooks",
                            "_forward_hooks_with_kwargs",
                            "_forward_hooks_always_called",
                        )
                    ):
                        getattr(module, name).pop(handle.id, None)
            try:
                capture._hook_audit(torch, model)
            except BaseException as exc:
                cleanup = cleanup or exc
            if cleanup is not None:
                state["cleanup_failure_type"] = type(cleanup).__name__
            if cleanup is not None and original is None:
                raise cleanup
        blobs = {"logits.fp32": raw}
        if reference:
            blobs.update(
                {"norm_input.fp32": state["pre_bytes"], "norm_output.fp32": state["post_bytes"]}
            )
        for name, value in blobs.items():
            e._publish(directory / name, value)
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "attempt_sha256": e.object_hash(attempt),
            "entry_sha256": e.object_hash(_read(directory / "entry.json")),
            "entries": 1,
            "model_call_returned": True,
            "files": {name: e.sha256(value) for name, value in blobs.items()},
            "chosen_token_id": None if reference else _argmax(raw, plan["manifest"]["vocab_size"]),
            "norm_pre_count": state["pre"],
            "norm_post_count": state["post"],
        }
        _write(directory / "result.json", result)
        return result
    except BaseException as exc:
        try:
            _write(
                directory / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "failure_type": type(exc).__name__,
                    "entries": state["entries"],
                    "model_call_returned": state["model_call_returned"],
                    "norm_pre_count": state["pre"],
                    "norm_post_count": state["post"],
                    "cleanup_failure_type": state["cleanup_failure_type"],
                    "attempt_sha256": e.object_hash(attempt),
                    "capture_usable": False,
                },
            )
        except BaseException:
            pass
        raise


def _call_attempt(plan, run_sha, fixture, values, phase, index):
    return {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "plan_sha256": e.object_hash(plan),
        "fixture_index": fixture,
        "phase": phase,
        "index": index,
        "input_token_ids": values,
        "input_ids_sha256": e.object_hash(values),
    }


def _generation(model, plan, observation, fixture, root, run_sha, counts):
    root = e._directory(root, create=True)
    attempt = {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "fixture_index": fixture,
        "observation_sha256": e.object_hash(observation),
        "policy": plan["generation_policy"],
    }
    _write(root / "attempt.json", attempt)
    ids, entries, results = [], [], []
    for step in range(CAP):
        call = _call_attempt(
            plan, run_sha, fixture, observation["prompt_token_ids"] + ids, "generation", step
        )
        result = _forward(model, plan, root / f"step_{step:02d}", call, counts)
        ids.append(result["chosen_token_id"])
        entries.append(result["entry_sha256"])
        results.append(e.object_hash(result))
        if ids[-1] in plan["manifest"]["eos_token_ids"]:
            break
    ledger = {
        "schema_version": SCHEMA,
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": entries,
        "step_result_sha256": results,
    }
    _write(root / "dispatch.json", ledger)
    generation = {
        "status": "completed",
        "stop_reason": "eos" if ids[-1] in plan["manifest"]["eos_token_ids"] else "cap",
        "token_ids": ids,
        "observed_tokens": len(ids),
        "attempt_sha256": e.object_hash(attempt),
        "dispatch_sha256": e.object_hash(ledger),
    }
    _write(root / "result.json", generation)
    return generation


def _outcome(plan, generation):
    tokens = generation["token_ids"]
    eos = bool(tokens and tokens[-1] in plan["manifest"]["eos_token_ids"])
    content = tokens[:-1] if eos else tokens
    return {
        "status": "missing",
        "applicable": None,
        "label": None,
        "reason": "not_measured",
        "disagreement": None,
        "endpoint": plan["manifest"]["endpoint"],
        "generation_sha256": e.object_hash(generation),
        "terminal_eos": eos,
        "capped": generation["stop_reason"] == "cap",
        "censored": not eos and len(content) < CAP,
    }


def _landmark_arguments(plan, prepared, generation, landmark, tokenizer):
    obs = {key: prepared[key] for key in ("observation_id", "messages", "prompt_token_ids")}
    obs["rendered_prompt_utf8"] = prepared["rendered_prompt_text"].encode()
    tokens = generation["token_ids"]
    eos = bool(tokens and tokens[-1] in plan["manifest"]["eos_token_ids"])
    content = tokens[:-1] if eos else tokens
    available = len(content) >= landmark
    decoded = (
        tokenizer.decode(content[:landmark], **plan["manifest"]["decoder"]) if available else None
    )
    require(decoded is None or type(decoded) is str, "prefix_decode")
    prefix = {
        "landmark": landmark,
        "status": "available" if available else "unreached",
        "reason": None if available else "early_eos",
        "token_ids": content[:landmark] if available else None,
        "decoded_utf8": decoded.encode() if available else None,
    }
    outcome = _outcome(plan, generation)
    return {
        "manifest": plan["manifest"],
        "observation": obs,
        "generation": generation,
        "prefix": prefix,
        "outcome": outcome,
    }


def _empty_records(plan):
    return [
        {
            **slot,
            "status": "unattempted",
            "capture_equal": None,
            "logits_equal": None,
            "result_sha256": None,
        }
        for slot in plan["slots"]
    ]


def _summary(records, counts, status, *, capture_count_known=True):
    compared = [row for row in records if row["status"] == "completed"]
    known_failure = status == "failed" or any(
        not row["capture_equal"] or not row["logits_equal"] for row in compared
    )
    resolved = all(row["status"] in ("completed", "unavailable") for row in records)
    controls = sum(
        row["landmark"] == 0 and row["capture_equal"] is True and row["logits_equal"] is True
        for row in compared
    )
    t8 = sum(
        row["landmark"] == 8 and row["capture_equal"] is True and row["logits_equal"] is True
        for row in compared
    )
    qualified = False if known_failure else True if resolved and controls == 2 and t8 >= 1 else None
    return {
        "planned_slots": 4,
        "completed_comparisons": len(compared),
        "unavailable_slots": sum(r["status"] == "unavailable" for r in records),
        "failed_slots": sum(r["status"] == "failed" for r in records),
        "unattempted_slots": sum(r["status"] == "unattempted" for r in records),
        "entries": {**counts, "capture": counts["capture"] if capture_count_known else None},
        "entry_lower_bounds": dict(counts),
        "entries_complete": capture_count_known,
        "total_entries": sum(counts.values()) if capture_count_known else None,
        "t0_exact": controls,
        "t8_exact": t8,
        "qualified": qualified,
    }


def run_qualification(model, tokenizer, *, plan, prepared, runtime_binding, directory):
    """One exclusive run. Hard failures stop; comparison mismatches do not gate.

    A successful return is separately recorded by the caller; retained publication
    alone cannot prove that caller observed a normal return. No loader is invoked.
    """
    import torch

    plan = validate_plan(plan)
    prepared = _prepared(plan, prepared)
    require(
        type(runtime_binding) is dict
        and e.object_hash(runtime_binding) == plan["bindings"]["runtime_binding_sha256"],
        "runtime_binding",
    )
    require(e.canonical(prepare_inputs(plan, tokenizer)) == e.canonical(prepared), "native_replay")
    _clean_model(model, plan)
    root = e._directory(directory, create=True)
    header = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "runtime_binding_sha256": e.object_hash(runtime_binding),
        "pid": os.getpid(),
        "sources": source_bindings(),
    }
    _write(root / "run.json", header)
    _write(root / "prepared.json", prepared)
    records = _empty_records(plan)
    counts = {"generation": 0, "capture": 0, "reference": 0}
    active = None
    capture_count_known = True
    try:
        for i, observation in enumerate(prepared["observations"]):
            generation = _generation(
                model, plan, observation, i, root / f"generation_{i}", e.object_hash(header), counts
            )
            for landmark in LANDMARKS:
                active = records[i * 2 + LANDMARKS.index(landmark)]
                slot_root = e._directory(root / active["slot_id"], create=True)
                kwargs = _landmark_arguments(plan, observation, generation, landmark, tokenizer)
                require(sum(counts.values()) < MAX_ENTRIES, "capture_entry_budget")
                try:
                    result = capture.capture_prefix(
                        model, blocks_path="model.layers", directory=slot_root / "adapter", **kwargs
                    )
                except BaseException:
                    try:
                        failure = _read(slot_root / "adapter" / "failure.json")
                        attempt = _read(slot_root / "adapter" / "attempt.json")
                        require(
                            failure["schema_version"] == capture.SCHEMA
                            and failure["status"] == "failed"
                            and failure["attempt_sha256"] == e.object_hash(attempt)
                            and type(failure["dispatches"]) is int
                            and failure["dispatches"] >= 0,
                            "adapter_failure_binding",
                        )
                        counts["capture"] += failure["dispatches"]
                    except BaseException:
                        capture_count_known = False
                        try:
                            attempt = _read(slot_root / "adapter" / "attempt.json")
                            entry = _read(slot_root / "adapter" / "dispatch.json")
                            values = observation["prompt_token_ids"] + (
                                kwargs["prefix"]["token_ids"] or []
                            )
                            expected_entry = {
                                "schema_version": capture.SCHEMA,
                                "attempt_sha256": e.object_hash(attempt),
                                "event": "model_forward_pre_hook_entry",
                                "input_ids_sha256": e.object_hash(values),
                                "input_length": len(values),
                                "dispatch_index": 0,
                            }
                            if e.canonical(entry) == e.canonical(expected_entry):
                                counts["capture"] += 1
                        except BaseException:
                            pass
                    raise
                counts["capture"] += result.receipt["dispatches"]
                caller_return = {
                    "schema_version": SCHEMA,
                    "event": "capture_prefix_returned_normally",
                    "run_sha256": e.object_hash(header),
                    "slot": {k: active[k] for k in ("fixture_index", "landmark", "slot_id")},
                    "adapter_result_sha256": e.sha256(result.receipt_bytes),
                    "bundle_sha256": result.bundle.sha256,
                }
                _write(slot_root / "caller-return.json", caller_return)
                loaded = capture.load_capture(slot_root / "adapter", plan["manifest"])
                require(
                    loaded.bundle == result.bundle and loaded.receipt_bytes == result.receipt_bytes,
                    "adapter_return_replay",
                )
                equal_capture = equal_logits = None
                adapter_logits_sha = reference_result_sha = None
                if result.receipt["status"] == "completed":
                    adapter_logits = _raw(
                        torch, result.forward_output.logits, (1, 1, plan["manifest"]["vocab_size"])
                    )
                    e._publish(slot_root / "adapter-logits.fp32", adapter_logits)
                    adapter_logits_sha = e.sha256(adapter_logits)
                    values = observation["prompt_token_ids"] + kwargs["prefix"]["token_ids"]
                    call = _call_attempt(
                        plan, e.object_hash(header), i, values, "reference", landmark
                    )
                    reference = _forward(
                        model, plan, slot_root / "reference", call, counts, reference=True
                    )
                    reference_result_sha = e.object_hash(reference)
                    equal_capture = result.bundle.blobs["capture.fp32"] == e._read_file(
                        slot_root / "reference" / "norm_input.fp32"
                    )
                    equal_logits = adapter_logits == e._read_file(
                        slot_root / "reference" / "logits.fp32"
                    )
                    require(reference["entries"] == 1, "reference_count")
                slot_result = {
                    "schema_version": SCHEMA,
                    "status": "completed" if equal_capture is not None else "unavailable",
                    "caller_return_sha256": e.object_hash(caller_return),
                    "adapter_result_sha256": e.sha256(result.receipt_bytes),
                    "capture_equal": equal_capture,
                    "logits_equal": equal_logits,
                    "adapter_logits_sha256": adapter_logits_sha,
                    "reference_result_sha256": reference_result_sha,
                }
                _write(slot_root / "result.json", slot_result)
                active.update(
                    status=slot_result["status"],
                    capture_equal=equal_capture,
                    logits_equal=equal_logits,
                    result_sha256=e.object_hash(slot_result),
                )
                active = None
        require(sum(counts.values()) <= MAX_ENTRIES, "total_entry_budget")
        terminal = {
            "schema_version": SCHEMA,
            "status": "finished",
            "run_sha256": e.object_hash(header),
            "records_sha256": e.object_hash(records),
            "summary": _summary(records, counts, "finished"),
            "failure_type": None,
        }
        _write(root / "records.json", records)
        _write(root / "terminal.json", terminal)
        return load_qualification(root, plan)
    except BaseException as exc:
        if active is not None:
            active["status"] = "failed"
        try:
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "failure_type": type(exc).__name__,
                    "records": records,
                    "summary": _summary(
                        records, counts, "failed", capture_count_known=capture_count_known
                    ),
                    "run_sha256": e.object_hash(header),
                },
            )
        except BaseException:
            pass
        raise


def _replay_call(root, plan, attempt, *, reference=False):
    require(e.canonical(_read(root / "attempt.json")) == e.canonical(attempt), "call_attempt")
    entry = {
        "schema_version": SCHEMA,
        "event": "model_forward_pre_hook_entry",
        "attempt_sha256": e.object_hash(attempt),
        "phase": "reference" if reference else "generation",
        "entry_index": 0,
    }
    require(e.canonical(_read(root / "entry.json")) == e.canonical(entry), "call_entry")
    result = _read(root / "result.json")
    names = {"logits.fp32", "norm_input.fp32", "norm_output.fp32"} if reference else {"logits.fp32"}
    require(
        {p.name for p in root.iterdir()} == names | {"attempt.json", "entry.json", "result.json"},
        "call_inventory",
    )
    blobs = {name: e._read_file(root / name) for name in names}
    for name, raw in blobs.items():
        _values(
            raw,
            plan["manifest"]["vocab_size"]
            if name == "logits.fp32"
            else plan["manifest"]["capture"]["hidden_width"],
        )
    expected = {
        "schema_version": SCHEMA,
        "status": "completed",
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": e.object_hash(entry),
        "entries": 1,
        "model_call_returned": True,
        "files": {name: e.sha256(raw) for name, raw in blobs.items()},
        "chosen_token_id": None
        if reference
        else _argmax(blobs["logits.fp32"], plan["manifest"]["vocab_size"]),
        "norm_pre_count": int(reference),
        "norm_post_count": int(reference),
    }
    require(e.canonical(result) == e.canonical(expected), "call_replay")
    return result, blobs


def load_qualification(directory, plan):
    """Replay complete retained evidence without model/tokenizer calls or repair.

    Incomplete roots are preserved but cannot be promoted to qualified results.
    A durable outer result cannot prove its caller observed the runner's return.
    """
    plan = validate_plan(plan)
    root = e._directory(directory)
    header, terminal = _read(root / "run.json"), _read(root / "terminal.json")
    e._keys(
        header,
        "schema_version plan_sha256 prepared_sha256 runtime_binding_sha256 pid sources",
        "run_fields",
    )
    require(
        header["schema_version"] == SCHEMA
        and header["plan_sha256"] == e.object_hash(plan)
        and header["runtime_binding_sha256"] == plan["bindings"]["runtime_binding_sha256"]
        and header["sources"] == source_bindings()
        and type(header["pid"]) is int
        and header["pid"] > 0,
        "run_binding",
    )
    prepared = _prepared(plan, _read(root / "prepared.json"))
    require(header["prepared_sha256"] == e.object_hash(prepared), "prepared_binding")
    expected_root = {
        "run.json",
        "prepared.json",
        "records.json",
        "terminal.json",
        "generation_0",
        "generation_1",
    } | {s["slot_id"] for s in plan["slots"]}
    require({p.name for p in root.iterdir()} == expected_root, "run_inventory")
    counts, records = {"generation": 0, "capture": 0, "reference": 0}, _empty_records(plan)
    for i, observation in enumerate(prepared["observations"]):
        gen_root = e._directory(root / f"generation_{i}")
        gen_attempt = {
            "schema_version": SCHEMA,
            "run_sha256": e.object_hash(header),
            "fixture_index": i,
            "observation_sha256": e.object_hash(observation),
            "policy": plan["generation_policy"],
        }
        require(
            e.canonical(_read(gen_root / "attempt.json")) == e.canonical(gen_attempt),
            "generation_attempt",
        )
        generation = _read(gen_root / "result.json")
        tokens = generation["token_ids"]
        e._ids(tokens, plan["manifest"]["vocab_size"], nonempty=True)
        require(
            len(tokens) <= CAP
            and not any(t in plan["manifest"]["eos_token_ids"] for t in tokens[:-1]),
            "generation_termination",
        )
        eos = tokens[-1] in plan["manifest"]["eos_token_ids"]
        require(eos or len(tokens) == CAP, "generation_cap")
        entries, results = [], []
        require(
            {p.name for p in gen_root.iterdir()}
            == {"attempt.json", "dispatch.json", "result.json"}
            | {f"step_{j:02d}" for j in range(len(tokens))},
            "generation_inventory",
        )
        for j, token in enumerate(tokens):
            attempt = _call_attempt(
                plan,
                e.object_hash(header),
                i,
                observation["prompt_token_ids"] + tokens[:j],
                "generation",
                j,
            )
            result, _ = _replay_call(e._directory(gen_root / f"step_{j:02d}"), plan, attempt)
            require(type(token) is int and result["chosen_token_id"] == token, "generation_argmax")
            entries.append(result["entry_sha256"])
            results.append(e.object_hash(result))
            counts["generation"] += 1
        dispatch = {
            "schema_version": SCHEMA,
            "attempt_sha256": e.object_hash(gen_attempt),
            "entry_sha256": entries,
            "step_result_sha256": results,
        }
        expected_generation = {
            "status": "completed",
            "stop_reason": "eos" if eos else "cap",
            "token_ids": tokens,
            "observed_tokens": len(tokens),
            "attempt_sha256": e.object_hash(gen_attempt),
            "dispatch_sha256": e.object_hash(dispatch),
        }
        require(
            e.canonical(_read(gen_root / "dispatch.json")) == e.canonical(dispatch)
            and e.canonical(generation) == e.canonical(expected_generation),
            "generation_binding",
        )
        for landmark in LANDMARKS:
            row = records[i * 2 + LANDMARKS.index(landmark)]
            slot = e._directory(root / row["slot_id"])
            adapter = capture.load_capture(slot / "adapter", plan["manifest"])
            meta = adapter.bundle.metadata
            require(
                e.canonical(meta["outcome"]) == e.canonical(_outcome(plan, generation)),
                "unmeasured_outcome",
            )
            require(
                e.canonical(meta["generation"]) == e.canonical(generation)
                and meta["observation"]["observation_id"] == observation["observation_id"]
                and meta["observation"]["messages"] == observation["messages"]
                and meta["observation"]["prompt_token_ids"] == observation["prompt_token_ids"]
                and adapter.bundle.blobs["prompt.utf8"]
                == observation["rendered_prompt_text"].encode()
                and type(meta["prefix"]["landmark"]) is int
                and meta["prefix"]["landmark"] == landmark,
                "adapter_cohort",
            )
            caller_return = {
                "schema_version": SCHEMA,
                "event": "capture_prefix_returned_normally",
                "run_sha256": e.object_hash(header),
                "slot": {k: row[k] for k in ("fixture_index", "landmark", "slot_id")},
                "adapter_result_sha256": e.sha256(adapter.receipt_bytes),
                "bundle_sha256": adapter.bundle.sha256,
            }
            require(
                e.canonical(_read(slot / "caller-return.json")) == e.canonical(caller_return),
                "normal_return_binding",
            )
            completed = adapter.receipt["status"] == "completed"
            names = {"adapter", "caller-return.json", "result.json"} | (
                {"adapter-logits.fp32", "reference"} if completed else set()
            )
            require({p.name for p in slot.iterdir()} == names, "slot_inventory")
            equal_capture = equal_logits = None
            adapter_logits_sha = reference_result_sha = None
            if completed:
                counts["capture"] += 1
                values = observation["prompt_token_ids"] + meta["prefix"]["token_ids"]
                call = _call_attempt(plan, e.object_hash(header), i, values, "reference", landmark)
                reference_result, blobs = _replay_call(
                    e._directory(slot / "reference"), plan, call, reference=True
                )
                reference_result_sha = e.object_hash(reference_result)
                counts["reference"] += 1
                adapter_logits = e._read_file(slot / "adapter-logits.fp32")
                _values(adapter_logits, plan["manifest"]["vocab_size"])
                adapter_logits_sha = e.sha256(adapter_logits)
                equal_capture = adapter.bundle.blobs["capture.fp32"] == blobs["norm_input.fp32"]
                equal_logits = adapter_logits == blobs["logits.fp32"]
            expected = {
                "schema_version": SCHEMA,
                "status": "completed" if completed else "unavailable",
                "caller_return_sha256": e.object_hash(caller_return),
                "adapter_result_sha256": e.sha256(adapter.receipt_bytes),
                "capture_equal": equal_capture,
                "logits_equal": equal_logits,
                "adapter_logits_sha256": adapter_logits_sha,
                "reference_result_sha256": reference_result_sha,
            }
            require(
                e.canonical(_read(slot / "result.json")) == e.canonical(expected), "slot_result"
            )
            row.update(
                status=expected["status"],
                capture_equal=equal_capture,
                logits_equal=equal_logits,
                result_sha256=e.object_hash(expected),
            )
    require(e.canonical(_read(root / "records.json")) == e.canonical(records), "records_replay")
    summary = _summary(records, counts, "finished")
    require(summary["total_entries"] <= MAX_ENTRIES, "replay_entry_ceiling")
    expected_terminal = {
        "schema_version": SCHEMA,
        "status": "finished",
        "run_sha256": e.object_hash(header),
        "records_sha256": e.object_hash(records),
        "summary": summary,
        "failure_type": None,
    }
    require(e.canonical(terminal) == e.canonical(expected_terminal), "terminal_replay")
    return {"summary": summary, "records": records, "terminal_sha256": e.object_hash(terminal)}
