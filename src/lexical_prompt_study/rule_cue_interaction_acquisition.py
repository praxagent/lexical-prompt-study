"""A197 fixed conditional path scoring; injected model/tokenizer, no loader.

Full native CPU FP32 vocabulary rows are retained privately. A successful return
receipt is necessary to use a measurement; an orphan readout never is. Native
rendering and execution are qualified separately from model-free receipt replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import sys

from . import landmark_evidence as e
from . import rule_cue_interaction_tasks as tasks
from . import special_token_inventory as inventory

SCHEMA = "a197-rule-cue-interaction-acquisition-v1"
MAX_ENTRIES = 128
MAX_PROMPT = 256
MAX_TARGET = 16
CHAT_TEMPLATE_DATE = "24 Sep 2026"
EOT = "<|eot_id|>"
EVIDENCE_SHA256 = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"


def require(condition, code):
    if not condition:
        raise ValueError("a197_" + code)


def same(left, right, code="typed_equality"):
    require(e.canonical(left) == e.canonical(right), code)


def source_bindings():
    require(
        e.sha256(Path(inventory.__file__).read_bytes()) == INVENTORY_SHA256,
        "inventory_source",
    )
    require(e.sha256(Path(e.__file__).read_bytes()) == EVIDENCE_SHA256, "evidence_source")
    return {
        "acquisition": e.sha256(Path(__file__).read_bytes()),
        "tasks": e.sha256(Path(tasks.__file__).read_bytes()),
        "evidence": EVIDENCE_SHA256,
        "special_token_inventory": INVENTORY_SHA256,
    }


def compile_plan(
    *, provenance, geometry, eos_token_ids, runtime_binding_sha256, protocol_sha256, tests_sha256
):
    e._keys(provenance, "model_sha256 tokenizer_sha256 chat_template_sha256", "provenance")
    for value in provenance.values():
        e._hash(value)
    e._keys(geometry, "vocab_size hidden_width model_layers context_limit", "geometry")
    for value in geometry.values():
        e._int(value, 1)
    e._ids(eos_token_ids, geometry["vocab_size"], nonempty=True)
    require(eos_token_ids == sorted(set(eos_token_ids)), "eos_order")
    for value in (runtime_binding_sha256, protocol_sha256, tests_sha256):
        e._hash(value)
    return e._snapshot(
        {
            "schema_version": SCHEMA,
            "provenance": provenance,
            "geometry": geometry,
            "eos_token_ids": eos_token_ids,
            "runtime_binding_sha256": runtime_binding_sha256,
            "protocol_sha256": protocol_sha256,
            "tests_sha256": tests_sha256,
            "sources": source_bindings(),
            "rows": tasks.build_roster(),
            "schedule": tasks.build_schedule(),
            "decoder": e.DECODER,
            "chat_template_date": CHAT_TEMPLATE_DATE,
            "eot_token": EOT,
            "maximum_prompt_tokens": MAX_PROMPT,
            "maximum_candidate_tokens_including_eot": MAX_TARGET,
            "planned_measurements": MAX_ENTRIES,
            "maximum_entries": MAX_ENTRIES,
            "execution": {
                "device": "cpu",
                "dtype": "float32",
                "threads": 8,
                "interop_threads": 1,
                "attention": "sdpa_math",
                "cache": False,
                "autocast": False,
                "score": "sum_including_eot",
                "repeat": "exact_native_bytes_first_copy",
            },
        }
    )


def validate_plan(plan):
    require(type(plan) is dict, "plan_type")
    expected = compile_plan(
        **{
            k: plan[k]
            for k in (
                "provenance",
                "geometry",
                "eos_token_ids",
                "runtime_binding_sha256",
                "protocol_sha256",
                "tests_sha256",
            )
        }
    )
    same(plan, expected, "plan_binding")
    return e._snapshot(plan)


# Preparation diagnostics expose only these finite stages/codes and a bounded
# exception class name. They never retain exception messages or native payloads.
INVENTORY_SHA256 = "75e9425bbde7fefa56798613c6ad17a643f077463af1fce8b4c2c40e716391f0"
PREPARATION_STAGES = frozenset(
    {
        "unclassified",
        "plan_validation",
        "template_before",
        "special_inventory_before",
        "terminal_identity",
        "payload_encoding",
        "prompt_construction",
        "candidate_construction",
        "special_inventory_after",
        "template_after",
        "prepared_validation",
    }
)
PREPARATION_GUARD_CODES = frozenset(
    {
        "operation_exception",
        "preparation_metadata_invalid",
        "prepared_contract",
        "chat_template",
        "native_eot",
        "literal_eot_encoding",
        "literal_eot_roundtrip",
        "encoded_ids",
        "payload_special_token",
        "rendered_prompt",
        "native_prompt_ids",
        "native_assistant_completion",
        "native_closed_ids",
        "joint_prompt_prefix",
        "target_roundtrip",
        "joint_roundtrip",
        "special_inventory_changed",
    }
)
GUARD_CODES = PREPARATION_GUARD_CODES


def _safe_error_type(error):
    name = type(error).__name__
    return (
        name
        if name.isascii() and name.isidentifier() and len(name) <= 128
        else "UnclassifiedException"
    )


class PreparationError(ValueError):
    """Safe native-preparation provenance; the original exception remains cause."""

    def __init__(self, stage, code, original_error_type):
        require(type(stage) is str and stage in PREPARATION_STAGES, "preparation_stage")
        require(type(code) is str and code in PREPARATION_GUARD_CODES, "preparation_code")
        require(
            type(original_error_type) is str
            and original_error_type.isascii()
            and original_error_type.isidentifier()
            and len(original_error_type) <= 128,
            "preparation_error_type",
        )
        self.stage, self.code, self.original_error_type = stage, code, original_error_type
        super().__init__("a197_preparation_" + stage + "_" + code)


class _PreparationGuard(ValueError):
    def __init__(self, code):
        require(code in PREPARATION_GUARD_CODES, "preparation_guard_code")
        self.code = code
        super().__init__(code)


def _prep_require(condition, code):
    if not condition:
        raise _PreparationGuard(code)


def _prep_same(left, right, code):
    _prep_require(e.canonical(left) == e.canonical(right), code)


def preparation_failure(error):
    """Return privacy-safe provenance, including unwrapped process interruptions."""
    if type(error) is PreparationError:
        # Validate again instead of trusting mutable exception attributes.
        checked = PreparationError(error.stage, error.code, error.original_error_type)
        return {
            "stage": checked.stage,
            "code": checked.code,
            "original_error_type": checked.original_error_type,
        }
    return {
        "stage": "unclassified",
        "code": "operation_exception",
        "original_error_type": _safe_error_type(error),
    }


def _encode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    _prep_require(type(ids) is list, "encoded_ids")
    return ids


def prepare_inputs(plan, tokenizer):
    """Audit actual tokenizer APIs only; never load a tokenizer or call a model.

    Ordinary errors carry finite safe provenance. KeyboardInterrupt/SystemExit
    keep their original identity and are classified by ``preparation_failure``.
    """
    state = {"stage": "plan_validation"}
    try:
        return _prepare_inputs(plan, tokenizer, state)
    except Exception as error:
        code = (
            error.code
            if type(error) is _PreparationGuard
            else "prepared_contract"
            if state["stage"] == "prepared_validation"
            else "operation_exception"
        )
        original_type = (
            "ValueError" if type(error) is _PreparationGuard else _safe_error_type(error)
        )
        raise PreparationError(state["stage"], code, original_type) from error


def _prepare_inputs(plan, tokenizer, state):
    plan = validate_plan(plan)
    state["stage"] = "template_before"
    _prep_require(
        type(tokenizer.chat_template) is str
        and e.sha256(tokenizer.chat_template.encode())
        == plan["provenance"]["chat_template_sha256"],
        "chat_template",
    )
    state["stage"] = "special_inventory_before"
    snapshot = inventory.collect_special_token_inventory(
        tokenizer, vocab_size=plan["geometry"]["vocab_size"]
    )
    special = snapshot["special_ids"]
    state["stage"] = "terminal_identity"
    eot = tokenizer.convert_tokens_to_ids(EOT)
    _prep_require(type(eot) is int and eot in plan["eos_token_ids"], "native_eot")
    terminal = inventory.validate_terminal_token(
        snapshot, terminal_token_id=eot, stop_token_ids=plan["eos_token_ids"]
    )
    _prep_same(_encode(tokenizer, EOT), [eot], "literal_eot_encoding")
    _prep_require(tokenizer.decode([eot], **e.DECODER) == EOT, "literal_eot_roundtrip")
    observations = []
    for row in plan["rows"]:
        state["stage"] = "payload_encoding"
        payloads = {message["role"]: message["content"] for message in row["messages"]}
        payloads.update(row["candidates"])
        payload_ids = {key: _encode(tokenizer, text) for key, text in payloads.items()}
        for ids in payload_ids.values():
            e._ids(ids, plan["geometry"]["vocab_size"], nonempty=True)
            _prep_require(not set(ids) & set(special), "payload_special_token")
        state["stage"] = "prompt_construction"
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            date_string=CHAT_TEMPLATE_DATE,
        )
        _prep_require(type(prompt) is str and prompt, "rendered_prompt")
        ids = _encode(tokenizer, prompt)
        _prep_same(
            tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                date_string=CHAT_TEMPLATE_DATE,
            ),
            ids,
            "native_prompt_ids",
        )
        candidates = {}
        for key, text in row["candidates"].items():
            state["stage"] = "candidate_construction"
            completed = tokenizer.apply_chat_template(
                row["messages"] + [{"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                date_string=CHAT_TEMPLATE_DATE,
            )
            _prep_require(completed == prompt + text + EOT, "native_assistant_completion")
            full = _encode(tokenizer, completed)
            _prep_same(
                tokenizer.apply_chat_template(
                    row["messages"] + [{"role": "assistant", "content": text}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=False,
                    date_string=CHAT_TEMPLATE_DATE,
                ),
                full,
                "native_closed_ids",
            )
            _prep_require(full[: len(ids)] == ids, "joint_prompt_prefix")
            target = full[len(ids) :]
            _prep_require(tokenizer.decode(target, **e.DECODER) == text + EOT, "target_roundtrip")
            _prep_require(tokenizer.decode(full, **e.DECODER) == completed, "joint_roundtrip")
            candidates[key] = {
                "text": text,
                "completed_utf8": completed,
                "completed_sha256": e.sha256(completed.encode()),
                "target_token_ids": target,
                "target_ids_sha256": e.object_hash(target),
            }
        observations.append(
            {
                "row_sha256": e.object_hash(row),
                "prompt_utf8": prompt,
                "prompt_sha256": e.sha256(prompt.encode()),
                "prompt_token_ids": ids,
                "prompt_ids_sha256": e.object_hash(ids),
                "candidates": candidates,
                "payload_token_ids": payload_ids,
                "payload_ids_sha256": e.object_hash(payload_ids),
            }
        )
    state["stage"] = "special_inventory_after"
    _prep_same(
        inventory.collect_special_token_inventory(
            tokenizer, vocab_size=plan["geometry"]["vocab_size"]
        ),
        snapshot,
        "special_inventory_changed",
    )
    state["stage"] = "template_after"
    _prep_require(
        type(tokenizer.chat_template) is str
        and e.sha256(tokenizer.chat_template.encode())
        == plan["provenance"]["chat_template_sha256"],
        "chat_template",
    )
    state["stage"] = "prepared_validation"
    return validate_prepared(
        plan,
        {
            "schema_version": SCHEMA,
            "plan_sha256": e.object_hash(plan),
            "eot_token_id": eot,
            "special_token_ids": special,
            "special_inventory": snapshot,
            "terminal_metadata": terminal,
            "observations": observations,
        },
    )


def validate_prepared(plan, prepared):
    e._keys(
        prepared,
        "schema_version plan_sha256 eot_token_id special_token_ids special_inventory terminal_metadata observations",
        "prepared_fields",
    )
    require(
        prepared["schema_version"] == SCHEMA and prepared["plan_sha256"] == e.object_hash(plan),
        "prepared_plan",
    )
    eot = prepared["eot_token_id"]
    require(type(eot) is int and eot in plan["eos_token_ids"], "prepared_eot")
    snapshot = inventory.validate_inventory(prepared["special_inventory"])
    require(snapshot["vocab_size"] == plan["geometry"]["vocab_size"], "inventory_vocab")
    terminal = inventory.validate_terminal_token(
        snapshot, terminal_token_id=eot, stop_token_ids=plan["eos_token_ids"]
    )
    same(prepared["terminal_metadata"], terminal, "terminal_metadata")
    special = prepared["special_token_ids"]
    same(special, snapshot["special_ids"], "special_inventory_ids")
    e._ids(special, plan["geometry"]["vocab_size"], nonempty=True)
    require(
        special == sorted(set(special)) and set(plan["eos_token_ids"]) <= set(special),
        "prepared_specials",
    )
    require(
        type(prepared["observations"]) is list and len(prepared["observations"]) == 32,
        "prepared_rows",
    )
    shared_targets = {}
    for row, obs in zip(plan["rows"], prepared["observations"], strict=True):
        e._keys(
            obs,
            "row_sha256 prompt_utf8 prompt_sha256 prompt_token_ids prompt_ids_sha256 candidates payload_token_ids payload_ids_sha256",
            "observation_fields",
        )
        ids, prompt = obs["prompt_token_ids"], obs["prompt_utf8"]
        e._ids(ids, plan["geometry"]["vocab_size"], nonempty=True)
        require(len(ids) <= MAX_PROMPT and type(prompt) is str and bool(prompt), "prompt_bounds")
        require(
            obs["row_sha256"] == e.object_hash(row)
            and obs["prompt_ids_sha256"] == e.object_hash(ids)
            and obs["prompt_sha256"] == e.sha256(prompt.encode()),
            "prompt_binding",
        )
        e._keys(obs["candidates"], "arithmetic_truth arithmetic_foil", "candidate_keys")
        e._keys(
            obs["payload_token_ids"], "system user arithmetic_truth arithmetic_foil", "payload_keys"
        )
        for payload in obs["payload_token_ids"].values():
            e._ids(payload, plan["geometry"]["vocab_size"], nonempty=True)
            require(not set(payload) & set(special), "prepared_payload_specials")
        require(
            obs["payload_ids_sha256"] == e.object_hash(obs["payload_token_ids"]), "payload_binding"
        )
        for key, path in obs["candidates"].items():
            e._keys(
                path,
                "text completed_utf8 completed_sha256 target_token_ids target_ids_sha256",
                "path_keys",
            )
            target = path["target_token_ids"]
            e._ids(target, plan["geometry"]["vocab_size"], nonempty=True)
            require(
                len(target) <= MAX_TARGET
                and target[-1] == eot
                and not any(x in special for x in target[:-1]),
                "target_eot_bounds",
            )
            require(
                len(ids) + len(target) - 1 <= plan["geometry"]["context_limit"], "context_bounds"
            )
            require(
                path["text"] == row["candidates"][key]
                and path["completed_utf8"] == prompt + path["text"] + EOT
                and path["completed_sha256"] == e.sha256(path["completed_utf8"].encode())
                and path["target_ids_sha256"] == e.object_hash(target),
                "candidate_binding",
            )
            pair_key = (row["core_index"], key)
            if pair_key in shared_targets:
                same(target, shared_targets[pair_key], "paired_target_ids")
            else:
                shared_targets[pair_key] = target
        require(
            obs["candidates"]["arithmetic_truth"]["target_token_ids"]
            != obs["candidates"]["arithmetic_foil"]["target_token_ids"],
            "distinct_paths",
        )
    return e._snapshot(prepared)


def _write(path, value):
    e._publish(Path(path), e.canonical(value))


def _read(path):
    raw = e._read_file(Path(path))
    value = json.loads(raw)
    require(e.canonical(value) == raw, "canonical_json")
    return value


def _inventory(root):
    root = e._directory(Path(root))
    found = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "inventory_symlink")
        if path.is_dir():
            continue
        found[str(path.relative_to(root))] = e.sha256(e._read_file(path))
    return found


def score_readout(raw, target_ids, vocab_size):
    """Literal FP32 rows, double-precision shifted sums, no target normalization."""
    e._int(vocab_size, 1)
    e._ids(target_ids, vocab_size, nonempty=True)
    require(type(raw) is bytes and len(raw) == 4 * vocab_size * len(target_ids), "readout_shape")
    chosen, maxima, log_sums, logprobs = [], [], [], []
    for index, token in enumerate(target_ids):
        start = index * vocab_size * 4
        values = [x[0] for x in struct.iter_unpack("<f", raw[start : start + vocab_size * 4])]
        require(all(math.isfinite(x) for x in values), "finite_readout")
        maximum = max(values)
        log_sum = math.log(math.fsum(math.exp(value - maximum) for value in values))
        logprob = (values[token] - maximum) - log_sum
        chosen.append(values[token])
        maxima.append(maximum)
        log_sums.append(log_sum)
        logprobs.append(logprob)
    score = math.fsum(logprobs)
    require(math.isfinite(score), "finite_score")
    return {
        "score": score,
        "chosen_logits": chosen,
        "row_maxima": maxima,
        "log_shifted_sums": log_sums,
        "token_logprobs": logprobs,
        "scored_tokens_including_eot": len(target_ids),
    }


def _hook_audit(torch, model, owned=None):
    state = torch.nn.modules.module
    require(not state._global_forward_hooks and not state._global_forward_pre_hooks, "global_hooks")
    for module in model.modules():
        require(not module._forward_hooks, "forward_hooks")
        expected = {owned.id} if owned is not None and module is model else set()
        require(
            set(module._forward_pre_hooks) == expected
            and set(module._forward_pre_hooks_with_kwargs) == expected,
            "pre_hooks",
        )


def _preflight(model, plan):
    import torch

    require(sys.byteorder == "little", "native_little_endian")
    require(torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 1, "cpu_threads")
    require(
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "determinism",
    )
    require(
        not torch.is_autocast_enabled("cpu")
        and not torch.is_autocast_enabled("cuda")
        and not torch.cuda.is_initialized(),
        "cpu_no_autocast",
    )
    require(
        torch.backends.cuda.math_sdp_enabled()
        and not torch.backends.cuda.flash_sdp_enabled()
        and not torch.backends.cuda.mem_efficient_sdp_enabled()
        and not torch.backends.cuda.cudnn_sdp_enabled(),
        "math_sdpa",
    )
    require(getattr(model.config, "_attn_implementation", None) == "sdpa", "model_attention")
    for attr, key in (
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_width"),
        ("num_hidden_layers", "model_layers"),
        ("max_position_embeddings", "context_limit"),
    ):
        same(getattr(model.config, attr), plan["geometry"][key], "model_geometry")
    for module in model.modules():
        require(
            not module.training
            and not hasattr(module, "_orig_mod")
            and not hasattr(module, "_hf_hook"),
            "eval_direct_modules",
        )
    require(
        getattr(model, "hf_device_map", None) in (None, {}, {"": "cpu"})
        and not getattr(model, "is_loaded_in_4bit", False)
        and not getattr(model, "is_loaded_in_8bit", False)
        and getattr(model, "quantization_method", None) is None,
        "direct_unquantized",
    )
    parameters = list(model.parameters())
    require(bool(parameters), "model_parameters")
    for value in (*parameters, *model.buffers()):
        require(
            value.layout == torch.strided
            and not value.is_complex()
            and not value.is_quantized
            and value.device.type == "cpu"
            and (not value.is_floating_point() or value.dtype == torch.float32),
            "cpu_fp32_tensors",
        )
    require(all(x.dtype == torch.float32 for x in parameters), "fp32_parameters")
    _hook_audit(torch, model)


def _attempt(plan, prepared, run_sha, slot):
    obs = prepared["observations"][slot["sequence_index"]]
    target = obs["candidates"][slot["candidate"]]["target_token_ids"]
    ids = obs["prompt_token_ids"] + target[:-1]
    return {
        "schema_version": SCHEMA,
        "run_sha256": run_sha,
        "plan_sha256": e.object_hash(plan),
        "evaluation_sha256": e.object_hash(slot),
        "evaluation_index": slot["evaluation_index"],
        "input_token_ids": ids,
        "input_ids_sha256": e.object_hash(ids),
        "prompt_tokens": len(obs["prompt_token_ids"]),
        "target_token_ids": target,
        "target_ids_sha256": e.object_hash(target),
        "logits_to_keep": len(target),
        "first_scored_absolute_position": len(obs["prompt_token_ids"]) - 1,
    }


def _forward(model, plan, root, attempt, counts):
    import torch

    root = e._directory(Path(root), create=True)
    _write(root / "attempt.json", attempt)
    ids, length = attempt["input_token_ids"], attempt["logits_to_keep"]
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device="cpu"),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cpu"),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device="cpu").unsqueeze(0),
        "past_key_values": None,
        "use_cache": False,
        "logits_to_keep": length,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    state = {"entries": 0, "model_call_returned": False, "cleanup_failure_type": None}
    handle, original = None, None

    def check(kwargs):
        require(set(kwargs) == set(inputs), "input_keys")
        for key, values in (
            ("input_ids", ids),
            ("attention_mask", [1] * len(ids)),
            ("position_ids", list(range(len(ids)))),
        ):
            value = kwargs[key]
            require(
                type(value) is torch.Tensor
                and value.dtype == torch.long
                and value.device.type == "cpu"
                and value.layout == torch.strided
                and tuple(value.shape) == (1, len(ids))
                and value.tolist() == [values],
                "actual_inputs",
            )
        require(
            kwargs["past_key_values"] is None
            and kwargs["use_cache"] is False
            and kwargs["output_hidden_states"] is False
            and kwargs["output_attentions"] is False,
            "forward_flags",
        )
        same(kwargs["logits_to_keep"], length, "target_rows")

    def entry(module, args, kwargs):
        state["entries"] += 1
        counts["scoring"] += 1
        require(
            state["entries"] == 1
            and counts["scoring"] <= MAX_ENTRIES
            and module is model
            and not args,
            "dispatch_ceiling",
        )
        check(kwargs)
        _hook_audit(torch, model, handle)
        require(
            torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cpu"),
            "inference_context",
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
            require(
                type(counts["scoring"]) is int and 0 <= counts["scoring"] < MAX_ENTRIES,
                "pre_dispatch_ceiling",
            )
            _preflight(model, plan)
            handle = model.register_forward_pre_hook(entry, with_kwargs=True)
            with torch.inference_mode():
                output = model(**inputs)
            state["model_call_returned"] = True
            check(inputs)
            require(state["entries"] == 1, "one_entry")
            logits = output.logits
            require(
                type(logits) is torch.Tensor
                and logits.dtype == torch.float32
                and logits.device.type == "cpu"
                and logits.layout == torch.strided
                and tuple(logits.shape) == (1, length, plan["geometry"]["vocab_size"]),
                "native_fp32_rows",
            )
            require(bool(torch.isfinite(logits).all()), "finite_logits")
            raw = bytes(logits.detach().clone().contiguous().view(torch.uint8).reshape(-1).tolist())
            metrics = score_readout(
                raw, attempt["target_token_ids"], plan["geometry"]["vocab_size"]
            )
        except BaseException as exc:
            original = exc
            raise
        finally:
            cleanup = None
            if handle is not None:
                try:
                    handle.remove()
                except BaseException as exc:
                    cleanup = exc
                finally:
                    model._forward_pre_hooks.pop(handle.id, None)
                    model._forward_pre_hooks_with_kwargs.pop(handle.id, None)
            try:
                _hook_audit(torch, model)
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
            **metrics,
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


def _replay_call(root, plan, attempt):
    require(
        {p.name for p in root.iterdir()}
        == {"attempt.json", "entry.json", "result.json", "logits.fp32", "return.json"},
        "complete_call_inventory",
    )
    same(_read(root / "attempt.json"), attempt, "call_attempt")
    entry = {
        "schema_version": SCHEMA,
        "event": "model_forward_pre_hook_entry",
        "attempt_sha256": e.object_hash(attempt),
        "entry_index": 0,
    }
    same(_read(root / "entry.json"), entry, "call_entry")
    raw = e._read_file(root / "logits.fp32")
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": e.object_hash(entry),
        "entries": 1,
        "model_call_returned": True,
        "logits_sha256": e.sha256(raw),
        **score_readout(raw, attempt["target_token_ids"], plan["geometry"]["vocab_size"]),
    }
    same(_read(root / "result.json"), result, "score_replay")
    same(
        _read(root / "return.json"),
        {"event": "measurement_returned_normally", "result_sha256": e.object_hash(result)},
        "normal_return",
    )
    return result


def _partial_count(root, attempt):
    if not root.exists():
        return 0, True
    e._directory(root)
    names = {p.name for p in root.iterdir()}
    allowed = {
        "attempt.json",
        "entry.json",
        "result.json",
        "logits.fp32",
        "failure.json",
        "return.json",
    }
    require(names <= allowed | {"." + x + ".tmp" for x in allowed}, "partial_inventory")
    if "attempt.json" not in names:
        require(names <= {".attempt.json.tmp"}, "unpublished_attempt")
        return 0, True
    same(_read(root / "attempt.json"), attempt, "partial_attempt")
    entry = int("entry.json" in names)
    if entry:
        same(
            _read(root / "entry.json"),
            {
                "schema_version": SCHEMA,
                "event": "model_forward_pre_hook_entry",
                "attempt_sha256": e.object_hash(attempt),
                "entry_index": 0,
            },
            "partial_entry",
        )
    if "failure.json" in names:
        value = _read(root / "failure.json")
        e._keys(
            value,
            "schema_version attempt_sha256 entries model_call_returned cleanup_failure_type failure_type usable",
            "failure_keys",
        )
        require(
            value["schema_version"] == SCHEMA
            and value["attempt_sha256"] == e.object_hash(attempt)
            and value["usable"] is False
            and type(value["entries"]) is int
            and entry <= value["entries"] <= 1 + entry
            and type(value["model_call_returned"]) is bool
            and (not value["model_call_returned"] or entry == 1)
            and type(value["failure_type"]) is str
            and (
                value["cleanup_failure_type"] is None or type(value["cleanup_failure_type"]) is str
            ),
            "partial_failure",
        )
        return value["entries"], True
    # Durable result alone is not a caller normal return and is never a score.
    # A second callback can fail before its receipt; the first proves only a lower bound.
    return entry, False


def _slot_record(slot, status="unattempted", result=None):
    return {
        "evaluation_id": slot["evaluation_id"],
        "evaluation_index": slot["evaluation_index"],
        "status": status,
        "measurement": None
        if result is None
        else {"score": result["score"], "logits_sha256": result["logits_sha256"]},
    }


def _summary(records, count, exact):
    return {
        "planned_measurements": MAX_ENTRIES,
        **{
            key + "_measurements": sum(r["status"] == key for r in records)
            for key in ("completed", "failed", "unattempted")
        },
        "scoring_entries": count if exact else None,
        "total_entries": count if exact else None,
        "entries_complete": exact,
    }


def run_acquisition(model, tokenizer, *, plan, prepared, runtime_binding, directory, event=None):
    """One fixed schedule; hard failures stop it, repeat/functional results never do."""
    plan = validate_plan(plan)
    prepared = validate_prepared(plan, prepared)
    require(e.object_hash(runtime_binding) == plan["runtime_binding_sha256"], "runtime_binding")
    root = e._directory(Path(directory), create=True)
    header = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "runtime_binding_sha256": e.object_hash(runtime_binding),
        "sources": source_bindings(),
    }
    _write(root / "run.json", header)
    _write(root / "prepared.json", prepared)
    e._directory(root / "calls", create=True)
    records = [_slot_record(slot) for slot in plan["schedule"]]
    counts, active = {"scoring": 0}, None
    try:
        if event is not None:
            event("science")
        for index, slot in enumerate(plan["schedule"]):
            active = index
            result = _forward(
                model,
                plan,
                root / "calls" / f"{index:03d}",
                _attempt(plan, prepared, e.object_hash(header), slot),
                counts,
            )
            _write(
                root / "calls" / f"{index:03d}" / "return.json",
                {"event": "measurement_returned_normally", "result_sha256": e.object_hash(result)},
            )
            records[index] = _slot_record(slot, "completed", result)
            active = None
        derived = tasks.analyze_measurements(
            plan["rows"], plan["schedule"], [r["measurement"] for r in records]
        )
        terminal = {
            "schema_version": SCHEMA,
            "status": "finished",
            "run_sha256": e.object_hash(header),
            "records": records,
            "summary": _summary(records, counts["scoring"], True),
            "context_records": derived["records"],
            "analysis": derived["analysis"],
            "artifact_sha256": _inventory(root),
        }
        _write(root / "terminal.json", terminal)
        return load_acquisition(root, plan)
    except BaseException as exc:
        try:
            if active is not None:
                slot = plan["schedule"][active]
                call = root / "calls" / f"{active:03d}"
                if (call / "return.json").exists():
                    result = _replay_call(
                        call, plan, _attempt(plan, prepared, e.object_hash(header), slot)
                    )
                    records[active] = _slot_record(slot, "completed", result)
                else:
                    records[active] = _slot_record(slot, "failed")
            derived = tasks.analyze_measurements(
                plan["rows"], plan["schedule"], [r["measurement"] for r in records]
            )
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "run_sha256": e.object_hash(header),
                    "active_index": active,
                    "failure_type": type(exc).__name__,
                    "records": records,
                    "context_records": derived["records"],
                    "analysis": derived["analysis"],
                    "summary": _summary(records, counts["scoring"], True),
                    "artifact_sha256": _inventory(root),
                },
            )
        except BaseException:
            pass
        raise


def load_acquisition(directory, plan):
    """Authenticate retained bytes without importing Torch or tokenizing anything."""
    plan = validate_plan(plan)
    root = e._directory(Path(directory))
    payloads = {"run.json", "prepared.json", "terminal.json", "failure.json"}
    require(
        {p.name for p in root.iterdir()}
        <= payloads | {"calls"} | {"." + x + ".tmp" for x in payloads},
        "root_inventory",
    )
    failed = (root / "failure.json").exists()
    name = "failure.json" if failed else "terminal.json"
    value = _read(root / name)
    require(
        value["schema_version"] == SCHEMA
        and value["status"] == ("failed" if failed else "finished"),
        "terminal_status",
    )
    inventory = _inventory(root)
    inventory.pop(name)
    same(inventory, value["artifact_sha256"], "terminal_inventory")
    prepared = validate_prepared(plan, _read(root / "prepared.json"))
    header = {
        "schema_version": SCHEMA,
        "plan_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "runtime_binding_sha256": plan["runtime_binding_sha256"],
        "sources": source_bindings(),
    }
    same(_read(root / "run.json"), header, "run_header")
    require(value["run_sha256"] == e.object_hash(header), "terminal_header")
    calls = e._directory(root / "calls")
    names = sorted(p.name for p in calls.iterdir())
    require(
        len(names) <= MAX_ENTRIES and names == [f"{i:03d}" for i in range(len(names))],
        "call_prefix",
    )
    active = value.get("active_index")
    require(
        not failed
        or active is None
        or type(active) is int
        and 0 <= active < MAX_ENTRIES
        and active in (len(names) - 1, len(names)),
        "active_slot",
    )
    records, count, exact = [], 0, True
    repeats = {}
    for index, slot in enumerate(plan["schedule"]):
        call = calls / f"{index:03d}"
        attempt = _attempt(plan, prepared, e.object_hash(header), slot)
        if (call / "return.json").exists():
            result = _replay_call(call, plan, attempt)
            count += 1
            raw = e._read_file(call / "logits.fp32")
            key = (slot["sequence_index"], slot["candidate"])
            if key in repeats and result["logits_sha256"] == repeats[key][0]:
                require(raw == repeats[key][1], "exact_repeat_bytes")
            else:
                repeats[key] = (result["logits_sha256"], raw)
            records.append(_slot_record(slot, "completed", result))
        else:
            require(failed and (not call.exists() or index == active), "incomplete_final_only")
            amount, known = _partial_count(call, attempt)
            count += amount
            exact = exact and known
            records.append(_slot_record(slot, "failed" if index == active else "unattempted"))
    require(count <= MAX_ENTRIES + int(failed), "receipt_entry_ceiling")
    same(records, value["records"], "slot_replay")
    derived = tasks.analyze_measurements(
        plan["rows"], plan["schedule"], [r["measurement"] for r in records]
    )
    same(derived["records"], value["context_records"], "context_replay")
    same(derived["analysis"], value["analysis"], "analysis_replay")
    summary = _summary(records, count, exact)
    if failed:
        declared = value["summary"]
        e._keys(
            declared,
            "planned_measurements completed_measurements failed_measurements unattempted_measurements scoring_entries total_entries entries_complete",
            "declared_summary",
        )
        for key in (
            "planned_measurements",
            "completed_measurements",
            "failed_measurements",
            "unattempted_measurements",
        ):
            same(declared[key], summary[key], "declared_slots")
        require(
            declared["entries_complete"] is True
            and type(declared["scoring_entries"]) is int
            and count <= declared["scoring_entries"] <= count + int(not exact),
            "declared_count",
        )
        same(declared["total_entries"], declared["scoring_entries"], "declared_total")
        summary.update(scoring_entries_lower_bound=count, total_entries_lower_bound=count)
    else:
        require(len(names) == MAX_ENTRIES and count == MAX_ENTRIES and exact, "complete_schedule")
        same(value["summary"], summary, "summary_replay")
    return {
        "status": value["status"],
        "records": records,
        "context_records": derived["records"],
        "summary": summary,
        "analysis": derived["analysis"],
        "reported_summary": value["summary"],
        "terminal_sha256": e.sha256(e._read_file(root / name)),
    }
