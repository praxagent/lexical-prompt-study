"""A188 CPU BF16 forward receipts and two fixed technical checks.

Injected, trusted model/tokenizer implementations only; no asset loader. Native
BF16 logits are retained unchanged. FP32 bytes are exact bit widening for replay,
not an FP32 model evaluation. Buffer dtypes are recorded; parameter dtype does
not establish the precision of every internal operation. Opaque source/weight
bindings require the separately frozen caller's authentication.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

from . import landmark_evidence as e
from . import prefix_only_capture as support

SCHEMA = "a188-bf16-forward-v1"
A183_SHA = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
A184_SHA = "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69"
MAX_ENTRIES, MAX_PROMPT = 2050, 256
TECHNICAL_MESSAGES = (
    ("system", "Follow the user's instruction."),
    ("user", "Reply with the digit 0."),
)
DATE = "26 Jul 2024"


def require(ok, code):
    if not ok:
        raise ValueError("a188_bf16_" + code)


def same(left, right, code):
    require(e.canonical(left) == e.canonical(right), code)


def source_bindings():
    require(e.sha256(Path(e.__file__).read_bytes()) == A183_SHA, "evidence_source")
    require(e.sha256(Path(support.__file__).read_bytes()) == A184_SHA, "support_source")
    return {
        "a183": A183_SHA,
        "a184": A184_SHA,
        "bf16_forward": e.sha256(Path(__file__).read_bytes()),
    }


def _write(path, value):
    e._publish(path, e.canonical(value))


def _read(path):
    return e._read_json(e._read_file(path))


def _inventory(root):
    root = e._directory(root)
    found = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "artifact_symlink")
        if path.is_file():
            found[str(path.relative_to(root))] = e.sha256(e._read_file(path))
        else:
            require(path.is_dir(), "artifact_type")
    return found


def _geometry(plan):
    e._keys(plan, "context_limit manifest", "geometry_plan")
    e._keys(plan["manifest"], "vocab_size capture", "geometry_manifest")
    cap = plan["manifest"]["capture"]
    e._keys(cap, "hidden_width model_layers layer_index", "geometry_compatibility")
    for value in (
        plan["context_limit"],
        plan["manifest"]["vocab_size"],
        cap["hidden_width"],
        cap["model_layers"],
    ):
        e._int(value, 1)
    e._int(cap["layer_index"])
    require(cap["layer_index"] == cap["model_layers"] - 1, "technical_last_layer")
    return e._snapshot(plan)


def clean_model(model, geometryplan):
    """Validate actual BF16 parameters; return metadata, never cast/mutate."""
    import torch

    plan = _geometry(geometryplan)
    source_bindings()
    require(isinstance(model, torch.nn.Module), "model_type")
    require(not hasattr(model, "_orig_mod"), "compiled_model")
    require(all(not module.training for module in model.modules()), "evaluation_mode")
    require(
        torch.get_num_threads() == 4 and torch.are_deterministic_algorithms_enabled(),
        "runtime_flags",
    )
    require(not torch.is_autocast_enabled("cpu") and sys.byteorder == "little", "precision")
    require(
        not getattr(model, "is_quantized", False)
        and not getattr(model, "quantization_method", None),
        "quantized_model",
    )
    require(not getattr(model.config, "quantization_config", None), "quantized_config")
    support._hook_audit(torch, model)
    cap, config = plan["manifest"]["capture"], model.config
    actual = {
        key: getattr(config, key)
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings")
    }
    expected = {
        "vocab_size": plan["manifest"]["vocab_size"],
        "hidden_size": cap["hidden_width"],
        "num_hidden_layers": cap["model_layers"],
        "max_position_embeddings": plan["context_limit"],
    }
    same(actual, expected, "config_geometry")
    require(
        config.output_hidden_states is False and config.output_attentions is False,
        "lazy_collectors",
    )
    require(getattr(config, "_attn_implementation", None) == "sdpa", "attention")
    blocks = model.get_submodule("model.layers")
    require(
        isinstance(blocks, torch.nn.ModuleList) and len(blocks) == cap["model_layers"], "blocks"
    )
    require(model.get_submodule("model.norm") is not model, "norm_module")
    tensors = []
    parameters = list(model.named_parameters())
    require(bool(parameters), "empty_parameters")
    for kind, named in (("parameter", parameters), ("buffer", list(model.named_buffers()))):
        for name, tensor in named:
            require(
                tensor.device.type == "cpu"
                and tensor.layout == torch.strided
                and not tensor.is_complex(),
                "model_cpu_strided",
            )
            if kind == "parameter":
                require(tensor.dtype == torch.bfloat16, "parameter_bf16")
            tensors.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
    info = {
        "schema_version": SCHEMA,
        "class": type(model).__module__ + "." + type(model).__qualname__,
        "geometry": actual,
        "attention_implementation": "sdpa",
        "parameter_dtype": "torch.bfloat16",
        "tensors": tensors,
        "tensor_metadata_sha256": e.object_hash(tensors),
        "weight_content_authenticated": False,
        "all_internal_operations_bf16_asserted": False,
    }
    _validate_model_info(info, plan)
    return info


def _validate_model_info(info, plan):
    e._keys(
        info,
        "schema_version class geometry attention_implementation parameter_dtype tensors tensor_metadata_sha256 weight_content_authenticated all_internal_operations_bf16_asserted",
        "model_metadata",
    )
    require(
        info["schema_version"] == SCHEMA
        and info["parameter_dtype"] == "torch.bfloat16"
        and info["attention_implementation"] == "sdpa"
        and info["weight_content_authenticated"] is False
        and info["all_internal_operations_bf16_asserted"] is False,
        "model_metadata_policy",
    )
    e._str(info["class"])
    cap = plan["manifest"]["capture"]
    same(
        info["geometry"],
        {
            "vocab_size": plan["manifest"]["vocab_size"],
            "hidden_size": cap["hidden_width"],
            "num_hidden_layers": cap["model_layers"],
            "max_position_embeddings": plan["context_limit"],
        },
        "model_metadata_geometry",
    )
    tensors = info["tensors"]
    require(type(tensors) is list and bool(tensors), "model_metadata_tensors")
    names, parameters = set(), 0
    for tensor in tensors:
        e._keys(tensor, "kind name shape dtype device", "model_tensor_metadata")
        require(
            tensor["kind"] in ("parameter", "buffer") and tensor["device"] == "cpu",
            "model_tensor_metadata_policy",
        )
        e._str(tensor["name"])
        e._str(tensor["dtype"])
        require(tensor["name"] not in names, "model_tensor_duplicate")
        names.add(tensor["name"])
        require(type(tensor["shape"]) is list, "model_tensor_shape")
        for size in tensor["shape"]:
            e._int(size)
        if tensor["kind"] == "parameter":
            parameters += 1
            require(tensor["dtype"] == "torch.bfloat16", "model_tensor_parameter_dtype")
        else:
            require(
                tensor["dtype"]
                in {
                    "torch.bfloat16",
                    "torch.float16",
                    "torch.float32",
                    "torch.float64",
                    "torch.bool",
                    "torch.uint8",
                    "torch.int8",
                    "torch.int16",
                    "torch.int32",
                    "torch.int64",
                    "torch.uint16",
                    "torch.uint32",
                    "torch.uint64",
                },
                "model_buffer_dtype",
            )
    require(
        parameters > 0 and info["tensor_metadata_sha256"] == e.object_hash(tensors),
        "model_tensor_metadata_binding",
    )


def widen_bf16(raw, width):
    """Independent little-endian IEEE bit widening, with finite-value guards."""
    e._int(width, 1)
    require(type(raw) is bytes and len(raw) == width * 2, "native_bytes")
    words = [word[0] for word in struct.iter_unpack("<H", raw)]
    require(all(word & 0x7F80 != 0x7F80 for word in words), "native_nonfinite")
    widened = b"".join(struct.pack("<I", word << 16) for word in words)
    require(
        all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", widened)),
        "widened_nonfinite",
    )
    return widened


def _argmax(raw, width):
    require(type(raw) is bytes and len(raw) == width * 4, "widened_bytes")
    values = [value[0] for value in struct.iter_unpack("<f", raw)]
    require(all(math.isfinite(value) for value in values), "widened_nonfinite")
    return max(range(width), key=values.__getitem__)


def _native_raw(torch, tensor, width):
    require(
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.bfloat16
        and tensor.layout == torch.strided
        and tuple(tensor.shape) == (1, 1, width),
        "native_logits",
    )
    require(bool(torch.isfinite(tensor).all().item()), "native_nonfinite")
    require(sys.byteorder == "little", "native_byte_order")
    raw = bytes(tensor.detach().clone().contiguous().view(torch.uint8).reshape(-1).tolist())
    return raw, widen_bf16(raw, width)


def _attempt(plan, attempt):
    e._keys(
        attempt,
        "schema_version run_sha256 plan_sha256 sequence_index step_index input_token_ids input_ids_sha256",
        "call_attempt",
    )
    require(attempt["schema_version"] == SCHEMA, "attempt_schema")
    for key in ("run_sha256", "plan_sha256", "input_ids_sha256"):
        e._hash(attempt[key])
    e._int(attempt["sequence_index"])
    e._int(attempt["step_index"])
    e._ids(attempt["input_token_ids"], plan["manifest"]["vocab_size"], nonempty=True)
    require(len(attempt["input_token_ids"]) <= plan["context_limit"], "input_context")
    require(attempt["input_ids_sha256"] == e.object_hash(attempt["input_token_ids"]), "input_hash")
    return e._snapshot(attempt)


def _forward(model, geometryplan, directory, attempt, counts):
    import torch

    plan = _geometry(geometryplan)
    attempt = _attempt(plan, attempt)
    model_info = clean_model(model, plan)
    require(
        type(counts) is dict
        and "generation" in counts
        and set(counts) <= {"generation", "technical"},
        "counter_keys",
    )
    for value in counts.values():
        e._int(value)
    require(sum(counts.values()) < MAX_ENTRIES, "entry_budget")
    root = e._directory(directory, create=True)
    _write(root / "attempt.json", attempt)
    inputs = support._tensor_inputs(torch, attempt["input_token_ids"])
    inputs.update(output_hidden_states=False, output_attentions=False)
    state = {"entries": 0, "model_call_returned": False, "cleanup_failure_type": None}
    handles, original = [], None

    def check(kwargs):
        require(
            kwargs.get("output_hidden_states") is False
            and kwargs.get("output_attentions") is False,
            "collector_flags",
        )
        support._check_inputs(
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
        support._hook_audit(torch, model, handles)
        check(kwargs)
        require(
            torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cpu")
            and torch.get_num_threads() == 4
            and torch.are_deterministic_algorithms_enabled(),
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
            support._hook_audit(torch, model)
            handles.append((model, "pre", model.register_forward_pre_hook(entry, with_kwargs=True)))
            with torch.inference_mode():
                output = model(**inputs)
            state["model_call_returned"] = True
            check(inputs)
            require(state["entries"] == 1, "one_entry")
            raw, widened = _native_raw(torch, output.logits, plan["manifest"]["vocab_size"])
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
                support._hook_audit(torch, model)
            except BaseException as exc:
                cleanup = cleanup or exc
            if cleanup is not None:
                state["cleanup_failure_type"] = type(cleanup).__name__
                if original is None:
                    raise cleanup
        same(clean_model(model, plan), model_info, "model_metadata_stable")
        e._publish(root / "logits.bf16", raw)
        e._publish(root / "logits.fp32", widened)
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "attempt_sha256": e.object_hash(attempt),
            "entry_sha256": e.object_hash(_read(root / "entry.json")),
            "entries": 1,
            "model_call_returned": True,
            "native_dtype": "bfloat16",
            "storage_dtype": "little_endian_bfloat16_and_exact_float32",
            "native_logits_sha256": e.sha256(raw),
            "logits_sha256": e.sha256(widened),
            "chosen_token_id": _argmax(widened, plan["manifest"]["vocab_size"]),
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


def _replay_call(root, geometryplan, attempt):
    plan = _geometry(geometryplan)
    attempt = _attempt(plan, attempt)
    root = e._directory(root)
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
        == {"attempt.json", "entry.json", "result.json", "logits.bf16", "logits.fp32"},
        "step_inventory",
    )
    native = e._read_file(root / "logits.bf16")
    widened = e._read_file(root / "logits.fp32")
    require(widen_bf16(native, plan["manifest"]["vocab_size"]) == widened, "exact_widening")
    expected = {
        "schema_version": SCHEMA,
        "status": "completed",
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": e.object_hash(expected_entry),
        "entries": 1,
        "model_call_returned": True,
        "native_dtype": "bfloat16",
        "storage_dtype": "little_endian_bfloat16_and_exact_float32",
        "native_logits_sha256": e.sha256(native),
        "logits_sha256": e.sha256(widened),
        "chosen_token_id": _argmax(widened, plan["manifest"]["vocab_size"]),
    }
    same(_read(root / "result.json"), expected, "step_result")
    return expected


def prepare_technical(geometryplan, tokenizer):
    """Render only the fixed harmless prompt; never enter the model."""
    plan = _geometry(geometryplan)
    messages = [{"role": role, "content": text} for role, text in TECHNICAL_MESSAGES]
    e._ids(tokenizer.all_special_ids, plan["manifest"]["vocab_size"], nonempty=True)
    special = sorted(set(tokenizer.all_special_ids))
    template = tokenizer.get_chat_template()
    e._str(template)
    payloads = [
        tokenizer.encode(message["content"], add_special_tokens=False) for message in messages
    ]
    for ids in payloads:
        e._ids(ids, plan["manifest"]["vocab_size"], nonempty=True)
        require(not set(ids).intersection(special), "payload_special")
    options = {"add_generation_prompt": True, "date_string": DATE, "return_dict": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    e._str(rendered)
    require(all(message["content"] in rendered for message in messages), "rendered_messages")
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    e._ids(ids, plan["manifest"]["vocab_size"], nonempty=True)
    same(ids, tokenizer.encode(rendered, add_special_tokens=False), "native_prompt_ids")
    prepared = {
        "schema_version": SCHEMA,
        "geometry_sha256": e.object_hash(plan),
        "messages": messages,
        "rendered_prompt_text": rendered,
        "prompt_token_ids": ids,
        "prompt_ids_sha256": e.object_hash(ids),
        "prompt_utf8_sha256": e.sha256(rendered.encode()),
        "chat_template_sha256": e.sha256(template.encode()),
        "special_token_ids": special,
        "payload_token_ids": payloads,
    }
    return validate_technical_prepared(plan, prepared)


def validate_technical_prepared(geometryplan, prepared):
    plan = _geometry(geometryplan)
    e._keys(
        prepared,
        "schema_version geometry_sha256 messages rendered_prompt_text prompt_token_ids prompt_ids_sha256 prompt_utf8_sha256 chat_template_sha256 special_token_ids payload_token_ids",
        "technical_prepared",
    )
    require(
        prepared["schema_version"] == SCHEMA and prepared["geometry_sha256"] == e.object_hash(plan),
        "technical_binding",
    )
    same(
        prepared["messages"],
        [{"role": role, "content": text} for role, text in TECHNICAL_MESSAGES],
        "technical_messages",
    )
    ids, special = prepared["prompt_token_ids"], prepared["special_token_ids"]
    e._ids(ids, plan["manifest"]["vocab_size"], nonempty=True)
    e._ids(special, plan["manifest"]["vocab_size"], nonempty=True)
    require(special == sorted(set(special)), "special_order")
    require(len(ids) <= min(MAX_PROMPT, plan["context_limit"]), "technical_context")
    e._str(prepared["rendered_prompt_text"])
    require(
        all(text in prepared["rendered_prompt_text"] for _role, text in TECHNICAL_MESSAGES),
        "technical_text",
    )
    require(
        prepared["prompt_ids_sha256"] == e.object_hash(ids)
        and prepared["prompt_utf8_sha256"] == e.sha256(prepared["rendered_prompt_text"].encode()),
        "technical_hashes",
    )
    e._hash(prepared["chat_template_sha256"])
    require(
        type(prepared["payload_token_ids"]) is list and len(prepared["payload_token_ids"]) == 2,
        "technical_payloads",
    )
    for values in prepared["payload_token_ids"]:
        e._ids(values, plan["manifest"]["vocab_size"], nonempty=True)
        require(not set(values).intersection(special), "technical_payload_special")
    return e._snapshot(prepared)


def _technical_attempt(plan, prepared, header, index):
    return {
        "schema_version": SCHEMA,
        "run_sha256": e.object_hash(header),
        "plan_sha256": e.object_hash(plan),
        "sequence_index": 0,
        "step_index": index,
        "input_token_ids": prepared["prompt_token_ids"],
        "input_ids_sha256": prepared["prompt_ids_sha256"],
    }


def _technical_result(root, header, results):
    native_equal = e._read_file(root / "call_0" / "logits.bf16") == e._read_file(
        root / "call_1" / "logits.bf16"
    )
    widened_equal = e._read_file(root / "call_0" / "logits.fp32") == e._read_file(
        root / "call_1" / "logits.fp32"
    )
    return {
        "schema_version": SCHEMA,
        "status": "completed",
        "run_sha256": e.object_hash(header),
        "entries": 2,
        "native_repeatable": native_equal,
        "widened_repeatable": widened_equal,
        "qualified": native_equal and widened_equal,
        "call_result_sha256": [e.object_hash(result) for result in results],
    }


def run_technical(model, *, plan, prepared, directory):
    plan = _geometry(plan)
    prepared = validate_technical_prepared(plan, prepared)
    model_info = clean_model(model, plan)
    root = e._directory(directory, create=True)
    header = {
        "schema_version": SCHEMA,
        "geometry_sha256": e.object_hash(plan),
        "prepared_sha256": e.object_hash(prepared),
        "sources": source_bindings(),
        "model_info": model_info,
    }
    counts = {"generation": 0}
    try:
        _write(root / "run.json", header)
        _write(root / "prepared.json", prepared)
        results = []
        for index in range(2):
            results.append(
                _forward(
                    model,
                    plan,
                    root / f"call_{index}",
                    _technical_attempt(plan, prepared, header, index),
                    counts,
                )
            )
        same(clean_model(model, plan), model_info, "model_metadata_stable")
        result = _technical_result(root, header, results)
        _write(root / "result.json", result)
        terminal = {
            "schema_version": SCHEMA,
            "status": "completed",
            "result_sha256": e.object_hash(result),
            "artifact_sha256": _inventory(root),
        }
        _write(root / "terminal.json", terminal)
        return load_technical(root, plan, prepared)
    except BaseException as exc:
        try:
            _write(
                root / "failure.json",
                {
                    "schema_version": SCHEMA,
                    "run_sha256": e.object_hash(header),
                    "failure_type": type(exc).__name__,
                    "observed_entries": counts["generation"],
                    "usable": False,
                },
            )
        except BaseException:
            pass
        raise


def load_technical(directory, plan, prepared):
    plan = _geometry(plan)
    prepared = validate_technical_prepared(plan, prepared)
    root = e._directory(directory)
    require(
        {p.name for p in root.iterdir()}
        == {"run.json", "prepared.json", "call_0", "call_1", "result.json", "terminal.json"},
        "technical_inventory",
    )
    header = _read(root / "run.json")
    e._keys(
        header,
        "schema_version geometry_sha256 prepared_sha256 sources model_info",
        "technical_header",
    )
    require(
        header["schema_version"] == SCHEMA
        and header["geometry_sha256"] == e.object_hash(plan)
        and header["prepared_sha256"] == e.object_hash(prepared),
        "technical_header_binding",
    )
    same(header["sources"], source_bindings(), "technical_sources")
    _validate_model_info(header["model_info"], plan)
    same(_read(root / "prepared.json"), prepared, "technical_native")
    results = [
        _replay_call(
            root / f"call_{index}", plan, _technical_attempt(plan, prepared, header, index)
        )
        for index in range(2)
    ]
    result = _technical_result(root, header, results)
    same(_read(root / "result.json"), result, "technical_result")
    inventory = _inventory(root)
    del inventory["terminal.json"]
    terminal = {
        "schema_version": SCHEMA,
        "status": "completed",
        "result_sha256": e.object_hash(result),
        "artifact_sha256": inventory,
    }
    same(_read(root / "terminal.json"), terminal, "technical_terminal")
    return {**result, "terminal_sha256": e.object_hash(terminal)}
