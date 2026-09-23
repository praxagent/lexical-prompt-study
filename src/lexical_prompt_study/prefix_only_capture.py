"""One observed-prefix CPU FP32 capture, with durable A183-compatible evidence.

No model loader, tokenizer, generator, fitting, retry, or acquisition schedule is
provided. Model weights and native token/text correspondence require a separate
qualified caller: opaque model/tokenizer pins are declarations, not proof. The
actual adapter source, runtime, module geometry, input and dispatch are recorded.
The model implementation must be trusted and separately qualified; these guards
do not establish arbitrary custom Module semantics or detect malicious code.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from . import landmark_evidence as evidence

EVIDENCE_SHA256 = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
SCHEMA = "prefix-only-capture-v1"


def _require(condition, code):
    if not condition:
        raise ValueError("prefix_capture_" + code)


def source_sha256():
    return evidence.sha256(Path(__file__).read_bytes())


def _source_check(manifest):
    _require(
        evidence.sha256(Path(evidence.__file__).read_bytes()) == EVIDENCE_SHA256, "evidence_source"
    )
    _require(manifest["provenance"]["capture_code_sha256"] == source_sha256(), "adapter_source")


def _arguments(bundle):
    meta, blobs = bundle.metadata, bundle.blobs
    observation = {
        key: meta["observation"][key] for key in ("observation_id", "messages", "prompt_token_ids")
    }
    observation["rendered_prompt_utf8"] = blobs["prompt.utf8"]
    prefix = {key: meta["prefix"][key] for key in ("landmark", "status", "reason", "token_ids")}
    prefix["decoded_utf8"] = blobs.get("prefix.utf8")
    return observation, meta["generation"], prefix, meta["outcome"]


def _runtime(torch):
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    _require(type(available) is int and available > 0, "memory_snapshot")
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "transformers": version("transformers"),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "autocast": bool(torch.is_autocast_enabled("cpu")),
        "available_memory_bytes": available,
        "reservation": "caller_owned_not_verified",
    }


def _hook_audit(torch, model, owned=()):
    module_api = torch.nn.modules.module
    _require(
        not module_api._global_forward_pre_hooks and not module_api._global_forward_hooks,
        "global_hooks",
    )
    allowed = {(id(module), kind, handle.id) for module, kind, handle in owned}
    actual = set()
    for module in model.modules():
        actual.update((id(module), "pre", key) for key in module._forward_pre_hooks)
        actual.update((id(module), "post", key) for key in module._forward_hooks)
    _require(actual == allowed, "model_hooks")


def _model_preflight(torch, model, blocks_path, manifest, sequence_length):
    _require(isinstance(model, torch.nn.Module), "model_type")
    _require(type(blocks_path) is str and bool(blocks_path), "blocks_path")
    _require(not hasattr(model, "_orig_mod"), "compiled_model")
    _require(all(not module.training for module in model.modules()), "evaluation_mode")
    _require(
        torch.get_num_threads() == 4 and not torch.is_autocast_enabled("cpu"), "precision_threads"
    )
    _require(sys.byteorder == "little", "host_byte_order")
    _hook_audit(torch, model)
    cap = manifest["capture"]
    config = model.config
    geometry = {
        key: getattr(config, key)
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings")
    }
    _require(
        all(type(value) is int and value > 0 for value in geometry.values()), "config_geometry"
    )
    _require(
        geometry["vocab_size"] == manifest["vocab_size"]
        and geometry["hidden_size"] == cap["hidden_width"]
        and geometry["num_hidden_layers"] == cap["model_layers"],
        "manifest_geometry",
    )
    _require(sequence_length <= geometry["max_position_embeddings"], "context_limit")
    blocks = model.get_submodule(blocks_path)
    _require(
        isinstance(blocks, torch.nn.ModuleList) and len(blocks) == cap["model_layers"], "blocks"
    )
    block = blocks[cap["layer_index"]]
    _require(block is not model, "block_identity")
    tensors = []
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        _require(value.device.type == "cpu" and value.layout == torch.strided, "model_cpu_strided")
        _require(not value.is_floating_point() or value.dtype == torch.float32, "model_fp32")
        _require(not value.is_complex(), "model_complex")
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    _require(bool(tensors), "empty_model")
    model_info = {
        "class": type(model).__module__ + "." + type(model).__qualname__,
        "block_class": type(block).__module__ + "." + type(block).__qualname__,
        "blocks_path": blocks_path,
        "layer_index": cap["layer_index"],
        "geometry": geometry,
        "tensor_metadata_sha256": evidence.object_hash(tensors),
        "tensor_count": len(tensors),
        "weight_content_authenticated": False,
        "attention_implementation": getattr(config, "_attn_implementation", None),
    }
    return block, model_info


def _tensor_inputs(torch, values):
    ids = torch.tensor([values], dtype=torch.long, device="cpu")
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(len(values), dtype=torch.long).unsqueeze(0),
        "past_key_values": None,
        "use_cache": False,
        "logits_to_keep": 1,
    }


def _check_inputs(torch, actual, values):
    _require(
        set(actual)
        == {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "use_cache",
            "logits_to_keep",
        },
        "forward_arguments",
    )
    _require(
        actual["past_key_values"] is None
        and actual["use_cache"] is False
        and type(actual["logits_to_keep"]) is int
        and actual["logits_to_keep"] == 1,
        "forward_policy",
    )
    for key, expected in (
        ("input_ids", values),
        ("attention_mask", [1] * len(values)),
        ("position_ids", list(range(len(values)))),
    ):
        tensor = actual[key]
        _require(
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.int64
            and tensor.device.type == "cpu"
            and tensor.layout == torch.strided
            and tuple(tensor.shape) == (1, len(values)),
            "forward_tensor",
        )
        _require(tensor.tolist() == [expected], "forward_boundary")


def _measurement(torch, model, block, values, width, directory, attempt, state):
    """One normal module dispatch; owned hooks are always removed, never retried."""
    state["capture"] = None
    handles = []
    inputs = _tensor_inputs(torch, values)

    def dispatch(module, args, kwargs):
        state["dispatches"] += 1
        _require(state["dispatches"] == 1 and module is model and not args, "dispatch_count")
        _hook_audit(torch, model, handles)
        _check_inputs(torch, kwargs, values)
        _require(
            torch.is_inference_mode_enabled()
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cpu"),
            "inference_mode",
        )
        receipt = {
            "schema_version": SCHEMA,
            "attempt_sha256": evidence.object_hash(attempt),
            "event": "model_forward_pre_hook_entry",
            "input_ids_sha256": evidence.object_hash(values),
            "input_length": len(values),
            "dispatch_index": 0,
        }
        evidence._publish(directory / "dispatch.json", evidence.canonical(receipt))
        state["dispatch"] = receipt
        return None

    def capture(module, args, output):
        state["hook_invocations"] += 1
        _require(state["hook_invocations"] == 1 and module is block, "capture_count")
        _hook_audit(torch, model, handles)
        _require(isinstance(output, torch.Tensor), "block_output_tensor")
        _require(
            output.device.type == "cpu"
            and output.dtype == torch.float32
            and output.layout == torch.strided,
            "source_cpu_fp32",
        )
        _require(tuple(output.shape) == (1, len(values), width), "block_output_shape")
        source = output[:, -1, :]
        _require(bool(torch.isfinite(source).all().item()), "source_nonfinite")
        state["capture"] = source.detach().clone().contiguous()
        return None

    original = None
    try:
        handles.append((model, "pre", model.register_forward_pre_hook(dispatch, with_kwargs=True)))
        handles.append((block, "post", block.register_forward_hook(capture)))
        _hook_audit(torch, model, handles)
        with torch.inference_mode():
            output = model(**inputs)
        _check_inputs(torch, inputs, values)
        _require(
            state["dispatches"] == state["hook_invocations"] == 1 and state["capture"] is not None,
            "incomplete_forward",
        )
    except BaseException as exc:
        original = exc
        raise
    finally:
        cleanup_error = None
        for module, kind, handle in reversed(handles):
            try:
                handle.remove()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            finally:
                # Only our exact IDs; never erase a foreign hook. This also
                # handles a failing custom remove() without leaving ours live.
                names = (
                    ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs")
                    if kind == "pre"
                    else (
                        "_forward_hooks",
                        "_forward_hooks_with_kwargs",
                        "_forward_hooks_always_called",
                    )
                )
                for name in names:
                    getattr(module, name).pop(handle.id, None)
        try:
            _hook_audit(torch, model)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if original is not None or cleanup_error is not None:
            state["capture"] = None
        if cleanup_error is not None:
            state["cleanup_failure_type"] = type(cleanup_error).__name__
            if original is None:
                raise cleanup_error
    # uint8 view reinterprets native bits; it does not convert FP32 values.
    raw = bytes(state["capture"].view(torch.uint8).reshape(-1).tolist())
    state["capture"] = None
    return output, raw, state


@dataclass(frozen=True)
class CaptureResult:
    """Evidence bytes are immutable; forward_output is the original mutable output."""

    bundle: evidence.ValidatedBundle
    receipt_bytes: bytes
    forward_output: object

    @property
    def receipt(self):
        return evidence._read_json(self.receipt_bytes)


def _completed_metadata(manifest, observation, prefix, raw, attempt, dispatch):
    prompt, ids = observation["prompt_token_ids"], prefix["token_ids"]
    values = prompt + ids
    item = {
        "observation_id": observation["observation_id"],
        "landmark": prefix["landmark"],
        "prompt_ids_sha256": evidence.object_hash(prompt),
        "prefix_ids_sha256": evidence.object_hash(ids),
        "prompt_utf8_sha256": evidence.sha256(observation["rendered_prompt_utf8"]),
        "prefix_utf8_sha256": evidence.sha256(prefix["decoded_utf8"]),
        "input_token_ids": values,
        "input_ids_sha256": evidence.object_hash(values),
        "input_length": len(values),
        "attention_mask": [1] * len(values),
        "position_ids": list(range(len(values))),
        "absolute_position": len(values) - 1,
        "policy": manifest["capture"],
        "provenance": manifest["provenance"],
        "shape": [1, manifest["capture"]["hidden_width"]],
        "byte_length": len(raw),
        "content_sha256": evidence.sha256(raw),
        "forward_receipt": {
            "attempt_sha256": evidence.object_hash(attempt),
            "dispatch_sha256": evidence.object_hash(dispatch),
            "source_sha256": source_sha256(),
            "input_ids_sha256": evidence.object_hash(values),
        },
    }
    return {"status": "completed", "reason": None, "evidence": item}


def capture_prefix(
    model, *, blocks_path, manifest, observation, generation, prefix, outcome, directory
):
    """Validate/snapshot first, then consume one exclusive attempt directory.

    Unavailable prefixes are retained without dispatch. Model preflight failures
    precede the attempt directory. Once claimed, every exception consumes the
    directory; a failed final forward never returns or publishes a usable capture.
    Only load_capture's whole terminal check certifies adapter publication. An
    orphan nested A183 bundle alone is not a successful adapter result.
    """
    import torch

    manifest = evidence.validate_manifest(manifest)
    _source_check(manifest)
    available = type(prefix) is dict and prefix.get("status") == "available"
    absent = {
        "status": "missing" if available else "unavailable",
        "reason": "not_attempted" if available else "prefix_unavailable",
        "evidence": None,
    }
    initial = evidence.validate_landmark(
        manifest, observation, generation, prefix, absent, None, outcome
    )
    observation, generation, prefix, outcome = _arguments(initial)
    values = observation["prompt_token_ids"] + (prefix["token_ids"] or [])
    runtime = _runtime(torch)
    block, model_info = (None, None)
    if available:
        block, model_info = _model_preflight(torch, model, blocks_path, manifest, len(values))
    attempt = {
        "schema_version": SCHEMA,
        "manifest_sha256": evidence.object_hash(manifest),
        "initial_bundle_sha256": initial.sha256,
        "adapter_source_sha256": source_sha256(),
        "evidence_source_sha256": EVIDENCE_SHA256,
        "observation_id": observation["observation_id"],
        "landmark": prefix["landmark"],
        "prefix_available": available,
        "input_ids_sha256": evidence.object_hash(values) if available else None,
        "input_length": len(values) if available else None,
        "generation_sha256": evidence.object_hash(generation),
        "provenance": manifest["provenance"],
        "runtime": runtime,
        "model": model_info,
    }
    _validate_attempt(attempt, manifest)
    directory = evidence._directory(directory, create=True)
    evidence._publish(directory / "attempt.json", evidence.canonical(attempt))
    state = {"dispatches": 0, "hook_invocations": 0, "dispatch": None, "cleanup_failure_type": None}
    try:
        output = None
        bundle = initial
        if available:
            output, raw, state = _measurement(
                torch,
                model,
                block,
                values,
                manifest["capture"]["hidden_width"],
                directory,
                attempt,
                state,
            )
            completed = _completed_metadata(
                manifest, observation, prefix, raw, attempt, state["dispatch"]
            )
            bundle = evidence.validate_landmark(
                manifest, observation, generation, prefix, completed, raw, outcome
            )
        publication = evidence.commit_landmark(directory / "evidence", bundle, manifest)
        receipt = {
            "schema_version": SCHEMA,
            "status": "completed" if available else "unavailable",
            "attempt_sha256": evidence.object_hash(attempt),
            "dispatch_sha256": evidence.object_hash(state["dispatch"])
            if state["dispatch"]
            else None,
            "dispatches": state["dispatches"],
            "hook_invocations": state["hook_invocations"],
            "bundle_sha256": bundle.sha256,
            "evidence_receipt_sha256": publication["receipt_sha256"],
            "source_device": "cpu" if available else None,
            "source_dtype": "float32" if available else None,
            "failure_type": None,
            "cleanup_failure_type": None,
        }
        evidence._publish(directory / "result.json", evidence.canonical(receipt))
        return CaptureResult(bundle, evidence.canonical(receipt), output)
    except BaseException as exc:
        # A failed publication can leave children or temporary files. Never repair
        # them or overwrite a possibly completed/uncertain terminal receipt.
        try:
            dispatch_path = directory / "dispatch.json"
            dispatch = (
                evidence._read_json(evidence._read_file(dispatch_path))
                if dispatch_path.exists()
                else None
            )
            failure = {
                "schema_version": SCHEMA,
                "status": "failed",
                "attempt_sha256": evidence.object_hash(attempt),
                "dispatch_sha256": evidence.object_hash(dispatch) if dispatch else None,
                "dispatches": state["dispatches"],
                "hook_invocations": state["hook_invocations"],
                "cleanup_failure_type": state["cleanup_failure_type"],
                "failure_type": type(exc).__name__,
                "bundle_sha256": None,
                "capture_usable": False,
            }
            evidence._publish(directory / "failure.json", evidence.canonical(failure))
        except BaseException:
            pass  # Preserve the original exception, including signals/SystemExit.
        raise


def _validate_attempt(attempt, manifest):
    evidence._keys(
        attempt,
        "schema_version manifest_sha256 initial_bundle_sha256 adapter_source_sha256 "
        "evidence_source_sha256 observation_id landmark prefix_available input_ids_sha256 "
        "input_length generation_sha256 provenance runtime model",
        "capture_attempt_fields",
    )
    runtime = attempt["runtime"]
    evidence._keys(
        runtime,
        "python executable torch transformers threads interop_threads device autocast "
        "available_memory_bytes reservation",
        "capture_runtime_fields",
    )
    for key in ("python", "executable", "torch", "transformers"):
        evidence._str(runtime[key])
    for key in ("threads", "interop_threads", "available_memory_bytes"):
        evidence._int(runtime[key], 1)
    _require(
        runtime["device"] == "cpu"
        and type(runtime["autocast"]) is bool
        and runtime["reservation"] == "caller_owned_not_verified",
        "runtime_policy",
    )
    if attempt["prefix_available"]:
        _require(runtime["threads"] == 4 and runtime["autocast"] is False, "runtime_precision")
        model = attempt["model"]
        evidence._keys(
            model,
            "class block_class blocks_path layer_index geometry tensor_metadata_sha256 "
            "tensor_count weight_content_authenticated attention_implementation",
            "model_receipt_fields",
        )
        for key in ("class", "block_class", "blocks_path"):
            evidence._str(model[key])
        evidence._hash(model["tensor_metadata_sha256"])
        evidence._int(model["tensor_count"], 1)
        evidence._int(model["layer_index"])
        _require(
            model["weight_content_authenticated"] is False
            and (
                model["attention_implementation"] is None
                or type(model["attention_implementation"]) is str
            ),
            "model_provenance_claim",
        )
        geometry = model["geometry"]
        evidence._keys(
            geometry,
            "vocab_size hidden_size num_hidden_layers max_position_embeddings",
            "geometry_fields",
        )
        for value in geometry.values():
            evidence._int(value, 1)
        cap = manifest["capture"]
        _require(
            model["layer_index"] == cap["layer_index"]
            and geometry["vocab_size"] == manifest["vocab_size"]
            and geometry["hidden_size"] == cap["hidden_width"]
            and geometry["num_hidden_layers"] == cap["model_layers"],
            "model_geometry_binding",
        )
        evidence._int(attempt["input_length"], 1)
        _require(
            attempt["input_length"] <= geometry["max_position_embeddings"], "retained_context_limit"
        )


def load_capture(directory, manifest):
    """Verify retained publication; never execute, retry, or infer missing success.

    A durable success record does not prove the original caller observed return.
    If a later failure marker could not be persisted, storage alone cannot prove
    that absence. Concurrent readers also cannot certify an in-progress writer's
    final fsync. This loader validates complete retained evidence, not those
    stronger process/durability claims.
    """
    manifest = evidence.validate_manifest(manifest)
    _source_check(manifest)
    directory = evidence._directory(directory)
    attempt = evidence._read_json(evidence._read_file(directory / "attempt.json"))
    _validate_attempt(attempt, manifest)
    result_raw = evidence._read_file(directory / "result.json")
    result = evidence._read_json(result_raw)
    evidence._keys(
        result,
        "schema_version status attempt_sha256 dispatch_sha256 dispatches hook_invocations "
        "bundle_sha256 evidence_receipt_sha256 source_device source_dtype failure_type cleanup_failure_type",
        "capture_result_fields",
    )
    _require(
        result["schema_version"] == SCHEMA and result["status"] in ("completed", "unavailable"),
        "terminal_status",
    )
    completed = result["status"] == "completed"
    _require(
        {p.name for p in directory.iterdir()}
        == {"attempt.json", "result.json", "evidence"}
        | ({"dispatch.json"} if completed else set()),
        "adapter_inventory",
    )
    _require(
        attempt["schema_version"] == SCHEMA
        and attempt["manifest_sha256"] == evidence.object_hash(manifest)
        and attempt["adapter_source_sha256"] == source_sha256()
        and attempt["evidence_source_sha256"] == EVIDENCE_SHA256
        and attempt["provenance"] == manifest["provenance"],
        "attempt_binding",
    )
    _require(
        type(attempt["prefix_available"]) is bool and attempt["prefix_available"] == completed,
        "attempt_availability",
    )
    _require(
        result["attempt_sha256"] == evidence.object_hash(attempt)
        and result["failure_type"] is None
        and result["cleanup_failure_type"] is None,
        "terminal_binding",
    )
    _require(
        type(result["dispatches"]) is int
        and type(result["hook_invocations"]) is int
        and result["dispatches"] == result["hook_invocations"] == int(completed),
        "terminal_counts",
    )
    _require(
        result["source_device"] == ("cpu" if completed else None)
        and result["source_dtype"] == ("float32" if completed else None),
        "source_type",
    )
    bundle = evidence.load_landmark(directory / "evidence", manifest)
    observation, generation, prefix, outcome = _arguments(bundle)
    absent = {
        "status": "missing" if completed else "unavailable",
        "reason": "not_attempted" if completed else "prefix_unavailable",
        "evidence": None,
    }
    initial = evidence.validate_landmark(
        manifest, observation, generation, prefix, absent, None, outcome
    )
    _require(
        attempt["initial_bundle_sha256"] == initial.sha256
        and attempt["observation_id"] == observation["observation_id"]
        and type(attempt["landmark"]) is int
        and attempt["landmark"] == prefix["landmark"]
        and attempt["generation_sha256"] == evidence.object_hash(generation),
        "initial_binding",
    )
    receipt = evidence._read_file(directory / "evidence" / "receipt.json")
    _require(
        result["bundle_sha256"] == bundle.sha256
        and result["evidence_receipt_sha256"] == evidence.sha256(receipt),
        "evidence_binding",
    )
    if completed:
        values = observation["prompt_token_ids"] + prefix["token_ids"]
        _require(
            type(attempt["input_length"]) is int
            and attempt["input_length"] == len(values)
            and attempt["input_ids_sha256"] == evidence.object_hash(values),
            "attempt_input",
        )
        dispatch = evidence._read_json(evidence._read_file(directory / "dispatch.json"))
        expected_dispatch = {
            "schema_version": SCHEMA,
            "attempt_sha256": evidence.object_hash(attempt),
            "event": "model_forward_pre_hook_entry",
            "input_ids_sha256": evidence.object_hash(values),
            "input_length": len(values),
            "dispatch_index": 0,
        }
        _require(
            evidence.canonical(dispatch) == evidence.canonical(expected_dispatch)
            and result["dispatch_sha256"] == evidence.object_hash(dispatch),
            "dispatch_binding",
        )
        forward = bundle.metadata["capture"]["evidence"]["forward_receipt"]
        _require(
            forward["attempt_sha256"] == result["attempt_sha256"]
            and forward["dispatch_sha256"] == result["dispatch_sha256"],
            "capture_dispatch_binding",
        )
    else:
        _require(
            attempt["input_length"] is None
            and attempt["input_ids_sha256"] is None
            and result["dispatch_sha256"] is None
            and attempt["model"] is None,
            "unavailable_dispatch",
        )
    return CaptureResult(bundle, result_raw, None)
