"""A189 CPU control fixtures and one root-owned tiny CUDA qualification.

CPU transport fixtures simulate receipt/control flow, not CUDA execution.
No pretrained model, native tokenizer, assets, completion or detector fitting.
"""

import copy
import struct
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import cuda_bf16_forward as b


def plan():
    return {
        "context_limit": 128,
        "manifest": {
            "vocab_size": 32,
            "capture": {"hidden_width": 8, "model_layers": 2, "layer_index": 1},
        },
    }


def attempt(ids=None):
    ids = [1, 2, 3] if ids is None else ids
    return {
        "schema_version": b.SCHEMA,
        "run_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "sequence_index": 0,
        "step_index": 0,
        "input_token_ids": ids,
        "input_ids_sha256": b.e.object_hash(ids),
    }


class Tokenizer:
    all_special_ids = [31]

    def get_chat_template(self):
        return "invented-bf16-template"

    def render(self, messages):
        return (
            "[system]" + messages[0]["content"] + "[user]" + messages[1]["content"] + "[assistant]"
        )

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs == {
            "add_generation_prompt": True,
            "date_string": "23 Sep 2026",
            "return_dict": False,
        }
        return [1, 2, 3] if tokenize else self.render(messages)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [1, 2, 3] if text.startswith("[system]") else [4, 5]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            max_position_embeddings=128,
            output_hidden_states=False,
            output_attentions=False,
            _attn_implementation="sdpa",
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, dtype=torch.bfloat16) for _ in range(2)]
        )
        self.model.norm = torch.nn.LayerNorm(8, dtype=torch.bfloat16)
        self.register_buffer("fp32_buffer", torch.tensor([2.0], dtype=torch.float32))
        self.register_buffer("integer_buffer", torch.tensor([1], dtype=torch.int64))
        self.mode, self.error, self.output, self.callback = None, None, None, None
        self.inputs = []
        self.eval()

    def forward(self, **kwargs):
        self.inputs.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        if self.callback:
            self.callback(kwargs)
        if self.error:
            raise self.error
        logits = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
        logits[0, 0, 2:4] = 2
        if self.mode == "float32":
            logits = logits.float()
        elif self.mode == "shape":
            logits = logits.repeat(1, 2, 1)
        elif self.mode == "nan":
            logits[0, 0, 0] = float("nan")
        elif self.mode == "inf":
            logits[0, 0, 0] = float("inf")
        elif self.mode == "tuple":
            logits = (logits,)
        elif self.mode == "vary":
            logits[0, 0, 7] = len(self.inputs)
        elif self.mode == "mutate":
            kwargs["input_ids"][0, 0] = 4
        elif self.mode == "parameter_mutate":
            self.model.layers[0].weight.data = self.model.layers[0].weight.data.float()
        elif self.mode == "reenter":
            self.mode = None
            self(**kwargs)
        self.output = SimpleNamespace(logits=logits)
        return self.output


def run(model, path, counts=None):
    return b._forward(
        model, plan(), path, attempt(), {"generation": 0} if counts is None else counts
    )


def hooks_clear(model):
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def test_known_bit_patterns_preserve_signed_zero_subnormal_extremes():
    words = [0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F]
    native = struct.pack("<8H", *words)
    widened = b.widen_bf16(native, len(words))
    assert widened == struct.pack("<8I", *(word << 16 for word in words))
    assert list(struct.unpack("<8f", widened))[:6] == [0.0, -0.0, 2**-133, -(2**-133), 1.0, -1.0]
    assert b._argmax(struct.pack("<4f", -0.0, 0.0, -1.0, -2.0), 4) == 0


@pytest.mark.parametrize("word", [0x7F80, 0xFF80, 0x7F81, 0x7FC0, 0xFFC0])
def test_struct_nonfinite_rejected(word):
    with pytest.raises(ValueError):
        b.widen_bf16(struct.pack("<H", word), 1)


@pytest.mark.parametrize(
    "raw,width", [(b"", 1), (b"abc", 1), (bytearray(2), 1), (b"\x00\x00", True), (b"\x00\x00", 0)]
)
def test_struct_shape_type_rejected(raw, width):
    with pytest.raises(ValueError):
        b.widen_bf16(raw, width)


def test_forward_native_widening_argmax_inputs_and_output_unchanged(cpu_transport, tmp_path):
    model, counts = Model(), {"generation": 0, "technical": 2}
    result = run(model, tmp_path / "call", counts)
    assert counts == {"generation": 1, "technical": 2}
    native = (tmp_path / "call/logits.bf16").read_bytes()
    assert native == struct.pack("<32H", *([0, 0, 0x4000, 0x4000] + [0] * 28))
    assert (tmp_path / "call/logits.fp32").read_bytes() == b.widen_bf16(native, 32)
    assert result["chosen_token_id"] == 2
    assert b._replay_call(tmp_path / "call", plan(), attempt()) == result
    assert model.output.logits.dtype == torch.bfloat16 and model.output.logits[0, 0, 2] == 2
    actual = model.inputs[0]
    assert actual["input_ids"].tolist() == [[1, 2, 3]]
    assert actual["attention_mask"].tolist() == [[1, 1, 1]]
    assert actual["position_ids"].tolist() == [[0, 1, 2]]
    assert actual["past_key_values"] is None and actual["use_cache"] is False
    assert (
        actual["logits_to_keep"] == 1
        and actual["output_hidden_states"] is False
        and actual["output_attentions"] is False
    )
    hooks_clear(model)
    with pytest.raises((ValueError, FileExistsError)):
        run(model, tmp_path / "call")
    assert len(model.inputs) == 1


@pytest.mark.parametrize(
    "mode", ["float32", "shape", "nan", "inf", "tuple", "mutate", "parameter_mutate", "reenter"]
)
def test_forward_failure_never_usable(cpu_transport, mode, tmp_path):
    model, counts = Model(), {"generation": 0}
    model.mode = mode
    with pytest.raises(ValueError):
        run(model, tmp_path / "call", counts)
    failure = b._read(tmp_path / "call/failure.json")
    assert failure["usable"] is False
    assert failure["entries"] == (2 if mode == "reenter" else 1)
    assert failure["model_call_returned"] is (mode != "reenter")
    assert not (tmp_path / "call/result.json").exists()
    hooks_clear(model)


@pytest.mark.parametrize("error", [RuntimeError("invented"), KeyboardInterrupt(), SystemExit(3)])
def test_original_baseexception_and_cleanup(cpu_transport, error, tmp_path):
    model = Model()
    model.error = error
    with pytest.raises(type(error)) as caught:
        run(model, tmp_path / "call")
    assert caught.value is error
    failure = b._read(tmp_path / "call/failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is False
    hooks_clear(model)


def test_cleanup_error_preserves_original(cpu_transport, tmp_path, monkeypatch):
    model, original = Model(), SystemExit(7)
    model.error = original
    register = model.register_forward_pre_hook

    def broken(*args, **kwargs):
        handle = register(*args, **kwargs)

        def remove():
            raise RuntimeError("cleanup")

        handle.remove = remove
        return handle

    monkeypatch.setattr(model, "register_forward_pre_hook", broken)
    with pytest.raises(SystemExit) as caught:
        run(model, tmp_path / "call")
    assert caught.value is original
    assert b._read(tmp_path / "call/failure.json")["cleanup_failure_type"] == "RuntimeError"
    hooks_clear(model)


def test_entry_write_failure_counts_dispatch_not_body(cpu_transport, tmp_path, monkeypatch):
    model, counts, write = Model(), {"generation": 0}, b._write

    def fail(path, value):
        if path.name == "entry.json":
            raise RuntimeError("entry publication")
        write(path, value)

    monkeypatch.setattr(b, "_write", fail)
    with pytest.raises(RuntimeError):
        run(model, tmp_path / "call", counts)
    assert counts["generation"] == 1 and not model.inputs
    failure = b._read(tmp_path / "call/failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is False
    hooks_clear(model)


@pytest.mark.parametrize(
    "counts",
    [
        {"generation": True},
        {"generation": -1},
        {"generation": 2048, "technical": 2},
        {"generation": 0, "capture": 0},
    ],
)
def test_counter_preflight_no_entry(cpu_transport, counts, tmp_path):
    model = Model()
    with pytest.raises(ValueError):
        run(model, tmp_path / "call", counts)
    assert not model.inputs and not (tmp_path / "call").exists()


def test_final_permitted_global_entry(cpu_transport, tmp_path):
    counts = {"generation": 2047, "technical": 2}
    run(Model(), tmp_path / "call", counts)
    assert sum(counts.values()) == 2050


@pytest.mark.parametrize(
    "tamper",
    [
        "native",
        "widened",
        "boolattempt",
        "boolentry",
        "boolresult",
        "chosen",
        "extra",
        "symlink",
        "failure",
    ],
)
def test_replay_rejects_tampering(cpu_transport, tamper, tmp_path):
    root = tmp_path / "call"
    run(Model(), root)
    if tamper in ("native", "widened"):
        file = root / ("logits.bf16" if tamper == "native" else "logits.fp32")
        raw = bytearray(file.read_bytes())
        raw[0] = 1
        file.write_bytes(raw)
    elif tamper in ("boolattempt", "boolentry", "boolresult", "chosen"):
        file, key, value = {
            "boolattempt": ("attempt.json", "sequence_index", False),
            "boolentry": ("entry.json", "entry_index", False),
            "boolresult": ("result.json", "entries", True),
            "chosen": ("result.json", "chosen_token_id", 3),
        }[tamper]
        data = b._read(root / file)
        data[key] = value
        (root / file).write_bytes(b.e.canonical(data))
    elif tamper == "extra":
        (root / "extra").write_bytes(b"x")
    elif tamper == "failure":
        (root / "failure.json").write_bytes(b"{}")
    elif tamper == "symlink":
        data = (root / "logits.bf16").read_bytes()
        (root / "logits.bf16").unlink()
        (tmp_path / "bytes").write_bytes(data)
        (root / "logits.bf16").symlink_to(tmp_path / "bytes")
    with pytest.raises((ValueError, FileNotFoundError)):
        b._replay_call(root, plan(), attempt())


def test_terminal_publication_failure_preserved_consumed(cpu_transport, tmp_path, monkeypatch):
    model, write = Model(), b._write
    prepared = b.prepare_technical(plan(), Tokenizer())

    def fail(path, value):
        write(path, value)
        if path.name == "terminal.json":
            raise RuntimeError("after terminal publication")

    monkeypatch.setattr(b, "_write", fail)
    with pytest.raises(RuntimeError):
        b.run_technical(model, plan=plan(), prepared=prepared, directory=tmp_path / "technical")
    assert len(model.inputs) == 2
    with pytest.raises(ValueError):
        b.load_technical(tmp_path / "technical", plan(), prepared)
    with pytest.raises((ValueError, FileExistsError)):
        b.run_technical(model, plan=plan(), prepared=prepared, directory=tmp_path / "technical")
    assert len(model.inputs) == 2


def test_integrated_identical_technical_checks(cpu_transport, tmp_path):
    model = Model()
    prepared = b.prepare_technical(plan(), Tokenizer())
    result = b.run_technical(
        model, plan=plan(), prepared=prepared, directory=tmp_path / "technical"
    )
    assert result["qualified"] is True and result["entries"] == 2
    assert result["native_repeatable"] is True and result["widened_repeatable"] is True
    assert len(model.inputs) == 2
    assert (
        model.inputs[0]["input_ids"].tolist()
        == model.inputs[1]["input_ids"].tolist()
        == [[1, 2, 3]]
    )
    assert result == b.load_technical(tmp_path / "technical", plan(), prepared)
    hooks_clear(model)


def test_integrated_mismatch_completed_false_no_tuning(cpu_transport, tmp_path):
    model = Model()
    model.mode = "vary"
    prepared = b.prepare_technical(plan(), Tokenizer())
    result = b.run_technical(
        model, plan=plan(), prepared=prepared, directory=tmp_path / "technical"
    )
    assert result["qualified"] is False and result["entries"] == 2
    assert len(model.inputs) == 2
    assert result == b.load_technical(tmp_path / "technical", plan(), prepared)


def test_integrated_failure_stops_before_second(cpu_transport, tmp_path):
    model = Model()
    model.mode = "float32"
    prepared = b.prepare_technical(plan(), Tokenizer())
    with pytest.raises(ValueError):
        b.run_technical(model, plan=plan(), prepared=prepared, directory=tmp_path / "technical")
    assert len(model.inputs) == 1
    assert b._read(tmp_path / "technical/failure.json")["observed_entries"] == 1
    with pytest.raises(ValueError):
        b.load_technical(tmp_path / "technical", plan(), prepared)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_token_ids", [True, 2, 3]),
        ("messages", [{"role": "user", "content": "other"}]),
        ("geometry_sha256", "0" * 64),
        ("prompt_ids_sha256", "0" * 64),
        ("special_token_ids", [True, 31]),
    ],
)
def test_technical_native_binding_guards(field, value):
    prepared = b.prepare_technical(plan(), Tokenizer())
    prepared[field] = value
    with pytest.raises(ValueError):
        b.validate_technical_prepared(plan(), prepared)


def test_native_special_and_token_boundary_guards():
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = [1, True, 31]
    with pytest.raises(ValueError):
        b.prepare_technical(plan(), tokenizer)
    tokenizer = Tokenizer()
    original = tokenizer.encode
    tokenizer.encode = lambda text, **kwargs: (
        [7, 8] if text.startswith("[system]") else original(text, **kwargs)
    )
    with pytest.raises(ValueError):
        b.prepare_technical(plan(), tokenizer)


@pytest.mark.parametrize("mutation", ["parameter", "boolshape", "duplicate", "assertion"])
def test_model_metadata_replay_rejects_false_declarations(cpu_transport, mutation):
    info = b.clean_model(Model(), plan())
    if mutation == "parameter":
        info["tensors"][0]["dtype"] = "torch.float32"
    elif mutation == "boolshape":
        info["tensors"][0]["shape"][0] = True
    elif mutation == "duplicate":
        info["tensors"].append(copy.deepcopy(info["tensors"][0]))
    else:
        info["all_internal_operations_bf16_asserted"] = True
    info["tensor_metadata_sha256"] = b.e.object_hash(info["tensors"])
    with pytest.raises(ValueError):
        b._validate_model_info(info, plan())


def test_replay_uses_no_torch_numpy_or_model_import(cpu_transport, tmp_path, monkeypatch):
    import builtins

    prepared = b.prepare_technical(plan(), Tokenizer())
    result = b.run_technical(
        Model(), plan=plan(), prepared=prepared, directory=tmp_path / "technical"
    )
    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "transformers", "tokenizers"}:
            raise AssertionError("replay imports model code")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(builtins, "__import__", restricted)
        assert b.load_technical(tmp_path / "technical", plan(), prepared) == result


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    flags = (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cuda.enable_math_sdp(flags[0])
    torch.backends.cuda.enable_flash_sdp(flags[1])
    torch.backends.cuda.enable_mem_efficient_sdp(flags[2])
    torch.backends.cuda.enable_cudnn_sdp(flags[3])
    torch.backends.cuda.matmul.allow_tf32 = flags[4]
    torch.backends.cudnn.allow_tf32 = flags[5]
    torch.set_float32_matmul_precision(flags[6])


@pytest.fixture
def cpu_transport(monkeypatch):
    """Invented CPU transport: qualify receipt/control flow, not GPU semantics."""

    def info(model, geometryplan):
        b._geometry(geometryplan)
        b.require(
            all(p.dtype == torch.bfloat16 for p in model.parameters()), "fixture_parameter_bf16"
        )
        tensors = [
            {
                "kind": kind,
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": "cuda:0",
            }
            for kind, named in (
                ("parameter", model.named_parameters()),
                ("buffer", model.named_buffers()),
            )
            for name, tensor in named
        ]
        result = {
            "schema_version": b.SCHEMA,
            "class": "invented.cpu_transport.Model",
            "geometry": {
                "vocab_size": 32,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "max_position_embeddings": 128,
            },
            "attention_implementation": "sdpa",
            "execution": b._expected_execution(),
            "parameter_dtype": "torch.bfloat16",
            "tensors": tensors,
            "tensor_metadata_sha256": b.e.object_hash(tensors),
            "weight_content_authenticated": False,
            "all_internal_operations_bf16_asserted": False,
        }
        b._validate_model_info(result, geometryplan)
        return result

    def raw(_torch, tensor, width):
        b.require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.bfloat16
            and tuple(tensor.shape) == (1, 1, width),
            "fixture_logits",
        )
        data = bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
        return data, b.widen_bf16(data, width)

    sync = []
    monkeypatch.setattr(b, "clean_model", info)
    monkeypatch.setattr(b, "_runtime_policy", lambda torch: b._expected_execution())
    monkeypatch.setattr(b, "_tensor_inputs", b.support._tensor_inputs)
    monkeypatch.setattr(b, "_check_inputs", b.support._check_inputs)
    monkeypatch.setattr(b, "_native_raw", raw)
    monkeypatch.setattr(b, "_synchronize", lambda torch: sync.append(True))
    return sync


def test_cpu_model_rejected_without_device_migration(monkeypatch):
    monkeypatch.setattr(b, "_runtime_policy", lambda torch: b._expected_execution())
    model = Model()
    with pytest.raises(ValueError, match="model_cuda_strided"):
        b.clean_model(model, plan())
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_cpu_native_logits_rejected():
    with pytest.raises(ValueError, match="native_logits"):
        b._native_raw(torch, torch.zeros((1, 1, 32), dtype=torch.bfloat16), 32)


def test_cpu_input_tensors_rejected():
    with pytest.raises(ValueError, match="forward_cuda_tensor"):
        b._check_inputs(torch, b.support._tensor_inputs(torch, [1, 2]), [1, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("math_sdp", False),
        ("flash_sdp", True),
        ("memory_efficient_sdp", True),
        ("cudnn_sdp", True),
        ("matmul_tf32", True),
        ("cudnn_tf32", True),
        ("cpu_autocast", True),
        ("cuda_autocast", True),
        ("threads", 1),
        ("deterministic_algorithms", False),
        ("deterministic_warn_only", True),
        ("float32_matmul_precision", "high"),
        ("cublas_workspace_config", ":16:8"),
        ("device", "cuda:1"),
    ],
)
def test_exact_execution_metadata_guards(field, value, cpu_transport):
    info = b.clean_model(Model(), plan())
    info["execution"][field] = value
    with pytest.raises(ValueError):
        b._validate_model_info(info, plan())


def test_runtime_backend_policy_without_cuda_execution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert b._runtime_policy(torch) == b._expected_execution()
    torch.backends.cuda.enable_cudnn_sdp(True)
    with pytest.raises(ValueError):
        b._runtime_policy(torch)


@pytest.mark.parametrize("sync_error", [RuntimeError("asynchronous kernel"), SystemExit(8)])
def test_sync_failure_invalidates_returned_forward(
    cpu_transport, sync_error, tmp_path, monkeypatch
):
    model = Model()

    def fail(_torch):
        raise sync_error

    monkeypatch.setattr(b, "_synchronize", fail)
    with pytest.raises(type(sync_error)) as caught:
        run(model, tmp_path / "call")
    assert caught.value is sync_error
    failure = b._read(tmp_path / "call/failure.json")
    assert (
        failure["entries"] == 1
        and failure["model_call_returned"] is True
        and failure["cuda_synchronized"] is False
        and failure["usable"] is False
    )
    assert not (tmp_path / "call/logits.bf16").exists()
    hooks_clear(model)


def test_sync_happens_before_readout_and_publication(cpu_transport, tmp_path, monkeypatch):
    events = []
    native = b._native_raw
    publish = b.e._publish
    monkeypatch.setattr(b, "_synchronize", lambda torch: events.append("sync"))

    def raw(*args):
        assert events == ["sync"]
        events.append("readout")
        return native(*args)

    def save(path, data):
        if path.name in ("logits.bf16", "logits.fp32", "result.json"):
            assert events[:2] == ["sync", "readout"]
        return publish(path, data)

    monkeypatch.setattr(b, "_native_raw", raw)
    monkeypatch.setattr(b.e, "_publish", save)
    result = run(Model(), tmp_path / "call")
    assert result["native_device"] == "cuda:0" and result["cuda_synchronized"] is True


def test_metadata_rejects_cpu_offload_declaration(cpu_transport):
    info = b.clean_model(Model(), plan())
    info["tensors"][0]["device"] = "cpu"
    info["tensor_metadata_sha256"] = b.e.object_hash(info["tensors"])
    with pytest.raises(ValueError):
        b._validate_model_info(info, plan())


def test_cuda_tiny_untrained_baseline_and_two_checks(tmp_path):
    """Root owns this single real CUDA fixture after the live headroom check."""
    assert torch.cuda.is_available()
    torch.manual_seed(189)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda:0"):
            model = LlamaForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    baseline = kwargs = None
    try:
        prepared = b.prepare_technical(plan(), Tokenizer())
        info = b.clean_model(model, plan())
        assert all(row["device"] == "cuda:0" for row in info["tensors"])
        kwargs = b._tensor_inputs(torch, prepared["prompt_token_ids"])
        with torch.inference_mode():
            baseline = model(**kwargs, output_hidden_states=False, output_attentions=False).logits
        torch.cuda.synchronize("cuda:0")
        assert baseline.dtype == torch.bfloat16 and str(baseline.device) == "cuda:0"
        expected = bytes(baseline.contiguous().view(torch.uint8).reshape(-1).cpu().tolist())
        result = b.run_technical(
            model, plan=plan(), prepared=prepared, directory=tmp_path / "technical"
        )
        assert result["qualified"] is True and result["entries"] == 2
        for index in range(2):
            assert (tmp_path / f"technical/call_{index}/logits.bf16").read_bytes() == expected
            receipt = b._read(tmp_path / f"technical/call_{index}/result.json")
            assert receipt["native_device"] == "cuda:0" and receipt["cuda_synchronized"] is True
        independent_fp32 = bytes(
            baseline.float().contiguous().view(torch.uint8).reshape(-1).cpu().tolist()
        )
        assert (tmp_path / "technical/call_0/logits.fp32").read_bytes() == independent_fp32
        with pytest.raises(ValueError):
            b._native_raw(torch, baseline.float(), 32)
        with pytest.raises(ValueError):
            b._native_raw(torch, baseline.cpu(), 32)
        hooks_clear(model)
        snapshot = copy.deepcopy(prepared)
        assert result == b.load_technical(tmp_path / "technical", plan(), prepared)
        assert prepared == snapshot
    finally:
        baseline = kwargs = model = None
        import gc

        gc.collect()
        torch.cuda.synchronize("cuda:0")
        torch.cuda.empty_cache()
