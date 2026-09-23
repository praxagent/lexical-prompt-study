"""Invented analytical/tiny-untrained BF16 qualification; no native assets."""

import copy
import struct
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import bf16_forward as b


@pytest.fixture(autouse=True)
def runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


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
            "date_string": "26 Jul 2024",
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


def test_forward_native_widening_argmax_inputs_and_output_unchanged(tmp_path):
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


def test_model_inventory_records_actual_buffer_dtypes():
    model = Model()
    result = b.clean_model(model, plan())
    assert {row["dtype"] for row in result["tensors"] if row["kind"] == "buffer"} == {
        "torch.float32",
        "torch.int64",
    }
    assert {row["dtype"] for row in result["tensors"] if row["kind"] == "parameter"} == {
        "torch.bfloat16"
    }
    assert result["weight_content_authenticated"] is False
    assert result["all_internal_operations_bf16_asserted"] is False
    assert model.fp32_buffer.dtype == torch.float32


@pytest.mark.parametrize(
    "bad",
    [
        "fp32",
        "training",
        "threads",
        "determinism",
        "autocast",
        "geometry",
        "boolgeometry",
        "attention",
        "hidden",
        "quantized",
        "prehook",
        "posthook",
        "globalhook",
    ],
)
def test_model_preflight_guards(bad):
    model, handle = Model(), None
    if bad == "fp32":
        model.float()
    elif bad == "training":
        model.train()
    elif bad == "threads":
        torch.set_num_threads(1)
    elif bad == "determinism":
        torch.use_deterministic_algorithms(False)
    elif bad == "geometry":
        model.config.hidden_size = 16
    elif bad == "boolgeometry":
        model.config.num_hidden_layers = True
    elif bad == "attention":
        model.config._attn_implementation = "eager"
    elif bad == "hidden":
        model.config.output_hidden_states = True
    elif bad == "quantized":
        model.is_quantized = True
    elif bad == "prehook":
        handle = model.register_forward_pre_hook(lambda *_: None)
    elif bad == "posthook":
        handle = model.model.layers[0].register_forward_hook(lambda *_: None)
    elif bad == "globalhook":
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *_: None)
    try:
        with torch.autocast("cpu", enabled=bad == "autocast"):
            with pytest.raises(ValueError):
                b.clean_model(model, plan())
    finally:
        if handle:
            handle.remove()


@pytest.mark.parametrize(
    "mode", ["float32", "shape", "nan", "inf", "tuple", "mutate", "parameter_mutate", "reenter"]
)
def test_forward_failure_never_usable(mode, tmp_path):
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
def test_original_baseexception_and_cleanup(error, tmp_path):
    model = Model()
    model.error = error
    with pytest.raises(type(error)) as caught:
        run(model, tmp_path / "call")
    assert caught.value is error
    failure = b._read(tmp_path / "call/failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is False
    hooks_clear(model)


def test_cleanup_error_preserves_original(tmp_path, monkeypatch):
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


def test_entry_write_failure_counts_dispatch_not_body(tmp_path, monkeypatch):
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
def test_counter_preflight_no_entry(counts, tmp_path):
    model = Model()
    with pytest.raises(ValueError):
        run(model, tmp_path / "call", counts)
    assert not model.inputs and not (tmp_path / "call").exists()


def test_final_permitted_global_entry(tmp_path):
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
def test_replay_rejects_tampering(tamper, tmp_path):
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


def test_terminal_publication_failure_preserved_consumed(tmp_path, monkeypatch):
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


def test_integrated_identical_technical_checks(tmp_path):
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


def test_integrated_mismatch_completed_false_no_tuning(tmp_path):
    model = Model()
    model.mode = "vary"
    prepared = b.prepare_technical(plan(), Tokenizer())
    result = b.run_technical(
        model, plan=plan(), prepared=prepared, directory=tmp_path / "technical"
    )
    assert result["qualified"] is False and result["entries"] == 2
    assert len(model.inputs) == 2
    assert result == b.load_technical(tmp_path / "technical", plan(), prepared)


def test_integrated_failure_stops_before_second(tmp_path):
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


def test_tiny_untrained_native_bf16_baseline_and_two_checks(tmp_path):
    torch.manual_seed(188)
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
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    prepared = b.prepare_technical(plan(), Tokenizer())
    kwargs = b.support._tensor_inputs(torch, prepared["prompt_token_ids"])
    with torch.inference_mode():
        baseline = model(**kwargs, output_hidden_states=False, output_attentions=False).logits
    assert baseline.dtype == torch.bfloat16
    expected = bytes(baseline.clone().contiguous().view(torch.uint8).reshape(-1).tolist())
    result = b.run_technical(
        model, plan=plan(), prepared=prepared, directory=tmp_path / "technical"
    )
    assert result["qualified"] is True and result["entries"] == 2
    assert (tmp_path / "technical/call_0/logits.bf16").read_bytes() == expected
    assert (tmp_path / "technical/call_1/logits.bf16").read_bytes() == expected
    independent_fp32 = bytes(baseline.float().contiguous().view(torch.uint8).reshape(-1).tolist())
    assert (tmp_path / "technical/call_0/logits.fp32").read_bytes() == independent_fp32
    hooks_clear(model)
    snapshot = copy.deepcopy(prepared)
    b.load_technical(tmp_path / "technical", plan(), prepared)
    assert prepared == snapshot


@pytest.mark.parametrize("mutation", ["parameter", "boolshape", "duplicate", "assertion"])
def test_model_metadata_replay_rejects_false_declarations(mutation):
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


def test_replay_uses_no_torch_numpy_or_model_import(tmp_path, monkeypatch):
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

    monkeypatch.setattr(builtins, "__import__", restricted)
    assert b.load_technical(tmp_path / "technical", plan(), prepared) == result
