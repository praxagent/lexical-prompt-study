"""A185 invented native declarations and analytical/tiny untrained qualification."""

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import native_prefix_qualification as a


@pytest.fixture(autouse=True)
def four_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


class Tokenizer:
    eos_token_id = 63
    all_special_ids = [63]

    def __init__(self):
        self.bad = None

    def get_chat_template(self):
        return "invented-native-template"

    def render(self, messages):
        return (
            "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"
        )

    def prompt(self, text):
        return [1, 2 if a.USERS[0] in text else 3, 4]

    def apply_chat_template(
        self, messages, *, tokenize, date_string, add_generation_prompt, return_dict=False
    ):
        assert date_string == "26 Jul 2024" and return_dict is False
        text = self.render(messages)
        if not tokenize:
            return text
        if add_generation_prompt:
            values = self.prompt(text)
            return values * 44 if self.bad == "long" else values
        assert messages[-1] == {"role": "assistant", "content": "probe"}
        return self.prompt(text) + [5, True if self.bad == "closed_bool" else self.eos_token_id]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in (a.SYSTEM, *a.USERS):
            return (
                [63]
                if self.bad == "special"
                else [True]
                if self.bad == "payload_bool"
                else [11, 12]
            )
        values = self.prompt(text)
        if self.bad == "reencode_bool":
            values[0] = True
        if text.endswith("probe"):
            values += [5]
        return values

    def decode(self, values, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        if values == [5]:
            return "probe"
        return "".join(chr(65 + value % 26) for value in values)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(8))

    def forward(self, value):
        return value.cumsum(1) @ self.weight


class Model(torch.nn.Module):
    def __init__(self, eos_after=(None, None)):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=2,
            max_position_embeddings=64,
            _attn_implementation="sdpa",
            output_hidden_states=False,
            output_attentions=False,
        )
        self.embed = torch.nn.Embedding(64, 8)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(), Block()])
        self.model.norm = torch.nn.LayerNorm(8)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(512).reshape(64, 8) / 100)
        self.eos_after = eos_after
        self.entries = []
        self.mode = None
        self.error = None
        self.callback = None
        self.eval()

    def forward(self, **kwargs):
        self.entries.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        if self.callback:
            self.callback()
        hidden = self.embed(kwargs["input_ids"])
        for block in self.model.layers:
            hidden = block(hidden)
        self.model.norm(hidden)
        if self.error:
            raise self.error
        fixture = 0 if int(kwargs["input_ids"][0, 1]) == 2 else 1
        generated = kwargs["input_ids"].shape[1] - 3
        eos = self.eos_after[fixture]
        chosen = 63 if eos is not None and generated >= eos else 0
        logits = torch.zeros((1, 1, 64), dtype=torch.float32)
        if chosen:
            logits[0, 0, chosen] = 1
        if self.mode == "mismatch" and self.model.norm._forward_pre_hooks:
            logits[0, 0, 7] += 0.25
        if self.mode == "nonfinite":
            logits[0, 0, 1] = float("nan")
        if self.mode == "dtype":
            logits = logits.double()
        if self.mode == "shape":
            logits = logits.expand(1, 2, 64)
        if self.mode == "mutate_input":
            kwargs["input_ids"][0, 0] += 1
        return SimpleNamespace(logits=logits)


def setup(model=None, tokenizer=None):
    model = Model() if model is None else model
    tokenizer = Tokenizer() if tokenizer is None else tokenizer
    runtime = {"declared_runtime": "invented-only"}
    plan = a.compile_plan(
        model_sha256=a.e.sha256(b"synthetic-model"),
        tokenizer_sha256=a.e.sha256(b"synthetic-tokenizer"),
        chat_template_sha256=a.e.sha256(tokenizer.get_chat_template().encode()),
        vocab_size=model.config.vocab_size,
        hidden_width=model.config.hidden_size,
        model_layers=model.config.num_hidden_layers,
        context_limit=model.config.max_position_embeddings,
        eos_token_ids=[tokenizer.eos_token_id],
        protocol_sha256=a.e.sha256(b"synthetic-protocol"),
        tests_sha256=a.e.sha256(Path(__file__).read_bytes()),
        runtime_binding_sha256=a.e.object_hash(runtime),
    )
    return model, tokenizer, plan, a.prepare_inputs(plan, tokenizer), runtime


def run(path, model=None, tokenizer=None):
    model, tokenizer, plan, prepared, runtime = setup(model, tokenizer)
    result = a.run_qualification(
        model, tokenizer, plan=plan, prepared=prepared, runtime_binding=runtime, directory=path
    )
    return result, model, plan, prepared


def read(path):
    return a._read(path)


def rewrite(path, value):
    path.write_bytes(a.e.canonical(value))


def clean(model):
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def test_fixed_plan_and_native_payload_snapshot():
    _, tokenizer, plan, prepared, _ = setup()
    assert plan["max_entries"] == 40 and plan["max_prompt_tokens"] == 128
    assert plan["generation_policy"]["cap"] == 16
    assert [(s["fixture_index"], s["landmark"]) for s in plan["slots"]] == [
        (0, 0),
        (0, 8),
        (1, 0),
        (1, 8),
    ]
    assert prepared["special_token_ids"] == [63]
    assert [o["prompt_token_ids"] for o in prepared["observations"]] == [[1, 2, 4], [1, 3, 4]]
    assert all(o["payload_token_ids"] == [[11, 12], [11, 12]] for o in prepared["observations"])
    snapshot = a.validate_plan(plan)
    plan["fixtures"][0]["messages"][0]["content"] = "changed"
    assert snapshot["fixtures"][0]["messages"][0]["content"] == a.SYSTEM
    assert tokenizer.all_special_ids == [63]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_entries", 41),
        ("max_prompt_tokens", 129),
        ("context_limit", True),
        ("slots", []),
        ("fixtures", []),
    ],
)
def test_fixed_plan_rejects_drift(field, value):
    _, _, plan, _, _ = setup()
    plan[field] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        a.validate_plan(plan)


@pytest.mark.parametrize("bad", ["long", "special", "payload_bool", "closed_bool", "reencode_bool"])
def test_native_boundary_rejects_bad_ids(bad):
    _, tok, plan, _, _ = setup()
    tok.bad = bad
    with pytest.raises(ValueError):
        a.prepare_inputs(plan, tok)


@pytest.mark.parametrize("special", [[True], [63, 63], [64], [], [12]])
def test_invalid_special_registry(special):
    _, tok, plan, _, _ = setup()
    tok.all_special_ids = special
    with pytest.raises(ValueError):
        a.prepare_inputs(plan, tok)


def test_full_cap_40_exact_prefix_schedule_and_replay(tmp_path):
    result, model, plan, _ = run(tmp_path / "run")
    assert result["summary"]["entries"] == {"generation": 32, "capture": 4, "reference": 4}
    assert result["summary"]["total_entries"] == 40 and result["summary"]["qualified"] is True
    assert result["summary"]["t0_exact"] == result["summary"]["t8_exact"] == 2
    expected = []
    for middle in (2, 3):
        prompt = [1, middle, 4]
        expected += [prompt + [0] * t for t in range(16)]
        expected += [prompt, prompt, prompt + [0] * 8, prompt + [0] * 8]
    assert [x["input_ids"].tolist()[0] for x in model.entries] == expected
    for entry in model.entries:
        n = entry["input_ids"].shape[1]
        assert entry["use_cache"] is False and entry["logits_to_keep"] == 1
        assert entry["past_key_values"] is None
        assert torch.equal(entry["attention_mask"], torch.ones((1, n), dtype=torch.int64))
        assert torch.equal(entry["position_ids"], torch.arange(n).reshape(1, n))
        assert entry.get("output_hidden_states", False) is False
    clean(model)
    assert a.load_qualification(tmp_path / "run", plan) == result
    assert len(model.entries) == 40


@pytest.mark.parametrize(
    "after,entries,available,qualified",
    [
        ((0, 0), 6, 0, None),
        ((7, 7), 20, 0, None),
        ((8, 8), 26, 2, True),
        ((0, 8), 16, 1, True),
        ((15, 15), 40, 2, True),
    ],
)
def test_eos_step_counts_and_non_eos_landmark_boundary(
    tmp_path, after, entries, available, qualified
):
    result, model, _, _ = run(tmp_path / "run", Model(after))
    assert result["summary"]["total_entries"] == len(model.entries) == entries
    assert result["summary"]["t8_exact"] == available
    assert result["summary"]["qualified"] is qualified
    for i in range(2):
        generation = read(tmp_path / "run" / f"generation_{i}" / "result.json")
        assert generation["stop_reason"] == "eos"
        assert generation["token_ids"] == [0] * after[i] + [63]
    clean(model)


def test_comparison_false_continues_entire_schedule(tmp_path):
    model = Model()
    model.mode = "mismatch"
    result, model, _, _ = run(tmp_path / "run", model)
    assert len(model.entries) == 40
    assert all(r["capture_equal"] is True and r["logits_equal"] is False for r in result["records"])
    assert result["summary"]["qualified"] is False


@pytest.mark.parametrize("mode", ["nonfinite", "dtype", "shape", "mutate_input"])
def test_returned_but_invalid_forward_is_failure_not_completed(tmp_path, mode):
    model = Model()
    model.mode = mode
    with pytest.raises(ValueError):
        run(tmp_path / "run", model)
    failure = read(tmp_path / "run" / "generation_0" / "step_00" / "failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is True
    assert failure["capture_usable"] is False
    outer = read(tmp_path / "run" / "failure.json")
    assert outer["summary"]["total_entries"] == 1
    assert outer["summary"]["qualified"] is False
    assert [x["status"] for x in outer["records"]] == ["unattempted"] * 4
    clean(model)


@pytest.mark.parametrize("error", [RuntimeError("original"), KeyboardInterrupt(), SystemExit(13)])
def test_original_forward_exception_and_cleanup(tmp_path, error):
    model = Model()
    model.error = error
    with pytest.raises(type(error)) as caught:
        run(tmp_path / "run", model)
    assert caught.value is error
    failure = read(tmp_path / "run" / "generation_0" / "step_00" / "failure.json")
    assert failure["model_call_returned"] is False and failure["entries"] == 1
    clean(model)


def test_model_preflight_fails_before_claim(tmp_path):
    model = Model().train()
    with pytest.raises(ValueError):
        run(tmp_path / "run", model)
    assert not (tmp_path / "run").exists() and not model.entries


def test_runtime_binding_rejected_before_claim(tmp_path):
    model, tok, plan, prepared, runtime = setup()
    runtime["changed"] = True
    with pytest.raises(ValueError):
        a.run_qualification(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists() and not model.entries


def test_consumed_directory_is_not_retried(tmp_path):
    result, model, plan, prepared = run(tmp_path / "run", Model((0, 0)))
    assert result["summary"]["total_entries"] == 6
    with pytest.raises((ValueError, FileExistsError)):
        a.run_qualification(
            model,
            Tokenizer(),
            plan=plan,
            prepared=prepared,
            runtime_binding={"declared_runtime": "invented-only"},
            directory=tmp_path / "run",
        )
    assert len(model.entries) == 6


@pytest.mark.parametrize(
    "site",
    [
        "run_pid",
        "call_index",
        "entry_index",
        "returned",
        "terminal_count",
        "extra_file",
        "generation_step_hash",
        "adapter_payload",
        "reference_payload",
    ],
)
def test_whole_root_replay_rejects_tampering(tmp_path, site):
    _, _, plan, _ = run(tmp_path / "run", Model((0, 0)))
    root = tmp_path / "run"
    if site == "extra_file":
        (root / "unclaimed").write_bytes(b"extra")
    elif site in ("adapter_payload", "reference_payload"):
        path = (
            root
            / "fixture_0_t0"
            / ("adapter-logits.fp32" if site == "adapter_payload" else "reference/logits.fp32")
        )
        path.write_bytes(struct.pack("<64f", *([0.125] * 64)))
    else:
        names = {
            "run_pid": ("run.json", "pid"),
            "call_index": ("generation_0/step_00/attempt.json", "index"),
            "entry_index": ("generation_0/step_00/entry.json", "entry_index"),
            "returned": ("generation_0/step_00/result.json", "model_call_returned"),
            "terminal_count": ("terminal.json", None),
            "generation_step_hash": ("generation_0/dispatch.json", None),
        }
        name, key = names[site]
        path = root / name
        value = read(path)
        if site == "terminal_count":
            value["summary"]["total_entries"] = True
        elif site == "generation_step_hash":
            value["step_result_sha256"][0] = "0" * 64
        else:
            value[key] = 1 if site == "returned" else False
        rewrite(path, value)
    with pytest.raises(ValueError):
        a.load_qualification(root, plan)


def test_snapshot_prepared_not_changed_by_caller_mutation(tmp_path):
    model, tok, plan, prepared, runtime = setup(Model((0, 0)))
    model.callback = lambda: prepared["observations"][0]["prompt_token_ids"].append(12)
    result = a.run_qualification(
        model,
        tok,
        plan=plan,
        prepared=prepared,
        runtime_binding=runtime,
        directory=tmp_path / "run",
    )
    assert result["summary"]["total_entries"] == 6
    assert read(tmp_path / "run" / "prepared.json")["observations"][0]["prompt_token_ids"] == [
        1,
        2,
        4,
    ]


def test_failure_after_capture_without_readable_failure_retains_unknown(tmp_path, monkeypatch):
    original = a.capture.capture_prefix
    error = RuntimeError("after completed adapter")

    def bad(*args, **kwargs):
        original(*args, **kwargs)
        raise error

    monkeypatch.setattr(a.capture, "capture_prefix", bad)
    with pytest.raises(RuntimeError) as caught:
        run(tmp_path / "run", Model((0, 0)))
    assert caught.value is error
    failure = read(tmp_path / "run" / "failure.json")
    assert failure["summary"]["entries"]["capture"] is None
    assert failure["summary"]["entry_lower_bounds"]["capture"] == 1
    assert failure["summary"]["total_entries"] is None
    assert failure["summary"]["entries_complete"] is False
    assert [r["status"] for r in failure["records"]] == [
        "failed",
        "unattempted",
        "unattempted",
        "unattempted",
    ]
    assert not (tmp_path / "run" / "fixture_0_t0" / "caller-return.json").exists()


def test_final_publication_failure_consumes_root(tmp_path, monkeypatch):
    original = a._write
    error = OSError("publication")

    def failing(path, value):
        original(path, value)
        if path.name == "terminal.json":
            raise error

    monkeypatch.setattr(a, "_write", failing)
    model, tok, plan, prepared, runtime = setup(Model((0, 0)))
    with pytest.raises(OSError) as caught:
        a.run_qualification(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    assert (tmp_path / "run" / "terminal.json").exists()
    assert (tmp_path / "run" / "failure.json").exists()
    with pytest.raises(ValueError):
        a.load_qualification(tmp_path / "run", plan)


@pytest.mark.parametrize(
    "values,want", [([0.0, -0.0, 0.0], 0), ([-4.0, -1.0, -1.0], 1), ([1.0, 2.0, 3.0], 2)]
)
def test_first_maximum_from_exact_saved_fp32(values, want):
    assert a._argmax(struct.pack("<" + "f" * len(values), *values), len(values)) == want


@pytest.mark.parametrize("bits", [0x7FC00000, 0x7F800000, 0xFF800000])
def test_nonfinite_raw_logits_rejected(bits):
    with pytest.raises(ValueError):
        a._argmax(struct.pack("<I", bits), 1)


@pytest.mark.parametrize("eos", [0, 63])
def test_tiny_untrained_llama_actual_norm_reference(tmp_path, eos):
    torch.manual_seed(29)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=eos,
        pad_token_id=eos,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(dtype=torch.float32).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    tok = Tokenizer()
    tok.eos_token_id = eos
    tok.all_special_ids = [eos]
    result, _, plan, _ = run(tmp_path / "run", model, tok)
    assert result["summary"]["total_entries"] == (6 if eos == 0 else 40)
    assert result["summary"]["qualified"] is (None if eos == 0 else True)
    assert all(
        r["capture_equal"] is True and r["logits_equal"] is True
        for r in result["records"]
        if r["status"] == "completed"
    )
    slot = tmp_path / "run" / "fixture_0_t0"
    bundle = a.capture.load_capture(slot / "adapter", plan["manifest"]).bundle
    assert bundle.blobs["capture.fp32"] == (slot / "reference" / "norm_input.fp32").read_bytes()
    assert (slot / "reference" / "norm_input.fp32").read_bytes() != (
        slot / "reference" / "norm_output.fp32"
    ).read_bytes()
    clean(model)
