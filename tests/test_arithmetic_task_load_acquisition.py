"""A187 invented generation-only qualification; no pretrained/native assets."""

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import arithmetic_task_load_acquisition as a


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


class Tokenizer:
    eos_token_id = 255
    all_special_ids = [255]

    def __init__(self):
        self.rows = a.tasks.build_roster()
        self.renderings = {self.render(row["messages"]): i for i, row in enumerate(self.rows)}
        self.payloads = {message["content"] for row in self.rows for message in row["messages"]}
        self.mode = None

    def get_chat_template(self):
        return "invented-a187-native-template"

    def render(self, messages):
        return (
            "[system]" + messages[0]["content"] + "[user]" + messages[1]["content"] + "[assistant]"
        )

    def prompt(self, text):
        return [1, self.renderings[text] + 10, 2]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, date_string, return_dict=False
    ):
        assert date_string == "26 Jul 2024" and return_dict is False
        rendered = self.render(messages)
        if not tokenize:
            return rendered
        if add_generation_prompt:
            ids = self.prompt(rendered)
            return ids * 86 if self.mode == "long" else ids
        return self.prompt(rendered) + [3, self.eos_token_id]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in self.payloads:
            return [255] if self.mode == "special" else [4, 5]
        probe = text.endswith("probe")
        ids = self.prompt(text[:-5] if probe else text)
        if self.mode == "bool":
            ids[0] = True
        return ids + [3] if probe else ids

    def decode(self, values, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        if values == [3]:
            return "probe"
        chunks = []
        for value in values:
            if value == 0:
                chunks.append(" ")
            elif value == 254:
                chunks.append("wrong")
            elif 20 <= value < 52:
                chunks.append(",".join(str(n) for n in self.rows[value - 20]["expected_sums"]))
            else:
                chunks.append("?")
        return "".join(chunks)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(8))

    def forward(self, hidden):
        return hidden.cumsum(1) @ self.weight


class Model(torch.nn.Module):
    def __init__(self, *, reached=False, wrong=False, cap=False):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=256,
            hidden_size=8,
            num_hidden_layers=2,
            max_position_embeddings=512,
            _attn_implementation="sdpa",
            output_hidden_states=False,
            output_attentions=False,
        )
        self.embed = torch.nn.Embedding(256, 8)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(), Block()])
        self.model.norm = torch.nn.LayerNorm(8)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(2048).reshape(256, 8) / 100)
        self.reached, self.wrong, self.cap = reached, wrong, cap
        self.entries = []
        self.mode = None
        self.error = None
        self.callback = None
        self.eval()

    def forward(self, **kwargs):
        self.entries.append(
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        )
        if self.callback:
            self.callback()
        hidden = self.embed(kwargs["input_ids"])
        for block in self.model.layers:
            hidden = block(hidden)
        self.model.norm(hidden)
        if self.error:
            raise self.error
        index = int(kwargs["input_ids"][0, 1]) - 10
        step = kwargs["input_ids"].shape[1] - 3
        if index == 0 and self.cap:
            chosen = 0
        elif index == 0 and self.reached:
            chosen = (
                0 if step < 8 else 254 if self.wrong and step == 8 else 20 if step == 8 else 255
            )
        else:
            chosen = 255
        logits = torch.zeros((1, 1, 256), dtype=torch.float32)
        logits[0, 0, chosen] = 1
        if self.mode == "nonfinite":
            logits[0, 0, 7] = float("nan")
        elif self.mode == "dtype":
            logits = logits.double()
        elif self.mode == "shape":
            logits = logits.expand(1, 2, 256)
        elif self.mode == "mutate":
            kwargs["input_ids"][0, 0] += 1
        return SimpleNamespace(logits=logits)


def setup(model=None, tok=None):
    model, tok = Model() if model is None else model, Tokenizer() if tok is None else tok
    runtime = {"synthetic_runtime": True}

    def h(text):
        return a.e.sha256(text.encode())

    plan = a.compile_plan(
        provenance={
            "model_sha256": h("model"),
            "tokenizer_sha256": h("tokenizer"),
            "chat_template_sha256": h(tok.get_chat_template()),
        },
        geometry={
            "vocab_size": model.config.vocab_size,
            "hidden_width": model.config.hidden_size,
            "model_layers": model.config.num_hidden_layers,
            "context_limit": model.config.max_position_embeddings,
        },
        eos_token_ids=[tok.eos_token_id],
        runtime_binding_sha256=a.e.object_hash(runtime),
        protocol_sha256=h("protocol"),
        tests_sha256=a.e.sha256(Path(__file__).read_bytes()),
    )
    return model, tok, plan, a.prepare_inputs(plan, tok), runtime


def run(path, model=None, tok=None):
    model, tok, plan, prepared, runtime = setup(model, tok)
    result = a.run_acquisition(
        model, tok, plan=plan, prepared=prepared, runtime_binding=runtime, directory=path
    )
    return result, model, plan, prepared


def invoke(path, model, tok, plan, prepared, runtime):
    return a.run_acquisition(
        model, tok, plan=plan, prepared=prepared, runtime_binding=runtime, directory=path
    )


def clean(model):
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def rewrite(path, value):
    path.write_bytes(a.e.canonical(value))


def refresh_failure(root):
    failure = a._read(root / "failure.json")
    inventory = a._inventory(root)
    inventory.pop("failure.json")
    failure["artifact_sha256"] = inventory
    rewrite(root / "failure.json", failure)


def test_exact32_plan_without_capture_fit_or_old_namespace():
    _, _, plan, prepared, _ = setup()
    assert len(plan["rows"]) == len(prepared["observations"]) == 32
    assert plan["rows"] == a.tasks.build_roster()
    assert plan["max_entries"] == 32 * 64 == 2048
    assert plan["max_prompt_tokens"] == 256 and plan["diagnostic"]["additional_forwards"] == 0
    assert "manifest" not in plan and "comparators" not in plan
    assert plan["step_receipt_schema"] == a.primitive.SCHEMA
    copy_plan = a.validate_plan(plan)
    plan["geometry"]["vocab_size"] += 1
    assert copy_plan["geometry"]["vocab_size"] == 256


@pytest.mark.parametrize(
    "field,value",
    [("rows", []), ("max_entries", 2049), ("max_prompt_tokens", 257), ("eos_token_ids", [True])],
)
def test_plan_drift(field, value):
    _, _, plan, _, _ = setup()
    plan[field] = value
    with pytest.raises((ValueError, TypeError)):
        a.validate_plan(plan)


@pytest.mark.parametrize("mode", ["long", "special", "bool"])
def test_native_guards(mode):
    _, tok, plan, _, _ = setup()
    tok.mode = mode
    with pytest.raises(ValueError):
        a.prepare_inputs(plan, tok)


def test_full32_early_eos_no_capture_or_extra_forwards(tmp_path):
    result, model, plan, _ = run(tmp_path / "run")
    assert result["status"] == "finished" and result["summary"]["total_entries"] == 32
    assert len(model.entries) == 32 and result["summary"]["known_labels"] == 32
    assert all(r["label"] == 0 and r["prefix_reached"] is False for r in result["records"])
    assert result == a.load_acquisition(tmp_path / "run", plan)
    assert not list((tmp_path / "run").rglob("adapter"))
    clean(model)


@pytest.mark.parametrize("wrong", [False, True])
def test_t8_is_checkpoint_not_capture_and_no_future_leak(tmp_path, wrong):
    result, model, plan, prepared = run(tmp_path / "run", Model(reached=True, wrong=wrong))
    assert result["summary"]["total_entries"] == len(model.entries) == 41
    prompt = prepared["observations"][0]["prompt_token_ids"]
    assert [x["input_ids"].tolist()[0] for x in model.entries[:10]] == [
        prompt + [0] * k for k in range(9)
    ] + [prompt + [0] * 8 + [254 if wrong else 20]]
    assert result["records"][0]["label"] == int(not wrong)
    assert result["records"][0]["prefix_reached"] is True
    assert result["records"][0]["completed_fields"] == 0
    assert result["records"][0]["unflagged_remaining_ge2"] is True
    assert (tmp_path / "run/rows/000/generation/prefix.utf8").read_bytes() == b"        "
    for call in model.entries:
        n = call["input_ids"].shape[1]
        assert call["use_cache"] is False and call["past_key_values"] is None
        assert call["logits_to_keep"] == 1 and call["output_hidden_states"] is False
        assert torch.equal(call["attention_mask"], torch.ones((1, n), dtype=torch.int64))
        assert torch.equal(call["position_ids"], torch.arange(n).reshape(1, n))
    clean(model)


def test_onecap_is64_sampled_not65_and_knownzero(tmp_path):
    result, model, _, _ = run(tmp_path / "run", Model(cap=True))
    assert result["summary"]["total_entries"] == len(model.entries) == 95
    assert result["records"][0]["label"] == 0 and result["records"][0]["capped"] is True


@pytest.mark.parametrize("mode", ["nonfinite", "dtype", "shape", "mutate"])
def test_readout_failure_exact_count_unknown_endpoint_and_allslots(tmp_path, mode):
    model, tok, plan, prepared, runtime = setup()
    model.mode = mode
    with pytest.raises(ValueError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    failure = a._read(tmp_path / "run/rows/000/generation/step_00/failure.json")
    assert failure["model_call_returned"] is True and failure["entries"] == 1
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["summary"]["total_entries"] == 1 and result["summary"]["entries_complete"] is True
    assert [r["status"] for r in result["records"]] == ["failed"] + ["unattempted"] * 31
    assert all(r["label"] is None for r in result["records"])
    clean(model)


@pytest.mark.parametrize("error", [RuntimeError("invented"), KeyboardInterrupt(), SystemExit(7)])
def test_original_exception_hook_cleanup_consumed_root(tmp_path, error):
    model, tok, plan, prepared, runtime = setup()
    model.error = error
    with pytest.raises(type(error)) as caught:
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert caught.value is error
    assert a.load_acquisition(tmp_path / "run", plan)["summary"]["total_entries"] == 1
    clean(model)
    model.error = None
    with pytest.raises((ValueError, FileExistsError)):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert len(model.entries) == 1


def test_interruption_after_t8_keeps_only_available_diagnostic(tmp_path):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    error = KeyboardInterrupt()

    def fail():
        if len(model.entries) == 9:
            raise error

    model.callback = fail
    with pytest.raises(KeyboardInterrupt):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    result = a.load_acquisition(tmp_path / "run", plan)
    first = result["records"][0]
    assert first["label"] is None and first["prefix_reached"] is True
    assert first["completed_fields"] == 0 and first["known_error"] is False
    assert first["unflagged_remaining_ge2"] is True and result["summary"]["total_entries"] == 9


@pytest.mark.parametrize("which", ["non_deterministic", "training", "runtime_binding"])
def test_preflight_failure_does_not_claim(which, tmp_path):
    model, tok, plan, prepared, runtime = setup()
    if which == "non_deterministic":
        torch.use_deterministic_algorithms(False)
    elif which == "training":
        model.train()
    else:
        runtime["changed"] = True
    with pytest.raises(ValueError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert not (tmp_path / "run").exists() and not model.entries


@pytest.mark.parametrize(
    "site",
    [
        "counter",
        "parent_attempt",
        "bool_step",
        "raw_logits",
        "score",
        "normal_return",
        "extra",
        "symlink",
    ],
)
def test_completed_or_partial_tamper_rejected(tmp_path, site):
    result, model, plan, _ = run(tmp_path / "run")
    root = tmp_path / "run"
    if site == "counter":
        val = a._read(root / "terminal.json")
        val["summary"]["generation_entries"] += 1
        rewrite(root / "terminal.json", val)
    elif site == "parent_attempt":
        p = root / "rows/000/generation/attempt.json"
        val = a._read(p)
        val["run_sha256"] = "0" * 64
        rewrite(p, val)
    elif site == "bool_step":
        p = root / "rows/000/generation/step_00/attempt.json"
        val = a._read(p)
        val["step_index"] = False
        rewrite(p, val)
    elif site == "raw_logits":
        (root / "rows/000/generation/step_00/logits.fp32").write_bytes(
            struct.pack("<256f", *([1.0] * 256))
        )
    elif site == "score":
        p = root / "rows/000/generation/score.json"
        val = a._read(p)
        val["label"] = 1
        rewrite(p, val)
    elif site == "normal_return":
        (root / "rows/000/generation-return.json").unlink()
    elif site == "extra":
        (root / "extra").write_bytes(b"x")
    else:
        p = root / "rows/000/generation/response.utf8"
        p.unlink()
        p.symlink_to(root / "prepared.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        a.load_acquisition(root, plan)


def test_failed_count_inflation_rejected_even_with_consistent_summary(tmp_path):
    model, tok, plan, prepared, runtime = setup()
    model.error = RuntimeError("invented")
    with pytest.raises(RuntimeError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    p = tmp_path / "run/failure.json"
    value = a._read(p)
    value["observed_generation_entries"] = 2
    value["summary"].update(generation_entries=2, total_entries=2, entry_lower_bound=2)
    rewrite(p, value)
    with pytest.raises(ValueError):
        a.load_acquisition(tmp_path / "run", plan)


def test_absent_final_failure_receipt_yields_unknown_not_declared_exact(tmp_path, monkeypatch):
    model, tok, plan, prepared, runtime = setup()
    error = RuntimeError("original")
    model.error = error
    original = a.primitive._write

    def fail_receipt(path, value):
        if path.name == "failure.json":
            raise OSError("invented receipt loss")
        return original(path, value)

    monkeypatch.setattr(a.primitive, "_write", fail_receipt)
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert caught.value is error
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["summary"]["generation_entries"] is None
    assert result["summary"]["total_entries"] is None
    assert result["summary"]["entry_lower_bound"] == 1
    assert result["summary"]["entries_complete"] is False


@pytest.mark.parametrize("filename", ["generation-return.json", "result.json", "terminal.json"])
def test_publication_failure_stays_consumed_and_unknown_as_needed(tmp_path, monkeypatch, filename):
    model, tok, plan, prepared, runtime = setup()
    original = a._write
    error = OSError("invented post-publication failure")

    def fail(path, value):
        original(path, value)
        selected = path.name == filename and (
            filename != "result.json" or path.parent.name == "000"
        )
        if selected:
            raise error

    monkeypatch.setattr(a, "_write", fail)
    with pytest.raises(OSError) as caught:
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert caught.value is error
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["status"] == "failed"
    assert result["summary"]["total_entries"] == (32 if filename == "terminal.json" else 1)
    assert result["records"][0]["label"] == (None if filename == "generation-return.json" else 0)


def test_tiny_untrained_llama32_cachefree_body_entries(tmp_path):
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    tok = Tokenizer()
    tok.eos_token_id = 0
    tok.all_special_ids = [0]
    result, _, _, _ = run(tmp_path / "run", model, tok)
    assert result["summary"]["total_entries"] == 32
    assert result["summary"]["success_labels"] == 0
    clean(model)


def test_sampled_eos_before_decode_failure_is_observed_but_endpoint_unknown(tmp_path):
    model, tok, plan, prepared, runtime = setup()
    decode = tok.decode
    original = RuntimeError("invented response decoder failure")

    def broken(values, **kwargs):
        if not values:
            raise original
        return decode(values, **kwargs)

    tok.decode = broken
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    assert caught.value is original
    result = a.load_acquisition(tmp_path / "run", plan)
    row = result["records"][0]
    assert row["label"] is None and row["terminal_eos"] is True
    assert row["prefix_reached"] is False and result["summary"]["total_entries"] == 1


def test_t8_decode_failure_preserves_reached_with_unknown_diagnostic(tmp_path):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    decode = tok.decode

    def broken(values, **kwargs):
        if values == [0] * 8:
            raise RuntimeError("invented prefix decoder failure")
        return decode(values, **kwargs)

    tok.decode = broken
    with pytest.raises(RuntimeError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    result = a.load_acquisition(tmp_path / "run", plan)
    row = result["records"][0]
    assert row["prefix_reached"] is True and row["label"] is None
    assert row["completed_fields"] is None and row["known_error"] is None
    assert result["summary"]["total_entries"] == 8


@pytest.mark.parametrize("filename", ["prefix.utf8", "prefix.json"])
def test_prefix_publication_failure_does_not_promote_orphan_text(tmp_path, monkeypatch, filename):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    publish = a.e._publish

    def broken(path, raw):
        publish(path, raw)
        if path.name == filename:
            raise OSError("invented prefix publication failure")

    monkeypatch.setattr(a.e, "_publish", broken)
    with pytest.raises(OSError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    result = a.load_acquisition(tmp_path / "run", plan)
    row = result["records"][0]
    assert row["prefix_reached"] is True and row["label"] is None
    assert row["completed_fields"] is None and row["known_error"] is None
    assert result["summary"]["total_entries"] == 8


def test_pre_hook_entry_is_distinct_from_model_body_entry(tmp_path, monkeypatch):
    model, tok, plan, prepared, runtime = setup()
    write = a.primitive._write

    def broken(path, value):
        if path.name == "entry.json":
            raise OSError("invented dispatch receipt failure")
        return write(path, value)

    monkeypatch.setattr(a.primitive, "_write", broken)
    with pytest.raises(OSError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["summary"]["total_entries"] == 1
    assert model.entries == []
    failure = a._read(tmp_path / "run/rows/000/generation/step_00/failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is False
    clean(model)


def test_partial_parent_binding_rejected_even_when_failure_inventory_rebound(tmp_path):
    model, tok, plan, prepared, runtime = setup()
    model.error = RuntimeError("invented failure")
    with pytest.raises(RuntimeError):
        invoke(tmp_path / "run", model, tok, plan, prepared, runtime)
    root = tmp_path / "run"
    path = root / "rows/000/generation/attempt.json"
    value = a._read(path)
    value["row_sha256"] = "0" * 64
    rewrite(path, value)
    refresh_failure(root)
    with pytest.raises(ValueError):
        a.load_acquisition(root, plan)
