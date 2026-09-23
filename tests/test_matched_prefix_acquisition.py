"""A186 invented acquisition fixtures; no native/pretrained loading or fitting."""

import copy
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import matched_prefix_acquisition as a


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
        return "invented-a186-native-template"

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
            elif 20 <= value < 132:
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

    groups = sorted({row["group_id"] for row in a.tasks.build_roster() if row["split"] == "fit"})
    comparator = {
        "model_sha256": h("learner"),
        "transform_sha256": h("transform"),
        "train_manifest_sha256": h("fit"),
        "calibration_manifest_sha256": h("none"),
        "fit_group_ids": groups,
        "calibration_group_ids": [],
        "fit_budget": 1,
        "calibration_budget": 0,
    }
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
        comparator_bindings={key: copy.deepcopy(comparator) for key in a.e.COMPARATORS},
    )
    return model, tok, plan, a.prepare_inputs(plan, tok), runtime


def run(path, model=None, tok=None):
    model, tok, plan, prepared, runtime = setup(model, tok)
    result = a.run_acquisition(
        model, tok, plan=plan, prepared=prepared, runtime_binding=runtime, directory=path
    )
    return result, model, plan, prepared


def clean(model):
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def rewrite(path, value):
    path.write_bytes(a.e.canonical(value))


def test_plan_has_exact_roster_grouping_policy_and_defensive_copies():
    _, _, plan, prepared, _ = setup()
    assert len(plan["rows"]) == len(prepared["observations"]) == 112
    assert plan["rows"] == a.tasks.build_roster()
    assert plan["manifest"]["landmarks"] == [8]
    assert plan["max_entries"] == 112 * (64 + 1) == 7280
    assert plan["max_prompt_tokens"] == 256
    assert len(plan["manifest"]["groups"]) == 48
    assert sum(r["split"] == "fit" for r in plan["rows"]) == 64
    assert all(r["format"] != "csv" for r in plan["rows"] if r["split"] == "fit")
    copied = a.validate_plan(plan)
    plan["rows"][0]["messages"][0]["content"] = "changed"
    assert copied["rows"][0]["messages"][0]["content"] == a.tasks.SYSTEM


@pytest.mark.parametrize(
    "field,value",
    [("rows", []), ("max_entries", 7281), ("max_prompt_tokens", 257), ("context_limit", True)],
)
def test_plan_drift_rejected(field, value):
    _, _, plan, _, _ = setup()
    plan[field] = value
    with pytest.raises((ValueError, TypeError)):
        a.validate_plan(plan)


@pytest.mark.parametrize("mode", ["long", "special", "bool"])
def test_native_prompt_constraints(mode):
    _, tok, plan, _, _ = setup()
    tok.mode = mode
    with pytest.raises(ValueError):
        a.prepare_inputs(plan, tok)


def test_complete_112_early_eos_rows_retain_null_features(tmp_path):
    result, model, plan, _ = run(tmp_path / "run")
    assert result["status"] == "finished"
    assert result["summary"]["entries"] == {"generation": 112, "capture": 0}
    assert len(model.entries) == 112
    assert result["summary"]["unavailable_captures"] == result["summary"]["known_labels"] == 112
    assert all(r["status"] == "completed" and r["label"] == 0 for r in result["records"])
    views = a.feature_views(tmp_path / "run", plan)
    assert (
        len(views["semantic_packets"])
        == len(views["internal_packets"])
        == len(views["labels"])
        == 112
    )
    assert all(p["common"] is None for p in views["semantic_packets"])
    assert all(p["residual_fp32le"] is None for p in views["internal_packets"])
    clean(model)


@pytest.mark.parametrize("wrong", [False, True])
def test_one_reached_t8_uses_only_exact_first_eight_without_reference(tmp_path, wrong):
    result, model, plan, prepared = run(tmp_path / "run", Model(reached=True, wrong=wrong))
    assert result["summary"]["entries"] == {"generation": 121, "capture": 1}
    assert len(model.entries) == 122
    first_prompt = prepared["observations"][0]["prompt_token_ids"]
    assert [v["input_ids"].tolist()[0] for v in model.entries[:10]] == [
        first_prompt + [0] * k for k in range(9)
    ] + [first_prompt + [0] * 8 + [254 if wrong else 20]]
    assert model.entries[10]["input_ids"].tolist()[0] == first_prompt + [0] * 8
    assert all(
        v["use_cache"] is False and v["past_key_values"] is None and v["logits_to_keep"] == 1
        for v in model.entries
    )
    for v in model.entries:
        n = v["input_ids"].shape[1]
        assert torch.equal(v["attention_mask"], torch.ones((1, n), dtype=torch.int64))
        assert torch.equal(v["position_ids"], torch.arange(n).reshape(1, n))
    assert result["records"][0]["label"] == int(not wrong)
    views = a.feature_views(tmp_path / "run", plan)
    common = views["semantic_packets"][0]["common"]
    assert set(common) == {
        "rendered_prompt_utf8",
        "prompt_token_ids",
        "prefix_utf8",
        "prefix_token_ids",
        "completed_fields",
        "prefix_known_error",
    }
    assert common["prefix_utf8"] == b"        " and common["prefix_token_ids"] == [0] * 8
    assert common["completed_fields"] == 0 and common["prefix_known_error"] is False
    assert len(views["internal_packets"][0]["residual_fp32le"]) == 32
    assert views["internal_packets"][0]["residual_shape"] == [1, 8]
    clean(model)


def test_feature_hashes_cannot_depend_on_future_outcome():
    base = {
        "schema_version": "a186-common-prefix-v1",
        "observation_id": "fixed",
        "common": {
            "rendered_prompt_utf8": b"prompt",
            "prompt_token_ids": [1, 2],
            "prefix_utf8": b"prefix",
            "prefix_token_ids": [0] * 8,
            "completed_fields": 0,
            "prefix_known_error": False,
        },
    }
    wanted = a.e.object_hash(
        {
            **base,
            "common": {
                **base["common"],
                "rendered_prompt_utf8": {"bytes_sha256": a.e.sha256(b"prompt"), "byte_length": 6},
                "prefix_utf8": {"bytes_sha256": a.e.sha256(b"prefix"), "byte_length": 6},
            },
        }
    )
    assert a._feature_hash(base) == wanted
    assert a._feature_hash(copy.deepcopy(base)) == wanted
    for future in (0, 1, None):
        envelope = {"feature": base, "label": future, "generation_length": 19 if future else 32}
        assert a._feature_hash(envelope["feature"]) == wanted


def test_cap_without_eos_is_known_zero_and_not_an_extra_step(tmp_path):
    result, model, plan, _ = run(tmp_path / "run", Model(cap=True))
    assert result["summary"]["entries"] == {"generation": 175, "capture": 1}
    assert len(model.entries) == 176 and result["records"][0]["label"] == 0
    generation = a._read(tmp_path / "run/rows/000/generation/result.json")
    assert generation["stop_reason"] == "cap" and len(generation["token_ids"]) == 64
    assert a.load_acquisition(tmp_path / "run", plan) == result


@pytest.mark.parametrize("mode", ["nonfinite", "dtype", "shape", "mutate"])
def test_bad_readout_stops_with_all_112_slots_and_truthful_returned_flag(tmp_path, mode):
    model, tok, plan, prepared, runtime = setup()
    model.mode = mode
    with pytest.raises(ValueError):
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    call_failure = a._read(tmp_path / "run/rows/000/generation/step_00/failure.json")
    assert call_failure["model_call_returned"] is True and call_failure["entries"] == 1
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["status"] == "failed" and result["summary"]["total_entries"] == 1
    assert result["summary"]["failed_rows"] == 1 and result["summary"]["unattempted_rows"] == 111
    assert all(r["label"] is None for r in result["records"])
    with pytest.raises(ValueError):
        a.feature_views(tmp_path / "run", plan)
    clean(model)


@pytest.mark.parametrize("error", [RuntimeError("original"), KeyboardInterrupt(), SystemExit(7)])
def test_original_base_exception_cleanup_no_retry(tmp_path, error):
    model, tok, plan, prepared, runtime = setup()
    model.error = error
    with pytest.raises(type(error)) as caught:
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    assert a.load_acquisition(tmp_path / "run", plan)["summary"]["entries"]["generation"] == 1
    clean(model)
    model.error = None
    with pytest.raises((ValueError, FileExistsError)):
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert len(model.entries) == 1


def test_capture_failure_keeps_known_outcome_but_no_internal_or_feature_export(
    tmp_path, monkeypatch
):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    original = a.capture.capture_prefix
    error = RuntimeError("lost caller return")

    def bad(*args, **kwargs):
        original(*args, **kwargs)
        raise error

    monkeypatch.setattr(a.capture, "capture_prefix", bad)
    with pytest.raises(RuntimeError) as caught:
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["records"][0]["label"] == 1
    assert result["records"][0]["capture_status"] == "infrastructure_failed"
    assert result["summary"]["entries"]["capture"] is None
    assert result["summary"]["entry_lower_bounds"]["capture"] == 1
    assert result["summary"]["total_entries"] is None
    with pytest.raises(ValueError):
        a.feature_views(tmp_path / "run", plan)
    clean(model)


@pytest.mark.parametrize("which", ["non_deterministic", "training", "runtime_binding"])
def test_preflight_rejects_before_claim(tmp_path, which):
    model, tok, plan, prepared, runtime = setup()
    if which == "non_deterministic":
        torch.use_deterministic_algorithms(False)
    elif which == "training":
        model.train()
    else:
        runtime["changed"] = True
    with pytest.raises(ValueError):
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists() and not model.entries


@pytest.mark.parametrize(
    "site",
    [
        "bool_step",
        "generation_parent",
        "raw_logits",
        "endpoint",
        "normal_return",
        "extra_file",
        "symlink",
    ],
)
def test_complete_replay_tamper_rejection(tmp_path, site):
    _, _, plan, _ = run(tmp_path / "run")
    root = tmp_path / "run"
    if site == "raw_logits":
        (root / "rows/000/generation/step_00/logits.fp32").write_bytes(
            struct.pack("<256f", *([1.0] * 256))
        )
    elif site == "extra_file":
        (root / "unexpected").write_bytes(b"unclaimed")
    elif site == "symlink":
        path = root / "rows/000/generation/response.utf8"
        path.unlink()
        path.symlink_to(root / "prepared.json")
    else:
        locations = {
            "bool_step": "rows/000/generation/step_00/attempt.json",
            "generation_parent": "rows/000/generation/dispatch.json",
            "endpoint": "rows/000/generation/outcome.json",
            "normal_return": "rows/000/caller-return.json",
        }
        path = root / locations[site]
        value = a._read(path)
        if site == "bool_step":
            value["step_index"] = False
        elif site == "generation_parent":
            value["step_result_sha256"][0] = "0" * 64
        elif site == "endpoint":
            value["label"] = 1
        else:
            value["event"] = "file_presence"
        rewrite(path, value)
    with pytest.raises(ValueError):
        a.load_acquisition(root, plan)


def test_final_publication_failure_cannot_be_promoted(tmp_path, monkeypatch):
    model, tok, plan, prepared, runtime = setup()
    original = a._write
    error = OSError("terminal publication")

    def fail(path, value):
        original(path, value)
        if path.name == "terminal.json":
            raise error

    monkeypatch.setattr(a, "_write", fail)
    with pytest.raises(OSError) as caught:
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["status"] == "failed" and result["summary"]["completed_rows"] == 112
    with pytest.raises(ValueError):
        a.feature_views(tmp_path / "run", plan)


def test_tiny_untrained_llama_complete_112_without_extra_t0(tmp_path):
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=0,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(dtype=torch.float32).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    tok = Tokenizer()
    tok.eos_token_id = 0
    tok.all_special_ids = [0]
    result, _, _, _ = run(tmp_path / "run", model, tok)
    assert result["summary"]["entries"] == {"generation": 112, "capture": 0}
    assert result["summary"]["unavailable_captures"] == 112
    clean(model)


@pytest.mark.parametrize("field,value", [("fit_budget", 2), ("calibration_budget", 1)])
def test_prospective_fit_budgets_fixed(field, value):
    _, _, plan, _, _ = setup()
    plan["manifest"]["comparators"]["text"][field] = value
    with pytest.raises(ValueError):
        a.validate_plan(plan)


@pytest.mark.parametrize("stage", ["caller-return.json", "result.json"])
def test_postcapture_publication_failure_counts_attempted_unusable(tmp_path, monkeypatch, stage):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    original = a._write
    error = OSError("post capture publication")

    def fail(path, value):
        original(path, value)
        if path.parent.name == "000" and path.name == stage:
            raise error

    monkeypatch.setattr(a, "_write", fail)
    with pytest.raises(OSError) as caught:
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    replay = a.load_acquisition(tmp_path / "run", plan)
    assert replay["records"][0]["label"] == 1
    assert replay["records"][0]["capture_status"] == "infrastructure_failed"
    assert replay["summary"]["failed_captures"] == 1
    assert replay["summary"]["entries"]["capture"] == 1
    with pytest.raises(ValueError):
        a.feature_views(tmp_path / "run", plan)


def test_capture_bookkeeping_failure_cannot_mask_original(tmp_path, monkeypatch):
    model, tok, plan, prepared, runtime = setup(Model(reached=True))
    error = SystemExit(29)

    def stop(*args, **kwargs):
        raise error

    def metadata_failure(*args, **kwargs):
        raise ValueError("secondary bookkeeping")

    monkeypatch.setattr(a.capture, "capture_prefix", stop)
    monkeypatch.setattr(a, "_failed_capture_count", metadata_failure)
    with pytest.raises(SystemExit) as caught:
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    assert caught.value is error
    result = a.load_acquisition(tmp_path / "run", plan)
    assert result["summary"]["entries"]["capture"] is None
    assert result["summary"]["entry_lower_bounds"]["capture"] == 0


def test_failed_counter_cannot_drop_retained_entry(tmp_path):
    model, tok, plan, prepared, runtime = setup()
    model.error = RuntimeError("stop")
    with pytest.raises(RuntimeError):
        a.run_acquisition(
            model,
            tok,
            plan=plan,
            prepared=prepared,
            runtime_binding=runtime,
            directory=tmp_path / "run",
        )
    path = tmp_path / "run/failure.json"
    failure = a._read(path)
    failure["summary"] = a._summary(
        failure["records"], {"generation": 0, "capture": 0}, "failed", True
    )
    rewrite(path, failure)
    with pytest.raises(ValueError, match="failed_count_lower_bound"):
        a.load_acquisition(tmp_path / "run", plan)


def test_tiny_untrained_chained_generation_and_exact_t8_capture(tmp_path):
    class LastIdentityTokenizer(Tokenizer):
        def prompt(self, text):
            return [1, 2, self.renderings[text] + 10]

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=255,
        pad_token_id=255,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(dtype=torch.float32).eval()
    # Exact causal token chain through zero attention/MLPs; no weight search.
    with torch.no_grad():
        for layer in model.model.layers:
            for value in (*layer.self_attn.parameters(), *layer.mlp.parameters()):
                value.zero_()
        model.model.embed_tokens.weight.zero_()
        model.model.embed_tokens.weight[:, 8] = 1
        model.model.embed_tokens.weight[10].zero_()
        model.model.embed_tokens.weight[10, 0] = 1
        for token in range(8):
            model.model.embed_tokens.weight[token].zero_()
            model.model.embed_tokens.weight[token, token + 1] = 1
        model.lm_head.weight.zero_()
        for token in range(8):
            model.lm_head.weight[token, token] = 1
        model.lm_head.weight[255, 8] = 1
    result, _, plan, _ = run(tmp_path / "run", model, LastIdentityTokenizer())
    assert result["summary"]["entries"] == {"generation": 120, "capture": 1}
    assert result["summary"]["completed_captures"] == 1
    generation = a._read(tmp_path / "run/rows/000/generation/result.json")
    assert generation["token_ids"] == list(range(8)) + [255]
    bundle = a.capture.load_capture(tmp_path / "run/rows/000/adapter", plan["manifest"]).bundle
    assert bundle.metadata["prefix"]["token_ids"] == list(range(8))
    expected = struct.pack("<16f", *([0.0] * 8 + [1.0] + [0.0] * 7))
    assert bundle.blobs["capture.fp32"] == expected
    clean(model)
