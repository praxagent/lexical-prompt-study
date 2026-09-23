"""A184 synthetic apparatus checks; all real model forwards use random tiny weights."""

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import landmark_evidence as e
from lexical_prompt_study import prefix_only_capture as a


def digest(value):
    return e.sha256(value.encode())


@pytest.fixture(autouse=True)
def four_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def arguments(model=None, *, t=2, generated=None, status="completed", stop="eos"):
    vocab, width, layers = (
        (32, 8, 2)
        if model is None
        else (model.config.vocab_size, model.config.hidden_size, model.config.num_hidden_layers)
    )
    generated = [4, 5, 6, vocab - 1] if generated is None else generated
    comparator = {
        "model_sha256": digest("learner"),
        "transform_sha256": digest("transform"),
        "train_manifest_sha256": digest("train"),
        "calibration_manifest_sha256": digest("cal"),
        "fit_group_ids": [],
        "calibration_group_ids": [],
        "fit_budget": 0,
        "calibration_budget": 0,
    }
    manifest = {
        "schema_version": e.SCHEMA,
        "study_id": "invented-adapter",
        "provenance": {
            "model_sha256": digest("declared-model"),
            "tokenizer_sha256": digest("declared-tokenizer"),
            "chat_template_sha256": digest("declared-template"),
            "capture_code_sha256": a.source_sha256(),
        },
        "vocab_size": vocab,
        "eos_token_ids": [vocab - 1],
        "generation": {"policy_sha256": digest("sample-policy"), "seed": 1, "max_new_tokens": 4},
        "landmarks": [0, 2, 4],
        "count_convention": e.COUNT_CONVENTION,
        "decoder": dict(e.DECODER),
        "capture": {
            "layer_index": layers - 1,
            "model_layers": layers,
            "hidden_width": width,
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": {"group": "evaluation"},
        "observations": [
            {
                "observation_id": "observation",
                "core_id": "core",
                "group_id": "group",
                "split": "evaluation",
                "strata": {"fixture": "invented"},
            }
        ],
        "endpoint": {
            "instrument_sha256": digest("instrument"),
            "target_sha256": digest("target"),
            "horizon_tokens": 4,
            "uncertainty_policy_sha256": digest("uncertainty"),
        },
        "comparators": {name: copy.deepcopy(comparator) for name in e.COMPARATORS},
    }
    observation = {
        "observation_id": "observation",
        "messages": [{"role": "user", "content": "  Synthetic\n"}],
        "rendered_prompt_utf8": b"[user]  Synthetic\n[assistant]",
        "prompt_token_ids": [1, 3],
    }
    generation = {
        "status": status,
        "stop_reason": stop,
        "token_ids": generated,
        "observed_tokens": len(generated),
        "attempt_sha256": digest("sampling-attempt"),
        "dispatch_sha256": digest("sampling-dispatch"),
    }
    if status == "missing":
        generation.update(attempt_sha256=None, dispatch_sha256=None)
    eos = bool(generated and generated[-1] == vocab - 1)
    content = generated[:-1] if eos else generated
    available = len(content) >= t
    prefix = {
        "landmark": t,
        "status": "available" if available else "unreached" if status == "completed" else "missing",
        "reason": None
        if available
        else "early_eos"
        if status == "completed"
        else "generation_" + status,
        "token_ids": content[:t] if available else None,
        "decoded_utf8": (b"" if t == 0 else "é\ufffd ".encode()) if available else None,
    }
    outcome = {
        "status": "known",
        "applicable": True,
        "label": 0,
        "reason": None,
        "disagreement": False,
        "endpoint": copy.deepcopy(manifest["endpoint"]),
        "generation_sha256": e.object_hash(generation),
        "terminal_eos": eos,
        "capped": stop == "cap",
        "censored": not eos and len(content) < 4,
    }
    return dict(
        manifest=manifest,
        observation=observation,
        generation=generation,
        prefix=prefix,
        outcome=outcome,
    )


class AnalyticalBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width))
        self.mode = None
        self.output = None

    def forward(self, hidden):
        value = hidden.cumsum(dim=1) @ self.weight
        if self.mode == "tuple":
            return (value,)
        if self.mode == "dtype":
            return value.double()
        if self.mode == "batch":
            return value.expand(2, -1, -1)
        if self.mode == "length":
            return value[:, :-1]
        if self.mode == "width":
            return value[:, :, :-1]
        if self.mode == "nonfinite":
            value = value.clone()
            value[:, -1, 0] = float("nan")
        self.output = value
        return value


class AnalyticalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            max_position_embeddings=16,
            _attn_implementation="analytical",
        )
        self.embed = torch.nn.Embedding(32, 8)
        self.core = torch.nn.Module()
        self.core.layers = torch.nn.ModuleList([AnalyticalBlock(8), AnalyticalBlock(8)])
        self.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(32 * 8).reshape(32, 8) / 100)
            self.head.weight.copy_(torch.arange(32 * 8).reshape(32, 8) / 200)
        self.failure = None
        self.double_block = False
        self.skip_block = False
        self.callback = None
        self.body_entries = 0
        self.actual_arguments = None
        self.output = None
        self.eval()

    def forward(self, **kwargs):
        self.body_entries += 1
        self.actual_arguments = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        assert torch.is_inference_mode_enabled()
        if self.callback:
            self.callback()
        value = self.embed(kwargs["input_ids"])
        for index, layer in enumerate(self.core.layers):
            if self.skip_block and index == 1:
                continue
            value = layer(value)
            if self.double_block and index == 1:
                value = layer(value)
        if self.failure is not None:
            raise self.failure
        self.output = SimpleNamespace(logits=self.head(self.norm(value[:, -1:, :])))
        return self.output


def run(model, path, **overrides):
    args = arguments(model, **overrides)
    return a.capture_prefix(model, blocks_path="core.layers", directory=path, **args), args


def raw_tensor(value):
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def assert_clean(model):
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


@pytest.mark.parametrize("t", [0, 2])
def test_analytical_boundary_noop_direct_state_and_replay(tmp_path, t):
    model = AnalyticalModel()
    args = arguments(model, t=t)
    ids = [1, 3] + args["prefix"]["token_ids"]
    with torch.inference_mode():
        hidden = model.embed(torch.tensor([ids]))
        hidden = hidden.cumsum(1) @ model.core.layers[0].weight
        hidden = hidden.cumsum(1) @ model.core.layers[1].weight
        reference = model.head(model.norm(hidden[:, -1:, :]))
    result = a.capture_prefix(
        model, blocks_path="core.layers", directory=tmp_path / "capture", **args
    )
    assert result.forward_output is model.output
    assert torch.equal(result.forward_output.logits, reference)
    assert result.bundle.blobs["capture.fp32"] == raw_tensor(hidden[:, -1])
    assert result.receipt["dispatches"] == result.receipt["hook_invocations"] == 1
    assert model.actual_arguments["input_ids"].tolist() == [ids]
    assert model.actual_arguments["attention_mask"].tolist() == [[1] * len(ids)]
    assert model.actual_arguments["position_ids"].tolist() == [list(range(len(ids)))]
    assert model.actual_arguments["use_cache"] is False
    assert model.actual_arguments["past_key_values"] is None
    assert model.actual_arguments["logits_to_keep"] == 1
    saved = a.load_capture(tmp_path / "capture", args["manifest"])
    assert saved.bundle == result.bundle and saved.forward_output is None
    assert_clean(model)
    with pytest.raises(FileExistsError):
        a.capture_prefix(model, blocks_path="core.layers", directory=tmp_path / "capture", **args)
    assert model.body_entries == 1


def test_owned_caller_snapshots_and_returned_capture_bytes(tmp_path):
    model = AnalyticalModel()
    args = arguments(model)
    original_manifest = copy.deepcopy(args["manifest"])

    def mutate_caller():
        args["manifest"]["provenance"]["model_sha256"] = digest("changed")
        args["observation"]["prompt_token_ids"][0] = 17
        args["generation"]["token_ids"][0] = 18
        args["prefix"]["token_ids"][0] = 19
        args["outcome"]["label"] = 1

    model.callback = mutate_caller
    result = a.capture_prefix(
        model, blocks_path="core.layers", directory=tmp_path / "capture", **args
    )
    raw = result.bundle.blobs["capture.fp32"]
    with torch.inference_mode():
        model.core.layers[-1].output.zero_()
        result.forward_output.logits.zero_()
    assert result.bundle.blobs["capture.fp32"] == raw
    assert result.bundle.metadata["observation"]["prompt_token_ids"] == [1, 3]
    assert result.bundle.metadata["outcome"]["label"] == 0
    assert a.load_capture(tmp_path / "capture", original_manifest).bundle == result.bundle


@pytest.mark.parametrize(
    "options,available",
    [
        ({"t": 0, "generated": [], "status": "missing", "stop": "missing"}, True),
        ({"generated": [4, 31]}, False),
        ({"generated": [4, 5, 31]}, True),
        ({"generated": [4, 5, 6, 7], "stop": "cap", "t": 4}, True),
        ({"generated": [4, 5], "status": "interrupted", "stop": "interrupted"}, True),
        ({"generated": [4], "status": "interrupted", "stop": "interrupted"}, False),
        (
            {"generated": [], "status": "infrastructure_failed", "stop": "infrastructure_failed"},
            False,
        ),
    ],
)
def test_availability_no_generation_or_future_eos_in_input(tmp_path, options, available):
    model = AnalyticalModel()
    result, args = run(model, tmp_path / "capture", **options)
    assert result.receipt["status"] == ("completed" if available else "unavailable")
    assert model.body_entries == int(available)
    assert result.receipt["dispatches"] == int(available)
    if available:
        assert 31 not in model.actual_arguments["input_ids"].tolist()[0]
    else:
        assert result.forward_output is None
        assert not (tmp_path / "capture" / "dispatch.json").exists()
    assert a.load_capture(tmp_path / "capture", args["manifest"]).bundle == result.bundle


@pytest.mark.parametrize(
    "mutation",
    [
        lambda args: args["prefix"]["token_ids"].append(6),
        lambda args: args["prefix"]["token_ids"].__setitem__(0, 31),
        lambda args: args["observation"]["prompt_token_ids"].__setitem__(0, True),
        lambda args: args["observation"]["prompt_token_ids"].__setitem__(0, 32),
        lambda args: args["manifest"]["provenance"].update(capture_code_sha256=digest("unbound")),
        lambda args: args["manifest"]["capture"].update(cache_policy="cached"),
        lambda args: args["manifest"]["capture"].update(layer_index=2),
    ],
)
def test_bad_declared_inputs_fail_before_attempt_and_dispatch(tmp_path, mutation):
    model, args = AnalyticalModel(), arguments()
    mutation(args)
    with pytest.raises(ValueError):
        a.capture_prefix(model, blocks_path="core.layers", directory=tmp_path / "invalid", **args)
    assert not (tmp_path / "invalid").exists()
    assert model.body_entries == 0


@pytest.mark.parametrize(
    "kind",
    [
        "training",
        "dtype",
        "context",
        "vocab",
        "width",
        "layers",
        "local_pre",
        "local_post",
        "global",
        "threads",
    ],
)
def test_model_preflight_rejections_have_no_attempt(tmp_path, kind):
    model, args = AnalyticalModel(), arguments()
    handles = []
    if kind == "training":
        model.train()
    elif kind == "dtype":
        model.double()
    elif kind == "context":
        model.config.max_position_embeddings = 3
    elif kind == "vocab":
        model.config.vocab_size = 33
    elif kind == "width":
        model.config.hidden_size = 9
    elif kind == "layers":
        model.config.num_hidden_layers = 1
    elif kind == "local_pre":
        handles.append(model.core.layers[0].register_forward_pre_hook(lambda *_: None))
    elif kind == "local_post":
        handles.append(model.core.layers[0].register_forward_hook(lambda *_: None))
    elif kind == "global":
        handles.append(torch.nn.modules.module.register_module_forward_hook(lambda *_: None))
    elif kind == "threads":
        torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError):
            a.capture_prefix(
                model, blocks_path="core.layers", directory=tmp_path / "invalid", **args
            )
        assert not (tmp_path / "invalid").exists()
        assert model.body_entries == 0
    finally:
        for handle in handles:
            handle.remove()


def test_active_cpu_autocast_is_rejected_before_dispatch(tmp_path):
    model, args = AnalyticalModel(), arguments()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="precision_threads"):
            a.capture_prefix(
                model, blocks_path="core.layers", directory=tmp_path / "invalid", **args
            )
    assert model.body_entries == 0 and not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "kind", ["tuple", "dtype", "batch", "length", "width", "nonfinite", "repeat", "skip"]
)
def test_bad_actual_block_output_consumes_attempt_without_capture(tmp_path, kind):
    model = AnalyticalModel()
    if kind == "repeat":
        model.double_block = True
    elif kind == "skip":
        model.skip_block = True
    else:
        model.core.layers[-1].mode = kind
    with pytest.raises(ValueError):
        run(model, tmp_path / "failed")
    failure = json.loads((tmp_path / "failed" / "failure.json").read_bytes())
    assert failure["dispatches"] == 1 and failure["capture_usable"] is False
    assert not (tmp_path / "failed" / "evidence").exists()
    assert_clean(model)
    with pytest.raises(FileNotFoundError):
        a.load_capture(tmp_path / "failed", arguments()["manifest"])


@pytest.mark.parametrize(
    "exc", [RuntimeError("synthetic private detail"), KeyboardInterrupt(), SystemExit(9)]
)
def test_failure_after_capture_preserves_original_exception_and_no_capture(tmp_path, exc):
    model = AnalyticalModel()
    model.failure = exc
    with pytest.raises(type(exc)) as caught:
        run(model, tmp_path / "failed")
    assert caught.value is exc
    raw = (tmp_path / "failed" / "failure.json").read_bytes()
    assert b"synthetic private detail" not in raw
    assert json.loads(raw)["hook_invocations"] == 1
    assert not (tmp_path / "failed" / "evidence").exists()
    assert_clean(model)


def test_cleanup_failure_preserves_original_and_removes_owned_ids(tmp_path, monkeypatch):
    model = AnalyticalModel()
    original = KeyboardInterrupt()
    model.failure = original
    register = model.core.layers[-1].register_forward_hook

    def broken_remove(*args, **kwargs):
        handle = register(*args, **kwargs)
        handle.remove = lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed"))
        return handle

    monkeypatch.setattr(model.core.layers[-1], "register_forward_hook", broken_remove)
    with pytest.raises(KeyboardInterrupt) as caught:
        run(model, tmp_path / "failed")
    assert caught.value is original
    failure = json.loads((tmp_path / "failed" / "failure.json").read_bytes())
    assert failure["cleanup_failure_type"] == "RuntimeError"
    assert_clean(model)


@pytest.mark.parametrize("filename", ["dispatch.json", "result.json"])
def test_failed_publication_is_consumed_and_nested_success_is_not_adapter_success(
    tmp_path, monkeypatch, filename
):
    model = AnalyticalModel()
    original = e._publish

    def interrupt(path, raw):
        if path.name == filename:
            raise KeyboardInterrupt()
        original(path, raw)

    monkeypatch.setattr(e, "_publish", interrupt)
    args = arguments()
    with pytest.raises(KeyboardInterrupt):
        a.capture_prefix(model, blocks_path="core.layers", directory=tmp_path / "orphan", **args)
    assert model.body_entries == int(filename == "result.json")
    assert_clean(model)
    with pytest.raises(FileNotFoundError):
        a.load_capture(tmp_path / "orphan", args["manifest"])
    monkeypatch.setattr(e, "_publish", original)
    with pytest.raises(FileExistsError):
        a.capture_prefix(model, blocks_path="core.layers", directory=tmp_path / "orphan", **args)
    if filename == "result.json":
        assert (
            e.load_landmark(tmp_path / "orphan" / "evidence", args["manifest"]).metadata["capture"][
                "status"
            ]
            == "completed"
        )


def test_exception_after_terminal_publication_marks_result_unusable(tmp_path, monkeypatch):
    model, args = AnalyticalModel(), arguments()
    original = e._publish

    def fail_after_result(path, raw):
        original(path, raw)
        if path.name == "result.json":
            raise KeyboardInterrupt()

    monkeypatch.setattr(e, "_publish", fail_after_result)
    with pytest.raises(KeyboardInterrupt):
        a.capture_prefix(model, blocks_path="core.layers", directory=tmp_path / "uncertain", **args)
    assert (tmp_path / "uncertain" / "result.json").exists()
    assert (tmp_path / "uncertain" / "failure.json").exists()
    with pytest.raises(ValueError, match="adapter_inventory"):
        a.load_capture(tmp_path / "uncertain", args["manifest"])
    assert_clean(model)


@pytest.mark.parametrize(
    "field,value", [("dispatches", True), ("hook_invocations", 0), ("source_dtype", "bfloat16")]
)
def test_terminal_tampering_is_rejected(tmp_path, field, value):
    model = AnalyticalModel()
    _, args = run(model, tmp_path / "capture")
    path = tmp_path / "capture" / "result.json"
    result = json.loads(path.read_bytes())
    result[field] = value
    path.write_bytes(e.canonical(result))
    with pytest.raises(ValueError):
        a.load_capture(tmp_path / "capture", args["manifest"])


def tiny_model():
    torch.manual_seed(71)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0,
        attn_implementation="sdpa",
        use_cache=False,
    )
    return LlamaForCausalLM(config).float().eval()


def tiny_kwargs(ids):
    values = torch.tensor([ids])
    return dict(
        input_ids=values,
        attention_mask=torch.ones_like(values),
        position_ids=torch.arange(len(ids)).unsqueeze(0),
        use_cache=False,
        past_key_values=None,
        logits_to_keep=1,
    )


@pytest.mark.parametrize("t,layer", [(0, 0), (0, 1), (2, 0), (2, 1)])
def test_tiny_untrained_llama_exact_postblock_and_unmodified_logits(tmp_path, t, layer):
    model = tiny_model()
    args = arguments(model, t=t)
    args["manifest"]["capture"]["layer_index"] = layer
    ids = args["observation"]["prompt_token_ids"] + args["prefix"]["token_ids"]
    # Independent pre-hook on the following layer/final norm sees the preceding
    # post-block residual, before final normalization. Removed before adapter.
    next_module = model.model.layers[layer + 1] if layer == 0 else model.model.norm
    reference = []
    normalized = []
    handle = next_module.register_forward_pre_hook(
        lambda _module, values: reference.append(values[0].detach().clone())
    )
    norm_handle = model.model.norm.register_forward_hook(
        lambda _module, _values, result: normalized.append(result.detach().clone())
    )
    try:
        with torch.inference_mode():
            normal = model(**tiny_kwargs(ids))
    finally:
        handle.remove()
        norm_handle.remove()
    result = a.capture_prefix(
        model, blocks_path="model.layers", directory=tmp_path / "capture", **args
    )
    assert torch.equal(result.forward_output.logits, normal.logits)
    assert result.bundle.blobs["capture.fp32"] == raw_tensor(reference[0][:, -1, :])
    if layer == 1:
        assert not torch.equal(reference[0], normalized[0])
        with torch.inference_mode():
            assert torch.equal(model.model.norm(reference[0]), normalized[0])
    assert_clean(model)
    assert a.load_capture(tmp_path / "capture", args["manifest"]).bundle == result.bundle


def test_tiny_untrained_causal_longer_input_controls_and_future_exclusion(tmp_path):
    model = tiny_model()
    args = arguments(model)
    prefix = [1, 3, 4, 5]
    states = []
    for suffix in ([6, 7], [8, 9]):
        captured = []
        handle = model.model.norm.register_forward_pre_hook(
            lambda _m, values: captured.append(values[0].detach().clone())
        )
        try:
            with torch.inference_mode():
                model(**tiny_kwargs(prefix + suffix))
        finally:
            handle.remove()
        states.append(captured[0][:, : len(prefix), :])
    assert torch.equal(states[0], states[1])
    result = a.capture_prefix(
        model, blocks_path="model.layers", directory=tmp_path / "capture", **args
    )
    actual = torch.tensor(struct.unpack("<16f", result.bundle.blobs["capture.fp32"])).reshape(1, 16)
    # Different full sequence geometry can change floating arithmetic slightly.
    assert torch.allclose(actual, states[0][:, -1, :], atol=1e-7, rtol=1e-6)
    assert result.bundle.metadata["capture"]["evidence"]["input_token_ids"] == prefix


def test_pinned_a183_source_is_unchanged():
    assert e.sha256(Path(e.__file__).read_bytes()) == a.EVIDENCE_SHA256
