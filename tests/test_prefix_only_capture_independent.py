"""Independent A184 prefix-only apparatus tests on invented/untrained fixtures.

No pretrained weights, tokenizer, encoder, generation or fitting is invoked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import landmark_evidence, prefix_only_capture as adapter


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def object_sha(value):
    return sha(encoded(value))


class ArithmeticBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 2.0


class ArithmeticModel(torch.nn.Module):
    """Known positional residuals and explicit call-input observations."""

    def __init__(self, width=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(width))
        self.layers = torch.nn.ModuleList([ArithmeticBlock()])
        self.config = SimpleNamespace(
            hidden_size=width,
            num_hidden_layers=1,
            vocab_size=64,
            max_position_embeddings=64,
            _attn_implementation="sdpa",
        )
        self.entered = []
        self.failure = None
        self.skip_block = False
        self.repeat_block = False
        self.mutate_input_after_capture = False
        self.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=1,
    ):
        self.entered.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )
        values = (
            input_ids.float().unsqueeze(-1) * 10 + torch.arange(self.config.hidden_size).float()
        )
        hidden = values if self.skip_block else self.layers[0](values)
        if self.repeat_block:
            hidden = self.layers[0](hidden)
        if self.failure is not None:
            raise self.failure
        if self.mutate_input_after_capture:
            input_ids[0, -1] = 21
        return SimpleNamespace(logits=hidden)


class RecordedLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.entered = []

    def forward(self, *args, **kwargs):
        self.entered.append(
            {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
                if name
                in ("input_ids", "attention_mask", "position_ids", "past_key_values", "use_cache")
            }
        )
        return super().forward(*args, **kwargs)


def tiny_model():
    torch.manual_seed(931)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return RecordedLlama(config).float().eval()


def observation_contract(adapter, model, *, t=2, tokens=None, reason="cap", layer=None):
    tokens = [4, 5, 6, 7] if tokens is None else list(tokens)
    layer = model.config.num_hidden_layers - 1 if layer is None else layer
    manifest = {
        "schema_version": "matched-landmark-v1",
        "study_id": "invented-adapter-independent",
        "provenance": {
            "model_sha256": sha(b"invented model declaration"),
            "tokenizer_sha256": sha(b"invented tokenizer declaration"),
            "chat_template_sha256": sha(b"invented template declaration"),
            "capture_code_sha256": sha(Path(adapter.__file__).read_bytes()),
        },
        "vocab_size": 64,
        "eos_token_ids": [63],
        "generation": {"policy_sha256": sha(b"invented policy"), "seed": 0, "max_new_tokens": 4},
        "landmarks": [t],
        "count_convention": "emitted_non_eos_tokens",
        "decoder": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        "capture": {
            "layer_index": layer,
            "model_layers": model.config.num_hidden_layers,
            "hidden_width": model.config.hidden_size,
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": {"invented-core": "evaluation"},
        "observations": [
            {
                "observation_id": "invented-row",
                "core_id": "invented-core",
                "group_id": "invented-core",
                "split": "evaluation",
                "strata": {},
            }
        ],
        "endpoint": {
            "instrument_sha256": sha(b"instrument"),
            "target_sha256": sha(b"target"),
            "horizon_tokens": 4,
            "uncertainty_policy_sha256": sha(b"uncertainty"),
        },
        "comparators": {},
    }
    for name in ("text", "text_internal"):
        manifest["comparators"][name] = {
            "model_sha256": sha(name.encode()),
            "transform_sha256": sha(b"transform"),
            "train_manifest_sha256": sha(b"train"),
            "calibration_manifest_sha256": sha(b"calibration"),
            "fit_group_ids": [],
            "calibration_group_ids": [],
            "fit_budget": 0,
            "calibration_budget": 0,
        }
    observation = {
        "observation_id": "invented-row",
        "messages": [{"role": "user", "content": "invented"}],
        "rendered_prompt_utf8": b"invented rendered prompt\n",
        "prompt_token_ids": [1, 2, 3],
    }
    generation = {
        "status": "completed" if reason in ("cap", "eos") else reason,
        "stop_reason": reason,
        "token_ids": tokens,
        "observed_tokens": len(tokens),
        "attempt_sha256": sha(b"generation attempt"),
        "dispatch_sha256": sha(b"generation dispatch"),
    }
    if reason == "missing":
        generation["attempt_sha256"] = generation["dispatch_sha256"] = None
    eos = bool(tokens and tokens[-1] == 63)
    content = tokens[:-1] if eos else tokens
    reached = len(content) >= t
    prefix = {
        "landmark": t,
        "status": "available" if reached else "unreached" if eos else "missing",
        "reason": None if reached else "early_eos" if eos else "generation_" + reason,
        "token_ids": content[:t] if reached else None,
        "decoded_utf8": (b"" if t == 0 else b"invented observed prefix") if reached else None,
    }
    outcome = {
        "status": "missing",
        "applicable": None,
        "label": None,
        "reason": "not_measured",
        "disagreement": None,
        "endpoint": copy.deepcopy(manifest["endpoint"]),
        "generation_sha256": object_sha(generation),
        "terminal_eos": eos,
        "capped": reason == "cap",
        "censored": not eos and len(content) < 4,
    }
    return {
        "manifest": manifest,
        "observation": observation,
        "generation": generation,
        "prefix": prefix,
        "outcome": outcome,
    }


def execute(tmp_path, model=None, **fixture):
    model = ArithmeticModel() if model is None else model
    declarations = observation_contract(adapter, model, **fixture)
    result = adapter.capture_prefix(
        model, blocks_path="layers", directory=tmp_path / "capture", **declarations
    )
    return model, declarations, result


def no_hooks(model):
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def failure_receipt(path):
    return json.loads((path / "failure.json").read_bytes())


@pytest.mark.parametrize("t", [0, 2])
def test_exact_boundary_and_known_residual_are_retained(tmp_path, t):
    model, declarations, result = execute(tmp_path, t=t)
    values = [1, 2, 3] + [4, 5][:t]
    entered = model.entered
    assert len(entered) == 1
    assert entered[0]["input_ids"].tolist() == [values]
    assert entered[0]["attention_mask"].tolist() == [[1] * len(values)]
    assert entered[0]["position_ids"].tolist() == [list(range(len(values)))]
    assert entered[0]["past_key_values"] is None
    assert entered[0]["use_cache"] is False
    assert entered[0]["logits_to_keep"] == 1
    assert entered[0]["inference_mode"] is True
    expected = struct.pack("<4f", *(values[-1] * 10 + k + 2 for k in range(4)))
    assert result.bundle.blobs["capture.fp32"] == expected
    assert result.receipt["dispatches"] == result.receipt["hook_invocations"] == 1
    capture = result.bundle.metadata["capture"]["evidence"]
    assert capture["input_token_ids"] == values
    assert capture["absolute_position"] == len(values) - 1
    assert capture["policy"]["source_dtype"] == "float32"
    loaded = adapter.load_capture(tmp_path / "capture", declarations["manifest"])
    assert loaded.bundle == result.bundle
    assert loaded.receipt_bytes == result.receipt_bytes
    assert loaded.forward_output is None
    assert (
        landmark_evidence.load_landmark(tmp_path / "capture" / "evidence", declarations["manifest"])
        == result.bundle
    )
    no_hooks(model)


@pytest.mark.parametrize(
    "t,tokens,reason,dispatches",
    [
        (2, [4, 63], "eos", 0),
        (2, [4, 5, 63], "eos", 1),
        (2, [], "missing", 0),
        (0, [], "missing", 1),
        (2, [4], "interrupted", 0),
    ],
)
def test_eos_and_missingness_define_boundary_without_later_survival_selection(
    tmp_path, t, tokens, reason, dispatches
):
    model, declarations, result = execute(tmp_path, t=t, tokens=tokens, reason=reason)
    assert len(model.entered) == dispatches
    assert result.receipt["dispatches"] == dispatches
    assert (tmp_path / "capture" / "dispatch.json").exists() == bool(dispatches)
    assert result.bundle.metadata["capture"]["status"] == (
        "completed" if dispatches else "unavailable"
    )
    assert ("capture.fp32" in result.bundle.blobs) == bool(dispatches)
    assert (
        adapter.load_capture(tmp_path / "capture", declarations["manifest"]).bundle == result.bundle
    )
    no_hooks(model)


def test_suffix_and_outcome_changes_do_not_enter_capture_or_predictor_features(tmp_path):
    model = ArithmeticModel()
    left = observation_contract(adapter, model, tokens=[4, 5, 6, 7])
    right = observation_contract(adapter, model, tokens=[4, 5, 19, 20])
    right["outcome"].update(
        status="known", applicable=True, label=1, reason=None, disagreement=False
    )
    first = adapter.capture_prefix(
        model, blocks_path="layers", directory=tmp_path / "first", **left
    )
    second = adapter.capture_prefix(
        model, blocks_path="layers", directory=tmp_path / "second", **right
    )
    assert [entry["input_ids"].tolist() for entry in model.entered] == [[[1, 2, 3, 4, 5]]] * 2
    assert first.bundle.metadata["generation"] != second.bundle.metadata["generation"]
    assert first.bundle.metadata["outcome"] != second.bundle.metadata["outcome"]
    assert first.bundle.blobs["capture.fp32"] == second.bundle.blobs["capture.fp32"]
    for comparator in ("text", "text_internal"):
        assert landmark_evidence.predictor_view(
            first.bundle, comparator=comparator
        ) == landmark_evidence.predictor_view(second.bundle, comparator=comparator)


@pytest.mark.parametrize(
    "case",
    ["future_prefix", "context_limit", "train_mode", "wrong_dtype", "autocast", "foreign_hook"],
)
def test_preflight_rejections_never_dispatch_or_consume_directory(tmp_path, case):
    model = ArithmeticModel()
    declarations = observation_contract(adapter, model)
    handle = None
    if case == "future_prefix":
        declarations["prefix"]["token_ids"] = [4, 6]
    elif case == "context_limit":
        model.config.max_position_embeddings = 4
    elif case == "train_mode":
        model.layers[0].train()
    elif case == "wrong_dtype":
        model.double()
    elif case == "foreign_hook":
        handle = model.layers[0].register_forward_hook(lambda *args: None)
    try:
        if case == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError):
                adapter.capture_prefix(
                    model, blocks_path="layers", directory=tmp_path / "capture", **declarations
                )
        else:
            with pytest.raises(ValueError):
                adapter.capture_prefix(
                    model, blocks_path="layers", directory=tmp_path / "capture", **declarations
                )
        assert model.entered == []
        assert not (tmp_path / "capture").exists()
        if handle is not None:
            assert handle.id in model.layers[0]._forward_hooks
    finally:
        if handle is not None:
            handle.remove()
    no_hooks(model)


class InvalidBlock(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, values):
        if self.mode == "dtype":
            return values.double()
        if self.mode == "shape":
            return values[:, -1:, :]
        if self.mode == "nonfinite":
            values = values.clone()
            values[:, -1, 0] = float("nan")
            return values
        return (values,)


@pytest.mark.parametrize("mode", ["dtype", "shape", "nonfinite", "tuple", "skip", "repeat"])
def test_invalid_or_nonunique_postblock_capture_is_failed_not_coerced(tmp_path, mode):
    model = ArithmeticModel()
    if mode in ("skip", "repeat"):
        setattr(model, mode + "_block", True)
    else:
        model.layers[0] = InvalidBlock(mode).eval()
    declarations = observation_contract(adapter, model)
    with pytest.raises(ValueError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    failure = failure_receipt(tmp_path / "capture")
    assert failure["dispatches"] == 1
    assert failure["hook_invocations"] == (0 if mode == "skip" else 2 if mode == "repeat" else 1)
    assert failure["capture_usable"] is False and failure["bundle_sha256"] is None
    assert not (tmp_path / "capture" / "evidence").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.load_capture(tmp_path / "capture", declarations["manifest"])
    no_hooks(model)


class LaterForwardError(RuntimeError):
    pass


class CleanupError(RuntimeError):
    pass


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_later_forward_failure_invalidates_earlier_capture_and_preserves_original(
    tmp_path, monkeypatch, cleanup_failure
):
    model = ArithmeticModel()
    original = LaterForwardError("invented error payload must not enter receipts")
    model.failure = original
    if cleanup_failure:
        remove = torch.utils.hooks.RemovableHandle.remove

        def failing_remove(handle):
            remove(handle)
            raise CleanupError("invented cleanup")

        monkeypatch.setattr(torch.utils.hooks.RemovableHandle, "remove", failing_remove)
    declarations = observation_contract(adapter, model)
    with pytest.raises(LaterForwardError) as raised:
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    assert raised.value is original
    failure = failure_receipt(tmp_path / "capture")
    assert failure["dispatches"] == failure["hook_invocations"] == 1
    assert failure["failure_type"] == "LaterForwardError"
    assert failure["cleanup_failure_type"] == ("CleanupError" if cleanup_failure else None)
    assert failure["capture_usable"] is False
    assert b"payload" not in (tmp_path / "capture" / "failure.json").read_bytes()
    assert not (tmp_path / "capture" / "evidence").exists()
    no_hooks(model)


def test_cleanup_failure_alone_prevents_success_and_removes_owned_hooks(tmp_path, monkeypatch):
    model = ArithmeticModel()

    def refusing_remove(handle):
        raise CleanupError("remove refused before removing owned hook")

    monkeypatch.setattr(torch.utils.hooks.RemovableHandle, "remove", refusing_remove)
    declarations = observation_contract(adapter, model)
    with pytest.raises(CleanupError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    failure = failure_receipt(tmp_path / "capture")
    assert failure["dispatches"] == failure["hook_invocations"] == 1
    assert failure["capture_usable"] is False
    assert not (tmp_path / "capture" / "result.json").exists()
    no_hooks(model)


def test_publication_failure_leaves_orphan_bundle_unusable_and_no_retry(tmp_path, monkeypatch):
    model = ArithmeticModel()
    declarations = observation_contract(adapter, model)
    publish = landmark_evidence._publish

    def stop_before_result(path, raw):
        if path.name == "result.json":
            raise OSError("invented result publication failure")
        return publish(path, raw)

    monkeypatch.setattr(landmark_evidence, "_publish", stop_before_result)
    with pytest.raises(OSError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    assert (tmp_path / "capture" / "evidence" / "receipt.json").is_file()
    assert failure_receipt(tmp_path / "capture")["capture_usable"] is False
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.load_capture(tmp_path / "capture", declarations["manifest"])
    with pytest.raises(FileExistsError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    assert len(model.entered) == 1
    no_hooks(model)


def test_claimed_hook_installation_failure_has_zero_dispatch_and_no_retry(tmp_path, monkeypatch):
    model = ArithmeticModel()
    declarations = observation_contract(adapter, model)

    def reject_registration(*args, **kwargs):
        raise RuntimeError("invented hook installation failure")

    monkeypatch.setattr(model, "register_forward_pre_hook", reject_registration)
    with pytest.raises(RuntimeError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    assert (tmp_path / "capture" / "attempt.json").is_file()
    assert not (tmp_path / "capture" / "dispatch.json").exists()
    failure = failure_receipt(tmp_path / "capture")
    assert failure["dispatches"] == failure["hook_invocations"] == 0
    assert failure["capture_usable"] is False
    assert model.entered == []
    with pytest.raises(FileExistsError):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    no_hooks(model)


def test_post_capture_input_mutation_invalidates_forward_binding(tmp_path):
    model = ArithmeticModel()
    model.mutate_input_after_capture = True
    declarations = observation_contract(adapter, model)
    with pytest.raises(ValueError, match="forward_boundary"):
        adapter.capture_prefix(
            model, blocks_path="layers", directory=tmp_path / "capture", **declarations
        )
    failure = failure_receipt(tmp_path / "capture")
    assert failure["dispatches"] == failure["hook_invocations"] == 1
    assert failure["capture_usable"] is False
    assert not (tmp_path / "capture" / "evidence").exists()
    no_hooks(model)


@pytest.mark.parametrize("mutation", ["orphan", "symlink"])
def test_outer_publication_inventory_cannot_hide_orphan_or_link(tmp_path, mutation):
    _, declarations, _ = execute(tmp_path)
    if mutation == "orphan":
        (tmp_path / "capture" / ".result.json.tmp").write_bytes(b"invented partial publication")
    else:
        dispatch = tmp_path / "capture" / "dispatch.json"
        original = tmp_path / "authentic-dispatch.json"
        original.write_bytes(dispatch.read_bytes())
        dispatch.unlink()
        dispatch.symlink_to(original)
    with pytest.raises(ValueError):
        adapter.load_capture(tmp_path / "capture", declarations["manifest"])


@pytest.mark.parametrize(
    "file_name,field,value",
    [
        ("result.json", "dispatches", True),
        ("result.json", "hook_invocations", 0),
        ("result.json", "source_dtype", "float64"),
        ("attempt.json", "input_length", 6),
        ("dispatch.json", "input_length", 6),
    ],
)
def test_loader_rejects_terminal_boundary_or_count_tampering(tmp_path, file_name, field, value):
    _, declarations, _ = execute(tmp_path)
    path = tmp_path / "capture" / file_name
    data = json.loads(path.read_bytes())
    data[field] = value
    path.write_bytes(encoded(data))
    with pytest.raises(ValueError):
        adapter.load_capture(tmp_path / "capture", declarations["manifest"])


def independent_last_block_reference(model, values):
    captured = []
    normalized = []

    def before_final_norm(module, args):
        captured.append(args[0].detach().clone())

    def after_final_norm(module, args, output):
        normalized.append(output.detach().clone())

    handles = [
        model.model.norm.register_forward_pre_hook(before_final_norm),
        model.model.norm.register_forward_hook(after_final_norm),
    ]
    try:
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([values]),
                attention_mask=torch.ones((1, len(values)), dtype=torch.long),
                position_ids=torch.arange(len(values)).unsqueeze(0),
                past_key_values=None,
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == len(normalized) == 1
    return output, captured[0], normalized[0]


@pytest.mark.parametrize("t", [0, 2])
def test_tiny_llama_matches_direct_postblock_not_final_normalization_and_is_noop(tmp_path, t):
    model = tiny_model()
    declarations = observation_contract(adapter, model, t=t)
    values = [1, 2, 3] + [4, 5][:t]
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    reference, postblock, normalized = independent_last_block_reference(model, values)
    assert not torch.equal(postblock, normalized)
    result = adapter.capture_prefix(
        model, blocks_path="model.layers", directory=tmp_path / "capture", **declarations
    )
    raw = result.bundle.blobs["capture.fp32"]
    captured = torch.tensor(struct.unpack("<16f", raw)).reshape(1, 16)
    assert torch.equal(captured, postblock[:, -1, :])
    assert torch.equal(result.forward_output.logits, reference.logits)
    assert model.entered[-1]["input_ids"].tolist() == [values]
    assert model.entered[-1]["past_key_values"] is None
    assert model.entered[-1]["use_cache"] is False
    assert all(torch.equal(state_before[name], value) for name, value in model.state_dict().items())
    no_hooks(model)


def test_tiny_llama_prefix_capture_agrees_with_causal_longer_controls(tmp_path):
    model = tiny_model()
    values = [1, 2, 3, 4, 5]
    _, longer_a, _ = independent_last_block_reference(model, values + [6, 7])
    _, longer_b, _ = independent_last_block_reference(model, values + [19, 20])
    declarations = observation_contract(adapter, model)
    result = adapter.capture_prefix(
        model, blocks_path="model.layers", directory=tmp_path / "capture", **declarations
    )
    captured = torch.tensor(struct.unpack("<16f", result.bundle.blobs["capture.fp32"])).reshape(
        1, 16
    )
    # Fixed numerical tolerance for different physical forward lengths. The
    # adapter itself receives only the five-token common-time prefix.
    torch.testing.assert_close(captured, longer_a[:, 4, :], rtol=0, atol=1e-6)
    torch.testing.assert_close(captured, longer_b[:, 4, :], rtol=0, atol=1e-6)
    assert model.entered[-1]["input_ids"].tolist() == [values]
    no_hooks(model)
