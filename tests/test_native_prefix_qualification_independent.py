"""Independent A185 tests on invented analytical and untrained fixtures only.

No native tokenizer, model assets, historical evidence or authored test builders.
"""

from __future__ import annotations

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lexical_prompt_study import landmark_evidence as evidence
from lexical_prompt_study import native_prefix_qualification as subject


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class AddBlock(torch.nn.Module):
    def forward(self, value):
        return value + 2.0


class ScaleNorm(torch.nn.Module):
    def forward(self, value):
        return value * 0.25


class GreedyFixture(torch.nn.Module):
    """Known residual and last-row logits, selected by actual input length."""

    def __init__(self, choices=None, *, prompt_length=3, width=4, vocab=64):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(width))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([AddBlock()])
        self.model.norm = ScaleNorm()
        self.config = SimpleNamespace(
            hidden_size=width,
            num_hidden_layers=1,
            vocab_size=vocab,
            max_position_embeddings=256,
            _attn_implementation="sdpa",
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        self.choices = [4] * 16 if choices is None else choices
        self.prompt_length = prompt_length
        self.inputs = []
        self.fail_at_input_length = None
        self.failure = RuntimeError("invented forward failure")
        self.logit_failure = None
        self.body_completed = 0
        self.residual_drift_entries = set()
        self.logit_drift_entries = set()
        self.fail_at_entry = None
        self.mutate_input = False
        self.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=1,
        output_hidden_states=False,
        output_attentions=False,
    ):
        self.inputs.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
                "inference_mode": torch.is_inference_mode_enabled(),
                "output_hidden_states": output_hidden_states,
                "output_attentions": output_attentions,
            }
        )
        hidden = (
            input_ids.float().unsqueeze(-1) * 10 + torch.arange(self.config.hidden_size).float()
        )
        if len(self.inputs) in self.residual_drift_entries:
            hidden = hidden + 1
        for block in self.model.layers:
            hidden = block(hidden)
        normalized = self.model.norm(hidden)
        if (
            input_ids.shape[1] == self.fail_at_input_length
            or len(self.inputs) == self.fail_at_entry
        ):
            raise self.failure
        index = max(0, min(len(self.choices) - 1, input_ids.shape[1] - self.prompt_length))
        choice = self.choices[index]
        logits = torch.full((1, 1, self.config.vocab_size), -2.0)
        for token in choice if isinstance(choice, tuple) else (choice,):
            logits[0, 0, token] = 3.0
        if len(self.inputs) in self.logit_drift_entries:
            logits[0, 0, 11] = -9.0
        if self.logit_failure == "nonfinite":
            logits[0, 0, 11] = float("nan")
        elif self.logit_failure == "dtype":
            logits = logits.double()
        elif self.logit_failure == "shape":
            logits = logits[:, :, :-1]
        self.body_completed += 1
        if self.mutate_input:
            input_ids[0, -1] = 30
        return SimpleNamespace(logits=logits, invented_normalized=normalized)


def no_hooks(model):
    assert all(
        not child._forward_hooks and not child._forward_pre_hooks for child in model.modules()
    )


def tiny_model():
    torch.manual_seed(185)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).float().eval()
    # All logits tie by construction; deterministic first-index argmax is zero,
    # distinct from the invented EOS63. This forces both t8 slots to be reached
    # without searching seeds or inspecting target behavior.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    return model


class InventedTokenizer:
    """An explicit finite fake; never loads or purports to qualify a tokenizer."""

    eos_token_id = 63
    all_special_ids = [1, 63]

    def __init__(self):
        self.rendered_to_ids = {}
        self.prompt_ids_to_rendered = {}
        self.decode_calls = []

    def get_chat_template(self):
        return "invented native-qualification template"

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        base = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        is_second = "integers" in base[-1]["content"]
        ids = [1, 2, 4 if is_second else 3]
        rendered = (
            "".join("<" + message["role"] + ">" + message["content"] for message in base)
            + "<assistant>"
        )
        if messages[-1]["role"] == "assistant":
            assert messages[-1]["content"] == "probe"
            ids += [20, 21, 22, 23, 24, 63]
            rendered += "probe<eos>"
        self.rendered_to_ids[rendered] = ids
        self.prompt_ids_to_rendered[tuple(ids)] = rendered
        return list(ids) if tokenize else rendered

    def encode(self, text, *, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        if text == "probe":
            return [20, 21, 22, 23, 24]
        if text in self.rendered_to_ids:
            return list(self.rendered_to_ids[text])
        if text.endswith("probe") and text[:-5] in self.rendered_to_ids:
            return list(self.rendered_to_ids[text[:-5]]) + [20, 21, 22, 23, 24]
        # Payload-only encoding used for special-token checks; invented and
        # deterministic, deliberately disjoint from BOS/EOS.
        return [2 + byte % 60 for byte in text.encode()]

    def decode(
        self, token_ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False, **kwargs
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        ids = list(token_ids)
        self.decode_calls.append(ids)
        if ids == [20, 21, 22, 23, 24]:
            return "probe"
        if ids == [20, 21, 22, 23, 24, 63]:
            return "probe<eos>"
        if tuple(ids) in self.prompt_ids_to_rendered:
            return self.prompt_ids_to_rendered[tuple(ids)]
        return "".join("<eos>" if token == 63 else "Ω" + str(token) for token in ids)


def fixture_plan(module, model):
    tokenizer = InventedTokenizer()
    runtime_binding = {
        "schema_version": "invented-runtime-binding",
        "scope": "untrained independent apparatus fixture",
    }
    plan = module.compile_plan(
        model_sha256=digest(b"invented model declaration"),
        tokenizer_sha256=digest(b"invented tokenizer declaration"),
        chat_template_sha256=digest(tokenizer.get_chat_template().encode()),
        vocab_size=model.config.vocab_size,
        hidden_width=model.config.hidden_size,
        model_layers=model.config.num_hidden_layers,
        context_limit=model.config.max_position_embeddings,
        eos_token_ids=[63],
        protocol_sha256=digest(b"invented protocol"),
        tests_sha256=digest(b"invented test binding"),
        runtime_binding_sha256=digest(canonical(runtime_binding)),
    )
    prepared = module.prepare_inputs(plan, tokenizer)
    return tokenizer, plan, prepared, runtime_binding


def run_fixture(tmp_path, model=None):
    model = GreedyFixture() if model is None else model
    tokenizer, plan, prepared, runtime_binding = fixture_plan(subject, model)
    result = subject.run_qualification(
        model,
        tokenizer,
        plan=plan,
        prepared=prepared,
        runtime_binding=runtime_binding,
        directory=tmp_path / "run",
    )
    return model, tokenizer, plan, result


def read_json(path):
    return json.loads(path.read_bytes())


def raw_values(path):
    return [item[0] for item in struct.iter_unpack("<f", path.read_bytes())]


def assert_private_endpoint_missing(bundle):
    outcome = bundle.metadata["outcome"]
    assert outcome["status"] == "missing"
    assert outcome["applicable"] is outcome["label"] is outcome["disagreement"] is None
    assert outcome["reason"] == "not_measured"


def test_full_fixed_cohort_has_exact_step_inputs_rows_and_40_entries(tmp_path):
    model, tokenizer, plan, result = run_fixture(tmp_path)
    assert result["summary"]["entries"] == {"generation": 32, "capture": 4, "reference": 4}
    assert result["summary"]["total_entries"] == len(model.inputs) == model.body_completed == 40
    assert result["summary"]["qualified"] is True
    assert len(result["records"]) == 4
    assert [row["status"] for row in result["records"]] == ["completed"] * 4
    offset = 0
    for fixture in range(2):
        prompt = [1, 2, 3 if fixture == 0 else 4]
        generation = read_json(tmp_path / "run" / f"generation_{fixture}" / "result.json")
        assert generation["token_ids"] == [4] * 16
        assert generation["stop_reason"] == "cap"
        for step in range(16):
            actual = model.inputs[offset + step]
            expected_input = prompt + [4] * step
            assert actual["input_ids"].tolist() == [expected_input]
            assert actual["attention_mask"].tolist() == [[1] * len(expected_input)]
            assert actual["position_ids"].tolist() == [list(range(len(expected_input)))]
            assert actual["past_key_values"] is None and actual["use_cache"] is False
            assert actual["logits_to_keep"] == 1 and actual["inference_mode"] is True
            assert actual["output_hidden_states"] is actual["output_attentions"] is False
            directory = tmp_path / "run" / f"generation_{fixture}" / f"step_{step:02d}"
            expected_row = [-2.0] * 64
            expected_row[4] = 3.0
            assert raw_values(directory / "logits.fp32") == expected_row
            assert read_json(directory / "attempt.json")["input_token_ids"] == expected_input
            assert read_json(directory / "result.json")["chosen_token_id"] == 4
        for j, landmark in enumerate((0, 8)):
            slot = tmp_path / "run" / f"fixture_{fixture}_t{landmark}"
            loaded = subject.capture.load_capture(slot / "adapter", plan["manifest"])
            expected_input = prompt + [4] * landmark
            assert model.inputs[offset + 16 + 2 * j]["input_ids"].tolist() == [expected_input]
            assert model.inputs[offset + 17 + 2 * j]["input_ids"].tolist() == [expected_input]
            assert (
                loaded.bundle.metadata["capture"]["evidence"]["input_token_ids"] == expected_input
            )
            assert_private_endpoint_missing(loaded.bundle)
            assert (slot / "caller-return.json").is_file()
            gold = [expected_input[-1] * 10 + index + 2 for index in range(4)]
            assert raw_values(slot / "reference" / "norm_input.fp32") == gold
            assert raw_values(slot / "reference" / "norm_output.fp32") == [
                value / 4 for value in gold
            ]
            assert loaded.bundle.blobs["capture.fp32"] == struct.pack("<4f", *gold)
            assert raw_values(slot / "adapter-logits.fp32") == expected_row
            assert (
                loaded.bundle.blobs["capture.fp32"]
                == (slot / "reference" / "norm_input.fp32").read_bytes()
            )
            assert (slot / "reference" / "norm_input.fp32").read_bytes() != (
                slot / "reference" / "norm_output.fp32"
            ).read_bytes()
        offset += 20
    assert subject.load_qualification(tmp_path / "run", plan) == result
    assert all(
        item["fit_budget"] == item["calibration_budget"] == 0
        for item in plan["manifest"]["comparators"].values()
    )
    no_hooks(model)


@pytest.mark.parametrize(
    "choices,expected_generation,expected_entries,qualified,statuses",
    [
        ([63], [63], 6, None, ["completed", "unavailable"] * 2),
        ([4] * 8 + [63], [4] * 8 + [63], 26, True, ["completed"] * 4),
        ([(7, 2), 63], [2, 63], 8, None, ["completed", "unavailable"] * 2),
    ],
)
def test_first_eos_and_lowest_tie_index_preserve_all_four_slots(
    tmp_path, choices, expected_generation, expected_entries, qualified, statuses
):
    model, _, plan, result = run_fixture(tmp_path, GreedyFixture(choices))
    assert result["summary"]["total_entries"] == len(model.inputs) == expected_entries
    assert result["summary"]["qualified"] is qualified
    assert [row["status"] for row in result["records"]] == statuses
    for fixture in range(2):
        generation = read_json(tmp_path / "run" / f"generation_{fixture}" / "result.json")
        assert generation["token_ids"] == expected_generation
        assert generation["stop_reason"] == "eos"
        assert not (
            tmp_path / "run" / f"generation_{fixture}" / f"step_{len(expected_generation):02d}"
        ).exists()
    assert subject.load_qualification(tmp_path / "run", plan) == result
    no_hooks(model)


def test_completed_comparison_false_continues_fixed_schedule(tmp_path):
    model = GreedyFixture()
    model.residual_drift_entries = {18}
    _, _, plan, result = run_fixture(tmp_path, model)
    assert result["summary"]["qualified"] is False
    assert result["summary"]["total_entries"] == len(model.inputs) == 40
    assert [row["status"] for row in result["records"]] == ["completed"] * 4
    assert result["records"][0]["capture_equal"] is False
    assert all(row["capture_equal"] is True for row in result["records"][1:])
    assert subject.load_qualification(tmp_path / "run", plan) == result


@pytest.mark.parametrize("logit_failure", ["nonfinite", "dtype", "shape"])
def test_invalid_full_logit_row_stops_after_one_returned_body(tmp_path, logit_failure):
    model = GreedyFixture()
    model.logit_failure = logit_failure
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    with pytest.raises(ValueError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    failure = read_json(tmp_path / "run" / "failure.json")
    assert model.body_completed == len(model.inputs) == 1
    assert failure["summary"]["entries"] == {"generation": 1, "capture": 0, "reference": 0}
    assert [row["status"] for row in failure["records"]] == ["unattempted"] * 4
    assert not (tmp_path / "run" / "generation_0" / "step_01").exists()
    assert not (tmp_path / "run" / "generation_1").exists()
    step_failure = read_json(tmp_path / "run" / "generation_0" / "step_00" / "failure.json")
    assert step_failure["model_call_returned"] is True
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_qualification(tmp_path / "run", plan)
    no_hooks(model)


@pytest.mark.parametrize("phase", ["generation", "reference"])
def test_late_body_failure_stops_no_retry_and_cleans_reference_hooks(tmp_path, phase):
    model = GreedyFixture()
    model.fail_at_entry = 3 if phase == "generation" else 18
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    with pytest.raises(RuntimeError) as exc:
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert exc.value is model.failure
    assert len(model.inputs) == model.fail_at_entry
    assert model.body_completed == model.fail_at_entry - 1
    failure = read_json(tmp_path / "run" / "failure.json")
    assert failure["summary"]["total_entries"] == model.fail_at_entry
    assert [row["status"] for row in failure["records"]] == (
        ["unattempted"] * 4 if phase == "generation" else ["failed"] + ["unattempted"] * 3
    )
    with pytest.raises(FileExistsError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert len(model.inputs) == model.fail_at_entry
    no_hooks(model)


@pytest.mark.parametrize(
    "failure_site,expected_entry", [("registration", 0), ("entry_publication", 1)]
)
def test_prehook_entry_is_not_forward_body_completion(
    tmp_path, monkeypatch, failure_site, expected_entry
):
    model = GreedyFixture()
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    if failure_site == "registration":

        def registration(*args, **kwargs):
            raise RuntimeError("invented registration failure")

        monkeypatch.setattr(model, "register_forward_pre_hook", registration)
    else:
        publish = evidence._publish

        def publication(path, raw):
            if path.name == "entry.json":
                raise RuntimeError("invented entry publication failure")
            return publish(path, raw)

        monkeypatch.setattr(evidence, "_publish", publication)
    with pytest.raises(RuntimeError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    failure = read_json(tmp_path / "run" / "failure.json")
    assert model.inputs == [] and model.body_completed == 0
    assert failure["summary"]["total_entries"] == expected_entry
    step_failure = read_json(tmp_path / "run" / "generation_0" / "step_00" / "failure.json")
    assert step_failure["model_call_returned"] is False
    assert [row["status"] for row in failure["records"]] == ["unattempted"] * 4
    no_hooks(model)


def test_post_entry_input_mutation_is_not_certified(tmp_path):
    model = GreedyFixture()
    model.mutate_input = True
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    with pytest.raises(ValueError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert len(model.inputs) == 1
    no_hooks(model)


def test_exception_after_adapter_returns_does_not_fabricate_zero_entries(tmp_path, monkeypatch):
    model = GreedyFixture([63])
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    original = subject.capture.capture_prefix

    def interrupted_return(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("invented lost caller return")

    monkeypatch.setattr(subject.capture, "capture_prefix", interrupted_return)
    with pytest.raises(RuntimeError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert len(model.inputs) == 2
    failure = read_json(tmp_path / "run" / "failure.json")
    count = failure["summary"]["entries"]["capture"]
    assert count is None or count >= 1
    assert failure["summary"]["entries_complete"] is False
    assert failure["summary"]["entry_lower_bounds"]["capture"] >= 1
    assert failure["summary"]["total_entries"] is None
    assert failure["summary"]["qualified"] is False
    assert not (tmp_path / "run" / "fixture_0_t0" / "caller-return.json").exists()
    assert [row["status"] for row in failure["records"]] == ["failed"] + ["unattempted"] * 3
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_qualification(tmp_path / "run", plan)
    no_hooks(model)


@pytest.mark.parametrize("boundary", ["step_result", "caller_return", "runner_terminal"])
def test_publication_failure_stops_without_rescuing_orphan_success(tmp_path, monkeypatch, boundary):
    model = GreedyFixture([63])
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    publish = evidence._publish

    def fail_publication(path, raw):
        match = (
            (
                boundary == "step_result"
                and path.name == "result.json"
                and path.parent.name == "step_00"
            )
            or (boundary == "caller_return" and path.name == "caller-return.json")
            or (boundary == "runner_terminal" and path.name == "terminal.json")
        )
        value = publish(path, raw)
        if match:
            raise OSError("invented failure after durable child publication")
        return value

    monkeypatch.setattr(evidence, "_publish", fail_publication)
    with pytest.raises(OSError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    expected = {"step_result": 1, "caller_return": 2, "runner_terminal": 6}[boundary]
    assert len(model.inputs) == expected
    failure = read_json(tmp_path / "run" / "failure.json")
    assert failure["summary"]["qualified"] is False
    assert failure["summary"]["total_entries"] == expected
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_qualification(tmp_path / "run", plan)
    with pytest.raises(FileExistsError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert len(model.inputs) == expected
    no_hooks(model)


class ReferenceCleanupError(RuntimeError):
    pass


def test_reference_cleanup_keeps_original_error_and_invalidates_captured_vectors(
    tmp_path, monkeypatch
):
    model = GreedyFixture([63])
    model.fail_at_entry = 3
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    register = model.model.norm.register_forward_hook

    def registration(*args, **kwargs):
        handle = register(*args, **kwargs)
        remove = handle.remove

        def broken_remove():
            remove()
            raise ReferenceCleanupError("invented reference cleanup failure")

        handle.remove = broken_remove
        return handle

    monkeypatch.setattr(model.model.norm, "register_forward_hook", registration)
    with pytest.raises(RuntimeError) as raised:
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert raised.value is model.failure
    reference = tmp_path / "run" / "fixture_0_t0" / "reference"
    failure = read_json(reference / "failure.json")
    assert failure["entries"] == failure["norm_pre_count"] == failure["norm_post_count"] == 1
    assert failure["model_call_returned"] is False
    assert failure["cleanup_failure_type"] == "ReferenceCleanupError"
    assert failure["capture_usable"] is False
    assert not (reference / "norm_input.fp32").exists()
    assert len(model.inputs) == 3
    no_hooks(model)


def test_known_endpoint_injection_cannot_be_replayed_as_apparatus_result(tmp_path, monkeypatch):
    model = GreedyFixture([63])
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    arguments = subject._landmark_arguments

    def mislabeled(*args, **kwargs):
        result = arguments(*args, **kwargs)
        result["outcome"].update(
            status="known", applicable=True, label=1, reason=None, disagreement=False
        )
        return result

    monkeypatch.setattr(subject, "_landmark_arguments", mislabeled)
    with pytest.raises(ValueError, match="unmeasured_outcome"):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    with pytest.raises(ValueError):
        subject.load_qualification(tmp_path / "run", plan)
    no_hooks(model)


@pytest.mark.parametrize("mutation", ["forward_ceiling", "prompt_cap"])
def test_fixed_plan_or_prompt_limits_fail_before_dispatch(tmp_path, mutation):
    model = GreedyFixture()
    tokenizer, plan, prepared, binding = fixture_plan(subject, model)
    if mutation == "forward_ceiling":
        plan["max_entries"] = 41
    else:
        observation = prepared["observations"][0]
        observation["prompt_token_ids"] = [2] * 129
        observation["prompt_ids_sha256"] = digest(canonical(observation["prompt_token_ids"]))
    with pytest.raises(ValueError):
        subject.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=tmp_path / "run",
        )
    assert model.inputs == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mutation", ["payload_eos", "boolean_special_id"])
def test_native_preparation_rejects_payload_specials_and_typed_id_corruption(monkeypatch, mutation):
    model = GreedyFixture()
    tokenizer, plan, _, _ = fixture_plan(subject, model)
    if mutation == "boolean_special_id":
        tokenizer.all_special_ids = [False, 1, 63]
    else:
        encode = tokenizer.encode

        def payload_eos(text, **kwargs):
            if text == "Follow the user's instruction.":
                return [63]
            return encode(text, **kwargs)

        monkeypatch.setattr(tokenizer, "encode", payload_eos)
    with pytest.raises(ValueError):
        subject.prepare_inputs(plan, tokenizer)
    assert model.inputs == []


@pytest.mark.parametrize(
    "kind", ["boolean_attempt", "unbound_adapter_logits", "unbound_step_payload"]
)
def test_retained_payload_and_attempt_mutations_are_rejected(tmp_path, kind):
    model = GreedyFixture([63])
    if kind == "unbound_adapter_logits":
        # First reference differs in residual only; change one adapter logit
        # while keeping the saved False logit comparison in a coherent fixture.
        model.logit_drift_entries = {3}
    _, _, plan, _ = run_fixture(tmp_path, model)
    root = tmp_path / "run"
    if kind == "boolean_attempt":
        path = root / "generation_0" / "step_00" / "attempt.json"
        data = read_json(path)
        data["index"] = False
        path.write_bytes(canonical(data))
    elif kind == "unbound_adapter_logits":
        path = root / "fixture_0_t0" / "adapter-logits.fp32"
        values = raw_values(path)
        values[11] = -8.0
        path.write_bytes(struct.pack("<64f", *values))
    else:
        directory = root / "generation_0" / "step_00"
        values = raw_values(directory / "logits.fp32")
        values[11] = -8.0  # Same selected EOS; changed retained full row.
        raw = struct.pack("<64f", *values)
        (directory / "logits.fp32").write_bytes(raw)
        data = read_json(directory / "result.json")
        data["files"]["logits.fp32"] = digest(raw)
        (directory / "result.json").write_bytes(canonical(data))
    with pytest.raises(ValueError):
        subject.load_qualification(root, plan)


def test_tiny_untrained_full_integration_uses_fixed_40_forwards(tmp_path):
    model, _, plan, result = run_fixture(tmp_path, tiny_model())
    assert result["summary"]["qualified"] is True
    assert result["summary"]["entries"] == {"generation": 32, "capture": 4, "reference": 4}
    assert result["summary"]["total_entries"] == 40
    assert [row["status"] for row in result["records"]] == ["completed"] * 4
    for i in range(2):
        generation = read_json(tmp_path / "run" / f"generation_{i}" / "result.json")
        assert generation["token_ids"] == [0] * 16
        assert generation["stop_reason"] == "cap"
    assert subject.load_qualification(tmp_path / "run", plan) == result
    no_hooks(model)
