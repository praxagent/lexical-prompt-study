import copy
import json
from types import SimpleNamespace

import pytest

from lexical_prompt_study import instruction_binding_runtime as rt
from lexical_prompt_study.instruction_binding_plan import compile_plan


class CharacterTokenizer:
    bos_token_id = 1

    def get_chat_template(self):
        return "synthetic native template"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) + 2 for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 2) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["date_string"] == rt.TEMPLATE_DATE
        text = "SYSTEM\n" + messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"
        if tokenize:
            assert kwargs["return_dict"] is False
            return self.encode(text, add_special_tokens=False)
        return text


@pytest.fixture
def plan():
    return compile_plan(stage="development_pilot", scaffolds={
        key: "Harmless synthetic fixture " + key for key in ("full", "sham", "replacement", "inert")
    }, bindings={})


@pytest.fixture
def config():
    return {
        "schema_version": "a163-runtime-v1", "model_path": "/synthetic/model",
        "model_revision": "a" * 40,
        "model_files_sha256": {name: "b" * 64 for name in (
            "config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")},
        "chat_template_sha256": rt.digest(CharacterTokenizer().get_chat_template().encode()),
        "runtime_versions": {name: "synthetic" for name in ("python", *rt.VERSION_PACKAGES)},
        "quantization": "nf4", "dtype": "bfloat16", "attention_implementation": "sdpa",
        "device": "cuda:0", "max_prompt_tokens": 4096, "max_new_tokens": 64,
        "seed": 20260916, "cpu_threads": 4, "gpu_memory_gib": 10,
    }


def test_prepared_tokens_use_native_system_user_and_exact_fixed_prefix(plan, config):
    token = CharacterTokenizer()
    trial = plan["trials"][0]
    prepared = rt.prepare_trial(token, trial, config)
    assert prepared["rendered_text"].startswith("SYSTEM\n" + trial["messages"][0]["content"])
    assert token.decode(prepared["forced_prefix_token_ids"]) == prepared["rendered_text"] + rt.ANSWER_PREFIX
    for key in ("selected", "unselected"):
        suffix = token.decode(prepared["candidate_token_ids"][key])
        assert suffix == trial[key + "_answer"] + '"}'
        assert token.encode(prepared["rendered_text"] + rt.ANSWER_PREFIX + suffix,
                            add_special_tokens=False) == (
            prepared["forced_prefix_token_ids"] + prepared["candidate_token_ids"][key])


def test_prefix_merging_is_rejected_instead_of_using_standalone_suffix_tokens(plan, config):
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            values = super().encode(text, **kwargs)
            if rt.ANSWER_PREFIX + "s" in text:
                values[len(text) - 5] += 1
            return values
    prepared = rt.prepare_trial(MergingTokenizer(), plan["trials"][0], config)
    assert prepared["likelihood_error"]["code"] == "likelihood_tokenization_failed"
    assert prepared["candidate_token_ids"] is None
    record, _ = rt.evaluate_trial(FakeRuntime(plan), plan, plan["trials"][0], prepared)
    assert record["generation_status"] == "completed"
    assert record["teacher_forced"]["status"] == "failed"


def test_template_drift_and_context_overflow_fail_closed(plan, config):
    with pytest.raises(ValueError, match="template drift"):
        rt.prepare_trial(CharacterTokenizer(), plan["trials"][0],
                         {**config, "chat_template_sha256": "a" * 64})
    with pytest.raises(ValueError, match="exceeds frozen limit"):
        rt.prepare_trial(CharacterTokenizer(), plan["trials"][0],
                         {**config, "max_prompt_tokens": 64})


@pytest.mark.parametrize("field,value", [
    ("max_new_tokens", 128), ("gpu_memory_gib", 11), ("cpu_threads", 8),
    ("quantization", "none"), ("device", "cpu"), ("seed", True),
    ("model_revision", "main"), ("max_prompt_tokens", 8193),
])
def test_config_rejects_unfrozen_runtime_contract(config, field, value):
    with pytest.raises(ValueError):
        rt.validate_config({**config, field: value})


def test_plan_oracle_and_actual_depth_are_independently_revalidated(plan):
    rt.validate_plan(plan)
    broken = copy.deepcopy(plan)
    broken["trials"][0]["selected_answer"] = broken["trials"][0]["unselected_answer"]
    with pytest.raises(ValueError, match="oracle mismatch"):
        rt.validate_plan(broken)
    broken = copy.deepcopy(plan)
    broken["trials"][0]["messages"][0]["role"] = "user"
    with pytest.raises(ValueError, match="system and user"):
        rt.validate_plan(broken)


class FakeRuntime:
    tokenizer = CharacterTokenizer()
    eos_ids = [0]

    def __init__(self, plan, finish="eos", fail_generation=False, fail_likelihood=False):
        self.plan, self.finish = plan, finish
        self.fail_generation, self.fail_likelihood = fail_generation, fail_likelihood
        self.calls = 0

    def generate(self, prepared):
        self.calls += 1
        if self.fail_generation:
            raise RuntimeError("private context must never be printed")
        trial = next(t for t in self.plan["trials"]
                     if t["messages"][0]["content"] in prepared["rendered_text"])
        text = json.dumps({"answer": trial["selected_answer"]})
        tokens = [3, 0] if self.finish == "eos" else [3] * 64
        return tokens, text, self.finish

    def likelihoods(self, prepared):
        if self.fail_likelihood:
            raise RuntimeError("private context")
        return {"status": "ok", "selected_logprob": -1.25, "alternative_logprob": -4.5,
                "selected_token_count": len(prepared["candidate_token_ids"]["selected"]),
                "alternative_token_count": len(prepared["candidate_token_ids"]["unselected"])}


def test_length_termination_fails_strict_endpoint_even_with_exact_json(plan, config):
    trial = plan["trials"][0]
    prepared = rt.prepare_trial(CharacterTokenizer(), trial, config)
    record, _ = rt.evaluate_trial(FakeRuntime(plan, finish="length"), plan, trial, prepared)
    assert record["score"] == {"category": "exact", "strict_correct": False}
    assert record["finish_reason"] == "length"
    assert record["teacher_forced"]["status"] == "ok"


@pytest.mark.parametrize("generation,likelihood", [(True, False), (False, True), (True, True)])
def test_generation_and_teacher_forcing_failures_remain_independent(plan, config, generation, likelihood):
    trial = plan["trials"][0]
    prepared = rt.prepare_trial(CharacterTokenizer(), trial, config)
    record, private = rt.evaluate_trial(FakeRuntime(plan, fail_generation=generation,
                                                  fail_likelihood=likelihood), plan, trial, prepared)
    assert (record["generation_status"] == "infrastructure_failed") is generation
    assert (record["teacher_forced"]["status"] == "failed") is likelihood
    if generation:
        assert record["score"] == {"category": None, "strict_correct": None}
    if likelihood:
        assert record["teacher_forced"]["selected_logprob"] is None
    assert "private context" not in json.dumps(private)


def arguments(tmp_path, plan, config):
    plan_path, config_path = tmp_path / "plan.json", tmp_path / "config.json"
    plan_path.write_bytes(rt.canonical(plan))
    config_path.write_bytes(rt.canonical(config))
    return dict(plan_path=plan_path, config_path=config_path, output_root=tmp_path / "run",
                expected_plan_sha256=rt.file_digest(plan_path),
                expected_config_sha256=rt.file_digest(config_path))


def test_durable_resume_and_verified_numeric_export_without_reloading_model(tmp_path, plan, config):
    plan["trials"] = plan["trials"][:2]
    args = arguments(tmp_path, plan, config)
    runtime = FakeRuntime(plan)
    summary = rt.run_plan(**args, runtime_loader=lambda c: runtime)
    assert summary["recorded"] == summary["generation_completed"] == 2
    assert runtime.calls == 2
    def forbidden(config):
        pytest.fail("completed resume loaded model")
    resumed = rt.run_plan(**args, runtime_loader=forbidden)
    assert resumed["launched_this_call"] == 0
    records = rt.export_records(**args)
    assert len(records) == 2
    assert all(record["score"]["strict_correct"] for record in records)
    assert "response_text" not in json.dumps(records)
    assert all(path.stat().st_mode & 0o777 == 0o600
               for path in args["output_root"].rglob("*.json"))


def test_resumed_receipt_rejects_forged_score_and_input_drift(tmp_path, plan, config):
    plan["trials"] = plan["trials"][:1]
    args = arguments(tmp_path, plan, config)
    rt.run_plan(**args, runtime_loader=lambda c: FakeRuntime(plan))
    result = next(args["output_root"].glob("trials/*/attempt-01/result.json"))
    content = json.loads(result.read_bytes())
    content["record"]["score"]["category"] = "other_selector"
    result.write_bytes(rt.canonical(content))
    with pytest.raises(ValueError, match="does not replay"):
        rt.export_records(**args)
    with pytest.raises(ValueError, match="input hash drift"):
        rt.run_plan(**{**args, "expected_plan_sha256": "a" * 64})


def test_interrupted_attempt_needs_explicit_retry_and_preserves_attempts(tmp_path, plan, config):
    plan["trials"] = plan["trials"][:1]
    args = arguments(tmp_path, plan, config)
    rt.run_plan(**args, runtime_loader=lambda c: FakeRuntime(plan))
    result = next(args["output_root"].glob("trials/*/attempt-01/result.json"))
    result.unlink()  # Synthetic interruption fixture, not an operational deletion.
    with pytest.raises(RuntimeError, match="explicit retry"):
        rt.run_plan(**args, runtime_loader=lambda c: FakeRuntime(plan))
    assert rt.export_records(**args) == []
    summary = rt.run_plan(**args, retry_failed=True, runtime_loader=lambda c: FakeRuntime(plan))
    assert summary["launched_this_call"] == 1
    assert len(list(args["output_root"].glob("trials/*/attempt-*/attempt.json"))) == 2


def test_partial_launch_keeps_unlaunched_cells_missing(tmp_path, plan, config):
    plan["trials"] = plan["trials"][:2]
    args = arguments(tmp_path, plan, config)
    summary = rt.run_plan(**args, max_trials=1, runtime_loader=lambda c: FakeRuntime(plan))
    assert summary["planned"] == 2 and summary["recorded"] == 1
    assert not summary["all_cells_recorded"]
    assert len(rt.export_records(**args)) == 1


def test_retry_only_repairs_failed_measurement_preserving_completed_generation(tmp_path, plan, config):
    plan["trials"] = plan["trials"][:1]
    args = arguments(tmp_path, plan, config)
    first = FakeRuntime(plan, fail_likelihood=True)
    rt.run_plan(**args, runtime_loader=lambda c: first)
    second = FakeRuntime(plan)
    rt.run_plan(**args, retry_failed=True, runtime_loader=lambda c: second)
    assert first.calls == 1 and second.calls == 0
    records = rt.export_records(**args)
    assert records[0]["score"]["strict_correct"] is True
    assert records[0]["teacher_forced"]["status"] == "ok"


def test_safe_exception_diagnostics_exclude_message_source_and_locals():
    try:
        raise RuntimeError("private fixture must not be retained in diagnostics")
    except RuntimeError as exc:
        detail = rt.safe_error(exc, "synthetic_failed")
    assert detail["exception_type"] == "RuntimeError"
    assert detail["frames"][0]["function"].startswith("test_safe_exception")
    assert set(detail["frames"][0]) == {"file", "function", "line"}
    assert "private fixture" not in json.dumps(detail)


def test_atomic_private_writes_are_immutable(tmp_path):
    path = tmp_path / "nested" / "receipt.json"
    rt.write_private(path, {"count": 1})
    rt.write_private(path, {"count": 1})
    with pytest.raises(ValueError, match="immutable"):
        rt.write_private(path, {"count": 2})
    assert json.loads(path.read_bytes()) == {"count": 1}


def test_teacher_forced_logprob_uses_causal_shift_and_all_suffix_tokens():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[0., 2., 1.], [3., 1., 0.], [99., -99., 0.]]])
    seen = {}
    def model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(logits=logits)
    score = rt.continuation_logprob(model, torch, [2, 2], [1, 0], "cpu")
    expected = (torch.log_softmax(logits[0, 0], 0)[1]
                + torch.log_softmax(logits[0, 1], 0)[0]).item()
    assert score == pytest.approx(expected)
    assert seen["input_ids"].tolist() == [[2, 2, 1, 0]]
    assert seen["logits_to_keep"] == 3 and seen["use_cache"] is False
    assert score != pytest.approx(torch.log_softmax(logits[0, -1], 0)[0].item())
