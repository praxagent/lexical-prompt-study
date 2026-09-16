import json

import pytest

from lexical_prompt_study import instruction_selection_runtime as rt
from lexical_prompt_study import instruction_selection_tasks as tasks


class Tokenizer:
    def get_chat_template(self):
        return "fixed synthetic template"

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(value - 10) for value in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        text = messages[0]["content"] + "\n---USER---\n" + messages[1]["content"] + "\n---ASSISTANT---\n"
        if tokenize:
            assert kwargs["return_dict"] is False
            return self.encode(text)
        return text


class FakeRuntime:
    tokenizer = Tokenizer()
    eos_ids = [1]

    def __init__(self, plan, mode="correct"):
        self.mode, self.calls = mode, 0
        self.trials = {self.tokenizer.apply_chat_template(trial["messages"], tokenize=False,
                      add_generation_prompt=True): trial for trial in plan["trials"]}

    def generate(self, prepared):
        self.calls += 1
        if self.mode == "infrastructure":
            raise RuntimeError("private synthetic request must not be printed")
        trial = self.trials[prepared["rendered_text"]]
        value = trial["unselected_answer" if self.mode == "wrong" else "selected_answer"]
        text = json.dumps({"answer": value})
        if self.mode == "cap":
            text += " " * (64 - len(text))
            return self.tokenizer.encode(text), text, "length"
        return self.tokenizer.encode(text) + [1], text, "eos"


@pytest.fixture
def inputs(tmp_path):
    manifest = tasks.make_cohort_manifest()
    scaffolds = {kind: "Public synthetic " + kind for kind in tasks.SCAFFOLDS}
    bindings = {"protocol_sha256": "a" * 64, "materials_sha256": "b" * 64}
    development = tasks.compile_plan(manifest, partition="development", scaffolds=scaffolds, bindings=bindings)
    heldout = tasks.compile_plan(manifest, partition="heldout", scaffolds=scaffolds, bindings=bindings)
    model = tmp_path / "model"
    model.mkdir()
    generation = model / "generation_config.json"
    generation.write_text('{"eos_token_id":[1]}')
    config = {
        "schema_version": "a163-runtime-v1", "model_path": str(model), "model_revision": "a" * 40,
        "model_files_sha256": {**{name: "b" * 64 for name in (
            "config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")},
            "generation_config.json": rt.engine.file_digest(generation)},
        "chat_template_sha256": rt.engine.digest(Tokenizer().get_chat_template().encode()),
        "runtime_versions": {name: "synthetic" for name in ("python", *rt.engine.VERSION_PACKAGES)},
        "quantization": "nf4", "dtype": "bfloat16", "attention_implementation": "sdpa", "device": "cuda:0",
        "max_prompt_tokens": 4096, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10,
    }
    result = {}
    for name, value in (("development", development), ("heldout", heldout), ("config", config)):
        path = tmp_path / (name + ".json")
        path.write_bytes(tasks.canonical(value))
        result[name] = value
        result[name + "_path"] = path
    result["tmp"] = tmp_path
    return result


def args(inputs, partition="development"):
    return {"plan_path": inputs[partition + "_path"], "config_path": inputs["config_path"],
            "output_root": inputs["tmp"] / (partition + "-run"),
            "expected_plan_sha256": rt.engine.file_digest(inputs[partition + "_path"]),
            "expected_config_sha256": rt.engine.file_digest(inputs["config_path"]),
            "tokenizer_loader": lambda config: Tokenizer()}


def test_generation_only_receipts_resume_and_analysis_integration(inputs):
    runtime = FakeRuntime(inputs["development"])
    value = rt.run_plan(**args(inputs), runtime_loader=lambda config: runtime)
    assert value["recorded"] == value["strict_correct"] == runtime.calls == 32
    assert value["likelihood_collected"] is False
    def forbidden(config):
        pytest.fail("completed resume loaded model")
    assert rt.run_plan(**args(inputs), runtime_loader=forbidden)["launched_this_call"] == 0
    records = rt.export_records(**args(inputs))
    assert len(records) == 32 and all(row["likelihood_collected"] is False for row in records)
    assert all("teacher_forced" not in row and "task_depth" not in row for row in records)
    from lexical_prompt_study.instruction_selection_analysis import analyze_plan_records
    assert analyze_plan_records(inputs["development"], records)["baseline_controls"]["gate_passed"]


@pytest.mark.parametrize("mode", ["wrong", "cap"])
def test_completed_task_failures_are_never_retried(inputs, mode):
    first = FakeRuntime(inputs["development"], mode)
    rt.run_plan(**args(inputs), max_trials=1, runtime_loader=lambda config: first)
    # A second call may launch the next cell, but must preserve the first failure.
    second = FakeRuntime(inputs["development"])
    rt.run_plan(**args(inputs), max_trials=1, retry_failed=True, runtime_loader=lambda config: second)
    records = rt.export_records(**args(inputs))
    assert records[0]["score"]["strict_correct"] is False
    assert len(list(args(inputs)["output_root"].glob("trials/*/attempt-02"))) == 0
    assert first.calls == second.calls == 1


def test_infrastructure_failure_preserved_then_explicitly_retried(inputs):
    rt.run_plan(**args(inputs), max_trials=1,
                runtime_loader=lambda config: FakeRuntime(inputs["development"], "infrastructure"))
    row = rt.export_records(**args(inputs))[0]
    assert row["generation_status"] == "infrastructure_failed" and row["score"]["strict_correct"] is None
    rt.run_plan(**args(inputs), max_trials=1, retry_failed=True,
                runtime_loader=lambda config: FakeRuntime(inputs["development"]))
    assert rt.export_records(**args(inputs))[0]["score"]["strict_correct"] is True
    assert len(list(args(inputs)["output_root"].glob("trials/*/attempt-02/result.json"))) == 1


@pytest.mark.parametrize("target", ["prepared", "eos", "text", "score"])
def test_replay_rejects_wrong_native_prompt_eos_response_or_score(inputs, target):
    rt.run_plan(**args(inputs), max_trials=1, runtime_loader=lambda config: FakeRuntime(inputs["development"]))
    root = args(inputs)["output_root"]
    attempt_path = next(root.glob("trials/*/attempt-01/attempt.json"))
    result_path = attempt_path.with_name("result.json")
    attempt, result = json.loads(attempt_path.read_bytes()), json.loads(result_path.read_bytes())
    if target == "prepared":
        attempt["prepared"]["prompt_token_ids"][0] += 1
    elif target == "eos":
        attempt["eos_token_ids"] = [2]
    elif target == "text":
        result["private"]["response_text"] = '{"answer":"s99"}'
    else:
        result["record"]["score"]["category"] = "other_selector"
    if target in ("prepared", "eos"):
        attempt_path.write_bytes(tasks.canonical(attempt))
        result["attempt_sha256"] = rt.engine.file_digest(attempt_path)
    result_path.write_bytes(tasks.canonical(result))
    with pytest.raises(ValueError):
        rt.export_records(**args(inputs))


def test_interrupted_attempt_requires_explicit_retry_and_is_unresolved(inputs):
    rt.run_plan(**args(inputs), max_trials=1, runtime_loader=lambda config: FakeRuntime(inputs["development"]))
    next(args(inputs)["output_root"].glob("trials/*/attempt-01/result.json")).unlink()
    assert rt.export_records(**args(inputs)) == []
    with pytest.raises(RuntimeError, match="explicit_retry"):
        rt.run_plan(**args(inputs), max_trials=1, runtime_loader=lambda config: FakeRuntime(inputs["development"]))


def test_heldout_requires_concrete_passed_dev_gate_before_model_loading(inputs):
    called = []
    def loader(config):
        called.append(True)
    with pytest.raises(ValueError, match="development_gate_required"):
        rt.run_plan(**args(inputs, "heldout"), runtime_loader=loader)
    assert not called
    rt.run_plan(**args(inputs), runtime_loader=lambda config: FakeRuntime(inputs["development"], "wrong"))
    heldout = {**args(inputs, "heldout"), "development_plan_path": inputs["development_path"],
               "development_plan_sha256": rt.engine.file_digest(inputs["development_path"]),
               "development_run_root": args(inputs)["output_root"]}
    with pytest.raises(ValueError, match="development_competence_gate_failed"):
        rt.run_plan(**heldout, runtime_loader=loader)
    assert not called


def test_heldout_passes_replayed_gate_with_same_instrument_and_immutable_receipts(inputs):
    rt.run_plan(**args(inputs), runtime_loader=lambda config: FakeRuntime(inputs["development"]))
    heldout = {**args(inputs, "heldout"), "development_plan_path": inputs["development_path"],
               "development_plan_sha256": rt.engine.file_digest(inputs["development_path"]),
               "development_run_root": args(inputs)["output_root"]}
    summary = rt.run_plan(**heldout, max_trials=1, runtime_loader=lambda config: FakeRuntime(inputs["heldout"]))
    assert summary["planned"] == 1152 and summary["recorded"] == 1
    assert not summary["all_cells_recorded"]
    header = json.loads((args(inputs, "heldout")["output_root"] / "run.json").read_bytes())
    assert header["schema_version"] == "a164-run-v1" and header["development_gate"]["gate_passed"]
    assert len(rt.export_records(**args(inputs, "heldout"))) == 1
