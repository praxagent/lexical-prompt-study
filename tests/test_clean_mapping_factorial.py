"""A173 public synthetic qualification; no target tokenizer, weights or model calls.

Native/scoring/lifecycle fixtures retain A171 conventions. Factorial stimuli,
schedule and paired arithmetic below are independently specified from protocol.
"""

import contextlib
import copy
import itertools
import json
import os
import signal
import types
from collections import Counter
from pathlib import Path

import pytest

from lexical_prompt_study import clean_mapping_factorial as a


EXPECTED_WORLDS = (("s20", "s27", "AB"), ("s24", "s21", "BA"),
                   ("s23", "s26", "BA"), ("s25", "s22", "AB"))
AXES = ("system_wording", "mapping_heading", "value_form")
LEVELS = (("legacy", "a171_standard"), ("off", "on"), ("short_symbol", "study_prefixed"))
EXPECTED_BASE = tuple(itertools.product(*LEVELS, ("A", "B")))
LEGACY = ('The authoritative query selector is {selector}. The user supplies two explicit answer values labeled A and B. '
          'Copy the value with the authoritative label. Output exactly one JSON object with the single field "answer" '
          'containing that value as a string. Do not transform the value or include other fields, explanations, or Markdown fences.')
STANDARD = ('Select the entry labeled {selector} from the user\'s mapping. Return that entry\'s mapped value '
            'as exactly one JSON object with the single key "answer" and a string value. Do not return any other text.')


class Tokenizer:
    """Character coding with explicit EOS IDs, independent of target tokenization."""

    eos_token_id, pad_token_id, bos_token_id = 1, None, 0

    def __len__(self):
        return 512

    def get_chat_template(self):
        return "a173 public synthetic native template"

    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith("<EOS>"):
                result.append(1)
                text = text[5:]
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result

    def decode(self, tokens, **kwargs):
        return "".join("<EOS>" if token in (1, 2) else chr(token - 10) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        text = messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"
        if not kwargs["add_generation_prompt"]:
            assert len(messages) == 3 and messages[2]["role"] == "assistant"
            text += messages[2]["content"] + "<EOS>"
        return self.encode(text) if tokenize else text



@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = tmp_path / "synthetic-model"
    model.mkdir()
    generation = model / "generation_config.json"
    generation.write_text('{"eos_token_id":[1,2]}')
    model_config = model / "config.json"
    model_config.write_text('{"max_position_embeddings":8192,"vocab_size":512}')
    digest = a.native.engine.file_digest
    config = {
        "schema_version": "a163-runtime-v1", "model_path": str(model), "model_revision": "a" * 40,
        "model_files_sha256": {
            **{name: "b" * 64 for name in
               ("tokenizer_config.json", "tokenizer.json", "model.safetensors")},
            "config.json": digest(model_config),
            "generation_config.json": digest(generation),
        },
        "chat_template_sha256": a.tasks.sha(Tokenizer().get_chat_template().encode()),
        "runtime_versions": {name: "synthetic" for name in ("python", *a.native.engine.VERSION_PACKAGES)},
        "quantization": "nf4", "dtype": "bfloat16", "attention_implementation": "sdpa", "device": "cuda:0",
        "max_prompt_tokens": 4064, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10,
    }
    config_path = tmp_path / "config.json"
    protocol = tmp_path / "protocol.md"
    reference = tmp_path / "reference.py"
    config_path.write_bytes(a.tasks.canonical(config))
    protocol.write_text("Public synthetic A173 protocol fixture")
    reference.write_text("Public synthetic reference binding fixture")
    monkeypatch.setattr(a, "CONFIG_SHA", digest(config_path))
    monkeypatch.setattr(a.native.engine, "file_digest", lambda path: a.REFERENCE_SHA if Path(path) == reference else digest(path))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    monkeypatch.setattr(a, "PRIVATE_RUNS_ROOT", tmp_path)
    plan = a.compile_plan(digest(config_path), digest(protocol), digest(Path(__file__)))
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(a.tasks.canonical(plan))
    args = {
        "plan_path": plan_path, "plan_sha256": digest(plan_path), "config_path": config_path,
        "config_sha256": digest(config_path), "protocol_path": protocol, "reference_script": reference,
        "output_root": tmp_path / "output", "tokenizer_loader": lambda config: Tokenizer(),
    }
    return plan, args



def prepare(fixture):
    return a.prepare_inputs(**{key: value for key, value in fixture[1].items() if key != "output_root"})



def encoded_answer(trial, field="selected_answer", eos=1):
    return Tokenizer().encode(json.dumps({"answer": trial[field]}, separators=(",", ":"))) + [eos]



@pytest.mark.parametrize("change", ["message", "oracle", "world", "schedule", "duplicate", "config", "protocol", "tests"])
def test_plan_drift_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == "message":
        plan["trials"][0]["messages"][1]["content"] += " "
    elif change == "oracle":
        plan["trials"][0]["selected_answer"] = "synthetic incorrect oracle"
    elif change == "world":
        plan["trials"][0]["world_index"] = 7
    elif change == "schedule":
        plan["trials"][0:2] = reversed(plan["trials"][0:2])
    elif change == "duplicate":
        plan["trials"].append(copy.deepcopy(plan["trials"][-1]))
    else:
        plan["bindings"][f"{change}_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.validate_plan(plan)



@pytest.mark.parametrize("answer_kind,category", [
    ("selected_answer", "correct"), ("unselected_answer", "other_mapped_value"),
    ("selector", "selected_label"), ("other_selector", "other_label"),
    ("unrecognized", "other_or_format"),
])
@pytest.mark.parametrize("selector", ["A", "B"])
def test_exhaustive_valid_answer_categories(fixture, answer_kind, category, selector):
    trial = next(trial for trial in fixture[0]["trials"] if trial["selector"] == selector)
    answer = trial.get(answer_kind, "B" if selector == "A" else "A")
    if answer_kind == "unrecognized":
        answer = "synthetic-other"
    tokens = Tokenizer().encode(json.dumps({"answer": answer}, separators=(",", ":"))) + [2]
    score = a.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    assert score["category"] == category
    assert score["strict_correct"] is (category == "correct")
    assert score["selected_label"] is (category == "selected_label")
    assert score["format_valid"] and score["eos_valid"] and not score["capped"]
    assert score["generated_token_count"] == len(tokens)
    assert score["attempted"] is True



@pytest.mark.parametrize("text", [
    '{"answer":"A","answer":"A"}', '{"answer":"A","extra":0}',
    '{"answer":0}', '{"answer":null}', '{"ANSWER":"A"}', '[{"answer":"A"}]',
    '"A"', '```json\n{"answer":"A"}\n```', '{"answer":"A"} trailing',
    '{"answer":"A"}{"answer":"A"}', '{"answer":', '',
])
def test_strict_format_rejections(fixture, text):
    trial = fixture[0]["trials"][0]
    score = a.score_tokens(Tokenizer(), Tokenizer().encode(text) + [1], trial, [1, 2])
    assert score["category"] == "other_or_format"
    assert not score["strict_correct"] and not score["selected_label"] and not score["format_valid"]
    assert score["eos_valid"] and not score["capped"]



def test_whitespace_terminal_eos_and_cap_precedence(fixture):
    trial = fixture[0]["trials"][0]
    body = Tokenizer().encode(json.dumps({"answer": trial["selected_answer"]}, separators=(",", ":")))
    last_slot_eos = body + [ord(" ") + 10] * (63 - len(body)) + [1]
    score = a.score_tokens(Tokenizer(), last_slot_eos, trial, [1, 2])
    assert len(last_slot_eos) == 64
    assert score["category"] == "correct" and score["eos_valid"] and not score["capped"]
    capped = last_slot_eos[:-1] + [ord(" ") + 10]
    score = a.score_tokens(Tokenizer(), capped, trial, [1, 2])
    assert score["category"] == "cap" and score["capped"] and not score["strict_correct"]
    short = a.score_tokens(Tokenizer(), body, trial, [1, 2])
    assert short["category"] == "other_or_format" and not short["eos_valid"] and not short["capped"]
    premature = a.score_tokens(Tokenizer(), [1] + body + [2], trial, [1, 2])
    assert premature["category"] == "other_or_format" and not premature["eos_valid"]



def test_native_preparation_replays_every_public_cell_without_consumption(fixture):
    prepared = prepare(fixture)
    assert prepared["planned_cells"] == prepared["maximum_generation_calls"] == 64
    assert prepared["eos_token_ids"] == [1, 2]
    assert prepared["native_prepared_sha256"] == a.object_sha(prepared["native_inputs"])
    tokenizer = Tokenizer()
    for trial in fixture[0]["trials"]:
        receipt = prepared["native_inputs"][trial["trial_id"]]
        text = tokenizer.apply_chat_template(trial["messages"], tokenize=False, add_generation_prompt=True)
        assert receipt["rendered_text"] == text
        assert receipt["prompt_token_ids"] == tokenizer.encode(text)
        assert receipt["intended_eos_token_id"] == 1
        assert receipt["assistant_probe_token_ids"] == encoded_answer(trial)
    assert not fixture[1]["output_root"].exists() and a._PROCESS_CONSUMED is False



@pytest.mark.parametrize("change", ["template", "rendering", "boundary", "decode", "assistant_eos", "native_eos", "context"])
def test_native_preflight_drift_stops_without_consumption(fixture, change):
    class BadTokenizer(Tokenizer):
        eos_token_id = 99 if change == "native_eos" else 1

        def get_chat_template(self):
            return super().get_chat_template() + (" changed" if change == "template" else "")

        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if change == "boundary" and text.endswith('"}'):
                result[0] += 1
            return result

        def decode(self, tokens, **kwargs):
            return super().decode(tokens, **kwargs) + (" altered" if change == "decode" else "")

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            if change == "rendering" and tokenize and kwargs["add_generation_prompt"]:
                result[0] += 1
            if change == "assistant_eos" and not kwargs["add_generation_prompt"]:
                result = result + [1] if tokenize else result + "<EOS>"
            if change == "context":
                result = result + [ord(" ") + 10] * 5000 if tokenize else result + " " * 5000
            return result

    fixture[1]["tokenizer_loader"] = lambda config: BadTokenizer()
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture[1]["output_root"].exists() and a._PROCESS_CONSUMED is False



def make_records(plan, assignment):
    records = []
    for trial in plan["trials"]:
        outcome = assignment(trial)
        if outcome is None:
            result = None
        elif outcome == "infrastructure":
            result = {"status": "infrastructure_failed"}
        else:
            answer = trial["selector"] if outcome == "label" else trial["selected_answer"] if outcome else trial["unselected_answer"]
            tokens = Tokenizer().encode(json.dumps({"answer": answer}, separators=(",", ":"))) + [1]
            result = {"status": "completed", "score": a.score_tokens(Tokenizer(), tokens, trial, [1, 2])}
        records.append(a._record(trial, result, attempted=outcome is not None))
    return records



def test_deeply_nested_decoded_json_is_behavioral_format_failure(fixture):
    class NestedTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return "[" * 2000 + "0" + "]" * 2000

    score = a.score_tokens(NestedTokenizer(), [30, 1], fixture[0]["trials"][0], [1, 2])
    assert score["category"] == "other_or_format" and not score["format_valid"]
    assert score["eos_valid"] and not score["strict_correct"]



def execute(fixture, *, fail_at=None, interrupt_at=None, signal_number=signal.SIGTERM,
            loader_error=None, response=None):
    plan, args = fixture
    calls, loaded = [], []

    def generate(runtime, prepared):
        index = len(calls)
        trial = plan["trials"][index]
        calls.append(trial["trial_id"])
        expected = Tokenizer().apply_chat_template(trial["messages"], tokenize=False, add_generation_prompt=True)
        assert prepared["rendered_text"] == expected
        assert prepared["prompt_token_ids"] == Tokenizer().encode(expected)
        if index == interrupt_at:
            if signal_number is None:
                raise KeyboardInterrupt
            if signal_number == signal.SIGALRM:
                a._deadline(signal_number, None)
            a._interrupt(signal_number, None)
        if index == fail_at:
            raise RuntimeError("synthetic confidential exception must not escape")
        return response(trial) if response else encoded_answer(trial)

    def load(config, engine):
        loaded.append(True)
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        if loader_error:
            raise loader_error
        return types.SimpleNamespace(tokenizer=Tokenizer(), eos_ids=[1, 2], device="cpu")

    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    aggregate = a.run_plan(**args, reference_loader=lambda path: reference, generation_runner=generate)
    assert len(loaded) == 1
    return aggregate, calls



def test_infrastructure_failure_is_unresolved_and_never_retried(fixture):
    aggregate, calls = execute(fixture, fail_at=2)
    assert len(calls) == len(set(calls)) == 64
    assert aggregate["status"] == "incomplete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert aggregate["overall"]["categories"]["infrastructure"] == 1
    assert aggregate["overall"]["strict_accuracy"] == {
        "point": None, "lower": 63 / 64, "upper": 1., "planned_outcomes": 64, "resolved_outcomes": 63}
    row = a.export_run(**fixture[1])["records"][2]
    assert row["strict_correct"] is None and row["selected_label"] is None and row["attempted"] is True
    result = json.loads((fixture[1]["output_root"] / "trials" / calls[2] / "result.json").read_text())
    assert "confidential" not in json.dumps(result)



@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_signal_deadline_and_keyboard_interrupt_preserve_coverage(fixture, number):
    aggregate, calls = execute(fixture, interrupt_at=3, signal_number=number)
    assert len(calls) == 4
    assert aggregate["coverage"] == {"attempted": 4, "completed": 3, "infrastructure_failed": 0,
                                     "interrupted": 1, "unattempted": 60, "missing": 61}
    receipt = json.loads((fixture[1]["output_root"] / "execution-finished.json").read_text())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {
        "signal_number": int(number) if number is not None else None,
        "signal_name": signal.Signals(number).name if number is not None else None,
        "pid": os.getpid(), "exception_type": "KeyboardInterrupt" if number is None else
        "A173Deadline" if number == signal.SIGALRM else "A173Interrupted"}
    exported = a.export_run(**fixture[1])
    assert exported["aggregate"] == aggregate
    assert exported["records"][3]["attempted"] is True and exported["records"][4]["attempted"] is False
    assert exported["records"][3]["category"] == exported["records"][4]["category"] == "missing"



def test_signal_context_restores_handlers_and_uses_two_hour_limit():
    old = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with a.bounded_signals():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 7199 < remaining <= 7200 and interval == 0
        assert signal.getsignal(signal.SIGALRM) is a._deadline
        assert signal.getsignal(signal.SIGTERM) is a._interrupt
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {number: signal.getsignal(number) for number in old} == old



def test_loader_failure_consumes_one_shot_and_cannot_change_output_path(fixture, monkeypatch):
    aggregate, calls = execute(fixture, loader_error=RuntimeError("synthetic loader failure"))
    assert calls == [] and aggregate["execution_status"] == "failed"
    assert aggregate["coverage"]["unattempted"] == aggregate["coverage"]["missing"] == 64
    original_root = fixture[1]["output_root"]
    fixture[1]["output_root"] = original_root.parent / "different-output"
    with pytest.raises(ValueError, match="fresh_process"):
        execute(fixture)
    fixture[1]["output_root"] = original_root
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        execute(fixture)



def test_system_exit_is_not_a_recoverable_evaluation_error(fixture):
    with pytest.raises(SystemExit):
        execute(fixture, loader_error=SystemExit(7))
    root = fixture[1]["output_root"]
    assert (root / "one-shot.json").exists()
    assert json.loads((root / "execution-finished.json").read_text())["status"] == "failed"
    assert a.export_run(**fixture[1])["aggregate"]["coverage"]["unattempted"] == 64



def test_exclusive_lock_prevents_execution(fixture):
    import fcntl
    root = fixture[1]["output_root"]
    root.mkdir()
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[1])



def rebind_model_metadata(fixture, monkeypatch, **changes):
    """Build a new synthetic bound plan, never relax production metadata guards."""
    _, args = fixture
    config = json.loads(args["config_path"].read_text())
    path = Path(config["model_path"]) / "config.json"
    metadata = json.loads(path.read_text())
    metadata.update(changes)
    path.write_bytes(a.tasks.canonical(metadata))
    config["model_files_sha256"]["config.json"] = a.native.engine.file_digest(path)
    args["config_path"].write_bytes(a.tasks.canonical(config))
    args["config_sha256"] = a.native.engine.file_digest(args["config_path"])
    monkeypatch.setattr(a, "CONFIG_SHA", args["config_sha256"])
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    plan = a.compile_plan(args["config_sha256"], a.native.engine.file_digest(args["protocol_path"]),
                          a.native.engine.file_digest(Path(__file__)))
    args["plan_path"].write_bytes(a.tasks.canonical(plan))
    args["plan_sha256"] = a.native.engine.file_digest(args["plan_path"])
    return plan, args



def test_model_context_limit_includes_all_64_generation_tokens(fixture, monkeypatch):
    largest = max(len(value["prompt_token_ids"]) for value in prepare(fixture)["native_inputs"].values())
    exact = rebind_model_metadata(fixture, monkeypatch, max_position_embeddings=largest + 64)
    native = prepare(exact)
    assert all(value["model_context_limit"] == largest + 64 for value in native["native_inputs"].values())
    too_short = rebind_model_metadata(exact, monkeypatch, max_position_embeddings=largest + 63)
    with pytest.raises(ValueError, match="generation_context_fit"):
        prepare(too_short)
    assert a._PROCESS_CONSUMED is False



def test_native_vocabulary_bound_is_not_inferred_from_decoder_acceptance(fixture, monkeypatch):
    native = prepare(fixture)["native_inputs"]
    highest = max(token for value in native.values() for token in value["prompt_token_ids"] + value["assistant_probe_token_ids"])
    exact = rebind_model_metadata(fixture, monkeypatch, vocab_size=highest + 1)
    assert len(prepare(exact)["native_inputs"]) == 64
    too_small = rebind_model_metadata(exact, monkeypatch, vocab_size=highest)
    with pytest.raises(ValueError, match="native_token_vocabulary"):
        prepare(too_small)



def test_model_metadata_content_must_match_bound_hash(fixture):
    config = json.loads(fixture[1]["config_path"].read_text())
    (Path(config["model_path"]) / "config.json").write_text('{"max_position_embeddings":9999,"vocab_size":512}')
    with pytest.raises(ValueError, match="model_config_binding"):
        prepare(fixture)



def test_out_of_vocabulary_injected_output_is_infrastructure_and_no_retry(fixture):
    target = fixture[0]["trials"][2]["trial_id"]

    def response(trial):
        return [512, 1] if trial["trial_id"] == target else encoded_answer(trial)

    aggregate, calls = execute(fixture, response=response)
    assert len(calls) == len(set(calls)) == 64
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert a.export_run(**fixture[1])["records"][2]["category"] == "infrastructure"



@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_preflight_interrupt_preserves_startup_receipt_and_consumes_claim(fixture, monkeypatch, number):
    def interrupted(*args):
        if number is None:
            raise KeyboardInterrupt
        if number == signal.SIGALRM:
            a._deadline(number, None)
        a._interrupt(number, None)

    monkeypatch.setattr(a, "_inputs", interrupted)
    exception = KeyboardInterrupt if number is None else a.A173Deadline if number == signal.SIGALRM else a.A173Interrupted
    with pytest.raises(exception):
        a.run_plan(**fixture[1])
    root = fixture[1]["output_root"]
    startup = json.loads((root / "startup-attempt.json").read_text())
    failed = json.loads((root / "startup-failure.json").read_text())
    assert failed["startup_attempt_sha256"] == a.object_sha(startup)
    assert failed["target_model_calls"] == 0 and not (root / "one-shot.json").exists()
    assert failed["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert failed["interruption"]["signal_number"] == (int(number) if number is not None else None)
    assert failed["interruption"]["signal_name"] == (signal.Signals(number).name if number is not None else None)
    assert failed["interruption"]["pid"] == os.getpid()
    assert a._PROCESS_CONSUMED is True
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**fixture[1])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1])



def test_preload_memory_failure_records_zero_calls_and_consumes_startup(fixture, monkeypatch):
    calls = []

    def forbidden(*args):
        calls.append(True)
        pytest.fail("model loader called below the 48 GiB memory gate")

    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - 1, load_cpu_reference=forbidden)
    with pytest.raises(ValueError, match="preload_memory_gate"):
        a.run_plan(**fixture[1], reference_loader=lambda path: reference)
    root = fixture[1]["output_root"]
    assert calls == [] and not (root / "one-shot.json").exists()
    failed = json.loads((root / "startup-failure.json").read_text())
    assert failed["target_model_calls"] == 0 and failed["status"] == "failed" and failed["interruption"] is None
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1], reference_loader=lambda path: reference)



def test_cli_exposes_only_safe_status_coverage_and_private_log(fixture, monkeypatch, capsys):
    def fake_run(**kwargs):
        print("synthetic private token and loader output")
        return {"status": "complete", "execution_status": "finished_schedule",
                "coverage": {"attempted": 64, "completed": 64}, "response_text": "synthetic private response"}

    monkeypatch.setattr(a, "run_plan", fake_run)
    args = fixture[1]
    argv = []
    for flag, key in (("plan", "plan_path"), ("plan-sha256", "plan_sha256"), ("config", "config_path"),
                      ("config-sha256", "config_sha256"), ("protocol", "protocol_path"),
                      ("reference-script", "reference_script"), ("output-root", "output_root")):
        argv.extend(["--" + flag, str(args[key])])
    previous = os.umask(0o077)
    try:
        assert a.main(argv) == 0
    finally:
        os.umask(previous)
    captured = capsys.readouterr()
    assert set(json.loads(captured.out)) == {"status", "execution_status", "coverage"}
    assert "synthetic private" not in captured.out + captured.err
    log = args["output_root"] / "execution.log"
    assert "synthetic private" in log.read_text() and log.stat().st_mode & 0o777 == 0o600



def test_exact_factorial_stimuli_and_counterbalanced_schedule(fixture):
    plan, _ = fixture
    trials = plan['trials']
    assert tuple(a.WORLDS) == EXPECTED_WORLDS
    assert tuple(a.BASE_SCHEDULE) == EXPECTED_BASE
    assert len(trials) == len({trial['trial_id'] for trial in trials}) == 64
    assert [trial['sequence_index'] for trial in trials] == list(range(64))
    positions = []
    for world, (value_a, value_b, order) in enumerate(EXPECTED_WORLDS):
        cells = [trial for trial in trials if trial['world_index'] == world]
        shift = 8 * (world // 2)
        expected = EXPECTED_BASE[shift:] + EXPECTED_BASE[:shift]
        if world % 2:
            expected = expected[::-1]
        actual = tuple(tuple(trial[key] for key in (*AXES, 'selector')) for trial in cells)
        assert actual == expected
        positions.append({cell: index for index, cell in enumerate(actual)})
        for trial in cells:
            prefix = 'a173_' if trial['value_form'] == 'study_prefixed' else ''
            mapping = {'A': prefix + value_a, 'B': prefix + value_b}
            expected_user = json.dumps({label: mapping[label] for label in order}, separators=(',', ':'))
            if trial['mapping_heading'] == 'on':
                expected_user = 'Mapping:\n' + expected_user
            expected_system = (LEGACY if trial['system_wording'] == 'legacy' else STANDARD).format(selector=trial['selector'])
            assert trial['messages'] == [{'role': 'system', 'content': expected_system},
                                         {'role': 'user', 'content': expected_user}]
            assert trial['selected_answer'] == mapping[trial['selector']]
            assert trial['unselected_answer'] == mapping['B' if trial['selector'] == 'A' else 'A']
            assert trial['presentation_order'] == order and trial['world_id'] == f'w{world:02d}'
            assert 'context' not in trial and 'wording' not in trial
        for heading, form in itertools.product(LEVELS[1], LEVELS[2]):
            selected = [cell for cell in cells if cell['mapping_heading'] == heading and cell['value_form'] == form]
            assert len({cell['messages'][1]['content'] for cell in selected}) == 1
    for left, right in itertools.combinations(EXPECTED_BASE, 2):
        assert sum(position[left] < position[right] for position in positions) == 2
    assert Counter(world[2] for world in EXPECTED_WORLDS) == {'AB': 2, 'BA': 2}
    assert Counter(world[2] for world in EXPECTED_WORLDS[::2]) == {'AB': 1, 'BA': 1}
    assert Counter(world[2] for world in EXPECTED_WORLDS[1::2]) == {'AB': 1, 'BA': 1}
    short_values = {value for world in EXPECTED_WORLDS for value in world[:2]}
    assert len(short_values) == 8 and short_values.isdisjoint({f's{i:02d}' for i in range(16)})
    assert all(trial['selected_answer'] not in ('A', 'B') for trial in trials)
    assert plan['factor_levels'] == dict(zip(AXES, map(list, LEVELS), strict=True))
    assert plan['primary_contrast_directions'] == {
        axis: {'positive': levels[1], 'negative': levels[0]} for axis, levels in zip(AXES, LEVELS, strict=True)}
    assert plan['planned_cells'] == plan['maximum_generation_calls'] == 64
    assert plan['retries'] == 0
    assert all(plan[key] is False for key in ('heldout_allowed', 'resume_allowed', 'baseline_gate',
                                              'likelihood_collected', 'prose_allowed', 'clarification_reminder_allowed'))


def test_protocol_literals_are_exact_and_pinned_helpers_are_identity_aliases():
    import re
    text = Path('plans/clean_mapping_factorial_a173.md').read_text()
    assert re.findall(r'```text\n([^\n]+)\n```', text) == [LEGACY, STANDARD]
    assert a.prepare_trial is a.previous.prepare_trial
    assert a.generate_tokens is a.previous.generate_tokens
    assert a._validate_tokens is a.previous._validate_tokens
    assert a.native.engine.file_digest(Path(a.previous.__file__)) == a.A171_SHA


@pytest.mark.parametrize('text,expected', [
    ('```', True), ('~~~', True), ('x```y', True), ('text\n~~~json', True),
    ('``', False), ('~~', False), ('` ` `', False), ('~ ~ ~', False),
    ('｀｀｀', False), ('～～～', False), ('', False), ('a173_s20', False),
])
def test_only_literal_ascii_fence_flag_is_added_to_pinned_primary_score(fixture, text, expected):
    trial = fixture[0]['trials'][0]
    tokens = Tokenizer().encode(text) + [1]
    original = a.previous.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    score = a.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    assert score == {**original, 'fence_marker': expected}
    assert set(score) - set(original) == {'fence_marker'}


def independent_contrast(records, axis):
    """Explicit positive-minus-negative averages over protocol-matched pairs."""
    axis_index = AXES.index(axis)
    other = [key for key in AXES if key != axis]
    other_levels = [LEVELS[AXES.index(key)] for key in other]
    differences = []
    for world, selector, first, second in itertools.product(range(4), ('A', 'B'), *other_levels):
        pair = {row[axis]: row['strict_correct'] for row in records
                if row['world_index'] == world and row['selector'] == selector
                and row[other[0]] == first and row[other[1]] == second}
        assert set(pair) == set(LEVELS[axis_index])
        differences.append(int(pair[LEVELS[axis_index][1]]) - int(pair[LEVELS[axis_index][0]]))
    assert len(differences) == 32
    return sum(differences) / 32


def test_three_main_effects_equal_independent_paired_arithmetic(fixture):
    def assignment(trial):
        system = trial['system_wording'] == 'a171_standard'
        heading = trial['mapping_heading'] == 'on'
        prefixed = trial['value_form'] == 'study_prefixed'
        return (3 * trial['world_index'] + (trial['selector'] == 'B') + system + 2 * heading + 4 * prefixed) % 7 < 3

    records = make_records(fixture[0], assignment)
    contrasts = a._contrasts(records)
    assert set(contrasts) == set(AXES)
    for axis in AXES:
        expected = independent_contrast(records, axis)
        assert contrasts[axis] == {'point': expected, 'lower': expected, 'upper': expected,
                                   'planned_outcomes': 64, 'resolved_outcomes': 64,
                                   'planned_pairs': 32, 'resolved_pairs': 32}


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('direction', [-1, 1])
def test_main_effect_extremes_and_signs(fixture, axis, direction):
    positive = LEVELS[AXES.index(axis)][1]
    records = make_records(fixture[0], lambda trial: (trial[axis] == positive) == (direction == 1))
    contrasts = a._contrasts(records)
    assert contrasts[axis]['point'] == contrasts[axis]['lower'] == contrasts[axis]['upper'] == direction
    assert all(contrasts[other]['point'] == 0 for other in AXES if other != axis)


def test_all_missing_main_effects_have_full_signed_bounds(fixture):
    records = make_records(fixture[0], lambda trial: None)
    for result in a._contrasts(records).values():
        assert result == {'point': None, 'lower': -1, 'upper': 1, 'planned_outcomes': 64,
                          'resolved_outcomes': 0, 'planned_pairs': 32, 'resolved_pairs': 0}
    assert all(row['fence_marker'] is None for row in records)


def test_partial_bounds_equal_exhaustive_assignment_extremes(fixture):
    def assignment(trial):
        index = trial['sequence_index']
        return None if index in (0, 19) else 'infrastructure' if index == 63 else index % 3 == 0

    records = make_records(fixture[0], assignment)
    unknown_indices = [i for i, row in enumerate(records) if row['strict_correct'] is None]
    assert unknown_indices == [0, 19, 63]
    for axis in AXES:
        all_values = []
        for outcomes in itertools.product((False, True), repeat=3):
            resolved = copy.deepcopy(records)
            for index, value in zip(unknown_indices, outcomes, strict=True):
                resolved[index]['strict_correct'] = value
            all_values.append(independent_contrast(resolved, axis))
        actual = a._contrasts(records)[axis]
        assert actual['point'] is None and actual['lower'] == min(all_values) and actual['upper'] == max(all_values)
        unknown_pairs = {(records[index]['world_index'], records[index]['selector'],
                          *(records[index][key] for key in AXES if key != axis)) for index in unknown_indices}
        assert actual['resolved_outcomes'] == 61 and actual['resolved_pairs'] == 32 - len(unknown_pairs)
    counts = a._counts(records)
    assert counts['categories']['infrastructure'] == 1 and counts['categories']['missing'] == 2
    assert counts['strict_accuracy']['point'] is None
    assert counts['indicators']['fence_marker']['missing'] == 3


@pytest.mark.parametrize('mutation', ['duplicate', 'count', 'integer', 'level', 'axis'])
def test_invalid_paired_cohort_rejected(fixture, mutation):
    records = make_records(fixture[0], lambda trial: True)
    axis = AXES[0]
    if mutation == 'duplicate':
        records[-1] = copy.deepcopy(records[0])
    elif mutation == 'count':
        records.pop()
    elif mutation == 'integer':
        records[0]['strict_correct'] = 1
    elif mutation == 'level':
        records[0][axis] = 'unknown'
    else:
        axis = 'context'
    with pytest.raises(ValueError):
        a._binary_bound(records, 'strict_correct', axis)


def test_complete_synthetic_schedule_all_eight_cells_and_export_privacy(fixture):
    aggregate, calls = execute(fixture)
    assert calls == [trial['trial_id'] for trial in fixture[0]['trials']]
    assert aggregate['status'] == 'complete' and aggregate['execution_status'] == 'finished_schedule'
    assert aggregate['coverage'] == {'attempted': 64, 'completed': 64, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 0, 'missing': 0}
    assert aggregate['overall']['categories']['correct'] == 64
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 0, 'false': 64, 'missing': 0}
    for system, heading, form in itertools.product(*LEVELS):
        cell = aggregate['by_condition'][system][heading][form]
        assert cell['planned_cells'] == cell['resolved_cells'] == cell['categories']['correct'] == 8
    assert set(aggregate['contrasts']) == {'strict_accuracy'}
    assert set(aggregate['contrasts']['strict_accuracy']) == set(AXES)
    assert all(value['point'] == 0 for value in aggregate['contrasts']['strict_accuracy'].values())
    assert aggregate['claim_boundaries']['interaction_estimands'] is False
    replay = a.export_run(**fixture[1])
    assert replay['aggregate'] == aggregate and len(replay['records']) == 64
    serialized = json.dumps(aggregate)
    assert all(secret not in serialized for secret in ('prompt_token_ids', 'response_text', 'generated_token_ids'))
    assert all(trial['trial_id'] not in serialized for trial in fixture[0]['trials'])
    assert (fixture[1]['output_root'] / 'native-inputs.json').stat().st_mode & 0o777 == 0o600


def test_caps_and_fences_never_gate_or_rescue_accuracy(fixture):
    marker = Tokenizer().encode('```')

    def response(trial):
        if trial['system_wording'] == 'legacy':
            return marker + [ord('x') + 10] * 61
        return encoded_answer(trial, field='selector')

    aggregate, calls = execute(fixture, response=response)
    assert len(calls) == 64 and aggregate['coverage']['completed'] == 64
    assert aggregate['overall']['categories']['cap'] == 32
    assert aggregate['overall']['categories']['selected_label'] == 32
    assert aggregate['overall']['strict_accuracy']['point'] == 0
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 32, 'false': 32, 'missing': 0}
    assert all(value['point'] == 0 for value in aggregate['contrasts']['strict_accuracy'].values())


def test_infrastructure_keeps_fence_unknown_and_all_primary_points_null(fixture):
    aggregate, _ = execute(fixture, fail_at=0)
    assert all(value['point'] is None for value in aggregate['contrasts']['strict_accuracy'].values())
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 0, 'false': 63, 'missing': 1}
    assert a.export_run(**fixture[1])['records'][0]['fence_marker'] is None


@pytest.mark.parametrize('tamper', ['native', 'attempt', 'tokens', 'score', 'fence', 'fence_integer',
                                   'orphan', 'order', 'receipt', 'signal', 'records', 'aggregate', 'numeric_bool'])
def test_export_rejects_tampered_evidence(fixture, tamper):
    execute(fixture)
    plan, args = fixture
    root = args['output_root']
    folder = root / 'trials' / plan['trials'][1]['trial_id']
    if tamper == 'orphan':
        (root / 'trials' / 'unplanned').mkdir()
    elif tamper == 'order':
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
    else:
        path = {'native': root / 'native-inputs.json', 'attempt': folder / 'attempt.json',
                'tokens': folder / 'result.json', 'score': folder / 'result.json',
                'fence': folder / 'result.json', 'fence_integer': folder / 'result.json',
                'receipt': root / 'execution-finished.json', 'signal': root / 'execution-finished.json',
                'records': root / 'records.json', 'aggregate': root / 'analysis.json',
                'numeric_bool': root / 'analysis.json'}[tamper]
        value = json.loads(path.read_text())
        if tamper == 'native':
            value[plan['trials'][0]['trial_id']]['model_vocab_size'] += 1
        elif tamper == 'attempt':
            value['sequence_index'] += 1
        elif tamper == 'tokens':
            value['generated_token_ids'][0] += 1
        elif tamper == 'score':
            value['score']['strict_correct'] = False
        elif tamper == 'fence':
            value['score']['fence_marker'] = True
        elif tamper == 'fence_integer':
            value['score']['fence_marker'] = 0
        elif tamper == 'receipt':
            value['run_sha256'] = '0' * 64
        elif tamper == 'signal':
            value['interruption'] = {'signal_number': 15, 'signal_name': 'SIGTERM', 'pid': os.getpid(),
                                     'exception_type': 'A173Interrupted'}
        elif tamper == 'records':
            value[0]['category'] = 'other_mapped_value'
        elif tamper == 'numeric_bool':
            value['overall']['strict_accuracy']['point'] = True
        else:
            value['contrasts']['strict_accuracy']['system_wording']['point'] = .5
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**args)


def test_timer_entry_failure_is_durable_and_cannot_restart(fixture, monkeypatch):
    @contextlib.contextmanager
    def failed_timer():
        raise RuntimeError('synthetic timer entry failure')
        yield

    monkeypatch.setattr(a, 'bounded_signals', failed_timer)
    with pytest.raises(RuntimeError, match='timer entry'):
        a.run_plan(**fixture[1])
    root = fixture[1]['output_root']
    failed = json.loads((root / 'startup-failure.json').read_text())
    startup = json.loads((root / 'startup-attempt.json').read_text())
    assert failed['startup_attempt_sha256'] == a.object_sha(startup)
    assert failed['target_model_calls'] == 0 and failed['status'] == 'failed'
    assert not (root / 'run.json').exists() and a._PROCESS_CONSUMED
    monkeypatch.setattr(a, '_PROCESS_CONSUMED', False)
    with pytest.raises(ValueError, match='consumed_run'):
        a.run_plan(**fixture[1])


def test_signal_after_result_publication_preserves_durable_completed_count(fixture, monkeypatch):
    original_write = a.native.engine.write_private
    triggered = []

    def interrupted_write(path, value):
        original_write(path, value)
        if Path(path).name == 'result.json' and not triggered:
            triggered.append(True)
            a._interrupt(signal.SIGTERM, None)

    monkeypatch.setattr(a.native.engine, 'write_private', interrupted_write)
    aggregate, calls = execute(fixture)
    assert len(calls) == 1 and aggregate['execution_status'] == 'interrupted'
    assert aggregate['coverage'] == {'attempted': 1, 'completed': 1, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 63, 'missing': 63}
    assert a.export_run(**fixture[1])['aggregate'] == aggregate
