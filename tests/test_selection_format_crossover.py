"""A176 synthetic qualification; no target tokenizer, weights or parent raw reads.

Independent literal templates, six-condition schedule and five contrast formulas
qualify the instrument alongside inherited synthetic lifecycle guards.
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

from lexical_prompt_study import selection_format_crossover as a


EXPECTED_WORLDS = (("s60", "s67", "AB"), ("s64", "s61", "BA"),
                   ("s63", "s66", "BA"), ("s65", "s62", "AB"))
AXES = ("selection_block", "format_request")
LEVELS = (("legacy", "a171_standard"), ("legacy", "a171_standard"))
EXPECTED_CONDITIONS = (
    ("normalized_legacy_legacy", "normalized", "legacy", "legacy"),
    ("normalized_legacy_standard", "normalized", "legacy", "a171_standard"),
    ("normalized_standard_legacy", "normalized", "a171_standard", "legacy"),
    ("normalized_standard_standard", "normalized", "a171_standard", "a171_standard"),
    ("original_legacy", "original", "legacy", "legacy"),
    ("original_standard", "original", "a171_standard", "a171_standard"),
)
EXPECTED_BASE = tuple(itertools.product((condition[0] for condition in EXPECTED_CONDITIONS), ("A", "B")))
LEGACY_SELECTION = ('The authoritative query selector is {selector}. The user supplies two explicit answer values labeled A and B. '
                    'Copy the value with the authoritative label.')
STANDARD_SELECTION = "Select the entry labeled {selector} from the user's mapping."
NORMALIZED_LEGACY_FORMAT = ('Output exactly one JSON object with the single field "answer" containing '
                            'the mapped value of the entry labeled {selector} as a string.')
NORMALIZED_STANDARD_FORMAT = ('Return the mapped value of the entry labeled {selector} as exactly one JSON object '
                              'with the single key "answer" and a string value.')
ORIGINAL_LEGACY_FORMAT = 'Output exactly one JSON object with the single field "answer" containing that value as a string.'
ORIGINAL_STANDARD_FORMAT = ('Return that entry\'s mapped value as exactly one JSON object with the single key "answer" '
                            'and a string value.')
CLOSING = 'Do not transform the value or include other fields, explanations, or Markdown fences.'


class Tokenizer:
    """Character coding with explicit EOS IDs, independent of target tokenization."""

    eos_token_id, pad_token_id, bos_token_id = 1, None, 0

    def __len__(self):
        return 512

    def get_chat_template(self):
        return "a176 public synthetic native template"

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
        "max_prompt_tokens": 4048, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10,
    }
    config_path = tmp_path / "config.json"
    protocol = tmp_path / "protocol.md"
    reference = tmp_path / "reference.py"
    config_path.write_bytes(a.tasks.canonical(config))
    protocol.write_text("Public synthetic A176 protocol fixture")
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
    assert prepared["planned_cells"] == prepared["maximum_generation_calls"] == 48
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
    assert len(calls) == len(set(calls)) == 48
    assert aggregate["status"] == "incomplete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert aggregate["overall"]["categories"]["infrastructure"] == 1
    assert aggregate["overall"]["strict_accuracy"] == {
        "point": None, "lower": 47 / 48, "upper": 1., "planned_outcomes": 48, "resolved_outcomes": 47}
    row = a.export_run(**fixture[1])["records"][2]
    assert row["strict_correct"] is None and row["selected_label"] is None and row["attempted"] is True
    result = json.loads((fixture[1]["output_root"] / "trials" / calls[2] / "result.json").read_text())
    assert "confidential" not in json.dumps(result)



@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_signal_deadline_and_keyboard_interrupt_preserve_coverage(fixture, number):
    aggregate, calls = execute(fixture, interrupt_at=3, signal_number=number)
    assert len(calls) == 4
    assert aggregate["coverage"] == {"attempted": 4, "completed": 3, "infrastructure_failed": 0,
                                     "interrupted": 1, "unattempted": 44, "missing": 45}
    receipt = json.loads((fixture[1]["output_root"] / "execution-finished.json").read_text())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {
        "signal_number": int(number) if number is not None else None,
        "signal_name": signal.Signals(number).name if number is not None else None,
        "pid": os.getpid(), "exception_type": "KeyboardInterrupt" if number is None else
        "A176Deadline" if number == signal.SIGALRM else "A176Interrupted"}
    exported = a.export_run(**fixture[1])
    assert exported["aggregate"] == aggregate
    assert exported["records"][3]["attempted"] is True and exported["records"][4]["attempted"] is False
    assert exported["records"][3]["category"] == exported["records"][4]["category"] == "missing"



def test_signal_context_restores_handlers_and_uses_one_hour_limit():
    old = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with a.bounded_signals():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 3599 < remaining <= 3600 and interval == 0
        assert signal.getsignal(signal.SIGALRM) is a._deadline
        assert signal.getsignal(signal.SIGTERM) is a._interrupt
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {number: signal.getsignal(number) for number in old} == old



def test_loader_failure_consumes_one_shot_and_cannot_change_output_path(fixture, monkeypatch):
    aggregate, calls = execute(fixture, loader_error=RuntimeError("synthetic loader failure"))
    assert calls == [] and aggregate["execution_status"] == "failed"
    assert aggregate["coverage"]["unattempted"] == aggregate["coverage"]["missing"] == 48
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
    assert a.export_run(**fixture[1])["aggregate"]["coverage"]["unattempted"] == 48



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
    assert len(prepare(exact)["native_inputs"]) == 48
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
    assert len(calls) == len(set(calls)) == 48
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
    exception = KeyboardInterrupt if number is None else a.A176Deadline if number == signal.SIGALRM else a.A176Interrupted
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
                "coverage": {"attempted": 48, "completed": 48}, "response_text": "synthetic private response"}

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



def test_exact_templates_conditions_and_counterbalanced_schedule(fixture):
    plan, _ = fixture
    trials = plan['trials']
    assert tuple(a.WORLDS) == EXPECTED_WORLDS
    assert tuple(a.CONDITIONS) == EXPECTED_CONDITIONS
    assert len(trials) == len({trial['trial_id'] for trial in trials}) == 48
    assert [trial['sequence_index'] for trial in trials] == list(range(48))
    positions = []
    for world, (value_a, value_b, order) in enumerate(EXPECTED_WORLDS):
        cells = [trial for trial in trials if trial['world_index'] == world]
        shift = 6 * (world // 2)
        expected = EXPECTED_BASE[shift:] + EXPECTED_BASE[:shift]
        if world % 2:
            expected = expected[::-1]
        actual = tuple((trial['condition'], trial['selector']) for trial in cells)
        assert actual == expected
        positions.append({cell: index for index, cell in enumerate(actual)})
        mapping = {'A': value_a, 'B': value_b}
        user = json.dumps({key: mapping[key] for key in order}, separators=(',', ':'))
        for trial in cells:
            condition = next(value for value in EXPECTED_CONDITIONS if value[0] == trial['condition'])
            assert tuple(trial[key] for key in ('condition', 'construction', *AXES)) == condition
            selection = LEGACY_SELECTION if trial['selection_block'] == 'legacy' else STANDARD_SELECTION
            if trial['construction'] == 'normalized':
                format_request = NORMALIZED_LEGACY_FORMAT if trial['format_request'] == 'legacy' else NORMALIZED_STANDARD_FORMAT
            else:
                format_request = ORIGINAL_LEGACY_FORMAT if trial['format_request'] == 'legacy' else ORIGINAL_STANDARD_FORMAT
            system = ' '.join((selection.format(selector=trial['selector']),
                               format_request.format(selector=trial['selector']), CLOSING))
            assert trial['messages'] == [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
            assert trial['selected_answer'] == mapping[trial['selector']]
            assert trial['unselected_answer'] == mapping['B' if trial['selector'] == 'A' else 'A']
            assert trial['presentation_order'] == order and trial['world_id'] == f'w{world:02d}'
            if trial['construction'] == 'original':
                assert system == a.previous.system_instruction(trial['selection_block'], 'legacy', trial['selector'])
            else:
                assert f"mapped value of the entry labeled {trial['selector']}" in system
                assert system != a.previous.system_instruction(trial['selection_block'], 'legacy', trial['selector'])
            assert not set(trial).intersection({'context', 'wording', 'mapping_heading', 'value_form', 'instruction_core', 'closing_constraint'})
        assert len({cell['messages'][1]['content'] for cell in cells}) == 1
    assert len(list(itertools.combinations(EXPECTED_BASE, 2))) == 66
    for left, right in itertools.combinations(EXPECTED_BASE, 2):
        assert sum(position[left] < position[right] for position in positions) == 2
    assert Counter(world[2] for world in EXPECTED_WORLDS) == {'AB': 2, 'BA': 2}
    assert Counter(world[2] for world in EXPECTED_WORLDS[::2]) == {'AB': 1, 'BA': 1}
    assert Counter(world[2] for world in EXPECTED_WORLDS[1::2]) == {'AB': 1, 'BA': 1}
    assert Counter(trial['condition'] for trial in trials) == {condition[0]: 8 for condition in EXPECTED_CONDITIONS}
    assert Counter(trial['construction'] for trial in trials) == {'normalized': 32, 'original': 16}
    values = {value for world in EXPECTED_WORLDS for value in world[:2]}
    assert len(values) == 8 and values.isdisjoint({f's{i:02d}' for i in range(60)})
    assert plan['planned_cells'] == plan['maximum_generation_calls'] == 48 and plan['max_new_tokens'] == 64
    assert plan['retries'] == 0
    assert all(plan[key] is False for key in ('heldout_allowed', 'resume_allowed', 'baseline_gate',
        'likelihood_collected', 'prose_allowed', 'clarification_reminder_allowed', 'mapping_heading_allowed', 'study_prefix_allowed'))


def test_protocol_templates_selector_multiplicity_and_identity_aliases():
    import re
    text = Path('plans/selection_format_crossover_a176.md').read_text()
    blocks = re.findall(r'```text\n([^\n]+)\n```', text)
    assert blocks == [LEGACY_SELECTION, STANDARD_SELECTION, NORMALIZED_LEGACY_FORMAT,
                      NORMALIZED_STANDARD_FORMAT, ORIGINAL_LEGACY_FORMAT, ORIGINAL_STANDARD_FORMAT, CLOSING]
    for selection in (LEGACY_SELECTION, STANDARD_SELECTION):
        for request in (NORMALIZED_LEGACY_FORMAT, NORMALIZED_STANDARD_FORMAT):
            assert (selection + ' ' + request + ' ' + CLOSING).count('{selector}') == 2
    assert (LEGACY_SELECTION + ORIGINAL_LEGACY_FORMAT).count('{selector}') == 1
    assert (STANDARD_SELECTION + ORIGINAL_STANDARD_FORMAT).count('{selector}') == 1
    assert a.prepare_trial is a.previous.prepare_trial
    assert a.generate_tokens is a.previous.generate_tokens
    assert a.score_tokens is a.previous.score_tokens
    assert a._validate_tokens is a.previous._validate_tokens
    assert a.native.engine.file_digest(Path(a.previous.__file__)) == a.A174_SHA


def independent_main(records, axis):
    other = next(key for key in AXES if key != axis)
    differences = []
    for world, selector, other_level in itertools.product(range(4), ('A', 'B'), ('legacy', 'a171_standard')):
        pair = {row[axis]: row['strict_correct'] for row in records
                if row['construction'] == 'normalized' and row['world_index'] == world
                and row['selector'] == selector and row[other] == other_level}
        assert set(pair) == {'legacy', 'a171_standard'}
        differences.append(int(pair['a171_standard']) - int(pair['legacy']))
    assert len(differences) == 16
    return sum(differences) / 16


def independent_interaction(records):
    values = []
    for world, selector in itertools.product(range(4), ('A', 'B')):
        cells = {(row['selection_block'], row['format_request']): row['strict_correct']
                 for row in records if row['construction'] == 'normalized'
                 and row['world_index'] == world and row['selector'] == selector}
        assert set(cells) == set(itertools.product(*LEVELS))
        values.append(int(cells['a171_standard', 'a171_standard']) + int(cells['legacy', 'legacy'])
                      - int(cells['a171_standard', 'legacy']) - int(cells['legacy', 'a171_standard']))
    return sum(values) / 8


def independent_normalization(records, level):
    norm = 'normalized_legacy_legacy' if level == 'legacy' else 'normalized_standard_standard'
    original = 'original_legacy' if level == 'legacy' else 'original_standard'
    values = []
    for world, selector in itertools.product(range(4), ('A', 'B')):
        cells = {row['condition']: row['strict_correct'] for row in records
                 if row['world_index'] == world and row['selector'] == selector
                 and row['condition'] in (norm, original)}
        assert set(cells) == {norm, original}
        values.append(int(cells[norm]) - int(cells[original]))
    return sum(values) / 8


def all_contrasts(records):
    return {**a._contrasts(records), **a._secondary_contrasts(records)}


def independent_all(records):
    return {**{axis: independent_main(records, axis) for axis in AXES},
            'selection_by_format_interaction': independent_interaction(records),
            **{'normalization_' + level: independent_normalization(records, level)
               for level in ('legacy', 'a171_standard')}}


def required_rows(records, key):
    if key in AXES or key == 'selection_by_format_interaction':
        return [row for row in records if row['construction'] == 'normalized']
    if key == 'normalization_legacy':
        names = ('normalized_legacy_legacy', 'original_legacy')
    else:
        names = ('normalized_standard_standard', 'original_standard')
    return [row for row in records if row['condition'] in names]


def test_all_five_contrasts_match_independent_arithmetic_and_planned_sizes(fixture):
    def assignment(trial):
        condition_index = next(index for index, condition in enumerate(EXPECTED_CONDITIONS) if condition[0] == trial['condition'])
        return (3 * trial['world_index'] + (trial['selector'] == 'B') + condition_index) % 7 < 3

    records = make_records(fixture[0], assignment)
    observed, expected = all_contrasts(records), independent_all(records)
    assert set(observed) == set(expected)
    for key, point in expected.items():
        result = observed[key]
        assert result['point'] == result['lower'] == result['upper'] == point
        assert result['planned_outcomes'] == result['resolved_outcomes'] == len(required_rows(records, key))
        if key == 'selection_by_format_interaction':
            assert result['planned_quartets'] == result['resolved_quartets'] == 8
        else:
            assert result['planned_pairs'] == result['resolved_pairs'] == (16 if key in AXES else 8)


@pytest.mark.parametrize('axis', AXES)
@pytest.mark.parametrize('direction', [-1, 1])
def test_normalized_main_extrema_ignore_adversarial_original_controls(fixture, axis, direction):
    def assignment(trial):
        positive = trial[axis] == 'a171_standard'
        return positive == (direction == 1) if trial['construction'] == 'normalized' else positive != (direction == 1)

    records = make_records(fixture[0], assignment)
    contrasts = a._contrasts(records)
    assert contrasts[axis]['point'] == contrasts[axis]['lower'] == contrasts[axis]['upper'] == direction
    assert all(contrasts[other]['point'] == 0 for other in AXES if other != axis)
    assert a._secondary_contrasts(records)['selection_by_format_interaction']['point'] == 0


@pytest.mark.parametrize('direction', [-1, 1])
def test_normalized_interaction_extrema_ignore_controls(fixture, direction):
    def assignment(trial):
        return (trial['selection_block'] == trial['format_request']) == (direction == 1) if trial['construction'] == 'normalized' else direction == -1

    records = make_records(fixture[0], assignment)
    result = a._secondary_contrasts(records)['selection_by_format_interaction']
    assert result['point'] == result['lower'] == result['upper'] == 2 * direction
    assert all(value['point'] == 0 for value in a._contrasts(records).values())


@pytest.mark.parametrize('level', ('legacy', 'a171_standard'))
@pytest.mark.parametrize('direction', [-1, 1])
def test_normalization_effect_extremes_and_positive_direction(fixture, level, direction):
    norm = 'normalized_legacy_legacy' if level == 'legacy' else 'normalized_standard_standard'
    original = 'original_legacy' if level == 'legacy' else 'original_standard'

    def assignment(trial):
        return direction == 1 if trial['condition'] == norm else direction == -1 if trial['condition'] == original else False

    result = a._secondary_contrasts(make_records(fixture[0], assignment))['normalization_' + level]
    assert result == {'point': direction, 'lower': direction, 'upper': direction,
                      'planned_outcomes': 16, 'resolved_outcomes': 16, 'planned_pairs': 8, 'resolved_pairs': 8}


def test_all_original_control_outcomes_can_change_without_affecting_normalized_estimands(fixture):
    def records(original_value):
        return make_records(fixture[0], lambda trial: original_value if trial['construction'] == 'original'
                            else (trial['world_index'] + (trial['selector'] == 'B') + (trial['selection_block'] == 'legacy')) % 3 == 0)

    first, second = records(False), records(True)
    assert a._contrasts(first) == a._contrasts(second)
    assert a._secondary_contrasts(first)['selection_by_format_interaction'] == a._secondary_contrasts(second)['selection_by_format_interaction']
    assert a._secondary_contrasts(first)['normalization_legacy'] != a._secondary_contrasts(second)['normalization_legacy']


@pytest.mark.parametrize('missing_condition,unknown_keys', [
    ('original_legacy', {'normalization_legacy'}),
    ('original_standard', {'normalization_a171_standard'}),
    ('normalized_legacy_standard', {*AXES, 'selection_by_format_interaction'}),
    ('normalized_legacy_legacy', {*AXES, 'selection_by_format_interaction', 'normalization_legacy'}),
    ('normalized_standard_standard', {*AXES, 'selection_by_format_interaction', 'normalization_a171_standard'}),
])
def test_missingness_is_local_to_each_prespecified_required_cohort(fixture, missing_condition, unknown_keys):
    records = make_records(fixture[0], lambda trial: None if trial['condition'] == missing_condition
                           and trial['world_index'] == 0 and trial['selector'] == 'A' else True)
    observed = all_contrasts(records)
    assert {key for key, value in observed.items() if value['point'] is None} == unknown_keys
    for key, value in observed.items():
        planned = 16 if key.startswith('normalization_') else 32
        assert value['planned_outcomes'] == planned and value['resolved_outcomes'] == planned - (key in unknown_keys)
        if key not in unknown_keys:
            assert value['point'] == value['lower'] == value['upper'] == 0
    assert a._counts(records)['strict_accuracy']['point'] is None


def test_all_missing_bounds_use_only_each_planned_cohort(fixture):
    observed = all_contrasts(make_records(fixture[0], lambda trial: None))
    for key, value in observed.items():
        interaction = key == 'selection_by_format_interaction'
        assert value['point'] is None and value['lower'] == (-2 if interaction else -1) and value['upper'] == (2 if interaction else 1)
        assert value['resolved_outcomes'] == 0
        assert value['planned_outcomes'] == (16 if key.startswith('normalization_') else 32)
        if interaction:
            assert value['planned_quartets'] == 8 and value['resolved_quartets'] == 0
        else:
            assert value['planned_pairs'] == (16 if key in AXES else 8) and value['resolved_pairs'] == 0


def test_partial_bounds_match_exhaustive_independent_assignments(fixture):
    missing = {(0, 'A', 'normalized_legacy_legacy'), (1, 'B', 'normalized_standard_legacy'),
               (2, 'A', 'original_legacy'), (3, 'B', 'original_standard')}
    records = make_records(fixture[0], lambda trial: None if (trial['world_index'], trial['selector'], trial['condition']) in missing
                           else trial['sequence_index'] % 3 == 0)
    unknown = [index for index, row in enumerate(records) if row['strict_correct'] is None]
    assert len(unknown) == 4
    possible = {key: [] for key in all_contrasts(records)}
    for outcomes in itertools.product((False, True), repeat=4):
        resolved = copy.deepcopy(records)
        for index, value in zip(unknown, outcomes, strict=True):
            resolved[index]['strict_correct'] = value
        for key, value in independent_all(resolved).items():
            possible[key].append(value)
    for key, actual in all_contrasts(records).items():
        needed = required_rows(records, key)
        assert actual['point'] is None and actual['lower'] == min(possible[key]) and actual['upper'] == max(possible[key])
        assert actual['planned_outcomes'] == len(needed)
        assert actual['resolved_outcomes'] == sum(row['strict_correct'] is not None for row in needed)
        unknown_units = {(row['world_index'], row['selector'], *(row[other] for other in AXES if other != key))
                         if key in AXES else (row['world_index'], row['selector'])
                         for row in needed if row['strict_correct'] is None}
        if key == 'selection_by_format_interaction':
            assert actual['resolved_quartets'] == 8 - len(unknown_units)
        else:
            assert actual['resolved_pairs'] == actual['planned_pairs'] - len(unknown_units)


def test_complete_synthetic_schedule_six_groups_and_private_export(fixture):
    aggregate, calls = execute(fixture)
    assert calls == [trial['trial_id'] for trial in fixture[0]['trials']]
    assert aggregate['status'] == 'complete' and aggregate['execution_status'] == 'finished_schedule'
    assert aggregate['coverage'] == {'attempted': 48, 'completed': 48, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 0, 'missing': 0}
    assert aggregate['overall']['categories']['correct'] == 48
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 0, 'false': 48, 'missing': 0}
    assert set(aggregate['by_condition']) == {condition[0] for condition in EXPECTED_CONDITIONS}
    for cell in aggregate['by_condition'].values():
        assert cell['planned_cells'] == cell['resolved_cells'] == cell['categories']['correct'] == 8
    assert set(aggregate['contrasts']) == {'strict_accuracy'}
    assert set(aggregate['contrasts']['strict_accuracy']) == set(AXES)
    assert set(aggregate['secondary_contrasts']['strict_accuracy']) == {
        'selection_by_format_interaction', 'normalization_legacy', 'normalization_a171_standard'}
    assert all(value['point'] == 0 for value in aggregate['contrasts']['strict_accuracy'].values())
    assert all(value['point'] == 0 for value in aggregate['secondary_contrasts']['strict_accuracy'].values())
    replay = a.export_run(**fixture[1])
    assert replay['aggregate'] == aggregate and len(replay['records']) == 48
    serialized = json.dumps(aggregate)
    assert all(secret not in serialized for secret in ('prompt_token_ids', 'response_text', 'generated_token_ids'))
    assert all(trial['trial_id'] not in serialized for trial in fixture[0]['trials'])


def test_original_controls_and_caps_never_gate_or_rescue_accuracy(fixture):
    marker = Tokenizer().encode('```')

    def response(trial):
        return marker + [ord('x') + 10] * 61 if trial['construction'] == 'original' else encoded_answer(trial, field='selector')

    aggregate, calls = execute(fixture, response=response)
    assert len(calls) == 48 and aggregate['coverage']['completed'] == 48
    assert aggregate['overall']['categories']['cap'] == 16
    assert aggregate['overall']['categories']['selected_label'] == 32
    assert aggregate['overall']['strict_accuracy']['point'] == 0
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 16, 'false': 32, 'missing': 0}
    assert all(value['point'] == 0 for value in aggregate['contrasts']['strict_accuracy'].values())


def test_missing_original_control_at_runtime_does_not_null_normalized_points(fixture):
    index = next(t['sequence_index'] for t in fixture[0]['trials'] if t['condition'] == 'original_legacy')
    aggregate, calls = execute(fixture, fail_at=index)
    assert len(calls) == 48 and aggregate['overall']['strict_accuracy']['point'] is None
    assert all(value['point'] == 0 for value in aggregate['contrasts']['strict_accuracy'].values())
    secondary = aggregate['secondary_contrasts']['strict_accuracy']
    assert secondary['selection_by_format_interaction']['point'] == 0
    assert secondary['normalization_a171_standard']['point'] == 0
    assert secondary['normalization_legacy']['point'] is None
    assert aggregate['overall']['indicators']['fence_marker']['missing'] == 1
    assert a.export_run(**fixture[1])['records'][index]['fence_marker'] is None


@pytest.mark.parametrize('key', ['selection_by_format_interaction', 'normalization_legacy', 'normalization_a171_standard'])
def test_secondary_contrast_tamper_is_rejected(fixture, key):
    execute(fixture)
    path = fixture[1]['output_root'] / 'analysis.json'
    value = json.loads(path.read_text())
    value['secondary_contrasts']['strict_accuracy'][key]['point'] = 1
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])


@pytest.mark.parametrize('text,expected', [
    ('```', True), ('~~~', True), ('x```y', True), ('text\n~~~json', True),
    ('``', False), ('~~', False), ('` ` `', False), ('~ ~ ~', False),
    ('｀｀｀', False), ('～～～', False), ('', False), ('a176_s20', False),
])
def test_only_literal_ascii_fence_flag_is_added_to_pinned_primary_score(fixture, text, expected):
    trial = fixture[0]['trials'][0]
    tokens = Tokenizer().encode(text) + [1]
    original = a.previous.previous.previous.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    score = a.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    assert score == {**original, 'fence_marker': expected}
    assert set(score) - set(original) == {'fence_marker'}



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
                                     'exception_type': 'A176Interrupted'}
        elif tamper == 'records':
            value[0]['category'] = 'other_mapped_value'
        elif tamper == 'numeric_bool':
            value['overall']['strict_accuracy']['point'] = True
        else:
            value['contrasts']['strict_accuracy']['selection_block']['point'] = .5
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
                                     'interrupted': 0, 'unattempted': 47, 'missing': 47}
    assert a.export_run(**fixture[1])['aggregate'] == aggregate



def test_boolean_signal_number_cannot_impersonate_sighup(fixture):
    execute(fixture, interrupt_at=1, signal_number=signal.SIGHUP)
    path = fixture[1]['output_root'] / 'execution-finished.json'
    receipt = json.loads(path.read_text())
    assert receipt['interruption']['signal_number'] == 1
    receipt['interruption']['signal_number'] = True
    path.write_bytes(a.tasks.canonical(receipt))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])
