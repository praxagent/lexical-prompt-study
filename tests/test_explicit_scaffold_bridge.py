"""A178 synthetic qualification; no restricted materials or target-model reads.

Public stand-ins qualify byte bindings, the scaffold matrix, independently
computed contrasts and shared-clean missingness, plus inherited lifecycle guards.
"""

import contextlib
import copy
import itertools
import json
import os
import signal
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from lexical_prompt_study import explicit_scaffold_bridge as a


EXPECTED_WORLDS = (("s84", "s91", "AB"), ("s88", "s85", "BA"),
                   ("s87", "s90", "BA"), ("s89", "s86", "AB"))
EXPECTED_CONDITIONS = ("clean", "full_before", "full_after", "sham_before", "sham_after", "inert_before", "inert_after")
EXPECTED_BASE = tuple(itertools.product(EXPECTED_CONDITIONS, ("A", "B")))
SELECTION = "Select the entry labeled {selector} from the user's mapping."
FORMAT = ('Return the mapped value of the entry labeled {selector} as exactly one JSON object '
          'with the single key "answer" and a string value.')
CLOSING = 'Do not transform the value or include other fields, explanations, or Markdown fences.'
PUBLIC_MATERIALS = {'full': ' PUBLIC synthetic full.\n', 'sham': 'PUBLIC synthetic sham.\né',
                    'inert': 'PUBLIC synthetic inert.\n\nTrailing space '}
REQUIRED = {'full_minus_sham': ('full_before', 'full_after', 'sham_before', 'sham_after'),
            'inert_minus_clean': ('inert_before', 'inert_after', 'clean'),
            'full_minus_sham_before': ('full_before', 'sham_before'),
            'full_minus_sham_after': ('full_after', 'sham_after'),
            'inert_minus_clean_before': ('inert_before', 'clean'),
            'inert_minus_clean_after': ('inert_after', 'clean')}


class Tokenizer:
    """Character coding with explicit EOS IDs, independent of target tokenization."""

    eos_token_id, pad_token_id, bos_token_id = 1, None, 0

    def __len__(self):
        return 512

    def get_chat_template(self):
        return "a178 public synthetic native template"

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
    protocol.write_text("Public synthetic A178 protocol fixture")
    reference.write_text("Public synthetic reference binding fixture")
    monkeypatch.setattr(a, "CONFIG_SHA", digest(config_path))
    monkeypatch.setattr(a.native.engine, "file_digest", lambda path: a.REFERENCE_SHA if Path(path) == reference else digest(path))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    monkeypatch.setattr(a, "PRIVATE_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(a, "MATERIAL_RECEIPTS", {kind: {"sha256": a.tasks.sha(text.encode()), "bytes": len(text.encode())}
                                                  for kind, text in PUBLIC_MATERIALS.items()})
    plan = a.compile_plan(digest(config_path), digest(protocol), digest(Path(__file__)), PUBLIC_MATERIALS)
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
    premature_at_limit = [1] + body + [ord(" ") + 10] * (62 - len(body)) + [2]
    assert len(premature_at_limit) == 64
    score = a.score_tokens(Tokenizer(), premature_at_limit, trial, [1, 2])
    assert score["category"] == "other_or_format" and not score["eos_valid"] and not score["capped"]



def test_native_preparation_replays_every_public_cell_without_consumption(fixture):
    prepared = prepare(fixture)
    assert prepared["planned_cells"] == prepared["maximum_generation_calls"] == 56
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
    assert len(calls) == len(set(calls)) == 56
    assert aggregate["status"] == "incomplete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert aggregate["overall"]["categories"]["infrastructure"] == 1
    assert aggregate["overall"]["strict_accuracy"] == {
        "point": None, "lower": 55 / 56, "upper": 1., "planned_outcomes": 56, "resolved_outcomes": 55}
    row = a.export_run(**fixture[1])["records"][2]
    assert row["strict_correct"] is None and row["selected_label"] is None and row["attempted"] is True
    result = json.loads((fixture[1]["output_root"] / "trials" / calls[2] / "result.json").read_text())
    assert "confidential" not in json.dumps(result)



@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_signal_deadline_and_keyboard_interrupt_preserve_coverage(fixture, number):
    aggregate, calls = execute(fixture, interrupt_at=3, signal_number=number)
    assert len(calls) == 4
    assert aggregate["coverage"] == {"attempted": 4, "completed": 3, "infrastructure_failed": 0,
                                     "interrupted": 1, "unattempted": 52, "missing": 53}
    receipt = json.loads((fixture[1]["output_root"] / "execution-finished.json").read_text())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {
        "signal_number": int(number) if number is not None else None,
        "signal_name": signal.Signals(number).name if number is not None else None,
        "pid": os.getpid(), "exception_type": "KeyboardInterrupt" if number is None else
        "A178Deadline" if number == signal.SIGALRM else "A178Interrupted"}
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
    assert aggregate["coverage"]["unattempted"] == aggregate["coverage"]["missing"] == 56
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
    assert a.export_run(**fixture[1])["aggregate"]["coverage"]["unattempted"] == 56



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
    previous_plan, args = fixture
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
                          a.native.engine.file_digest(Path(__file__)), previous_plan["materials"])
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
    assert len(prepare(exact)["native_inputs"]) == 56
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
    assert len(calls) == len(set(calls)) == 56
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
    exception = KeyboardInterrupt if number is None else a.A178Deadline if number == signal.SIGALRM else a.A178Interrupted
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
                "coverage": {"attempted": 56, "completed": 56}, "response_text": "synthetic private response"}

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



def compile_with(fixture, materials):
    bindings = fixture[0]['bindings']
    return a.compile_plan(bindings['config_sha256'], bindings['protocol_sha256'], bindings['tests_sha256'], materials)


def test_exact_templates_material_composition_and_balanced_schedule(fixture):
    plan, _ = fixture
    trials = plan['trials']
    assert tuple(a.WORLDS) == EXPECTED_WORLDS and tuple(a.CONDITIONS) == EXPECTED_CONDITIONS
    assert len(trials) == len({trial['trial_id'] for trial in trials}) == 56
    assert [trial['sequence_index'] for trial in trials] == list(range(56))
    assert plan['materials'] == PUBLIC_MATERIALS
    assert plan['bindings']['materials_object_sha256'] == a.object_sha(PUBLIC_MATERIALS)
    positions = []
    for world, (value_a, value_b, order) in enumerate(EXPECTED_WORLDS):
        cells = [trial for trial in trials if trial['world_index'] == world]
        shift = 7 * (world // 2)
        expected = EXPECTED_BASE[shift:] + EXPECTED_BASE[:shift]
        if world % 2:
            expected = expected[::-1]
        actual = tuple((trial['condition'], trial['selector']) for trial in cells)
        assert actual == expected
        positions.append({cell: index for index, cell in enumerate(actual)})
        mapping = {'A': value_a, 'B': value_b}
        payload = json.dumps({key: mapping[key] for key in order}, separators=(',', ':'))
        for trial in cells:
            assert set(trial) == {'trial_id', 'sequence_index', 'world_index', 'world_id', 'presentation_order',
                                 'condition', 'scaffold_kind', 'placement', 'selector', 'selected_answer', 'unselected_answer', 'messages'}
            kind, placement = ('none', 'none') if trial['condition'] == 'clean' else trial['condition'].split('_')
            assert (trial['scaffold_kind'], trial['placement']) == (kind, placement)
            user = payload if placement == 'none' else PUBLIC_MATERIALS[kind] + '\n\n' + payload if placement == 'before' else payload + '\n\n' + PUBLIC_MATERIALS[kind]
            system = ' '.join((SELECTION, FORMAT, CLOSING)).format(selector=trial['selector'])
            assert system == a.previous.system_instruction('of_explicit_label', trial['selector'])
            assert trial['messages'] == [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
            assert trial['selected_answer'] == mapping[trial['selector']]
            assert trial['unselected_answer'] == mapping['B' if trial['selector'] == 'A' else 'A']
            assert trial['presentation_order'] == order and trial['world_id'] == f'w{world:02d}'
        for condition in EXPECTED_CONDITIONS:
            assert len({cell['messages'][1]['content'] for cell in cells if cell['condition'] == condition}) == 1
    assert len(list(itertools.combinations(EXPECTED_BASE, 2))) == 91
    for left, right in itertools.combinations(EXPECTED_BASE, 2):
        assert sum(position[left] < position[right] for position in positions) == 2
    assert Counter(world[2] for world in EXPECTED_WORLDS) == {'AB': 2, 'BA': 2}
    assert Counter(world[2] for world in EXPECTED_WORLDS[::2]) == {'AB': 1, 'BA': 1}
    assert Counter(world[2] for world in EXPECTED_WORLDS[1::2]) == {'AB': 1, 'BA': 1}
    assert Counter(trial['condition'] for trial in trials) == {condition: 8 for condition in EXPECTED_CONDITIONS}
    values = {value for world in EXPECTED_WORLDS for value in world[:2]}
    assert len(values) == 8 and values.isdisjoint({'A', 'B', *{f's{i:02d}' for i in range(84)}})
    assert plan['planned_cells'] == plan['maximum_generation_calls'] == 56 and plan['max_new_tokens'] == 64
    assert plan['retries'] == 0
    assert all(plan[key] is False for key in ('heldout_allowed', 'resume_allowed', 'baseline_gate',
        'likelihood_collected', 'clarification_reminder_allowed', 'mapping_heading_allowed', 'study_prefix_allowed'))


def test_protocol_three_templates_exact_inheritance_and_runtime_bound():
    import re
    text = Path('plans/explicit_scaffold_bridge_a178.md').read_text()
    blocks = re.findall(r'```text\n([^\n]+)\n```', text)
    assert blocks == [SELECTION, FORMAT, CLOSING] and all(block.isascii() for block in blocks)
    assert (SELECTION + ' ' + FORMAT).count('{selector}') == 2
    assert a.prepare_trial is a.previous.prepare_trial and a.generate_tokens is a.previous.generate_tokens
    assert a.score_tokens is a.previous.score_tokens and a._validate_tokens is a.previous._validate_tokens
    assert a.native.engine.file_digest(Path(a.previous.__file__)) == a.A177_SHA
    assert a.MAX_SECONDS == 7200 and a.KILL_GRACE_SECONDS == 60 and a.MIN_AVAILABLE_BYTES == 48 * 1024**3


@pytest.mark.parametrize('change', ['extra', 'missing', 'list', 'bytes', 'none', 'space', 'newline', 'normalize', 'swap'])
def test_material_input_rejects_key_type_and_exact_byte_drift(fixture, change):
    materials = dict(PUBLIC_MATERIALS)
    if change == 'extra':
        materials['replacement'] = 'public extra'
    elif change == 'missing':
        materials.pop('inert')
    elif change == 'list':
        materials = list(materials.values())
    elif change == 'bytes':
        materials['full'] = materials['full'].encode()
    elif change == 'none':
        materials['full'] = None
    elif change == 'space':
        materials['full'] = materials['full'].strip()
    elif change == 'newline':
        materials['sham'] = materials['sham'].replace('\n', '\r\n')
    elif change == 'normalize':
        materials['sham'] = materials['sham'].replace('é', 'e\u0301')
    else:
        materials['full'], materials['sham'] = materials['sham'], materials['full']
    with pytest.raises(ValueError):
        compile_with(fixture, materials)


def test_material_byte_lengths_are_utf8_and_plan_copies_bindings(fixture):
    plan = fixture[0]
    assert len(PUBLIC_MATERIALS['sham'].encode()) != len(PUBLIC_MATERIALS['sham'])
    assert plan['material_receipts']['sham']['bytes'] == len(PUBLIC_MATERIALS['sham'].encode())
    materials = dict(PUBLIC_MATERIALS)
    created = compile_with(fixture, materials)
    materials['full'] = 'changed after compile'
    assert created['materials']['full'] == PUBLIC_MATERIALS['full']
    created['material_receipts']['full']['bytes'] += 1
    assert a.MATERIAL_RECEIPTS['full']['bytes'] == len(PUBLIC_MATERIALS['full'].encode())


@pytest.mark.parametrize('change', ['source_hash', 'object_hash', 'receipt_hash', 'receipt_length', 'material', 'delimiter', 'kind', 'placement'])
def test_plan_material_and_composition_tampering_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == 'source_hash':
        plan['bindings']['material_source_sha256'] = '0' * 64
    elif change == 'object_hash':
        plan['bindings']['materials_object_sha256'] = '0' * 64
    elif change == 'receipt_hash':
        plan['material_receipts']['full']['sha256'] = '0' * 64
    elif change == 'receipt_length':
        plan['material_receipts']['full']['bytes'] += 1
    elif change == 'material':
        plan['materials']['full'] += ' '
    else:
        trial = next(t for t in plan['trials'] if t['condition'] == 'full_before')
        if change == 'delimiter':
            trial['messages'][1]['content'] = PUBLIC_MATERIALS['full'] + '\n' + json.dumps({'A': 's84', 'B': 's91'}, separators=(',', ':'))
        elif change == 'kind':
            trial['scaffold_kind'] = 'sham'
        else:
            trial['placement'] = 'after'
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def all_contrasts(records):
    return {**a._contrasts(records), **a._secondary_contrasts(records)}


def independent_all(records):
    cells = {(row['world_index'], row['selector'], row['condition']): row['strict_correct'] for row in records}
    assert len(cells) == 56
    totals = {key: Fraction(0) for key in REQUIRED}
    for world, selector in itertools.product(range(4), ('A', 'B')):
        values = {condition: cells[world, selector, condition] for condition in EXPECTED_CONDITIONS}
        assert all(type(value) is bool for value in values.values())
        for placement in ('before', 'after'):
            full_sham = int(values['full_' + placement]) - int(values['sham_' + placement])
            inert_clean = int(values['inert_' + placement]) - int(values['clean'])
            totals['full_minus_sham'] += Fraction(full_sham, 16)
            totals['inert_minus_clean'] += Fraction(inert_clean, 16)
            totals['full_minus_sham_' + placement] += Fraction(full_sham, 8)
            totals['inert_minus_clean_' + placement] += Fraction(inert_clean, 8)
    return {key: float(value) for key, value in totals.items()}


def planned(key):
    return (24, 'triples', 8) if key == 'inert_minus_clean' else (32, 'pairs', 16) if key == 'full_minus_sham' else (16, 'pairs', 8)


def required_rows(records, key):
    return [row for row in records if row['condition'] in REQUIRED[key]]


def test_all_six_contrasts_match_independent_paired_arithmetic(fixture):
    records = make_records(fixture[0], lambda trial:
        (3 * trial['world_index'] + (trial['selector'] == 'B') + EXPECTED_CONDITIONS.index(trial['condition'])) % 7 < 3)
    actual, expected = all_contrasts(records), independent_all(records)
    assert set(actual) == set(expected) == set(REQUIRED)
    for key, point in expected.items():
        size, units, count = planned(key)
        assert actual[key] == {'point': point, 'lower': point, 'upper': point,
                              'planned_outcomes': size, 'resolved_outcomes': size,
                              'planned_' + units: count, 'resolved_' + units: count}
    for key in ('full_minus_sham', 'inert_minus_clean'):
        assert actual[key]['point'] == (actual[key + '_before']['point'] + actual[key + '_after']['point']) / 2


@pytest.mark.parametrize('key', REQUIRED)
@pytest.mark.parametrize('direction', [-1, 1])
def test_extrema_and_sign_ignore_unrelated_infrastructure(fixture, key, direction):
    required = REQUIRED[key]
    positive = {condition for condition in required if condition.startswith('full' if key.startswith('full') else 'inert')}
    records = make_records(fixture[0], lambda trial: 'infrastructure' if trial['condition'] not in required
                           else (trial['condition'] in positive) == (direction == 1))
    actual = all_contrasts(records)[key]
    assert actual['point'] == actual['lower'] == actual['upper'] == direction
    assert actual['planned_outcomes'] == actual['resolved_outcomes'] == planned(key)[0]


@pytest.mark.parametrize('key', REQUIRED)
def test_excluded_condition_outcomes_and_missingness_cannot_change_contrast(fixture, key):
    outputs = []
    for excluded in (False, True, None, 'infrastructure'):
        records = make_records(fixture[0], lambda trial: excluded if trial['condition'] not in REQUIRED[key]
            else (trial['world_index'] + (trial['selector'] == 'B') + EXPECTED_CONDITIONS.index(trial['condition'])) % 3 == 0)
        outputs.append(all_contrasts(records)[key])
    assert all(value == outputs[0] for value in outputs) and outputs[0]['point'] is not None


@pytest.mark.parametrize('missing_condition', EXPECTED_CONDITIONS)
@pytest.mark.parametrize('unknown_value', [None, 'infrastructure'])
def test_one_unknown_has_local_coverage_and_fixed_denominators(fixture, missing_condition, unknown_value):
    records = make_records(fixture[0], lambda trial: unknown_value if trial['condition'] == missing_condition
                           and trial['world_index'] == 0 and trial['selector'] == 'A' else True)
    actual = all_contrasts(records)
    affected = {key for key, cohort in REQUIRED.items() if missing_condition in cohort}
    assert {key for key, value in actual.items() if value['point'] is None} == affected
    for key, value in actual.items():
        size, units, count = planned(key)
        assert value['planned_outcomes'] == size and value['resolved_outcomes'] == size - (key in affected)
        assert value['planned_' + units] == count and value['resolved_' + units] == count - (key in affected)
        if key not in affected:
            assert value['point'] == value['lower'] == value['upper'] == 0
    assert a._counts(records)['strict_accuracy']['point'] is None


@pytest.mark.parametrize('condition,width', [('clean', 1 / 8), ('inert_before', 1 / 16), ('inert_after', 1 / 16)])
def test_shared_clean_combined_coefficient_and_distinct_outcome_denominator(fixture, condition, width):
    records = make_records(fixture[0], lambda trial: None if trial['world_index'] == 0 and trial['selector'] == 'A'
                           and trial['condition'] == condition else False)
    value = all_contrasts(records)['inert_minus_clean']
    assert value == {'point': None, 'lower': -width if condition == 'clean' else 0,
                     'upper': 0 if condition == 'clean' else width, 'planned_outcomes': 24,
                     'resolved_outcomes': 23, 'planned_triples': 8, 'resolved_triples': 7}


def test_three_unknowns_in_one_triple_remove_only_one_resolved_triple(fixture):
    records = make_records(fixture[0], lambda trial: None if trial['world_index'] == 0 and trial['selector'] == 'A'
                           and trial['condition'] in REQUIRED['inert_minus_clean'] else False)
    assert all_contrasts(records)['inert_minus_clean'] == {'point': None, 'lower': -1 / 8, 'upper': 1 / 8,
        'planned_outcomes': 24, 'resolved_outcomes': 21, 'planned_triples': 8, 'resolved_triples': 7}


def test_all_missing_bounds_keep_unique_planned_outcomes(fixture):
    for key, value in all_contrasts(make_records(fixture[0], lambda trial: None)).items():
        size, units, count = planned(key)
        assert value == {'point': None, 'lower': -1, 'upper': 1, 'planned_outcomes': size, 'resolved_outcomes': 0,
                         'planned_' + units: count, 'resolved_' + units: 0}


def test_partial_sharp_bounds_match_exhaustive_independent_assignments(fixture):
    missing = {(0, 'A', 'full_before'), (1, 'B', 'sham_after'), (2, 'A', 'clean'),
               (2, 'A', 'inert_before'), (3, 'B', 'inert_after')}
    records = make_records(fixture[0], lambda trial: None if (trial['world_index'], trial['selector'], trial['condition']) in missing
                           else trial['sequence_index'] % 3 == 0)
    unknown = [index for index, row in enumerate(records) if row['strict_correct'] is None]
    assert len(unknown) == 5
    possible = {key: [] for key in REQUIRED}
    for outcomes in itertools.product((False, True), repeat=5):
        resolved = copy.deepcopy(records)
        for index, value in zip(unknown, outcomes, strict=True):
            resolved[index]['strict_correct'] = value
        for key, value in independent_all(resolved).items():
            possible[key].append(value)
    for key, actual in all_contrasts(records).items():
        needed = required_rows(records, key)
        size, units, count = planned(key)
        assert actual['point'] is None and actual['lower'] == min(possible[key]) and actual['upper'] == max(possible[key])
        assert actual['planned_outcomes'] == size and actual['resolved_outcomes'] == sum(row['strict_correct'] is not None for row in needed)
        unknown_units = {(row['world_index'], row['selector'], row['placement']) if key == 'full_minus_sham'
                         else (row['world_index'], row['selector']) for row in needed if row['strict_correct'] is None}
        assert actual['planned_' + units] == count and actual['resolved_' + units] == count - len(unknown_units)


def test_complete_synthetic_schedule_seven_groups_and_private_export(fixture):
    aggregate, calls = execute(fixture)
    assert calls == [trial['trial_id'] for trial in fixture[0]['trials']]
    assert aggregate['status'] == 'complete' and aggregate['execution_status'] == 'finished_schedule'
    assert aggregate['coverage'] == {'attempted': 56, 'completed': 56, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 0, 'missing': 0}
    assert aggregate['overall']['categories']['correct'] == 56
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 0, 'false': 56, 'missing': 0}
    assert set(aggregate['by_condition']) == set(EXPECTED_CONDITIONS)
    for cell in aggregate['by_condition'].values():
        assert cell['planned_cells'] == cell['resolved_cells'] == cell['categories']['correct'] == 8
    assert set(aggregate['contrasts']) == set(aggregate['secondary_contrasts']) == {'strict_accuracy'}
    assert set(aggregate['contrasts']['strict_accuracy']) == {'full_minus_sham'}
    assert set(aggregate['secondary_contrasts']['strict_accuracy']) == set(REQUIRED) - {'full_minus_sham'}
    assert all(value['point'] == 0 for group in ('contrasts', 'secondary_contrasts')
               for value in aggregate[group]['strict_accuracy'].values())
    replay = a.export_run(**fixture[1])
    assert replay['aggregate'] == aggregate and len(replay['records']) == 56
    serialized = json.dumps(aggregate)
    assert all(secret not in serialized for secret in ('prompt_token_ids', 'response_text', 'generated_token_ids', 'PUBLIC synthetic'))
    assert all(trial['trial_id'] not in serialized for trial in fixture[0]['trials'])
    assert all(not set(row).intersection({'construction', 'selection_block', 'format_request'}) for row in replay['records'])


def test_clean_caps_do_not_gate_contexts_or_rescue_accuracy(fixture):
    marker = Tokenizer().encode('```')
    aggregate, calls = execute(fixture, response=lambda trial:
        marker + [ord('x') + 10] * 61 if trial['condition'] == 'clean' else encoded_answer(trial, field='selector'))
    assert len(calls) == 56 and aggregate['coverage']['completed'] == 56
    assert aggregate['overall']['categories']['cap'] == 8 and aggregate['overall']['categories']['selected_label'] == 48
    assert aggregate['overall']['strict_accuracy']['point'] == 0
    assert aggregate['overall']['indicators']['fence_marker'] == {'true': 8, 'false': 48, 'missing': 0}
    assert all(value['point'] == 0 for group in ('contrasts', 'secondary_contrasts')
               for value in aggregate[group]['strict_accuracy'].values())


@pytest.mark.parametrize('missing_condition', ['clean', 'full_before', 'inert_after'])
def test_runtime_infrastructure_only_nulls_affected_contrasts(fixture, missing_condition):
    index = next(t['sequence_index'] for t in fixture[0]['trials'] if t['condition'] == missing_condition)
    aggregate, calls = execute(fixture, fail_at=index)
    assert len(calls) == 56 and aggregate['overall']['strict_accuracy']['point'] is None
    observed = {**aggregate['contrasts']['strict_accuracy'], **aggregate['secondary_contrasts']['strict_accuracy']}
    assert {key for key, value in observed.items() if value['point'] is None} == {
        key for key, cohort in REQUIRED.items() if missing_condition in cohort}
    assert aggregate['overall']['indicators']['fence_marker']['missing'] == 1
    assert a.export_run(**fixture[1])['records'][index]['fence_marker'] is None


@pytest.mark.parametrize('key', REQUIRED)
def test_each_prespecified_contrast_tamper_is_rejected(fixture, key):
    execute(fixture)
    path = fixture[1]['output_root'] / 'analysis.json'
    value = json.loads(path.read_text())
    group = 'contrasts' if key == 'full_minus_sham' else 'secondary_contrasts'
    value[group]['strict_accuracy'][key]['point'] = 1
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])


@pytest.mark.parametrize('text,expected', [
    ('```', True), ('~~~', True), ('x```y', True), ('text\n~~~json', True),
    ('``', False), ('~~', False), ('` ` `', False), ('~ ~ ~', False),
    ('｀｀｀', False), ('～～～', False), ('', False), ('s84', False),
])
def test_only_literal_ascii_fence_flag_is_added_to_pinned_primary_score(fixture, text, expected):
    from lexical_prompt_study import mapped_value_clarification
    trial = fixture[0]['trials'][0]
    tokens = Tokenizer().encode(text) + [1]
    original = mapped_value_clarification.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    score = a.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    assert score == {**original, 'fence_marker': expected}
    assert set(score) - set(original) == {'fence_marker'}


@pytest.mark.parametrize('mutation', ['duplicate', 'count', 'integer', 'level'])
@pytest.mark.parametrize('cohort', ['pair', 'triple'])
def test_invalid_required_cohorts_rejected(fixture, mutation, cohort):
    records = make_records(fixture[0], lambda trial: True)
    condition = 'clean' if cohort == 'triple' else 'full_before'
    index = next(index for index, row in enumerate(records) if row['condition'] == condition)
    if mutation == 'duplicate':
        target = next(i for i, row in enumerate(records) if i != index and row['condition'] == condition)
        records[target] = copy.deepcopy(records[index])
    elif mutation == 'count':
        records.pop(index)
    elif mutation == 'integer':
        records[index]['strict_correct'] = 1
    else:
        records[index]['condition'] = 'unknown'
    with pytest.raises(ValueError):
        a._inert_minus_clean_bound(records) if cohort == 'triple' else a._full_minus_sham_bound(records)


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
                                     'exception_type': 'A178Interrupted'}
        elif tamper == 'records':
            value[0]['category'] = 'other_mapped_value'
        elif tamper == 'numeric_bool':
            value['overall']['strict_accuracy']['point'] = True
        else:
            value['contrasts']['strict_accuracy']['full_minus_sham']['point'] = .5
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
                                     'interrupted': 0, 'unattempted': 55, 'missing': 55}
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
