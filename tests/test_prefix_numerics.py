"""A170 public synthetic qualification: no pretrained weights or private inputs."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import types

import pytest

from lexical_prompt_study import prefix_numerics as a


class Tokenizer:
    eos_token_id, pad_token_id = 1, None
    pieces = {"<EOS>": 1, '{"': 2, "answer": 3, '":"': 4}

    def get_chat_template(self):
        return "a170 public synthetic native template"

    def encode(self, text, **kwargs):
        tokens = []
        while text:
            for piece, token in self.pieces.items():
                if text.startswith(piece):
                    tokens.append(token)
                    text = text[len(piece):]
                    break
            else:
                tokens.append(ord(text[0]) + 10)
                text = text[1:]
        return tokens

    def decode(self, tokens, **kwargs):
        reverse = {value: key for key, value in self.pieces.items()}
        return "".join(reverse[value] if value in reverse else chr(value - 10) for value in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"
        if not kwargs["add_generation_prompt"]:
            assert len(messages) == 3
            text += messages[2]["content"] + "<EOS>"
        return self.encode(text) if tokenize else text


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = tmp_path / "synthetic-model"
    model.mkdir()
    generation = model / "generation_config.json"
    generation.write_text('{"eos_token_id":[1]}')
    config = {"schema_version": "a163-runtime-v1", "model_path": str(model), "model_revision": "a" * 40,
              "model_files_sha256": {**{name: "b" * 64 for name in
                  ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")},
                  "generation_config.json": a.native.engine.file_digest(generation)},
              "chat_template_sha256": a.tasks.sha(Tokenizer().get_chat_template().encode()),
              "runtime_versions": {name: "synthetic" for name in ("python", *a.native.engine.VERSION_PACKAGES)},
              "quantization": "nf4", "dtype": "bfloat16", "attention_implementation": "sdpa", "device": "cuda:0",
              "max_prompt_tokens": 4096, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10}
    config_path, protocol, reference = tmp_path / "config.json", tmp_path / "protocol.md", tmp_path / "reference.py"
    config_path.write_bytes(a.tasks.canonical(config))
    protocol.write_text("Public synthetic A170 protocol fixture")
    reference.write_text("Public synthetic reference binding fixture")
    digest = a.native.engine.file_digest
    # Production source and reference constants remain strict; only synthetic
    # test config/reference files receive explicit substituted bindings.
    monkeypatch.setattr(a, "CONFIG_SHA", digest(config_path))
    monkeypatch.setattr(a.native.engine, "file_digest", lambda p: a.REFERENCE_SHA if Path(p) == reference else digest(p))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda p: config["model_files_sha256"])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    plan = a.compile_plan(digest(config_path), digest(protocol))
    path = tmp_path / "plan.json"
    path.write_bytes(a.tasks.canonical(plan))
    args = {"plan_path": path, "plan_sha256": digest(path), "config_path": config_path,
            "config_sha256": digest(config_path), "protocol_path": protocol, "reference_script": reference,
            "output_root": tmp_path / "output", "tokenizer_loader": lambda _: Tokenizer()}
    return plan, args


def prepare(fixture):
    return a.prepare_inputs(**{key: value for key, value in fixture[1].items() if key != "output_root"})


def test_exact_public_stimuli_schedule_and_native_audit(fixture):
    plan, args = fixture
    native = prepare(fixture)
    assert [c["filler_repetitions"] for c in plan["contexts"]] == [0, 24, 72]
    for context in plan["contexts"]:
        assert context["messages"] == [{"role": "system", "content": a.SYSTEM}, {"role": "user", "content":
            ("The field is quiet. " * context["filler_repetitions"]).rstrip() + '\n\nReturn {"answer":"s00"}.'}]
        assert list(context["canonical_candidates"]) == list(a.VALUES)
    assert plan["contexts"][0]["messages"][1]["content"].startswith("\n\n")
    assert len(plan["evaluations"]) == 30 and len({e["evaluation_id"] for e in plan["evaluations"]}) == 30
    for context in plan["contexts"]:
        evaluations = [e for e in plan["evaluations"] if e["context_id"] == context["context_id"]]
        assert [(e["method"], e["candidate"]) for e in evaluations] == [("prefix_pre", None)] + [
            (method, value) for method in a.FULL_METHODS for value in a.VALUES] + [("prefix_post", None)]
        p = native["native_inputs"][context["context_id"]]
        assert p["shared_json_prefix_token_ids"] == [2, 3, 4]
        assert p["shared_json_prefix_text"] == a.JSON_PREFIX
        assert p["candidate_lengths"] == {"s00": 9, "s01": 9, "A": 7, "B": 7}
        assert p["padding_token_id"] == 1 and p["padding_token_source"] == "native_eos_fallback"
        assert all(tokens[-1] == 1 and 1 not in tokens[:-1] for tokens in p["candidate_token_ids"].values())
    assert len({len(p["prompt_token_ids"]) for p in native["native_inputs"].values()}) == 3
    assert native["native_prepared_sha256"] == a.object_sha(native["native_inputs"])
    assert not args["output_root"].exists() and not a._PROCESS_CONSUMED
    assert not any("a167" in key or "a165" in key for key in plan)


@pytest.mark.parametrize("change", ["message", "candidate", "evaluation", "config", "prefix_count", "extra"])
def test_plan_drift_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == "message":
        plan["contexts"][0]["messages"][1]["content"] = plan["contexts"][0]["messages"][1]["content"].strip()
    elif change == "candidate":
        plan["contexts"][0]["canonical_candidates"]["s00"] = '{"answer":"s02"}'
    elif change == "evaluation":
        plan["evaluations"][1:3] = reversed(plan["evaluations"][1:3])
    elif change == "config":
        plan["bindings"]["config_sha256"] = "0" * 64
    elif change == "prefix_count":
        plan["prefix_tokens"] = 2
    else:
        plan["evaluations"].append(copy.deepcopy(plan["evaluations"][-1]))
    with pytest.raises(ValueError):
        a.validate_plan(plan)


@pytest.mark.parametrize("change", ["prompt", "eos", "premature_eos", "prefix", "equal_all_lengths", "unequal_pair", "minimal"])
def test_native_drift_rejected_before_loading(fixture, change):
    class BadTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if change == "prompt" and text.endswith(('s00"}', 's01"}', 'A"}', 'B"}')):
                result[0] += 1
            if change == "premature_eos":
                result = [1 if token == ord("B") + 10 else token for token in result]
            if change == "equal_all_lengths":
                # Keep exact decode possible but coalesce every value to one token.
                for value, token in (("s00", 5), ("s01", 6)):
                    parts = super().encode(value)
                    for i in range(len(result) - len(parts), -1, -1):
                        if result[i:i + len(parts)] == parts:
                            result[i:i + len(parts)] = [token]
            if change == "unequal_pair" and text.endswith('s01"}'):
                result.insert(-2, ord(" ") + 10)
            if change == "minimal" and text.endswith('{"answer'):
                result[0] += 1
            return result
        def decode(self, tokens, **kwargs):
            if change == "prefix" and tokens == [2, 3, 4]:
                return "invalid"
            if change == "premature_eos":
                tokens = [ord("B") + 10 if t == 1 else t for t in tokens]
            if change == "equal_all_lengths":
                return "".join("s00" if t == 5 else "s01" if t == 6 else super(BadTokenizer, self).decode([t]) for t in tokens)
            return super().decode(tokens, **kwargs)
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            if change == "eos" and not kwargs["add_generation_prompt"]:
                return result + [1] if tokenize else result + "<EOS>"
            return result
    fixture[1]["tokenizer_loader"] = lambda _: BadTokenizer()
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture[1]["output_root"].exists()


def test_native_partial_syntax_prefix_and_explicit_pad(fixture):
    class PartialTokenizer(Tokenizer):
        pad_token_id = 9
        pieces = {"<EOS>": 1, '{"': 2, "answer": 3, '":': 4, '"s': 5, '"A': 6, '"B': 7}
    fixture[1]["tokenizer_loader"] = lambda _: PartialTokenizer()
    values = prepare(fixture)["native_inputs"].values()
    for value in values:
        assert value["shared_json_prefix_text"] == '{"answer":'
        assert value["padding_token_id"] == 9 and value["padding_token_source"] == "tokenizer_pad_token_id"


def test_three_constructions_mask_geometry_and_eos_positions(fixture):
    inputs = prepare(fixture)["native_inputs"]
    for e in fixture[0]["evaluations"]:
        p = inputs[e["context_id"]]
        spec = a._forward_spec(p, e)
        n = len(p["prompt_token_ids"])
        assert spec["predictor_positions"] == [n - 1, n, n + 1]
        assert spec["physical_sequence_length"] - spec["logits_to_keep"] == n - 1
        assert spec["prefix_target_token_ids"] == [2, 3, 4]
        if e["candidate"] is None:
            assert spec["input_token_ids"] == p["prompt_token_ids"] + [2, 3]
            assert spec["logits_to_keep"] == 3 and spec["candidate_target_count"] is None
        else:
            target = p["candidate_token_ids"][e["candidate"]]
            assert spec["input_token_ids"][n:n + len(target)] == target
            assert spec["input_token_ids"][n + len(target) - 1] == 1
            pads = 9 - len(target) if e["method"] == "fixed_full" else 0
            assert spec["right_pad_count"] == pads
            assert spec["attention_mask"] == [1] * (n + len(target)) + [0] * pads


class PositionModel:
    """Real torch tensors; deterministic position- and physical-shape effects."""
    def __init__(self, torch, fail_at=None, interrupt_at=None, signal_number=signal.SIGTERM):
        self.torch, self.calls, self.specs = torch, 0, []
        self.fail_at, self.interrupt_at, self.signal_number = fail_at, interrupt_at, signal_number

    def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        assert not use_cache
        index = self.calls
        self.calls += 1
        self.specs.append((input_ids.tolist(), attention_mask.tolist(), logits_to_keep))
        if index == self.interrupt_at:
            if self.signal_number == signal.SIGALRM:
                a._deadline(self.signal_number, None)
            a._interrupt(self.signal_number, None)
        if index == self.fail_at:
            raise RuntimeError("synthetic confidential exception must not escape")
        torch = self.torch
        n = input_ids.shape[1]
        rows = torch.arange(n, dtype=torch.float32)[None, :, None]
        vocab = torch.arange(160, dtype=torch.float32)[None, None, :]
        # Not a causal model: exposes physical-shape differences intentionally.
        logits = (rows * vocab / 10000 + vocab / 100 + n * vocab / 100000).expand(1, n, 160).clone()
        return types.SimpleNamespace(logits=logits[:, -logits_to_keep:, :])


def execute(fixture, *, fail_at=None, interrupt_at=None, signal_number=signal.SIGTERM, loader_error=None):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    model = PositionModel(torch, fail_at, interrupt_at, signal_number)
    loaded = []
    def load(config, engine):
        loaded.append(True)
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        if loader_error:
            raise loader_error
        return types.SimpleNamespace(model=model, torch=torch, tokenizer=Tokenizer(), eos_ids=[1], device="cpu")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    result = a.run_plan(**fixture[1], reference_loader=lambda _: reference)
    assert len(loaded) == 1
    return result, model


def test_shifted_prefix_only_readout_float64_and_unscored_nan():
    torch = pytest.importorskip("torch")
    spec = {"input_token_ids": [7, 8, 2, 3, 4, 5, 1], "attention_mask": [1] * 7,
            "physical_sequence_length": 7, "prefix_target_token_ids": [2, 3, 4], "logits_to_keep": 6}
    class Model:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [spec["input_token_ids"]]
            assert kwargs["attention_mask"].tolist() == [spec["attention_mask"]]
            assert kwargs["use_cache"] is False and kwargs["logits_to_keep"] == 6
            logits = torch.zeros((1, 7, 10), dtype=torch.float32)
            for row, token in enumerate((2, 3, 4), start=1):
                logits[0, row, token] = row
            logits[0, 4:, :] = float("nan")  # All suffix/EOS rows are unscored.
            return types.SimpleNamespace(logits=logits[:, -6:, :])
    measured = a.measure_prefix(Model(), torch, spec)
    expected = [x - math.log(math.exp(x) + 9) for x in (1., 2., 3.)]
    assert measured["chosen_logits"] == [1., 2., 3.]
    assert measured["token_logprobs"] == pytest.approx(expected, abs=1e-14)
    assert measured["json_prefix_logprob"] == pytest.approx(math.fsum(expected), abs=1e-14)
    assert set(measured) == {"chosen_logits", "log_normalizers", "token_logprobs", "json_prefix_logprob"}


@pytest.mark.parametrize("bad", ["nan", "inf", "dtype", "rows"])
def test_invalid_measured_logits_rejected(bad):
    torch = pytest.importorskip("torch")
    spec = {"input_token_ids": [7, 8, 2, 3], "attention_mask": [1] * 4,
            "physical_sequence_length": 4, "prefix_target_token_ids": [2, 3, 4], "logits_to_keep": 3}
    def model(**kwargs):
        logits = torch.zeros(1, 2 if bad == "rows" else 3, 10, dtype=torch.float64 if bad == "dtype" else torch.float32)
        if bad in ("nan", "inf"):
            logits[0, 0, 9] = float(bad)  # Even an unchosen prefix logit matters to normalization.
        return types.SimpleNamespace(logits=logits)
    with pytest.raises(ValueError):
        a.measure_prefix(model, torch, spec)


def measurement(values):
    return {"chosen_logits": list(values), "log_normalizers": [0.] * 3,
            "token_logprobs": list(values), "json_prefix_logprob": math.fsum(values)}


def test_groups_signed_deltas_tolerance_and_missingness():
    values = {value: measurement([-1. - i / 10, -2., -3.]) for i, value in enumerate(a.VALUES)}
    group = a._group(values)
    assert group["prefix_sum_spread_nats"] == pytest.approx(.3)
    assert group["descriptive_guard_passed"] is False
    assert group["per_token_spreads"]["chosen_logits"] == pytest.approx([.3, 0, 0])
    assert len(group["pairwise_left_minus_right"]) == 6
    delta = group["pairwise_left_minus_right"]["s00_minus_B"]
    assert delta["json_prefix_logprob_nats"] == pytest.approx(.3)
    assert delta["chosen_logits"] == pytest.approx([.3, 0, 0])
    tolerance_values = {value: measurement([0., 0., 0.]) for value in a.VALUES}
    tolerance_values["B"] = measurement([-a.PREFIX_ABSOLUTE_TOLERANCE_NATS, 0., 0.])
    assert a._group(tolerance_values)["descriptive_guard_passed"] is True
    tolerance_values["B"] = measurement([-math.nextafter(a.PREFIX_ABSOLUTE_TOLERANCE_NATS, math.inf), 0., 0.])
    assert a._group(tolerance_values)["descriptive_guard_passed"] is False
    values["B"] = None
    missing = a._group(values)
    assert missing["completed_candidates"] == 3 and missing["prefix_sum_spread_nats"] is None
    assert missing["descriptive_guard_passed"] is None
    assert missing["per_token_spreads"]["log_normalizers"] == [None] * 3
    assert missing["pairwise_left_minus_right"]["s00_minus_B"]["json_prefix_logprob_nats"] is None
    assert missing["pairwise_left_minus_right"]["s00_minus_s01"]["json_prefix_logprob_nats"] is not None


def test_complete_real_tensor_schedule_and_export(fixture):
    aggregate, model = execute(fixture)
    assert model.calls == 30 and aggregate["status"] == "complete"
    assert aggregate["coverage"] == {"attempted": 30, "completed": 30, "infrastructure_failed": 0, "interrupted": 0, "unattempted": 0}
    replay = a.export_run(**fixture[1])
    assert replay["aggregate"] == aggregate and len(replay["records"]) == 3
    assert aggregate["descriptive_prefix_guard"]["failed"] == 3
    assert aggregate["descriptive_prefix_guard"]["passed"] == 3
    for row in replay["records"]:
        assert row["reference_post_minus_pre"]["json_prefix_logprob_nats"] == 0
        assert row["within_candidate_groups"]["fixed_full"]["prefix_sum_spread_nats"] == 0
        assert row["within_candidate_groups"]["variable_full"]["prefix_sum_spread_nats"] > 0
        assert row["candidate_minus_reference_pre"]["fixed_full"]["s00"]["json_prefix_logprob_nats"] != 0
    serialized = json.dumps(aggregate)
    assert "chosen_logits" not in serialized and "prompt_token_ids" not in serialized
    assert all(c["context_id"] not in serialized for c in fixture[0]["contexts"])
    assert (fixture[1]["output_root"] / "native-inputs.json").stat().st_mode & 0o777 == 0o600


def test_measurement_error_does_not_retry_or_change_schedule(fixture):
    aggregate, model = execute(fixture, fail_at=2)
    assert model.calls == 30 and aggregate["execution_status"] == "finished_schedule" and aggregate["status"] == "incomplete"
    assert aggregate["coverage"]["infrastructure_failed"] == 1 and aggregate["coverage"]["unattempted"] == 0
    assert aggregate["descriptive_prefix_guard"]["missing"] == 1
    row = a.export_run(**fixture[1])["records"][0]
    assert row["measurements"]["variable_full"]["s01"] is None
    assert row["candidate_minus_reference_pre"]["variable_full"]["s01"]["chosen_logits"] == [None] * 3
    result = json.loads((fixture[1]["output_root"] / "evaluations" / fixture[0]["evaluations"][2]["evaluation_id"] / "result.json").read_bytes())
    assert "confidential" not in json.dumps(result)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM])
def test_terminal_signal_preserves_actual_number_pid_and_missingness(fixture, number):
    aggregate, model = execute(fixture, interrupt_at=3, signal_number=number)
    assert model.calls == 4
    assert aggregate["coverage"] == {"attempted": 4, "completed": 3, "infrastructure_failed": 0, "interrupted": 1, "unattempted": 26}
    receipt = json.loads((fixture[1]["output_root"] / "execution-finished.json").read_bytes())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {"signal_number": int(number), "pid": os.getpid(),
        "exception_type": "A170Deadline" if number == signal.SIGALRM else "A170Interrupted"}
    assert a.export_run(**fixture[1])["aggregate"] == aggregate


def test_signal_context_deadline_and_handlers_restore(monkeypatch):
    old = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with a.bounded_signals():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 1799 < remaining <= 1800 and interval == 0
        assert signal.getsignal(signal.SIGALRM) is a._deadline
        assert signal.getsignal(signal.SIGTERM) is a._interrupt
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {n: signal.getsignal(n) for n in old} == old


def test_loader_failure_consumes_one_shot_and_all_measurements_missing(fixture, monkeypatch):
    aggregate, model = execute(fixture, loader_error=RuntimeError("synthetic loader failure"))
    assert model.calls == 0 and aggregate["execution_status"] == "failed"
    assert aggregate["coverage"]["unattempted"] == 30 and aggregate["descriptive_prefix_guard"]["missing"] == 6
    with pytest.raises(ValueError, match="fresh_process"):
        execute(fixture)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        execute(fixture)


def test_preload_memory_gate_blocks_model(fixture):
    def forbidden(*args):
        pytest.fail("model loaded below 48 GiB")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - 1, load_cpu_reference=forbidden)
    with pytest.raises(ValueError, match="memory_gate"):
        a.run_plan(**fixture[1], reference_loader=lambda _: reference)
    assert not (fixture[1]["output_root"] / "one-shot.json").exists()


@pytest.mark.parametrize("tamper", ["mask", "position", "chosen", "probability", "sum", "orphan", "order", "receipt", "signal", "summary"])
def test_replay_rejects_tampering(fixture, tamper):
    execute(fixture)
    plan, args = fixture
    root = args["output_root"]
    folder = root / "evaluations" / plan["evaluations"][1]["evaluation_id"]
    if tamper == "orphan":
        (root / "evaluations" / "unknown").mkdir()
        with pytest.raises(ValueError):
            a.export_run(**args)
        return
    if tamper == "order":
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
        with pytest.raises(ValueError, match="attempt_sequence"):
            a.export_run(**args)
        return
    filename = "attempt.json" if tamper in ("mask", "position") else "result.json"
    path = (root / "execution-finished.json" if tamper in ("receipt", "signal") else
            root / "records.json" if tamper == "summary" else folder / filename)
    value = json.loads(path.read_bytes())
    if tamper == "mask":
        value["forward_spec"]["attention_mask"][-1] = 0
    elif tamper == "position":
        value["forward_spec"]["predictor_positions"][0] += 1
    elif tamper == "chosen":
        value["measurement"]["chosen_logits"][0] += 1
    elif tamper == "probability":
        value["measurement"]["token_logprobs"][0] += 1
    elif tamper == "sum":
        value["measurement"]["json_prefix_logprob"] += 1
    elif tamper == "receipt":
        value["run_sha256"] = "0" * 64
    elif tamper == "signal":
        value["interruption"] = {"signal_number": 15, "pid": os.getpid(), "exception_type": "A170Interrupted"}
    else:
        value[0]["reference_post_minus_pre"]["json_prefix_logprob_nats"] += 1
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**args)


def test_exclusive_lock_prevents_execution(fixture):
    import fcntl
    root = fixture[1]["output_root"]
    root.mkdir()
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[1])


def test_system_exit_is_terminal_and_not_swallowed_as_measurement_failure(fixture):
    with pytest.raises(SystemExit):
        execute(fixture, loader_error=SystemExit(7))
    root = fixture[1]["output_root"]
    assert (root / "one-shot.json").exists()
    assert json.loads((root / "execution-finished.json").read_bytes())["status"] == "failed"
    exported = a.export_run(**fixture[1])
    assert exported["aggregate"]["coverage"]["unattempted"] == 30


def test_cli_prints_only_safe_status_and_coverage(fixture, monkeypatch, capsys):
    def fake_run(**kwargs):
        print("synthetic private token and loader output")
        return {"status": "complete", "execution_status": "finished_schedule",
                "coverage": {"attempted": 30, "completed": 30}, "chosen_logits": [123.], "secret": "not for stdout"}
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
    assert "synthetic private" not in captured.out + captured.err and "chosen_logits" not in captured.out
    log = args["output_root"] / "execution.log"
    assert "synthetic private" in log.read_text() and log.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_tiny_random_causal_model_against_independent_full_logits(attention):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(4)
    torch.manual_seed(71)
    config = transformers.LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=1, pad_token_id=1, attention_dropout=0.)
    config._attn_implementation = attention
    model = transformers.LlamaForCausalLM(config).float().eval()  # Random local weights only.
    paths = {"s00": [2, 3, 4, 8, 9, 5, 1], "s01": [2, 3, 4, 8, 10, 5, 1],
             "A": [2, 3, 4, 11, 5, 1], "B": [2, 3, 4, 12, 5, 1]}
    for prompt in ([7, 6], [7, 6] * 9):
        prepared = {"prompt_token_ids": prompt, "shared_json_prefix_token_ids": [2, 3, 4],
                    "candidate_token_ids": paths, "longest_target_count": 7, "intended_eos_token_id": 1,
                    "padding_token_id": 1}
        observed = []
        for method, candidate in [("prefix_pre", None)] + [(m, v) for m in a.FULL_METHODS for v in a.VALUES] + [("prefix_post", None)]:
            spec = a._forward_spec(prepared, {"method": method, "candidate": candidate})
            actual = a.measure_prefix(model, torch, spec)
            ids = torch.tensor([spec["input_token_ids"]], dtype=torch.long)
            mask = torch.tensor([spec["attention_mask"]], dtype=torch.long)
            with torch.inference_mode():
                # Independent oracle materializes every logit and explicitly
                # indexes absolute predictor positions, not the scorer's slice.
                full = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=0).logits[0]
                selected = full[torch.tensor([len(prompt) - 1, len(prompt), len(prompt) + 1])].double()
                expected_logits = selected[torch.arange(3), torch.tensor([2, 3, 4])]
                expected_normalizers = torch.logsumexp(selected, -1)
                expected_probs = expected_logits - expected_normalizers
            assert actual["chosen_logits"] == pytest.approx(expected_logits.tolist(), abs=2e-7)
            assert actual["log_normalizers"] == pytest.approx(expected_normalizers.tolist(), abs=2e-7)
            assert actual["token_logprobs"] == pytest.approx(expected_probs.tolist(), abs=4e-7)
            observed.append(actual)
        for result in observed:
            assert result["token_logprobs"] == pytest.approx(observed[0]["token_logprobs"], abs=4e-7)
        assert observed[0] == observed[-1]  # Separate equal-shape reference calls.
