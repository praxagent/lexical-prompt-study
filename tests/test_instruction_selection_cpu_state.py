"""Synthetic A166 triplets; never import or load a Torch model."""
import builtins
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import sys
import types

import pytest

from lexical_prompt_study import instruction_selection_cpu_state as a

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a166_synthetic_native", ROOT / "tests/test_instruction_selection_runtime.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    data = helper.inputs.__wrapped__(tmp_path)
    materials = {kind: "Public synthetic " + kind for kind in a.tasks.SCAFFOLDS}
    parent_plan = a.parent.compile_plan(data["development"], scaffolds=materials, protocol_sha256="c" * 64)
    header = {"schema_version": "a165-run-v1", "plan_sha256": a.object_sha(parent_plan),
        "config_sha256": a.native.engine.file_digest(data["config_path"]), "source_hashes": a.parent.validate_sources(),
        "cpu_reference_script_sha256": a.REFERENCE_SHA, "reference_protocol_sha256": a.parent.REFERENCE_PROTOCOL_SHA,
        "reference_amendment_sha256": a.parent.REFERENCE_AMENDMENT_SHA, "heldout_allowed": False,
        "retries": 0, "exploratory_development_only": True, "eos_token_ids": [1], "settings": {
            "device": "cpu", "dtype": "float32", "quantization": None, "attention_implementation": "sdpa",
            "cpu_threads": 4, "seed": data["config"]["seed"], "max_new_tokens": 64}}
    protocol = tmp_path / "a166.md"
    protocol.write_text("Public synthetic fixed-triplet protocol")
    plan = a.compile_plan(parent_plan, a165_run_header=header,
                          protocol_sha256=a.native.engine.file_digest(protocol))
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(a.tasks.canonical(plan))
    reference = tmp_path / "reference.py"
    reference.write_text("# public synthetic reference stand-in")
    original_digest = a.native.engine.file_digest
    monkeypatch.setattr(a.native.engine, "file_digest", lambda p: a.REFERENCE_SHA if Path(p) == reference else original_digest(p))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: data["config"]["model_files_sha256"])
    args = {"plan_path": plan_path, "plan_sha256": original_digest(plan_path), "config_path": data["config_path"],
        "config_sha256": original_digest(data["config_path"]), "protocol_path": protocol, "reference_script": reference,
        "output_root": tmp_path / "output", "tokenizer_loader": lambda config: helper.Tokenizer()}
    return data, plan, args


class StateRuntime:
    tokenizer = helper.Tokenizer()
    eos_ids = [1]

    def __init__(self, plan, mode="stable", interrupt_after=None):
        self.plan, self.mode, self.interrupt_after = plan, mode, interrupt_after
        self.calls = 0

    def generate(self, prepared):
        if self.calls == self.interrupt_after:
            a._interrupt(signal.SIGTERM, None)
        trial = self.plan["trials"][self.calls]
        assert prepared["rendered_text"] == self.tokenizer.apply_chat_template(
            trial["messages"], tokenize=False, add_generation_prompt=True)
        self.calls += 1
        phase, index = trial["diagnostic_phase"], trial["triplet_index"]
        if self.mode == "infrastructure" and phase == "inert" and index == 0:
            raise RuntimeError("private synthetic error")
        if phase == "inert" and self.mode != "all_correct":
            if self.mode == "cap":
                return self.tokenizer.encode("x" * 64), "x" * 64, "length"
            text = "not json"
        else:
            wrong = (self.mode == "deterioration" and index == 0 and phase == "post"
                     or self.mode == "pre_failure" and index == 0 and phase == "pre"
                     or self.mode == "mixed_deterioration" and (index == 1 and phase == "pre"
                                                               or index == 0 and phase == "post"))
            text = json.dumps({"answer": trial["unselected_answer" if wrong else "selected_answer"]})
            if self.mode == "token_drift" and phase == "post" and index == 0:
                text = " " + text
        return self.tokenizer.encode(text) + [1], text, "eos"


def execute(fixture, mode="stable", interrupt_after=None, loader_error=None):
    model = StateRuntime(fixture[1], mode, interrupt_after)
    loaded = []
    def load(config, engine):
        loaded.append(True)
        if loader_error:
            raise loader_error
        return model
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    result = a.run_plan(**fixture[2], reference_loader=lambda path: reference)
    assert len(loaded) == 1
    return result, model


def export(fixture):
    return a.export_run(**fixture[2])


def test_fixed_first_four_world_selection_original_order_and_identical_native_baselines(fixture):
    data, plan, args = fixture
    originals = data["development"]["trials"]
    worlds = list(dict.fromkeys(t["world_id"] for t in originals))[:4]
    selected = [t for t in originals if t["world_id"] in worlds]
    assert len(plan["trials"]) == len({t["trial_id"] for t in plan["trials"]}) == 24
    assert [(t["world_id"], t["selector"]) for t in plan["trials"][::3]] == [
        (t["world_id"], t["selector"]) for t in selected]
    parent_ids = {t["trial_id"] for t in plan["a165_plan"]["trials"]}
    for index in range(8):
        pre, inert, post = plan["trials"][3 * index:3 * index + 3]
        assert [t["diagnostic_phase"] for t in (pre, inert, post)] == list(a.PHASES)
        assert pre["messages"] == post["messages"] == selected[index]["messages"]
        assert inert["scaffold_kind"] == "inert" and inert["placement"] == "before"
        source = next(t for t in plan["a165_plan"]["trials"] if t["trial_id"] == inert["source_a165_trial_id"])
        assert inert["messages"] == source["messages"]
        assert not {pre["trial_id"], inert["trial_id"], post["trial_id"]} & parent_ids
    inputs = a.prepare_inputs(**{k: v for k, v in args.items() if k != "output_root"})
    assert inputs["planned_cells"] == 24
    assert inputs["native_prepared_sha256"] == a.object_sha(inputs["native_inputs"])
    assert not args["output_root"].exists() and a._PROCESS_CONSUMED is False


@pytest.mark.parametrize("mutation", ["selector", "placement", "messages", "order", "subset", "phase", "world", "source", "config"])
def test_changed_selection_prompt_instrument_or_parent_config_rejected(fixture, mutation):
    plan = copy.deepcopy(fixture[1])
    if mutation == "selector":
        plan["trials"][0]["selector"] = "B"
    elif mutation == "placement":
        plan["trials"][1]["placement"] = "after"
    elif mutation == "messages":
        plan["trials"][2]["messages"][1]["content"] += "changed"
    elif mutation == "order":
        plan["trials"][0], plan["trials"][3] = plan["trials"][3], plan["trials"][0]
    elif mutation == "subset":
        plan["trials"].pop()
    elif mutation == "phase":
        plan["trials"][0]["diagnostic_phase"] = "post"
    elif mutation == "world":
        plan["trials"][0]["world_id"] = "0" * 64
    elif mutation == "source":
        plan["bindings"]["a166_source_sha256"] = "0" * 64
    else:
        plan["bindings"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_complete_stable_reproduction_replays_and_exports_only_aggregate_counts(fixture):
    result, model = execute(fixture)
    assert model.calls == 24 and result["decision"] == "carries_less_persistent_state_concern"
    assert result["coverage"]["completed"] == 24 and result["pre_post"]["token_identical_pairs"] == 8
    assert result["arms"]["inert"]["strict_failed"] == 8
    replay = export(fixture)
    assert replay["aggregate"] == result and len(replay["records"]) == 24
    text = json.dumps(result)
    assert all(t["trial_id"] not in text and t["world_id"] not in text for t in fixture[1]["trials"])
    assert "pre_tokens_sha256" not in text and "response_text" not in text
    assert len(replay["private_token_comparison"]) == 8


@pytest.mark.parametrize("mode,decision", [("deterioration", "investigate_state"),
    ("mixed_deterioration", "investigate_state"), ("pre_failure", "inconclusive"),
    ("all_correct", "inconclusive"), ("token_drift", "inconclusive"),
    ("infrastructure", "inconclusive"), ("cap", "carries_less_persistent_state_concern")])
def test_predefined_branches_without_adaptive_expansion(fixture, mode, decision):
    result, model = execute(fixture, mode)
    assert model.calls == 24 and result["decision"] == decision
    if mode == "pre_failure":
        assert result["pre_post"]["strict_improvement_pairs"] == 1
    if mode in ("deterioration", "mixed_deterioration"):
        assert result["pre_post"]["strict_deterioration_pairs"] == 1
        assert result["observed_deterioration_requires_investigation"]
    if mode == "token_drift":
        assert result["pre_post"]["token_identical_pairs"] == 7
        assert result["pre_post"]["strict_unchanged_pairs"] == 8
    if mode == "cap":
        assert result["arms"]["inert"]["capped"] == 8
    if mode == "infrastructure":
        assert result["status"] == "incomplete" and result["arms"]["inert"]["infrastructure_failed"] == 1
        assert result["arms"]["inert"]["strict_accuracy_bounds_pp"] == [0, 12.5]


def test_late_interruption_keeps_all_eight_pair_denominators_and_outstanding_attempt(fixture):
    result, model = execute(fixture, interrupt_after=23)
    assert model.calls == 23 and result["status"] == "incomplete" and result["decision"] == "inconclusive"
    assert result["interrupted_attempts"] == 1 and result["unattempted_cells"] == 0
    assert result["pre_post"]["completed_pairs"] == 7
    assert result["pre_post"]["strict_difference_post_minus_pre_pp"] is None
    assert result["pre_post"]["strict_difference_bounds_pp"] == [-12.5, 0]
    assert result["arms"]["post"]["strict_accuracy_bounds_pp"] == [87.5, 100]
    assert export(fixture)["aggregate"] == result
    assert not issubclass(a.A166Interrupted, Exception)


def test_observed_deterioration_still_flags_investigation_with_other_missing_triplets(fixture):
    result, _ = execute(fixture, "deterioration", interrupt_after=4)
    assert result["decision"] == "investigate_state" and result["coverage"]["completed"] == 4
    assert "missing_or_infrastructure_failure" in result["inconclusive_reasons"]


def test_deadline_during_load_retains_all_planned_unresolved_cells(fixture):
    result, model = execute(fixture, loader_error=a.A166Deadline())
    assert model.calls == 0 and result["coverage"]["missing"] == 24
    assert result["pre_post"]["strict_difference_bounds_pp"] == [-100, 100]
    assert result["unattempted_cells"] == 24
    finished = json.loads((fixture[2]["output_root"] / "execution-finished.json").read_bytes())
    assert finished["status"] == "deadline" and export(fixture)["aggregate"] == result


def test_consumed_process_and_consumed_root_both_reject_before_loading(fixture, monkeypatch):
    execute(fixture)
    def forbidden(*args):
        pytest.fail("consumed run loaded reference")
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**{**fixture[2], "output_root": fixture[2]["output_root"].with_name("second")}, reference_loader=forbidden)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[2], reference_loader=forbidden)


def test_exclusive_process_lock_rejects_duplicate_and_export(fixture):
    root = fixture[2]["output_root"]
    root.mkdir()
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[2])
        with pytest.raises(BlockingIOError):
            export(fixture)


@pytest.mark.parametrize("mutation", ["prompt", "eos", "response", "tokens", "score", "score_type", "elapsed", "native_inputs", "orphan", "unknown", "export", "canonical"])
def test_native_receipt_replay_detects_drift(fixture, mutation):
    execute(fixture)
    root = fixture[2]["output_root"]
    trial = fixture[1]["trials"][0]
    folder = root / "trials" / trial["trial_id"]
    path = folder / "result.json"
    if mutation in ("prompt", "eos"):
        path = folder / "attempt.json"
    elif mutation == "native_inputs":
        path = root / "native-inputs.json"
    elif mutation == "export":
        path = root / "records.json"
    if mutation == "orphan":
        (folder / "attempt.json").unlink()
    elif mutation == "unknown":
        (root / "trials" / ("0" * 24)).mkdir()
    elif mutation == "canonical":
        path.write_text(json.dumps(json.loads(path.read_bytes()), indent=2))
    else:
        value = json.loads(path.read_bytes())
        if mutation == "prompt":
            value["prepared"]["prompt_token_ids"][0] += 1
        elif mutation == "eos":
            value["eos_token_ids"] = [2]
        elif mutation == "response":
            value["private"]["response_text"] += " "
        elif mutation == "tokens":
            value["private"]["generated_token_ids"][0] += 1
        elif mutation == "score":
            value["record"]["score"]["strict_correct"] = False
        elif mutation == "score_type":
            value["record"]["score"]["strict_correct"] = 1
        elif mutation == "elapsed":
            value["elapsed_seconds"] = -1
        elif mutation == "native_inputs":
            value[trial["trial_id"]]["prompt_token_ids"][0] += 1
        elif mutation == "export":
            value.pop()
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        export(fixture)


def test_missing_terminal_followed_by_later_attempt_is_rejected(fixture):
    execute(fixture)
    root = fixture[2]["output_root"]
    (root / "trials" / fixture[1]["trials"][1]["trial_id"] / "result.json").unlink()
    with pytest.raises(ValueError, match="attempt_sequence"):
        export(fixture)


@pytest.mark.parametrize("mutation", ["source", "model", "config", "reference", "protocol", "memory"])
def test_preflight_drift_fails_before_model_load_or_attempt(fixture, monkeypatch, mutation):
    data, _, args = fixture
    original = a.native.engine.file_digest
    if mutation == "source":
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p).name == "instruction_selection_runtime.py" else original(p))
    elif mutation == "model":
        monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: {})
    elif mutation == "config":
        config = dict(data["config"], seed=99)
        args["config_path"].write_bytes(a.tasks.canonical(config))
        args["config_sha256"] = original(args["config_path"])
    elif mutation == "reference":
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p) == args["reference_script"] else original(p))
    elif mutation == "protocol":
        args["protocol_path"].write_text("changed")
    def forbidden(*args):
        pytest.fail("preflight failure loaded model")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - (mutation == "memory"),
                                      load_cpu_reference=forbidden)
    with pytest.raises(ValueError):
        a.run_plan(**args, reference_loader=lambda path: reference)
    assert not list(args["output_root"].glob("trials/*/attempt.json"))


def test_signal_scope_installs_baseexception_and_restores_handlers():
    original = signal.getsignal(signal.SIGTERM)
    with a.bounded_signals():
        assert signal.getsignal(signal.SIGTERM) is a._interrupt
        with pytest.raises(a.A166Interrupted):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with pytest.raises(a.A166Deadline):
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    assert signal.getsignal(signal.SIGTERM) == original
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_main_keeps_loading_text_private_and_outputs_no_trial_ids(fixture, monkeypatch, capsys):
    model = StateRuntime(fixture[1])
    def load(config, engine):
        print("private synthetic loading")
        sys.stderr.write("private synthetic progress\n")
        return model
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    original = a.run_plan
    monkeypatch.setattr(a, "run_plan", lambda **kw: original(**kw, reference_loader=lambda path: reference,
                        tokenizer_loader=lambda config: helper.Tokenizer()))
    argv = []
    for key, value in fixture[2].items():
        if key != "tokenizer_loader":
            argv += ["--" + key.removesuffix("_path").replace("_", "-"), str(value)]
    assert a.main(argv) == 0
    captured = capsys.readouterr()
    assert "private synthetic" not in captured.out + captured.err
    assert json.loads(captured.out)["planned_cells"] == 24
    assert "private synthetic loading" in (fixture[2]["output_root"] / "execution.log").read_text()
    assert all(t["trial_id"] not in captured.out for t in fixture[1]["trials"])


def test_import_performs_no_model_library_import_or_call(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "accelerate", "bitsandbytes"}:
            pytest.fail("import attempted to access model runtime")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("lexical_prompt_study.a166_import_test", Path(a.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._PROCESS_CONSUMED is False
