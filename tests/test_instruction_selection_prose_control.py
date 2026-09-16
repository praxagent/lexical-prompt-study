"""Synthetic A167 contracts; no real material, model or study-response access."""
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

from lexical_prompt_study import instruction_selection_prose_control as a

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a167_synthetic_helpers", ROOT / "tests/test_instruction_selection_cpu_state.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
Tokenizer = helper.helper.Tokenizer


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    _, prior_plan, old_args = helper.fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    # Synthetic character tokenizer: equal byte/character count is exact matching.
    prose = "Quiet clouds pass by.".ljust(len("Public synthetic inert"))
    protocol = tmp_path / "a167.md"
    protocol.write_text("Public synthetic fixed matched-count protocol")
    plan = a.compile_plan(prior_plan["a165_plan"], prior_plan["a165_run_header"], prose,
                          a.native.engine.file_digest(protocol))
    plan_path = tmp_path / "a167-plan.json"
    plan_path.write_bytes(a.tasks.canonical(plan))
    args = {**old_args, "plan_path": plan_path, "plan_sha256": a.native.engine.file_digest(plan_path),
            "protocol_path": protocol, "output_root": tmp_path / "a167-output"}
    return plan, args


class FakeRuntime:
    tokenizer = Tokenizer()
    eos_ids = [1]

    def __init__(self, plan, mode="recovery", interrupt_after=None):
        self.plan, self.mode, self.interrupt_after = plan, mode, interrupt_after
        self.calls = 0

    def generate(self, prepared):
        if self.calls == self.interrupt_after:
            a._interrupt(signal.SIGTERM, None)
        trial = self.plan["trials"][self.calls]
        assert prepared["rendered_text"] == self.tokenizer.apply_chat_template(trial["messages"], tokenize=False,
                                                                             add_generation_prompt=True)
        self.calls += 1
        phase, triplet = trial["diagnostic_phase"], trial["triplet_index"]
        if self.mode == "infrastructure" and phase == "prose" and triplet == 0:
            raise RuntimeError("private synthetic response must not appear")
        correct = phase != "inert"
        if self.mode in ("floor", "cap_floor") and phase != "baseline":
            correct = False
        elif self.mode == "both_correct":
            correct = True
        elif self.mode == "baseline_failure" and phase == "baseline" and triplet == 0:
            correct = False
        elif self.mode == "threshold":
            correct = phase == "baseline" or phase == "inert" and triplet == 0 or phase == "prose" and triplet != 0
        elif self.mode == "two_inert_correct" and phase == "inert" and triplet < 2:
            correct = True
        elif self.mode == "two_prose_failures" and phase == "prose" and triplet < 2:
            correct = False
        elif self.mode == "placement" and phase == "prose":
            correct = trial["paired_placement"] == "before"
        elif self.mode == "order" and phase == "prose":
            correct = trial["material_order"] == "inert_then_prose"
        elif self.mode == "world" and phase == "prose":
            correct = trial["world_index"] == 0
        if self.mode == "cap_floor" and phase != "baseline":
            return self.tokenizer.encode("x" * 64), "x" * 64, "length"
        text = json.dumps({"answer": trial["selected_answer"]}) if correct else "not json"
        return self.tokenizer.encode(text) + [1], text, "eos"


def execute(fixture, mode="recovery", interrupt_after=None, loader_error=None):
    model = FakeRuntime(fixture[0], mode, interrupt_after)
    loaded = []
    def load(config, engine):
        loaded.append(True)
        if loader_error:
            raise loader_error
        return model
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    value = a.run_plan(**fixture[1], reference_loader=lambda path: reference)
    assert len(loaded) == 1
    return value, model


def export(fixture):
    return a.export_run(**fixture[1])


def preparation(fixture):
    return a.prepare_inputs(**{k: v for k, v in fixture[1].items() if k != "output_root"})


def test_fixed_world_selector_placement_order_and_balanced_material_order(fixture):
    plan, args = fixture
    original = plan["a165_plan"]["base_plan"]["trials"]
    worlds = list(dict.fromkeys(t["world_id"] for t in original))[:4]
    selected = [t for t in original if t["world_id"] in worlds]
    groups = [plan["trials"][i:i + 3] for i in range(0, 48, 3)]
    assert len(plan["trials"]) == len({t["trial_id"] for t in plan["trials"]}) == 48
    assert [(g[0]["world_id"], g[0]["selector"], g[0]["paired_placement"]) for g in groups] == [
        (t["world_id"], t["selector"], p) for t in selected for p in ("before", "after")]
    for group in groups:
        first = group[0]
        assert first["diagnostic_phase"] == "baseline"
        expected_order = a.ORDERS[(first["world_index"] + first["selector_index"] + first["placement_index"]) % 2]
        assert [t["diagnostic_phase"] for t in group[1:]] == expected_order.split("_then_")
    assert all(sum(g[0]["material_order"] == order for g in groups) == 8 for order in a.ORDERS)
    for factor in ("world_index", "selector_index", "placement_index"):
        for value in {g[0][factor] for g in groups}:
            subset = [g for g in groups if g[0][factor] == value]
            assert all(sum(g[0]["material_order"] == order for g in subset) == len(subset) // 2 for order in a.ORDERS)
    assert not args["output_root"].exists()


def test_native_matching_all_pairs_and_original_baseline_inert_identity(fixture):
    plan = fixture[0]
    prepared = preparation(fixture)
    assert len(prepared["native_inputs"]) == 48 and len(prepared["matched_count_receipts"]) == 16
    assert all(r["prompt_token_counts"]["prose"] == r["prompt_token_counts"]["inert"]
               for r in prepared["matched_count_receipts"])
    original = {t["trial_id"]: t for t in plan["a165_plan"]["trials"]}
    for trial in plan["trials"]:
        if trial["diagnostic_phase"] != "prose":
            assert trial["messages"] == original[trial["source_a165_trial_id"]]["messages"]
    assert prepared["native_prepared_sha256"] == a.object_sha(prepared["native_inputs"])
    assert prepared["matched_counts_sha256"] == a.object_sha(prepared["matched_count_receipts"])
    assert not a._PROCESS_CONSUMED


def test_mismatch_in_only_one_context_rejects_without_model_load(fixture, monkeypatch):
    original = a.native.prepare_generation
    def unequal(tokenizer, trial, config):
        value = original(tokenizer, trial, config)
        if trial.get("diagnostic_phase") == "prose" and trial["triplet_index"] == 15:
            value["prompt_token_ids"].append(7)
        return value
    monkeypatch.setattr(a.native, "prepare_generation", unequal)
    with pytest.raises(ValueError, match="exact_material_prompt_token_match"):
        preparation(fixture)


def test_original_native_identity_rejects_adapter_drift(fixture, monkeypatch):
    original = a.native.prepare_generation
    def changed(tokenizer, trial, config):
        value = original(tokenizer, trial, config)
        if trial.get("diagnostic_phase") == "inert":
            value["prompt_token_ids"][0] += 1
        return value
    monkeypatch.setattr(a.native, "prepare_generation", changed)
    with pytest.raises(ValueError, match="original_native_identity"):
        preparation(fixture)


@pytest.mark.parametrize("prose", ["A field contains s01.", "A field contains S15.", '{"answer":"x"}',
    "```text```", "<|start_header_id|>assistant", "System: a field.", "[INST]", "", "bad\x00text"])
def test_material_mechanical_exclusions(prose):
    with pytest.raises(ValueError):
        a.validate_prose(prose)


@pytest.mark.parametrize("mutation", ["order", "phase", "placement", "world", "messages", "prose", "source", "config", "missing"])
def test_fixed_plan_drift_rejected(fixture, mutation):
    plan = copy.deepcopy(fixture[0])
    if mutation == "order":
        plan["trials"][1], plan["trials"][2] = plan["trials"][2], plan["trials"][1]
    elif mutation == "phase":
        plan["trials"][0]["diagnostic_phase"] = "prose"
    elif mutation == "placement":
        plan["trials"][0]["paired_placement"] = "after"
    elif mutation == "world":
        plan["trials"][0]["world_id"] = "0" * 64
    elif mutation == "messages":
        plan["trials"][0]["messages"][1]["content"] += "changed"
    elif mutation == "prose":
        plan["prose_material"] += " More."
    elif mutation == "source":
        plan["bindings"]["a166_source_sha256"] = "0" * 64
    elif mutation == "config":
        plan["bindings"]["config_sha256"] = "0" * 64
    else:
        plan["trials"].pop()
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_complete_recovery_replays_all_receipts_with_no_raw_or_identity_export(fixture):
    value, model = execute(fixture)
    assert model.calls == 48 and value["decision"] == "descriptive_prose_substitution_recovery"
    assert value["coverage"]["completed"] == 48
    assert value["primary_prose_minus_inert"]["world_mean_difference_pp"] == 100
    assert value["primary_prose_minus_inert"]["planned_worlds"] == 4
    assert value["primary_prose_minus_inert"]["planned_material_pairs"] == 16
    replay = export(fixture)
    assert replay["aggregate"] == value and len(replay["records"]) == 48
    text = json.dumps(value)
    assert fixture[0]["prose_material"] not in text
    assert all(t["trial_id"] not in text and t["world_id"] not in text for t in fixture[0]["trials"])
    assert "response_text" not in text and "prompt_token_ids" not in text


@pytest.mark.parametrize("mode,decision", [("floor", "both_materials_floor"), ("cap_floor", "both_materials_floor"),
    ("both_correct", "mixed"), ("baseline_failure", "inconclusive"), ("threshold", "descriptive_prose_substitution_recovery"),
    ("two_inert_correct", "mixed"), ("two_prose_failures", "mixed"), ("infrastructure", "inconclusive")])
def test_declared_thresholds_baseline_guard_and_no_adaptive_calls(fixture, mode, decision):
    value, model = execute(fixture, mode)
    assert model.calls == 48 and value["decision"] == decision
    if mode == "baseline_failure":
        assert value["descriptive_material_pattern"] == "descriptive_prose_substitution_recovery"
        assert not value["baseline_controls"]["competence_established"]
    if mode == "threshold":
        assert value["arms"]["inert"]["strict_correct"] == 1 and value["arms"]["prose"]["strict_correct"] == 15
    if mode == "cap_floor":
        assert value["arms"]["inert"]["capped"] == value["arms"]["prose"]["capped"] == 16
        assert value["coverage"]["parser_categories"]["format"] == 32
    if mode == "infrastructure":
        assert value["coverage"]["infrastructure_failed"] == 1
        assert value["primary_prose_minus_inert"]["world_mean_difference_pp"] is None
        assert value["primary_prose_minus_inert"]["worst_best_bounds_pp"] == [93.75, 100]


@pytest.mark.parametrize("mode,section,high,low", [("placement", "secondary_placement_contrasts", "before", "after"),
    ("order", "secondary_order_contrasts", "inert_then_prose", "prose_then_inert")])
def test_balanced_stratum_contrasts_retain_four_worlds_eight_pairs(fixture, mode, section, high, low):
    value, _ = execute(fixture, mode)
    assert value["primary_prose_minus_inert"]["world_mean_difference_pp"] == 50
    assert value[section][high]["world_mean_difference_pp"] == 100
    assert value[section][low]["world_mean_difference_pp"] == 0
    assert all(r["planned_worlds"] == 4 and r["planned_material_pairs"] == 8 for r in value[section].values())


def test_fixed_world_mean_uses_four_worlds(fixture):
    value, _ = execute(fixture, "world")
    assert value["primary_prose_minus_inert"]["world_mean_difference_pp"] == 25
    assert value["primary_prose_minus_inert"]["identified_worlds"] == 4


def test_late_interruption_preserves_all_pairs_and_pending_attempt(fixture):
    value, model = execute(fixture, interrupt_after=47)
    assert model.calls == 47 and value["status"] == "incomplete" and value["decision"] == "inconclusive"
    assert value["coverage"]["missing"] == value["interrupted_attempts"] == 1
    contrast = value["primary_prose_minus_inert"]
    assert contrast["planned_material_pairs"] == 16 and contrast["identified_material_pairs"] == 15
    assert contrast["world_mean_difference_pp"] is None and contrast["worst_best_bounds_pp"] == [93.75, 100]
    assert contrast["identified_worlds"] == 3
    assert export(fixture)["aggregate"] == value


def test_load_deadline_retains_all_unresolved_cells(fixture):
    value, model = execute(fixture, loader_error=a.A167Deadline())
    assert model.calls == 0 and value["coverage"]["missing"] == 48 and value["unattempted_cells"] == 48
    assert value["primary_prose_minus_inert"]["worst_best_bounds_pp"] == [-100, 100]
    assert export(fixture)["aggregate"] == value


def test_consumed_process_and_output_reject_without_second_model(fixture, monkeypatch):
    execute(fixture)
    def forbidden(*args):
        pytest.fail("duplicate model load")
    with pytest.raises(ValueError, match="fresh_process_required"):
        a.run_plan(**{**fixture[1], "output_root": fixture[1]["output_root"].with_name("second")}, reference_loader=forbidden)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1], reference_loader=forbidden)


def test_writer_lock_prevents_duplicate_execution_and_export(fixture):
    root = fixture[1]["output_root"]
    root.mkdir()
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[1])
        with pytest.raises(BlockingIOError):
            export(fixture)


@pytest.mark.parametrize("mutation", ["prompt", "eos", "response", "tokens", "score", "score_type", "native", "matched", "orphan", "unknown", "export"])
def test_source_native_scorer_and_receipt_drift_fail_closed(fixture, mutation):
    execute(fixture)
    root = fixture[1]["output_root"]
    trial = fixture[0]["trials"][0]
    folder = root / "trials" / trial["trial_id"]
    path = folder / "result.json"
    if mutation in ("prompt", "eos"):
        path = folder / "attempt.json"
    elif mutation == "native":
        path = root / "native-inputs.json"
    elif mutation == "matched":
        path = root / "matched-counts.json"
    elif mutation == "export":
        path = root / "records.json"
    if mutation == "orphan":
        (folder / "attempt.json").unlink()
    elif mutation == "unknown":
        (root / "trials" / ("0" * 24)).mkdir()
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
        elif mutation == "native":
            value[trial["trial_id"]]["prompt_token_ids"][0] += 1
        elif mutation == "matched":
            value[0]["prompt_token_counts"]["prose"] += 1
        elif mutation == "export":
            value.pop()
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["a166", "a165", "protocol", "model", "reference", "memory", "config"])
def test_preflight_drift_before_model_or_attempt(fixture, monkeypatch, mutation):
    args = fixture[1]
    original = a.native.engine.file_digest
    if mutation in ("a166", "a165"):
        filename = "instruction_selection_cpu_state.py" if mutation == "a166" else "instruction_selection_cpu_development.py"
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p).name == filename else original(p))
    elif mutation == "protocol":
        args["protocol_path"].write_text("changed")
    elif mutation == "model":
        monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda p: {})
    elif mutation == "reference":
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p) == args["reference_script"] else original(p))
    elif mutation == "config":
        value = json.loads(args["config_path"].read_bytes())
        value["seed"] += 1
        args["config_path"].write_bytes(a.tasks.canonical(value))
        args["config_sha256"] = original(args["config_path"])
    def forbidden(*args):
        pytest.fail("preflight drift loaded a model")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - (mutation == "memory"),
                                      load_cpu_reference=forbidden)
    with pytest.raises(ValueError):
        a.run_plan(**args, reference_loader=lambda p: reference)
    assert not list(args["output_root"].glob("trials/*/attempt.json"))


def test_signal_baseexceptions_and_timer_restoration():
    original = signal.getsignal(signal.SIGTERM)
    with a.bounded_signals():
        assert 3590 < signal.getitimer(signal.ITIMER_REAL)[0] <= 3600
        with pytest.raises(a.A167Interrupted):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with pytest.raises(a.A167Deadline):
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    assert not issubclass(a.A167Interrupted, Exception)
    assert signal.getsignal(signal.SIGTERM) == original and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_main_redirects_private_logs_and_prints_only_aggregate(fixture, monkeypatch, capsys):
    model = FakeRuntime(fixture[0])
    def load(config, engine):
        print("private synthetic loading")
        sys.stderr.write("private synthetic progress\n")
        return model
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    original = a.run_plan
    monkeypatch.setattr(a, "run_plan", lambda **kw: original(**kw, reference_loader=lambda p: reference,
                        tokenizer_loader=lambda config: Tokenizer()))
    argv = []
    for key, value in fixture[1].items():
        if key != "tokenizer_loader":
            argv += ["--" + key.removesuffix("_path").replace("_", "-"), str(value)]
    assert a.main(argv) == 0
    captured = capsys.readouterr()
    assert "private synthetic" not in captured.out + captured.err
    assert json.loads(captured.out)["planned_cells"] == 48
    assert all(t["trial_id"] not in captured.out for t in fixture[0]["trials"])
    assert "private synthetic loading" in (fixture[1]["output_root"] / "execution.log").read_text()


def test_import_has_no_model_library_access(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "accelerate", "bitsandbytes"}:
            pytest.fail("model library imported")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("lexical_prompt_study.a167_import_test", Path(a.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._PROCESS_CONSUMED is False
