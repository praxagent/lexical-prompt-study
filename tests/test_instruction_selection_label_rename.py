"""Synthetic A168 contracts; no real material, model or study-response access."""
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

from lexical_prompt_study import instruction_selection_label_rename as a

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a168_synthetic_helpers", ROOT / "tests/test_instruction_selection_prose_control.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
Tokenizer = helper.Tokenizer


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prior, old_args = helper.fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    material = "A calm cloud passes.".ljust(len("Public synthetic inert"))
    prior = a.prose.compile_plan(prior["a165_plan"], prior["a165_run_header"], material,
                                a.native.engine.file_digest(old_args["protocol_path"]))
    old_args["plan_path"].write_bytes(a.tasks.canonical(prior))
    old_args["plan_sha256"] = a.native.engine.file_digest(old_args["plan_path"])
    old_inputs = a.prose._inputs(**{key: value for key, value in old_args.items() if key != "output_root"})
    _, config, sources, _, eos, prepared, matched = old_inputs
    header = {**a.prose._fixed_header(prior, config, sources, eos, prepared, matched,
        old_args["plan_sha256"], old_args["config_sha256"], old_args["reference_script"]),
        "pid": 123, "available_memory_bytes": a.MIN_AVAILABLE_BYTES}
    protocol = tmp_path / "a168.md"
    protocol.write_text("Public synthetic fixed label-renaming protocol")
    plan = a.compile_plan(prior, header, a.native.engine.file_digest(protocol))
    plan_path = tmp_path / "a168-plan.json"
    plan_path.write_bytes(a.tasks.canonical(plan))
    reference = tmp_path / "distinct-a168-reference.py"
    reference.write_text("# public synthetic separately frozen reference")
    original_digest = a.native.engine.file_digest
    monkeypatch.setattr(a.native.engine, "file_digest", lambda p: a.REFERENCE_SHA if Path(p) == reference else original_digest(p))
    args = {**old_args, "plan_path": plan_path, "plan_sha256": a.native.engine.file_digest(plan_path),
            "protocol_path": protocol, "reference_script": reference, "output_root": tmp_path / "a168-output"}
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
        phase, pair = trial["diagnostic_phase"], trial["pair_index"]
        selector = trial["selector"]
        other = "B" if selector == "A" else "A"
        if self.mode == "infrastructure_" + phase and pair == 0:
            raise RuntimeError("private synthetic error must not appear")
        kind = "selected_display_label" if phase == "original" else "correct_value"
        if self.mode == "all_correct":
            kind = "correct_value"
        elif self.mode == "role_transfer":
            kind = "selected_display_label"
        elif self.mode == "classes" and phase == "renamed":
            kind = a.RESPONSE_CLASSES[pair % len(a.RESPONSE_CLASSES)]
        elif self.mode == "world" and phase == "renamed" and trial["world_index"] > 0:
            kind = "other_given_value"
        elif self.mode == "original_other_label" and phase == "original":
            kind = "other_display_label"
        values = {"correct_value": trial["selected_answer"], "other_given_value": trial["unselected_answer"],
            "selected_display_label": trial["displayed_label_mapping"][selector],
            "other_display_label": trial["displayed_label_mapping"][other], "legacy_selected_label": selector,
            "legacy_other_label": other, "other_allowed_symbol": next(x for x in a.tasks.SYMBOLS
                if x not in (trial["selected_answer"], trial["unselected_answer"])), "other_string": "synthetic",
            "format_or_cap": "unused"}
        text = json.dumps({"answer": values[kind]})
        if kind == "format_or_cap":
            text = "[]"
        if self.mode == "cap" and phase == "renamed":
            text = json.dumps({"answer": trial["selected_answer"]}).ljust(64)
            return self.tokenizer.encode(text), text, "length"
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
    result = a.run_plan(**fixture[1], reference_loader=lambda path: reference)
    assert len(loaded) == 1
    return result, model


def export(fixture):
    return a.export_run(**fixture[1])


def preparation(fixture):
    return a.prepare_inputs(**{key: value for key, value in fixture[1].items() if key != "output_root"})


def test_all_original_prose_pairs_new_identities_and_balanced_order(fixture):
    plan = fixture[0]
    originals = [t for t in plan["a167_plan"]["trials"] if t["diagnostic_phase"] == "prose"]
    groups = [plan["trials"][i:i + 2] for i in range(0, 32, 2)]
    assert len(plan["trials"]) == len({t["trial_id"] for t in plan["trials"]}) == 32
    assert [(g[0]["world_id"], g[0]["selector"], g[0]["paired_placement"]) for g in groups] == [
        (t["world_id"], t["selector"], t["paired_placement"]) for t in originals]
    assert all(sum(g[0]["label_order"] == order for g in groups) == 8 for order in a.ORDERS)
    for group in groups:
        first = group[0]
        order = a.ORDERS[(first["world_index"] + first["selector_index"] + first["placement_index"]) % 2]
        assert [t["diagnostic_phase"] for t in group] == order.split("_then_")
    for factor in ("world_index", "selector_index", "placement_index"):
        for value in {g[0][factor] for g in groups}:
            subset = [g for g in groups if g[0][factor] == value]
            assert all(sum(g[0]["label_order"] == order for g in subset) == len(subset) // 2 for order in a.ORDERS)
    assert not {t["trial_id"] for t in originals} & {t["trial_id"] for t in plan["trials"]}


def test_rename_scope_preserves_article_prose_value_order_and_internal_oracle(fixture):
    plan = fixture[0]
    originals = {t["trial_id"]: t for t in plan["a167_plan"]["trials"]}
    material = plan["a167_plan"]["prose_material"]
    assert material.startswith("A ")
    for trial in plan["trials"]:
        source = originals[trial["source_a167_trial_id"]]
        for key in ("selector", "world", "query_order", "selected_answer", "unselected_answer", "placement"):
            assert trial[key] == source[key]
        if trial["diagnostic_phase"] == "original":
            assert trial["messages"] == source["messages"]
            continue
        system = source["messages"][0]["content"].replace("selector is A.", "selector is C.").replace(
            "selector is B.", "selector is D.").replace("labeled A and B.", "labeled C and D.")
        assert trial["messages"][0]["content"] == system
        user = trial["messages"][1]["content"]
        task = user[len(material) + 2:] if trial["paired_placement"] == "before" else user[:-len(material) - 2]
        preserved = user[:len(material)] if trial["paired_placement"] == "before" else user[-len(material):]
        assert preserved == material
        table = json.loads(task)
        assert list(table) == [a.DISPLAY_LABELS["renamed"][key] for key in trial["query_order"]]
        assert table == {a.DISPLAY_LABELS["renamed"][key]: value for key, value in trial["world"]["answers"].items()}


def test_native_matching_and_parent_header_allows_distinct_frozen_reference_path(fixture):
    plan, args = fixture
    assert plan["a167_run_header"]["reference_script_path"] != str(args["reference_script"])
    prepared = preparation(fixture)
    assert prepared["planned_cells"] == len(prepared["native_inputs"]) == 32
    assert len(prepared["matched_count_receipts"]) == 16
    assert all(r["prompt_token_counts"]["original"] == r["prompt_token_counts"]["renamed"]
               for r in prepared["matched_count_receipts"])
    assert not args["output_root"].exists() and a._PROCESS_CONSUMED is False


def test_actual_native_token_count_mismatch_blocks_before_any_call(fixture):
    class UnequalTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return super().encode(text.replace('"C":', '"CC":'), **kwargs)
    fixture[1]["tokenizer_loader"] = lambda config: UnequalTokenizer()
    with pytest.raises(ValueError, match="exact_renamed_prompt_token_match"):
        preparation(fixture)
    assert not fixture[1]["output_root"].exists()


def test_original_native_identity_guard(fixture, monkeypatch):
    original = a.native.prepare_generation
    def changed(tokenizer, trial, config):
        value = original(tokenizer, trial, config)
        if trial.get("diagnostic_phase") == "original":
            value["prompt_token_ids"][0] += 1
        return value
    monkeypatch.setattr(a.native, "prepare_generation", changed)
    with pytest.raises(ValueError, match="original_native_identity"):
        preparation(fixture)


@pytest.mark.parametrize("mutation", ["system", "prose", "selector", "phase", "mapping", "order", "subset", "source"])
def test_fixed_plan_rejects_selection_prompt_and_identity_drift(fixture, mutation):
    plan = copy.deepcopy(fixture[0])
    if mutation in ("system", "prose"):
        plan["trials"][0]["messages"][mutation == "prose"]["content"] += " changed"
    elif mutation == "selector":
        plan["trials"][0]["selector"] = "C"
    elif mutation == "phase":
        plan["trials"][0]["diagnostic_phase"] = "renamed"
    elif mutation == "mapping":
        plan["trials"][0]["displayed_label_mapping"]["A"] = "Z"
    elif mutation == "order":
        plan["trials"][:2] = reversed(plan["trials"][:2])
    elif mutation == "subset":
        plan["trials"].pop()
    else:
        plan["bindings"]["a168_source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_complete_recovery_replay_four_world_means_and_private_aggregate(fixture):
    result, model = execute(fixture)
    assert model.calls == 32 and result["status"] == "complete"
    primary = result["primary_renamed_minus_original"]
    assert primary["planned_pairs"] == primary["identified_pairs"] == 16
    assert primary["world_mean_difference_pp"] == 100
    assert [world["difference_pp"] for world in primary["fixed_world_means"]] == [100] * 4
    subset = result["selected_label_error_transfer"]
    assert subset["known_members_K"] == 16 and subset["unknown_membership_pairs"] == 0
    assert subset["known_member_category_rate_bounds_pp"]["correct_value"] == [100, 100]
    assert export(fixture)["aggregate"] == result
    text = json.dumps(result)
    assert all(t["trial_id"] not in text and t["world_id"] not in text for t in fixture[0]["trials"])
    assert "response_text" not in text and "generated_token_ids" not in text
    assert fixture[0]["a167_plan"]["prose_material"] not in text


def test_exclusive_response_classes_from_validated_private_json(fixture):
    result, model = execute(fixture, "classes")
    assert model.calls == 32
    counts = result["response_classes"]["renamed"]["counts"]
    assert counts == {key: 2 if index < 7 else 1 for index, key in enumerate(a.RESPONSE_CLASSES)}
    assert sum(counts.values()) == 16
    originals = result["response_classes"]["original"]["counts"]
    assert originals["selected_display_label"] == 16
    assert originals["legacy_selected_label"] == originals["legacy_other_label"] == 0
    assert result["selected_label_error_transfer"]["known_member_counterparts"]["counts"] == counts


@pytest.mark.parametrize("mode", ["role_transfer", "original_other_label", "all_correct", "world", "cap"])
def test_secondary_patterns_without_mechanism_threshold_or_expansion(fixture, mode):
    result, model = execute(fixture, mode)
    assert model.calls == 32 and "decision" not in result
    if mode == "world":
        primary = result["primary_renamed_minus_original"]
        assert primary["world_mean_difference_pp"] == 25
        assert [w["difference_pp"] for w in primary["fixed_world_means"]] == [100, 0, 0, 0]
    if mode in ("all_correct", "original_other_label"):
        subset = result["selected_label_error_transfer"]
        assert subset["known_members_K"] == 0 and subset["status"] == "uninformative_K_zero"
        assert all(rate is None for rate in subset["known_member_category_rate_bounds_pp"].values())
    if mode == "cap":
        assert result["response_classes"]["renamed"]["counts"]["format_or_cap"] == 16
        assert result["arms"]["renamed"]["capped"] == 16
        assert result["arms"]["renamed"]["strict_correct"] == 0


@pytest.mark.parametrize("phase", ["original", "renamed"])
def test_infrastructure_failure_runs_remaining_schedule_and_preserves_membership_bounds(fixture, phase):
    result, model = execute(fixture, "infrastructure_" + phase)
    assert model.calls == 32 and result["status"] == "incomplete"
    assert result["arms"][phase]["infrastructure_failed"] == 1
    primary = result["primary_renamed_minus_original"]
    assert primary["planned_pairs"] == 16 and primary["world_mean_difference_pp"] is None
    assert primary["worst_best_bounds_pp"] == [93.75, 100]
    subset = result["selected_label_error_transfer"]
    if phase == "original":
        assert subset["known_members_K"] == 15 and subset["unknown_membership_pairs"] == 1
        assert subset["membership_count_bounds"] == [15, 16]
        assert subset["unknown_membership_counterparts"]["counts"]["correct_value"] == 1
        assert subset["possible_full_subset_category_rate_bounds_pp"]["correct_value"] == [93.75, 100]
    else:
        assert subset["known_members_K"] == 16 and subset["unknown_membership_pairs"] == 0
        assert subset["known_member_counterparts"]["unresolved"] == 1
        assert subset["known_member_category_rate_bounds_pp"]["correct_value"] == [93.75, 100]


def test_partial_schedule_keeps_all_pairs_and_missing_counterpart(fixture):
    result, model = execute(fixture, interrupt_after=1)
    assert model.calls == 1 and result["interrupted_attempts"] == 1 and result["unattempted_cells"] == 30
    assert result["primary_renamed_minus_original"]["worst_best_bounds_pp"] == [-93.75, 100]
    subset = result["selected_label_error_transfer"]
    assert subset["known_members_K"] == 1 and subset["unknown_membership_pairs"] == 15
    assert subset["known_member_counterparts"]["unresolved"] == 1
    assert subset["known_member_category_rate_bounds_pp"]["selected_display_label"] == [0, 100]
    assert export(fixture)["aggregate"] == result


def test_deadline_during_load_keeps_unknown_membership_and_all_denominators(fixture):
    result, model = execute(fixture, loader_error=a.A168Deadline())
    assert model.calls == 0 and result["coverage"]["missing"] == 32
    assert result["primary_renamed_minus_original"]["worst_best_bounds_pp"] == [-100, 100]
    subset = result["selected_label_error_transfer"]
    assert subset["known_members_K"] == 0 and subset["unknown_membership_pairs"] == 16
    assert subset["status"] == "inconclusive_membership" and subset["membership_count_bounds"] == [0, 16]
    assert subset["possible_full_subset_category_rate_bounds_pp"]["correct_value"] == [0, 100]
    assert result["response_classes"]["original"]["count_bounds"]["legacy_selected_label"] == [0, 0]
    assert json.loads((fixture[1]["output_root"] / "execution-finished.json").read_bytes())["status"] == "deadline"
    assert export(fixture)["aggregate"] == result


def test_process_root_and_flock_prevent_duplicate_execution(fixture, monkeypatch):
    execute(fixture)
    def forbidden(*args):
        pytest.fail("duplicate run loaded reference")
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**{**fixture[1], "output_root": fixture[1]["output_root"].with_name("second")}, reference_loader=forbidden)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1], reference_loader=forbidden)
    with (fixture[1]["output_root"] / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            export(fixture)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[1], reference_loader=forbidden)


@pytest.mark.parametrize("mutation", ["attempt", "attempt_boolean", "decoded", "score_type", "native", "matched", "export", "export_boolean", "orphan", "unknown"])
def test_receipt_replay_rejects_drift_and_orphan_results(fixture, mutation):
    execute(fixture)
    root = fixture[1]["output_root"]
    trial = fixture[0]["trials"][0]
    folder = root / "trials" / trial["trial_id"]
    if mutation == "orphan":
        (folder / "attempt.json").unlink()
    elif mutation == "unknown":
        (root / "trials" / "unknown").mkdir()
    else:
        path = (folder / "attempt.json" if mutation in ("attempt", "attempt_boolean") else root / "native-inputs.json" if mutation == "native"
                else root / "matched-counts.json" if mutation == "matched" else root / "records.json" if mutation in ("export", "export_boolean")
                else folder / "result.json")
        value = json.loads(path.read_bytes())
        if mutation == "attempt":
            value["sequence_index"] = 5
        elif mutation == "attempt_boolean":
            value["sequence_index"] = False
        elif mutation == "decoded":
            value["private"]["response_text"] += " "
        elif mutation == "score_type":
            value["record"]["score"]["strict_correct"] = 0
        elif mutation == "native":
            value[trial["trial_id"]]["prompt_token_ids"][0] += 1
        elif mutation == "matched":
            value[0]["prompt_token_counts"]["renamed"] += 1
        elif mutation == "export_boolean":
            value[0]["score"]["strict_correct"] = 0
        else:
            value.pop()
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["a167", "a166", "protocol", "model", "reference", "memory", "config", "parent_native"])
def test_source_config_model_and_parent_freeze_drift_block_before_model_calls(fixture, monkeypatch, mutation):
    args = fixture[1]
    original = a.native.engine.file_digest
    if mutation in ("a167", "a166"):
        filename = "instruction_selection_prose_control.py" if mutation == "a167" else "instruction_selection_cpu_state.py"
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p).name == filename else original(p))
    elif mutation == "protocol":
        args["protocol_path"].write_text("changed")
    elif mutation == "model":
        monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: {})
    elif mutation == "reference":
        monkeypatch.setattr(a.native.engine, "file_digest", lambda p: "0" * 64 if Path(p) == args["reference_script"] else original(p))
    elif mutation == "config":
        value = json.loads(args["config_path"].read_bytes())
        value["seed"] += 1
        args["config_path"].write_bytes(a.tasks.canonical(value))
        args["config_sha256"] = original(args["config_path"])
    elif mutation == "parent_native":
        plan = fixture[0]
        header = copy.deepcopy(plan["a167_run_header"])
        header["native_prepared_sha256"] = "0" * 64
        replacement = a.compile_plan(plan["a167_plan"], header, plan["bindings"]["protocol_sha256"])
        args["plan_path"].write_bytes(a.tasks.canonical(replacement))
        args["plan_sha256"] = original(args["plan_path"])
    def forbidden(*args):
        pytest.fail("preflight drift loaded model")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - (mutation == "memory"),
                                      load_cpu_reference=forbidden)
    with pytest.raises(ValueError):
        a.run_plan(**args, reference_loader=lambda path: reference)
    assert not list(args["output_root"].glob("trials/*/attempt.json"))


def test_alarm_and_signal_baseexceptions_restore_handlers():
    original = signal.getsignal(signal.SIGTERM)
    with a.bounded_signals():
        assert 2390 < signal.getitimer(signal.ITIMER_REAL)[0] <= 2400
        with pytest.raises(a.A168Interrupted):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with pytest.raises(a.A168Deadline):
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    assert not issubclass(a.A168Interrupted, Exception) and not issubclass(a.A168Deadline, Exception)
    assert signal.getsignal(signal.SIGTERM) == original and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_cli_prints_aggregate_only_and_redirects_private_logs(fixture, monkeypatch, capsys):
    model = FakeRuntime(fixture[0])
    def load(config, engine):
        print("private synthetic model loading")
        sys.stderr.write("private synthetic progress\n")
        return model
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    original = a.run_plan
    monkeypatch.setattr(a, "run_plan", lambda **kwargs: original(**kwargs, reference_loader=lambda path: reference,
                        tokenizer_loader=lambda config: Tokenizer()))
    argv = []
    for key, value in fixture[1].items():
        if key != "tokenizer_loader":
            argv += ["--" + key.removesuffix("_path").replace("_", "-"), str(value)]
    assert a.main(argv) == 0
    captured = capsys.readouterr()
    assert "private synthetic" not in captured.out + captured.err
    assert json.loads(captured.out)["planned_cells"] == 32
    assert all(t["trial_id"] not in captured.out for t in fixture[0]["trials"])
    assert "private synthetic model loading" in (fixture[1]["output_root"] / "execution.log").read_text()


def test_import_has_no_model_library_access(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "accelerate", "bitsandbytes"}:
            pytest.fail("model library imported")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("lexical_prompt_study.a168_import_test", Path(a.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._PROCESS_CONSUMED is False
