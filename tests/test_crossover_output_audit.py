"""A175 public synthetic tests; never inspect parent experimental responses."""

import copy
import itertools
import json
import os
import signal
from collections import Counter
from pathlib import Path

import pytest

from lexical_prompt_study import crossover_output_audit as a


TRIAL = {"selector": "A", "selected_answer": "public_v1", "unselected_answer": "public_v2"}
OBJECT_CATEGORIES = ("selected_mapped_value", "other_mapped_value", "selected_label", "other_label", "other_string")
FEATURES = ("fence", "any_answer_object", "repetition")
EOS = [999]
PLAIN_TOKENS = [10, 11, 12, 13, 14, 999]


def inspect(text, tokens=None, trial=None, eos_ids=None):
    return a.inspect_output(text, PLAIN_TOKENS if tokens is None else tokens,
                            TRIAL if trial is None else trial, EOS if eos_ids is None else eos_ids)


def answer(value):
    return json.dumps({"answer": value}, separators=(",", ":"))


def assert_object_counts(features, expected):
    counts = {category: expected.get(category, 0) for category in OBJECT_CATEGORIES}
    assert features["answer_object_counts"] == counts
    assert features["answer_object_presence"] == {key: value > 0 for key, value in counts.items()}
    assert features["answer_object_count"] == sum(counts.values())
    assert features["any_answer_object"] is (sum(counts.values()) > 0)


@pytest.mark.parametrize("text,expected", [
    ("plain public text", False), ("` one and ~~ two", False),
    ("inline ``` marker", True), ("~~~", True), ("````", True), ("~~~~", True),
    ("'```' quoted", True), ("` ` ` spaced", False), ("～" * 3, False),
])
def test_exact_ascii_fence_presence(text, expected):
    assert inspect(text)["fence"] is expected


@pytest.mark.parametrize("selector", ["A", "B"])
@pytest.mark.parametrize("kind", OBJECT_CATEGORIES)
def test_all_object_categories_use_exact_parent_oracle(selector, kind):
    trial = {**TRIAL, "selector": selector}
    value = {"selected_mapped_value": trial["selected_answer"], "other_mapped_value": trial["unselected_answer"],
             "selected_label": selector, "other_label": "B" if selector == "A" else "A", "other_string": "unrecognized"}[kind]
    features = inspect(answer(value), trial=trial)
    assert_object_counts(features, {kind: 1})
    assert features["fence"] is False and features["repetition"] is False and features["overlap"] == "010"


@pytest.mark.parametrize("text", [
    '{"answer":"public_v1","answer":"public_v2"}', '{"answer":"public_v1","extra":0}',
    '{"answer":true}', '{"answer":null}', '{"answer":23}', '{"ANSWER":"public_v1"}',
    '{"answer":[]}', '{"answer":NaN}', '{"answer":Infinity}', '{"answer":-Infinity}',
    '{"answer":"public_v1"', '{answer:"public_v1"}', '{"answer":"public_v1",}',
    '{"answer":"bad\ncontrol"}', '"public_v1"', '[]', '',
])
def test_nonmatching_or_malformed_candidate_offsets_do_not_count(text):
    assert_object_counts(inspect(text), {})


@pytest.mark.parametrize("text", [
    'prefix {"answer":"public_v1"} trailing words',
    '```json\n{"answer":"public_v1"}\n```',
    'The quoted example is \'{"answer":"public_v1"}\'.',
    '[{"answer":"public_v1"}]',
    '{"outer":{"answer":"public_v1"},"extra":0}',
    '{"answer":{"answer":"public_v1"}}',
    '{"x":{"answer":"public_v1"},"x":1}',
    '{broken prefix {"answer":"public_v1"}',
    '{ "answer" : "public_v1" }',
])
def test_every_literal_brace_is_scanned_independently(text):
    assert_object_counts(inspect(text), {"selected_mapped_value": 1})


def test_escaped_string_is_not_unescaped_or_repaired():
    assert_object_counts(inspect(json.dumps(answer("public_v1"))), {})
    assert_object_counts(inspect('{"answer":"literal { text"}'), {"other_string": 1})


def test_object_totals_and_presence_are_distinct_and_overlap():
    text = " ".join([answer("public_v1"), answer("public_v1"), answer("public_v2"),
                     answer("A"), answer("B"), answer("other"), answer("other")])
    features = inspect(text)
    assert_object_counts(features, {"selected_mapped_value": 2, "other_mapped_value": 1,
                                    "selected_label": 1, "other_label": 1, "other_string": 2})
    assert features["answer_object_count"] == 7 and sum(features["answer_object_presence"].values()) == 5
    assert features["overlap"] == "010"


def test_case_sensitive_values_and_escaped_json_characters():
    text = answer("PUBLIC_V1") + answer("a") + answer(" public_v1") + answer("public_v1 ") + '{"answer":"public_\\u00761"}'
    assert_object_counts(inspect(text), {"selected_mapped_value": 1, "other_string": 4})


def test_recursion_error_at_outer_offsets_does_not_hide_inner_object():
    text = '{"nest":' * 1100 + answer("public_v1") + '}' * 1100
    assert_object_counts(inspect(text), {"selected_mapped_value": 1})


@pytest.mark.parametrize("length,expected", [(1, False), (3, False), (4, False), (6, False), (7, True), (8, True)])
def test_overlapping_four_gram_threshold(length, expected):
    assert inspect("public text", tokens=[7] * length + [999])["repetition"] is expected


def test_four_gram_recurrence_need_not_be_consecutive():
    three = [1, 2, 3, 4, 20, 1, 2, 3, 4, 21, 1, 2, 3, 4, 22]
    four = three + [1, 2, 3, 4, 23]
    assert inspect("public text", tokens=three)["repetition"] is False
    assert inspect("public text", tokens=four)["repetition"] is True
    assert inspect("public text", tokens=[1, 2, 3, 4] * 4)["repetition"] is True
    assert inspect("public text", tokens=([1, 2, 3, 4] * 4)[:-1])["repetition"] is False


def test_remove_only_one_final_eos_and_preserve_every_earlier_token():
    assert inspect("public text", tokens=[9] * 7, eos_ids=[9])["repetition"] is False
    assert inspect("public text", tokens=[9] * 8, eos_ids=[9])["repetition"] is True
    assert inspect("public text", tokens=[9] * 7 + [8], eos_ids=[9])["repetition"] is True
    assert inspect("public text", tokens=[9], eos_ids=[9])["repetition"] is False
    assert inspect("public text", tokens=[7] * 7 + [999], eos_ids=[999, 1000])["repetition"] is True


def test_detector_never_mutates_original_arrays_or_oracle():
    tokens = [7] * 7 + [999]
    trial = copy.deepcopy(TRIAL)
    token_copy, trial_copy = tokens.copy(), copy.deepcopy(trial)
    inspect(answer(trial["selected_answer"]), tokens=tokens, trial=trial)
    assert tokens == token_copy and trial == trial_copy


def test_exact_numeric_feature_schema_and_all_eight_overlaps():
    assert tuple(a.OBJECT_CATEGORIES) == OBJECT_CATEGORIES and tuple(a.FEATURES) == FEATURES
    overlaps = Counter()
    for fence, obj, repetition in itertools.product((False, True), repeat=3):
        text = ("``` " if fence else "") + (answer("public_v1") if obj else "public text")
        tokens = [7] * 7 if repetition else [1, 2, 3, 4]
        features = inspect(text, tokens=tokens)
        assert set(features) == {"fence", "any_answer_object", "repetition", "answer_object_count",
                                 "answer_object_counts", "answer_object_presence", "overlap"}
        assert [features[key] for key in FEATURES] == [fence, obj, repetition]
        assert features["overlap"] == "".join(str(int(value)) for value in (fence, obj, repetition))
        overlaps[features["overlap"]] += 1
        serialized = json.dumps(features)
        assert "public_v1" not in serialized and "response_text" not in serialized and "token" not in serialized
    assert overlaps == {"".join(code): 1 for code in itertools.product("01", repeat=3)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.canonical(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Wholly synthetic parent and freeze; no real A174 artifacts are opened."""
    parent, root = tmp_path / "synthetic-parent", tmp_path / "synthetic-audit"
    parent.mkdir()
    root.mkdir()
    monkeypatch.setattr(a, "PARENT_ROOT", parent)
    monkeypatch.setattr(a, "AUDIT_ROOT", root)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    trials = []
    for world, instruction_core, closing_constraint, selector in itertools.product(range(4), a.INSTRUCTION_CORES, a.CLOSING_CONSTRAINTS, ("A", "B")):
        index = len(trials)
        trials.append({"trial_id": f"{index:024x}", "sequence_index": index, "world_index": world,
                       "instruction_core": instruction_core, "closing_constraint": closing_constraint, "selector": selector,
                       "selected_answer": "public_v1" if selector == "A" else "public_v2",
                       "unselected_answer": "public_v2" if selector == "A" else "public_v1"})
    parent_plan = {"schema_version": "a174-plan-v1", "trials": trials}
    write_json(parent / "inputs/plan.json", parent_plan)
    write_json(parent / "inputs/runtime-config.json", {"public_synthetic_configuration": True})
    header = {"schema_version": "a174-run-v1", "plan_sha256": a.file_sha(parent / "inputs/plan.json"),
              "config_sha256": a.file_sha(parent / "inputs/runtime-config.json"), "eos_token_ids": EOS}
    write_json(parent / "execution/run.json", header)
    (parent / "execution/.lock").touch()
    records, raw_results, result_hashes = [], {}, {}
    for trial in trials:
        index = trial["sequence_index"]
        capped = trial["instruction_core"] == trial["closing_constraint"] == "a171_standard"
        correct = trial["instruction_core"] == "legacy"
        text = (("``` " if index % 8 & 4 else "") + (answer(trial["selected_answer"]) if index % 8 & 2 else "")
                + " trailing public fixture text")
        tokens = [7] * 64 if index % 8 & 1 else list(range(20, 84))
        if correct:
            text, tokens = answer(trial["selected_answer"]), PLAIN_TOKENS.copy()
        elif not capped:
            tokens = tokens[:15] + [999]
        score = {"category": "correct" if correct else "cap" if capped else "other_or_format",
                 "strict_correct": correct, "selected_label": False,
                 "format_valid": correct, "eos_valid": not capped, "capped": capped, "attempted": True,
                 "fence_marker": "```" in text or "~~~" in text,
                 "reason": None if correct else "token_limit_without_terminal_eos" if capped else "invalid_format",
                 "response_text": text, "generated_token_count": len(tokens)}
        record = {"trial_id": trial["trial_id"], "sequence_index": index, "instruction_core": trial["instruction_core"],
                  "closing_constraint": trial["closing_constraint"], "selector": trial["selector"],
                  **{key: value for key, value in score.items() if key != "response_text"}}
        records.append(record)
        attempt = {"schema_version": "a174-attempt-v1", "run_sha256": a.object_sha(header),
                   "trial_id": trial["trial_id"], "trial_sha256": a.object_sha(trial)}
        result = {"schema_version": "a174-result-v1", "status": "completed", "error": None,
                  "attempt_sha256": a.object_sha(attempt), "score": score, "generated_token_ids": tokens}
        folder = parent / "execution/trials" / trial["trial_id"]
        write_json(folder / "attempt.json", attempt)
        write_json(folder / "result.json", result)
        result_hashes[trial["trial_id"]] = a.file_sha(folder / "result.json")
        raw_results[trial["trial_id"]] = result
    coverage = {"attempted": 32, "completed": 32, "infrastructure_failed": 0,
                "interrupted": 0, "unattempted": 0, "missing": 0}
    analysis = {"coverage": coverage, "status": "complete", "execution_status": "finished_schedule",
                "numeric_records_sha256": a.object_sha(records), "plan_object_sha256": a.object_sha(parent_plan),
                "provenance": {"run_sha256": a.object_sha(header)}}
    write_json(parent / "execution/records.json", records)
    write_json(parent / "execution/analysis.json", analysis)
    write_json(parent / "execution/execution-finished.json", {"schema_version": "a174-execution-finished-v1",
        "status": "finished_schedule", "run_sha256": a.object_sha(header), "interruption": None})
    verifier_path = parent / "frozen/verification/independent_a174.py"
    verifier_path.parent.mkdir(parents=True)
    verifier_path.write_text("# Public synthetic parent verifier fixture.\n")
    write_json(parent / "REVIEW.json", {"status": "pass", "synthetic": True})
    write_json(parent / "NATIVE-AUDIT.json", {"status": "pass", "synthetic": True})
    parent_freeze = {"source_files_sha256": {"verification/independent_a174.py": a.file_sha(verifier_path)},
                     "input_sha256": {name: a.file_sha(parent / "inputs" / name) for name in ("plan.json", "runtime-config.json")},
                     "model_files_sha256": {"synthetic-model-metadata": "c" * 64},
                     "review_sha256": a.file_sha(parent / "REVIEW.json"),
                     "native_audit_sha256": a.file_sha(parent / "NATIVE-AUDIT.json")}
    write_json(parent / "FREEZE.json", parent_freeze)
    verification = {"schema_version": "a174-independent-verification-v1", "status": "pass", "coverage": coverage,
                    "terminal_status": "finished_schedule", "independent_native_constructions": 32,
                    "independent_completed_response_scores": 32, "independent_all_planned_rows_and_contrasts_replayed": True,
                    "model_forward_reexecuted": False, "source_files": 1, "input_files": 2, "model_files": 1,
                    "verifier_sha256": a.file_sha(verifier_path), "result_sha256": result_hashes}
    for key, relative in (("freeze_sha256", "FREEZE.json"), ("analysis_sha256", "execution/analysis.json"),
                          ("records_sha256", "execution/records.json"), ("run_sha256", "execution/run.json"),
                          ("execution_finished_sha256", "execution/execution-finished.json")):
        verification[key] = a.file_sha(parent / relative)
    write_json(parent / "VERIFICATION.json", verification)
    pins = {relative: a.file_sha(parent / relative) for relative in a.PARENT_ARTIFACT_SHA256}
    monkeypatch.setattr(a, "PARENT_ARTIFACT_SHA256", pins)
    bindings = {"root": str(parent), "artifact_sha256": pins, "result_sha256": result_hashes}
    repository = Path(__file__).absolute().parents[1]
    frozen_files = {
        "src/lexical_prompt_study/output_structure_audit.py": repository / "src/lexical_prompt_study/output_structure_audit.py",
        "src/lexical_prompt_study/crossover_output_audit.py": Path(a.__file__),
        "plans/crossover_output_audit_a175.md": repository / "plans/crossover_output_audit_a175.md",
        "tests/test_crossover_output_audit.py": Path(__file__),
    }
    for relative, source in frozen_files.items():
        target = root / "frozen" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    protocol = root / "frozen/plans/crossover_output_audit_a175.md"
    plan = a.compile_plan(bindings, a.file_sha(protocol), a.file_sha(Path(__file__)))
    write_json(root / "inputs/plan.json", plan)
    write_json(root / "REVIEW.json", {"status": "pass", "synthetic": True})
    freeze = {"schema_version": "a175-freeze-v1", "source_commit": "a" * 40, "pushed_commit": "a" * 40,
              "source_files_sha256": {relative: a.file_sha(root / "frozen" / relative) for relative in frozen_files},
              "input_sha256": {"plan.json": a.file_sha(root / "inputs/plan.json")},
              "plan_sha256": a.file_sha(root / "inputs/plan.json"), **plan["bindings"],
              "parent_verification_sha256": pins["VERIFICATION.json"], "target_model_calls": 0,
              "review_sha256": a.file_sha(root / "REVIEW.json")}
    write_json(root / "FREEZE.json", freeze)
    args = {"plan_path": root / "inputs/plan.json", "plan_sha256": a.file_sha(root / "inputs/plan.json"),
            "protocol_path": protocol, "freeze_path": root / "FREEZE.json", "freeze_sha256": a.file_sha(root / "FREEZE.json"),
            "output_root": root / "execution"}
    return {"parent": parent, "root": root, "trials": trials, "parent_records": records, "raw_results": raw_results,
            "bindings": bindings, "plan": plan, "args": args}


def numeric_records(fixture):
    result = []
    for trial, parent_record in zip(fixture["trials"], fixture["parent_records"], strict=True):
        original = fixture["raw_results"][trial["trial_id"]]
        features = inspect(original["score"]["response_text"], original["generated_token_ids"], trial)
        result.append({"trial_id": trial["trial_id"], "sequence_index": trial["sequence_index"],
                       "instruction_core": trial["instruction_core"], "closing_constraint": trial["closing_constraint"],
                       "original_category": parent_record["category"], "original_capped": parent_record["capped"], "features": features})
    return result


def independent_group(rows):
    feature_counts = {name: 0 for name in FEATURES}
    object_counts, presence_counts = dict.fromkeys(OBJECT_CATEGORIES, 0), dict.fromkeys(OBJECT_CATEGORIES, 0)
    overlap_counts = dict.fromkeys((f"{number:03b}" for number in range(8)), 0)
    total = 0
    for row in rows:
        features = row["features"]
        total += features["answer_object_count"]
        for name in FEATURES:
            feature_counts[name] += int(features[name])
        for name in OBJECT_CATEGORIES:
            object_counts[name] += features["answer_object_counts"][name]
            presence_counts[name] += int(features["answer_object_presence"][name])
        bits = "".join(str(int(features[name])) for name in FEATURES)
        overlap_counts[bits] += 1
    return {"outputs": len(rows), "planned_outputs": len(rows), "resolved_outputs": len(rows),
            "feature_presence": feature_counts, "answer_object_count": total, "answer_object_counts": object_counts,
            "answer_object_presence": presence_counts, "overlaps": overlap_counts}


def test_summaries_match_independent_counts_groups_and_empty_strata(fixture):
    records = numeric_records(fixture)
    summary = a.summarize(records)
    assert summary["overall"] == independent_group(records)
    assert summary["by_cap_status"]["cap"] == independent_group([r for r in records if r["original_capped"]])
    assert summary["by_cap_status"]["noncap"] == independent_group([r for r in records if not r["original_capped"]])
    assert summary["by_cap_status"]["cap"]["outputs"] == 8 and summary["by_cap_status"]["noncap"]["outputs"] == 24
    for instruction_core, closing_constraint in itertools.product(a.INSTRUCTION_CORES, a.CLOSING_CONSTRAINTS):
        subset = [row for row in records if (row["instruction_core"], row["closing_constraint"]) == (instruction_core, closing_constraint)]
        group = summary["by_condition"][instruction_core][closing_constraint]
        assert group["all"] == independent_group(subset) and group["all"]["outputs"] == 8
        for name, capped in (("cap", True), ("noncap", False)):
            assert group["by_cap_status"][name] == independent_group([row for row in subset if row["original_capped"] is capped])
    assert summary["by_condition"]["a171_standard"]["a171_standard"]["by_cap_status"]["noncap"] == independent_group([])
    overall = summary["overall"]
    assert sum(overall["overlaps"].values()) == 32
    for index, feature in enumerate(FEATURES):
        assert sum(count for key, count in overall["overlaps"].items() if key[index] == "1") == overall["feature_presence"][feature]


def test_incomplete_records_cannot_produce_complete_case_summaries(fixture):
    records = numeric_records(fixture)
    records[17]["features"] = None
    with pytest.raises(ValueError, match="no_complete_case_summary"):
        a.summarize(records)


def test_compile_plan_never_opens_any_parent_file(fixture, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.is_relative_to(fixture["parent"]):
            pytest.fail("compile_plan opened synthetic parent evidence")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    plan = a.compile_plan(fixture["bindings"], fixture["plan"]["bindings"]["protocol_sha256"],
                          fixture["plan"]["bindings"]["tests_sha256"])
    assert plan == fixture["plan"]


@pytest.mark.parametrize("change", ["detector", "gram_rule", "count", "resume", "model", "object_categories", "parent_manifest"])
def test_plan_mutations_rejected(fixture, change):
    plan = copy.deepcopy(fixture["plan"])
    if change == "detector":
        plan["detectors"]["new_detector"] = "unrequested"
    elif change == "gram_rule":
        plan["detectors"]["repetition"] = "changed threshold"
    elif change == "count":
        plan["planned_outputs"] = 31
    elif change == "resume":
        plan["resume_allowed"] = True
    elif change == "model":
        plan["model_calls_allowed"] = 1
    elif change == "object_categories":
        plan["object_categories"].reverse()
    else:
        del plan["parent_bindings"]["result_sha256"][next(iter(plan["parent_bindings"]["result_sha256"]))]
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_complete_audit_replay_and_parent_evidence_immutability(fixture):
    parent = fixture["parent"]
    before = {path.relative_to(parent).as_posix(): a.file_sha(path) for path in parent.rglob("*") if path.is_file()}
    aggregate = a.run_plan(**fixture["args"])
    assert aggregate["status"] == "complete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"] == {"attempted": 32, "processed": 32, "unattempted": 0, "missing": 0}
    assert aggregate["target_model_calls"] == 0 and aggregate["posthoc"] is True
    assert aggregate["summaries"] == a.summarize(numeric_records(fixture))
    replay = a.export_run(**fixture["args"])
    assert replay["aggregate"] == aggregate and replay["records"] == numeric_records(fixture)
    after = {path.relative_to(parent).as_posix(): a.file_sha(path) for path in parent.rglob("*") if path.is_file()}
    assert before == after
    serialized = json.dumps(replay)
    assert "response_text" not in serialized and "generated_token_ids" not in serialized
    assert "public_v1" not in serialized and "public fixture text" not in serialized
    assert (fixture["args"]["output_root"] / "records.json").stat().st_mode & 0o777 == 0o600


def test_all_32_parent_hashes_and_attempt_receipts_precede_interpretation(fixture, monkeypatch):
    paths = {fixture["parent"] / "execution/trials" / trial["trial_id"] / "result.json": trial for trial in fixture["trials"]}
    checked, inspected = set(), []
    original_read, original_parse = a._bound_bytes, a._parent_result

    def read(path, digest):
        if path in paths:
            trial = paths[path]
            attempt = fixture["args"]["output_root"] / "outputs" / trial["trial_id"] / "attempt.json"
            assert attempt.exists(), "raw bytes were read before their durable attempt"
        raw = original_read(path, digest)
        if path in paths:
            checked.add(paths[path]["trial_id"])
        return raw

    def parse(raw, trial, record, header):
        assert len(checked) == 32, "text/token interpretation began before every result hash passed"
        inspected.append(trial["trial_id"])
        return original_parse(raw, trial, record, header)

    monkeypatch.setattr(a, "_bound_bytes", read)
    monkeypatch.setattr(a, "_parent_result", parse)
    result = a.run_plan(**fixture["args"])
    assert result["status"] == "complete"
    assert inspected == [trial["trial_id"] for trial in fixture["trials"]]


def test_result_hash_failure_stops_before_any_interpretation(fixture, monkeypatch):
    trial = fixture["trials"][5]
    path = fixture["parent"] / "execution/trials" / trial["trial_id"] / "result.json"
    path.write_bytes(path.read_bytes() + b" ")

    def forbidden(*args):
        pytest.fail("parent response interpreted before all result hashes passed")

    monkeypatch.setattr(a, "_parent_result", forbidden)
    aggregate = a.run_plan(**fixture["args"])
    assert aggregate["execution_status"] == "failed" and aggregate["summaries"] is None
    assert aggregate["coverage"] == {"attempted": 6, "processed": 0, "unattempted": 26, "missing": 32}
    records = json.loads((fixture["args"]["output_root"] / "records.json").read_text())
    assert len(records) == 32 and all(row["features"] is None for row in records)
    with pytest.raises(ValueError, match="artifact_hash_drift"):
        a.export_run(**fixture["args"])


@pytest.mark.parametrize("relative", ["execution/analysis.json", "execution/records.json", "VERIFICATION.json"])
def test_parent_metadata_hash_drift_blocks_all_raw_output_reads(fixture, monkeypatch, relative):
    path = fixture["parent"] / relative
    path.write_bytes(path.read_bytes() + b" ")

    def forbidden(*args):
        pytest.fail("raw result hashing started despite parent metadata drift")

    monkeypatch.setattr(a, "_result_bytes", forbidden)
    aggregate = a.run_plan(**fixture["args"])
    assert aggregate["coverage"] == {"attempted": 0, "processed": 0, "unattempted": 32, "missing": 32}
    assert aggregate["execution_status"] == "failed" and aggregate["summaries"] is None
    assert aggregate["numeric_records_sha256"] is None


@pytest.mark.parametrize("phase", ["hash", "inspect"])
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_interruption_preserves_two_phase_coverage_and_replays_completed_only(fixture, monkeypatch, phase, number):
    original = a._bound_bytes if phase == "hash" else a.inspect_output
    calls = []
    result_paths = {fixture["parent"] / "execution/trials" / trial["trial_id"] / "result.json" for trial in fixture["trials"]}

    def interrupted(*args):
        relevant = phase == "inspect" or args[0] in result_paths
        if relevant:
            calls.append(True)
            if len(calls) == 4:
                if number is None:
                    raise KeyboardInterrupt
                if number == signal.SIGALRM:
                    a._deadline(number, None)
                a._interrupt(number, None)
        return original(*args)

    attribute = "_bound_bytes" if phase == "hash" else "inspect_output"
    monkeypatch.setattr(a, attribute, interrupted)
    aggregate = a.run_plan(**fixture["args"])
    expected_processed, expected_attempted = (0, 4) if phase == "hash" else (3, 32)
    assert len(calls) == 4 and aggregate["summaries"] is None
    assert aggregate["coverage"] == {"attempted": expected_attempted, "processed": expected_processed,
                                     "unattempted": 32 - expected_attempted, "missing": 32 - expected_processed}
    root = fixture["args"]["output_root"]
    receipt = json.loads((root / "execution-finished.json").read_text())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {
        "signal_number": int(number) if number is not None else None,
        "signal_name": signal.Signals(number).name if number is not None else None,
        "pid": os.getpid(), "exception_type": "KeyboardInterrupt" if number is None else
        "A175Deadline" if number == signal.SIGALRM else "A175Interrupted"}
    monkeypatch.setattr(a, attribute, original)
    inspector = a.inspect_output
    replayed = []

    def count_replay(*args):
        replayed.append(True)
        return inspector(*args)

    monkeypatch.setattr(a, "inspect_output", count_replay)
    replay = a.export_run(**fixture["args"])
    assert len(replayed) == expected_processed and replay["aggregate"] == aggregate
    assert sum(row["features"] is None for row in replay["records"]) == 32 - expected_processed


def test_signal_after_durable_result_publication_keeps_that_result(fixture, monkeypatch):
    original = a._write_new
    results = []

    def write(path, value):
        original(path, value)
        if path.name == "result.json":
            results.append(True)
            if len(results) == 2:
                a._interrupt(signal.SIGTERM, None)

    monkeypatch.setattr(a, "_write_new", write)
    aggregate = a.run_plan(**fixture["args"])
    assert aggregate["coverage"] == {"attempted": 32, "processed": 2, "unattempted": 0, "missing": 30}
    assert aggregate["summaries"] is None
    assert a.export_run(**fixture["args"])["aggregate"] == aggregate


def test_detector_error_is_terminal_with_no_retry_or_feature_imputation(fixture, monkeypatch):
    original = a.inspect_output
    calls = []

    def fail(*args):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("synthetic private response must never escape")
        return original(*args)

    monkeypatch.setattr(a, "inspect_output", fail)
    aggregate = a.run_plan(**fixture["args"])
    assert len(calls) == 3 and aggregate["execution_status"] == "failed" and aggregate["summaries"] is None
    assert aggregate["coverage"]["processed"] == 2
    error = (fixture["args"]["output_root"] / "error.json").read_text()
    assert "private response" not in error


def test_one_shot_cannot_repeat_or_change_output_directory(fixture, monkeypatch):
    a.run_plan(**fixture["args"])
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**fixture["args"])
    changed = {**fixture["args"], "output_root": fixture["root"] / "different-execution"}
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**changed)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="private_audit_output"):
        a.run_plan(**changed)
    with pytest.raises(ValueError, match="consumed_audit"):
        a.run_plan(**fixture["args"])


def test_parent_exclusive_lock_failure_gets_terminal_receipt_without_raw_reads(fixture):
    import fcntl
    with (fixture["parent"] / "execution/.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        aggregate = a.run_plan(**fixture["args"])
    assert aggregate["execution_status"] == "failed" and aggregate["summaries"] is None
    assert aggregate["coverage"] == {"attempted": 0, "processed": 0, "unattempted": 32, "missing": 32}
    assert (fixture["args"]["output_root"] / "execution-finished.json").exists()
    replay = a.export_run(**fixture["args"])
    assert replay["records"] is None and replay["aggregate"] == aggregate


def test_audit_exclusive_lock_prevents_consumption(fixture):
    import fcntl
    root = fixture["args"]["output_root"]
    root.mkdir()
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture["args"])
    assert not (root / "one-shot.json").exists()


def test_bounded_signals_use_ten_minutes_and_restore_handlers():
    old = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with a.bounded_signals():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 599 < remaining <= 600 and interval == 0
        assert signal.getsignal(signal.SIGALRM) is a._deadline
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {number: signal.getsignal(number) for number in old} == old


@pytest.mark.parametrize("tamper", ["feature", "boolean_count", "overlap", "attempt", "records", "aggregate", "signal", "orphan", "gap"])
def test_export_rejects_feature_receipt_and_summary_tampering(fixture, tamper):
    a.run_plan(**fixture["args"])
    root = fixture["args"]["output_root"]
    folder = root / "outputs" / fixture["trials"][1]["trial_id"]
    if tamper == "orphan":
        (root / "outputs" / ("f" * 24)).mkdir()
    elif tamper == "gap":
        (folder / "result.json").unlink()
    else:
        path = {"feature": folder / "result.json", "boolean_count": folder / "result.json",
                "overlap": folder / "result.json", "attempt": folder / "attempt.json",
                "records": root / "records.json", "aggregate": root / "analysis.json",
                "signal": root / "execution-finished.json"}[tamper]
        value = json.loads(path.read_text())
        if tamper == "feature":
            value["record"]["features"]["fence"] = not value["record"]["features"]["fence"]
        elif tamper == "boolean_count":
            count = value["record"]["features"]["answer_object_count"]
            assert count in (0, 1)
            value["record"]["features"]["answer_object_count"] = bool(count)
        elif tamper == "overlap":
            value["record"]["features"]["overlap"] = "111"
        elif tamper == "attempt":
            value["sequence_index"] += 1
        elif tamper == "records":
            value[0]["features"]["repetition"] = not value[0]["features"]["repetition"]
        elif tamper == "aggregate":
            value["summaries"]["overall"]["outputs"] -= 1
        else:
            value["interruption"] = {"signal_number": 15, "signal_name": "SIGTERM", "pid": os.getpid(),
                                     "exception_type": "A175Interrupted"}
        write_json(path, value)
    with pytest.raises(ValueError):
        a.export_run(**fixture["args"])


def test_cli_returns_only_status_coverage_and_no_private_exception_text(fixture, monkeypatch, capsys):
    def fake_run(**kwargs):
        raise RuntimeError("synthetic private response and token array must not escape")

    monkeypatch.setattr(a, "run_plan", fake_run)
    argv = []
    for flag, key in (("plan", "plan_path"), ("plan-sha256", "plan_sha256"), ("protocol", "protocol_path"),
                      ("freeze", "freeze_path"), ("freeze-sha256", "freeze_sha256"), ("output-root", "output_root")):
        argv.extend(["--" + flag, str(fixture["args"][key])])
    previous = os.umask(0o077)
    try:
        assert a.main(argv) == 1
    finally:
        os.umask(previous)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "a175_failed", "error_type": "RuntimeError", "target_model_calls": 0}
    assert "synthetic private" not in captured.out + captured.err


@pytest.mark.parametrize("change", ["source", "protocol", "tests", "extra_input", "review", "unpushed"])
def test_audit_freeze_and_review_gates_precede_parent_response_reads(fixture, monkeypatch, change):
    root, args = fixture["root"], fixture["args"]
    if change in ("source", "protocol", "tests"):
        relative = {"source": "src/lexical_prompt_study/crossover_output_audit.py",
                    "protocol": "plans/crossover_output_audit_a175.md", "tests": "tests/test_crossover_output_audit.py"}[change]
        path = root / "frozen" / relative
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "extra_input":
        write_json(root / "inputs/unplanned.json", {"unexpected": True})
    else:
        freeze = json.loads(args["freeze_path"].read_text())
        if change == "review":
            write_json(root / "REVIEW.json", {"status": "fail"})
            freeze["review_sha256"] = a.file_sha(root / "REVIEW.json")
        else:
            freeze["pushed_commit"] = "b" * 40
        write_json(args["freeze_path"], freeze)
        args["freeze_sha256"] = a.file_sha(args["freeze_path"])

    def forbidden(*args):
        pytest.fail("parent response bytes were read despite failed audit freeze/review")

    monkeypatch.setattr(a, "_result_bytes", forbidden)
    with pytest.raises(ValueError):
        a.run_plan(**args)
    assert not (args["output_root"] / "one-shot.json").exists()


def test_malformed_detector_schema_is_terminal_before_numeric_publication(fixture, monkeypatch):
    original = a.inspect_output
    calls = []

    def invalid(*args):
        calls.append(True)
        result = original(*args)
        result["answer_object_count"] = bool(result["answer_object_count"])
        return result

    monkeypatch.setattr(a, "inspect_output", invalid)
    aggregate = a.run_plan(**fixture["args"])
    assert len(calls) == 1 and aggregate["execution_status"] == "failed"
    assert aggregate["coverage"] == {"attempted": 32, "processed": 0, "unattempted": 0, "missing": 32}
    assert aggregate["summaries"] is None
    assert not any((fixture["args"]["output_root"] / "outputs").glob("*/result.json"))


def test_detector_is_exact_pinned_a172_alias_without_new_feature_fields():
    from lexical_prompt_study import output_structure_audit as original
    assert a.inspect_output is original.inspect_output
    assert a.file_sha(Path(original.__file__)) == '4a5628bfd6bc835fb476e717d01197b4294c855ad7dfaf6798814f7245740c02'
    assert a.OBJECT_CATEGORIES == original.OBJECT_CATEGORIES
    assert a.FEATURES == original.FEATURES
    assert a.OVERLAPS == original.OVERLAPS


def test_parent_fence_indicator_must_agree_with_unchanged_detector(fixture, monkeypatch):
    original = a.inspect_output
    calls = []

    def inconsistent(*args):
        calls.append(True)
        value = original(*args)
        value['fence'] = not value['fence']
        value['overlap'] = ''.join('1' if value[key] else '0' for key in FEATURES)
        return value

    monkeypatch.setattr(a, 'inspect_output', inconsistent)
    aggregate = a.run_plan(**fixture['args'])
    assert len(calls) == 1 and aggregate['execution_status'] == 'failed'
    assert aggregate['coverage'] == {'attempted': 32, 'processed': 0, 'unattempted': 0, 'missing': 32}
    assert aggregate['summaries'] is None


def test_frozen_detector_helper_drift_prevents_parent_result_reads(fixture, monkeypatch):
    helper = fixture['root'] / 'frozen/src/lexical_prompt_study/output_structure_audit.py'
    helper.write_bytes(helper.read_bytes() + b'\n')

    def forbidden(*args):
        pytest.fail('parent output read despite frozen detector helper drift')

    monkeypatch.setattr(a, '_result_bytes', forbidden)
    with pytest.raises(ValueError):
        a.run_plan(**fixture['args'])
    assert not (fixture['args']['output_root'] / 'one-shot.json').exists()
