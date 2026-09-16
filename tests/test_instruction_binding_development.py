import copy
import json

import pytest

from lexical_prompt_study import instruction_binding_development as dev


@pytest.fixture
def inputs():
    pins = dev.source_hashes()
    bindings = {"compiler_sha256": pins["instruction_binding_plan.py"],
                "generator_sha256": pins["instruction_binding_tasks.py"],
                "protocol_sha256": "a" * 64, "materials_sha256": "b" * 64}
    scaffolds = {key: "Public synthetic fixture " + key for key in dev.tasks.SCAFFOLD_KINDS}
    plans = [dev.compiler.compile_plan(stage=stage, scaffolds=scaffolds, bindings=bindings)
             for stage in dev.PHASES.values()]
    raw = [dev.canonical(plan) for plan in plans]
    return {"raw_plans": raw, "expected_sha256": [dev.sha(value) for value in raw],
            "phase": "initial", "protocol_sha256": "c" * 64, "other_known_world_ids": []}


def completed(plan):
    records = []
    for trial in plan["trials"]:
        record = dev.runtime.numeric_record(plan, trial)
        record.update(generation_status="completed", finish_reason="eos",
                      score={"category": "exact", "strict_correct": True},
                      teacher_forced={"status": "ok", "selected_logprob": -1.,
                                      "alternative_logprob": -4., "selected_token_count": 3,
                                      "alternative_token_count": 3})
        records.append(record)
    return records


def mutate_source(inputs, index, mutate):
    changed = copy.deepcopy(inputs)
    plan = json.loads(changed["raw_plans"][index])
    mutate(plan)
    changed["raw_plans"][index] = dev.canonical(plan)
    changed["expected_sha256"][index] = dev.sha(changed["raw_plans"][index])
    return changed


def test_baseline_plans_preserve_live_world_user_order_and_pair_identity(inputs):
    plan = dev.build_baseline_plan(**inputs)
    original = {trial["trial_id"]: trial for trial in json.loads(inputs["raw_plans"][0])["trials"]}
    assert len(plan["trials"]) == 32
    assert {trial["scaffold_kind"] for trial in plan["trials"]} == {"none"}
    assert {trial["renderer"] for trial in plan["trials"]} == {"json", "line_table"}
    for trial in plan["trials"]:
        source = original[trial["source_trial_id"]]
        assert trial["trial_id"] != source["trial_id"]
        assert trial["messages"][1] == source["messages"][1]
        assert trial["messages"][0]["content"].startswith(source["messages"][0]["content"] + "\n\n")
        for field in ("world_id", "world", "query_order", "table_order", "selected_answer",
                      "unselected_answer", "selector", "diagnostic"):
            assert trial[field] == source[field]
    for a, b in zip(plan["trials"][::2], plan["trials"][1::2], strict=True):
        assert a["messages"][1] == b["messages"][1]
        assert a["messages"][0]["content"].replace("selector is A", "selector is B", 1) == b["messages"][0]["content"]


def test_fixed_examples_have_independently_computed_chains_and_different_answers():
    expected = (("s2", "s5"), ("s5", "s1"))
    for example, answers in zip(dev.EXAMPLES, expected, strict=True):
        assert tuple(dev.tasks.oracle(example, selector) for selector in ("A", "B")) == answers
        for renderer in dev.tasks.RENDERERS:
            text = dev.worked_examples(renderer)
            for answer in answers:
                assert json.dumps({"answer": answer}, separators=(",", ":")) in text


def test_source_hashes_and_message_changes_are_bound_to_new_identity(inputs):
    original = dev.build_baseline_plan(**inputs)
    revised = dev.build_baseline_plan(**{**inputs, "protocol_sha256": "d" * 64})
    assert {r["trial_id"] for r in original["trials"]}.isdisjoint(r["trial_id"] for r in revised["trials"])
    assert original["development_amendment"]["source_hashes"] == dev.source_hashes()
    with pytest.raises(ValueError, match="source_plan_hash"):
        dev.build_baseline_plan(**{**inputs, "expected_sha256": ["f" * 64, inputs["expected_sha256"][1]]})


def test_external_example_overlap_is_rejected_and_dev_examples_are_disjoint(inputs):
    plan = dev.build_baseline_plan(**inputs)
    assert not {trial["world_id"] for raw in inputs["raw_plans"]
                for trial in json.loads(raw)["trials"]}.intersection(w.world_id for w in dev.EXAMPLES)
    assert plan["development_amendment"]["root_full_cohort_disjointness_check_required"]
    with pytest.raises(ValueError, match="example_world_overlap"):
        dev.build_baseline_plan(**{**inputs, "other_known_world_ids": [dev.EXAMPLES[0].world_id]})


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.update(partition="heldout"), "development_only"),
    (lambda p: p.update(stage="development"), "duplicate_source_stage"),
    (lambda p: p["trials"].pop(), "complete_source"),
    (lambda p: p["bindings"].update(compiler_sha256="f" * 64), "compiler_binding"),
    (lambda p: p["trials"][0]["messages"][0].update(content="changed instruction"), "unmodified_system"),
])
def test_rejects_heldout_wrong_stage_missing_baselines_or_modified_instrument(inputs, mutation, code):
    with pytest.raises(ValueError, match=code):
        dev.build_baseline_plan(**mutate_source(inputs, 0, mutation))


def test_complete_gate_uses_planned_cells_pairs_and_depth_denominators(inputs):
    plan = dev.build_baseline_plan(**inputs)
    summary = dev.summarize_baseline_records(plan, completed(plan))
    assert summary["gate_passed"] and summary["planned_cells"] == 32
    for renderer in summary["renderers"].values():
        assert renderer["planned_cells"] == renderer["strict_correct"] == 16
        assert renderer["planned_pairs"] == renderer["both_selectors_correct"] == 8
        assert renderer["likelihood_ok"] == 16
        assert all(row["planned_cells"] == 8 and row["planned_pairs"] == 4
                   for row in renderer["by_depth"].values())


def test_one_capped_correct_answer_fails_pair_gate_and_remains_exact_category(inputs):
    plan = dev.build_baseline_plan(**inputs)
    records = completed(plan)
    records[0]["finish_reason"] = "length"
    records[0]["score"]["strict_correct"] = False
    summary = dev.summarize_baseline_records(plan, records)
    row = summary["renderers"][records[0]["renderer"]]
    assert row["strict_correct"] == 15 and row["both_selectors_correct"] == 7
    assert row["categories"]["exact"] == 16 and row["length_terminated"] == 1
    assert not summary["gate_passed"]


def test_missing_generation_and_failed_likelihood_have_separate_denominators(inputs):
    plan = dev.build_baseline_plan(**inputs)
    records = completed(plan)
    records[0]["teacher_forced"] = dev.runtime.numeric_record(plan, plan["trials"][0])["teacher_forced"]
    assert dev.summarize_baseline_records(plan, records)["gate_passed"]
    summary = dev.summarize_baseline_records(plan, records[1:])
    assert not summary["gate_passed"]
    assert summary["renderers"][records[0]["renderer"]]["missing_cells"] == 1
    records[0]["teacher_forced"]["status"] = "ok"
    records[0]["teacher_forced"]["selected_logprob"] = float("nan")
    with pytest.raises(ValueError):
        dev.summarize_baseline_records(plan, records)


@pytest.mark.parametrize("kind", ["duplicate", "unrelated", "wrong_metadata", "extra_field"])
def test_unplanned_or_ambiguous_records_are_rejected(inputs, kind):
    plan = dev.build_baseline_plan(**inputs)
    records = completed(plan)
    if kind == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif kind == "unrelated":
        records[0]["trial_id"] = "f" * 24
    elif kind == "wrong_metadata":
        records[0]["scaffold"] = "full"
    else:
        records[0]["response_text"] = "synthetic"
    with pytest.raises(ValueError):
        dev.summarize_baseline_records(plan, records)


def test_remaining_half_requires_passed_initial_records_and_uses_disjoint_worlds(inputs):
    first = dev.build_baseline_plan(**inputs)
    with pytest.raises(ValueError, match="initial_gate_required"):
        dev.build_baseline_plan(**{**inputs, "phase": "remaining"})
    with pytest.raises(ValueError, match="initial_competence_gate_failed"):
        dev.build_baseline_plan(**{**inputs, "phase": "remaining", "initial_records": []})
    second = dev.build_baseline_plan(**{**inputs, "phase": "remaining", "initial_records": completed(first)})
    assert len(second["trials"]) == 32
    assert {r["world_id"] for r in first["trials"]}.isdisjoint(r["world_id"] for r in second["trials"])
    assert second["development_amendment"]["initial_gate"]["records_sha256"] == dev.sha(dev.canonical(completed(first)))


def test_cross_half_duplicates_are_rejected(inputs):
    changed = copy.deepcopy(inputs)
    duplicate = json.loads(changed["raw_plans"][0])
    duplicate["stage"] = "development"
    changed["raw_plans"][1] = dev.canonical(duplicate)
    changed["expected_sha256"][1] = dev.sha(changed["raw_plans"][1])
    with pytest.raises(ValueError, match="duplicate_source_trial|cross_half_world_overlap"):
        dev.build_baseline_plan(**changed)


@pytest.mark.parametrize("field", ["system", "user", "id", "source"])
def test_summary_refuses_altered_plan_or_source_binding(inputs, field):
    plan = dev.build_baseline_plan(**inputs)
    records = completed(plan)
    if field == "system":
        plan["trials"][0]["messages"][0]["content"] += " extra instruction"
    elif field == "user":
        plan["trials"][0]["messages"][1]["content"] += " extra payload"
    elif field == "id":
        plan["trials"][0]["trial_id"] = "f" * 24
    else:
        plan["development_amendment"]["source_hashes"]["instruction_binding_development.py"] = "f" * 64
    with pytest.raises(ValueError):
        dev.summarize_baseline_records(plan, records)
