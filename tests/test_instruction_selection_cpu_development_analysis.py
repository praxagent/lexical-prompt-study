from copy import deepcopy
import hashlib
import json

import pytest

from lexical_prompt_study import instruction_selection_cpu_development as runtime
from lexical_prompt_study import instruction_selection_cpu_development_analysis as analysis
from lexical_prompt_study import instruction_selection_tasks as tasks


def fixture():
    materials = {name: "Invented harmless " + name for name in tasks.SCAFFOLDS}
    base = tasks.compile_plan(tasks.make_cohort_manifest(), partition="development", scaffolds=materials,
        bindings={"protocol_sha256": "a" * 64, "materials_sha256": "b" * 64})
    plan = runtime.compile_plan(base, scaffolds=materials, protocol_sha256="c" * 64)
    records = []
    for trial in plan["trials"]:
        row = runtime.numeric_record(trial)
        row.update(generation_status="completed", finish_reason="eos",
                   score={"category": "exact", "strict_correct": True})
        records.append(row)
    return plan, records


def result(plan, records):
    return analysis.analyze_plan_records(plan, records)


def primary(value):
    return value["primary_full_minus_sham_strict_correctness"]


def wrong(row, category="other_selector", *, cap=False):
    row["finish_reason"] = "length" if cap else "eos"
    row["score"] = {"category": category, "strict_correct": category == "exact" and not cap}


def test_complete_null_is_descriptive_only_and_has_no_inferential_outputs():
    plan, records = fixture()
    value = result(plan, records)
    assert value["world_count"] == 16 and value["coverage"]["expected"] == 288
    assert value["status"] == "complete" and value["baseline_controls"]["gate_passed"]
    assert primary(value)["estimate_pp"] == 0 and primary(value)["worst_best_bounds_pp"] == [0, 0]
    assert value["claim_boundaries"]["implementation_selected_after_competence_diagnostic"]
    assert not value["claim_boundaries"]["confirmatory_result"]
    assert not value["claim_boundaries"]["heldout_allowed"]
    raw = json.dumps(value)
    assert "bootstrap" not in raw and "equivalence" not in raw and "interval_95" not in raw


def test_baseline_only_gate_can_pass_without_fabricating_unrun_scaffold_outcomes():
    plan, records = fixture()
    value = result(plan, records[:32])
    assert value["status"] == "incomplete" and value["baseline_controls"]["gate_passed"]
    assert value["coverage"]["missing"] == 256
    assert primary(value)["estimate_pp"] is None
    assert primary(value)["worst_best_bounds_pp"] == [-100, 100]


def test_baseline_gate_agrees_with_runtime_and_requires_world_pairs():
    plan, records = fixture()
    wrong(records[0])
    assert result(plan, records)["baseline_controls"]["gate_passed"]
    wrong(records[2])  # Different world, cell accuracy30/32 but both-correct14/16.
    controls = result(plan, records)["baseline_controls"]
    assert controls["no_scaffold_strict_accuracy"]["estimate_pp"] == 93.75
    assert controls["selector_pair_both_correct"]["estimate_pp"] == 87.5
    assert controls["gate_passed"] == runtime.baseline_gate(plan, records)["gate_passed"] is False


def test_two_errors_in_one_world_pass_both_thresholds_but_three_errors_do_not():
    plan, records = fixture()
    wrong(records[0])
    wrong(records[1])
    assert result(plan, records)["baseline_controls"]["gate_passed"]
    wrong(records[2])
    assert not result(plan, records)["baseline_controls"]["gate_passed"]


def test_missing_partner_blocks_gate_even_when_pair_outcome_identified():
    plan, records = fixture()
    wrong(records[0])
    records.pop(1)
    controls = result(plan, records)["baseline_controls"]
    assert controls["selector_pair_both_correct"]["binary_outcome_fully_identified"]
    assert not controls["all_baseline_cells_completed"] and not controls["gate_passed"]


def test_world_mean_averages_four_placement_selector_pairs():
    plan, records = fixture()
    for row in records:
        if (row["scaffold"], row["placement"], row["selector"]) == ("full", "before", "A"):
            wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == -25
    assert primary(value)["worlds"] == primary(value)["identified_worlds"] == 16
    assert value["secondary_placement_contrasts"]["before"]["estimate_pp"] == -50
    assert value["secondary_placement_contrasts"]["after"]["estimate_pp"] == 0


def test_opposite_placement_effects_remain_visible_under_pooled_zero():
    plan, records = fixture()
    for row in records:
        if (row["scaffold"], row["placement"]) in (("full", "before"), ("sham", "after")):
            wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0
    assert value["secondary_placement_contrasts"]["before"]["estimate_pp"] == -100
    assert value["secondary_placement_contrasts"]["after"]["estimate_pp"] == 100


def test_presentation_contrasts_keep_eight_world_denominators():
    plan, records = fixture()
    for row in records:
        if row["scaffold"] == "full" and row["query_order"] == "AB":
            wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == -50
    assert value["secondary_query_order_contrasts"]["AB"]["estimate_pp"] == -100
    assert value["secondary_query_order_contrasts"]["BA"]["estimate_pp"] == 0


def test_one_missing_full_cell_gives_sharp_bounds_over_all_scheduled_cells():
    plan, records = fixture()
    records.pop(next(i for i, row in enumerate(records) if row["scaffold"] == "full"))
    value = result(plan, records)
    assert primary(value)["estimate_pp"] is None
    assert primary(value)["worst_best_bounds_pp"] == [-100 / 64, 0]
    assert primary(value)["identified_worlds"] == 15
    assert value["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is None


def test_infrastructure_failure_is_unknown_not_a_task_failure():
    plan, records = fixture()
    row = next(row for row in records if row["scaffold"] == "sham")
    row.update(generation_status="infrastructure_failed", finish_reason=None,
               score={"category": None, "strict_correct": None})
    value = result(plan, records)
    assert value["coverage"]["infrastructure_failed"] == 1 and value["coverage"]["missing"] == 0
    assert primary(value)["worst_best_bounds_pp"] == [0, 100 / 64]


def test_missing_auxiliary_arm_does_not_erase_fully_observed_primary():
    plan, records = fixture()
    records = [row for row in records if row["scaffold"] != "replacement"]
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0 and value["status"] == "incomplete"
    assert value["secondary_control_contrasts"]["full_minus_replacement"]["worst_best_bounds_pp"] == [0, 100]


def test_empty_records_keep_all288_denominators_and_block_baseline_gate():
    plan, _ = fixture()
    value = result(plan, [])
    assert value["coverage"]["missing"] == 288
    assert value["coverage"]["strict_accuracy_bounds_pp"] == [0, 100]
    assert primary(value)["worst_best_bounds_pp"] == [-100, 100]
    assert not value["baseline_controls"]["gate_passed"]


def test_floor_guard_uses_fixed10percent_without_erasing_estimate():
    plan, records = fixture()
    for scaffold in ("full", "sham"):
        group = [row for row in records if row["scaffold"] == scaffold]
        for row in group[6:]:
            wrong(row, "format")
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0
    assert value["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is True
    group = [row for row in records if row["scaffold"] == "full"]
    wrong(group[6], "exact")  # Seven of64 is above10%; six of64 is below.
    assert result(plan, records)["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is False


def test_failed_baseline_guard_does_not_erase_scaffold_estimate():
    plan, records = fixture()
    for row in records[:4]:
        wrong(row)
    value = result(plan, records)
    assert not value["interpretation_guards"]["baseline_competence_passed"]
    assert primary(value)["estimate_pp"] == 0


def test_cap_is_failure_even_if_parser_exact_and_decomposition_is_exhaustive():
    plan, records = fixture()
    full = [row for row in records if row["scaffold"] == "full"]
    for row, category in zip(full[:4], ("format", "other_selector", "other_answer", "exact"), strict=True):
        wrong(row, category, cap=category == "exact")
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == -4 * 100 / 64
    arm = value["scaffold_coverage"]["full"]
    assert arm["strict_correct"] == 60 and arm["capped"] == 1
    assert arm["parser_categories"]["exact"] == 61 and arm["uncapped_categories"]["exact"] == 60
    pieces = value["strict_outcome_decomposition_contrasts"]
    assert all(pieces[key]["estimate_pp"] == 100 / 64 for key in ("format", "other_selector", "other_answer", "capped"))
    assert sum(part["estimate_pp"] for part in pieces.values()) == 0


def test_secondary_controls_use_same_world_weights_and_repeated_baseline_not_independent_cells():
    plan, records = fixture()
    for row in records:
        if row["scaffold"] == "inert" and row["selector"] == "A":
            wrong(row)
    values = result(plan, records)["secondary_control_contrasts"]
    assert values["full_minus_inert"]["estimate_pp"] == 50
    assert values["inert_minus_none"]["estimate_pp"] == -50
    assert values["sham_minus_none"]["estimate_pp"] == 0


def test_record_order_does_not_change_statistics_or_hashes():
    plan, records = fixture()
    left = result(plan, records)
    records.reverse()
    assert result(plan, records) == left


@pytest.mark.parametrize("mutation", ("old_schema", "raw", "metadata", "bool", "cap", "duplicate", "unknown", "likelihood"))
def test_invalid_numeric_records_fail_closed(mutation):
    plan, records = fixture()
    if mutation == "old_schema":
        records[0]["schema_version"] = "a164-cell-v1"
    elif mutation == "raw":
        records[0]["response_text"] = "Invented harmless output"
    elif mutation == "metadata":
        records[0]["query_order"] = "BA" if records[0]["query_order"] == "AB" else "AB"
    elif mutation == "bool":
        records[0]["score"]["strict_correct"] = 1
    elif mutation == "cap":
        records[0]["finish_reason"] = "length"
    elif mutation == "duplicate":
        records.append(deepcopy(records[0]))
    elif mutation == "unknown":
        records[0]["trial_id"] = "f" * 24
    else:
        records[0]["likelihood_collected"] = True
    with pytest.raises(analysis.AnalysisError):
        result(plan, records)


@pytest.mark.parametrize("mutation", ("missing_cell", "duplicate_cell", "new_world", "query_order", "baseline_order",
                                     "base_hash", "world_count", "reused_id", "heldout"))
def test_plan_universe_and_original_development_identity_are_enforced(mutation):
    plan, records = fixture()
    if mutation == "missing_cell":
        plan["trials"].pop()
    elif mutation == "duplicate_cell":
        plan["trials"][-1] = deepcopy(plan["trials"][-2])
    elif mutation == "new_world":
        plan["trials"][-1]["world_id"] = "f" * 64
    elif mutation == "query_order":
        plan["trials"][0]["query_order"] = "BA" if plan["trials"][0]["query_order"] == "AB" else "AB"
    elif mutation == "baseline_order":
        plan["trials"][0], plan["trials"][32] = plan["trials"][32], plan["trials"][0]
    elif mutation == "base_hash":
        plan["base_plan_sha256"] = "f" * 64
    elif mutation == "world_count":
        plan["world_count"] = 64
    elif mutation == "reused_id":
        plan["trials"][0]["trial_id"] = plan["base_plan"]["trials"][0]["trial_id"]
    else:
        plan["partition"] = "heldout"
    with pytest.raises(analysis.AnalysisError):
        result(plan, records)


def test_summary_does_not_export_source_prompt_answer_or_world_identity():
    plan, records = fixture()
    raw = json.dumps(result(plan, records))
    assert "Invented harmless" not in raw and plan["trials"][0]["world_id"] not in raw
    assert '"messages"' not in raw and '"selected_answer"' not in raw


def test_cli_hash_binding_no_overwrite_and_safe_stdout(tmp_path, capsys):
    plan, records = fixture()
    paths = []
    for name, value in (("plan", plan), ("records", records)):
        path = tmp_path / (name + ".json")
        path.write_bytes(analysis.canonical(value))
        paths.extend(["--" + name, str(path), "--" + name + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()])
    output = tmp_path / "summary.json"
    args = [*paths, "--output", str(output)]
    assert analysis.main(args) == 0
    stdout = capsys.readouterr().out
    assert "Invented" not in stdout and plan["trials"][0]["world_id"] not in stdout
    assert json.loads(output.read_bytes())["world_count"] == 16
    frozen = output.read_bytes()
    assert analysis.main(args) == 2 and output.read_bytes() == frozen
    capsys.readouterr()
    (tmp_path / "records.json").write_text("[]")
    assert analysis.main([*paths, "--output", str(tmp_path / "bad.json")]) == 2
    assert not (tmp_path / "bad.json").exists()
