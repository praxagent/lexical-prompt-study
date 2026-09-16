from copy import deepcopy
import hashlib
import json

import pytest

from lexical_prompt_study import instruction_selection_analysis as analysis
from lexical_prompt_study import instruction_selection_tasks as tasks


def fixture(partition="heldout"):
    plan = tasks.compile_plan(tasks.make_cohort_manifest(), partition=partition,
        scaffolds={name: "Invented harmless " + name for name in tasks.SCAFFOLDS},
        bindings={"protocol_sha256": "a" * 64, "materials_sha256": "b" * 64})
    records = []
    for trial in plan["trials"]:
        row = {key: trial[key] for key in analysis.META_FIELDS if key != "scaffold"}
        row.update(schema_version="a164-cell-v1", scaffold=trial["scaffold_kind"],
                   generation_status="completed", finish_reason="eos",
                   score={"category": "exact", "strict_correct": True}, likelihood_collected=False)
        records.append(row)
    return plan, records


def result(plan, records):
    return analysis.analyze_plan_records(plan, records, bootstrap_replicates=200)


def wrong(row, category="other_selector", *, cap=False):
    row["finish_reason"] = "length" if cap else "eos"
    row["score"] = {"category": category, "strict_correct": category == "exact" and not cap}


def primary(value):
    return value["primary_full_minus_sham_strict_correctness"]


def test_development_has_only_baseline_gate_and_no_invented_scaffold_estimate():
    plan, records = fixture("development")
    value = result(plan, records)
    assert value["world_count"] == 16 and value["coverage"]["expected"] == 32
    assert value["baseline_controls"]["gate_passed"]
    assert primary(value) is None and value["likelihood_collected"] is False


def test_baseline_gate_requires_both_selectors_not_only_high_cell_accuracy():
    plan, records = fixture("development")
    wrong(records[0])
    assert result(plan, records)["baseline_controls"]["gate_passed"]
    wrong(records[2])  # Different world: 30/32 correct, but only14/16 paired successes.
    value = result(plan, records)["baseline_controls"]
    assert value["no_scaffold_strict_accuracy"]["estimate_pp"] == 93.75
    assert value["selector_pair_both_correct"]["estimate_pp"] == 87.5
    assert not value["gate_passed"]


def test_missing_partner_never_passes_gate_even_if_pair_failure_is_identified():
    plan, records = fixture("development")
    wrong(records[0])
    records.pop(1)
    value = result(plan, records)["baseline_controls"]
    assert value["selector_pair_both_correct"]["fully_observed"]
    assert not value["all_baseline_cells_completed"] and not value["gate_passed"]


def test_heldout_fixed64_complete_null_reports_qualified_approximate_exclusion_only():
    plan, records = fixture()
    value = result(plan, records)
    assert value["world_count"] == 64 and value["coverage"]["expected"] == 1152
    assert primary(value)["estimate_pp"] == 0 and primary(value)["approximate_interval_95_pp"] == [0, 0]
    assert primary(value)["approximate_interval_excludes_loss_of_10pp_or_more"] is True
    assert value["claim_boundaries"]["task_pair_population_size"] == 240
    assert not value["claim_boundaries"]["exact_design_based_coverage"]
    assert not value["claim_boundaries"]["equivalence_claim"]


def test_large_full_arm_damage_is_retained():
    plan, records = fixture()
    for row in records:
        if row["scaffold"] == "full":
            wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == -100
    assert primary(value)["approximate_interval_excludes_loss_of_10pp_or_more"] is False
    assert value["interpretation_guards"]["practical_loss_exclusion_interpretable"]


def test_both_arm_zero_floor_blocks_null_claim_without_erasing_estimate():
    plan, records = fixture()
    for row in records:
        if row["scaffold"] in ("full", "sham"):
            wrong(row, "format")
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0
    assert value["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is True
    assert primary(value)["approximate_interval_excludes_loss_of_10pp_or_more"] is None


def test_floor_guard_is_predeclared_10percent_not_just_exact_zero():
    plan, records = fixture()
    for scaffold in ("full", "sham"):
        group = [row for row in records if row["scaffold"] == scaffold]
        for row in group[25:]:
            wrong(row)
    assert result(plan, records)["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is True
    group = [row for row in records if row["scaffold"] == "full"]
    wrong(group[25], "exact")  #26/256 crosses10%; no outcome-driven threshold change.
    assert result(plan, records)["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is False


def test_failed_heldout_baseline_blocks_exclusion():
    plan, records = fixture()
    baseline_a = [row for row in records if row["scaffold"] == "none" and row["selector"] == "A"]
    for row in baseline_a[:7]:
        wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0
    assert not value["baseline_controls"]["gate_passed"]
    assert primary(value)["approximate_interval_excludes_loss_of_10pp_or_more"] is None


def test_missing_one_full_cell_gives_exact_scheduled_cohort_bounds():
    plan, records = fixture()
    records.pop(next(i for i, row in enumerate(records) if row["scaffold"] == "full"))
    value = result(plan, records)
    assert primary(value)["estimate_pp"] is None
    assert primary(value)["worst_best_bounds_pp"] == [-100 / (64 * 4), 0]
    assert primary(value)["complete_worlds"] == 63
    assert value["interpretation_guards"]["both_full_and_sham_at_or_below_floor"] is None


def test_missing_all_primary_cells_remains_minus100_to100():
    plan, records = fixture()
    records = [row for row in records if row["scaffold"] not in ("full", "sham")]
    value = result(plan, records)
    assert primary(value)["worst_best_bounds_pp"] == [-100, 100]
    assert primary(value)["estimate_pp"] is None and primary(value)["complete_worlds"] == 0


def test_infrastructure_failure_has_same_bounds_as_missing_not_failure_label():
    plan, records = fixture()
    row = next(row for row in records if row["scaffold"] == "sham")
    row.update(generation_status="infrastructure_failed", finish_reason=None,
               score={"category": None, "strict_correct": None})
    value = result(plan, records)
    assert primary(value)["worst_best_bounds_pp"] == [0, 100 / (64 * 4)]
    assert value["coverage"]["infrastructure_failed"] == 1


def test_auxiliary_missing_preserves_numeric_primary_but_conservative_exclusion_guard_blocks():
    plan, records = fixture()
    records = [row for row in records if row["scaffold"] != "replacement"]
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0 and primary(value)["fully_observed"]
    assert primary(value)["approximate_interval_excludes_loss_of_10pp_or_more"] is None


def test_cap_is_strict_failure_even_when_parser_category_exact():
    plan, records = fixture()
    row = next(row for row in records if row["scaffold"] == "full")
    wrong(row, "exact", cap=True)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == -100 / (64 * 4)
    assert value["scaffold_coverage"]["full"]["parser_categories"]["exact"] == 256
    assert value["strict_outcome_decomposition_contrasts"]["capped"]["estimate_pp"] == 100 / (64 * 4)


def test_format_and_wrong_selector_damage_remain_distinct():
    plan, records = fixture()
    group = [row for row in records if row["scaffold"] == "full"]
    wrong(group[0], "format")
    wrong(group[1], "other_selector")
    value = result(plan, records)["strict_outcome_decomposition_contrasts"]
    assert value["format"]["estimate_pp"] == value["other_selector"]["estimate_pp"] == 100 / (64 * 4)


def test_opposite_placement_effects_are_visible_under_pooled_zero():
    plan, records = fixture()
    for row in records:
        if (row["scaffold"], row["placement"]) in (("full", "before"), ("sham", "after")):
            wrong(row)
    value = result(plan, records)
    assert primary(value)["estimate_pp"] == 0
    assert value["secondary_placement_contrasts"]["before"]["estimate_pp"] == -100
    assert value["secondary_placement_contrasts"]["after"]["estimate_pp"] == 100


def test_identical_world_contrast_has_zero_bootstrap_width_not_cell_variance():
    plan, records = fixture()
    for row in records:
        if (row["scaffold"], row["placement"], row["selector"]) == ("sham", "after", "B"):
            wrong(row)
    value = primary(result(plan, records))
    assert value["estimate_pp"] == 25 and value["approximate_interval_95_pp"] == [25, 25]


def test_perfectly_clustered_cell_errors_use64_worlds_not256_pairs():
    plan, records = fixture()
    negative = set(sorted({row["world_id"] for row in records})[:32])
    for row in records:
        if row["scaffold"] == ("full" if row["world_id"] in negative else "sham"):
            wrong(row)
    value = analysis.analyze_plan_records(plan, records, bootstrap_replicates=1000)
    interval = primary(value)["approximate_interval_95_pp"]
    assert primary(value)["estimate_pp"] == 0 and interval[1] - interval[0] >= 40


def test_row_order_does_not_change_statistics_or_hashes():
    plan, records = fixture()
    left = result(plan, records)
    records.reverse()
    right = result(plan, records)
    assert left == right


@pytest.mark.parametrize("mutation", ("likelihood", "raw", "metadata", "bool", "cap", "duplicate"))
def test_invalid_or_unbound_numeric_records_are_rejected(mutation):
    plan, records = fixture("development")
    if mutation == "likelihood":
        records[0]["likelihood_collected"] = True
    elif mutation == "raw":
        records[0]["response_text"] = "Invented response"
    elif mutation == "metadata":
        records[0]["query_order"] = "BA" if records[0]["query_order"] == "AB" else "AB"
    elif mutation == "bool":
        records[0]["score"]["strict_correct"] = 1
    elif mutation == "cap":
        records[0]["finish_reason"] = "length"
    else:
        records.append(deepcopy(records[0]))
    with pytest.raises(analysis.AnalysisError):
        result(plan, records)


def test_fixed64_and_query_presentation_balance_are_required():
    plan, records = fixture()
    plan["world_count"] = 63
    with pytest.raises(analysis.AnalysisError, match="fixed_counts"):
        result(plan, records)
    plan, records = fixture()
    for trial in plan["trials"]:
        trial["query_order"] = "AB"
    with pytest.raises(analysis.AnalysisError, match="presentation_balance"):
        result(plan, records)


def test_summary_has_no_raw_prompt_answers_or_world_identity():
    plan, records = fixture()
    value = result(plan, records)
    raw = json.dumps(value)
    assert "Invented harmless" not in raw and plan["trials"][0]["world_id"] not in raw
    assert '"selected_answer"' not in raw and '"messages"' not in raw


def test_cli_hash_binding_fixed10000_and_exclusive_output(tmp_path, capsys):
    plan, records = fixture("development")
    paths = []
    for name, value in (("plan", plan), ("records", records)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        paths += ["--" + name, str(path), "--" + name + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()]
    args = paths + ["--output", str(tmp_path / "summary.json")]
    assert analysis.main(args) == 0
    assert json.loads((tmp_path / "summary.json").read_bytes())["bootstrap_replicates"] == 10000
    assert analysis.main(args) == 2
    assert "Invented harmless" not in capsys.readouterr().out
