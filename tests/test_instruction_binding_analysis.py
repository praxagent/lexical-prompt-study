from copy import deepcopy
import hashlib
import json
import random

import pytest

from lexical_prompt_study import instruction_binding_analysis as analysis
from lexical_prompt_study.instruction_binding_plan import compile_plan


def inputs(stage="development_pilot"):
    plan = compile_plan(stage=stage,
        scaffolds={kind: "Harmless synthetic context " + kind
                   for kind in ("full", "sham", "replacement", "inert")},
        bindings={"protocol_sha256": "a" * 64, "generator_sha256": "b" * 64,
                  "materials_sha256": "c" * 64, "compiler_sha256": "d" * 64})
    records = []
    for trial in plan["trials"]:
        records.append({"schema_version": "a163-cell-v1", "trial_id": trial["trial_id"],
            "world_id": trial["world_id"], "task_depth": trial["depth"],
            "partition": plan["partition"], "renderer": trial["renderer"],
            "query_order": trial["query_order"], "table_order": trial["table_order"],
            "scaffold": trial["scaffold_kind"], "placement": trial["placement"],
            "selector": trial["selector"], "diagnostic": trial["diagnostic"],
            "generation_status": "completed", "finish_reason": "eos",
            "score": {"category": "exact", "strict_correct": True},
            "teacher_forced": {"status": "ok", "selected_logprob": -.2,
                "alternative_logprob": -1.2, "selected_token_count": 2,
                "alternative_token_count": 3}})
    return plan, records


def analyze(plan, records):
    return analysis.analyze_plan_records(plan, records, bootstrap_replicates=100)


def fail_generation(row):
    row.update(generation_status="infrastructure_failed", finish_reason=None,
               score={"category": None, "strict_correct": None})


def wrong(row, category="other_selector"):
    row["score"] = {"category": category, "strict_correct": False}


def test_all_correct_fixed_matrix_has_zero_effect_and_separate_diagnostics():
    plan, records = inputs()
    result = analyze(plan, records)
    assert result["status"] == "complete"
    assert result["coverage"]["all"]["expected"] == 160
    assert result["coverage"]["primary_matrix"]["expected"] == 144
    assert result["coverage"]["diagnostics"]["expected"] == 16
    primary = result["primary_full_minus_sham_strict_correctness"]
    assert primary["estimate"] == 0 and primary["interval_95"] == [0, 0]
    assert primary["worlds"] == primary["complete_worlds"] == 8
    assert primary["equivalence_claim"] is False
    assert primary["interval_excludes_loss_of_10pp_or_more"] is True
    assert all(c["development_90_percent_gate_passed"] for c in result["baseline_controls"].values())
    serialized = json.dumps(result)
    assert "Harmless synthetic context" not in serialized and '"messages"' not in serialized
    assert '"selected_answer"' not in serialized
    assert all(row["world_id"] not in serialized for row in records)


def test_constant_world_effect_does_not_treat_cells_as_independent():
    plan, records = inputs()
    for row in records:
        if row["scaffold"] == "full" and row["task_depth"] == 1:
            wrong(row)
    result = analyze(plan, records)
    primary = result["primary_full_minus_sham_strict_correctness"]
    assert primary["estimate"] == -50
    assert primary["interval_95"] == [-50, -50]  # Each family has a constant effect.
    assert result["secondary_depth_contrasts"]["1"]["estimate"] == -100
    assert result["secondary_depth_contrasts"]["2"]["estimate"] == 0
    assert result["strict_outcome_decomposition_contrasts"]["other_selector"]["estimate"] == 50


def test_opposite_placement_effects_remain_visible():
    plan, records = inputs()
    for row in records:
        if (row["scaffold"], row["placement"]) in (("full", "before"), ("sham", "after")):
            wrong(row)
    result = analyze(plan, records)
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] == 0
    assert result["secondary_placement_contrasts"]["before"]["estimate"] == -100
    assert result["secondary_placement_contrasts"]["after"]["estimate"] == 100


@pytest.mark.parametrize("scaffold,bounds", [("full", [-3.125, 0]), ("sham", [0, 3.125])])
@pytest.mark.parametrize("mode", ["absent", "failure"])
def test_unresolved_primary_cell_has_all_world_binary_bounds(scaffold, bounds, mode):
    plan, records = inputs()
    target = next(row for row in records if row["scaffold"] == scaffold)
    if mode == "absent":
        records.remove(target)
    else:
        fail_generation(target)
    result = analyze(plan, records)
    primary = result["primary_full_minus_sham_strict_correctness"]
    assert primary["estimate"] is None and primary["worst_best_bounds"] == bounds
    assert primary["worlds"] == 8 and primary["complete_worlds"] == 7
    assert primary["interval_kind"] == "bootstrap_outer_interval_for_worst_best_bounds"
    assert primary["interval_95"][0] <= bounds[0] <= bounds[1] <= primary["interval_95"][1]
    assert primary["interval_excludes_loss_of_10pp_or_more"] is None


def test_all_primary_outcomes_missing_have_minus100_plus100_bounds():
    plan, records = inputs()
    records = [row for row in records if row["scaffold"] not in ("full", "sham")]
    result = analyze(plan, records)
    primary = result["primary_full_minus_sham_strict_correctness"]
    assert primary["estimate"] is None and primary["worst_best_bounds"] == [-100, 100]
    assert primary["interval_95"] == [-100, 100]
    assert result["conditional_answer_margin"]["full_cohort_estimate_nats"] is None
    assert result["conditional_answer_margin"]["complete_world_description"] is None


def test_missing_auxiliary_and_failed_diagnostics_do_not_erase_observed_primary():
    plan, records = inputs()
    records = [row for row in records if row["scaffold"] != "inert"]
    for row in records:
        if row["diagnostic"]:
            wrong(row, "format")
    result = analyze(plan, records)
    assert result["status"] == "incomplete"
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] == 0
    assert result["baseline_controls"]["json"]["development_90_percent_gate_passed"] is True
    assert result["baseline_controls"]["line_table"]["development_90_percent_gate_passed"] is False
    assert result["exploratory_control_contrasts"]["full_minus_inert"]["estimate"] is None


def test_cap_counts_as_failure_even_for_valid_exact_json_parser_category():
    plan, records = inputs()
    for row in records:
        if row["scaffold"] == "full":
            row["finish_reason"] = "length"
            row["score"]["strict_correct"] = False
    result = analyze(plan, records)
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] == -100
    assert result["scaffold_coverage"]["full"]["parser_categories"]["exact"] == 32
    assert result["scaffold_coverage"]["full"]["strict_correct"] == 0
    assert result["strict_outcome_decomposition_contrasts"]["capped"]["estimate"] == 100
    assert result["strict_outcome_decomposition_contrasts"]["format"]["estimate"] == 0


def test_format_failures_are_not_wrong_selector_answers():
    plan, records = inputs()
    for row in records:
        if row["scaffold"] == "full":
            wrong(row, "format")
    result = analyze(plan, records)
    assert result["strict_outcome_decomposition_contrasts"]["format"]["estimate"] == 100
    assert result["strict_outcome_decomposition_contrasts"]["other_selector"]["estimate"] == 0


def test_teacher_forced_secondary_uses_summed_loglikelihoods_without_length_normalization():
    plan, records = inputs()
    for row in records:
        if row["scaffold"] == "full":
            row["teacher_forced"]["selected_logprob"] = -.8
    result = analyze(plan, records)
    margin = result["conditional_answer_margin"]
    assert margin["full_cohort_estimate_nats"] == pytest.approx(-.6)
    assert margin["full_cohort_interval_95_nats"] == pytest.approx([-.6, -.6])
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] == 0


def test_missing_likelihood_does_not_impute_zero_or_fail_observed_generation():
    plan, records = inputs()
    row = next(row for row in records if row["scaffold"] == "full")
    row["teacher_forced"] = dict.fromkeys(analysis.TF_FIELDS)
    row["teacher_forced"]["status"] = "failed"
    result = analyze(plan, records)
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] == 0
    margin = result["conditional_answer_margin"]
    assert margin["full_cohort_estimate_nats"] is None and margin["missing_score_bounds"] is None
    assert margin["complete_worlds"] == 7 and margin["complete_world_description"]["worlds"] == 7
    assert result["coverage"]["all"]["teacher_forced_failed"] == 1


def test_generation_failure_and_likelihood_success_have_separate_coverage():
    plan, records = inputs()
    fail_generation(next(row for row in records if row["scaffold"] == "full"))
    result = analyze(plan, records)
    assert result["primary_full_minus_sham_strict_correctness"]["estimate"] is None
    assert result["conditional_answer_margin"]["full_cohort_estimate_nats"] == 0


def test_heldout_fixed64_and_no_diagnostic_leakage():
    plan, records = inputs("heldout")
    result = analyze(plan, records)
    assert result["world_count"] == 64 and result["coverage"]["all"]["expected"] == 1152
    assert result["coverage"]["diagnostics"]["expected"] == 0
    assert result["primary_renderer"] == "line_table"
    assert result["baseline_controls"]["line_table"]["development_90_percent_gate_passed"] is None
    lost = plan["trials"][0]["world_id"]
    plan["trials"] = [row for row in plan["trials"] if row["world_id"] != lost]
    plan["world_count"] -= 1
    plan["trial_count"] = len(plan["trials"])
    with pytest.raises(analysis.AnalysisError, match="fixed_world_count"):
        analyze(plan, records)


def test_source_row_order_does_not_change_statistics():
    plan, records = inputs()
    for row in records:
        if row["scaffold"] == "full" and int(row["world_id"], 16) % 3 == 0:
            wrong(row)
    original = analyze(plan, records)
    shuffled_plan = deepcopy(plan)
    random.Random(17).shuffle(shuffled_plan["trials"])
    random.Random(18).shuffle(records)
    shuffled = analyze(shuffled_plan, records)
    for key in ("primary_full_minus_sham_strict_correctness", "secondary_depth_contrasts",
                "secondary_placement_contrasts", "conditional_answer_margin", "numeric_records_sha256"):
        assert original[key] == shuffled[key]


@pytest.mark.parametrize("mutation,code", [
    (lambda row: row.update(renderer="line_table"), "record_metadata_binding"),
    (lambda row: row.update(query_order="BA" if row["query_order"] == "AB" else "AB"), "record_metadata_binding"),
    (lambda row: row.update(diagnostic=1), "record_metadata_binding"),
    (lambda row: row["score"].update(strict_correct=1), "strict_score_consistency"),
    (lambda row: row.update(finish_reason="length"), "strict_score_consistency"),
    (lambda row: row["teacher_forced"].update(selected_logprob=float("nan")), "likelihood_values"),
    (lambda row: row["teacher_forced"].update(selected_logprob=.2), "likelihood_values"),
    (lambda row: row["teacher_forced"].update(selected_token_count=True), "candidate_token_counts"),
    (lambda row: row["teacher_forced"].update(status="failed"), "missing_likelihood"),
    (lambda row: row.update(response_text="must not export"), "record_schema"),
])
def test_record_contract_rejects_drift_and_invalid_scores(mutation, code):
    plan, records = inputs()
    mutation(records[0])
    with pytest.raises(analysis.AnalysisError, match=code):
        analyze(plan, records)


def test_duplicate_terminal_attempts_are_rejected_not_counted_as_replication():
    plan, records = inputs()
    with pytest.raises(analysis.AnalysisError, match="duplicate_terminal_record"):
        analyze(plan, records + [deepcopy(records[0])])


def test_presentation_counterbalance_is_validated_from_plan_metadata():
    plan, records = inputs()
    first_world = plan["trials"][0]["world_id"]
    for row in plan["trials"]:
        if row["world_id"] == first_world:
            row["query_order"] = "BA"
    with pytest.raises(analysis.AnalysisError, match="presentation_balance"):
        analyze(plan, records)


def test_missing_selector_pair_not_silently_removed_from_controls():
    plan, records = inputs()
    target = next(row for row in records if row["scaffold"] == "none" and not row["diagnostic"])
    records.remove(target)
    result = analyze(plan, records)
    control = result["baseline_controls"]["json"]
    assert control["development_90_percent_gate_passed"] is False
    assert control["selector_pair_both_correct"]["worst_best_bounds"] == [87.5, 100]
    assert control["no_scaffold_strict_accuracy"]["worst_best_bounds"] == [93.75, 100]


def test_cli_hash_binds_inputs_and_creates_aggregate_once(tmp_path):
    plan, records = inputs()
    paths = {}
    for name, value in (("plan", plan), ("records", records)):
        path = tmp_path / (name + ".json")
        raw = analysis.canonical(value) + b"\n"
        path.write_bytes(raw)
        paths[name] = (path, hashlib.sha256(raw).hexdigest())
    output = tmp_path / "aggregate.json"
    args = ["--plan", str(paths["plan"][0]), "--plan-sha256", paths["plan"][1],
            "--records", str(paths["records"][0]), "--records-sha256", paths["records"][1],
            "--output", str(output)]
    assert analysis.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["bootstrap_replicates"] == 10_000 and value["bootstrap_seed"] == 20260915
    assert value["input_files"]["plan_sha256"] == paths["plan"][1]
    assert analysis.main(args) == 2
    paths["records"][0].write_text("[]")
    args[-1] = str(tmp_path / "should-not-exist.json")
    assert analysis.main(args) == 2 and not (tmp_path / "should-not-exist.json").exists()
