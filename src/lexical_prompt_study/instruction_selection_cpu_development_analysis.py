"""Descriptive A165 development aggregates, with no inference or model calls.

Use only receipt-validated numeric exports from the A165 runtime. All 288 planned
cells remain in denominators. Sixteen previously observed development worlds
cannot provide a new confirmatory result or a held-out population estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

WORLD_COUNT = 16
CELL_COUNT = 288
FLOOR_PERCENT = 10
CATEGORIES = ("exact", "other_selector", "other_answer", "format")
SCAFFOLDS = ("full", "sham", "replacement", "inert")
META_FIELDS = ("trial_id", "world_id", "partition", "task_family", "renderer", "query_order",
               "scaffold", "placement", "selector")
RECORD_FIELDS = {"schema_version", *META_FIELDS, "generation_status", "finish_reason", "score",
                 "likelihood_collected"}


class AnalysisError(ValueError):
    """Closed errors never disclose source content or private identifiers."""


def require(ok, code):
    if not ok:
        raise AnalysisError("a165_analysis_" + code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(value, length=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _key(row):
    return row["world_id"], row["scaffold"], row["placement"], row["selector"]


def _metadata(trial):
    require(type(trial) is dict, "trial_type")
    row = {key: trial.get(key) for key in META_FIELDS if key != "scaffold"}
    row["scaffold"] = trial.get("scaffold_kind")
    require(_digest(row["world_id"]) and _digest(row["trial_id"], 24), "identifiers")
    require(row["partition"] == "development" and row["task_family"] == "direct_answer_selection"
            and row["renderer"] == "json" and row["query_order"] in ("AB", "BA")
            and row["selector"] in ("A", "B"), "trial_metadata")
    require((row["scaffold"] == "none" and row["placement"] == "none")
            or (row["scaffold"] in SCAFFOLDS and row["placement"] in ("before", "after")), "scaffold_scope")
    return row


def manifest_from_plan(plan):
    """Validate numeric scope; the runtime additionally validates prompt/oracle bytes."""
    require(type(plan) is dict and plan.get("schema_version") == "a165-plan-v1", "plan_schema")
    require(plan.get("partition") == "development" and plan.get("stage") == "development_scaffold"
            and plan.get("task_family") == "direct_answer_selection" and plan.get("renderer") == "json"
            and plan.get("likelihood_collected") is False, "plan_instrument")
    require(_digest(plan.get("cohort_sha256")), "cohort_binding")
    bindings = plan.get("bindings")
    require(type(bindings) is dict and {"protocol_sha256", "materials_sha256", "task_source_sha256"} <= set(bindings)
            and all(type(key) is str and re.fullmatch(r"[a-z][a-z0-9_]*_sha256", key)
                    and _digest(value) for key, value in bindings.items()), "source_bindings")
    base = plan.get("base_plan")
    require(type(base) is dict and base.get("schema_version") == "a164-plan-v1"
            and base.get("partition") == "development" and base.get("stage") == "development_baseline"
            and base.get("world_count") == WORLD_COUNT and base.get("trial_count") == 32
            and base.get("likelihood_collected") is False and base.get("cohort_sha256") == plan["cohort_sha256"]
            and _digest(plan.get("base_plan_sha256")) and sha(base) == plan["base_plan_sha256"], "original_development_binding")
    require(type(base.get("trials")) is list and len(base["trials"]) == 32, "original_baseline_count")
    original, original_keys, original_ids = {}, set(), set()
    for trial in base["trials"]:
        row = _metadata(trial)
        require(row["scaffold"] == "none" and _key(row) not in original_keys
                and row["trial_id"] not in original_ids, "original_baseline_matrix")
        require(original.setdefault(row["world_id"], row["query_order"]) == row["query_order"],
                "original_presentation_drift")
        original_keys.add(_key(row))
        original_ids.add(row["trial_id"])
    require(len(original) == WORLD_COUNT and original_keys == {
        (world, "none", "none", selector) for world in original for selector in ("A", "B")
    }, "original_world_count")
    require(type(plan.get("trials")) is list and len(plan["trials"]) == CELL_COUNT, "trials")
    worlds, cells, identifiers, keys = {}, [], set(), set()
    for index, trial in enumerate(plan["trials"]):
        row = _metadata(trial)
        require(row["world_id"] in original and row["query_order"] == original[row["world_id"]],
                "original_world_identity")
        require((index < 32) == (row["scaffold"] == "none"), "baseline_first")
        world = {"world_id": row["world_id"], "query_order": row["query_order"]}
        require(worlds.setdefault(row["world_id"], world) == world, "world_presentation_drift")
        require(row["trial_id"] not in identifiers and row["trial_id"] not in original_ids
                and _key(row) not in keys, "duplicate_or_reused_planned_cell")
        identifiers.add(row["trial_id"])
        keys.add(_key(row))
        cells.append(row)
    require(len(worlds) == WORLD_COUNT and type(plan.get("world_count")) is int
            and plan["world_count"] == WORLD_COUNT and type(plan.get("trial_count")) is int
            and plan["trial_count"] == CELL_COUNT, "fixed_counts")
    require(all(sum(world["query_order"] == order for world in worlds.values()) == 8
                for order in ("AB", "BA")), "presentation_balance")
    matrix = {("none", "none", selector) for selector in ("A", "B")} | {
        (scaffold, placement, selector) for scaffold in SCAFFOLDS
        for placement in ("before", "after") for selector in ("A", "B")}
    for world in worlds:
        require({(r["scaffold"], r["placement"], r["selector"]) for r in cells if r["world_id"] == world}
                == matrix, "complete_planned_matrix")
    return {"schema_version": "a165-analysis-manifest-v1",
            "worlds": sorted(worlds.values(), key=lambda world: world["world_id"]),
            "expected_cells": sorted(cells, key=_key), "cohort_sha256": plan["cohort_sha256"],
            "base_plan_sha256": plan["base_plan_sha256"], "provenance": dict(bindings),
            "plan_object_sha256": sha(plan)}


def validate_records(manifest, records):
    require(type(records) is list, "records_type")
    expected = {row["trial_id"]: row for row in manifest["expected_cells"]}
    result = {}
    for row in records:
        require(type(row) is dict and set(row) == RECORD_FIELDS
                and row["schema_version"] == "a165-cell-v1" and row["likelihood_collected"] is False,
                "record_schema")
        identity = row["trial_id"]
        require(type(identity) is str and identity in expected and identity not in result, "terminal_record_identity")
        require(all(row[key] == expected[identity][key] and type(row[key]) is type(expected[identity][key])
                    for key in META_FIELDS), "record_metadata_binding")
        require(type(row["score"]) is dict and set(row["score"]) == {"category", "strict_correct"}, "score_schema")
        require(row["generation_status"] in ("completed", "infrastructure_failed"), "generation_status")
        if row["generation_status"] == "completed":
            require(row["finish_reason"] in ("eos", "length") and row["score"]["category"] in CATEGORIES
                    and type(row["score"]["strict_correct"]) is bool
                    and row["score"]["strict_correct"] == (row["finish_reason"] == "eos"
                                                          and row["score"]["category"] == "exact"),
                    "strict_score_consistency")
        else:
            require(row["finish_reason"] is None and row["score"]["category"] is None
                    and row["score"]["strict_correct"] is None, "missing_generation")
        result[identity] = row
    return {_key(row): row for row in result.values()}


def _binary(row, outcome="strict_correct"):
    if row is None or row["generation_status"] != "completed":
        return 0., 1.
    if outcome == "strict_correct":
        value = row["score"]["strict_correct"]
    elif outcome == "capped":
        value = row["finish_reason"] == "length"
    else:
        value = row["finish_reason"] == "eos" and row["score"]["category"] == outcome
    return float(value), float(value)


def _row(rows, world, scaffold, placement, selector):
    return rows.get((world["world_id"], scaffold, "none" if scaffold == "none" else placement, selector))


def _summary(values):
    require(bool(values), "empty_estimator")
    low = sum(value[0] for value in values) / len(values)
    high = sum(value[1] for value in values) / len(values)
    identified = sum(a == b for a, b in values)
    return {"worlds": len(values), "identified_worlds": identified,
            "binary_outcome_fully_identified": identified == len(values),
            "estimate_pp": 100 * low if identified == len(values) else None,
            "worst_best_bounds_pp": [100 * low, 100 * high], "unit": "percentage_points"}


def _contrast(manifest, rows, positive, negative, *, outcome="strict_correct", placement=None):
    positions = (placement,) if placement else ("before", "after")
    values = []
    for world in manifest["worlds"]:
        low = high = 0.
        for position in positions:
            for selector in ("A", "B"):
                plus = _binary(_row(rows, world, positive, position, selector), outcome)
                minus = _binary(_row(rows, world, negative, position, selector), outcome)
                low += plus[0] - minus[1]
                high += plus[1] - minus[0]
        values.append((low / (2 * len(positions)), high / (2 * len(positions))))
    return _summary(values)


def _coverage(cells, rows):
    result = {"expected": len(cells), "completed": 0, "infrastructure_failed": 0, "missing": 0,
              "strict_correct": 0, "capped": 0, "parser_categories": dict.fromkeys(CATEGORIES, 0),
              "uncapped_categories": dict.fromkeys(CATEGORIES, 0)}
    for cell in cells:
        row = rows.get(_key(cell))
        if row is None:
            result["missing"] += 1
        elif row["generation_status"] == "infrastructure_failed":
            result["infrastructure_failed"] += 1
        else:
            result["completed"] += 1
            result["strict_correct"] += row["score"]["strict_correct"]
            result["capped"] += row["finish_reason"] == "length"
            result["parser_categories"][row["score"]["category"]] += 1
            if row["finish_reason"] == "eos":
                result["uncapped_categories"][row["score"]["category"]] += 1
    unknown = result["missing"] + result["infrastructure_failed"]
    result["strict_accuracy_bounds_pp"] = [100 * result["strict_correct"] / len(cells),
        100 * (result["strict_correct"] + unknown) / len(cells)]
    return result


def _controls(manifest, rows):
    def subset(worlds):
        accuracy, pairs = [], []
        correct_cells = correct_pairs = completed = 0
        for world in worlds:
            a, b = [_binary(_row(rows, world, "none", "none", selector)) for selector in ("A", "B")]
            accuracy.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            pairs.append((a[0] * b[0], a[1] * b[1]))
            correct_cells += a[0] + b[0]
            correct_pairs += a[0] * b[0]
            completed += (a[0] == a[1]) + (b[0] == b[1])
        full = completed == len(worlds) * 2
        passed = full and 100 * correct_cells >= 90 * len(worlds) * 2 and 100 * correct_pairs >= 90 * len(worlds)
        return {"planned_cells": len(worlds) * 2, "completed_cells": completed,
                "no_scaffold_strict_accuracy": _summary(accuracy),
                "selector_pair_both_correct": _summary(pairs),
                "all_baseline_cells_completed": full, "gate_passed": bool(passed)}
    result = subset(manifest["worlds"])
    result["by_query_order"] = {order: subset([w for w in manifest["worlds"] if w["query_order"] == order])
                                for order in ("AB", "BA")}
    result["presentation_stratum_gates_are_diagnostic"] = True
    return result


def analyze_plan_records(plan, records):
    """Pure descriptive aggregation; no bootstrap, intervals or significance tests."""
    manifest = manifest_from_plan(plan)
    rows = validate_records(manifest, records)
    coverage = _coverage(manifest["expected_cells"], rows)
    complete = coverage["completed"] == CELL_COUNT
    controls = _controls(manifest, rows)
    arms = {scaffold: _coverage([r for r in manifest["expected_cells"] if r["scaffold"] == scaffold], rows)
            for scaffold in (*SCAFFOLDS, "none")}
    measured = all(arms[s]["completed"] == arms[s]["expected"] for s in ("full", "sham"))
    floor = all(100 * arms[s]["strict_correct"] <= FLOOR_PERCENT * arms[s]["expected"]
                for s in ("full", "sham")) if measured else None
    return {"schema_version": "a165-analysis-v1", "partition": "development", "stage": "development_scaffold",
        "status": "complete" if complete else "incomplete", "world_count": WORLD_COUNT,
        "task_family": "direct_answer_selection", "renderer": "json", "likelihood_collected": False,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": sha(manifest), "plan_object_sha256": manifest["plan_object_sha256"],
        "numeric_records_sha256": sha(sorted(records, key=lambda row: row["trial_id"])),
        "cohort_sha256": manifest["cohort_sha256"], "base_plan_sha256": manifest["base_plan_sha256"],
        "provenance": manifest["provenance"], "coverage": coverage, "baseline_controls": controls,
        "scaffold_coverage": arms,
        "primary_full_minus_sham_strict_correctness": _contrast(manifest, rows, "full", "sham"),
        "secondary_placement_contrasts": {p: _contrast(manifest, rows, "full", "sham", placement=p)
                                           for p in ("before", "after")},
        "secondary_query_order_contrasts": {order: _contrast({**manifest,
            "worlds": [w for w in manifest["worlds"] if w["query_order"] == order]}, rows, "full", "sham")
            for order in ("AB", "BA")},
        "strict_outcome_decomposition_contrasts": {category: _contrast(manifest, rows, "full", "sham",
            outcome=category) for category in ("strict_correct", "other_selector", "other_answer", "format", "capped")},
        "decomposition_note": "Other-selector, other-answer and format indicators exclude capped responses; raw parser categories remain in coverage.",
        "secondary_control_contrasts": {f"{plus}_minus_{minus}": _contrast(manifest, rows, plus, minus)
            for plus, minus in (("full", "replacement"), ("full", "inert"), ("sham", "none"), ("inert", "none"))},
        "interpretation_guards": {"baseline_competence_passed": controls["gate_passed"],
            "complete_scheduled_coverage": complete, "both_arm_floor_threshold_percent": FLOOR_PERCENT,
            "both_full_and_sham_at_or_below_floor": floor, "floor_guard_is_statistical_test": False,
            "failed_controls_or_floor_do_not_erase_numeric_estimates": True},
        "claim_boundaries": {"analysis_kind": "descriptive_exploratory_development",
            "previously_observed_development_worlds": True, "implementation_selected_after_competence_diagnostic": True,
            "worlds_are_fixed_descriptive_units": True, "cells_are_independent_units": False,
            "confirmatory_result": False, "general_task_population_estimate": False,
            "missing_bounds_are_confidence_intervals": False, "a164_gate_reopened": False,
            "heldout_allowed": False, "quantization_only_causal_claim": False,
            "format_failure_implies_selector_rebinding": False, "human_accuracy_or_harm_prevention": False,
            "raw_text_answers_or_world_ids_exported": False,
            "numeric_values_require_receipt_validated_runtime_export": True}}


def _read(path, expected):
    raw = path.read_bytes()
    require(_digest(expected) and hashlib.sha256(raw).hexdigest() == expected, "file_hash")
    return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "records", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "records-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_plan_records(_read(args.plan, args.plan_sha256), _read(args.records, args.records_sha256))
        result["input_files"] = {"plan_sha256": args.plan_sha256, "records_sha256": args.records_sha256}
        raw = canonical(result)
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"status": result["status"], "worlds": WORLD_COUNT,
                          "result_sha256": hashlib.sha256(raw).hexdigest()}))
        return 0
    except (ValueError, OSError, TypeError, KeyError):
        print('{"status":"a165_analysis_invalid_inputs"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
