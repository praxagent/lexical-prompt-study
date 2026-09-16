"""Aggregate-only fixed-cohort A164 analysis; no model calls or raw exports.

Input records must come from the frozen runtime's receipt-validated export.
The world bootstrap is approximate: it does not replay balanced derangement
sampling and does not imply IID general-task or exact design-based coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SEED = 20260915
BOOTSTRAP_REPLICATES = 10_000
FLOOR_PERCENT = 10
CATEGORIES = ("exact", "other_selector", "other_answer", "format")
SCAFFOLDS = ("full", "sham", "replacement", "inert")
META_FIELDS = ("trial_id", "world_id", "partition", "task_family", "renderer", "query_order",
               "scaffold", "placement", "selector")
RECORD_FIELDS = {"schema_version", *META_FIELDS, "generation_status", "finish_reason", "score",
                 "likelihood_collected"}


class AnalysisError(ValueError):
    """Closed errors never echo private content or identifiers."""


def require(ok, code):
    if not ok:
        raise AnalysisError("a164_analysis_" + code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(value, length=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _key(row):
    return row["world_id"], row["scaffold"], row["placement"], row["selector"]


def manifest_from_plan(plan):
    """Extract only typed metadata. Runtime separately validates task/oracle bytes."""
    require(type(plan) is dict and plan.get("schema_version") == "a164-plan-v1", "plan_schema")
    partition = plan.get("partition")
    require(partition in ("development", "heldout"), "partition")
    require(plan.get("stage") == ("development_baseline" if partition == "development" else "heldout")
            and plan.get("task_family") == "direct_answer_selection" and plan.get("renderer") == "json"
            and plan.get("likelihood_collected") is False, "plan_instrument")
    require(_digest(plan.get("cohort_sha256")), "cohort_binding")
    bindings = plan.get("bindings")
    require(type(bindings) is dict and {"protocol_sha256", "materials_sha256", "task_source_sha256"} <= set(bindings)
            and all(type(key) is str and re.fullmatch(r"[a-z][a-z0-9_]*_sha256", key)
                    and _digest(value) for key, value in bindings.items()), "source_bindings")
    require(type(plan.get("trials")) is list and bool(plan["trials"]), "trials")
    worlds, cells, identifiers, keys = {}, [], set(), set()
    for trial in plan["trials"]:
        require(type(trial) is dict, "trial_type")
        row = {key: trial.get(key) for key in META_FIELDS if key != "scaffold"}
        row["scaffold"] = trial.get("scaffold_kind")
        require(_digest(row["world_id"]) and _digest(row["trial_id"], 24), "identifiers")
        require(row["partition"] == partition and row["task_family"] == "direct_answer_selection"
                and row["renderer"] == "json" and row["query_order"] in ("AB", "BA")
                and row["selector"] in ("A", "B"), "trial_metadata")
        require((row["scaffold"] == "none" and row["placement"] == "none")
                or (partition == "heldout" and row["scaffold"] in SCAFFOLDS
                    and row["placement"] in ("before", "after")), "scaffold_scope")
        world = {"world_id": row["world_id"], "query_order": row["query_order"]}
        require(worlds.setdefault(row["world_id"], world) == world, "world_presentation_drift")
        require(row["trial_id"] not in identifiers and _key(row) not in keys, "duplicate_planned_cell")
        identifiers.add(row["trial_id"])
        keys.add(_key(row))
        cells.append(row)
    count = 16 if partition == "development" else 64
    per_world = 2 if partition == "development" else 18
    require(len(worlds) == count and type(plan.get("world_count")) is int and plan["world_count"] == count
            and type(plan.get("trial_count")) is int and plan["trial_count"] == len(cells) == count * per_world,
            "fixed_counts")
    require(all(sum(world["query_order"] == order for world in worlds.values()) == count // 2
                for order in ("AB", "BA")), "presentation_balance")
    matrix = {("none", "none", selector) for selector in ("A", "B")}
    if partition == "heldout":
        matrix |= {(scaffold, placement, selector) for scaffold in SCAFFOLDS
                   for placement in ("before", "after") for selector in ("A", "B")}
    for key in worlds:
        require({(r["scaffold"], r["placement"], r["selector"]) for r in cells if r["world_id"] == key}
                == matrix, "complete_planned_matrix")
    return {"schema_version": "a164-analysis-manifest-v1", "partition": partition,
            "stage": plan["stage"], "worlds": sorted(worlds.values(), key=lambda world: world["world_id"]),
            "expected_cells": sorted(cells, key=_key), "cohort_sha256": plan["cohort_sha256"],
            "provenance": dict(bindings), "plan_object_sha256": sha(plan)}


def validate_records(manifest, records):
    require(type(records) is list, "records_type")
    expected = {row["trial_id"]: row for row in manifest["expected_cells"]}
    result = {}
    for row in records:
        require(type(row) is dict and set(row) == RECORD_FIELDS
                and row["schema_version"] == "a164-cell-v1" and row["likelihood_collected"] is False,
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
    return np.asarray(values)


def _summary(values, replicates):
    require(len(values) > 0, "empty_estimator")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    boots = values[draws].mean(axis=1)
    bounds = values.mean(axis=0)
    complete_mask = values[:, 0] == values[:, 1]
    complete = bool(complete_mask.all())
    observed = values[complete_mask, 0]
    return {"worlds": len(values), "complete_worlds": len(observed), "fully_observed": complete,
            "estimate_pp": float(bounds[0] * 100) if complete else None,
            "worst_best_bounds_pp": (bounds * 100).tolist(),
            "approximate_interval_95_pp": [float(np.quantile(boots[:, 0], .025) * 100),
                                           float(np.quantile(boots[:, 1], .975) * 100)],
            "interval_kind": "approximate_world_percentile" if complete else "approximate_bootstrap_outer_bounds",
            "complete_world_sd_pp": float(np.std(observed, ddof=1) * 100) if len(observed) > 1 else None,
            "unit": "percentage_points"}


def _coverage(cells, rows):
    result = {"expected": len(cells), "completed": 0, "infrastructure_failed": 0, "missing": 0,
              "strict_correct": 0, "capped": 0, "parser_categories": dict.fromkeys(CATEGORIES, 0)}
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
    return result


def _controls(manifest, rows, replicates):
    def subset(worlds):
        accuracy, pairs = [], []
        for world in worlds:
            a, b = [_binary(_row(rows, world, "none", "none", selector)) for selector in ("A", "B")]
            accuracy.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            pairs.append((a[0] * b[0], a[1] * b[1]))
        accuracy, pairs = np.asarray(accuracy), np.asarray(pairs)
        acc, pair = _summary(accuracy, replicates), _summary(pairs, replicates)
        # Missing cells always prevent passage, even if an observed wrong partner
        # makes the binary pair indicator identifiable as zero.
        full = bool(np.all(accuracy[:, 0] == accuracy[:, 1]))
        passed = full and acc["estimate_pp"] >= 90 and pair["estimate_pp"] >= 90
        return {"no_scaffold_strict_accuracy": acc, "selector_pair_both_correct": pair,
                "all_baseline_cells_completed": full, "gate_passed": bool(passed)}
    result = subset(manifest["worlds"])
    result["by_query_order"] = {order: subset([w for w in manifest["worlds"] if w["query_order"] == order])
                                for order in ("AB", "BA")}
    result["presentation_stratum_gates_are_diagnostic"] = True
    return result


def analyze_plan_records(plan, records, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    """Pure numeric analysis. Custom replicate counts are for synthetic tests."""
    require(type(bootstrap_replicates) is int and bootstrap_replicates >= 100, "bootstrap_replicates")
    manifest = manifest_from_plan(plan)
    rows = validate_records(manifest, records)
    coverage = _coverage(manifest["expected_cells"], rows)
    complete = coverage["completed"] == coverage["expected"]
    controls = _controls(manifest, rows, bootstrap_replicates)
    result = {"schema_version": "a164-analysis-v1", "partition": manifest["partition"],
        "status": "complete" if complete else "incomplete", "world_count": len(manifest["worlds"]),
        "task_family": "direct_answer_selection", "renderer": "json", "likelihood_collected": False,
        "fixed_heldout_worlds": 64, "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": SEED,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": sha(manifest), "plan_object_sha256": manifest["plan_object_sha256"],
        "numeric_records_sha256": sha(sorted(records, key=lambda row: row["trial_id"])),
        "cohort_sha256": manifest["cohort_sha256"], "provenance": manifest["provenance"],
        "coverage": coverage, "baseline_controls": controls,
        "primary_full_minus_sham_strict_correctness": None,
        "interpretation_guards": {"baseline_competence_passed": controls["gate_passed"],
                                  "complete_scheduled_coverage": complete,
                                  "both_arm_floor_threshold_percent": FLOOR_PERCENT,
                                  "both_full_and_sham_at_or_below_floor": None,
                                  "practical_loss_exclusion_interpretable": False,
                                  "floor_guard_is_statistical_test": False},
        "claim_boundaries": {"task_pair_population_size": 240,
            "cohort_design": "five_disjoint_balanced_derangements",
            "bootstrap_replays_restricted_sampling": False, "exact_design_based_coverage": False,
            "iid_general_task_population": False, "cells_are_independent_units": False,
            "uncertainty_scope": "approximate_whole_world_resampling_conditional_on_fixed_implementation",
            "fixed_64_world_primary": True, "adaptive_extension_included": False,
            "equivalence_claim": False, "format_failure_implies_selector_rebinding": False,
            "human_accuracy_or_harm_prevention": False, "raw_text_answers_or_world_ids_exported": False,
            "numeric_values_require_receipt_validated_runtime_export": True}}
    if manifest["partition"] == "development":
        return result
    def summarize(values):
        return _summary(values, bootstrap_replicates)
    arms = {scaffold: _coverage([r for r in manifest["expected_cells"] if r["scaffold"] == scaffold], rows)
            for scaffold in (*SCAFFOLDS, "none")}
    primary = summarize(_contrast(manifest, rows, "full", "sham"))
    primary["prespecified_practical_loss_pp"] = -10
    measured = all(arms[s]["completed"] == arms[s]["expected"] for s in ("full", "sham"))
    floor = all(100 * arms[s]["strict_correct"] <= FLOOR_PERCENT * arms[s]["expected"]
                for s in ("full", "sham")) if measured else None
    eligible = complete and controls["gate_passed"] and floor is False
    primary["approximate_interval_excludes_loss_of_10pp_or_more"] = (
        primary["approximate_interval_95_pp"][0] > -10 if eligible else None)
    result["interpretation_guards"].update(both_full_and_sham_at_or_below_floor=floor,
                                         practical_loss_exclusion_interpretable=bool(eligible))
    result.update(primary_full_minus_sham_strict_correctness=primary, scaffold_coverage=arms,
        secondary_placement_contrasts={p: summarize(_contrast(manifest, rows, "full", "sham", placement=p))
                                       for p in ("before", "after")},
        secondary_query_order_contrasts={order: summarize(_contrast(manifest, rows, "full", "sham")[
            [w["query_order"] == order for w in manifest["worlds"]]]) for order in ("AB", "BA")},
        strict_outcome_decomposition_contrasts={category: summarize(_contrast(manifest, rows, "full", "sham",
            outcome=category)) for category in ("strict_correct", "other_selector", "other_answer", "format", "capped")},
        decomposition_note="Other-selector, other-answer and format indicators exclude capped responses; raw parser categories remain in coverage.",
        secondary_control_contrasts={f"{plus}_minus_{minus}": summarize(_contrast(manifest, rows, plus, minus))
            for plus, minus in (("full", "replacement"), ("full", "inert"), ("sham", "none"), ("inert", "none"))})
    return result


def _read(path, expected):
    raw = path.read_bytes()
    require(_digest(expected) and hashlib.sha256(raw).hexdigest() == expected, "file_hash")
    return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_plan_records(_read(args.plan, args.plan_sha256), _read(args.records, args.records_sha256))
        result["input_files"] = {"plan_sha256": args.plan_sha256, "records_sha256": args.records_sha256}
        raw = canonical(result) + b"\n"
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"status": result["status"], "worlds": result["world_count"],
                          "result_sha256": hashlib.sha256(raw).hexdigest()}))
        return 0
    except (ValueError, OSError, TypeError, KeyError):
        print('{"status":"a164_analysis_invalid_inputs"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
