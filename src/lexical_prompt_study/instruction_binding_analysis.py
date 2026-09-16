"""Fixed-cohort A163 numeric analysis; no model calls or raw-output exports.

The plan fixes every scheduled cell. Missing terminal records remain unresolved
in binary bounds. Worlds, stratified by task depth, are the resampling units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SEED = 20260915
BOOTSTRAP_REPLICATES = 10_000
CATEGORIES = ("exact", "other_selector", "other_answer", "format")
SCAFFOLDS = ("full", "sham", "replacement", "inert")
META_FIELDS = ("trial_id", "world_id", "task_depth", "partition", "renderer", "query_order",
               "table_order", "scaffold", "placement", "selector", "diagnostic")
RECORD_FIELDS = {"schema_version", *META_FIELDS, "generation_status", "finish_reason", "score",
                 "teacher_forced"}
TF_FIELDS = {"status", "selected_logprob", "alternative_logprob", "selected_token_count",
             "alternative_token_count"}


class AnalysisError(ValueError):
    """Closed errors never echo private input text or row identifiers."""


def require(value: bool, code: str) -> None:
    if not value:
        raise AnalysisError("a163_analysis_" + code)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(value, length=64) -> bool:
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _key(value: dict) -> tuple:
    return (value["world_id"], value["renderer"], value["scaffold"], value["placement"],
            value["selector"], value["diagnostic"])


def manifest_from_plan(plan: dict) -> dict:
    """Extract numeric metadata only; never return messages, answers, or world contents."""
    require(type(plan) is dict and plan.get("schema_version") == "a163-plan-v1", "plan_schema")
    require(plan.get("partition") in ("development", "heldout"), "partition")
    partition = plan["partition"]
    require(plan.get("stage") in (("heldout",) if partition == "heldout"
                                  else ("development_pilot", "development")), "stage")
    require(type(plan.get("trials")) is list and bool(plan["trials"]), "trials")
    bindings = plan.get("bindings")
    require(type(bindings) is dict and bool(bindings)
            and all(type(k) is str and re.fullmatch(r"[a-z][a-z0-9_]*_sha256", k)
                    and _digest(v) for k, v in bindings.items()), "plan_bindings")
    worlds, expected, ids, keys = {}, [], set(), set()
    primary_renderer = "line_table" if partition == "heldout" else "json"
    for trial in plan["trials"]:
        require(type(trial) is dict, "trial_schema")
        row = {"trial_id": trial.get("trial_id"), "world_id": trial.get("world_id"),
               "task_depth": trial.get("depth"), "partition": partition,
               "renderer": trial.get("renderer"), "query_order": trial.get("query_order"),
               "table_order": trial.get("table_order"), "scaffold": trial.get("scaffold_kind"),
               "placement": trial.get("placement"), "selector": trial.get("selector"),
               "diagnostic": trial.get("diagnostic")}
        require(_digest(row["world_id"]) and _digest(row["trial_id"], 24), "identifiers")
        require(type(row["task_depth"]) is int and row["task_depth"] in (1, 2), "task_depth")
        require(row["query_order"] in ("AB", "BA")
                and row["table_order"] in ("forward", "reverse")
                and (row["task_depth"] == 2 or row["table_order"] == "forward"), "presentation")
        require(row["renderer"] in ("json", "line_table") and row["selector"] in ("A", "B")
                and type(row["diagnostic"]) is bool, "condition")
        require((row["scaffold"] == "none" and row["placement"] == "none")
                or (row["scaffold"] in SCAFFOLDS and row["placement"] in ("before", "after")),
                "scaffold_placement")
        require((not row["diagnostic"] and row["renderer"] == primary_renderer)
                or (partition == "development" and row["diagnostic"]
                    and row["renderer"] == "line_table" and row["scaffold"] == "none"),
                "diagnostic_scope")
        world = {k: row[k] for k in ("world_id", "task_depth", "query_order", "table_order")}
        require(worlds.setdefault(row["world_id"], world) == world, "world_metadata_drift")
        require(row["trial_id"] not in ids and _key(row) not in keys, "duplicate_planned_cell")
        ids.add(row["trial_id"])
        keys.add(_key(row))
        expected.append(row)
    count = len(worlds)
    require(type(plan.get("world_count")) is int and plan["world_count"] == count
            and type(plan.get("trial_count")) is int and plan["trial_count"] == len(expected),
            "plan_counts")
    require(count == 64 if partition == "heldout" else count in (8, 16), "fixed_world_count")
    if plan["stage"] == "development_pilot":
        require(count == 8, "pilot_world_count")
    for depth in (1, 2):
        family = [w for w in worlds.values() if w["task_depth"] == depth]
        require(len(family) == count // 2, "family_balance")
        combinations = [(q, t) for q in ("AB", "BA")
                        for t in (("forward", "reverse") if depth == 2 else ("forward",))]
        require(all(sum((w["query_order"], w["table_order"]) == combination for w in family)
                    == len(family) // len(combinations) for combination in combinations),
                "presentation_balance")
    main_cells = {("none", "none", selector) for selector in ("A", "B")}
    main_cells |= {(s, p, a) for s in SCAFFOLDS for p in ("before", "after") for a in ("A", "B")}
    for identifier in worlds:
        observed = {(r["scaffold"], r["placement"], r["selector"]) for r in expected
                    if r["world_id"] == identifier and not r["diagnostic"]}
        require(observed == main_cells, "full_main_matrix")
        diagnostic = [r for r in expected if r["world_id"] == identifier and r["diagnostic"]]
        require(len(diagnostic) == (2 if partition == "development" else 0), "diagnostic_matrix")
    return {"schema_version": "a163-analysis-manifest-v1", "partition": partition,
            "stage": plan["stage"], "primary_renderer": primary_renderer,
            "worlds": sorted(worlds.values(), key=lambda w: w["world_id"]),
            "expected_cells": sorted(expected, key=_key), "provenance": dict(bindings),
            "plan_object_sha256": sha(plan)}


def _validate_records(manifest: dict, records: list) -> dict:
    require(type(records) is list, "records_type")
    expected = {r["trial_id"]: r for r in manifest["expected_cells"]}
    result = {}
    for row in records:
        require(type(row) is dict and set(row) == RECORD_FIELDS
                and row["schema_version"] == "a163-cell-v1", "record_schema")
        identifier = row["trial_id"]
        require(type(identifier) is str and identifier in expected, "unplanned_record")
        require(identifier not in result, "duplicate_terminal_record")
        require(all(row[k] == expected[identifier][k] and type(row[k]) is type(expected[identifier][k])
                    for k in META_FIELDS), "record_metadata_binding")
        require(type(row["score"]) is dict and set(row["score"]) == {"category", "strict_correct"},
                "score_schema")
        require(row["generation_status"] in ("completed", "infrastructure_failed"), "generation_status")
        if row["generation_status"] == "completed":
            require(row["finish_reason"] in ("eos", "length")
                    and row["score"]["category"] in CATEGORIES
                    and type(row["score"]["strict_correct"]) is bool
                    and row["score"]["strict_correct"] == (
                        row["finish_reason"] == "eos" and row["score"]["category"] == "exact"),
                    "strict_score_consistency")
        else:
            require(row["finish_reason"] is None and row["score"]["category"] is None
                    and row["score"]["strict_correct"] is None, "missing_generation")
        tf = row["teacher_forced"]
        require(type(tf) is dict and set(tf) == TF_FIELDS and tf["status"] in ("ok", "failed"),
                "teacher_forced_schema")
        if tf["status"] == "failed":
            require(all(tf[k] is None for k in TF_FIELDS - {"status"}), "missing_likelihood")
        else:
            require(all(type(tf[k]) in (int, float) and math.isfinite(tf[k]) and tf[k] <= 0
                        for k in ("selected_logprob", "alternative_logprob")), "likelihood_values")
            require(all(type(tf[k]) is int and tf[k] > 0
                        for k in ("selected_token_count", "alternative_token_count")),
                    "candidate_token_counts")
        result[identifier] = row
    return {_key(row): row for row in result.values()}


def _binary(row: dict | None, outcome: str) -> tuple[float, float]:
    if row is None or row["generation_status"] != "completed":
        return 0.0, 1.0
    if outcome == "strict_correct":
        value = row["score"]["strict_correct"]
    elif outcome == "capped":
        value = row["finish_reason"] == "length"
    else:
        value = row["finish_reason"] == "eos" and row["score"]["category"] == outcome
    return float(value), float(value)


def _lookup(rows, world, renderer, scaffold, placement, selector, diagnostic=False):
    return rows.get((world["world_id"], renderer, scaffold, placement, selector, diagnostic))


def _contrast(manifest, rows, positive, negative, outcome="strict_correct", placement=None):
    placements = (placement,) if placement is not None else ("before", "after")
    values = []
    for world in manifest["worlds"]:
        low = high = 0.0
        for position in placements:
            for selector in ("A", "B"):
                pair = []
                for scaffold in (positive, negative):
                    row = _lookup(rows, world, manifest["primary_renderer"], scaffold,
                                  "none" if scaffold == "none" else position, selector)
                    pair.append(_binary(row, outcome))
                low += pair[0][0] - pair[1][1]
                high += pair[0][1] - pair[1][0]
        values.append((low / (2 * len(placements)), high / (2 * len(placements))))
    return np.asarray(values, dtype=float)


def _summarize_bounds(values, depths, *, replicates, depth=None, scale=100.0):
    selected_depths = (depth,) if depth is not None else (1, 2)
    strata = [np.flatnonzero(depths == d) for d in selected_depths]
    require(all(len(indices) > 0 for indices in strata), "empty_stratum")
    rng = np.random.default_rng(SEED)
    means = np.mean([values[indices].mean(axis=0) for indices in strata], axis=0)
    boots = np.zeros((replicates, 2))
    for indices in strata:
        drawn = rng.integers(0, len(indices), size=(replicates, len(indices)))
        boots += values[indices][drawn].mean(axis=1) / len(strata)
    included = np.concatenate(strata)
    complete = bool(np.all(values[included, 0] == values[included, 1]))
    observed = values[included][values[included, 0] == values[included, 1], 0]
    return {"worlds": len(included), "complete_worlds": len(observed),
            "estimate": float(means[0] * scale) if complete else None,
            "worst_best_bounds": [float(v * scale) for v in means],
            "interval_95": [float(np.quantile(boots[:, 0], .025) * scale),
                            float(np.quantile(boots[:, 1], .975) * scale)],
            "interval_kind": ("family_stratified_world_percentile" if complete
                              else "bootstrap_outer_interval_for_worst_best_bounds"),
            "complete_world_sd": float(np.std(observed, ddof=1) * scale) if len(observed) > 1 else None,
            "unit": "percentage_points" if scale == 100 else "nats",
            "fully_observed": complete}


def _margin(manifest, rows, *, replicates, placement=None):
    values, depths, expected_depths = [], [], []
    positions = (placement,) if placement else ("before", "after")
    for world in manifest["worlds"]:
        expected_depths.append(world["task_depth"])
        terms = []
        for position in positions:
            for selector in ("A", "B"):
                pair = []
                for scaffold in ("full", "sham"):
                    row = _lookup(rows, world, manifest["primary_renderer"], scaffold, position, selector)
                    tf = row["teacher_forced"] if row is not None else None
                    pair.append(tf["selected_logprob"] - tf["alternative_logprob"]
                                if tf is not None and tf["status"] == "ok" else None)
                terms.append(pair[0] - pair[1] if all(v is not None for v in pair) else None)
        if all(value is not None for value in terms):
            value = float(np.mean(terms))
            values.append((value, value))
            depths.append(world["task_depth"])
    count = len(manifest["worlds"])
    summary = None
    if set(depths) == {1, 2}:
        summary = _summarize_bounds(np.asarray(values), np.asarray(depths), replicates=replicates, scale=1)
    complete = len(values) == count
    return {"expected_worlds": count, "complete_worlds": len(values),
            "complete_worlds_by_depth": {str(d): depths.count(d) for d in (1, 2)},
            "full_cohort_estimate_nats": summary["estimate"] if complete else None,
            "complete_world_description": summary,
            "full_cohort_interval_95_nats": summary["interval_95"] if complete else None,
            "missing_score_bounds": None,
            "limitation": "Conditional on canonical assistant prefix; incomplete coverage has no finite full-cohort bounds."}


def _coverage(cells, rows):
    counts = {"expected": len(cells), "completed": 0, "infrastructure_failed": 0, "missing": 0,
              "capped": 0, "strict_correct": 0, "teacher_forced_ok": 0, "teacher_forced_failed": 0,
              "parser_categories": dict.fromkeys(CATEGORIES, 0)}
    for cell in cells:
        row = rows.get(_key(cell))
        if row is None:
            counts["missing"] += 1
            continue
        counts[row["generation_status"]] += 1
        if row["generation_status"] == "completed":
            counts["capped"] += row["finish_reason"] == "length"
            counts["strict_correct"] += row["score"]["strict_correct"]
            counts["parser_categories"][row["score"]["category"]] += 1
        counts["teacher_forced_" + row["teacher_forced"]["status"]] += 1
    return counts


def _controls(manifest, rows, *, replicates):
    results = {}
    depths = np.asarray([w["task_depth"] for w in manifest["worlds"]])
    for renderer in sorted({r["renderer"] for r in manifest["expected_cells"]}):
        diagnostic = renderer != manifest["primary_renderer"]
        accuracy, switches = [], []
        for world in manifest["worlds"]:
            a, b = [_binary(_lookup(rows, world, renderer, "none", "none", selector, diagnostic),
                             "strict_correct") for selector in ("A", "B")]
            accuracy.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            switches.append((a[0] * b[0], a[1] * b[1]))
        acc = _summarize_bounds(np.asarray(accuracy), depths, replicates=replicates)
        switch = _summarize_bounds(np.asarray(switches), depths, replicates=replicates)
        results[renderer] = {"diagnostic_only": diagnostic,
            "no_scaffold_strict_accuracy": acc, "selector_pair_both_correct": switch,
            "development_90_percent_gate_passed": (
                acc["fully_observed"] and switch["fully_observed"]
                and acc["estimate"] >= 90 and switch["estimate"] >= 90)
                if manifest["partition"] == "development" else None,
            "by_depth": {str(d): {
                "no_scaffold_strict_accuracy": _summarize_bounds(np.asarray(accuracy), depths,
                                                                  replicates=replicates, depth=d),
                "selector_pair_both_correct": _summarize_bounds(np.asarray(switches), depths,
                                                                 replicates=replicates, depth=d)}
                for d in (1, 2)}}
    return results


def analyze_plan_records(plan: dict, records: list, *, bootstrap_replicates=BOOTSTRAP_REPLICATES) -> dict:
    """Return aggregate-only estimates. Custom bootstrap counts are for synthetic tests."""
    require(type(bootstrap_replicates) is int and bootstrap_replicates >= 100, "bootstrap_replicates")
    manifest = manifest_from_plan(plan)
    rows = _validate_records(manifest, records)
    depths = np.asarray([world["task_depth"] for world in manifest["worlds"]])
    def summarize(values, depth=None):
        return _summarize_bounds(values, depths, replicates=bootstrap_replicates, depth=depth)
    primary_values = _contrast(manifest, rows, "full", "sham")
    primary = summarize(primary_values)
    primary["prespecified_practical_loss_pp"] = -10
    primary["interval_excludes_loss_of_10pp_or_more"] = (
        primary["interval_95"][0] > -10 if primary["fully_observed"] else None)
    primary["equivalence_claim"] = False
    main = [row for row in manifest["expected_cells"] if not row["diagnostic"]]
    diagnostic = [row for row in manifest["expected_cells"] if row["diagnostic"]]
    all_coverage = _coverage(manifest["expected_cells"], rows)
    result = {
        "schema_version": "a163-analysis-v1", "partition": manifest["partition"], "stage": manifest["stage"],
        "status": "complete" if all_coverage["completed"] == all_coverage["expected"] else "incomplete",
        "world_count": len(manifest["worlds"]), "primary_renderer": manifest["primary_renderer"],
        "fixed_heldout_worlds": 64, "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": SEED,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": sha(manifest), "plan_object_sha256": manifest["plan_object_sha256"],
        "numeric_records_sha256": sha(sorted(records, key=lambda row: row["trial_id"])),
        "provenance": manifest["provenance"],
        "coverage": {"all": all_coverage, "primary_matrix": _coverage(main, rows),
                     "diagnostics": _coverage(diagnostic, rows)},
        "primary_full_minus_sham_strict_correctness": primary,
        "secondary_placement_contrasts": {p: summarize(_contrast(manifest, rows, "full", "sham", placement=p))
                                            for p in ("before", "after")},
        "secondary_depth_contrasts": {str(d): summarize(primary_values, depth=d) for d in (1, 2)},
        "strict_outcome_decomposition_contrasts": {
            name: summarize(_contrast(manifest, rows, "full", "sham", outcome=name))
            for name in ("strict_correct", "other_selector", "other_answer", "format", "capped")},
        "exploratory_control_contrasts": {
            f"{positive}_minus_{negative}": summarize(_contrast(manifest, rows, positive, negative))
            for positive, negative in (("full", "inert"), ("full", "replacement"),
                                       ("sham", "none"), ("inert", "none"))},
        "scaffold_coverage": {s: _coverage([r for r in main if r["scaffold"] == s], rows)
                              for s in (*SCAFFOLDS, "none")},
        "baseline_controls": _controls(manifest, rows, replicates=bootstrap_replicates),
        "conditional_answer_margin": _margin(manifest, rows, replicates=bootstrap_replicates),
        "conditional_answer_margin_by_placement": {
            p: _margin(manifest, rows, replicates=bootstrap_replicates, placement=p)
            for p in ("before", "after")},
        "claim_boundaries": {"fixed_sample_primary": True, "adaptive_extension_included": False,
                             "cells_are_independent_units": False, "diagnostics_in_primary": False,
                             "format_failure_implies_rebinding": False,
                             "teacher_forced_margin_is_unconditional_generation": False,
                             "human_ground_truth_or_harm_prevention": False,
                             "raw_text_answers_or_world_identifiers_in_output": False},
    }
    return result


def _read(path, expected_sha):
    raw = path.read_bytes()
    require(_digest(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha, "file_hash")
    return json.loads(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_plan_records(_read(args.plan, args.plan_sha256),
                                      _read(args.records, args.records_sha256))
        result["input_files"] = {"plan_sha256": args.plan_sha256, "records_sha256": args.records_sha256}
        raw = canonical(result) + b"\n"
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"status": result["status"], "worlds": result["world_count"],
                          "result_sha256": hashlib.sha256(raw).hexdigest()}))
        return 0
    except (ValueError, OSError, TypeError, KeyError):
        print('{"status":"a163_analysis_invalid_inputs"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
