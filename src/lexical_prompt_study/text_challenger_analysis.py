"""Aggregate-only analysis of the frozen lexical challenger's complete matrix.

Reads inventory, execution metadata and numeric cell summaries only. It does
not load prompts, generations, per-row predictions, or fitting dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

PLACEMENTS = ("scaffold_before_request", "scaffold_after_request")
FAMILIES = (None, "attack_block_mask", "structural_sham", "harmless_structured_wrapper")
GROUPS = ("all", "unsafe_direct", "safe_classify_exact", "safe_refuse_exact", "safe_acknowledge_exact")
HORIZONS = ("128", "256", "512", "1024")
MODELS = ("training_prevalence", "prompt_text", "prompt_plus_prefix_hash", "jlens_t8",
          "prompt_plus_prefix_hash_and_jlens")
COUNTS = ("selected", "known", "unknown", "positive", "request_cores", "right_censored", "capped_negative")
METRICS = ("log_loss", "brier", "roc_auc", "average_precision")
COMPARISONS = (("prompt_plus_prefix_hash", "jlens_t8"),
               ("prompt_plus_prefix_hash_and_jlens", "prompt_plus_prefix_hash"),
               ("prompt_plus_prefix_hash", "prompt_text"))
CELL_FIELDS = {"schema_version", "placement", "fold", "held_family", "target_horizon", "training_rows",
               "test_rows", "models", "interpretation", "confirmation_opened", "vocabulary_sizes",
               "status", "elapsed_seconds", "inventory_sha256", "eligible_rows"}
STATES = ("complete", "incomplete", "execution_failed", "missing", "interrupted")


class AnalysisError(ValueError):
    """Errors contain fixed codes, never source payloads or identifiers."""


def require(value, code):
    if not value:
        raise AnalysisError("text_matrix_" + code)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink")
    require(path.is_file() and 0 < path.stat().st_size <= 16 * 1024**2, "file_bounds")
    raw = path.read_bytes()
    require(expected is None or sha(raw) == expected, "file_hash")
    return raw


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def decode(raw):
    def invalid(_):
        raise AnalysisError("text_matrix_nonfinite_json")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def key(cell):
    return cell["placement"], cell["held_family"], cell["fold"]


def integer(value, lower=0):
    return type(value) is int and value >= lower


def finite(value, lower=0, upper=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= lower
            and (upper is None or value <= upper))


def validate_inventory(inventory):
    require(inventory["schema_version"] == "text-challenger-inventory-v1"
            and inventory["actual_fitting_performed"] is False
            and inventory["confirmation_opened"] is False, "inventory_schema")
    expected = {(placement, family, fold) for placement in PLACEMENTS
                for family in FAMILIES for fold in range(5)}
    cells = {}
    for cell in inventory["cells"]:
        require(key(cell) in expected and key(cell) not in cells, "inventory_matrix")
        require(type(cell["fold"]) is int and cell["both_training_classes"] is True
                and all(integer(cell[n], 1) for n in ("training_rows", "test_rows", "training_cores", "test_cores")),
                "inventory_support")
        require(cell["training_cores"] + cell["test_cores"] == inventory["request_cores"], "core_partition")
        cells[key(cell)] = cell
    require(set(cells) == expected, "inventory_matrix")
    return cells


def validate_metric(metric, expected):
    require(type(metric) is dict and set(metric) == set(COUNTS + METRICS), "metric_schema")
    require(all(integer(metric[n]) for n in COUNTS), "metric_counts")
    n, known, positive = metric["selected"], metric["known"], metric["positive"]
    require(known + metric["unknown"] == n and positive <= known and n <= expected["test_rows"]
            and metric["request_cores"] <= min(n, expected["test_cores"])
            and metric["right_censored"] <= n
            and metric["capped_negative"] <= min(known - positive, metric["right_censored"]), "metric_support")
    require((finite(metric["log_loss"]) and finite(metric["brier"], upper=1)) if known
            else metric["log_loss"] is None and metric["brier"] is None, "probability_metrics")
    two_classes = 0 < positive < known
    require(all(finite(metric[n], upper=1) for n in ("roc_auc", "average_precision")) if two_classes
            else metric["roc_auc"] is None and metric["average_precision"] is None, "ranking_support")


def validate_cell(value, expected, inventory_sha, eligible_rows):
    require(type(value) is dict and set(value) == CELL_FIELDS, "cell_schema")
    require(value["schema_version"] == "text-challenger-cell-v1" and key(value) == key(expected)
            and type(value["fold"]) is int and value["inventory_sha256"] == inventory_sha
            and value["eligible_rows"] == eligible_rows and value["target_horizon"] == 1024
            and value["training_rows"] == expected["training_rows"] and value["test_rows"] == expected["test_rows"]
            and value["interpretation"] == "retrospective_classifier_proxy"
            and value["confirmation_opened"] is False, "cell_binding")
    require(value["status"] in ("complete", "incomplete") and set(value["models"]) == set(MODELS)
            and finite(value["elapsed_seconds"])
            and type(value["vocabulary_sizes"]) is list and len(value["vocabulary_sizes"]) == 2
            and all(integer(n, 1) for n in value["vocabulary_sizes"]), "cell_status")
    reference = {}
    for model_name, model in value["models"].items():
        require(model["status"] in ("available", "unavailable_deadline", "unavailable_convergence"), "model_status")
        if model["status"] != "available":
            require(set(model) == {"status"}, "unavailable_payload")
            continue
        expected_fields = {"status", "metrics"} | ({"feature_dimensions"} if model_name != "training_prevalence" else set())
        require(set(model) == expected_fields and set(model["metrics"]) == set(HORIZONS), "model_schema")
        if "feature_dimensions" in model:
            require(integer(model["feature_dimensions"], 1), "dimensions")
        selected_by_group = {}
        for horizon, groups in model["metrics"].items():
            require(set(groups) == set(GROUPS), "intent_coverage")
            for group, metric in groups.items():
                validate_metric(metric, expected)
                support = tuple(metric[n] for n in COUNTS)
                require(reference.setdefault((horizon, group), support) == support, "paired_support")
                subset = (metric["selected"], metric["request_cores"])
                require(selected_by_group.setdefault(group, subset) == subset, "horizon_selection")
            require(groups["all"]["selected"] == expected["test_rows"]
                    and groups["all"]["request_cores"] == expected["test_cores"], "whole_cell_support")
            require(all(sum(groups[g][n] for g in GROUPS[1:]) == groups["all"][n]
                        for n in COUNTS if n != "request_cores"), "intent_partition")
    require(value["models"]["training_prevalence"]["status"] == "available", "prevalence_missing")
    require((value["status"] == "complete") == all(m["status"] == "available" for m in value["models"].values()),
            "cell_completion")


@contextmanager
def snapshot_lock(root):
    lock_path = root / ".launcher.lock"
    if not lock_path.exists():
        yield
        return
    require(not lock_path.is_symlink(), "lock_symlink")
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AnalysisError("text_matrix_active_writer") from None
        yield


def load_matrix(manifest_path, manifest_sha):
    """Validate hashes, pilot preservation and durable numeric terminal receipts."""
    manifest = decode(read(manifest_path, manifest_sha))
    require(manifest["schema_version"] == "text-challenger-extension-execution-v1"
            and manifest["pilot_outcomes_inspected"] is True
            and manifest["retrospective_exploratory"] is True
            and manifest["confirmation_opened"] is False, "execution_manifest")
    inventory_sha = manifest["inventory_sha256"]
    inventory = decode(read(manifest["inventory_path"], inventory_sha))
    expected = validate_inventory(inventory)
    require(all(manifest[k] == inventory[k] for k in ("source_pins", "input_pins", "runtime")), "frozen_inventory")
    root = Path(manifest["output_root"])
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "output_root")
    read(root / "launch.py", manifest["launcher_source_sha256"])
    require(read(root / "manifest.json") == read(manifest_path), "copied_manifest")
    planned = list(manifest["completed_pilot_cells"]) + list(manifest["cells"])
    require(len(planned) == 40 and len({key(c) for c in planned}) == 40
            and {key(c) for c in planned} == set(expected), "planned_coverage")
    pilot_keys = {(p, family, 0) for p in PLACEMENTS for family in (None, "structural_sham")}
    require({key(c) for c in manifest["completed_pilot_cells"]} == pilot_keys, "pilot_selection")
    records, artifacts = {}, []
    with snapshot_lock(root):
        for pilot in manifest["completed_pilot_cells"]:
            raw = read(pilot["result_path"], pilot["result_sha256"])
            value = decode(raw)
            validate_cell(value, expected[key(pilot)], inventory_sha, inventory["eligible_rows"])
            require(value["status"] == "complete", "pilot_changed")
            records[key(pilot)] = {"state": "complete", "origin": "pilot", "result": value}
            artifacts.append({"origin": "pilot", "cell": list(key(pilot)), "sha256": sha(raw)})
        for index, cell in enumerate(manifest["cells"]):
            require(cell["index"] == index and all(cell[n] == expected[key(cell)][n] for n in expected[key(cell)]),
                    "extension_selection")
            require(Path(cell["output_name"]).name == cell["output_name"], "output_name")
            validated_path = root / f"cell-{index:03d}-validated.json"
            exit_path = root / f"cell-{index:03d}-exit.json"
            result_path = root / cell["output_name"]
            record = {"state": "missing", "origin": "extension", "result": None}
            if not validated_path.exists():
                if exit_path.exists() or result_path.exists():
                    record["state"] = "interrupted"
                records[key(cell)] = record
                continue
            terminal_raw = read(validated_path)
            terminal = decode(terminal_raw)
            exit_raw = read(exit_path)
            exit_value = decode(exit_raw)
            require(terminal["schema_version"] == "text-challenger-extension-cell-exit-v1"
                    and terminal["manifest_sha256"] == manifest_sha and terminal["index"] == index
                    and key(terminal) == key(cell) and terminal["status"] in ("complete", "incomplete", "execution_failed")
                    and terminal["returncode"] == exit_value["returncode"], "terminal_binding")
            require({k: v for k, v in terminal.items() if k not in ("status", "result_sha256")}
                    == {k: v for k, v in exit_value.items() if k != "result_sha256"}, "exit_binding")
            require((terminal["status"] != "execution_failed") == (terminal["returncode"] == 0), "terminal_status")
            record["state"] = terminal["status"]
            if terminal["result_sha256"] is not None:
                raw = read(result_path, terminal["result_sha256"])
                value = decode(raw)
                validate_cell(value, expected[key(cell)], inventory_sha, inventory["eligible_rows"])
                if terminal["status"] != "execution_failed":
                    require(value["status"] == terminal["status"], "result_status")
                    record["result"] = value
                artifacts.append({"origin": "extension", "cell": list(key(cell)), "sha256": sha(raw)})
            else:
                require(terminal["status"] == "execution_failed", "missing_result")
            artifacts.append({"origin": "terminal_receipts", "cell": list(key(cell)),
                              "validated_sha256": sha(terminal_raw), "exit_sha256": sha(exit_raw)})
            records[key(cell)] = record
        finished = None
        if (root / "finished.json").exists():
            finished_raw = read(root / "finished.json")
            finished = decode(finished_raw)
            require(finished["manifest_sha256"] == manifest_sha, "finished_binding")
            counts = Counter(r["state"] for r in records.values() if r["origin"] == "extension")
            require(finished["completed"] == counts["complete"] and finished["incomplete"] == counts["incomplete"]
                    and finished["execution_failed"] == counts["execution_failed"], "finished_counts")
            artifacts.append({"origin": "finished", "sha256": sha(finished_raw)})
    return inventory, records, {"execution_manifest_sha256": manifest_sha, "inventory_sha256": inventory_sha,
        "freeze_commit": manifest["freeze_commit"], "launcher_source_sha256": manifest["launcher_source_sha256"],
        "pilot_outcomes_inspected": True, "artifacts": artifacts,
        "execution_finished_status": finished["status"] if finished is not None else None}


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_matrix(inventory, records):
    expected = validate_inventory(inventory)
    require(set(records) == set(expected), "record_coverage")
    inventory_sha = sha(canonical(inventory))
    for cell_key, record in records.items():
        require(record["state"] in STATES and record["origin"] in ("pilot", "extension"), "record_state")
        if record["state"] in ("complete", "incomplete"):
            validate_cell(record["result"], expected[cell_key], inventory_sha, inventory["eligible_rows"])
            require(record["result"]["status"] == record["state"], "record_status")
        else:
            require(record["result"] is None, "failed_result_used")
    summaries, comparisons = [], []
    for placement in PLACEMENTS:
        for family in FAMILIES:
            cells = [records[(placement, family, fold)] for fold in range(5)]
            for horizon in HORIZONS:
                for group in GROUPS:
                    by_model = {}
                    for model_name in MODELS:
                        folds = []
                        for fold, record in enumerate(cells):
                            require(record["state"] in STATES, "record_state")
                            model = record["result"]["models"][model_name] if record["result"] is not None else None
                            entry = {"fold": fold, "origin": record["origin"], "cell_state": record["state"],
                                     "model_status": model["status"] if model is not None else "unavailable_cell",
                                     "metrics": model["metrics"][horizon][group] if model is not None and model["status"] == "available" else None}
                            folds.append(entry)
                        available = [r["metrics"] for r in folds if r["metrics"] is not None]
                        known = sum(r["known"] for r in available)
                        summary = {"placement": placement, "held_family": family, "horizon": int(horizon),
                            "intent": group, "model": model_name, "expected_folds": 5,
                            "available_folds": len(available), "all_folds_available": len(available) == 5,
                            "known_rows": known, "unknown_rows": sum(r["unknown"] for r in available),
                            "observed_selected_rows": sum(r["selected"] for r in available),
                            "positive_rows": sum(r["positive"] for r in available),
                            "observed_request_cores_across_disjoint_folds": sum(r["request_cores"] for r in available),
                            "right_censored_rows": sum(r["right_censored"] for r in available),
                            "capped_negative_rows": sum(r["capped_negative"] for r in available),
                            "planned_all_intent_rows": sum(expected[(placement, family, f)]["test_rows"] for f in range(5)),
                            "scope": "full_known_row_cohort" if len(available) == 5 else "available_folds_only",
                            "weighted_log_loss": sum(r["known"] * r["log_loss"] for r in available if r["known"]) / known if known else None,
                            "weighted_brier": sum(r["known"] * r["brier"] for r in available if r["known"]) / known if known else None,
                            "ranking_summary_kind": "unweighted_mean_of_available_fold_metrics_not_pooled",
                            "mean_available_fold_roc_auc": _mean([r["roc_auc"] for r in available if r["roc_auc"] is not None]),
                            "mean_available_fold_average_precision": _mean([r["average_precision"] for r in available if r["average_precision"] is not None]),
                            "ranking_available_folds": sum(r["roc_auc"] is not None for r in available), "folds": folds}
                        summaries.append(summary)
                        by_model[model_name] = folds
                    for left, right in COMPARISONS:
                        paired = []
                        for a, b in zip(by_model[left], by_model[right], strict=True):
                            if a["metrics"] is None or b["metrics"] is None:
                                continue
                            am, bm = a["metrics"], b["metrics"]
                            require(all(am[n] == bm[n] for n in COUNTS), "comparison_support")
                            paired.append((a["fold"], am, bm))
                        known = sum(a["known"] for _, a, _ in paired)
                        comparisons.append({"placement": placement, "held_family": family, "horizon": int(horizon),
                            "intent": group, "left_model": left, "right_model": right,
                            "direction": "left_minus_right_negative_probability_loss_favors_left",
                            "paired_folds": [fold for fold, _, _ in paired], "all_folds_paired": len(paired) == 5,
                            "paired_known_rows": known,
                            "weighted_log_loss_difference": sum(a["known"] * (a["log_loss"] - b["log_loss"]) for _, a, b in paired if a["known"]) / known if known else None,
                            "weighted_brier_difference": sum(a["known"] * (a["brier"] - b["brier"]) for _, a, b in paired if a["known"]) / known if known else None})
    states = Counter(r["state"] for r in records.values())
    model_states = Counter(model["status"] for record in records.values() if record["result"] is not None
                           for model in record["result"]["models"].values())
    model_states["unavailable_cell"] = len(MODELS) * sum(r["result"] is None for r in records.values())
    family_coverage = []
    for placement in PLACEMENTS:
        for family in FAMILIES:
            current = [records[(placement, family, f)] for f in range(5)]
            family_coverage.append({"placement": placement, "held_family": family, "expected_folds": 5,
                "cells_by_state": {s: sum(r["state"] == s for r in current) for s in STATES},
                "planned_test_rows": sum(expected[(placement, family, f)]["test_rows"] for f in range(5))})
    return {"schema_version": "text-challenger-matrix-analysis-v1",
        "status": "complete" if states["complete"] == 40 else "incomplete",
        "coverage": {"expected_cells": 40, "cells_by_state": {s: states[s] for s in STATES},
                     "expected_model_cells": 200, "model_cells_by_status": dict(model_states),
                     "pilot_cells": sum(r["origin"] == "pilot" for r in records.values()),
                     "extension_cells": sum(r["origin"] == "extension" for r in records.values())},
        "coverage_by_placement_family": family_coverage,
        "population": {"eligible_rows": inventory["eligible_rows"], "request_cores": inventory["request_cores"]},
        "summaries": summaries, "paired_comparisons": comparisons,
        "claim_boundaries": {"retrospective_exploratory": True, "pilot_outcomes_inspected": True,
            "classifier_proxy_labels": True, "fixed_prefix_hash_is_lossy": True,
            "learned_prefix_or_pretrained_semantic_baseline": False,
            "ordinary_and_family_stress_samples_overlap": True,
            "pooled_ranking_reconstructed": False, "row_independent_inference": False,
            "confidence_intervals_available": False, "human_accuracy": False,
            "raw_prompts_generations_or_item_predictions_read": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inventory, records, provenance = load_matrix(args.execution_manifest, args.manifest_sha256)
        result = summarize_matrix(inventory, records)
        result["provenance"] = provenance
        result["analysis_source_sha256"] = sha(Path(__file__).read_bytes())
        raw = canonical(result)
        require(not any(p.is_symlink() for p in (args.output, *args.output.parents)), "output_symlink")
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"status": result["status"], "sha256": sha(raw), "coverage": result["coverage"]}))
        return 0
    except (AnalysisError, ValueError, OSError, KeyError, TypeError):
        print('{"status":"text_matrix_analysis_rejected"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
