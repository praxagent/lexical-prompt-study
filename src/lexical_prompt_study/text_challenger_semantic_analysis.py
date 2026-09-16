"""Aggregate-only paired readout of 200 lexical and 80 prompt-semantic fits.

Never reads prompt packets, the cache index, embedding receipts, model weights,
per-item outcomes, or fitted models. The numeric fit inventory and standalone
encoder config carry provenance without exposing encoded study material.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from . import text_challenger_analysis as lexical

LEXICAL_ANALYSIS_SHA = "335d4bd8dc4c2afc7377cd320d0da05c16e850ff6246cebdedbb0cceb86d239b"
SEMANTIC_LAUNCHER_SHA = "a56729ba7b5535516e85266a61d58b9d0ba48e6901d0a9a855360bc064bd9cec"
SEMANTIC_SOURCES = {
    "src/lexical_prompt_study/text_challenger_semantic.py": "d9240be3d5a55306f570e783850337378a0bc51a0d29d554e73286d4c4de87c4",
    "tests/test_text_challenger_semantic.py": "a16f210e0bd22347646949fb02f5365a02c5d9b47792f85d33c387df29f76869",
    "plans/text_challenger_semantic_20260915.md": "73edfe8bc5bd2d5d635578b291871719373a8f53e6cf1dbf076354864ea3f26f",
}
SEMANTIC_MODELS = ("prompt_prefix_hash_semantic", "prompt_prefix_hash_semantic_and_jlens")
MODELS = lexical.MODELS + SEMANTIC_MODELS
COMPARISONS = ((SEMANTIC_MODELS[0], "prompt_plus_prefix_hash"),
               (SEMANTIC_MODELS[1], SEMANTIC_MODELS[0]),
               (SEMANTIC_MODELS[1], "prompt_plus_prefix_hash_and_jlens"))
STATES = ("complete", "incomplete", "unavailable", "execution_failed", "interrupted", "not_started")
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt"}
ENCODING = {
    "schema_version": "prompt-semantic-encoding-v1", "model_id": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "dimensions": 384,
    "content_tokens_per_chunk": 510, "overlap_tokens": 0, "special_tokens_per_chunk": 2,
    "pooling": "cls", "chunk_l2_normalize": True, "pool_weight": "content_token_count",
    "document_l2_normalize": True, "retrieval_instruction": None, "full_prompt_coverage": True,
    "dtype": "float32", "device": "cpu", "attention_implementation": "eager",
    "deterministic_algorithms": True, "eval_mode": True, "gradient_enabled": False,
    "batch_size": 4, "threads": 4, "benchmark_prompt_count": 32, "seed": 20260915,
}
SEMANTIC_CELL_FIELDS = (lexical.CELL_FIELDS - {"eligible_rows"}) | {
    "semantic_response_prefix_available", "preparation_sha256", "cache_sha256"}
UNAVAILABLE_CELL_FIELDS = SEMANTIC_CELL_FIELDS - {"elapsed_seconds", "vocabulary_sizes"}


def require(value, code):
    lexical.require(value, "semantic_" + code)


def is_digest(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def launcher_canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def matrix():
    return [{"cell_index": i, "placement": p, "held_family": family, "fold": fold}
            for i, (p, family, fold) in enumerate((p, h, f) for p in lexical.PLACEMENTS
                for h in lexical.FAMILIES for f in range(5))]


def validate_numeric_lineage(header, fit_inventory, encoder_config, original_inventory):
    """Prove identical population/folds/target and frozen encoder provenance."""
    require(fit_inventory["schema_version"] == "prompt-semantic-fit-inventory-v1"
            and fit_inventory["actual_fitting_performed"] is False, "fit_inventory")
    inputs = header["inputs"]
    require(fit_inventory["preparation_sha256"] == inputs["preparation"]["sha256"]
            and fit_inventory["cache_sha256"] == inputs["cache"]["sha256"], "input_lineage")
    prepared = fit_inventory["preparation"]
    require(lexical.sha(lexical.canonical(prepared)) == inputs["preparation"]["sha256"], "preparation_hash")
    require(prepared["schema_version"] == "prompt-semantic-preparation-v1"
            and prepared["encoding"] == ENCODING and prepared["models"] == list(SEMANTIC_MODELS)
            and prepared["semantic_response_prefix_available"] is False
            and prepared["new_fits_in_full_matrix"] == 80 and is_digest(prepared["packet_sha256"])
            and lexical.integer(prepared["unique_prompts"], 1), "preparation_scope")
    base = prepared["base_inventory"]
    require(set(base) == set(original_inventory), "base_inventory_schema")
    require(all(base[k] == original_inventory[k] for k in original_inventory if k != "source_pins"),
            "same_population_folds_runtime")
    require(all(base["source_pins"].get(k) == v for k, v in original_inventory["source_pins"].items()),
            "base_source_pins")
    require(fit_inventory["fit_runtime"] == original_inventory["runtime"], "fit_runtime")
    require(all(prepared["source_pins"].get(k) == v for k, v in SEMANTIC_SOURCES.items())
            and header["source_pins"] == SEMANTIC_SOURCES, "semantic_source_pins")
    require(base["source_pins"].get("src/lexical_prompt_study/text_challenger_semantic.py")
            == SEMANTIC_SOURCES["src/lexical_prompt_study/text_challenger_semantic.py"], "semantic_helper_source")
    require(encoder_config["schema_version"] == "prompt-semantic-encoder-v1"
            and encoder_config["encoding"] == ENCODING
            and encoder_config["source_pins"] == prepared["source_pins"]
            and set(encoder_config["model_files_sha256"]) == MODEL_FILES
            and all(is_digest(v) for v in encoder_config["model_files_sha256"].values()), "encoder_config")
    require(Path(encoder_config["model_path"]).is_absolute()
            and Path(encoder_config["model_path"]).name == ENCODING["revision"], "model_revision")
    runtime = encoder_config["runtime"]
    require(set(runtime) == {"python", "executable", "versions"}
            and set(runtime["versions"]) == {"torch", "transformers", "tokenizers", "numpy"}
            and all(type(v) is str and bool(v) for v in runtime["versions"].values()), "encoder_runtime")
    return {"encoding": ENCODING, "model_files_sha256": encoder_config["model_files_sha256"],
            "encoder_runtime": runtime, "source_pins": prepared["source_pins"],
            "packet_sha256": prepared["packet_sha256"], "cache_sha256": fit_inventory["cache_sha256"],
            "encoder_config_sha256": fit_inventory["cache_encoder_config_sha256"],
            "preparation_sha256": fit_inventory["preparation_sha256"], "unique_prompts": prepared["unique_prompts"]}


def validate_semantic_cell(value, expected, inputs, original=None):
    unavailable = value.get("status") == "unavailable_training_classes_or_test_support"
    require(set(value) == (UNAVAILABLE_CELL_FIELDS if unavailable else SEMANTIC_CELL_FIELDS), "cell_schema")
    require(value["schema_version"] == "prompt-semantic-cell-v1" and lexical.key(value) == lexical.key(expected)
            and type(value["fold"]) is int and value["target_horizon"] == 1024
            and value["training_rows"] == expected["training_rows"] and value["test_rows"] == expected["test_rows"]
            and value["confirmation_opened"] is False and value["semantic_response_prefix_available"] is False
            and value["interpretation"] == "retrospective_classifier_proxy"
            and all(value[k + "_sha256"] == inputs[k]["sha256"] for k in ("preparation", "cache", "inventory")),
            "cell_binding")
    if unavailable:
        require(value["models"] == {}, "unavailable_payload")
        return
    require(value["status"] in ("complete", "incomplete") and set(value["models"]) == set(SEMANTIC_MODELS)
            and lexical.finite(value["elapsed_seconds"])
            and type(value["vocabulary_sizes"]) is list and len(value["vocabulary_sizes"]) == 2
            and all(lexical.integer(n, 1) for n in value["vocabulary_sizes"]), "cell_status")
    if original is not None:
        require(value["vocabulary_sizes"] == original["vocabulary_sizes"], "lexical_vocabulary_support")
    reference = {}
    if original is not None:
        for horizon, groups in original["models"]["training_prevalence"]["metrics"].items():
            for group, metric in groups.items():
                reference[horizon, group] = tuple(metric[n] for n in lexical.COUNTS)
    for name, model in value["models"].items():
        require(model["status"] in ("available", "unavailable_deadline", "unavailable_convergence"), "model_status")
        if model["status"] != "available":
            require(set(model) == {"status"}, "unavailable_model")
            continue
        require(set(model) == {"status", "feature_dimensions", "metrics"}
                and lexical.integer(model["feature_dimensions"], 1)
                and set(model["metrics"]) == set(lexical.HORIZONS), "model_schema")
        dimensions = sum(value["vocabulary_sizes"]) + 256 + 6 + 384 + (31 if name == SEMANTIC_MODELS[1] else 0)
        require(model["feature_dimensions"] == dimensions, "semantic_feature_dimensions")
        selected = {}
        for horizon, groups in model["metrics"].items():
            require(set(groups) == set(lexical.GROUPS), "intent_coverage")
            for group, metric in groups.items():
                lexical.validate_metric(metric, expected)
                counts = tuple(metric[n] for n in lexical.COUNTS)
                require(reference.setdefault((horizon, group), counts) == counts, "cross_model_support")
                cohort = metric["selected"], metric["request_cores"]
                require(selected.setdefault(group, cohort) == cohort, "horizon_selection")
            require(groups["all"]["selected"] == expected["test_rows"]
                    and groups["all"]["request_cores"] == expected["test_cores"], "test_support")
            require(all(sum(groups[g][n] for g in lexical.GROUPS[1:]) == groups["all"][n]
                        for n in lexical.COUNTS if n != "request_cores"), "intent_partition")
    require((value["status"] == "complete") == all(m["status"] == "available" for m in value["models"].values()),
            "complete_status")


def expected_command(header, root, cell):
    args = [header["fit_python"], "-m", "lexical_prompt_study.text_challenger_semantic", "run"]
    for name in ("preparation", "cache", "inventory"):
        args += ["--" + name, header["inputs"][name]["path"], "--" + name + "-sha256",
                 header["inputs"][name]["sha256"]]
    args += ["--placement", cell["placement"], "--fold", str(cell["fold"]), "--max-seconds",
             str(header["fit_start_deadline_seconds"]), "--output", str(root / f"cell-{cell['cell_index']:02d}.json")]
    if cell["held_family"] is not None:
        args += ["--held-family", cell["held_family"]]
    return args


@contextmanager
def snapshot_lock(root):
    path = root / ".lock"
    if not path.exists():
        yield
        return
    require(not path.is_symlink(), "lock_symlink")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise lexical.AnalysisError("text_matrix_semantic_active_writer") from None
        yield


def load_semantic_run(launch_path, launch_sha, encoder_config_path, original_inventory, original_records):
    root = Path(launch_path).absolute().parent
    header = lexical.decode(lexical.read(launch_path, launch_sha))
    require(header["schema_version"] == "prompt-semantic-matrix-launch-v1"
            and header["launcher_sha256"] == SEMANTIC_LAUNCHER_SHA and header["source_pins"] == SEMANTIC_SOURCES
            and header["cells"] == matrix() and header["maximum_new_fits"] == 80
            and header["automatic_retry_attempted_cells"] is False, "launch_binding")
    require(lexical.integer(header["fit_start_deadline_seconds"], 1)
            and header["fit_start_deadline_seconds"] <= 1800
            and header["fit_start_deadline_seconds"] < header["cell_hard_timeout_seconds"] <= 3600
            and 1 <= header["invocation_wall_limit_seconds"] <= 86400, "launch_bounds")
    inputs = header["inputs"]
    require(set(inputs) == {"preparation", "cache", "inventory"}
            and all(set(v) == {"path", "sha256"} and Path(v["path"]).is_absolute() and is_digest(v["sha256"])
                    for v in inputs.values()), "input_binding")
    # Deliberately open the numeric inventory and standalone config only.
    inventory = lexical.decode(lexical.read(inputs["inventory"]["path"], inputs["inventory"]["sha256"]))
    config = lexical.decode(lexical.read(encoder_config_path, inventory["cache_encoder_config_sha256"]))
    provenance = validate_numeric_lineage(header, inventory, config, original_inventory)
    expected = lexical.validate_inventory(original_inventory)
    records, artifacts = {}, []
    with snapshot_lock(root):
        for cell in matrix():
            prefix = root / f"cell-{cell['cell_index']:02d}"
            attempt_path = prefix.with_suffix(".attempt.json")
            execution_path = prefix.with_suffix(".execution.json")
            result_path = prefix.with_suffix(".json")
            record = {"state": "not_started", "result": None}
            if not attempt_path.exists():
                require(not execution_path.exists() and not result_path.exists(), "unbound_terminal")
                records[lexical.key(cell)] = record
                continue
            attempt_raw = lexical.read(attempt_path)
            attempt = lexical.decode(attempt_raw)
            require(attempt == {"schema_version": "prompt-semantic-matrix-attempt-v1", "launch_sha256": launch_sha,
                "cell": cell, "command_sha256": lexical.sha(launcher_canonical(expected_command(header, root, cell)))},
                "attempt_binding")
            artifact = {"cell": list(lexical.key(cell)), "attempt_sha256": lexical.sha(attempt_raw)}
            if not execution_path.exists():
                record["state"] = "interrupted"
                records[lexical.key(cell)] = record
                artifacts.append(artifact)
                continue
            execution_raw = lexical.read(execution_path)
            execution = lexical.decode(execution_raw)
            require(set(execution) == {"schema_version", "returncode", "timed_out", "elapsed_seconds",
                    "stdout_sha256", "stderr_sha256", "attempt_sha256", "result_sha256"}
                    and execution["schema_version"] == "prompt-semantic-matrix-execution-v1"
                    and execution["attempt_sha256"] == lexical.sha(attempt_raw)
                    and type(execution["returncode"]) is int and type(execution["timed_out"]) is bool
                    and lexical.finite(execution["elapsed_seconds"])
                    and is_digest(execution["stdout_sha256"]) and is_digest(execution["stderr_sha256"]), "execution_binding")
            artifact["execution_sha256"] = lexical.sha(execution_raw)
            record["state"] = "execution_failed"
            if execution["result_sha256"] is None:
                require(not result_path.exists(), "unbound_result")
            else:
                require(is_digest(execution["result_sha256"]), "result_digest")
                raw = lexical.read(result_path, execution["result_sha256"])
                result = lexical.decode(raw)
                validate_semantic_cell(result, expected[lexical.key(cell)], inputs,
                                       original_records[lexical.key(cell)]["result"])
                artifact["result_sha256"] = lexical.sha(raw)
                if execution["returncode"] == 0 and not execution["timed_out"]:
                    record["state"] = "unavailable" if result["status"].startswith("unavailable_") else result["status"]
                    record["result"] = result
            artifacts.append(artifact)
            records[lexical.key(cell)] = record
    provenance.update(launch_sha256=launch_sha, launcher_sha256=header["launcher_sha256"],
                      inventory_sha256=inputs["inventory"]["sha256"], artifacts=artifacts)
    return records, provenance, inputs


def model_at(record, model):
    if record["result"] is None:
        return None
    return record["result"]["models"].get(model)


def summarize_semantic(original_inventory, original_records, semantic_records, inputs):
    expected = lexical.validate_inventory(original_inventory)
    require(set(semantic_records) == set(expected), "semantic_coverage")
    old = lexical.summarize_matrix(original_inventory, original_records)
    for cell_key, record in semantic_records.items():
        require(record["state"] in STATES, "record_state")
        if record["state"] in ("complete", "incomplete", "unavailable"):
            validate_semantic_cell(record["result"], expected[cell_key], inputs, original_records[cell_key]["result"])
            wanted = "unavailable" if record["result"]["status"].startswith("unavailable_") else record["result"]["status"]
            require(record["state"] == wanted, "record_status")
        else:
            require(record["result"] is None, "failed_result_used")
    summaries, comparisons = [], []
    for placement in lexical.PLACEMENTS:
        for family in lexical.FAMILIES:
            for horizon in lexical.HORIZONS:
                for group in lexical.GROUPS:
                    folds_by_model = {}
                    for name in MODELS:
                        source = semantic_records if name in SEMANTIC_MODELS else original_records
                        folds = []
                        for fold in range(5):
                            record = source[placement, family, fold]
                            model = model_at(record, name)
                            metric = model["metrics"][horizon][group] if model is not None and model["status"] == "available" else None
                            folds.append({"fold": fold, "cell_state": record["state"],
                                "model_status": model["status"] if model is not None else "unavailable_cell", "metrics": metric})
                        folds_by_model[name] = folds
                        if name not in SEMANTIC_MODELS:
                            continue
                        available = [r["metrics"] for r in folds if r["metrics"] is not None]
                        known = sum(r["known"] for r in available)
                        summaries.append({"placement": placement, "held_family": family, "horizon": int(horizon),
                            "intent": group, "model": name, "expected_folds": 5, "available_folds": len(available),
                            "all_folds_available": len(available) == 5, "known_rows": known,
                            "unknown_rows": sum(r["unknown"] for r in available),
                            "observed_selected_rows": sum(r["selected"] for r in available),
                            "positive_rows": sum(r["positive"] for r in available),
                            "right_censored_rows": sum(r["right_censored"] for r in available),
                            "capped_negative_rows": sum(r["capped_negative"] for r in available),
                            "observed_request_cores_across_disjoint_folds": sum(r["request_cores"] for r in available),
                            "scope": "full_known_row_cohort" if len(available) == 5 else "available_folds_only",
                            "weighted_log_loss": sum(r["known"] * r["log_loss"] for r in available if r["known"]) / known if known else None,
                            "weighted_brier": sum(r["known"] * r["brier"] for r in available if r["known"]) / known if known else None,
                            "ranking_summary_kind": "unweighted_mean_of_available_fold_metrics_not_pooled",
                            "mean_available_fold_roc_auc": lexical._mean([r["roc_auc"] for r in available if r["roc_auc"] is not None]),
                            "mean_available_fold_average_precision": lexical._mean([r["average_precision"] for r in available if r["average_precision"] is not None]),
                            "ranking_available_folds": sum(r["roc_auc"] is not None for r in available), "folds": folds})
                    for left, right in COMPARISONS:
                        pairs = []
                        for a, b in zip(folds_by_model[left], folds_by_model[right], strict=True):
                            if a["metrics"] is None or b["metrics"] is None:
                                continue
                            am, bm = a["metrics"], b["metrics"]
                            require(all(am[n] == bm[n] for n in lexical.COUNTS), "paired_denominator")
                            pairs.append((a["fold"], am, bm))
                        known = sum(a["known"] for _, a, _ in pairs)
                        comparisons.append({"placement": placement, "held_family": family, "horizon": int(horizon),
                            "intent": group, "left_model": left, "right_model": right,
                            "direction": "left_minus_right_negative_probability_loss_favors_left",
                            "paired_folds": [f for f, _, _ in pairs], "all_folds_paired": len(pairs) == 5,
                            "paired_known_rows": known,
                            "weighted_log_loss_difference": sum(a["known"] * (a["log_loss"] - b["log_loss"]) for _, a, b in pairs if a["known"]) / known if known else None,
                            "weighted_brier_difference": sum(a["known"] * (a["brier"] - b["brier"]) for _, a, b in pairs if a["known"]) / known if known else None})
    cell_states = Counter(r["state"] for r in semantic_records.values())
    model_states = Counter()
    for record in semantic_records.values():
        for name in SEMANTIC_MODELS:
            model = model_at(record, name)
            model_states[model["status"] if model is not None else "unavailable_cell"] += 1
    return {"schema_version": "prompt-semantic-matrix-analysis-v1",
        "status": "complete" if old["status"] == "complete" and cell_states["complete"] == 40 else "incomplete",
        "models": list(MODELS), "training_target_horizon": 1024,
        "coverage": {"expected_original_model_cells": 200, "expected_semantic_model_cells": 80,
            "expected_total_model_cells": 280, "original": old["coverage"],
            "semantic_cells_by_state": {s: cell_states[s] for s in STATES},
            "semantic_model_cells_by_status": dict(model_states)},
        "semantic_coverage_by_placement_family": [{"placement": p, "held_family": family,
            "expected_folds": 5, "cells_by_state": {
                state: sum(semantic_records[p, family, f]["state"] == state for f in range(5)) for state in STATES}}
            for p in lexical.PLACEMENTS for family in lexical.FAMILIES],
        "population": old["population"], "summaries": old["summaries"] + summaries,
        "semantic_paired_comparisons": comparisons, "original_paired_comparisons": old["paired_comparisons"],
        "claim_boundaries": {**{k: v for k, v in old["claim_boundaries"].items()
                                if k != "learned_prefix_or_pretrained_semantic_baseline"},
            "frozen_prompt_semantic_representation_included": True,
            "learned_response_prefix_vocabulary_included": False,
            "semantic_response_prefix_available": False, "fixed_chunk_pooling_loses_cross_chunk_relations": True,
            "semantic_prompt_cache_or_embedding_vectors_read": False,
            "nonfinal_horizon_metrics_are_descriptive": True,
            "full_semantic_baseline_universality_claim": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical-execution-manifest", type=Path, required=True)
    parser.add_argument("--lexical-manifest-sha256", required=True)
    parser.add_argument("--semantic-launch", type=Path, required=True)
    parser.add_argument("--semantic-launch-sha256", required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        lexical.read(lexical.__file__, LEXICAL_ANALYSIS_SHA)
        inventory, old_records, old_provenance = lexical.load_matrix(args.lexical_execution_manifest,
                                                                     args.lexical_manifest_sha256)
        new_records, provenance, inputs = load_semantic_run(args.semantic_launch, args.semantic_launch_sha256,
                                                          args.encoder_config, inventory, old_records)
        result = summarize_semantic(inventory, old_records, new_records, inputs)
        result["provenance"] = {"lexical": old_provenance, "semantic": provenance,
            "analysis_source_sha256": lexical.sha(Path(__file__).read_bytes()),
            "lexical_analysis_source_sha256": LEXICAL_ANALYSIS_SHA}
        raw = lexical.canonical(result)
        require(not any(p.is_symlink() for p in (args.output, *args.output.parents)), "output_symlink")
        with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(lexical.canonical({"status": result["status"], "sha256": lexical.sha(raw),
                                 "coverage": result["coverage"]}).decode().strip())
        return 0
    except (lexical.AnalysisError, ValueError, OSError, KeyError, TypeError):
        print('{"status":"semantic_matrix_analysis_rejected"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
