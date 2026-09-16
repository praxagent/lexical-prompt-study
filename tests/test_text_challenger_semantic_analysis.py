from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from lexical_prompt_study import text_challenger_semantic_analysis as analysis
from test_text_challenger_analysis import fixtures as lexical_fixtures


def fixtures(tmp_path):
    original_inventory, old_records = lexical_fixtures()
    root = tmp_path / "semantic-μ"
    root.mkdir()
    source_pins = dict(analysis.SEMANTIC_SOURCES)
    prepared_base = deepcopy(original_inventory)
    prepared_base["source_pins"].update(source_pins)
    prepared = {"schema_version": "prompt-semantic-preparation-v1", "encoding": analysis.ENCODING,
        "models": list(analysis.SEMANTIC_MODELS), "packet_sha256": "d" * 64,
        "base_inventory": prepared_base, "source_pins": source_pins, "unique_prompts": 30,
        "semantic_response_prefix_available": False, "new_fits_in_full_matrix": 80}
    config = {"schema_version": "prompt-semantic-encoder-v1", "encoding": analysis.ENCODING,
        "source_pins": source_pins, "model_path": "/invented/" + analysis.ENCODING["revision"],
        "model_files_sha256": {name: "e" * 64 for name in analysis.MODEL_FILES},
        "runtime": {"python": "synthetic", "executable": "/invented/python", "versions": {
            name: "synthetic" for name in ("torch", "transformers", "tokenizers", "numpy")}}}
    config_path = root / "encoder-config.json"
    config_path.write_bytes(analysis.lexical.canonical(config))
    inputs = {"preparation": {"path": str(root / "must-not-open-preparation.json"),
                              "sha256": analysis.lexical.sha(analysis.lexical.canonical(prepared))},
              "cache": {"path": str(root / "must-not-open-cache.json"), "sha256": "f" * 64}}
    fit_inventory = {"schema_version": "prompt-semantic-fit-inventory-v1", "preparation": prepared,
        "preparation_sha256": inputs["preparation"]["sha256"], "cache_sha256": inputs["cache"]["sha256"],
        "cache_encoder_config_sha256": analysis.lexical.sha(config_path.read_bytes()),
        "fit_runtime": original_inventory["runtime"], "actual_fitting_performed": False}
    inventory_path = root / "fit-inventory.json"
    inventory_path.write_bytes(analysis.lexical.canonical(fit_inventory))
    inputs["inventory"] = {"path": str(inventory_path), "sha256": analysis.lexical.sha(inventory_path.read_bytes())}
    header = {"schema_version": "prompt-semantic-matrix-launch-v1",
        "launcher_sha256": analysis.SEMANTIC_LAUNCHER_SHA, "source_pins": source_pins,
        "repo": "/invented/repo", "fit_python": "/invented/python", "inputs": inputs,
        "cells": analysis.matrix(), "maximum_new_fits": 80,
        "fit_start_deadline_seconds": 900, "cell_hard_timeout_seconds": 1200,
        "invocation_wall_limit_seconds": 21600, "automatic_retry_attempted_cells": False}
    launch_path = root / "launch.json"
    launch_path.write_bytes(analysis.launcher_canonical(header))
    launch_sha = analysis.lexical.sha(launch_path.read_bytes())
    records = {}
    for key, old in old_records.items():
        value = deepcopy(old["result"])
        del value["eligible_rows"]
        value.update(schema_version="prompt-semantic-cell-v1", semantic_response_prefix_available=False,
            **{k + "_sha256": inputs[k]["sha256"] for k in inputs})
        value["models"] = {}
        for i, name in enumerate(analysis.SEMANTIC_MODELS):
            model = deepcopy(old["result"]["models"]["prompt_plus_prefix_hash"])
            model["feature_dimensions"] = sum(value["vocabulary_sizes"]) + 646 + 31 * i
            for groups in model["metrics"].values():
                for metric in groups.values():
                    metric["log_loss"] -= .02 + .01 * i
                    metric["brier"] -= .01 + .005 * i
            value["models"][name] = model
        records[key] = {"state": "complete", "result": value}
    return original_inventory, old_records, records, header, launch_path, launch_sha, config_path


def write_cell(root, header, launch_sha, cell, record, *, returncode=0, timed_out=False,
               attempted=True, executed=True, result=True):
    prefix = root / f"cell-{cell['cell_index']:02d}"
    if not attempted:
        return
    attempt = {"schema_version": "prompt-semantic-matrix-attempt-v1", "launch_sha256": launch_sha,
        "cell": cell, "command_sha256": analysis.lexical.sha(analysis.launcher_canonical(
            analysis.expected_command(header, root, cell)))}
    attempt_raw = analysis.launcher_canonical(attempt)
    prefix.with_suffix(".attempt.json").write_bytes(attempt_raw)
    if result:
        result_raw = analysis.lexical.canonical(record["result"])
        prefix.with_suffix(".json").write_bytes(result_raw)
    if not executed:
        return
    execution = {"schema_version": "prompt-semantic-matrix-execution-v1", "returncode": returncode,
        "timed_out": timed_out, "elapsed_seconds": .1, "stdout_sha256": "1" * 64,
        "stderr_sha256": "2" * 64, "attempt_sha256": analysis.lexical.sha(attempt_raw),
        "result_sha256": analysis.lexical.sha(result_raw) if result else None}
    prefix.with_suffix(".execution.json").write_bytes(analysis.launcher_canonical(execution))


def row(result, model):
    return next(r for r in result["summaries"] if r["model"] == model and r["placement"] == analysis.lexical.PLACEMENTS[0]
                and r["held_family"] is None and r["horizon"] == 1024 and r["intent"] == "all")


def pair(result, left, right):
    return next(r for r in result["semantic_paired_comparisons"] if r["left_model"] == left and r["right_model"] == right
                and r["placement"] == analysis.lexical.PLACEMENTS[0] and r["held_family"] is None
                and r["horizon"] == 1024 and r["intent"] == "all")


def test_seven_model_coverage_weighted_losses_and_three_paired_comparisons(tmp_path):
    inv, old, new, header, *_ = fixtures(tmp_path)
    result = analysis.summarize_semantic(inv, old, new, header["inputs"])
    assert result["status"] == "complete"
    assert result["coverage"]["expected_total_model_cells"] == 280
    assert result["coverage"]["original"]["model_cells_by_status"]["available"] == 200
    assert result["coverage"]["semantic_model_cells_by_status"]["available"] == 80
    assert len(result["summaries"]) == 1120
    assert row(result, analysis.SEMANTIC_MODELS[0])["weighted_log_loss"] == pytest.approx(55 / 150 - .02)
    assert row(result, analysis.SEMANTIC_MODELS[0])["weighted_brier"] == pytest.approx(55 / 300 - .01)
    assert row(result, analysis.SEMANTIC_MODELS[0])["ranking_available_folds"] == 4
    for (left, right), expected in zip(analysis.COMPARISONS, (-.02, -.01, -.03), strict=True):
        compared = pair(result, left, right)
        assert compared["weighted_log_loss_difference"] == pytest.approx(expected)
        assert compared["paired_known_rows"] == 60 and compared["all_folds_paired"] is True
    assert result["claim_boundaries"]["confidence_intervals_available"] is False
    assert "learned_prefix_or_pretrained_semantic_baseline" not in result["claim_boundaries"]


def test_pairing_excludes_missing_old_fold_and_unavailable_semantic_model(tmp_path):
    inv, old, new, header, *_ = fixtures(tmp_path)
    p = analysis.lexical.PLACEMENTS[0]
    old[p, None, 4] = {"state": "missing", "origin": "extension", "result": None}
    current = new[p, None, 3]
    current["state"] = current["result"]["status"] = "incomplete"
    current["result"]["models"][analysis.SEMANTIC_MODELS[0]] = {"status": "unavailable_convergence"}
    result = analysis.summarize_semantic(inv, old, new, header["inputs"])
    compared = pair(result, *analysis.COMPARISONS[0])
    assert compared["paired_folds"] == [0, 1, 2] and compared["paired_known_rows"] == 24
    assert compared["weighted_log_loss_difference"] == pytest.approx(-.02)
    assert compared["all_folds_paired"] is False and result["status"] == "incomplete"
    assert result["coverage"]["semantic_model_cells_by_status"]["unavailable_convergence"] == 1


def test_unavailable_training_cell_retains_no_fabricated_model_scores(tmp_path):
    inv, old, new, header, *_ = fixtures(tmp_path)
    record = new[analysis.lexical.PLACEMENTS[0], None, 4]
    record["state"] = "unavailable"
    record["result"]["status"] = "unavailable_training_classes_or_test_support"
    record["result"]["models"] = {}
    del record["result"]["vocabulary_sizes"]
    del record["result"]["elapsed_seconds"]
    result = analysis.summarize_semantic(inv, old, new, header["inputs"])
    assert result["coverage"]["semantic_cells_by_state"]["unavailable"] == 1
    assert result["coverage"]["semantic_model_cells_by_status"]["unavailable_cell"] == 2
    assert row(result, analysis.SEMANTIC_MODELS[0])["known_rows"] == 40
    assert row(result, analysis.SEMANTIC_MODELS[0])["all_folds_available"] is False


@pytest.mark.parametrize("mutation,code", [
    (lambda v: v.update(inventory_sha256="0" * 64), "cell_binding"),
    (lambda v: v.update(semantic_response_prefix_available=True), "cell_binding"),
    (lambda v: v.update(vocabulary_sizes=[11, 20]), "lexical_vocabulary_support"),
    (lambda v: v["models"][analysis.SEMANTIC_MODELS[0]].update(feature_dimensions=384), "semantic_feature_dimensions"),
    (lambda v: v["models"][analysis.SEMANTIC_MODELS[0]]["metrics"]["1024"]["all"].update(right_censored=1), "cross_model_support"),
])
def test_mismatched_feature_population_or_denominators_rejected(tmp_path, mutation, code):
    inv, old, new, header, *_ = fixtures(tmp_path)
    cell = inv["cells"][0]
    value = new[analysis.lexical.key(cell)]["result"]
    mutation(value)
    with pytest.raises(analysis.lexical.AnalysisError, match=code):
        analysis.validate_semantic_cell(value, cell, header["inputs"], old[analysis.lexical.key(cell)]["result"])


def test_loader_never_reads_packet_cache_or_embeddings(tmp_path):
    inv, old, new, header, launch, digest, config = fixtures(tmp_path)
    for cell in analysis.matrix():
        write_cell(launch.parent, header, digest, cell, new[analysis.lexical.key(cell)])
    loaded, provenance, inputs = analysis.load_semantic_run(launch, digest, config, inv, old)
    assert not Path(header["inputs"]["cache"]["path"]).exists()
    assert not Path(header["inputs"]["preparation"]["path"]).exists()
    assert len(loaded) == 40 and all(r["state"] == "complete" for r in loaded.values())
    assert set(provenance["model_files_sha256"]) == analysis.MODEL_FILES
    assert analysis.summarize_semantic(inv, old, loaded, inputs)["status"] == "complete"


def test_unattempted_interrupted_and_failed_are_preserved_without_scores(tmp_path):
    inv, old, new, header, launch, digest, config = fixtures(tmp_path)
    cells = analysis.matrix()
    write_cell(launch.parent, header, digest, cells[0], new[analysis.lexical.key(cells[0])], executed=False)
    write_cell(launch.parent, header, digest, cells[1], new[analysis.lexical.key(cells[1])], returncode=1, result=False)
    write_cell(launch.parent, header, digest, cells[2], new[analysis.lexical.key(cells[2])], timed_out=True)
    loaded, _, inputs = analysis.load_semantic_run(launch, digest, config, inv, old)
    assert loaded[analysis.lexical.key(cells[0])]["state"] == "interrupted"
    assert all(loaded[analysis.lexical.key(c)]["state"] == "execution_failed" for c in cells[1:3])
    assert all(r["result"] is None for r in loaded.values())
    result = analysis.summarize_semantic(inv, old, loaded, inputs)
    assert result["coverage"]["semantic_cells_by_state"]["not_started"] == 37
    assert result["coverage"]["semantic_model_cells_by_status"]["unavailable_cell"] == 80
    assert pair(result, *analysis.COMPARISONS[0])["weighted_log_loss_difference"] is None


@pytest.mark.parametrize("victim", ("attempt", "execution", "result", "config"))
def test_tampered_durable_bindings_or_config_rejected(tmp_path, victim):
    inv, old, new, header, launch, digest, config = fixtures(tmp_path)
    cell = analysis.matrix()[0]
    write_cell(launch.parent, header, digest, cell, new[analysis.lexical.key(cell)])
    if victim == "config":
        path = config
    else:
        path = launch.parent / ("cell-00.json" if victim == "result" else f"cell-00.{victim}.json")
    value = json.loads(path.read_text())
    value["unexpected"] = "synthetic mutation"
    path.write_bytes(analysis.lexical.canonical(value))
    with pytest.raises(analysis.lexical.AnalysisError):
        analysis.load_semantic_run(launch, digest, config, inv, old)


def test_different_original_population_cannot_be_paired(tmp_path):
    inv, old, _, _, launch, digest, config = fixtures(tmp_path)
    inv["eligible_rows"] += 1
    with pytest.raises(analysis.lexical.AnalysisError, match="same_population"):
        analysis.load_semantic_run(launch, digest, config, inv, old)


def test_active_semantic_writer_is_rejected(tmp_path):
    inv, old, _, _, launch, digest, config = fixtures(tmp_path)
    with (launch.parent / ".lock").open("wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(analysis.lexical.AnalysisError, match="active_writer"):
            analysis.load_semantic_run(launch, digest, config, inv, old)


def test_result_without_attempt_is_rejected(tmp_path):
    inv, old, new, _, launch, digest, config = fixtures(tmp_path)
    (launch.parent / "cell-00.json").write_bytes(analysis.lexical.canonical(next(iter(new.values()))["result"]))
    with pytest.raises(analysis.lexical.AnalysisError, match="unbound_terminal"):
        analysis.load_semantic_run(launch, digest, config, inv, old)


def test_horizon_intent_and_family_outputs_remain_separate(tmp_path):
    inv, old, new, header, *_ = fixtures(tmp_path)
    result = analysis.summarize_semantic(inv, old, new, header["inputs"])
    semantic = [r for r in result["summaries"] if r["model"] in analysis.SEMANTIC_MODELS]
    assert len(semantic) == 2 * 4 * 4 * 5 * 2
    assert {r["horizon"] for r in semantic} == {128, 256, 512, 1024}
    assert {r["intent"] for r in semantic} == set(analysis.lexical.GROUPS)
    assert {r["held_family"] for r in semantic} == set(analysis.lexical.FAMILIES)
