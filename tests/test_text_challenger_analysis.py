from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from lexical_prompt_study import text_challenger_analysis as analysis


def fixtures():
    cells = [{"placement": p, "held_family": family, "fold": fold, "training_rows": 100,
              "test_rows": 4 * (fold + 1), "training_cores": 8, "test_cores": 2,
              "both_training_classes": True}
             for p in analysis.PLACEMENTS for family in analysis.FAMILIES for fold in range(5)]
    inventory = {"schema_version": "text-challenger-inventory-v1", "actual_fitting_performed": False,
        "confirmation_opened": False, "cells": cells, "request_cores": 10, "eligible_rows": 120,
        "source_pins": {}, "input_pins": {}, "runtime": {}}
    digest = analysis.sha(analysis.canonical(inventory))
    records = {}
    for cell in cells:
        n = cell["fold"] + 1
        def metric(size, cores, positive):
            return {"selected": size, "known": size, "unknown": 0, "positive": positive,
                "request_cores": cores, "right_censored": 0, "capped_negative": 0,
                "log_loss": n / 10, "brier": n / 20,
                "roc_auc": .5 + n / 20 if 0 < positive < size else None,
                "average_precision": .4 + n / 20 if 0 < positive < size else None}
        groups = {g: metric(n, min(n, 2), 1) for g in analysis.GROUPS[1:]}
        groups["all"] = metric(4 * n, 2, 4)
        models = {model: {"status": "available", "metrics": {h: deepcopy(groups) for h in analysis.HORIZONS},
                          **({"feature_dimensions": 31} if model != "training_prevalence" else {})}
                  for model in analysis.MODELS}
        result = {"schema_version": "text-challenger-cell-v1", "placement": cell["placement"],
            "held_family": cell["held_family"], "fold": cell["fold"], "target_horizon": 1024,
            "training_rows": cell["training_rows"], "test_rows": cell["test_rows"], "models": models,
            "interpretation": "retrospective_classifier_proxy", "confirmation_opened": False,
            "vocabulary_sizes": [10, 20], "status": "complete", "elapsed_seconds": .1,
            "inventory_sha256": digest, "eligible_rows": 120}
        pilot = cell["fold"] == 0 and cell["held_family"] in (None, "structural_sham")
        records[analysis.key(cell)] = {"state": "complete", "origin": "pilot" if pilot else "extension",
                                       "result": result}
    return inventory, records


def summary(result, *, model="prompt_plus_prefix_hash", family=None, intent="all", horizon=1024):
    return next(r for r in result["summaries"] if r["placement"] == analysis.PLACEMENTS[0]
                and r["held_family"] == family and r["model"] == model
                and r["intent"] == intent and r["horizon"] == horizon)


def test_probability_metrics_weight_known_rows_and_ranking_stays_foldwise():
    inventory, records = fixtures()
    result = analysis.summarize_matrix(inventory, records)
    row = summary(result)
    assert result["status"] == "complete"
    assert result["coverage"]["expected_cells"] == 40
    assert result["coverage"]["pilot_cells"] == 4
    assert result["coverage"]["expected_model_cells"] == 200
    assert result["coverage"]["model_cells_by_status"]["available"] == 200
    assert row["known_rows"] == 60 and row["available_folds"] == 5
    assert row["weighted_log_loss"] == pytest.approx(55 / 150)
    assert row["weighted_brier"] == pytest.approx(55 / 300)
    assert row["weighted_log_loss"] != pytest.approx(.3)
    assert row["ranking_available_folds"] == 4
    assert row["mean_available_fold_roc_auc"] == pytest.approx((.6 + .65 + .7 + .75) / 4)
    assert row["folds"][0]["metrics"]["roc_auc"] is None
    assert "pooled_roc_auc" not in row and "confidence_interval" not in row
    assert row["observed_request_cores_across_disjoint_folds"] == 10


def test_missing_cell_remains_missing_without_zero_imputation_or_false_completion():
    inventory, records = fixtures()
    records[(analysis.PLACEMENTS[0], None, 4)] = {"state": "missing", "origin": "extension", "result": None}
    result = analysis.summarize_matrix(inventory, records)
    row = summary(result)
    assert result["status"] == "incomplete"
    assert result["coverage"]["cells_by_state"]["missing"] == 1
    assert result["coverage"]["model_cells_by_status"]["unavailable_cell"] == 5
    assert row["scope"] == "available_folds_only" and row["known_rows"] == 40
    assert row["weighted_log_loss"] == pytest.approx(.3)
    assert row["planned_all_intent_rows"] == 60
    assert row["folds"][4]["metrics"] is None
    assert row["folds"][4]["model_status"] == "unavailable_cell"


def test_paired_comparison_uses_only_common_available_folds():
    inventory, records = fixtures()
    row = records[(analysis.PLACEMENTS[0], None, 4)]
    row["result"]["models"]["jlens_t8"] = {"status": "unavailable_convergence"}
    row["state"] = row["result"]["status"] = "incomplete"
    for record in records.values():
        model = record["result"]["models"]["jlens_t8"]
        if model["status"] == "available":
            for groups in model["metrics"].values():
                for metric in groups.values():
                    metric["log_loss"] += .2
                    metric["brier"] += .1
    result = analysis.summarize_matrix(inventory, records)
    pair = next(r for r in result["paired_comparisons"] if r["placement"] == analysis.PLACEMENTS[0]
                and r["held_family"] is None and r["horizon"] == 1024 and r["intent"] == "all"
                and r["right_model"] == "jlens_t8")
    assert pair["paired_folds"] == [0, 1, 2, 3]
    assert pair["paired_known_rows"] == 40 and pair["all_folds_paired"] is False
    assert pair["weighted_log_loss_difference"] == pytest.approx(-.2)
    assert pair["weighted_brier_difference"] == pytest.approx(-.1)


def test_intents_and_family_stress_are_not_pooled_together():
    inventory, records = fixtures()
    result = analysis.summarize_matrix(inventory, records)
    assert len(result["summaries"]) == 2 * 4 * 4 * 5 * 5
    assert summary(result, intent="unsafe_direct")["known_rows"] == 15
    assert summary(result, family="structural_sham")["known_rows"] == 60
    assert result["claim_boundaries"]["ordinary_and_family_stress_samples_overlap"] is True


def test_unknown_labels_preserve_selected_and_censoring_denominators():
    inventory, records = fixtures()
    record = records[(analysis.PLACEMENTS[0], None, 4)]
    for model in record["result"]["models"].values():
        for groups in model["metrics"].values():
            for group in ("unsafe_direct", "all"):
                groups[group]["known"] -= 1
                groups[group]["unknown"] += 1
                groups[group]["right_censored"] += 1
    result = analysis.summarize_matrix(inventory, records)
    row = summary(result)
    assert row["observed_selected_rows"] == 60 and row["known_rows"] == 59 and row["unknown_rows"] == 1
    assert row["right_censored_rows"] == 1 and row["capped_negative_rows"] == 0
    assert row["weighted_log_loss"] == pytest.approx((22 - .5) / 59)


@pytest.mark.parametrize("mutation,code", [
    (lambda v: v.update(fold=4), "cell_binding"),
    (lambda v: v.update(inventory_sha256="0" * 64), "cell_binding"),
    (lambda v: v["models"]["jlens_t8"]["metrics"]["1024"]["all"].update(positive=3, roc_auc=.5, average_precision=.5), "paired_support"),
    (lambda v: v["models"]["jlens_t8"]["metrics"]["1024"]["all"].update(brier=float("nan")), "probability_metrics"),
])
def test_wrong_binding_support_and_nonfinite_metrics_rejected(mutation, code):
    inventory, records = fixtures()
    cell = inventory["cells"][0]
    value = records[analysis.key(cell)]["result"]
    mutation(value)
    with pytest.raises(analysis.AnalysisError, match=code):
        analysis.validate_cell(value, cell, analysis.sha(analysis.canonical(inventory)), 120)


def test_duplicate_fold_or_incomplete_inventory_rejected():
    inventory, _ = fixtures()
    inventory["cells"][-1] = deepcopy(inventory["cells"][0])
    with pytest.raises(analysis.AnalysisError, match="inventory_matrix"):
        analysis.validate_inventory(inventory)


def durable_fixture(tmp_path):
    inventory, records = fixtures()
    root = tmp_path / "extension"
    root.mkdir()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(analysis.canonical(inventory))
    launcher = b"# Synthetic fixture: never executed.\n"
    (root / "launch.py").write_bytes(launcher)
    pilot, cells = [], []
    for cell in inventory["cells"]:
        record = records[analysis.key(cell)]
        if record["origin"] == "pilot":
            path = tmp_path / f"pilot-{len(pilot)}.json"
            raw = analysis.canonical(record["result"])
            path.write_bytes(raw)
            pilot.append({**cell, "result_path": str(path), "result_sha256": analysis.sha(raw)})
        else:
            cells.append({**cell, "index": len(cells), "output_name": f"new-{len(cells)}.json"})
    manifest = {"schema_version": "text-challenger-extension-execution-v1", "pilot_outcomes_inspected": True,
        "retrospective_exploratory": True, "confirmation_opened": False,
        "inventory_path": str(inventory_path), "inventory_sha256": analysis.sha(inventory_path.read_bytes()),
        "source_pins": {}, "input_pins": {}, "runtime": {}, "output_root": str(root),
        "launcher_source_sha256": analysis.sha(launcher), "completed_pilot_cells": pilot, "cells": cells,
        "freeze_commit": "synthetic"}
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(analysis.canonical(manifest))
    manifest_sha = analysis.sha(manifest_path.read_bytes())
    for cell in cells:
        raw = analysis.canonical(records[analysis.key(cell)]["result"])
        (root / cell["output_name"]).write_bytes(raw)
        terminal = {"schema_version": "text-challenger-extension-cell-exit-v1", "manifest_sha256": manifest_sha,
            "index": cell["index"], "placement": cell["placement"], "held_family": cell["held_family"],
            "fold": cell["fold"], "returncode": 0, "result_sha256": None}
        (root / f"cell-{cell['index']:03d}-exit.json").write_bytes(analysis.canonical(terminal))
        terminal.update(status="complete", result_sha256=analysis.sha(raw))
        (root / f"cell-{cell['index']:03d}-validated.json").write_bytes(analysis.canonical(terminal))
    return manifest_path, manifest_sha, manifest


def test_durable_matrix_validates_all_pilot_and_terminal_hashes(tmp_path):
    manifest_path, digest, _ = durable_fixture(tmp_path)
    inventory, records, provenance = analysis.load_matrix(manifest_path, digest)
    assert len(records) == 40 and provenance["pilot_outcomes_inspected"] is True
    assert analysis.summarize_matrix(inventory, records)["status"] == "complete"


@pytest.mark.parametrize("which", ["pilot", "new", "receipt"])
def test_tampered_pilot_result_or_new_result_or_receipt_rejected(tmp_path, which):
    path, digest, manifest = durable_fixture(tmp_path)
    if which == "pilot":
        victim = Path(manifest["completed_pilot_cells"][0]["result_path"])
    elif which == "new":
        victim = path.parent / manifest["cells"][0]["output_name"]
    else:
        victim = path.parent / "cell-000-validated.json"
    value = json.loads(victim.read_text())
    if which == "receipt":
        value["manifest_sha256"] = "0" * 64
    else:
        value["models"]["training_prevalence"]["metrics"]["1024"]["all"]["log_loss"] += .1
    victim.write_bytes(analysis.canonical(value))
    with pytest.raises(analysis.AnalysisError):
        analysis.load_matrix(path, digest)


def test_interrupted_result_without_validated_terminal_is_not_scored(tmp_path):
    path, digest, _ = durable_fixture(tmp_path)
    (path.parent / "cell-000-validated.json").unlink()
    inventory, records, _ = analysis.load_matrix(path, digest)
    result = analysis.summarize_matrix(inventory, records)
    assert result["coverage"]["cells_by_state"]["interrupted"] == 1
    assert result["status"] == "incomplete"


def test_active_writer_is_rejected(tmp_path):
    path, digest, _ = durable_fixture(tmp_path)
    with (path.parent / ".launcher.lock").open("wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(analysis.AnalysisError, match="active_writer"):
            analysis.load_matrix(path, digest)
