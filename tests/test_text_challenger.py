from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lexical_prompt_study import text_challenger as challenger


def records():
    out = []
    for core in range(15):
        for placement in challenger.PLACEMENTS:
            for family in challenger.FAMILIES:
                for label in (False, True):
                    signal = "positive" if label else "negative"
                    out.append({"core": f"synthetic-core-{core}", "fold": core % 5,
                        "placement": placement, "family": family, "intent": "unsafe_direct",
                        "origin": "original_eos", "prompt": f"Harmless invented {signal} task with words.",
                        "prefix_hash": [float(not label), float(label)] + [0.0] * 254,
                        "internal": [float(label)] * 31, "structural": [1.0] * 6,
                        "scores": [{"requested_horizon": h, "binary_prediction": label,
                                    "right_censored": False, "observed_token_count": 12}
                                   for h in challenger.HORIZONS]})
    return out


class TestTextChallenger(unittest.TestCase):
    def test_folds_hold_all_variants_of_a_core_together(self):
        data = records()
        train, test = challenger.cell_masks(data, challenger.PLACEMENTS[0], 0)
        training_cores = {r["core"] for r, keep in zip(data, train, strict=True) if keep}
        test_cores = {r["core"] for r, keep in zip(data, test, strict=True) if keep}
        assert not training_cores & test_cores
        assert len(training_cores) == 12 and len(test_cores) == 3
        assert all(r["placement"] == challenger.PLACEMENTS[0] for r, keep in zip(data, train | test, strict=True) if keep)


    def test_same_core_in_two_folds_is_rejected(self):
        data = records()
        data[0]["fold"] = 4
        with self.assertRaisesRegex(challenger.ChallengerError, "core_leakage"):
            challenger.cell_masks(data, challenger.PLACEMENTS[0], 0)


    def test_structural_family_stress_excludes_family_and_core_from_training(self):
        data = records()
        train, test = challenger.cell_masks(data, challenger.PLACEMENTS[1], 1, "structural_sham")
        assert all(r["family"] != "structural_sham" for r, keep in zip(data, train, strict=True) if keep)
        assert all(r["family"] == "structural_sham" and r["fold"] == 1 for r, keep in zip(data, test, strict=True) if keep)


    def test_unknown_training_label_excluded_but_unknown_test_retained(self):
        data = records()
        for record in data:
            record["scores"][-1]["binary_prediction"] = None
        train, test = challenger.cell_masks(data, challenger.PLACEMENTS[0], 0)
        assert not train.any() and test.any()


    def test_vocabulary_and_document_frequencies_do_not_see_test_documents(self):
        data = records()
        training = data[:4]
        testing = deepcopy(data[4:8])
        for record in testing:
            record["prompt"] += " EXCLUSIVELYHELDOUTWORD"
        matrices, vectorizers = challenger.vectorize_text(training, testing)
        assert all("EXCLUSIVELYHELDOUTWORD" not in v.vocabulary_ for v in vectorizers)
        assert matrices[0].shape[1] == matrices[1].shape[1]
        assert len(vectorizers) == 2


    def test_metrics_keep_unknown_and_censoring_separate(self):
        data = records()[:3]
        for score in data[2]["scores"]:
            score["binary_prediction"] = None
            score["right_censored"] = True
        result = challenger.metrics(data, np.array([.5, .5, .9]))["1024"]["all"]
        assert result["known"] == 2 and result["unknown"] == 1
        assert result["positive"] == 1 and result["right_censored"] == 1
        assert result["capped_negative"] == 0
        self.assertAlmostEqual(result["log_loss"], math.log(2))
        self.assertAlmostEqual(result["brier"], .25)


    def test_empty_intent_and_one_class_ranking_are_unavailable(self):
        data = [records()[0]]
        result = challenger.metrics(data, np.array([.1]))["128"]
        assert result["all"]["roc_auc"] is None and result["all"]["average_precision"] is None
        assert result["safe_classify_exact"]["known"] == 0
        assert result["safe_classify_exact"]["log_loss"] is None


    def test_matched_synthetic_fit_exposes_only_aggregate_outputs(self):
        result = challenger.evaluate_cell(records(), challenger.PLACEMENTS[0], 0)
        assert result["status"] == "complete"
        assert set(result["models"]) == set(challenger.MODELS)
        scores = {name: row["metrics"]["1024"]["all"] for name, row in result["models"].items()}
        assert scores["prompt_plus_prefix_hash"]["log_loss"] < scores["training_prevalence"]["log_loss"]
        assert scores["jlens_t8"]["roc_auc"] == 1
        output = json.dumps(result)
        assert "Harmless invented" not in output and "synthetic-core" not in output
        assert '"vocabulary"' not in output and '"vocabulary_"' not in output and '"predictions"' not in output


    def test_single_class_training_is_explicitly_unavailable(self):
        data = records()
        for record in data:
            for score in record["scores"]:
                score["binary_prediction"] = False
        result = challenger.evaluate_cell(data, challenger.PLACEMENTS[0], 0)
        assert result["status"] == "unavailable_training_classes_or_test_support"
        assert result["models"] == {}


    def test_deadline_does_not_fabricate_scores(self):
        result = challenger.evaluate_cell(records(), challenger.PLACEMENTS[0], 0, max_seconds=0)
        assert result["status"] == "incomplete"
        assert all(result["models"][name] == {"status": "unavailable_deadline"} for name in challenger.MODELS[1:])


    def test_predictions_must_be_finite_probabilities(self):
        with self.assertRaisesRegex(challenger.ChallengerError, "predictions"):
            challenger.metrics(records()[:1], np.array([np.nan]))


    def test_prefix_features_bind_original_regime_receipt_and_exact_eight_tokens(self):
        meta = {"regime": "original", "prefix_token_count": 8,
                "source_generation_receipt_sha256": "a" * 64, "prefix_token_ids_sha256": "b" * 64}
        readout = {"prefix_token_ids_sha256": "b" * 64}
        vector = np.zeros(256)
        assert len(challenger.validate_prefix(meta, vector, "a" * 64, readout)) == 256
        for key, value in (("regime", "continuation"), ("prefix_token_count", 32),
                           ("source_generation_receipt_sha256", "c" * 64),
                           ("prefix_token_ids_sha256", "c" * 64)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(challenger.ChallengerError, "prefix_binding"):
                    challenger.validate_prefix(dict(meta, **{key: value}), vector, "a" * 64, readout)


    def test_prefix_vector_must_have_finite_fixed_dimensions(self):
        meta = {"regime": "original", "prefix_token_count": 8,
                "source_generation_receipt_sha256": "a" * 64, "prefix_token_ids_sha256": "b" * 64}
        for vector in (np.zeros(255), np.full(256, np.nan)):
            with self.assertRaisesRegex(challenger.ChallengerError, "prefix_features"):
                challenger.validate_prefix(meta, vector, "a" * 64, {"prefix_token_ids_sha256": "b" * 64})


    def test_output_is_private_and_cannot_overwrite(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        path = tmp_path / "result.json"
        challenger.immutable(path, {"status": "synthetic"})
        assert path.stat().st_mode & 0o777 == 0o600
        with self.assertRaises(FileExistsError):
            challenger.immutable(path, {"status": "changed"})


if __name__ == "__main__":
    unittest.main()
