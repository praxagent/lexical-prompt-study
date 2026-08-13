from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .jlens_falsification_topology import _apply_fit, _hashed_ngrams
from .weaponization_analysis import (
    STRUCTURAL_FIELDS,
    _average_precision,
    _roc_auc,
    fit_ridge_logistic,
)


MODEL_NAMES = (
    "prompt_full_hashed",
    "request_only_hashed",
    "structural",
    "jlens_trajectory",
    "sae_readouts",
    "prompt_plus_jlens",
    "all_prefill_readouts",
)


def _features(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    prompt = np.stack([_hashed_ngrams(str(row["prompt_text"])) for row in rows])
    request = np.stack([_hashed_ngrams(str(row["request_text"])) for row in rows])
    structural = np.asarray(
        [
            [float(row["structural_metrics"][field]) for field in STRUCTURAL_FIELDS]
            for row in rows
        ],
        dtype=np.float64,
    )
    jlens = np.asarray(
        [row["jlens_refusal_minus_compliance_trajectory"] for row in rows],
        dtype=np.float64,
    )
    sae = np.asarray(
        [
            [
                row["feature_6779_magnitude"],
                row["frozen_subspace_score"],
                row["sae_normalized_reconstruction_error"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "prompt_full_hashed": prompt,
        "request_only_hashed": request,
        "structural": structural,
        "jlens_trajectory": jlens,
        "sae_readouts": sae,
        "prompt_plus_jlens": np.column_stack([prompt, jlens]),
        "all_prefill_readouts": np.column_stack([prompt, request, structural, jlens, sae]),
    }


def _folds_by_request(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    request_ids = sorted(
        {str(row["request_id"]) for row in rows},
        key=lambda request_id: (
            next(
                str(row["request_sha256"])
                for row in rows
                if row["request_id"] == request_id
            ),
            request_id,
        ),
    )
    assignment = {request_id: index % 5 for index, request_id in enumerate(request_ids)}
    return np.asarray([assignment[str(row["request_id"])] for row in rows])


def _cross_validated_predictions(
    matrix: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        fit = fit_ridge_logistic(matrix[train], labels[train], penalty=0.01)
        predictions[test] = _apply_fit(matrix[test], fit)
    if not np.isfinite(predictions).all():
        raise ValueError("cross-validation prediction coverage drift")
    return predictions


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _roc_auc(labels, predictions),
        "average_precision": _average_precision(labels, predictions),
    }


def analyze_cross_validation(
    *, topology_path: Path, acquisition_root: Path, score_root: Path, output_path: Path
) -> dict[str, Any]:
    topology = json.loads(topology_path.read_text())
    observations = {
        row["trial_id"]: row
        for row in topology["observations"]
        if row["request_class"] == "harmful_request"
    }
    receipts = {
        path.stem: json.loads(path.read_text())
        for path in sorted((acquisition_root / "receipts").glob("*.json"))
        if path.stem in observations
    }
    scores = {
        path.stem: json.loads(path.read_text())
        for path in sorted((score_root / "trials").glob("*.json"))
    }
    if set(observations) != set(receipts) or set(observations) != set(scores):
        raise ValueError("cross-validation input trial drift")
    rows = []
    for trial_id in sorted(observations):
        observation = observations[trial_id]
        receipt = receipts[trial_id]
        score = scores[trial_id]
        if (
            receipt["prompt_sha256"] != observation["prompt_sha256"]
            or score["generation_receipt_sha256"]
            != sha256_file(acquisition_root / "receipts" / f"{trial_id}.json")
        ):
            raise ValueError("cross-validation linkage drift")
        rows.append(
            {
                **observation,
                **receipt,
                "harmbench_binary_prediction": bool(score["binary_prediction"]),
            }
        )
    if len(rows) != 960:
        raise ValueError("cross-validation harmful row count drift")
    labels = np.asarray([int(row["harmbench_binary_prediction"]) for row in rows])
    matrices = _features(rows)
    request_folds = _folds_by_request(rows)
    mask_folds = np.asarray([int(row["attack_block_mask"]) for row in rows])
    output_metrics: dict[str, Any] = {}
    private_predictions: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for name in MODEL_NAMES:
        request_predictions = _cross_validated_predictions(
            matrices[name], labels, request_folds
        )
        mask_predictions = _cross_validated_predictions(matrices[name], labels, mask_folds)
        output_metrics[name] = {
            "five_fold_unseen_request": _metrics(labels, request_predictions),
            "sixteen_fold_unseen_mask": _metrics(labels, mask_predictions),
            "feature_dimension": int(matrices[name].shape[1]),
        }
        private_predictions[name] = {
            "five_fold_unseen_request": request_predictions.tolist(),
            "sixteen_fold_unseen_mask": mask_predictions.tolist(),
        }
    output = {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-signal-falsification-v1",
        "status": "post_outcome_cross_validated_success_prediction_complete",
        "warning": "Model families and cross-validation were specified after A132 was opened. Estimates are exploratory and need prospective replication.",
        "endpoint": "HarmBench binary success among 960 harmful-request mutation rows only",
        "folds": {
            "unseen_request": "five deterministic folds assigning all 16 masks for one request to the same fold",
            "unseen_mask": "leave one of 16 exact-length attack-block masks out",
        },
        "fit": "class-balanced deterministic ridge logistic, L2 penalty 0.01, no hyperparameter selection",
        "metrics": output_metrics,
        "bindings": {
            "private_topology_sha256": sha256_file(topology_path),
            "acquisition_summary_sha256": sha256_file(acquisition_root / "summary.json"),
            "score_summary_sha256": sha256_file(score_root / "summary.json"),
        },
        "claim_boundaries": {
            "prospective": False,
            "independent_confirmation": False,
            "deployable_threshold": False,
            "unopened_v2_confirmation_opened": False,
            "deployment_authorized": False,
        },
        "raw_prompt_request_token_generation_or_row_level_content_public": False,
    }
    raw = canonical_json_bytes(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    private_sha = sha256_bytes(canonical_json_bytes(private_predictions))
    return {**output, "output_sha256": sha256_bytes(raw), "private_predictions_sha256": private_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_cross_validation(
        topology_path=args.topology,
        acquisition_root=args.acquisition_root,
        score_root=args.score_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_sha256": result["output_sha256"],
                "private_predictions_sha256": result["private_predictions_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
