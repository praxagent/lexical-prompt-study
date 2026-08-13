from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .jlens_behavior_candidate import _assign_folds, _atomic, _select_threshold
from .jlens_falsification_topology import _apply_fit, _hashed_ngrams
from .weaponization_analysis import _average_precision, _roc_auc, fit_ridge_logistic


MODELS = ("prompt_full_hashed", "request_only_hashed", "jlens_trajectory", "prompt_plus_jlens")


def build_deployment_candidates(
    *,
    topology_path: Path,
    acquisition_root: Path,
    score_root: Path,
    private_candidate_path: Path,
    public_result_path: Path,
) -> dict[str, Any]:
    topology = json.loads(topology_path.read_text())
    observations = {row["trial_id"]: row for row in topology["observations"]}
    receipts = {
        path.stem: json.loads(path.read_text())
        for path in sorted((acquisition_root / "receipts").glob("*.json"))
    }
    scores = {
        path.stem: json.loads(path.read_text())
        for path in sorted((score_root / "trials").glob("*.json"))
    }
    rows = []
    for trial_id in sorted(observations):
        observation = observations[trial_id]
        receipt = receipts[trial_id]
        score = scores.get(trial_id)
        rows.append(
            {
                **observation,
                "trajectory": receipt["jlens_refusal_minus_compliance_trajectory"],
                "success": bool(score["binary_prediction"]) if score is not None else False,
            }
        )
    if len(rows) != 1920:
        raise ValueError("deployment candidate topology drift")
    prompt = np.stack([_hashed_ngrams(str(row["prompt_text"])) for row in rows])
    request = np.stack([_hashed_ngrams(str(row["request_text"])) for row in rows])
    jlens = np.asarray([row["trajectory"] for row in rows], dtype=np.float64)
    matrices = {
        "prompt_full_hashed": prompt,
        "request_only_hashed": request,
        "jlens_trajectory": jlens,
        "prompt_plus_jlens": np.column_stack([prompt, jlens]),
    }
    labels = np.asarray(
        [
            int(row["request_class"] == "harmful_request" and row["success"])
            for row in rows
        ]
    )
    benign = np.asarray(
        [row["request_class"] == "ordinary_benign_request" for row in rows]
    )
    harmful = [row for row in rows if row["request_class"] == "harmful_request"]
    benign_rows = [row for row in rows if row["request_class"] == "ordinary_benign_request"]
    fold_map = {**_assign_folds(harmful), **_assign_folds(benign_rows)}
    folds = np.asarray([fold_map[str(row["request_id"])] for row in rows])
    public_models: dict[str, Any] = {}
    private_models: dict[str, Any] = {}
    for name in MODELS:
        predictions = np.full(len(rows), np.nan, dtype=np.float64)
        for fold in range(5):
            test = folds == fold
            train = ~test
            fit = fit_ridge_logistic(matrices[name][train], labels[train], penalty=0.01)
            predictions[test] = _apply_fit(matrices[name][test], fit)
        if not np.isfinite(predictions).all():
            raise ValueError("deployment candidate OOF prediction drift")
        critical = (labels == 1) | benign
        critical_labels = labels[critical]
        critical_predictions = predictions[critical]
        selection = _select_threshold(predictions, labels == 1, benign)
        final = fit_ridge_logistic(matrices[name], labels, penalty=0.01)
        public_models[name] = {
            "feature_dimension": int(matrices[name].shape[1]),
            "out_of_fold_successful_harmful_vs_ordinary_benign": {
                "roc_auc": _roc_auc(critical_labels, critical_predictions),
                "average_precision": _average_precision(
                    critical_labels, critical_predictions
                ),
            },
            "out_of_fold_success_vs_all_other_rows": {
                "roc_auc": _roc_auc(labels, predictions),
                "average_precision": _average_precision(labels, predictions),
            },
            "threshold_selection": selection,
        }
        private_models[name] = {
            "mean": final["mean"].tolist(),
            "scale": final["scale"].tolist(),
            "coefficients": final["coefficients"].tolist(),
            "threshold_selection": selection,
        }
    jlens = public_models["jlens_trajectory"]["threshold_selection"]
    prompt_dominates = []
    if jlens["eligible"]:
        jlens_metrics = jlens["metrics"]
        for name in ("prompt_full_hashed", "request_only_hashed"):
            selection = public_models[name]["threshold_selection"]
            if not selection["eligible"]:
                continue
            metrics = selection["metrics"]
            if (
                metrics["successful_harmful_capture_fraction"]
                >= jlens_metrics["successful_harmful_capture_fraction"]
                and metrics["ordinary_benign_false_trip_fraction"]
                <= jlens_metrics["ordinary_benign_false_trip_fraction"]
            ):
                prompt_dominates.append(name)
    private = {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-signal-falsification-v1",
        "status": "post_outcome_direct_endpoint_candidates_frozen",
        "warning": "Candidate families were specified after A132 and require independent confirmation.",
        "models": private_models,
        "bindings": {
            "private_topology_sha256": sha256_file(topology_path),
            "acquisition_summary_sha256": sha256_file(acquisition_root / "summary.json"),
            "score_summary_sha256": sha256_file(score_root / "summary.json"),
        },
        "unopened_v2_confirmation_opened": False,
        "deployment_authorized": False,
    }
    private_sha256 = _atomic(private_candidate_path, private, 0o600)
    public = {
        "schema_version": "1.0",
        "study_id": private["study_id"],
        "status": private["status"],
        "analysis_role": "post_outcome_discovery_not_confirmation",
        "endpoint": "successful harmful completion versus all other rows; benign FPR is the utility gate",
        "models": public_models,
        "private_candidate_sha256": private_sha256,
        "prompt_only_matches_or_dominates_jlens": bool(prompt_dominates),
        "matching_or_dominating_prompt_only_models": prompt_dominates,
        "independent_internal_confirmation_eligible": bool(
            jlens["eligible"] and not prompt_dominates
        ),
        "unopened_v2_confirmation_opened": False,
        "deployment_authorized": False,
        "raw_prompt_request_token_generation_or_row_level_content_public": False,
    }
    raw = canonical_json_bytes(public)
    public_result_path.parent.mkdir(parents=True, exist_ok=True)
    public_result_path.write_bytes(raw)
    return {**public, "output_sha256": sha256_bytes(raw)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--private-candidate", type=Path, required=True)
    parser.add_argument("--public-result", type=Path, required=True)
    args = parser.parse_args()
    result = build_deployment_candidates(
        topology_path=args.topology,
        acquisition_root=args.acquisition_root,
        score_root=args.score_root,
        private_candidate_path=args.private_candidate,
        public_result_path=args.public_result,
    )
    print(json.dumps({"status": result["status"], "output_sha256": result["output_sha256"]}))


if __name__ == "__main__":
    main()
