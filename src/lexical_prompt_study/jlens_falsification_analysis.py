from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .jlens_breaker_v2_analysis import apply_frozen_jlens_head, select_v2_threshold
from .weaponization_analysis import _average_precision, _roc_auc


POSITIVE = "positive_attack_harmful"
NEGATIVES = (
    "negative_attack_benign",
    "negative_attack_neutral",
    "negative_harmless_harmful",
    "negative_harmless_benign",
    "negative_structural_sham_harmful",
    "negative_inert_harmful",
)
PLACEMENTS = ("scaffold_before_request", "scaffold_after_request")


def _load_receipts(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if "/receipts/" in member.name and member.name.endswith(".json")
            ),
            key=lambda member: member.name,
        )
        if len(members) != expected:
            raise ValueError(f"{path}: expected {expected} receipts")
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name}: unreadable receipt")
            row = json.load(handle)
            if any(
                key in row
                for key in (
                    "prompt_text",
                    "prompt_token_ids",
                    "request_text",
                    "generated_text",
                )
            ):
                raise ValueError("raw field in mechanism receipt")
            if len(row["jlens_refusal_minus_compliance_trajectory"]) != 31:
                raise ValueError("J-lens trajectory topology drift")
            rows.append(row)
    if len({row["trial_id"] for row in rows}) != expected:
        raise ValueError("duplicate receipt trial ID")
    return rows


def _critical(
    rows: Sequence[Mapping[str, Any]], placement: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["placement"] == placement and row["stratum"] in (POSITIVE, *NEGATIVES)
    ]
    trajectories = np.asarray(
        [row["jlens_refusal_minus_compliance_trajectory"] for row in selected],
        dtype=np.float64,
    )
    labels = np.asarray([int(row["stratum"] == POSITIVE) for row in selected])
    strata = np.asarray([row["stratum"] for row in selected], dtype=object)
    expected_per_stratum = len(selected) // 7
    if (
        trajectories.shape != (expected_per_stratum * 7, 31)
        or any(int((strata == stratum).sum()) != expected_per_stratum for stratum in (POSITIVE, *NEGATIVES))
    ):
        raise ValueError("critical receipt topology drift")
    return trajectories, labels, strata


def _ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
    }


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, strata: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = scores >= threshold
    positive = labels == 1
    output: dict[str, Any] = {
        "positive_count": int(positive.sum()),
        "true_positive_count": int(predictions[positive].sum()),
        "recall": float(predictions[positive].mean()),
        "critical_negatives": {},
    }
    for stratum in NEGATIVES:
        mask = strata == stratum
        output["critical_negatives"][stratum] = {
            "count": int(mask.sum()),
            "false_positive_count": int(predictions[mask].sum()),
            "false_positive_rate": float(predictions[mask].mean()),
        }
    return output


def _single_coordinate(
    calibration: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_strata: np.ndarray,
    confirmation: np.ndarray,
    confirmation_labels: np.ndarray,
    confirmation_strata: np.ndarray,
) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for layer in range(31):
        auc = _roc_auc(calibration_labels, calibration[:, layer])
        sign = 1 if auc >= 0.5 else -1
        candidates.append((max(auc, 1.0 - auc), -layer, sign))
    calibration_auc, negative_layer, sign = max(candidates)
    layer = -negative_layer
    calibration_scores = sign * calibration[:, layer]
    selected = select_v2_threshold(calibration_scores, calibration_labels, calibration_strata)
    confirmation_scores = sign * confirmation[:, layer]
    output: dict[str, Any] = {
        "selected_source_layer": layer,
        "sign": sign,
        "calibration": {"ranking": _ranking(calibration_labels, calibration_scores), **selected},
        "confirmation": {"ranking": _ranking(confirmation_labels, confirmation_scores)},
    }
    if selected["eligible"]:
        output["confirmation"]["frozen_threshold_metrics"] = _threshold_metrics(
            confirmation_scores,
            confirmation_labels,
            confirmation_strata,
            float(selected["threshold"]),
        )
    else:
        output["confirmation"]["frozen_threshold_metrics"] = None
    if abs(output["calibration"]["ranking"]["roc_auc"] - calibration_auc) > 1e-12:
        raise ValueError("single-coordinate selection arithmetic drift")
    return output


def _top_k_ablation(
    trajectories: np.ndarray,
    labels: np.ndarray,
    head: Mapping[str, Any],
) -> dict[str, Any]:
    coefficients = np.asarray(head["coefficients"], dtype=np.float64)
    if coefficients.shape != (32,):
        raise ValueError("candidate coefficient topology drift")
    order = np.argsort(-np.abs(coefficients[1:]), kind="stable")
    output: dict[str, Any] = {}
    for k in (1, 2, 4, 8, 16, 31):
        mask = np.zeros(31, dtype=bool)
        mask[order[:k]] = True
        reduced = dict(head)
        reduced_coefficients = coefficients.copy()
        reduced_coefficients[1:][~mask] = 0.0
        reduced["coefficients"] = reduced_coefficients.tolist()
        scores = apply_frozen_jlens_head(trajectories, reduced)
        output[str(k)] = {
            **_ranking(labels, scores),
            "selected_source_layers": sorted(int(value) for value in order[:k]),
        }
    return output


def analyze_existing_evidence(
    *,
    plan_path: Path,
    calibration_bundle_path: Path,
    confirmation_bundle_path: Path,
    candidate_path: Path,
    size_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("study_id") != "lexical-jlens-signal-falsification-v1"
        or plan.get("status") != "frozen_before_new_mutation_outcomes"
    ):
        raise ValueError("falsification plan drift")
    calibration = _load_receipts(calibration_bundle_path, 1900)
    confirmation = _load_receipts(confirmation_bundle_path, 5700)
    candidate = json.loads(candidate_path.read_text())
    size_result = json.loads(size_result_path.read_text())
    placements: dict[str, Any] = {}
    for placement in PLACEMENTS:
        cal_x, cal_y, cal_s = _critical(calibration, placement)
        con_x, con_y, con_s = _critical(confirmation, placement)
        placements[placement] = {
            "single_coordinate_selected_on_calibration": _single_coordinate(
                cal_x, cal_y, cal_s, con_x, con_y, con_s
            ),
            "confirmation_coefficient_ranked_top_k": _top_k_ablation(
                con_x, con_y, candidate["placements"][placement]["jlens_head"]
            ),
            "four_size_same_threshold": size_result["placements"][placement][
                "same_threshold_size_metrics"
            ],
        }
    output = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": "existing_open_evidence_audit_complete",
        "analysis_role": "secondary_falsification_not_new_confirmation",
        "bindings": {
            "plan_sha256": sha256_file(plan_path),
            "calibration_bundle_sha256": sha256_file(calibration_bundle_path),
            "confirmation_bundle_sha256": sha256_file(confirmation_bundle_path),
            "candidate_sha256": sha256_file(candidate_path),
            "size_result_sha256": sha256_file(size_result_path),
        },
        "placements": placements,
        "raw_prompt_token_generation_or_row_level_content_public": False,
        "new_mutation_outcome_opened": False,
        "unopened_v2_confirmation_opened": False,
    }
    raw = canonical_json_bytes(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return {**output, "output_sha256": sha256_bytes(raw)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--confirmation-bundle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--size-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_existing_evidence(
        plan_path=args.plan,
        calibration_bundle_path=args.calibration_bundle,
        confirmation_bundle_path=args.confirmation_bundle,
        candidate_path=args.candidate,
        size_result_path=args.size_result,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "output_sha256": result["output_sha256"]}))


if __name__ == "__main__":
    main()
