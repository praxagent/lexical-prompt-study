from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from .intervention_plan import validate_intervention_plan
from .interventions import validate_intervention_receipt


def calibration_condition_id(sign: int, rho: float) -> str:
    if sign not in {-1, 1} or rho <= 0 or not np.isfinite(rho):
        raise ValueError("invalid calibration condition")
    sign_name = "negative" if sign == -1 else "positive"
    rho_name = format(rho, ".10g").replace(".", "p")
    return f"primary_{sign_name}_rho_{rho_name}"


def planned_calibration_conditions(plan: dict) -> list[str]:
    conditions = ["zero"]
    for rho in plan["discovery_alpha_calibration"]["rho_ladder"]:
        conditions.extend(
            [
                calibration_condition_id(-1, float(rho)),
                calibration_condition_id(1, float(rho)),
            ]
        )
    if len(conditions) != len(set(conditions)):
        raise ValueError("duplicate calibration condition")
    return conditions


def _interval(values: np.ndarray, *, replicates: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def analyze_alpha_calibration(rows: list[dict], plan: dict) -> dict:
    calibration = plan["discovery_alpha_calibration"]
    expected_behaviors = int(calibration["behavior_count"])
    conditions = planned_calibration_conditions(plan)
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        condition = row["condition_id"]
        if condition not in conditions:
            raise ValueError(f"unexpected calibration condition: {condition}")
        behavior_id = row["behavior_id"]
        if condition in indexed[behavior_id]:
            raise ValueError(f"duplicate calibration row: {behavior_id}/{condition}")
        score = row["evaluator_score"]
        if score is None or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("calibration evaluator score outside [0, 1]")
        indexed[behavior_id][condition] = row
    if len(indexed) != expected_behaviors:
        raise ValueError("calibration behavior count mismatch")
    if any(set(value) != set(conditions) for value in indexed.values()):
        raise ValueError("calibration condition topology mismatch")

    behavior_ids = sorted(indexed)
    zero = np.array(
        [indexed[item]["zero"]["evaluator_score"] for item in behavior_ids],
        dtype=float,
    )
    scoring = calibration["scoring"]
    gates = calibration["eligible_alpha_rule"]
    candidates = []
    for candidate_index, rho_value in enumerate(calibration["rho_ladder"]):
        rho = float(rho_value)
        negative_id = calibration_condition_id(-1, rho)
        positive_id = calibration_condition_id(1, rho)
        negative = np.array(
            [indexed[item][negative_id]["evaluator_score"] for item in behavior_ids],
            dtype=float,
        )
        positive = np.array(
            [indexed[item][positive_id]["evaluator_score"] for item in behavior_ids],
            dtype=float,
        )
        restoring_sign = -1 if negative.mean() <= positive.mean() else 1
        restoring = negative if restoring_sign == -1 else positive
        opposite = positive if restoring_sign == -1 else negative
        signed_half_span = (opposite - restoring) / 2
        restoring_minus_zero = restoring - zero
        opposite_minus_zero = opposite - zero
        phase_rows = [
            indexed[behavior_id][condition]
            for behavior_id in behavior_ids
            for condition in ("zero", negative_id, positive_id)
        ]
        nonzero_rows = [
            indexed[behavior_id][condition]
            for behavior_id in behavior_ids
            for condition in (negative_id, positive_id)
        ]
        realized_alphas = np.array(
            [float(row["requested_alpha"]) for row in nonzero_rows],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(realized_alphas))
            or np.any(realized_alphas <= 0)
            or not np.allclose(
                realized_alphas,
                realized_alphas[0],
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError("calibration alpha drift within dose")
        parse_failures = sum(not bool(row["evaluator_parse_ok"]) for row in phase_rows)
        runtime_errors = sum(row.get("error") is not None for row in phase_rows)
        max_relative_error = max(
            float(row["maximum_requested_realized_relative_error"])
            for row in nonzero_rows
        )
        max_residual_ratio = max(
            float(row["maximum_event_delta_to_pre_residual_norm"])
            for row in nonzero_rows
        )
        zero_truncation = np.mean(
            [bool(indexed[item]["zero"]["truncated"]) for item in behavior_ids]
        )
        steered_truncation = max(
            np.mean(
                [bool(indexed[item][condition]["truncated"]) for item in behavior_ids]
            )
            for condition in (negative_id, positive_id)
        )
        interval = _interval(
            signed_half_span,
            replicates=int(scoring["bootstrap_replicates"]),
            seed=int(scoring["bootstrap_seed"]) + candidate_index,
        )
        metrics = {
            "rho": rho,
            "restoring_sign": restoring_sign,
            "alpha": float(realized_alphas[0]),
            "mean_signed_half_span": float(signed_half_span.mean()),
            "signed_half_span_bootstrap_95_interval": interval,
            "mean_restoring_minus_zero": float(restoring_minus_zero.mean()),
            "mean_opposite_minus_zero": float(opposite_minus_zero.mean()),
            "parse_failure_fraction": parse_failures / len(phase_rows),
            "runtime_errors": runtime_errors,
            "maximum_requested_realized_relative_error": max_relative_error,
            "maximum_event_delta_to_pre_residual_norm": max_residual_ratio,
            "maximum_length_truncation_rate_increase_vs_zero": float(
                steered_truncation - zero_truncation
            ),
        }
        metrics["eligible"] = bool(
            runtime_errors == gates["runtime_errors"]
            and metrics["parse_failure_fraction"]
            <= gates["maximum_parse_failure_fraction"]
            and max_relative_error
            <= gates["maximum_requested_realized_relative_error"]
            and max_residual_ratio
            <= gates["maximum_event_delta_to_pre_residual_norm"]
            and metrics["mean_signed_half_span"]
            >= gates["minimum_mean_signed_half_span"]
            and interval[0]
            > gates["signed_half_span_bootstrap_ci_lower_must_exceed"]
            and metrics["mean_restoring_minus_zero"]
            <= gates["maximum_mean_restoring_minus_zero"]
            and metrics["mean_opposite_minus_zero"]
            >= gates["minimum_mean_opposite_minus_zero"]
            and metrics["maximum_length_truncation_rate_increase_vs_zero"]
            <= gates["maximum_length_truncation_rate_increase_vs_zero"]
        )
        candidates.append(metrics)

    selected = next((item for item in candidates if item["eligible"]), None)
    return {
        "schema_version": "1.0",
        "protocol_id": plan["protocol_id"],
        "status": "selected" if selected is not None else "stopped_no_eligible_alpha",
        "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "n_behaviors": len(behavior_ids),
        "conditions": conditions,
        "candidates": candidates,
        "selection": selected,
        "confirmatory_outcomes_opened": False,
    }


def analyze_alpha_calibration_receipts(
    *,
    intervention_plan_path: Path,
    public_plan_path: Path,
    gate3_analysis_path: Path,
    generation_root: Path,
    score_root: Path,
    output_path: Path,
) -> dict:
    validate_intervention_plan(
        intervention_plan_path,
        public_plan_path,
        gate3_analysis_path,
    )
    plan = json.loads(intervention_plan_path.read_text())
    plan_sha = sha256_file(intervention_plan_path)
    generation_paths = sorted(
        (generation_root / "receipts" / "trials").glob("*.json")
    )
    score_paths = sorted((score_root / "trials").glob("*.json"))
    if not generation_paths or len(generation_paths) != len(score_paths):
        raise ValueError("calibration generation/score count mismatch")
    scores = {
        json.loads(path.read_text())["trial_id"]: path for path in score_paths
    }
    rows = []
    generation_hashes = []
    score_hashes = []
    run_ids = set()
    source_commits = set()
    for generation_path in generation_paths:
        receipt = validate_intervention_receipt(
            json.loads(generation_path.read_text())
        )
        if receipt.phase != "discovery_calibration":
            raise ValueError("non-calibration receipt in calibration analysis")
        if receipt.intervention_plan_sha256 != plan_sha:
            raise ValueError("calibration intervention plan hash drift")
        if receipt.trial_id not in scores:
            raise ValueError(f"missing calibration score: {receipt.trial_id}")
        score_path = scores[receipt.trial_id]
        score = json.loads(score_path.read_text())
        if score["generation_receipt_sha256"] != sha256_file(generation_path):
            raise ValueError(f"calibration score linkage drift: {receipt.trial_id}")
        steps = receipt.intervention_steps
        rows.append(
            {
                "trial_id": receipt.trial_id,
                "behavior_id": receipt.behavior_id,
                "condition_id": receipt.condition_id,
                "evaluator_score": score["yes_probability"],
                "evaluator_parse_ok": score["parse_ok"],
                "requested_alpha": receipt.requested_alpha,
                "maximum_requested_realized_relative_error": max(
                    (
                        step.requested_realized_relative_error
                        for step in steps
                    ),
                    default=0.0,
                ),
                "maximum_event_delta_to_pre_residual_norm": max(
                    (step.delta_to_pre_residual_norm for step in steps),
                    default=0.0,
                ),
                "truncated": receipt.truncated,
                "error": receipt.error,
                "generation_receipt_sha256": sha256_file(generation_path),
                "score_receipt_sha256": sha256_file(score_path),
            }
        )
        generation_hashes.append(sha256_file(generation_path))
        score_hashes.append(sha256_file(score_path))
        run_ids.add(receipt.run_id)
        source_commits.add(receipt.source_commit)
    if len(run_ids) != 1 or len(source_commits) != 1:
        raise ValueError("calibration run/source topology drift")
    result = analyze_alpha_calibration(rows, plan)
    result.update(
        {
            "intervention_plan_sha256": plan_sha,
            "public_plan_sha256": sha256_file(public_plan_path),
            "gate3_analysis_sha256": sha256_file(gate3_analysis_path),
            "generation_receipts_sha256": sha256_bytes(
                canonical_json_bytes(sorted(generation_hashes))
            ),
            "score_receipts_sha256": sha256_bytes(
                canonical_json_bytes(sorted(score_hashes))
            ),
            "run_id": next(iter(run_ids)),
            "generation_source_commit": next(iter(source_commits)),
            "analysis_source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "analysis_implementation_sha256": sha256_file(Path(__file__)),
        }
    )
    write_json_atomic(output_path, result)
    return result
