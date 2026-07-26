from __future__ import annotations

from collections import defaultdict

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes


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
        if score is not None and not 0 <= score <= 1:
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
            "alpha": float(nonzero_rows[0]["requested_alpha"]),
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
