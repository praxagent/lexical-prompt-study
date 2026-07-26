from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import EVALUATOR_REVISION
from .followup_design import PLACEMENTS
from .followup_plan import validate_followup_plan
from .hashing import sha256_file, write_json_atomic
from .models import FollowupTrialReceipt


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(axis=1)


def analyze_followup_behavior_discovery(
    *,
    public_plan_path: Path,
    generation_root: Path,
    score_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(plan)
    rule = plan["placement_factor"]["behavioral_family"]["discovery"]
    generation_binding = plan["compute"]["scientific_runs"]["g2_discovery"][
        "result_binding"
    ]
    replicates = int(rule["bootstrap_replicates"])
    seed = int(rule["bootstrap_seed"])

    generation_paths = sorted((generation_root / "receipts" / "trials").glob("*.json"))
    score_paths = sorted((score_root / "trials").glob("*.json"))
    if len(generation_paths) != 140 or len(score_paths) != 140:
        raise ValueError(
            "follow-up discovery requires 140 generations and scores, got "
            f"{len(generation_paths)} and {len(score_paths)}"
        )
    generations: dict[str, tuple[Path, FollowupTrialReceipt]] = {}
    for path in generation_paths:
        receipt = FollowupTrialReceipt.model_validate_json(path.read_text())
        if receipt.partition != "discovery":
            raise ValueError(f"{path}: non-discovery generation receipt")
        if (
            receipt.source_commit != generation_binding["source_commit"]
            or receipt.plan_sha256 != generation_binding["public_plan_sha256"]
            or receipt.private_plan_sha256
            != generation_binding["private_plan_sha256"]
            or receipt.run_id != generation_binding["run_id"]
        ):
            raise ValueError(f"{path}: frozen generation provenance drift")
        if receipt.trial_id in generations:
            raise ValueError(f"duplicate generation trial {receipt.trial_id}")
        generations[receipt.trial_id] = (path, receipt)

    rows = []
    seen_scores = set()
    for score_path in score_paths:
        score = json.loads(score_path.read_text())
        trial_id = score["trial_id"]
        if trial_id in seen_scores:
            raise ValueError(f"duplicate score trial {trial_id}")
        seen_scores.add(trial_id)
        if trial_id not in generations:
            raise ValueError(f"score without generation trial {trial_id}")
        generation_path, generation = generations[trial_id]
        if score["generation_receipt_sha256"] != sha256_file(generation_path):
            raise ValueError(f"{trial_id}: generation receipt hash mismatch")
        if score["evaluator_revision"] != EVALUATOR_REVISION:
            raise ValueError(f"{trial_id}: evaluator revision drift")
        yes = float(score["yes_probability"])
        no = float(score["no_probability"])
        if (
            not score["parse_ok"]
            or not math.isfinite(yes)
            or not math.isfinite(no)
            or abs(yes + no - 1.0) > 1e-5
        ):
            raise ValueError(f"{trial_id}: invalid HarmBench score")
        rows.append(
            {
                "trial_id": trial_id,
                "behavior_id": generation.behavior_id,
                "category": generation.category,
                "arm": generation.arm,
                "placement": generation.placement,
                "shared_reference": generation.shared_reference,
                "score": yes,
                "binary_prediction": bool(score["binary_prediction"]),
                "truncated": generation.truncated,
                "refusal_detected": generation.refusal_detected,
                "generation_receipt_sha256": sha256_file(generation_path),
                "score_receipt_sha256": sha256_file(score_path),
            }
        )
    if set(generations) != seen_scores:
        raise ValueError("generation/score trial sets differ")

    topology = Counter((row["arm"], row["placement"] or "shared") for row in rows)
    expected_topology = {("base", "shared"): 20}
    expected_topology.update(
        {
            (arm, placement): 20
            for arm in ("inert_length", "structural_sham", "full")
            for placement in PLACEMENTS
        }
    )
    if topology != expected_topology:
        raise ValueError(f"follow-up discovery topology drift: {topology}")

    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["placement"] is None:
            continue
        stratum = indexed[row["placement"]][row["behavior_id"]]
        if row["arm"] in stratum:
            raise ValueError("duplicate behavior/placement/arm row")
        stratum[row["arm"]] = row

    ordering_results = {}
    paired_by_order = {}
    for order_index, placement in enumerate(PLACEMENTS):
        by_behavior = indexed[placement]
        if len(by_behavior) != 20 or any(
            set(arms) != {"inert_length", "structural_sham", "full"}
            for arms in by_behavior.values()
        ):
            raise ValueError(f"{placement}: incomplete ordering-specific arms")
        behavior_ids = sorted(by_behavior)
        paired = np.asarray(
            [
                by_behavior[behavior_id]["full"]["score"]
                - by_behavior[behavior_id]["structural_sham"]["score"]
                for behavior_id in behavior_ids
            ],
            dtype=np.float64,
        )
        paired_by_order[placement] = (behavior_ids, paired)
        interval = _interval(
            _bootstrap_mean(paired, replicates, seed + order_index)
        )
        estimate = float(paired.mean())
        arm_summaries = {}
        for arm in ("inert_length", "structural_sham", "full"):
            arm_rows = [by_behavior[behavior_id][arm] for behavior_id in behavior_ids]
            arm_summaries[arm] = {
                "mean_score": float(np.mean([row["score"] for row in arm_rows])),
                "binary_rate": float(
                    np.mean([row["binary_prediction"] for row in arm_rows])
                ),
                "refusal_rate": float(
                    np.mean([row["refusal_detected"] for row in arm_rows])
                ),
                "truncation_rate": float(
                    np.mean([row["truncated"] for row in arm_rows])
                ),
            }
        mean_passed = estimate >= float(rule["minimum_mean_paired_effect"])
        interval_passed = interval[0] > float(rule["bootstrap_lower_bound_above"])
        ordering_results[placement] = {
            "n_behavior_pairs": len(behavior_ids),
            "arm_summaries": arm_summaries,
            "full_minus_structural_sham": {
                "estimate": estimate,
                "bootstrap_95_interval": interval,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed + order_index,
                "minimum_mean_paired_effect": rule["minimum_mean_paired_effect"],
                "bootstrap_lower_bound_above": rule[
                    "bootstrap_lower_bound_above"
                ],
                "mean_gate_passed": mean_passed,
                "interval_gate_passed": interval_passed,
                "continuation_gate_passed": mean_passed and interval_passed,
            },
            "paired_values": [
                {"behavior_id": behavior_id, "full_minus_sham": float(value)}
                for behavior_id, value in zip(behavior_ids, paired, strict=True)
            ],
        }

    before_ids, before = paired_by_order["ep_before_request"]
    after_ids, after = paired_by_order["ep_after_request"]
    if before_ids != after_ids:
        raise ValueError("placement interaction behavior IDs differ")
    interaction = before - after
    result = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "partition": "discovery",
        "status": "complete",
        "plan_sha256": sha256_file(public_plan_path),
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "source_commit": _source_commit(),
        "n_generation_receipts": len(generation_paths),
        "n_score_receipts": len(score_paths),
        "pooled_estimate_reported": False,
        "ordering_results": ordering_results,
        "placement_interaction": {
            "estimand": (
                "paired difference: ep_before_request effect minus "
                "ep_after_request effect"
            ),
            "estimate": float(interaction.mean()),
            "bootstrap_95_interval": _interval(
                _bootstrap_mean(interaction, replicates, seed + len(PLACEMENTS))
            ),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed + len(PLACEMENTS),
            "descriptive_only": True,
            "equivalence_or_no_moderation_claim": False,
        },
        "both_orderings_continuation_gate_passed": all(
            ordering_results[placement]["full_minus_structural_sham"][
                "continuation_gate_passed"
            ]
            for placement in PLACEMENTS
        ),
        "shared_base_summary": {
            "n": 20,
            "mean_score": float(
                np.mean(
                    [
                        row["score"]
                        for row in rows
                        if row["arm"] == "base" and row["placement"] is None
                    ]
                )
            ),
        },
        "source_receipts": rows,
    }
    write_json_atomic(output_path, result)
    return result


def _sign_flip_p_value(values: np.ndarray, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null_statistics = np.abs((signs * values).mean(axis=1))
    observed = abs(float(values.mean()))
    return float((np.count_nonzero(null_statistics >= observed) + 1) / (draws + 1))


def _holm_adjusted(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(
        PLACEMENTS,
        key=lambda placement: (raw[placement], PLACEMENTS.index(placement)),
    )
    adjusted = {}
    running = 0.0
    family_size = len(ordered)
    for rank, placement in enumerate(ordered):
        value = min(1.0, (family_size - rank) * raw[placement])
        running = max(running, value)
        adjusted[placement] = running
    return adjusted


def analyze_followup_behavior_calibration(
    *,
    public_plan_path: Path,
    generation_root: Path,
    score_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(plan)
    rule = plan["placement_factor"]["behavioral_family"]["calibration"]
    generation_binding = plan["compute"]["scientific_runs"][
        "g2_calibration_generation"
    ]["result_binding"]
    generation_paths = sorted((generation_root / "receipts" / "trials").glob("*.json"))
    score_paths = sorted((score_root / "trials").glob("*.json"))
    if len(generation_paths) != 140 or len(score_paths) != 140:
        raise ValueError(
            "follow-up calibration requires 140 generations and scores, got "
            f"{len(generation_paths)} and {len(score_paths)}"
        )

    generations: dict[str, tuple[Path, FollowupTrialReceipt]] = {}
    provenance = set()
    for path in generation_paths:
        receipt = FollowupTrialReceipt.model_validate_json(path.read_text())
        if receipt.partition != "calibration":
            raise ValueError(f"{path}: non-calibration generation receipt")
        if (
            receipt.source_commit != generation_binding["source_commit"]
            or receipt.plan_sha256 != generation_binding["public_plan_sha256"]
            or receipt.private_plan_sha256
            != generation_binding["private_plan_sha256"]
            or receipt.run_id != generation_binding["run_id"]
        ):
            raise ValueError(f"{path}: frozen calibration provenance drift")
        if receipt.trial_id in generations:
            raise ValueError(f"duplicate generation trial {receipt.trial_id}")
        provenance.add(
            (
                receipt.source_commit,
                receipt.plan_sha256,
                receipt.private_plan_sha256,
                receipt.run_id,
            )
        )
        generations[receipt.trial_id] = (path, receipt)
    if len(provenance) != 1:
        raise ValueError("calibration generation provenance is not singular")

    rows = []
    seen_scores = set()
    for score_path in score_paths:
        score = json.loads(score_path.read_text())
        trial_id = score["trial_id"]
        if trial_id in seen_scores or trial_id not in generations:
            raise ValueError(f"invalid or duplicate calibration score {trial_id}")
        seen_scores.add(trial_id)
        generation_path, generation = generations[trial_id]
        if score["generation_receipt_sha256"] != sha256_file(generation_path):
            raise ValueError(f"{trial_id}: generation receipt hash mismatch")
        if score["evaluator_revision"] != EVALUATOR_REVISION:
            raise ValueError(f"{trial_id}: evaluator revision drift")
        yes = float(score["yes_probability"])
        no = float(score["no_probability"])
        if (
            not score["parse_ok"]
            or not math.isfinite(yes)
            or not math.isfinite(no)
            or abs(yes + no - 1.0) > 1e-5
        ):
            raise ValueError(f"{trial_id}: invalid HarmBench score")
        rows.append(
            {
                "trial_id": trial_id,
                "behavior_id": generation.behavior_id,
                "category": generation.category,
                "arm": generation.arm,
                "placement": generation.placement,
                "shared_reference": generation.shared_reference,
                "score": yes,
                "binary_prediction": bool(score["binary_prediction"]),
                "truncated": generation.truncated,
                "refusal_detected": generation.refusal_detected,
                "generation_receipt_sha256": sha256_file(generation_path),
                "score_receipt_sha256": sha256_file(score_path),
            }
        )
    if set(generations) != seen_scores:
        raise ValueError("calibration generation/score trial sets differ")
    topology = Counter((row["arm"], row["placement"] or "shared") for row in rows)
    expected_topology = {("base", "shared"): 20}
    expected_topology.update(
        {
            (arm, placement): 20
            for arm in ("inert_length", "structural_sham", "full")
            for placement in PLACEMENTS
        }
    )
    if topology != expected_topology:
        raise ValueError(f"follow-up calibration topology drift: {topology}")

    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["placement"] is not None:
            indexed[row["placement"]][row["behavior_id"]][row["arm"]] = row
    paired_by_order = {}
    raw_p_values = {}
    for order_index, placement in enumerate(PLACEMENTS):
        by_behavior = indexed[placement]
        if len(by_behavior) != 20 or any(
            set(arms) != {"inert_length", "structural_sham", "full"}
            for arms in by_behavior.values()
        ):
            raise ValueError(f"{placement}: incomplete calibration arms")
        behavior_ids = sorted(by_behavior)
        paired = np.asarray(
            [
                by_behavior[behavior_id]["full"]["score"]
                - by_behavior[behavior_id]["structural_sham"]["score"]
                for behavior_id in behavior_ids
            ],
            dtype=np.float64,
        )
        paired_by_order[placement] = (behavior_ids, paired)
        raw_p_values[placement] = _sign_flip_p_value(
            paired,
            int(rule["randomization_draws"]),
            int(rule["randomization_seed"]) + order_index,
        )
    adjusted_p_values = _holm_adjusted(raw_p_values)

    ordering_results = {}
    for order_index, placement in enumerate(PLACEMENTS):
        behavior_ids, paired = paired_by_order[placement]
        estimate = float(paired.mean())
        positive_fraction = float(np.mean(paired > 0))
        mean_passed = estimate >= float(rule["minimum_mean_paired_effect"])
        sign_passed = positive_fraction >= float(
            rule["minimum_positive_sign_fraction"]
        )
        holm_passed = adjusted_p_values[placement] < float(rule["family_alpha"])
        ordering_results[placement] = {
            "n_behavior_pairs": len(behavior_ids),
            "full_minus_structural_sham": {
                "estimate": estimate,
                "positive_sign_fraction": positive_fraction,
                "minimum_mean_paired_effect": rule["minimum_mean_paired_effect"],
                "minimum_positive_sign_fraction": rule[
                    "minimum_positive_sign_fraction"
                ],
                "randomization_draws": rule["randomization_draws"],
                "randomization_seed": int(rule["randomization_seed"]) + order_index,
                "raw_p_value": raw_p_values[placement],
                "holm_adjusted_p_value": adjusted_p_values[placement],
                "family_alpha": rule["family_alpha"],
                "mean_gate_passed": mean_passed,
                "sign_gate_passed": sign_passed,
                "holm_gate_passed": holm_passed,
                "continuation_gate_passed": mean_passed
                and sign_passed
                and holm_passed,
            },
            "paired_values": [
                {"behavior_id": behavior_id, "full_minus_sham": float(value)}
                for behavior_id, value in zip(behavior_ids, paired, strict=True)
            ],
        }

    before_ids, before = paired_by_order["ep_before_request"]
    after_ids, after = paired_by_order["ep_after_request"]
    if before_ids != after_ids:
        raise ValueError("calibration placement behavior IDs differ")
    interaction = before - after
    result = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "partition": "calibration",
        "status": "complete",
        "plan_sha256": sha256_file(public_plan_path),
        "generation_provenance": {
            "source_commit": next(iter(provenance))[0],
            "public_plan_sha256": next(iter(provenance))[1],
            "private_plan_sha256": next(iter(provenance))[2],
            "run_id": next(iter(provenance))[3],
        },
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "source_commit": _source_commit(),
        "n_generation_receipts": len(generation_paths),
        "n_score_receipts": len(score_paths),
        "pooled_estimate_reported": False,
        "ordering_results": ordering_results,
        "placement_interaction": {
            "estimand": (
                "paired difference: ep_before_request effect minus "
                "ep_after_request effect"
            ),
            "estimate": float(interaction.mean()),
            "bootstrap_95_interval": _interval(
                _bootstrap_mean(
                    interaction,
                    int(rule["interaction_bootstrap_replicates"]),
                    int(rule["interaction_bootstrap_seed"]),
                )
            ),
            "bootstrap_replicates": rule["interaction_bootstrap_replicates"],
            "bootstrap_seed": rule["interaction_bootstrap_seed"],
            "descriptive_only": True,
            "equivalence_or_no_moderation_claim": False,
        },
        "both_orderings_continuation_gate_passed": all(
            ordering_results[placement]["full_minus_structural_sham"][
                "continuation_gate_passed"
            ]
            for placement in PLACEMENTS
        ),
        "source_receipts": rows,
    }
    write_json_atomic(output_path, result)
    return result
