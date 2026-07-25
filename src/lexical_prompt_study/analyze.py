from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .hashing import sha256_file, write_json_atomic


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def analyze_behavior_gate(
    *,
    public_plan_path: Path,
    generation_root: Path,
    score_root: Path,
    output_path: Path,
    split: str,
) -> dict:
    plan = json.loads(public_plan_path.read_text())
    generation_paths = sorted((generation_root / "receipts" / "trials").glob("*.json"))
    score_paths = sorted((score_root / "trials").glob("*.json"))
    generation = {json.loads(path.read_text())["trial_id"]: path for path in generation_paths}
    scores = [json.loads(path.read_text()) for path in score_paths]
    all_rows = []
    for score, score_path in zip(scores, score_paths, strict=True):
        trial_id = score["trial_id"]
        if trial_id not in generation:
            raise ValueError(f"score without generation receipt: {trial_id}")
        if score["generation_receipt_sha256"] != sha256_file(generation[trial_id]):
            raise ValueError(f"generation receipt hash mismatch: {trial_id}")
        gen = json.loads(generation[trial_id].read_text())
        if gen["split"] == split and gen["turn"] in (1, 2):
            all_rows.append(
                {
                    "trial_id": trial_id,
                    "behavior_id": gen["behavior_id"],
                    "category": gen["category"],
                    "arm": gen["arm"],
                    "turn": gen["turn"],
                    "score": score["yes_probability"],
                    "binary": score["binary_prediction"],
                    "truncated": gen["truncated"],
                    "refusal": gen["refusal_detected"],
                    "divider": gen["divider_detected"],
                    "post_divider": gen["post_divider_present"],
                    "generation_receipt_sha256": sha256_file(generation[trial_id]),
                    "score_receipt_sha256": sha256_file(score_path),
                }
            )
    expected_behaviors = 20 if split == "discovery" else 40
    if len(all_rows) != expected_behaviors * 4 * 2:
        raise ValueError(
            f"expected {expected_behaviors * 4 * 2} phase rows, got {len(all_rows)}"
        )
    rows = [row for row in all_rows if row["turn"] == 2]
    if len(rows) != expected_behaviors * 4:
        raise ValueError(f"expected {expected_behaviors * 4} turn-2 rows, got {len(rows)}")
    by_behavior: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["arm"] in by_behavior[row["behavior_id"]]:
            raise ValueError(f"duplicate arm for {row['behavior_id']}/{row['arm']}")
        by_behavior[row["behavior_id"]][row["arm"]] = row
    required_arms = {"base", "full", "structural_sham", "inert_length"}
    if any(set(arms) != required_arms for arms in by_behavior.values()):
        raise ValueError("one or more behaviors lacks a required arm")
    behavior_ids = sorted(by_behavior)
    paired = np.array(
        [
            by_behavior[behavior_id]["full"]["score"]
            - by_behavior[behavior_id]["structural_sham"]["score"]
            for behavior_id in behavior_ids
        ]
    )
    rng = np.random.default_rng(plan["thresholds"]["bootstrap_seed"])
    replicates = int(plan["thresholds"]["bootstrap_replicates"])
    indices = rng.integers(0, len(paired), size=(replicates, len(paired)))
    bootstrap = paired[indices].mean(axis=1)
    arm_summaries = {}
    for arm in sorted(required_arms):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "mean_score": float(np.mean([row["score"] for row in arm_rows])),
            "binary_rate": float(np.mean([row["binary"] for row in arm_rows])),
            "refusal_rate": float(np.mean([row["refusal"] for row in arm_rows])),
            "divider_rate": float(np.mean([row["divider"] for row in arm_rows])),
            "post_divider_rate": float(np.mean([row["post_divider"] for row in arm_rows])),
            "truncation_rate": float(np.mean([row["truncated"] for row in arm_rows])),
        }
    mean_threshold = plan["thresholds"][
        "gate1_discovery_mean_full_minus_sham"
        if split == "discovery"
        else "gate2_confirmatory_mean_full_minus_sham"
    ]
    ci_threshold = plan["thresholds"][
        "gate1_discovery_bootstrap_ci_lower"
        if split == "discovery"
        else "gate2_confirmatory_bootstrap_ci_lower"
    ]
    ci = _percentile_interval(bootstrap)
    result = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "split": split,
        "plan_sha256": sha256_file(public_plan_path),
        "n_behaviors": len(behavior_ids),
        "n_turn2_rows": len(rows),
        "arm_summaries": arm_summaries,
        "primary_contrast": {
            "estimand": "mean paired HarmBench affirmative probability: full minus structural_sham",
            "estimate": float(paired.mean()),
            "bootstrap_95_interval": ci,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": plan["thresholds"]["bootstrap_seed"],
            "mean_threshold": mean_threshold,
            "ci_lower_threshold": ci_threshold,
            "mean_gate_passed": float(paired.mean()) >= mean_threshold,
            "ci_gate_passed": ci[0] > ci_threshold,
            "continuation_gate_passed": float(paired.mean()) >= mean_threshold
            and ci[0] > ci_threshold,
        },
        "paired_values": [
            {"behavior_id": behavior_id, "full_minus_sham": float(value)}
            for behavior_id, value in zip(behavior_ids, paired, strict=True)
        ],
        "source_receipts": rows,
        "phase_source_receipts": all_rows,
    }
    write_json_atomic(output_path, result)
    return result
