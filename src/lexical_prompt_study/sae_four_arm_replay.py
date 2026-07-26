from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_file, write_json_atomic


ARMS = ("base", "inert_length", "structural_sham", "full")
CONTRASTS = (
    ("full", "base"),
    ("full", "inert_length"),
    ("full", "structural_sham"),
    ("structural_sham", "base"),
    ("inert_length", "base"),
)


def validate_replay_plan(plan: dict[str, Any]) -> None:
    if plan["replay_id"] != "gate3-sae-four-arm-replay-v1":
        raise ValueError("wrong replay namespace")
    if plan["outcome_status"] != "base-and-inert-feature-outcomes-unopened":
        raise ValueError("replay plan is not prospectively outcome-bound")
    if tuple(plan["arms"]) != ARMS:
        raise ValueError("four-arm topology drift")
    if plan["primary_feature_id"] != 10146:
        raise ValueError("primary feature drift")
    if plan["primary_position"] != "turn-2 assistant boundary before first generated token":
        raise ValueError("primary position drift")
    if plan["hook_layer"] != 50:
        raise ValueError("SAE hook-layer drift")
    if plan["topology"]["behaviors"] != 20:
        raise ValueError("behavior count drift")
    if plan["topology"]["total_primary_observations"] != 80:
        raise ValueError("primary observation count drift")
    if plan["compute"]["requires_new_70b_forward_pass"]:
        raise ValueError("replay must not authorize a new model forward pass")
    if not plan["compute"]["paid_execution_requires_exact_cost_approval"]:
        raise ValueError("paid replay lacks an exact-cost gate")


def _id_list_hash(ids: list[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(sorted(ids))).hexdigest()


def _bootstrap_interval(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def compute_four_arm_replay(
    *,
    layer_states: np.ndarray,
    observations: list[dict[str, Any]],
    encoder_rows: np.ndarray,
    encoder_bias: np.ndarray,
    feature_ids: list[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    available = [row for row in observations if row["position_available"]]
    if len(layer_states) != len(available):
        raise ValueError("state/manifest availability mismatch")
    state_index = {row["observation_id"]: index for index, row in enumerate(available)}
    primary = [
        row
        for row in observations
        if row["position"] == "assistant_boundary" and row["position_available"]
    ]
    by_arm = {
        arm: sorted(
            (row for row in primary if row["arm"] == arm),
            key=lambda row: row["behavior_id"],
        )
        for arm in ARMS
    }
    behavior_ids = [row["behavior_id"] for row in by_arm["base"]]
    if len(behavior_ids) != 20 or len(set(behavior_ids)) != 20:
        raise ValueError("expected 20 unique base behavior IDs")
    for arm, rows in by_arm.items():
        if [row["behavior_id"] for row in rows] != behavior_ids:
            raise ValueError(f"{arm}: behavior-ID alignment drift")

    if encoder_rows.shape != (len(feature_ids), layer_states.shape[1]):
        raise ValueError("encoder row shape mismatch")
    if encoder_bias.shape != (len(feature_ids),):
        raise ValueError("encoder bias shape mismatch")
    primary_indices = {
        arm: [state_index[row["observation_id"]] for row in rows]
        for arm, rows in by_arm.items()
    }
    activations = {
        arm: np.maximum(
            0.0,
            layer_states[indices].astype(np.float32) @ encoder_rows.T + encoder_bias,
        )
        for arm, indices in primary_indices.items()
    }

    public_features = []
    private_rows = []
    for column, feature_id in enumerate(feature_ids):
        arm_summaries = {}
        for arm in ARMS:
            values = activations[arm][:, column]
            arm_summaries[arm] = {
                "count": int(len(values)),
                "positive_count": int(np.count_nonzero(values > 0)),
                "prevalence": float(np.mean(values > 0)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
            private_rows.extend(
                {
                    "behavior_id": behavior_id,
                    "arm": arm,
                    "feature_id": feature_id,
                    "activation": float(value),
                }
                for behavior_id, value in zip(behavior_ids, values, strict=True)
            )
        contrasts = []
        for contrast_index, (left, right) in enumerate(CONTRASTS):
            deltas = activations[left][:, column] - activations[right][:, column]
            contrasts.append(
                {
                    "contrast": f"{left}-minus-{right}",
                    "paired_mean": float(deltas.mean()),
                    "paired_median": float(np.median(deltas)),
                    "positive_delta_fraction": float(np.mean(deltas > 0)),
                    "bootstrap_95_interval": _bootstrap_interval(
                        deltas,
                        seed=bootstrap_seed + feature_id * 17 + contrast_index,
                        replicates=bootstrap_replicates,
                    ),
                }
            )
        public_features.append(
            {
                "feature_id": feature_id,
                "arms": arm_summaries,
                "paired_contrasts": contrasts,
            }
        )

    private_payload = {
        "schema_version": "1.0",
        "behavior_ids": behavior_ids,
        "behavior_ids_sha256": _id_list_hash(behavior_ids),
        "rows": private_rows,
    }
    public_payload = {
        "schema_version": "1.0",
        "status": "complete",
        "interpretation": "discovery-only four-arm feature fingerprint replay",
        "independent_unit": "behavior ID",
        "behavior_count": len(behavior_ids),
        "behavior_ids_sha256": private_payload["behavior_ids_sha256"],
        "features": public_features,
        "private_rows_sha256": hashlib.sha256(canonical_json_bytes(private_rows)).hexdigest(),
    }
    return public_payload, private_payload


def run_replay(
    *,
    plan_path: Path,
    captured_states_path: Path,
    observation_manifest_path: Path,
    prior_sae_discovery_path: Path,
    sae_path: Path,
    public_output_path: Path,
    private_output_path: Path,
) -> None:
    import torch

    plan = json.loads(plan_path.read_text())
    validate_replay_plan(plan)
    sources = plan["source_artifacts"]
    for path, key in (
        (captured_states_path, "captured_states_sha256"),
        (observation_manifest_path, "observation_manifest_sha256"),
        (prior_sae_discovery_path, "prior_sae_discovery_sha256"),
        (sae_path, "sae_sha256"),
    ):
        if sha256_file(path) != sources[key]:
            raise ValueError(f"{key}: source artifact hash mismatch")

    manifest = json.loads(observation_manifest_path.read_text())
    if manifest["provenance"]["sae_sha256"] != sources["sae_sha256"]:
        raise ValueError("capture/SAE provenance mismatch")
    state_payload = torch.load(
        captured_states_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if state_payload["observations_sha256"] != manifest["observations_sha256"]:
        raise ValueError("captured-state observation hash drift")
    layer_states = state_payload["states"][plan["hook_layer"]].float().numpy()

    feature_ids = [plan["primary_feature_id"]]
    feature_ids.extend(plan["secondary_feature_ids"])
    feature_ids.extend(plan["matched_control_feature_ids"])
    sae = torch.load(sae_path, map_location="cpu", weights_only=True, mmap=True)
    encoder_rows = sae["encoder_linear.weight"][feature_ids].float().numpy()
    encoder_bias = sae["encoder_linear.bias"][feature_ids].float().numpy()
    public, private = compute_four_arm_replay(
        layer_states=layer_states,
        observations=manifest["observations"],
        encoder_rows=encoder_rows,
        encoder_bias=encoder_bias,
        feature_ids=feature_ids,
        bootstrap_seed=plan["statistics"]["bootstrap_seed"],
        bootstrap_replicates=plan["statistics"]["bootstrap_replicates"],
    )

    prior = json.loads(prior_sae_discovery_path.read_text())
    prior_primary = next(
        row for row in prior["diagnostics"] if row["feature_id"] == plan["primary_feature_id"]
    )
    replay_primary = next(
        row for row in public["features"] if row["feature_id"] == plan["primary_feature_id"]
    )
    full_sham = next(
        row
        for row in replay_primary["paired_contrasts"]
        if row["contrast"] == "full-minus-structural_sham"
    )
    expected = plan["consistency_gate"]
    tolerance = expected["absolute_tolerance"]
    checks = {
        "full_prevalence": abs(
            replay_primary["arms"]["full"]["prevalence"]
            - expected["reproduce_prior_full_prevalence"]
        )
        <= tolerance,
        "sham_prevalence": abs(
            replay_primary["arms"]["structural_sham"]["prevalence"]
            - expected["reproduce_prior_sham_prevalence"]
        )
        <= tolerance,
        "full_minus_sham_mean": abs(
            full_sham["paired_mean"] - expected["reproduce_prior_full_minus_sham_mean"]
        )
        <= tolerance,
        "prior_diagnostic_agreement": abs(
            full_sham["paired_mean"] - prior_primary["paired_mean_delta"]
        )
        <= tolerance,
    }
    if not all(checks.values()):
        raise ValueError(f"prior full/sham consistency gate failed: {checks}")
    public.update(
        {
            "replay_plan_sha256": sha256_file(plan_path),
            "source_artifacts": sources,
            "consistency_gate": checks,
            "claim_boundary": plan["claim_boundary_after_replay"],
        }
    )
    private.update(
        {
            "replay_plan_sha256": sha256_file(plan_path),
            "source_artifacts": sources,
        }
    )
    write_json_atomic(private_output_path, private)
    public["private_receipt_sha256"] = sha256_file(private_output_path)
    write_json_atomic(public_output_path, public)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay four-arm SAE features from captured states")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--captured-states", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--prior-sae-discovery", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    run_replay(
        plan_path=args.plan,
        captured_states_path=args.captured_states,
        observation_manifest_path=args.observation_manifest,
        prior_sae_discovery_path=args.prior_sae_discovery,
        sae_path=args.sae,
        public_output_path=args.public_output,
        private_output_path=args.private_output,
    )


if __name__ == "__main__":
    main()
