from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from .models import MechanismReceipt

TRANSPORTS = ("jacobian_lens", "identity", "random_gaussian")
POSITION_ORDER = (None, 0, 1, 2, 4, 8, 16)
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 10_000


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired_deltas(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    by_behavior: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if not row["position_available"]:
            continue
        by_behavior[row["behavior_id"]][row["arm"]] = float(
            row["margin"]["refusal_minus_compliance_margin"]
        )
    behavior_ids = sorted(
        behavior_id
        for behavior_id, arms in by_behavior.items()
        if "full" in arms and "structural_sham" in arms
    )
    values = np.array(
        [
            by_behavior[behavior_id]["full"]
            - by_behavior[behavior_id]["structural_sham"]
            for behavior_id in behavior_ids
        ],
        dtype=np.float64,
    )
    return behavior_ids, values


def _matched_controls(
    diagnostics: list[dict[str, Any]],
    selected_feature_ids: list[int],
    count: int = 3,
) -> list[dict[str, Any]]:
    primary_id = selected_feature_ids[0]
    by_id = {int(item["feature_id"]): item for item in diagnostics}
    target = by_id[primary_id]
    candidates = []
    for item in diagnostics:
        feature_id = int(item["feature_id"])
        if (
            feature_id in selected_feature_ids
            or float(item["decoder_norm"]) <= 0
            or float(item["all_prevalence"]) <= 0
            or abs(float(item["paired_standardized_delta"])) > 0.20
        ):
            continue
        distance = abs(
            math.log(float(item["decoder_norm"]) / float(target["decoder_norm"]))
        ) + 4.0 * abs(
            float(item["all_prevalence"]) - float(target["all_prevalence"])
        )
        candidates.append((distance, feature_id, item))
    if len(candidates) < count:
        raise ValueError("insufficient eligible norm/frequency-matched SAE controls")
    selected = sorted(candidates, key=lambda value: (value[0], value[1]))[:count]
    return [
        {
            "feature_id": feature_id,
            "distance": float(distance),
            "decoder_norm": float(item["decoder_norm"]),
            "all_prevalence": float(item["all_prevalence"]),
            "paired_standardized_delta": float(item["paired_standardized_delta"]),
            "eligibility": (
                "positive decoder norm and prevalence; absolute discovery "
                "standardized delta <= 0.20"
            ),
        }
        for distance, feature_id, item in selected
    ]


def analyze_mechanisms(input_root: Path, output_path: Path) -> dict[str, Any]:
    summary_path = input_root / "summary.json"
    moment_path = input_root / "vocabulary-moment-validation.json"
    sae_path = input_root / "sae-discovery.json"
    manifest_path = input_root / "observation-manifest.json"
    run_summary = json.loads(summary_path.read_text())
    moment = json.loads(moment_path.read_text())
    sae = json.loads(sae_path.read_text())
    layer_paths = sorted((input_root / "layers").glob("layer-*.json"))
    expected_layers = list(range(79))
    if [int(path.stem.split("-")[1]) for path in layer_paths] != expected_layers:
        raise ValueError("mechanism layer topology mismatch")
    all_rows: list[dict[str, Any]] = []
    layer_hashes = []
    maximum_random_norm_relative_error = 0.0
    for path in layer_paths:
        payload = json.loads(path.read_text())
        layer = int(path.stem.split("-")[1])
        if (
            payload["layer"] != layer
            or payload["run_id"] != run_summary["run_id"]
            or payload["source_commit"] != run_summary["source_commit"]
            or payload["public_plan_sha256"] != run_summary["public_plan_sha256"]
        ):
            raise ValueError(f"layer {layer}: provenance drift")
        if len(payload["receipts"]) != run_summary["observations"] * len(TRANSPORTS):
            raise ValueError(f"layer {layer}: receipt count mismatch")
        seen = set()
        for raw in payload["receipts"]:
            receipt = MechanismReceipt.model_validate(raw)
            key = (receipt.observation_id, receipt.transport)
            if key in seen:
                raise ValueError(f"layer {layer}: duplicate receipt {key}")
            seen.add(key)
            if receipt.transport == "random_gaussian":
                relative_error = abs(
                    receipt.realized_frobenius_norm - receipt.fitted_frobenius_norm
                ) / receipt.fitted_frobenius_norm
                maximum_random_norm_relative_error = max(
                    maximum_random_norm_relative_error, relative_error
                )
            all_rows.append(receipt.model_dump(mode="json"))
        layer_hashes.append({"path": str(path), "sha256": sha256_file(path)})

    layerwise = []
    for transport_index, transport in enumerate(TRANSPORTS):
        for layer in expected_layers:
            rows = [
                row
                for row in all_rows
                if row["transport"] == transport
                and row["layer"] == layer
                and row["position"] == "assistant_boundary"
                and row["arm"] in {"full", "structural_sham"}
            ]
            behavior_ids, deltas = _paired_deltas(rows)
            if len(behavior_ids) != 20:
                raise ValueError(
                    f"{transport} layer {layer}: expected 20 assistant-boundary pairs"
                )
            layerwise.append(
                {
                    "transport": transport,
                    "layer": layer,
                    "n_behavior_pairs": len(behavior_ids),
                    "mean_full_minus_sham_margin": float(deltas.mean()),
                    "bootstrap_95_interval": _bootstrap_interval(
                        deltas,
                        BOOTSTRAP_SEED + transport_index * 1000 + layer,
                    ),
                    "behavior_ids_sha256": sha256_bytes(
                        canonical_json_bytes(behavior_ids)
                    ),
                }
            )

    trajectory = []
    for transport in TRANSPORTS:
        for layer in expected_layers:
            for token_index in POSITION_ORDER:
                position = "assistant_boundary" if token_index is None else "generated"
                rows = [
                    row
                    for row in all_rows
                    if row["transport"] == transport
                    and row["layer"] == layer
                    and row["position"] == position
                    and row["position_token_index"] == token_index
                    and row["arm"] in {"full", "structural_sham"}
                ]
                behavior_ids, deltas = _paired_deltas(rows)
                trajectory.append(
                    {
                        "transport": transport,
                        "layer": layer,
                        "position": position,
                        "position_token_index": token_index,
                        "n_behavior_pairs": len(behavior_ids),
                        "mean_full_minus_sham_margin": (
                            float(deltas.mean()) if deltas.size else None
                        ),
                        "behavior_ids_sha256": sha256_bytes(
                            canonical_json_bytes(behavior_ids)
                        ),
                    }
                )

    diagnostics = sae["diagnostics"]
    selected_ids = [int(value) for value in sae["selected_feature_ids"]]
    diagnostic_by_id = {int(item["feature_id"]): item for item in diagnostics}
    selected = [diagnostic_by_id[feature_id] for feature_id in selected_ids]
    if not all(
        float(item["decoder_norm"]) > 0
        and float(item["full_prevalence"]) >= 0.10
        and float(item["paired_mean_delta"]) > 0
        and float(item["paired_standardized_delta"]) > 0
        for item in selected
    ):
        raise ValueError("selected SAE candidate violates frozen eligibility")
    controls = _matched_controls(diagnostics, selected_ids)

    identity = np.array(
        [
            item["mean_full_minus_sham_margin"]
            for item in layerwise
            if item["transport"] == "identity"
        ]
    )
    fitted = np.array(
        [
            item["mean_full_minus_sham_margin"]
            for item in layerwise
            if item["transport"] == "jacobian_lens"
        ]
    )
    random = np.array(
        [
            item["mean_full_minus_sham_margin"]
            for item in layerwise
            if item["transport"] == "random_gaussian"
        ]
    )
    result = {
        "schema_version": "1.0",
        "study_id": run_summary["study_id"],
        "split": "discovery",
        "status": "complete",
        "interpretation": "descriptive readout and discovery-only SAE selection",
        "source_commit": run_summary["source_commit"],
        "analysis_source_commit": _source_commit(),
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "public_plan_sha256": run_summary["public_plan_sha256"],
        "model_revision": run_summary["model_revision"],
        "tokenizer_revision": run_summary["tokenizer_revision"],
        "lens_sha256": run_summary["lens_sha256"],
        "sae_sha256": run_summary["sae_sha256"],
        "source_files": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "moment_validation": {
                "path": str(moment_path),
                "sha256": sha256_file(moment_path),
            },
            "sae_discovery": {"path": str(sae_path), "sha256": sha256_file(sae_path)},
            "observation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "layers": layer_hashes,
        },
        "counts": {
            "layers": len(layer_paths),
            "receipt_rows": len(all_rows),
            "scheduled_observations": run_summary["observations"],
            "available_observations": run_summary["available_observations"],
            "missing_observations": (
                run_summary["observations"] - run_summary["available_observations"]
            ),
            "sae_features": len(diagnostics),
        },
        "validity_gates": {
            "exact_layer_topology": True,
            "exact_artifact_and_plan_bindings": True,
            "complete_receipt_topology": True,
            "vocabulary_moment_analytic_pass": (
                max(
                    moment["mean_abs_error"],
                    moment["std_abs_error"],
                    moment["probe_max_abs_error"],
                )
                <= moment["analytic_tolerance"]
            ),
            "vocabulary_moment_module_pass": (
                moment["module_probe_max_abs_error"] <= moment["module_tolerance"]
            ),
            "maximum_random_transport_norm_relative_error": (
                maximum_random_norm_relative_error
            ),
            "published_same_checkpoint_fixture": "not available; not treated as passed",
        },
        "layerwise_primary": {
            "position": "turn-2 assistant boundary",
            "independent_unit": "behavior ID",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "rows": layerwise,
            "curve_diagnostics": {
                "fitted_vs_identity_rmse": float(np.sqrt(np.mean((fitted - identity) ** 2))),
                "fitted_vs_random_rmse": float(np.sqrt(np.mean((fitted - random) ** 2))),
                "fitted_identity_correlation": float(np.corrcoef(fitted, identity)[0, 1]),
                "fitted_random_correlation": float(np.corrcoef(fitted, random)[0, 1]),
            },
        },
        "trajectory": {
            "position_semantics": (
                "assistant boundary or generated index k after including token IDs 0..k"
            ),
            "rows": trajectory,
        },
        "sae_discovery": {
            "hook_layer": sae["hook_layer"],
            "selection_rule": sae["selection_rule"],
            "selected_feature_ids": selected_ids,
            "selected_diagnostics": selected,
            "matched_control_rule": {
                "candidate_filter": (
                    "not selected; positive decoder norm and prevalence; "
                    "absolute paired standardized delta <= 0.20"
                ),
                "distance": (
                    "abs(log(decoder_norm/primary_norm)) + "
                    "4*abs(all_prevalence-primary_prevalence)"
                ),
                "tie_break": "ascending feature ID",
            },
            "matched_controls": controls,
            "activation_sparsity": sae["activation_sparsity"],
            "reconstruction_relative_error": sae["reconstruction_relative_error"],
        },
        "non_claims": [
            "descriptive J-lens differences are not causal mechanisms",
            "SAE feature labels are not mechanistic evidence",
            "candidate and matched-control selection are discovery-only",
            "no held-out intervention outcome has been opened",
        ],
    }
    result["gate3_artifact_sha256"] = sha256_bytes(canonical_json_bytes(result))
    write_json_atomic(output_path, result)
    return result
