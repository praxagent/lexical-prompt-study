from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .factorial_runner import factorial_receipt_manifest_sha256
from .hashing import sha256_file, write_json_atomic
from .models import FactorialTrialReceipt


REQUEST_CLASSES = [
    "minimal_neutral_carrier",
    "ordinary_benign_request",
    "harmful_request",
]
MATERIALS = ["inert_length", "structural_sham", "full_scaffold"]
PLACEMENTS = ["ep_before_request", "ep_after_request"]
SIZE_IDS = ["blocks-001", "blocks-002", "blocks-003", "blocks-004"]
TOKEN_COUNTS = [64, 128, 188, 252]
CLAIM_LIMITS = {
    "secondary_descriptive_not_confirmatory",
    "fixed_model_sae_materials_requests_and_assistant_boundary_only",
    "no_harmful_request_detector_claim",
    "no_request_independence_claim",
    "no_monotonic_or_semantic_component_claim",
    "no_causal_mechanism_claim",
    "no_threshold_fit_or_deployment_claim",
}
PARTIAL_SIZE_IDS = SIZE_IDS[:3]
PARTIAL_TOKEN_COUNTS = dict(zip(PARTIAL_SIZE_IDS, TOKEN_COUNTS[:3], strict=True))
ALL_TOKEN_COUNTS = dict(zip(SIZE_IDS, TOKEN_COUNTS, strict=True))
METRICS = [
    "feature_6779_magnitude",
    "frozen_subspace_score",
    "assistant_boundary_jlens_margin",
    "refusal_detected",
]
CONTRASTS = [
    ("full_content_increment", "full_scaffold", "structural_sham"),
    ("sham_over_inert_increment", "structural_sham", "inert_length"),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_factorial_dose_analysis_plan(plan: dict[str, Any]) -> None:
    _require(plan["schema_version"] == "1.0", "dose-analysis schema drift")
    _require(
        plan["study_id"] == "lexical-scaffold-8b-factorial-v1"
        and plan["analysis_id"] == "factorial-8b-secondary-dose-v1"
        and plan["status"] == "prospective_before_secondary_dose_outcomes",
        "dose-analysis identity or prospective status drift",
    )
    bindings = plan["bindings"]
    for field in (
        "public_plan_sha256",
        "private_plan_sha256",
        "canonical_result_sha256",
        "canonical_execution_receipt_sha256",
        "canonical_matrix_receipt_manifest_sha256",
        "dose_authorization_sha256",
        "dose_observation_manifest_sha256",
    ):
        _require(_is_sha256(bindings[field]), f"dose-analysis {field} drift")
    _require(
        bindings["dose_execution_receipt_sha256"] is None,
        "dose outcome was bound before prospective plan freeze",
    )
    inputs = plan["inputs"]
    _require(
        inputs["new_partial_dose_receipts"] == 540
        and inputs["reused_canonical_receipts"] == 180
        and inputs["canonical_regeneration_forbidden"] is True
        and inputs["literal_sentinels_excluded"] is True
        and inputs["held_out_confirmation_excluded"] is True
        and inputs["raw_prompt_or_generation_text_opening_forbidden"] is True,
        "dose-analysis input boundary drift",
    )
    factors = plan["factors"]
    _require(
        factors["request_classes"] == REQUEST_CLASSES
        and factors["materials"] == MATERIALS
        and factors["placements"] == PLACEMENTS
        and factors["size_ids"] == SIZE_IDS
        and factors["injected_token_counts"] == TOKEN_COUNTS
        and factors["independent_unit"] == "prompt_family_id"
        and factors["expected_units_per_cell"] == 10
        and factors["placement_pooling"] is False
        and factors["size_pooling"] is False,
        "dose-analysis factor topology drift",
    )
    readouts = plan["readouts"]
    _require(
        readouts["primary"] == "feature_6779_magnitude"
        and readouts["strict_positive_prevalence"] == "descriptive_only"
        and readouts["secondary"]
        == [
            "frozen_subspace_score",
            "assistant_boundary_jlens_margin",
            "refusal_detected",
        ],
        "dose-analysis readout drift",
    )
    contrasts = plan["paired_contrasts_at_each_request_class_placement_and_size"]
    _require(
        contrasts
        == [
            {
                "name": "full_content_increment",
                "left": "full_scaffold",
                "right": "structural_sham",
            },
            {
                "name": "sham_over_inert_increment",
                "left": "structural_sham",
                "right": "inert_length",
            },
        ],
        "dose-analysis paired contrast drift",
    )
    uncertainty = plan["uncertainty"]
    _require(
        uncertainty["method"] == "paired_prompt_family_nonparametric_bootstrap"
        and uncertainty["replicates"] == 10000
        and uncertainty["master_seed"] == 20260728
        and uncertainty["interval"] == "two_sided_95_percent_percentile"
        and uncertainty["simultaneous_or_familywise_claim"] is False
        and uncertainty["p_values"] is False
        and uncertainty["formal_pass_fail_decision"] is False,
        "dose-analysis uncertainty drift",
    )
    shape = plan["dose_shape_policy"]
    _require(
        shape["plot_all_four_realized_token_counts"] is True
        and shape["monotonicity_test"] is False
        and shape["monotonicity_claim_forbidden"] is True
        and shape["linear_dose_response_claim_forbidden"] is True
        and shape["semantic_component_ablation_claim_forbidden"] is True,
        "dose-shape claim boundary drift",
    )
    missingness = plan["missingness"]
    _require(
        missingness["two_deterministic_attempts_per_new_observation"] is True
        and missingness["complete_case_substitution"] is False
        and missingness["imputation"] is False,
        "dose-analysis missingness drift",
    )
    _require(
        set(plan["claim_limits"]) == CLAIM_LIMITS,
        "dose-analysis claim limits drift",
    )


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _validate_restricted_artifact(
    receipt: FactorialTrialReceipt,
    restricted_root: Path,
) -> None:
    artifact = restricted_root / Path(receipt.restricted_artifact_path).name
    _require(artifact.is_file(), f"{receipt.trial_id}: restricted artifact missing")
    _require(
        sha256_file(artifact) == receipt.restricted_artifact_sha256,
        f"{receipt.trial_id}: restricted artifact hash drift",
    )


def _load_receipts(
    *,
    receipt_root: Path,
    restricted_root: Path,
    expected_count: int,
    expected_manifest_sha256: str,
    expected_provenance: dict[str, str],
) -> list[FactorialTrialReceipt]:
    paths = sorted(receipt_root.glob("*.json"))
    _require(len(paths) == expected_count, "dose receipt count drift")
    _require(
        factorial_receipt_manifest_sha256(receipt_root)
        == expected_manifest_sha256,
        "dose receipt manifest drift",
    )
    receipts = []
    for path in paths:
        receipt = FactorialTrialReceipt.model_validate_json(path.read_text())
        _require(path.stem == receipt.trial_id, "dose receipt filename drift")
        for field, expected in expected_provenance.items():
            _require(
                getattr(receipt, field) == expected,
                f"{receipt.trial_id}: {field} drift",
            )
        _require(
            receipt.request_class != "literal_sentinel"
            and receipt.task_completion is None
            and receipt.utility_score is None,
            f"{receipt.trial_id}: dose receipt scope drift",
        )
        _require(
            receipt.feature_6779_positive
            == (receipt.feature_6779_magnitude > 0),
            f"{receipt.trial_id}: feature prevalence drift",
        )
        _validate_restricted_artifact(receipt, restricted_root)
        receipts.append(receipt)
    _require(
        len({receipt.trial_id for receipt in receipts}) == expected_count,
        "duplicate dose trial ID",
    )
    return receipts


def _validate_execution_receipt(
    *,
    execution: dict[str, Any],
    execution_path: Path,
    plan: dict[str, Any],
    public_plan_path: Path,
    private_plan_path: Path,
    authorization_path: Path,
    summary_path: Path,
) -> None:
    _require(
        execution["schema_version"] == "1.0"
        and execution["study_id"] == plan["study_id"]
        and execution["analysis_id"] == plan["analysis_id"]
        and execution["status"] == "secondary_dose_generation_complete",
        "dose execution identity drift",
    )
    _require(
        execution["authorization_sha256"] == sha256_file(authorization_path)
        == plan["bindings"]["dose_authorization_sha256"],
        "dose execution authorization drift",
    )
    _require(
        execution["public_plan_sha256"] == sha256_file(public_plan_path)
        == plan["bindings"]["public_plan_sha256"]
        and execution["private_plan_sha256"] == sha256_file(private_plan_path)
        == plan["bindings"]["private_plan_sha256"],
        "dose execution plan binding drift",
    )
    _require(
        execution["summary_sha256"] == sha256_file(summary_path),
        "dose execution summary drift",
    )
    _require(
        execution["dose_observation_manifest_sha256"]
        == plan["bindings"]["dose_observation_manifest_sha256"]
        and execution["canonical_result_sha256"]
        == plan["bindings"]["canonical_result_sha256"]
        and execution["canonical_execution_receipt_sha256"]
        == plan["bindings"]["canonical_execution_receipt_sha256"]
        and execution["canonical_matrix_receipt_manifest_sha256"]
        == plan["bindings"]["canonical_matrix_receipt_manifest_sha256"],
        "dose execution predecessor binding drift",
    )
    completion = execution["completion"]
    _require(
        completion
        == {
            "planned_observations": 540,
            "final_receipt_count": 540,
            "missing_after_two_attempts": 0,
            "failure_receipt_count": 0,
            "canonical_observations_regenerated": 0,
        },
        "dose execution completion drift",
    )
    boundaries = execution["analysis_boundaries_at_binding"]
    _require(
        boundaries["receipt_outcome_analysis_started"] is False
        and boundaries["raw_prompt_or_generated_text_opened"] is False
        and boundaries["placement_pooling"] is False
        and boundaries["size_pooling"] is False
        and boundaries["detector_threshold_fit"] is False
        and boundaries["held_out_confirmation_opened"] is False,
        "dose execution outcome boundary drift",
    )
    _require(_is_sha256(sha256_file(execution_path)), "dose execution receipt hash drift")


def _validate_partial_topology(
    receipts: list[FactorialTrialReceipt],
) -> dict[str, list[str]]:
    topology = Counter(
        (
            row.request_class,
            row.placement,
            row.material,
            row.size_id,
            row.injected_token_count,
        )
        for row in receipts
    )
    expected = Counter(
        {
            (request_class, placement, material, size_id, token_count): 10
            for request_class in REQUEST_CLASSES
            for placement in PLACEMENTS
            for material in MATERIALS
            for size_id, token_count in PARTIAL_TOKEN_COUNTS.items()
        }
    )
    _require(topology == expected, "secondary-dose partial topology drift")
    by_family: dict[tuple[str, str], list[FactorialTrialReceipt]] = defaultdict(list)
    for row in receipts:
        by_family[(row.request_class, row.prompt_family_id)].append(row)
    _require(len(by_family) == 30, "secondary-dose prompt-family count drift")
    selected: dict[str, list[str]] = defaultdict(list)
    for (request_class, family_id), rows in by_family.items():
        _require(len(rows) == 18, f"{family_id}: partial family incomplete")
        _require(
            len({row.request_id for row in rows}) == 1
            and len({row.request_sha256 for row in rows}) == 1,
            f"{family_id}: request identity drift",
        )
        for placement in PLACEMENTS:
            for size_id, token_count in PARTIAL_TOKEN_COUNTS.items():
                matched = [
                    row
                    for row in rows
                    if row.placement == placement and row.size_id == size_id
                ]
                _require(
                    len(matched) == 3
                    and {row.material for row in matched} == set(MATERIALS)
                    and {row.injected_token_count for row in matched} == {token_count}
                    and len({row.render_group_sha256 for row in matched}) == 1,
                    f"{family_id}/{placement}/{size_id}: matching drift",
                )
        selected[request_class].append(family_id)
    return {
        request_class: sorted(family_ids)
        for request_class, family_ids in selected.items()
    }


def _select_canonical_receipts(
    receipts: list[FactorialTrialReceipt],
    selected_families: dict[str, list[str]],
) -> list[FactorialTrialReceipt]:
    selected = [
        row
        for row in receipts
        if row.request_class in REQUEST_CLASSES
        and row.prompt_family_id in selected_families[row.request_class]
        and row.material in MATERIALS
    ]
    topology = Counter(
        (row.request_class, row.placement, row.material, row.size_id)
        for row in selected
    )
    expected = Counter(
        {
            (request_class, placement, material, "blocks-004"): 10
            for request_class in REQUEST_CLASSES
            for placement in PLACEMENTS
            for material in MATERIALS
        }
    )
    _require(topology == expected, "canonical dose-reuse topology drift")
    _require(
        all(row.injected_token_count == 252 for row in selected),
        "canonical dose token count drift",
    )
    return selected


def _validate_complete_topology(receipts: list[FactorialTrialReceipt]) -> None:
    topology = Counter(
        (
            row.request_class,
            row.placement,
            row.material,
            row.size_id,
            row.injected_token_count,
        )
        for row in receipts
    )
    expected = Counter(
        {
            (request_class, placement, material, size_id, token_count): 10
            for request_class in REQUEST_CLASSES
            for placement in PLACEMENTS
            for material in MATERIALS
            for size_id, token_count in ALL_TOKEN_COUNTS.items()
        }
    )
    _require(topology == expected, "complete secondary-dose topology drift")


def _metric_value(receipt: FactorialTrialReceipt, field: str) -> float:
    value = getattr(receipt, field)
    _require(value is not None, f"{receipt.trial_id}: missing {field}")
    return float(value)


def _cell_summaries(receipts: list[FactorialTrialReceipt]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[FactorialTrialReceipt]] = defaultdict(
        list
    )
    for row in receipts:
        grouped[(row.request_class, row.placement, row.material, row.size_id)].append(
            row
        )
    rows = []
    for request_class in REQUEST_CLASSES:
        for placement in PLACEMENTS:
            for size_id, token_count in ALL_TOKEN_COUNTS.items():
                for material in MATERIALS:
                    group = grouped[(request_class, placement, material, size_id)]
                    _require(len(group) == 10, "secondary-dose cell count drift")
                    metrics = {}
                    for field in METRICS:
                        values = np.asarray(
                            [_metric_value(item, field) for item in group],
                            dtype=np.float64,
                        )
                        metrics[field] = {
                            "mean": float(values.mean()),
                            "minimum": float(values.min()),
                            "maximum": float(values.max()),
                        }
                    positive = sum(item.feature_6779_positive for item in group)
                    rows.append(
                        {
                            "request_class": request_class,
                            "placement": placement,
                            "size_id": size_id,
                            "injected_token_count": token_count,
                            "material": material,
                            "n_prompt_families": 10,
                            "feature_6779_positive_count": positive,
                            "feature_6779_positive_prevalence": positive / 10,
                            "metrics": metrics,
                        }
                    )
    return rows


def _paired_vectors(
    receipts: list[FactorialTrialReceipt],
    field: str,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    indexed = {
        (
            row.request_class,
            row.prompt_family_id,
            row.placement,
            row.size_id,
            row.material,
        ): _metric_value(row, field)
        for row in receipts
    }
    vectors = {}
    for request_class in REQUEST_CLASSES:
        family_ids = sorted(
            {
                row.prompt_family_id
                for row in receipts
                if row.request_class == request_class
            }
        )
        _require(len(family_ids) == 10, f"{request_class}: dose family count drift")
        for placement in PLACEMENTS:
            for size_id in SIZE_IDS:
                for contrast, left, right in CONTRASTS:
                    vectors[(request_class, placement, size_id, contrast)] = np.asarray(
                        [
                            indexed[
                                (request_class, family_id, placement, size_id, left)
                            ]
                            - indexed[
                                (request_class, family_id, placement, size_id, right)
                            ]
                            for family_id in family_ids
                        ],
                        dtype=np.float64,
                    )
    return vectors


def _percentile_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _metric_contrasts(
    receipts: list[FactorialTrialReceipt],
    *,
    field: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    vectors = _paired_vectors(receipts, field)
    estimands: list[tuple[tuple[str, ...], str, np.ndarray]] = []
    for key, values in vectors.items():
        request_class, placement, size_id, contrast = key
        estimands.append(
            (
                ("paired", field, request_class, placement, size_id, contrast),
                "paired_contrast",
                values,
            )
        )
    for placement in PLACEMENTS:
        for size_id in SIZE_IDS:
            harmful = vectors[
                ("harmful_request", placement, size_id, "full_content_increment")
            ]
            for comparator, label in (
                ("ordinary_benign_request", "harmful_minus_ordinary_benign"),
                ("minimal_neutral_carrier", "harmful_minus_minimal_neutral"),
            ):
                comparison = vectors[
                    (comparator, placement, size_id, "full_content_increment")
                ]
                estimands.append(
                    (
                        ("interaction", field, placement, size_id, label),
                        "request_class_interaction",
                        np.concatenate([harmful, comparison]),
                    )
                )
    rule = plan["uncertainty"]
    replicates = int(rule["replicates"])
    master_seed = int(rule["master_seed"])
    all_keys = []
    for metric in METRICS:
        for request_class in REQUEST_CLASSES:
            for placement in PLACEMENTS:
                for size_id in SIZE_IDS:
                    for contrast, _, _ in CONTRASTS:
                        all_keys.append(
                            (
                                "paired",
                                metric,
                                request_class,
                                placement,
                                size_id,
                                contrast,
                            )
                        )
        for placement in PLACEMENTS:
            for size_id in SIZE_IDS:
                for label in (
                    "harmful_minus_ordinary_benign",
                    "harmful_minus_minimal_neutral",
                ):
                    all_keys.append(("interaction", metric, placement, size_id, label))
    stream_index = {
        key: index for index, key in enumerate(sorted(all_keys))
    }
    output = []
    for key, kind, values in sorted(estimands, key=lambda item: item[0]):
        seed = master_seed + stream_index[key]
        if kind == "paired_contrast":
            request_class, placement, size_id, contrast = (
                key[2],
                key[3],
                key[4],
                key[5],
            )
            estimate = float(values.mean())
            lower, upper = _percentile_interval(
                values,
                replicates=replicates,
                seed=seed,
            )
            output.append(
                {
                    "kind": kind,
                    "metric": field,
                    "request_class": request_class,
                    "placement": placement,
                    "size_id": size_id,
                    "injected_token_count": ALL_TOKEN_COUNTS[size_id],
                    "contrast": contrast,
                    "n_prompt_families": 10,
                    "estimate": estimate,
                    "pointwise_95_lower": lower,
                    "pointwise_95_upper": upper,
                    "bootstrap_stream_seed": seed,
                }
            )
        else:
            placement, size_id, contrast = key[2], key[3], key[4]
            harmful, comparison = values[:10], values[10:]
            estimate = float(harmful.mean() - comparison.mean())
            rng = np.random.default_rng(seed)
            harmful_indices = rng.integers(0, 10, size=(replicates, 10))
            comparison_indices = rng.integers(0, 10, size=(replicates, 10))
            bootstrap = harmful[harmful_indices].mean(axis=1) - comparison[
                comparison_indices
            ].mean(axis=1)
            lower, upper = np.quantile(
                bootstrap,
                [0.025, 0.975],
                method="linear",
            )
            output.append(
                {
                    "kind": kind,
                    "metric": field,
                    "request_class": "harmful_request",
                    "comparator_request_class": (
                        "ordinary_benign_request"
                        if contrast == "harmful_minus_ordinary_benign"
                        else "minimal_neutral_carrier"
                    ),
                    "placement": placement,
                    "size_id": size_id,
                    "injected_token_count": ALL_TOKEN_COUNTS[size_id],
                    "contrast": contrast,
                    "n_prompt_families_per_request_class": 10,
                    "estimate": estimate,
                    "pointwise_95_lower": float(lower),
                    "pointwise_95_upper": float(upper),
                    "bootstrap_stream_seed": seed,
                }
            )
    return {
        "field": field,
        "bootstrap_replicates": replicates,
        "interval": rule["interval"],
        "interval_role": "pointwise_descriptive_not_simultaneous",
        "p_values": False,
        "formal_pass_fail_decision": False,
        "rows": output,
    }


def analyze_factorial_dose(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    analysis_plan_path: Path,
    execution_receipt_path: Path,
    authorization_path: Path,
    summary_path: Path,
    dose_root: Path,
    canonical_root: Path,
    canonical_execution_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = json.loads(analysis_plan_path.read_text())
    validate_factorial_dose_analysis_plan(plan)
    execution = json.loads(execution_receipt_path.read_text())
    _validate_execution_receipt(
        execution=execution,
        execution_path=execution_receipt_path,
        plan=plan,
        public_plan_path=public_plan_path,
        private_plan_path=private_plan_path,
        authorization_path=authorization_path,
        summary_path=summary_path,
    )
    canonical_execution = json.loads(canonical_execution_receipt_path.read_text())
    _require(
        sha256_file(canonical_execution_receipt_path)
        == execution["canonical_execution_receipt_sha256"],
        "canonical execution receipt binding drift",
    )
    dose = _load_receipts(
        receipt_root=dose_root / "receipts" / "trials",
        restricted_root=dose_root / "restricted",
        expected_count=540,
        expected_manifest_sha256=execution["dose_receipt_manifest_sha256"],
        expected_provenance={
            "public_plan_sha256": execution["public_plan_sha256"],
            "private_plan_sha256": execution["private_plan_sha256"],
            "assay_receipt_sha256": execution["assay_receipt_sha256"],
            "source_commit": execution["source_commit"],
            "run_id": execution["run_id"],
        },
    )
    selected_families = _validate_partial_topology(dose)
    canonical = _load_receipts(
        receipt_root=canonical_root / "receipts" / "trials",
        restricted_root=canonical_root / "restricted",
        expected_count=420,
        expected_manifest_sha256=execution[
            "canonical_matrix_receipt_manifest_sha256"
        ],
        expected_provenance={
            "public_plan_sha256": execution["public_plan_sha256"],
            "private_plan_sha256": execution["private_plan_sha256"],
            "assay_receipt_sha256": execution["assay_receipt_sha256"],
            "source_commit": canonical_execution["matrix"]["source_commit"],
            "run_id": canonical_execution["matrix"]["run_id"],
        },
    )
    canonical_selected = _select_canonical_receipts(canonical, selected_families)
    combined = [*dose, *canonical_selected]
    _require(len(combined) == 720, "secondary-dose combined count drift")
    _validate_complete_topology(combined)
    identity = {
        (
            row.model_revision,
            row.tokenizer_revision,
            row.lens_sha256,
            row.sae_sha256,
        )
        for row in combined
    }
    _require(len(identity) == 1, "secondary-dose artifact identity drift")
    metric_results = {
        field: _metric_contrasts(combined, field=field, plan=plan)
        for field in METRICS
    }
    model_revision, tokenizer_revision, lens_sha256, sae_sha256 = next(iter(identity))
    result = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "analysis_id": plan["analysis_id"],
        "status": "complete",
        "public_plan_sha256": sha256_file(public_plan_path),
        "analysis_plan_sha256": sha256_file(analysis_plan_path),
        "execution_receipt_sha256": sha256_file(execution_receipt_path),
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "source_commit": _source_commit(),
        "source_receipts": {
            "new_partial_dose_count": len(dose),
            "new_partial_dose_manifest_sha256": execution[
                "dose_receipt_manifest_sha256"
            ],
            "reused_canonical_count": len(canonical_selected),
            "canonical_matrix_manifest_sha256": execution[
                "canonical_matrix_receipt_manifest_sha256"
            ],
            "restricted_artifact_hashes_verified": True,
        },
        "artifact_identity": {
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "lens_sha256": lens_sha256,
            "sae_sha256": sae_sha256,
        },
        "factors": {
            "request_classes": REQUEST_CLASSES,
            "materials": MATERIALS,
            "placements": PLACEMENTS,
            "size_ids": SIZE_IDS,
            "injected_token_counts": TOKEN_COUNTS,
            "independent_unit": "prompt_family_id",
            "units_per_cell": 10,
            "selected_prompt_family_ids": selected_families,
        },
        "cell_summaries": _cell_summaries(combined),
        "metric_contrasts": metric_results,
        "dose_shape_policy": plan["dose_shape_policy"],
        "claim_limits": {
            "secondary_descriptive_not_confirmatory": True,
            "placement_pooling": False,
            "size_pooling": False,
            "monotonicity_test": False,
            "linear_dose_response_claim": False,
            "semantic_component_ablation_claim": False,
            "harmful_request_detector_claim": False,
            "causal_mechanism_claim": False,
            "threshold_fit": False,
            "held_out_confirmation_opened": False,
            "semantic_task_completion_scored": False,
            "harmbench_scored": False,
        },
    }
    _require(
        all(
            math.isfinite(row["estimate"])
            for metric in metric_results.values()
            for row in metric["rows"]
        ),
        "non-finite secondary-dose estimate",
    )
    write_json_atomic(output_path, result)
    return result
