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


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_local_restricted_artifact(
    receipt: FactorialTrialReceipt,
    restricted_root: Path,
) -> None:
    local_path = restricted_root / Path(receipt.restricted_artifact_path).name
    _require(local_path.is_file(), f"{receipt.trial_id}: missing restricted artifact")
    _require(
        sha256_file(local_path) == receipt.restricted_artifact_sha256,
        f"{receipt.trial_id}: restricted artifact hash drift",
    )


def _load_receipt_lane(
    *,
    receipt_root: Path,
    restricted_root: Path,
    expected_count: int,
    expected_manifest_sha256: str,
    expected_public_plan_sha256: str,
    expected_private_plan_sha256: str,
    expected_assay_receipt_sha256: str,
    expected_source_commit: str,
    expected_run_id: str,
    literal_sentinel: bool,
) -> list[FactorialTrialReceipt]:
    paths = sorted(receipt_root.glob("*.json"))
    _require(len(paths) == expected_count, "factorial receipt lane count drift")
    _require(
        factorial_receipt_manifest_sha256(receipt_root)
        == expected_manifest_sha256,
        "factorial receipt lane manifest drift",
    )
    receipts = []
    for path in paths:
        receipt = FactorialTrialReceipt.model_validate_json(path.read_text())
        _require(path.stem == receipt.trial_id, "factorial receipt filename drift")
        _require(
            receipt.public_plan_sha256 == expected_public_plan_sha256
            and receipt.private_plan_sha256 == expected_private_plan_sha256
            and receipt.assay_receipt_sha256 == expected_assay_receipt_sha256
            and receipt.source_commit == expected_source_commit
            and receipt.run_id == expected_run_id,
            f"{receipt.trial_id}: factorial receipt provenance drift",
        )
        _require(
            (receipt.request_class == "literal_sentinel") is literal_sentinel,
            f"{receipt.trial_id}: wrong factorial receipt lane",
        )
        _require(
            receipt.task_completion is None and receipt.utility_score is None,
            f"{receipt.trial_id}: generation receipt contains semantic judgment",
        )
        _require(
            receipt.feature_6779_positive
            == (receipt.feature_6779_magnitude > 0),
            f"{receipt.trial_id}: feature prevalence drift",
        )
        _validate_local_restricted_artifact(receipt, restricted_root)
        receipts.append(receipt)
    _require(
        len({receipt.trial_id for receipt in receipts}) == expected_count,
        "duplicate factorial trial ID",
    )
    return receipts


def _validate_matrix_topology(
    receipts: list[FactorialTrialReceipt],
    classes: list[str],
    placements: list[str],
    materials: list[str],
) -> None:
    topology = Counter(
        (
            receipt.request_class,
            receipt.material,
            receipt.placement or "shared",
        )
        for receipt in receipts
    )
    expected = Counter()
    for request_class in classes:
        expected[(request_class, materials[0], "shared")] = 20
        for material in materials[1:]:
            for placement in placements:
                expected[(request_class, material, placement)] = 20
    _require(topology == expected, f"factorial canonical topology drift: {topology}")

    by_family: dict[tuple[str, str], list[FactorialTrialReceipt]] = defaultdict(list)
    for receipt in receipts:
        by_family[(receipt.request_class, receipt.prompt_family_id)].append(receipt)
    _require(len(by_family) == 60, "factorial prompt-family count drift")
    for key, rows in by_family.items():
        _require(len(rows) == 7, f"{key}: incomplete factorial family")
        _require(
            len({row.request_id for row in rows}) == 1
            and len({row.request_sha256 for row in rows}) == 1,
            f"{key}: request identity drift",
        )
        base = [row for row in rows if row.material == materials[0]]
        _require(
            len(base) == 1
            and base[0].placement is None
            and base[0].shared_reference is True,
            f"{key}: shared base topology drift",
        )
        injected = [row for row in rows if row.material != materials[0]]
        _require(
            {
                (row.material, row.placement)
                for row in injected
            }
            == {
                (material, placement)
                for material in materials[1:]
                for placement in placements
            },
            f"{key}: injected arm topology drift",
        )
        for placement in placements:
            matched = [row for row in injected if row.placement == placement]
            _require(
                len({row.injected_token_count for row in matched}) == 1
                and len({row.render_group_sha256 for row in matched}) == 1,
                f"{key}/{placement}: size or render matching drift",
            )


def _metric_arrays(
    receipts: list[FactorialTrialReceipt],
    *,
    classes: list[str],
    placements: list[str],
    materials: list[str],
    field: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str], dict[tuple[str, str | None], float]] = defaultdict(
        dict
    )
    for receipt in receipts:
        value = float(getattr(receipt, field))
        indexed[(receipt.request_class, receipt.prompt_family_id)][
            (receipt.material, receipt.placement)
        ] = value

    arrays: dict[str, np.ndarray] = {}
    arm_cells = []
    for request_class in classes:
        family_ids = sorted(
            family_id
            for row_class, family_id in indexed
            if row_class == request_class
        )
        _require(len(family_ids) == 20, f"{request_class}: family count drift")
        components = np.empty((20, len(placements), 3), dtype=np.float64)
        for family_index, family_id in enumerate(family_ids):
            cells = indexed[(request_class, family_id)]
            base = cells[(materials[0], None)]
            for placement_index, placement in enumerate(placements):
                inert = cells[(materials[1], placement)]
                sham = cells[(materials[2], placement)]
                full = cells[(materials[3], placement)]
                components[family_index, placement_index] = (
                    inert - base,
                    sham - inert,
                    full - sham,
                )
        arrays[request_class] = components
        for placement in placements:
            for material in materials:
                placement_key = None if material == materials[0] else placement
                values = [
                    indexed[(request_class, family_id)][(material, placement_key)]
                    for family_id in family_ids
                ]
                arm_cells.append(
                    {
                        "request_class": request_class,
                        "placement": placement,
                        "material": material,
                        "n_prompt_families": len(values),
                        "mean": float(np.mean(values)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
    return arrays, arm_cells


def _vector_from_class_means(
    means: dict[str, np.ndarray],
    classes: list[str],
    placements: list[str],
) -> np.ndarray:
    values = []
    for request_class in classes:
        for placement_index, _ in enumerate(placements):
            values.extend(means[request_class][placement_index].tolist())
    harmful = means["harmful_request"]
    benign = means["ordinary_benign_request"]
    neutral = means["minimal_neutral_carrier"]
    for placement_index, _ in enumerate(placements):
        values.extend(
            (
                harmful[placement_index, 2] - benign[placement_index, 2],
                harmful[placement_index, 2] - neutral[placement_index, 2],
            )
        )
    return np.asarray(values, dtype=np.float64)


def _vector_labels(
    classes: list[str],
    placements: list[str],
    components: list[str],
    interactions: list[str],
) -> list[dict[str, str]]:
    labels = []
    for request_class in classes:
        for placement in placements:
            for component in components:
                labels.append(
                    {
                        "kind": "paired_component",
                        "request_class": request_class,
                        "placement": placement,
                        "contrast": component,
                    }
                )
    for placement in placements:
        for interaction in interactions:
            labels.append(
                {
                    "kind": "request_class_interaction",
                    "request_class": "harmful_request",
                    "placement": placement,
                    "contrast": interaction,
                }
            )
    return labels


def _primary_analysis_from_arrays(
    arrays: dict[str, np.ndarray],
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    classes = list(analysis_plan["request_class_order"])
    placements = list(analysis_plan["placement_order"])
    components = list(analysis_plan["paired_component_order"])
    interactions = list(analysis_plan["interaction_order"])
    rule = analysis_plan["uncertainty"]
    replicates = int(rule["bootstrap_replicates"])
    seed = int(rule["seed"])
    for request_class in classes:
        _require(
            arrays[request_class].shape == (20, len(placements), len(components)),
            f"{request_class}: primary component array shape drift",
        )

    point_means = {
        request_class: arrays[request_class].mean(axis=0)
        for request_class in classes
    }
    point = _vector_from_class_means(point_means, classes, placements)
    rng = np.random.default_rng(seed)
    bootstrap_means = {}
    for request_class in classes:
        indices = rng.integers(0, 20, size=(replicates, 20))
        bootstrap_means[request_class] = arrays[request_class][indices].mean(axis=1)

    bootstrap_columns = []
    for request_class in classes:
        for placement_index, _ in enumerate(placements):
            for component_index, _ in enumerate(components):
                bootstrap_columns.append(
                    bootstrap_means[request_class][
                        :, placement_index, component_index
                    ]
                )
    harmful = bootstrap_means["harmful_request"]
    benign = bootstrap_means["ordinary_benign_request"]
    neutral = bootstrap_means["minimal_neutral_carrier"]
    for placement_index, _ in enumerate(placements):
        bootstrap_columns.extend(
            (
                harmful[:, placement_index, 2] - benign[:, placement_index, 2],
                harmful[:, placement_index, 2] - neutral[:, placement_index, 2],
            )
        )
    bootstrap = np.column_stack(bootstrap_columns)
    _require(
        bootstrap.shape == (replicates, len(point)),
        "primary bootstrap vector shape drift",
    )
    maximum_deviation = np.max(np.abs(bootstrap - point), axis=1)
    critical = float(
        np.quantile(
            maximum_deviation,
            float(rule["quantile"]),
            method=str(rule["numpy_quantile_method"]),
        )
    )
    labels = _vector_labels(classes, placements, components, interactions)
    _require(len(labels) == len(point), "primary label/vector length drift")
    margin = float(analysis_plan["decision_rule"]["practical_margin_feature_units"])
    contrasts = []
    for label, estimate in zip(labels, point, strict=True):
        lower = float(estimate - critical)
        upper = float(estimate + critical)
        contrasts.append(
            {
                **label,
                "estimate": float(estimate),
                "simultaneous_95_lower": lower,
                "simultaneous_95_upper": upper,
                "practical_margin": margin,
                "lower_exceeds_margin": lower > margin,
            }
        )

    decisions = {}
    for placement in placements:
        full_by_class = {
            row["request_class"]: row
            for row in contrasts
            if row["kind"] == "paired_component"
            and row["placement"] == placement
            and row["contrast"] == "full_content_increment"
        }
        interaction_rows = [
            row
            for row in contrasts
            if row["kind"] == "request_class_interaction"
            and row["placement"] == placement
        ]
        interaction_present = all(
            row["simultaneous_95_lower"] > margin for row in interaction_rows
        )
        similarity = all(
            row["simultaneous_95_lower"] > margin
            for row in full_by_class.values()
        ) and all(
            row["simultaneous_95_lower"] > -margin
            and row["simultaneous_95_upper"] < margin
            for row in interaction_rows
        )
        if interaction_present:
            decision = "harmful_request_interaction_present"
        elif similarity:
            decision = "request_class_similarity_on_fixed_panels"
        else:
            decision = "mixed_or_inconclusive"
        decisions[placement] = {
            "decision": decision,
            "harmful_request_interaction_present": interaction_present,
            "request_class_similarity_on_fixed_panels": similarity,
            "failure_to_show_interaction_is_independence_evidence": False,
        }
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "familywise_confidence": float(rule["quantile"]),
        "simultaneous_critical_value": critical,
        "contrast_count": len(contrasts),
        "contrasts": contrasts,
        "placement_decisions": decisions,
        "placement_pooling": False,
        "size_pooling": False,
        "interval_role": rule["interval_role"],
    }


def _descriptive_metric(
    receipts: list[FactorialTrialReceipt],
    *,
    classes: list[str],
    placements: list[str],
    materials: list[str],
    field: str,
) -> dict[str, Any]:
    arrays, arm_cells = _metric_arrays(
        receipts,
        classes=classes,
        placements=placements,
        materials=materials,
        field=field,
    )
    component_cells = []
    component_names = (
        "inert_injection_increment",
        "sham_over_inert_increment",
        "full_content_increment",
    )
    for request_class in classes:
        for placement_index, placement in enumerate(placements):
            for component_index, component in enumerate(component_names):
                values = arrays[request_class][:, placement_index, component_index]
                component_cells.append(
                    {
                        "request_class": request_class,
                        "placement": placement,
                        "contrast": component,
                        "n_prompt_families": len(values),
                        "mean": float(values.mean()),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    return {
        "field": field,
        "arm_cells": arm_cells,
        "paired_components": component_cells,
        "descriptive_only": True,
        "placement_pooling": False,
    }


def _cellwise_summaries(
    receipts: list[FactorialTrialReceipt],
    *,
    classes: list[str],
    placements: list[str],
    materials: list[str],
) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], list[FactorialTrialReceipt]] = defaultdict(list)
    for receipt in receipts:
        report_placements = (
            placements if receipt.placement is None else [receipt.placement]
        )
        for placement in report_placements:
            cells[(receipt.request_class, placement, receipt.material)].append(receipt)
    expected_keys = {
        (request_class, placement, material)
        for request_class in classes
        for placement in placements
        for material in materials
    }
    _require(set(cells) == expected_keys, "factorial reporting-cell topology drift")
    rows = []
    for request_class in classes:
        for placement in placements:
            for material in materials:
                group = cells[(request_class, placement, material)]
                parser_status = Counter(
                    str(receipt.phase_parser_result.get("status")) for receipt in group
                )
                finish_reasons = Counter(receipt.finish_reason for receipt in group)
                reconstruction = np.asarray(
                    [
                        receipt.sae_reconstruction_relative_error
                        for receipt in group
                    ],
                    dtype=np.float64,
                )
                tokens = np.asarray(
                    [receipt.generated_token_count for receipt in group],
                    dtype=np.float64,
                )
                positive = sum(receipt.feature_6779_positive for receipt in group)
                refusal_values = [
                    receipt.refusal_detected
                    for receipt in group
                    if receipt.refusal_detected is not None
                ]
                rows.append(
                    {
                        "request_class": request_class,
                        "placement": placement,
                        "material": material,
                        "n": len(group),
                        "feature_6779_strict_positive_count": positive,
                        "feature_6779_strict_positive_prevalence": positive
                        / len(group),
                        "sae_reconstruction_relative_error": {
                            "mean": float(reconstruction.mean()),
                            "minimum": float(reconstruction.min()),
                            "maximum": float(reconstruction.max()),
                        },
                        "parser_status_counts": dict(sorted(parser_status.items())),
                        "refusal_count": int(sum(bool(value) for value in refusal_values)),
                        "refusal_denominator": len(refusal_values),
                        "finish_reason_counts": dict(sorted(finish_reasons.items())),
                        "truncation_count": int(sum(row.truncated for row in group)),
                        "generated_token_count": {
                            "mean": float(tokens.mean()),
                            "minimum": int(tokens.min()),
                            "maximum": int(tokens.max()),
                        },
                        "semantic_task_completion_judged": False,
                        "harmbench_scored": False,
                    }
                )
    return {
        "cells": rows,
        "strict_positive_prevalence_role": "descriptive_only",
        "semantic_task_completion_status": "deferred",
        "harmbench_status": "deferred",
    }


def analyze_factorial_canonical(
    *,
    public_plan_path: Path,
    analysis_plan_path: Path,
    execution_receipt_path: Path,
    matrix_root: Path,
    sentinel_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    analysis_plan = json.loads(analysis_plan_path.read_text())
    execution = json.loads(execution_receipt_path.read_text())
    _require(
        public_plan["study_id"]
        == analysis_plan["study_id"]
        == execution["study_id"],
        "factorial study identity drift",
    )
    _require(
        analysis_plan["execution_receipt_sha256"]
        == sha256_file(execution_receipt_path),
        "factorial analysis/execution binding drift",
    )
    _require(
        execution["public_plan_sha256"] == sha256_file(public_plan_path)
        and execution["status"] == "canonical_generation_complete",
        "factorial execution receipt drift",
    )
    classes = list(analysis_plan["request_class_order"])
    placements = list(analysis_plan["placement_order"])
    materials = list(analysis_plan["material_order"])
    matrix = _load_receipt_lane(
        receipt_root=matrix_root / "receipts" / "trials",
        restricted_root=matrix_root / "restricted",
        expected_count=420,
        expected_manifest_sha256=execution["matrix"]["receipt_manifest_sha256"],
        expected_public_plan_sha256=execution["public_plan_sha256"],
        expected_private_plan_sha256=execution["private_plan_sha256"],
        expected_assay_receipt_sha256=execution["assay_receipt_sha256"],
        expected_source_commit=execution["matrix"]["source_commit"],
        expected_run_id=execution["matrix"]["run_id"],
        literal_sentinel=False,
    )
    sentinels = _load_receipt_lane(
        receipt_root=sentinel_root / "receipts" / "trials",
        restricted_root=sentinel_root / "restricted",
        expected_count=2,
        expected_manifest_sha256=execution["descriptive_sentinels"][
            "receipt_manifest_sha256"
        ],
        expected_public_plan_sha256=execution["public_plan_sha256"],
        expected_private_plan_sha256=execution["private_plan_sha256"],
        expected_assay_receipt_sha256=execution["assay_receipt_sha256"],
        expected_source_commit=execution["descriptive_sentinels"]["source_commit"],
        expected_run_id=execution["descriptive_sentinels"]["run_id"],
        literal_sentinel=True,
    )
    _validate_matrix_topology(matrix, classes, placements, materials)
    _require(
        {receipt.placement for receipt in sentinels} == set(placements)
        and all(receipt.material == "full_scaffold" for receipt in sentinels),
        "descriptive sentinel topology drift",
    )
    artifact_identity = {
        (
            receipt.model_revision,
            receipt.tokenizer_revision,
            receipt.lens_sha256,
            receipt.sae_sha256,
        )
        for receipt in [*matrix, *sentinels]
    }
    _require(len(artifact_identity) == 1, "factorial artifact identity is not singular")

    primary_arrays, primary_arm_cells = _metric_arrays(
        matrix,
        classes=classes,
        placements=placements,
        materials=materials,
        field=analysis_plan["primary_readout"],
    )
    primary = _primary_analysis_from_arrays(primary_arrays, analysis_plan)
    primary["arm_cells"] = primary_arm_cells
    sentinel_rows = []
    for receipt in sorted(sentinels, key=lambda row: placements.index(row.placement)):
        sentinel_rows.append(
            {
                "placement": receipt.placement,
                "trial_id": receipt.trial_id,
                "feature_6779_magnitude": receipt.feature_6779_magnitude,
                "feature_6779_positive": receipt.feature_6779_positive,
                "frozen_subspace_score": receipt.frozen_subspace_score,
                "assistant_boundary_jlens_margin": (
                    receipt.assistant_boundary_jlens_margin
                ),
                "sae_reconstruction_relative_error": (
                    receipt.sae_reconstruction_relative_error
                ),
                "refusal_detected": receipt.refusal_detected,
                "parser_status": receipt.phase_parser_result.get("status"),
                "generated_token_count": receipt.generated_token_count,
                "finish_reason": receipt.finish_reason,
                "truncated": receipt.truncated,
                "descriptive_n_equals_1": True,
            }
        )
    model_revision, tokenizer_revision, lens_sha256, sae_sha256 = next(
        iter(artifact_identity)
    )
    result = {
        "schema_version": "1.0",
        "study_id": analysis_plan["study_id"],
        "status": "complete",
        "public_plan_sha256": sha256_file(public_plan_path),
        "analysis_plan_sha256": sha256_file(analysis_plan_path),
        "execution_receipt_sha256": sha256_file(execution_receipt_path),
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "source_commit": _source_commit(),
        "source_receipts": {
            "matrix_count": len(matrix),
            "matrix_manifest_sha256": execution["matrix"][
                "receipt_manifest_sha256"
            ],
            "sentinel_count": len(sentinels),
            "sentinel_manifest_sha256": execution["descriptive_sentinels"][
                "receipt_manifest_sha256"
            ],
            "restricted_artifact_hashes_verified": True,
        },
        "artifact_identity": {
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "lens_sha256": lens_sha256,
            "sae_sha256": sae_sha256,
        },
        "primary_feature_6779": primary,
        "secondary_frozen_subspace": _descriptive_metric(
            matrix,
            classes=classes,
            placements=placements,
            materials=materials,
            field="frozen_subspace_score",
        ),
        "secondary_jacobian_lens": _descriptive_metric(
            matrix,
            classes=classes,
            placements=placements,
            materials=materials,
            field="assistant_boundary_jlens_margin",
        ),
        "cellwise_diagnostics": _cellwise_summaries(
            matrix,
            classes=classes,
            placements=placements,
            materials=materials,
        ),
        "descriptive_literal_sentinels": {
            "rows": sentinel_rows,
            "inferential_pooling": False,
            "false_positive_stratum": False,
            "threshold_fit": False,
        },
        "claim_limits": {
            "placement_pooling": False,
            "size_pooling": False,
            "threshold_fit": False,
            "held_out_confirmation_opened": False,
            "semantic_task_completion_scored": False,
            "harmbench_scored": False,
            "population_inference": False,
        },
    }
    _require(
        all(
            math.isfinite(float(row["estimate"]))
            for row in result["primary_feature_6779"]["contrasts"]
        ),
        "non-finite primary factorial estimate",
    )
    write_json_atomic(output_path, result)
    return result
