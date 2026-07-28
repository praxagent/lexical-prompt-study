from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .figures import _configure, _save_all
from .hashing import sha256_file, write_json_atomic


PLACEMENTS = ("ep_before_request", "ep_after_request")
PLACEMENT_LABELS = {
    "ep_before_request": "Scaffold before request",
    "ep_after_request": "Scaffold after request",
}
CLASSES = (
    "minimal_neutral_carrier",
    "ordinary_benign_request",
    "harmful_request",
)
CLASS_LABELS = {
    "minimal_neutral_carrier": "Neutral carrier",
    "ordinary_benign_request": "Benign request",
    "harmful_request": "Harmful request",
}
MATERIALS = ("inert_length", "structural_sham", "full_scaffold")
MATERIAL_LABELS = {
    "inert_length": "Inert length",
    "structural_sham": "Structural sham",
    "full_scaffold": "Full scaffold",
}
MATERIAL_COLORS = {
    "inert_length": "#7C3AED",
    "structural_sham": "#0284C7",
    "full_scaffold": "#D97706",
}
MATERIAL_MARKERS = {
    "inert_length": "o",
    "structural_sham": "s",
    "full_scaffold": "D",
}
CLASS_COLORS = {
    "minimal_neutral_carrier": "#6B7280",
    "ordinary_benign_request": "#0284C7",
    "harmful_request": "#D97706",
}
CLASS_MARKERS = {
    "minimal_neutral_carrier": "o",
    "ordinary_benign_request": "s",
    "harmful_request": "D",
}
TOKENS = (64, 128, 188, 252)


def _save_figure(fig, stem: Path) -> dict[str, str]:
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg = stem.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    hashes["svg"] = sha256_file(svg)
    return hashes


def _receipt(
    *,
    result_path: Path,
    output_dir: Path,
    stem_name: str,
    hashes: dict[str, str],
    metadata: dict[str, Any],
) -> Path:
    stem = output_dir / stem_name
    payload = {
        **metadata,
        "source_receipts": [
            {"path": str(result_path), "sha256": sha256_file(result_path)}
        ],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-factorial-dose --result {result_path} "
                f"--out {output_dir}"
            ),
        },
        "outputs": {
            kind: {
                "path": str(stem.with_suffix(f".{kind}")),
                "sha256": digest,
            }
            for kind, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": (
                "direct panel labels, marker shapes, line styles, and zero lines"
            ),
            "text_equivalent": "alt_text plus complete derived_data",
        },
        "verification": {
            "status": "pending",
            "verified_utc": None,
            "byte_identity": None,
        },
    }
    path = stem.with_suffix(".receipt.json")
    write_json_atomic(path, payload)
    return path


def _cell(
    result: dict[str, Any],
    *,
    placement: str,
    request_class: str,
    token_count: int,
    material: str,
) -> dict[str, Any]:
    return next(
        row
        for row in result["cell_summaries"]
        if row["placement"] == placement
        and row["request_class"] == request_class
        and row["injected_token_count"] == token_count
        and row["material"] == material
    )


def _contrast(
    result: dict[str, Any],
    *,
    metric: str,
    placement: str,
    request_class: str,
    token_count: int,
    contrast: str,
) -> dict[str, Any]:
    return next(
        row
        for row in result["metric_contrasts"][metric]["rows"]
        if row["kind"] == "paired_contrast"
        and row["placement"] == placement
        and row["request_class"] == request_class
        and row["injected_token_count"] == token_count
        and row["contrast"] == contrast
    )


def _validate_source(result: dict[str, Any]) -> None:
    if (
        result["status"] != "complete"
        or result["analysis_id"] != "factorial-8b-secondary-dose-v1"
        or result["source_receipts"]["new_partial_dose_count"] != 540
        or result["source_receipts"]["reused_canonical_count"] != 180
        or len(result["cell_summaries"]) != 72
        or result["claim_limits"]["placement_pooling"] is not False
        or result["claim_limits"]["size_pooling"] is not False
        or result["claim_limits"]["threshold_fit"] is not False
        or result["claim_limits"]["held_out_confirmation_opened"] is not False
        or result["claim_limits"]["monotonicity_test"] is not False
    ):
        raise ValueError("secondary-dose figure source boundary drift")


def generate_factorial_dose_figures(
    result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _configure()
    result = json.loads(result_path.read_text())
    _validate_source(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_index = []

    # E16: primary SAE feature magnitude across all controls and sizes.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 7.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    dose_rows = []
    for row_index, placement in enumerate(PLACEMENTS):
        for column_index, request_class in enumerate(CLASSES):
            ax = axes[row_index, column_index]
            for material in MATERIALS:
                selected = [
                    _cell(
                        result,
                        placement=placement,
                        request_class=request_class,
                        token_count=token_count,
                        material=material,
                    )
                    for token_count in TOKENS
                ]
                values = [
                    row["metrics"]["feature_6779_magnitude"]["mean"]
                    for row in selected
                ]
                ax.plot(
                    TOKENS,
                    values,
                    color=MATERIAL_COLORS[material],
                    marker=MATERIAL_MARKERS[material],
                    label=MATERIAL_LABELS[material],
                    linewidth=1.8,
                    markersize=5,
                )
                dose_rows.extend(
                    {
                        "placement": placement,
                        "request_class": request_class,
                        "material": material,
                        "injected_token_count": token_count,
                        "mean_feature_6779_magnitude": value,
                    }
                    for token_count, value in zip(TOKENS, values, strict=True)
                )
            ax.set_title(
                f"{CLASS_LABELS[request_class]}\n{PLACEMENT_LABELS[placement]}"
            )
            ax.set_xticks(TOKENS)
            ax.set_xlabel("Injected tokens")
            ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    axes[0, 0].set_ylabel("Mean feature 6779 magnitude")
    axes[1, 0].set_ylabel("Mean feature 6779 magnitude")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    stem = output_dir / "E16-dose-feature-6779-magnitude"
    hashes = _save_figure(fig, stem)
    path = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E16",
            "title": "Feature 6779 magnitude across scaffold size",
            "question": (
                "How does the frozen feature respond as injected lexical material "
                "grows, with length, structure, request class, and placement visible?"
            ),
            "description": (
                "Exact ten-family cell means at four realized token counts. Lines "
                "connect ordered sizes for visualization only."
            ),
            "alt_text": (
                "Six line-chart panels separate three request classes and both "
                "scaffold placements. Inert-length controls stay near zero. Sham "
                "and full-scaffold feature magnitudes are generally small at 64 "
                "and 128 tokens and larger at 188 and 252 tokens, with material "
                "and placement differences. Connected lines are not monotonicity "
                "or linear-dose tests."
            ),
            "row_filter": "all 72 request-class by placement by material by size cells",
            "counts": {"cells": 72, "prompt_families_per_cell": 10},
            "uncertainty": "none on this descriptive cell-mean plot",
            "permitted_inference": (
                "whitespace-aligned lexical-prefix dose curves on the frozen panels"
            ),
            "non_claims": [
                "no monotonicity or linear-dose test",
                "no semantic component ablation",
                "no placement pooling",
                "no harmful-request detector claim",
            ],
            "derived_data": dose_rows,
        },
    )
    receipt_index.append({"path": str(path), "sha256": sha256_file(path)})

    # E17: strict-positive prevalence, shown as exact counts.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 7.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    prevalence_rows = []
    for row_index, placement in enumerate(PLACEMENTS):
        for column_index, request_class in enumerate(CLASSES):
            ax = axes[row_index, column_index]
            for material in MATERIALS:
                selected = [
                    _cell(
                        result,
                        placement=placement,
                        request_class=request_class,
                        token_count=token_count,
                        material=material,
                    )
                    for token_count in TOKENS
                ]
                values = [
                    row["feature_6779_positive_prevalence"] for row in selected
                ]
                ax.plot(
                    TOKENS,
                    values,
                    color=MATERIAL_COLORS[material],
                    marker=MATERIAL_MARKERS[material],
                    label=MATERIAL_LABELS[material],
                    linewidth=1.8,
                    markersize=5,
                )
                for token_count, value, row in zip(
                    TOKENS, values, selected, strict=True
                ):
                    prevalence_rows.append(
                        {
                            "placement": placement,
                            "request_class": request_class,
                            "material": material,
                            "injected_token_count": token_count,
                            "positive_count": row[
                                "feature_6779_positive_count"
                            ],
                            "n": row["n_prompt_families"],
                            "prevalence": value,
                        }
                    )
            ax.set_title(
                f"{CLASS_LABELS[request_class]}\n{PLACEMENT_LABELS[placement]}"
            )
            ax.set_xticks(TOKENS)
            ax.set_xlabel("Injected tokens")
            ax.set_ylim(-0.04, 1.04)
    axes[0, 0].set_ylabel("Strict-positive prevalence")
    axes[1, 0].set_ylabel("Strict-positive prevalence")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    stem = output_dir / "E17-dose-feature-6779-prevalence"
    hashes = _save_figure(fig, stem)
    path = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E17",
            "title": "Feature 6779 strict-positive prevalence across size",
            "question": (
                "How often is the frozen feature nonzero at each material size?"
            ),
            "description": (
                "Exact positive counts divided by ten prompt families per cell; "
                "strict positivity is descriptive and is not a detector threshold."
            ),
            "alt_text": (
                "Six placement-separated panels show exact feature-positive "
                "prevalence. Inert length is zero in nearly every cell. Structural "
                "sham and full scaffold become more frequently positive at larger "
                "sizes, including neutral and benign requests. Strict positivity "
                "therefore does not establish harmful-request specificity."
            ),
            "row_filter": "all 72 dose cells",
            "counts": {"cells": 72, "prompt_families_per_cell": 10},
            "uncertainty": "exact descriptive census of each frozen ten-family cell",
            "permitted_inference": "strict-positive prevalence on the frozen panels",
            "non_claims": [
                "not a calibrated threshold",
                "no sensitivity or specificity estimate",
                "no harmful-request detector claim",
                "no placement pooling",
            ],
            "derived_data": prevalence_rows,
        },
    )
    receipt_index.append({"path": str(path), "sha256": sha256_file(path)})

    # E18: primary paired material contrasts with frozen pointwise intervals.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    contrast_rows = []
    contrast_styles = {
        "full_content_increment": ("Full minus sham", "#D97706", "D", "-"),
        "sham_over_inert_increment": ("Sham minus inert", "#0284C7", "s", "--"),
    }
    for row_index, placement in enumerate(PLACEMENTS):
        for column_index, request_class in enumerate(CLASSES):
            ax = axes[row_index, column_index]
            for contrast, (label, color, marker, linestyle) in contrast_styles.items():
                selected = [
                    _contrast(
                        result,
                        metric="feature_6779_magnitude",
                        placement=placement,
                        request_class=request_class,
                        token_count=token_count,
                        contrast=contrast,
                    )
                    for token_count in TOKENS
                ]
                estimates = np.asarray([row["estimate"] for row in selected])
                lower = np.asarray([row["pointwise_95_lower"] for row in selected])
                upper = np.asarray([row["pointwise_95_upper"] for row in selected])
                ax.errorbar(
                    TOKENS,
                    estimates,
                    yerr=[estimates - lower, upper - estimates],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                    linewidth=1.6,
                    capsize=3,
                    markersize=5,
                )
                contrast_rows.extend(selected)
            ax.axhline(0, color="#6B7280", linewidth=0.9)
            ax.set_title(
                f"{CLASS_LABELS[request_class]}\n{PLACEMENT_LABELS[placement]}"
            )
            ax.set_xticks(TOKENS)
            ax.set_xlabel("Injected tokens")
    axes[0, 0].set_ylabel("Paired feature-magnitude contrast")
    axes[1, 0].set_ylabel("Paired feature-magnitude contrast")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    stem = output_dir / "E18-dose-feature-6779-contrasts"
    hashes = _save_figure(fig, stem)
    path = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E18",
            "title": "Which scaffold component adds feature 6779 activation?",
            "question": (
                "At each size, how much activation is added by structure over "
                "inert length and by full scaffold content over matched structure?"
            ),
            "description": (
                "Paired ten-family mean contrasts with prespecified 10,000-replicate "
                "pointwise percentile bootstrap intervals."
            ),
            "alt_text": (
                "Six panels show sham-minus-inert and full-minus-sham feature "
                "contrasts at four sizes. Both components vary with size and "
                "placement. Full-minus-sham is clearly positive at 252 tokens in "
                "every panel, but is small, uncertain, or negative in several "
                "shorter cells. Intervals are pointwise and descriptive."
            ),
            "row_filter": (
                "feature_6779_magnitude paired contrasts; all request classes, "
                "placements, and sizes"
            ),
            "counts": {"contrasts": 48, "prompt_families_per_contrast": 10},
            "independent_unit": "prompt_family_id",
            "uncertainty": (
                "10,000-replicate paired-family pointwise percentile bootstrap; "
                "not simultaneous and no p-values"
            ),
            "permitted_inference": (
                "size-specific decomposition of the frozen feature readout on "
                "the fixed panels"
            ),
            "non_claims": [
                "no monotonicity test",
                "no cross-size familywise inference",
                "no semantic component identification",
                "no detector or causal claim",
            ],
            "derived_data": contrast_rows,
        },
    )
    receipt_index.append({"path": str(path), "sha256": sha256_file(path)})

    # E19: secondary internal readouts, full minus sham only.
    metric_specs = (
        (
            "frozen_subspace_score",
            "Frozen eight-feature subspace",
            "Full minus sham subspace score",
        ),
        (
            "assistant_boundary_jlens_margin",
            "Assistant-boundary Jacobian lens",
            "Full minus sham J-lens margin",
        ),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    secondary_rows = []
    for row_index, (metric, title, ylabel) in enumerate(metric_specs):
        for column_index, placement in enumerate(PLACEMENTS):
            ax = axes[row_index, column_index]
            for request_class in CLASSES:
                selected = [
                    _contrast(
                        result,
                        metric=metric,
                        placement=placement,
                        request_class=request_class,
                        token_count=token_count,
                        contrast="full_content_increment",
                    )
                    for token_count in TOKENS
                ]
                estimates = np.asarray([row["estimate"] for row in selected])
                lower = np.asarray([row["pointwise_95_lower"] for row in selected])
                upper = np.asarray([row["pointwise_95_upper"] for row in selected])
                ax.errorbar(
                    TOKENS,
                    estimates,
                    yerr=[estimates - lower, upper - estimates],
                    color=CLASS_COLORS[request_class],
                    marker=CLASS_MARKERS[request_class],
                    label=CLASS_LABELS[request_class],
                    linewidth=1.6,
                    capsize=3,
                    markersize=5,
                )
                secondary_rows.extend(selected)
            ax.axhline(0, color="#6B7280", linewidth=0.9)
            ax.set_title(f"{title}\n{PLACEMENT_LABELS[placement]}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(TOKENS)
            ax.set_xlabel("Injected tokens")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    stem = output_dir / "E19-dose-secondary-readouts"
    hashes = _save_figure(fig, stem)
    path = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E19",
            "title": "Secondary internal readouts do not collapse to one dose curve",
            "question": (
                "Do the frozen subspace and Jacobian-lens readouts track the "
                "feature-6779 full-minus-sham curve?"
            ),
            "description": (
                "Placement-separated full-minus-sham contrasts for the frozen "
                "eight-feature subspace and assistant-boundary Jacobian-lens margin."
            ),
            "alt_text": (
                "Four panels plot full-minus-sham contrasts across injected size. "
                "The frozen subspace and Jacobian-lens trajectories differ across "
                "request classes, sizes, and placements. The J-lens curve changes "
                "sign in multiple panels, so it is not a simple copy of the "
                "feature-6779 magnitude curve."
            ),
            "row_filter": (
                "full_content_increment only for two secondary readouts, all "
                "request classes, placements, and sizes"
            ),
            "counts": {"contrasts": 48, "prompt_families_per_contrast": 10},
            "independent_unit": "prompt_family_id",
            "uncertainty": (
                "10,000-replicate paired-family pointwise percentile bootstrap"
            ),
            "permitted_inference": (
                "secondary descriptive size trajectories on the same frozen panels"
            ),
            "non_claims": [
                "no secondary decision gate",
                "no causal localization",
                "no monotonicity test",
                "no placement pooling",
            ],
            "derived_data": secondary_rows,
        },
    )
    receipt_index.append({"path": str(path), "sha256": sha256_file(path)})

    # E20: generation-phase parser refusal prevalence.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 7.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    refusal_rows = []
    for row_index, placement in enumerate(PLACEMENTS):
        for column_index, request_class in enumerate(CLASSES):
            ax = axes[row_index, column_index]
            for material in MATERIALS:
                selected = [
                    _cell(
                        result,
                        placement=placement,
                        request_class=request_class,
                        token_count=token_count,
                        material=material,
                    )
                    for token_count in TOKENS
                ]
                values = [
                    row["metrics"]["refusal_detected"]["mean"] for row in selected
                ]
                ax.plot(
                    TOKENS,
                    values,
                    color=MATERIAL_COLORS[material],
                    marker=MATERIAL_MARKERS[material],
                    label=MATERIAL_LABELS[material],
                    linewidth=1.8,
                    markersize=5,
                )
                refusal_rows.extend(
                    {
                        "placement": placement,
                        "request_class": request_class,
                        "material": material,
                        "injected_token_count": token_count,
                        "refusal_count": int(round(value * 10)),
                        "n": 10,
                        "refusal_prevalence": value,
                    }
                    for token_count, value in zip(TOKENS, values, strict=True)
                )
            ax.set_title(
                f"{CLASS_LABELS[request_class]}\n{PLACEMENT_LABELS[placement]}"
            )
            ax.set_xticks(TOKENS)
            ax.set_xlabel("Injected tokens")
            ax.set_ylim(-0.04, 1.04)
    axes[0, 0].set_ylabel("Parser-detected refusal prevalence")
    axes[1, 0].set_ylabel("Parser-detected refusal prevalence")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    stem = output_dir / "E20-dose-refusal-prevalence"
    hashes = _save_figure(fig, stem)
    path = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E20",
            "title": "Generation-phase refusal outcomes across scaffold size",
            "question": (
                "How do parser-detected refusals vary with material, size, request "
                "class, and placement?"
            ),
            "description": (
                "Exact parser-detected refusal counts divided by ten prompt "
                "families per cell. No semantic task-completion judge was run."
            ),
            "alt_text": (
                "Six panels show refusal prevalence for inert length, structural "
                "sham, and full scaffold at four sizes. The trajectories depend "
                "strongly on material, request class, and placement and do not "
                "mirror feature 6779 with one universal rule."
            ),
            "row_filter": "all 72 dose cells; generation-phase refusal parser",
            "counts": {"cells": 72, "prompt_families_per_cell": 10},
            "uncertainty": "exact descriptive census of each frozen ten-family cell",
            "permitted_inference": (
                "parser-detected refusal prevalence on the frozen generation panel"
            ),
            "non_claims": [
                "no HarmBench scoring",
                "no semantic task-completion or utility judgment",
                "no causal relation to feature 6779",
                "no deployment claim",
            ],
            "derived_data": refusal_rows,
        },
    )
    receipt_index.append({"path": str(path), "sha256": sha256_file(path)})

    provenance = output_dir / "provenance.factorial-dose.json"
    write_json_atomic(
        provenance,
        {
            "schema_version": "1.0",
            "source_result": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
            },
            "figures": receipt_index,
        },
    )
    return {"figures": len(receipt_index), "provenance": str(provenance)}


def verify_factorial_dose_figures(
    result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    receipt_names = (
        "E16-dose-feature-6779-magnitude.receipt.json",
        "E17-dose-feature-6779-prevalence.receipt.json",
        "E18-dose-feature-6779-contrasts.receipt.json",
        "E19-dose-secondary-readouts.receipt.json",
        "E20-dose-refusal-prevalence.receipt.json",
    )
    comparisons = []
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_factorial_dose_figures(result_path, temporary)
        for name in receipt_names:
            expected_path = output_dir / name
            expected = json.loads(expected_path.read_text())
            actual = json.loads((temporary / name).read_text())
            output_comparisons = [
                {
                    "output_type": output_type,
                    "byte_identical": (
                        expected["outputs"][output_type]["sha256"]
                        == actual["outputs"][output_type]["sha256"]
                    ),
                }
                for output_type in ("svg", "png", "pdf")
            ]
            if not all(row["byte_identical"] for row in output_comparisons):
                raise ValueError(f"{name}: byte verification failed")
            expected["verification"] = {
                "status": "verified",
                "verified_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "byte_identity": output_comparisons,
            }
            write_json_atomic(expected_path, expected)
            comparisons.extend(
                {"receipt": name, **row} for row in output_comparisons
            )
        provenance_path = output_dir / "provenance.factorial-dose.json"
        provenance = json.loads(provenance_path.read_text())
        for row in provenance["figures"]:
            row["sha256"] = sha256_file(Path(row["path"]))
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
