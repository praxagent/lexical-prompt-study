from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
MATERIALS = (
    "no_scaffold",
    "inert_length",
    "structural_sham",
    "full_scaffold",
)
MATERIAL_LABELS = {
    "no_scaffold": "No scaffold",
    "inert_length": "Inert length",
    "structural_sham": "Structural sham",
    "full_scaffold": "Full scaffold",
}
MATERIAL_COLORS = {
    "no_scaffold": "#6B7280",
    "inert_length": "#7C3AED",
    "structural_sham": "#0284C7",
    "full_scaffold": "#D97706",
}
CLASS_COLORS = {
    "minimal_neutral_carrier": "#6B7280",
    "ordinary_benign_request": "#0284C7",
    "harmful_request": "#D97706",
}


def _save_figure(fig, stem: Path) -> dict[str, str]:
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    hashes["svg"] = sha256_file(svg_path)
    return hashes


def _save_receipt(
    *,
    result_path: Path,
    output_dir: Path,
    stem_name: str,
    hashes: dict[str, str],
    metadata: dict,
) -> Path:
    stem = output_dir / stem_name
    receipt = {
        **metadata,
        "source_receipts": [
            {"path": str(result_path), "sha256": sha256_file(result_path)}
        ],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-factorial --result {result_path} "
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
                "direct labels, marker or hatch differences, and reference lines"
            ),
            "text_equivalent": "alt_text plus derived_data",
        },
        "verification": {
            "status": "pending",
            "verified_utc": None,
            "byte_identity": None,
        },
    }
    path = stem.with_suffix(".receipt.json")
    write_json_atomic(path, receipt)
    return path


def generate_factorial_figures(result_path: Path, output_dir: Path) -> dict:
    _configure()
    result = json.loads(result_path.read_text())
    if (
        result["status"] != "complete"
        or result["source_receipts"]["matrix_count"] != 420
        or result["source_receipts"]["sentinel_count"] != 2
        or result["claim_limits"]["placement_pooling"] is not False
        or result["claim_limits"]["threshold_fit"] is not False
    ):
        raise ValueError("factorial figure source boundary drift")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []

    # E13: complete strict-positive prevalence matrix.
    cell_rows = result["cellwise_diagnostics"]["cells"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.9),
        sharey=True,
        constrained_layout=True,
    )
    width = 0.19
    x = np.arange(len(CLASSES))
    prevalence_data = []
    for ax, placement in zip(axes, PLACEMENTS, strict=True):
        for material_index, material in enumerate(MATERIALS):
            selected = [
                next(
                    row
                    for row in cell_rows
                    if row["placement"] == placement
                    and row["request_class"] == request_class
                    and row["material"] == material
                )
                for request_class in CLASSES
            ]
            offset = (material_index - 1.5) * width
            bars = ax.bar(
                x + offset,
                [row["feature_6779_strict_positive_prevalence"] for row in selected],
                width,
                label=MATERIAL_LABELS[material],
                color=MATERIAL_COLORS[material],
                hatch=("", "..", "//", "xx")[material_index],
                edgecolor="white",
            )
            for bar, row in zip(bars, selected, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{row['feature_6779_strict_positive_count']}/{row['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90 if 0 < row["feature_6779_strict_positive_count"] < 20 else 0,
                )
                prevalence_data.append(
                    {
                        "placement": placement,
                        "request_class": row["request_class"],
                        "material": material,
                        "positive_count": row[
                            "feature_6779_strict_positive_count"
                        ],
                        "n": row["n"],
                        "prevalence": row[
                            "feature_6779_strict_positive_prevalence"
                        ],
                    }
                )
        ax.set_xticks(x, [CLASS_LABELS[value] for value in CLASSES])
        ax.set_ylim(0, 1.14)
        ax.set_title(PLACEMENT_LABELS[placement])
        ax.axhline(0, color="#6B7280", linewidth=0.8)
    axes[0].set_ylabel("Feature 6779 strict-positive prevalence")
    axes[0].legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    stem = output_dir / "E13-factorial-feature-6779-prevalence"
    hashes = _save_figure(fig, stem)
    receipt = _save_receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E13",
            "title": "Feature 6779 across the complete scaffold-by-request matrix",
            "question": (
                "Does feature 6779 require harmful content, inert length, matched "
                "structure, or the full scaffold?"
            ),
            "description": (
                "Strict-positive prevalence for all request classes, materials, "
                "and frozen placements; shared no-scaffold values are repeated "
                "only for visual alignment."
            ),
            "alt_text": (
                "Two placement panels. No-scaffold and inert-length prevalence are "
                "zero of 20 for neutral, benign, and harmful requests. Full-scaffold "
                "prevalence is 20 of 20 in every cell. Structural sham ranges from "
                "10 to 20 of 20, demonstrating that positive activation is not "
                "specific to the full scaffold or harmful requests."
            ),
            "permitted_inference": (
                "placement-stratified strict-positive prevalence on the pinned "
                "Llama 3.1 8B model, SAE, materials, and fixed request panels"
            ),
            "non_claims": [
                "strict positivity is not a detector threshold",
                "no harmful-request specificity",
                "no causal or deployment claim",
                "no placement pooling",
            ],
            "row_filter": "all 420 canonical receipts; canonical size only",
            "counts": {
                "canonical_receipts": 420,
                "independent_prompt_families_per_request_class": 20,
                "plotted_cells": len(prevalence_data),
            },
            "derived_data": prevalence_data,
        },
    )
    receipts.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E14: frozen primary full-content effects and interactions.
    contrasts = result["primary_feature_6779"]["contrasts"]
    critical = result["primary_feature_6779"]["simultaneous_critical_value"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    plotted_contrasts = []
    for ax, placement in zip(axes, PLACEMENTS, strict=True):
        selected = [
            next(
                row
                for row in contrasts
                if row["placement"] == placement
                and row["kind"] == "paired_component"
                and row["request_class"] == request_class
                and row["contrast"] == "full_content_increment"
            )
            for request_class in CLASSES
        ]
        selected.extend(
            [
                next(
                    row
                    for row in contrasts
                    if row["placement"] == placement
                    and row["contrast"] == contrast
                )
                for contrast in (
                    "harmful_vs_benign_full_content",
                    "harmful_vs_neutral_full_content",
                )
            ]
        )
        positions = np.arange(len(selected))
        estimates = np.asarray([row["estimate"] for row in selected])
        lower = np.asarray([row["simultaneous_95_lower"] for row in selected])
        upper = np.asarray([row["simultaneous_95_upper"] for row in selected])
        colors = [
            CLASS_COLORS["minimal_neutral_carrier"],
            CLASS_COLORS["ordinary_benign_request"],
            CLASS_COLORS["harmful_request"],
            "#374151",
            "#9CA3AF",
        ]
        markers = ("o", "s", "D", "^", "v")
        for index, row in enumerate(selected):
            ax.errorbar(
                positions[index],
                estimates[index],
                yerr=[
                    [estimates[index] - lower[index]],
                    [upper[index] - estimates[index]],
                ],
                fmt=markers[index],
                color=colors[index],
                markeredgecolor="#111827",
                capsize=4,
                linewidth=1.6,
                markersize=7,
            )
            plotted_contrasts.append(row)
        ax.axhline(0, color="#6B7280", linewidth=0.9)
        ax.axhline(0.05, color="#16A34A", linewidth=1.0, linestyle="--")
        ax.axhline(-0.05, color="#16A34A", linewidth=1.0, linestyle=":")
        ax.set_xticks(
            positions,
            (
                "Neutral\nfull−sham",
                "Benign\nfull−sham",
                "Harmful\nfull−sham",
                "Harmful−\nbenign",
                "Harmful−\nneutral",
            ),
        )
        ax.set_title(
            f"{PLACEMENT_LABELS[placement]}\n"
            f"{result['primary_feature_6779']['placement_decisions'][placement]['decision'].replace('_', ' ')}"
        )
    axes[0].set_ylabel("Feature 6779 activation contrast")
    stem = output_dir / "E14-factorial-feature-6779-primary-contrasts"
    hashes = _save_figure(fig, stem)
    receipt = _save_receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E14",
            "title": "Feature 6779 full-content effects and request-class interactions",
            "question": (
                "Does the full scaffold add activation beyond structural sham, "
                "and is that increment uniquely larger for harmful requests?"
            ),
            "description": (
                "Frozen full-minus-sham class effects and harmful-minus-comparator "
                "interactions with one familywise simultaneous stability interval."
            ),
            "alt_text": (
                "Both placement panels show positive full-minus-sham effects for "
                "neutral, benign, and harmful requests. The harmful-minus-benign "
                "interaction interval crosses the practical margin in both panels, "
                "so neither placement passes the harmful-specific interaction gate. "
                "Both frozen decisions are mixed or inconclusive."
            ),
            "permitted_inference": (
                "full scaffold content adds feature magnitude beyond matched sham "
                "on every fixed request panel; harmful-specific interaction is not established"
            ),
            "non_claims": [
                "no request-class independence claim",
                "no population inference",
                "no placement comparison or pooling",
                "no detector threshold or causal claim",
            ],
            "row_filter": (
                "prespecified full_content_increment and two prespecified "
                "request-class interactions; placements separate"
            ),
            "independent_unit": "prompt_family_id",
            "uncertainty": (
                "100,000-replicate familywise max-absolute-centered-deviation "
                "cluster bootstrap; common critical value "
                f"{critical}"
            ),
            "counts": {
                "primary_vector_contrasts": result["primary_feature_6779"][
                    "contrast_count"
                ],
                "plotted_contrasts": len(plotted_contrasts),
                "prompt_families_per_request_class": 20,
            },
            "derived_data": {
                "plotted_contrasts": plotted_contrasts,
                "placement_decisions": result["primary_feature_6779"][
                    "placement_decisions"
                ],
                "simultaneous_critical_value": critical,
            },
        },
    )
    receipts.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E15: secondary full-content contrasts, never used for the primary decision.
    metrics = (
        (
            "secondary_frozen_subspace",
            "Frozen 8-feature subspace",
            "Subspace score: full − structural sham",
        ),
        (
            "secondary_jacobian_lens",
            "Jacobian-lens boundary margin",
            "J-lens margin: full − structural sham",
        ),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.0),
        constrained_layout=True,
    )
    secondary_data = []
    for row_index, (key, metric_label, ylabel) in enumerate(metrics):
        rows = result[key]["paired_components"]
        for column_index, placement in enumerate(PLACEMENTS):
            ax = axes[row_index, column_index]
            selected = [
                next(
                    row
                    for row in rows
                    if row["placement"] == placement
                    and row["request_class"] == request_class
                    and row["contrast"] == "full_content_increment"
                )
                for request_class in CLASSES
            ]
            positions = np.arange(len(selected))
            ax.axhline(0, color="#6B7280", linewidth=0.9)
            for index, row in enumerate(selected):
                ax.scatter(
                    index,
                    row["mean"],
                    color=CLASS_COLORS[row["request_class"]],
                    marker=("o", "s", "D")[index],
                    s=55,
                    edgecolor="#111827",
                    linewidth=0.6,
                    zorder=3,
                )
                ax.vlines(
                    index,
                    row["minimum"],
                    row["maximum"],
                    color=CLASS_COLORS[row["request_class"]],
                    alpha=0.55,
                    linewidth=2,
                )
                secondary_data.append({"metric": key, **row})
            ax.set_xticks(positions, [CLASS_LABELS[value] for value in CLASSES])
            ax.set_title(f"{metric_label} · {PLACEMENT_LABELS[placement]}")
            ax.set_ylabel(ylabel)
    stem = output_dir / "E15-factorial-secondary-readouts"
    hashes = _save_figure(fig, stem)
    receipt = _save_receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E15",
            "title": "Secondary subspace and Jacobian-lens readouts",
            "question": (
                "Do the frozen eight-feature subspace and assistant-boundary "
                "Jacobian-lens margin show the same full-over-sham pattern?"
            ),
            "description": (
                "Placement-separated mean full-minus-sham contrasts with the "
                "observed prompt-family range for both secondary internal readouts."
            ),
            "alt_text": (
                "Four panels show frozen-subspace and Jacobian-lens full-minus-sham "
                "contrasts for neutral, benign, and harmful requests in each "
                "placement. Points are means and vertical lines are observed ranges. "
                "These readouts are descriptive and were not reused for the primary "
                "feature-6779 decision."
            ),
            "permitted_inference": (
                "secondary descriptive internal readouts on the same complete "
                "receipt matrix"
            ),
            "non_claims": [
                "no secondary hypothesis gate",
                "no causal localization",
                "no placement pooling",
                "no detector performance",
            ],
            "row_filter": (
                "full_content_increment only; all three request classes; "
                "placements and metrics separate"
            ),
            "counts": {
                "plotted_cells": len(secondary_data),
                "prompt_families_per_cell": 20,
            },
            "derived_data": secondary_data,
        },
    )
    receipts.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    provenance_path = output_dir / "provenance.factorial.json"
    write_json_atomic(
        provenance_path,
        {
            "schema_version": "1.0",
            "source_result": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
            },
            "figures": receipts,
        },
    )
    return {"figures": len(receipts), "provenance": str(provenance_path)}


def verify_factorial_figures(result_path: Path, output_dir: Path) -> dict:
    receipt_names = (
        "E13-factorial-feature-6779-prevalence.receipt.json",
        "E14-factorial-feature-6779-primary-contrasts.receipt.json",
        "E15-factorial-secondary-readouts.receipt.json",
    )
    comparisons = []
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_factorial_figures(result_path, temporary)
        for name in receipt_names:
            expected_path = output_dir / name
            actual = json.loads((temporary / name).read_text())
            expected = json.loads(expected_path.read_text())
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
        provenance_path = output_dir / "provenance.factorial.json"
        provenance = json.loads(provenance_path.read_text())
        for row in provenance["figures"]:
            row["sha256"] = sha256_file(Path(row["path"]))
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
