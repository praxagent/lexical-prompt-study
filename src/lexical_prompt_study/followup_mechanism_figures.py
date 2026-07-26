from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .figures import COLORS, _configure, _save_all
from .hashing import sha256_file, write_json_atomic

PLACEMENTS = ("ep_before_request", "ep_after_request")
PARTITIONS = ("discovery", "calibration")
TRANSPORTS = ("jacobian_lens", "identity", "random_gaussian")
PLACEMENT_LABELS = {
    "ep_before_request": "Scaffold before request",
    "ep_after_request": "Scaffold after request",
}
TRANSPORT_LABELS = {
    "jacobian_lens": "Jacobian lens",
    "identity": "Identity",
    "random_gaussian": "Matched random",
}
TRANSPORT_COLORS = {
    "jacobian_lens": "#D97706",
    "identity": "#0284C7",
    "random_gaussian": "#6B7280",
}


def _save_receipt(
    *,
    output_dir: Path,
    stem_name: str,
    metadata: dict,
    source_path: Path,
    hashes: dict[str, str],
) -> Path:
    stem = output_dir / stem_name
    receipt = {
        **metadata,
        "source_receipts": [
            {"path": str(source_path), "sha256": sha256_file(source_path)}
        ],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-followup-mechanisms --result {source_path} "
                f"--out {output_dir}"
            ),
        },
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": "direct labels, marker or hatch differences, and reference lines",
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


def _save_figure(fig, stem: Path) -> dict[str, str]:
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    hashes["svg"] = sha256_file(svg_path)
    return hashes


def generate_followup_mechanism_figures(
    result_path: Path,
    output_dir: Path,
) -> dict:
    _configure()
    result = json.loads(result_path.read_text())
    if (
        result["status"] != "complete"
        or result["placement_orderings"] != list(PLACEMENTS)
        or result["pooled_placement_estimate_reported"] is not False
    ):
        raise ValueError("follow-up mechanism result boundary drift")
    selected = result["sae"]["selected_candidate"]
    if selected is None or result["sae"]["threshold"]["status"] != "not_fit":
        raise ValueError("expected a selected candidate with an unfitted threshold")
    candidate_id = f"{selected['kind']}:{'-'.join(map(str, selected['feature_ids']))}"
    matrix = result["sae"]["candidate_arm_matrix"]
    receipts: list[dict[str, str]] = []

    # E09: complete selected-feature arm matrix, never pooling placement.
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    arm_keys = [
        ("shared_base", None, "base"),
        ("ep_before_request", "inert_length", "inert_length"),
        ("ep_before_request", "structural_sham", "structural_sham"),
        ("ep_before_request", "full", "full"),
        ("ep_after_request", "inert_length", "inert_length"),
        ("ep_after_request", "structural_sham", "structural_sham"),
        ("ep_after_request", "full", "full"),
    ]
    labels = [
        "Base\n(shared)",
        "Inert\nbefore",
        "Sham\nbefore",
        "Full\nbefore",
        "Inert\nafter",
        "Sham\nafter",
        "Full\nafter",
    ]
    derived_matrix = {}
    for ax, partition in zip(axes, PARTITIONS, strict=True):
        payload = matrix[partition][candidate_id]
        summaries = []
        for placement, arm, color_arm in arm_keys:
            row = (
                payload["shared_base"]
                if placement == "shared_base"
                else payload[placement][arm]
            )
            summaries.append(
                {
                    "placement": None if placement == "shared_base" else placement,
                    "arm": "base" if placement == "shared_base" else arm,
                    **row,
                }
            )
        derived_matrix[partition] = summaries
        bars = ax.bar(
            np.arange(len(summaries)),
            [row["mean"] for row in summaries],
            color=[COLORS[color_arm] for _, _, color_arm in arm_keys],
            hatch=("", "..", "//", "xx", "..", "//", "xx"),
            edgecolor="white",
        )
        for bar, row in zip(bars, summaries, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.015,
                f"{int(row['prevalence_positive'] * row['n'])}/{row['n']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylim(0, max(row["mean"] for row in summaries) * 1.23)
        ax.set_title(partition.capitalize())
        ax.axhline(0, color="#6B7280", linewidth=0.8)
    axes[0].set_ylabel(f"SAE feature {selected['feature_ids'][0]} mean activation")
    stem = output_dir / "E09-followup-selected-sae-arm-matrix"
    hashes = _save_figure(fig, stem)
    receipt_path = _save_receipt(
        output_dir=output_dir,
        stem_name=stem.name,
        source_path=result_path,
        hashes=hashes,
        metadata={
            "figure_id": "E09",
            "title": "Llama 3.1 8B selected SAE feature: complete arm matrix",
            "question": "Does the selected feature separate the full scaffold from every control in both orderings?",
            "description": "Mean activation with positive-count annotations for all arms, partitions, and placements.",
            "alt_text": (
                f"Two-panel arm matrix for feature {selected['feature_ids'][0]}. "
                "Base and inert-length activation prevalence are zero in both "
                "partitions. Full prevalence is 20 of 20 in both orderings and "
                "partitions. Structural-sham prevalence is larger for the "
                "before-request ordering, so separation is mainly in magnitude, "
                "not activation presence."
            ),
            "permitted_inference": "replicated ordering-specific activation-magnitude contrast on discovery and calibration",
            "non_claims": [
                "no common detector threshold",
                "no ordinary-benign or structured-benign false-positive estimate",
                "no causal or deployment claim",
            ],
            "row_filter": "selected discovery/calibration candidate; all four arms; orderings separate",
            "counts": {"partitions": 2, "placement_levels": 2, "n_per_arm_stratum": 20},
            "derived_data": derived_matrix,
        },
    )
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    # E10: frozen calibration choice between exactly two discovery candidates.
    ranking = result["sae"]["calibration_ranking"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    x = np.arange(len(ranking))
    width = 0.34
    candidate_labels = [
        "Single\nfeature 6779" if row["kind"] == "single_feature" else "8-feature\nsubspace"
        for row in ranking
    ]
    for placement_index, placement in enumerate(PLACEMENTS):
        offset = (placement_index - 0.5) * width
        standardized_bars = axes[0].bar(
            x + offset,
            [row["ordering_results"][placement]["standardized"] for row in ranking],
            width,
            color=("#D97706", "#0284C7")[placement_index],
            hatch=("", "//")[placement_index],
            label=PLACEMENT_LABELS[placement],
        )
        raw_bars = axes[1].bar(
            x + offset,
            [row["ordering_results"][placement]["mean"] for row in ranking],
            width,
            color=("#D97706", "#0284C7")[placement_index],
            hatch=("", "//")[placement_index],
        )
        short_label = "Before" if placement == "ep_before_request" else "After"
        for ax, bars in ((axes[0], standardized_bars), (axes[1], raw_bars)):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    short_label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0].set_ylabel("Paired standardized effect")
    axes[1].set_ylabel("Paired raw mean difference")
    for ax, title in zip(
        axes,
        ("Frozen maximin selection metric", "Magnitude differs from selection metric"),
        strict=True,
    ):
        ax.set_xticks(x, candidate_labels)
        ax.set_title(title)
        ax.axhline(0, color="#6B7280", linewidth=0.8)
    axes[0].set_ylim(0, 1.06)
    axes[1].set_ylim(
        0,
        max(
            row["ordering_results"][placement]["mean"]
            for row in ranking
            for placement in PLACEMENTS
        )
        * 1.14,
    )
    stem = output_dir / "E10-followup-sae-candidate-selection"
    hashes = _save_figure(fig, stem)
    receipt_path = _save_receipt(
        output_dir=output_dir,
        stem_name=stem.name,
        source_path=result_path,
        hashes=hashes,
        metadata={
            "figure_id": "E10",
            "title": "Frozen calibration choice between two SAE candidates",
            "question": "Why did the single feature advance instead of the eight-feature subspace?",
            "description": "Ordering-specific standardized and raw full-minus-sham effects on calibration.",
            "alt_text": (
                "Two grouped-bar panels compare the single feature and eight-feature "
                "subspace separately for scaffold-before and scaffold-after. The "
                "single feature has the slightly higher worst-order standardized "
                "effect, while the subspace has larger raw magnitude."
            ),
            "permitted_inference": "the prospectively frozen maximin rule selects feature 6779 on calibration",
            "non_claims": ["not a post-hoc detector threshold", "not held-out confirmation"],
            "row_filter": "exactly the two discovery-frozen candidates; calibration only",
            "counts": {"candidates": 2, "placement_levels": 2, "n_pairs_per_level": 20},
            "derived_data": ranking,
        },
    )
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    # E11: complete J-lens trajectory, split and ordering kept separate.
    rows = result["jlens"]["rows"]
    if len(rows) != 2 * 2 * 31 * 3:
        raise ValueError("J-lens row topology drift")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True, constrained_layout=True)
    for row_index, partition in enumerate(PARTITIONS):
        for column_index, placement in enumerate(PLACEMENTS):
            ax = axes[row_index, column_index]
            for transport in TRANSPORTS:
                selected_rows = sorted(
                    (
                        row
                        for row in rows
                        if row["partition"] == partition
                        and row["placement"] == placement
                        and row["transport"] == transport
                    ),
                    key=lambda row: row["layer"],
                )
                layers = np.asarray([row["layer"] for row in selected_rows])
                means = np.asarray(
                    [row["full_minus_structural_sham"] for row in selected_rows]
                )
                lower = np.asarray([row["bootstrap_95"][0] for row in selected_rows])
                upper = np.asarray([row["bootstrap_95"][1] for row in selected_rows])
                ax.plot(
                    layers,
                    means,
                    color=TRANSPORT_COLORS[transport],
                    label=TRANSPORT_LABELS[transport],
                    marker={"jacobian_lens": "o", "identity": "s", "random_gaussian": "^"}[
                        transport
                    ],
                    markevery=5,
                    linewidth=1.6,
                    markersize=4,
                )
                ax.fill_between(
                    layers,
                    lower,
                    upper,
                    color=TRANSPORT_COLORS[transport],
                    alpha=0.10,
                )
            ax.axhline(0, color="#111827", linewidth=0.8, linestyle="--")
            ax.set_title(f"{partition.capitalize()} · {PLACEMENT_LABELS[placement]}")
            ax.set_xlabel("Source layer")
            if column_index == 0:
                ax.set_ylabel("Full − sham refusal/compliance margin")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
    stem = output_dir / "E11-followup-jlens-trajectories"
    hashes = _save_figure(fig, stem)
    receipt_path = _save_receipt(
        output_dir=output_dir,
        stem_name=stem.name,
        source_path=result_path,
        hashes=hashes,
        metadata={
            "figure_id": "E11",
            "title": "Llama 3.1 8B J-lens trajectories by placement",
            "question": "Where does the full-minus-sham refusal/compliance readout differ across layers?",
            "description": "All 31 layers under fitted J-lens, identity, and Frobenius-matched random transport.",
            "alt_text": (
                "Four trajectory panels keep discovery versus calibration and "
                "scaffold-before versus scaffold-after separate. Each panel plots "
                "the paired full-minus-sham refusal-minus-compliance margin across "
                "31 source layers for Jacobian-lens, identity, and matched-random "
                "transport, with paired bootstrap bands."
            ),
            "permitted_inference": "secondary descriptive layerwise readout on the pinned probe and state boundary",
            "non_claims": [
                "no causal localization",
                "no hidden-thought interpretation",
                "no placement pooling or equivalence claim",
            ],
            "row_filter": "all public J-lens rows; partitions and placements shown separately",
            "uncertainty": "10,000-replicate paired bootstrap over 20 behavior IDs per stratum",
            "counts": {"rows": len(rows), "layers": 31, "transports": 3, "n_pairs_per_row": 20},
            "derived_data": rows,
        },
    )
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    provenance = {
        "schema_version": "1.0",
        "source_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "figures": receipts,
    }
    provenance_path = output_dir / "provenance.followup-mechanisms.json"
    write_json_atomic(provenance_path, provenance)
    return {"figures": len(receipts), "provenance": str(provenance_path)}


def verify_followup_mechanism_figures(
    result_path: Path,
    output_dir: Path,
) -> dict:
    receipt_names = (
        "E09-followup-selected-sae-arm-matrix.receipt.json",
        "E10-followup-sae-candidate-selection.receipt.json",
        "E11-followup-jlens-trajectories.receipt.json",
    )
    comparisons = []
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_followup_mechanism_figures(result_path, temporary)
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
        provenance_path = output_dir / "provenance.followup-mechanisms.json"
        provenance = json.loads(provenance_path.read_text())
        for row in provenance["figures"]:
            row["sha256"] = sha256_file(Path(row["path"]))
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
