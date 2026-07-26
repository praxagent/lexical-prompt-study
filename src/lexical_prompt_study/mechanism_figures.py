from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .figures import _configure, _save_all
from .hashing import sha256_file, write_json_atomic

TRANSPORT_LABELS = {
    "jacobian_lens": "Fitted J-lens",
    "identity": "Identity / logit",
    "random_gaussian": "Random Gaussian",
}
TRANSPORT_COLORS = {
    "jacobian_lens": "#D97706",
    "identity": "#0284C7",
    "random_gaussian": "#6B7280",
}
TRANSPORT_STYLES = {
    "jacobian_lens": "-",
    "identity": "--",
    "random_gaussian": ":",
}


def _receipt(
    *,
    figure_id: str,
    title: str,
    description: str,
    alt_text: str,
    source_path: Path,
    generator_path: Path,
    stem: Path,
    hashes: dict[str, str],
    derived_data: dict,
    non_claims: list[str],
) -> dict:
    return {
        "figure_id": figure_id,
        "title": title,
        "description": description,
        "alt_text": alt_text,
        "source_receipts": [{"path": str(source_path), "sha256": sha256_file(source_path)}],
        "derived_data": derived_data,
        "non_claims": non_claims,
        "generator": {
            "path": str(generator_path),
            "sha256": sha256_file(generator_path),
            "plotting_library": "matplotlib",
            "command": (
                f"lexical-study figures-mechanisms --analysis {source_path} "
                f"--out {stem.parent}"
            ),
        },
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "verification": {
            "status": "pending",
            "verified_utc": None,
            "byte_identity": None,
        },
    }


def generate_mechanism_figures(analysis_path: Path, output_dir: Path) -> dict:
    _configure()
    analysis = json.loads(analysis_path.read_text())
    generator_path = Path(__file__)
    receipts = []

    # E03: identical statistic and layer surface for fitted, identity, and random.
    fig, ax = plt.subplots(figsize=(10.5, 5.1), constrained_layout=True)
    layerwise = analysis["layerwise_primary"]["rows"]
    e03_data = []
    for transport in TRANSPORT_LABELS:
        rows = sorted(
            (row for row in layerwise if row["transport"] == transport),
            key=lambda row: row["layer"],
        )
        layers = np.array([row["layer"] for row in rows])
        means = np.array([row["mean_full_minus_sham_margin"] for row in rows])
        lower = np.array([row["bootstrap_95_interval"][0] for row in rows])
        upper = np.array([row["bootstrap_95_interval"][1] for row in rows])
        ax.plot(
            layers,
            means,
            color=TRANSPORT_COLORS[transport],
            linestyle=TRANSPORT_STYLES[transport],
            linewidth=2.0,
            label=TRANSPORT_LABELS[transport],
        )
        ax.fill_between(
            layers,
            lower,
            upper,
            color=TRANSPORT_COLORS[transport],
            alpha=0.12,
            linewidth=0,
        )
        e03_data.extend(rows)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.axvline(50, color="#7C3AED", linewidth=1.0, alpha=0.7)
    ax.text(50.7, ax.get_ylim()[0] * 0.94, "SAE hook: L50", color="#7C3AED", fontsize=9)
    ax.set_xlabel("Residual-stream layer")
    ax.set_ylabel("Paired full − sham refusal/compliance margin")
    ax.set_title("Descriptive layerwise readout at the turn-2 assistant boundary")
    ax.legend(frameon=False, ncol=3)
    stem = output_dir / "E03-layerwise-jlens-margin"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    receipt = _receipt(
        figure_id="E03",
        title="Layerwise J-lens refusal/compliance margin",
        description=(
            "Paired full-minus-structural-sham probe margin at the shared assistant "
            "boundary under fitted, identity, and random transports."
        ),
        alt_text=(
            "Three layerwise curves with behavior-bootstrap intervals. The fitted "
            "J-lens diverges from identity in middle-to-late layers, while the "
            "single seeded random transport also produces large structured swings."
        ),
        source_path=analysis_path,
        generator_path=generator_path,
        stem=stem,
        hashes=hashes,
        derived_data={
            "rows": e03_data,
            "curve_diagnostics": analysis["layerwise_primary"]["curve_diagnostics"],
        },
        non_claims=[
            "readout differences are not causal mechanisms",
            "a single random transport is not a calibrated null distribution",
            "visually strongest layers are descriptive maxima",
        ],
    )
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    # E04: one common color surface across the three controls.
    trajectory = analysis["trajectory"]["rows"]
    layers = sorted({int(row["layer"]) for row in trajectory})
    positions = [None, 0, 1, 2, 4, 8, 16]
    matrices = {}
    for transport in TRANSPORT_LABELS:
        lookup = {
            (int(row["layer"]), row["position_token_index"]): row[
                "mean_full_minus_sham_margin"
            ]
            for row in trajectory
            if row["transport"] == transport
        }
        matrices[transport] = np.array(
            [
                [lookup[(layer, position)] for layer in layers]
                for position in positions
            ],
            dtype=float,
        )
    limit = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices.values())
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.7), constrained_layout=True)
    image = None
    for ax, transport in zip(axes, TRANSPORT_LABELS, strict=True):
        image = ax.imshow(
            matrices[transport],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=[min(layers) - 0.5, max(layers) + 0.5, -0.5, len(positions) - 0.5],
        )
        ax.set_yticks(
            range(len(positions)),
            ["boundary", "0", "1", "2", "4", "8", "16"],
        )
        ax.set_ylabel("Position")
        ax.set_title(TRANSPORT_LABELS[transport], loc="left")
    axes[-1].set_xlabel("Residual-stream layer")
    fig.colorbar(image, ax=axes, label="Paired full − sham margin", shrink=0.92)
    stem = output_dir / "E04-layer-position-trajectory"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    receipt = _receipt(
        figure_id="E04",
        title="Layer-by-position trajectory map",
        description=(
            "Common-scale heatmaps across the assistant boundary and six frozen "
            "generated-token indices."
        ),
        alt_text=(
            "Three common-scale heatmaps for fitted J-lens, identity, and seeded "
            "random transport. Color shows the paired full-minus-sham probe margin "
            "across 79 layers and seven frozen positions."
        ),
        source_path=analysis_path,
        generator_path=generator_path,
        stem=stem,
        hashes=hashes,
        derived_data={
            "positions": positions,
            "layers": layers,
            "common_absolute_color_limit": limit,
            "rows": trajectory,
        },
        non_claims=[
            "heatmap maxima are descriptive and uncorrected for selection",
            "readout differences are not hidden thoughts or causal mechanisms",
        ],
    )
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    # E05: retain the whole candidate surface, highlighting selections and controls.
    feature_rows = analysis["sae_discovery"]["feature_map_rows"]
    selected_ids = set(analysis["sae_discovery"]["selected_feature_ids"])
    control_ids = {
        int(item["feature_id"]) for item in analysis["sae_discovery"]["matched_controls"]
    }
    decoder_norm = np.array([row["decoder_norm"] for row in feature_rows], dtype=float)
    paired_delta = np.array([row["paired_mean_delta"] for row in feature_rows], dtype=float)
    finite = (decoder_norm > 0) & np.isfinite(decoder_norm) & np.isfinite(paired_delta)
    fig, ax = plt.subplots(figsize=(10.5, 5.7), constrained_layout=True)
    density = ax.hexbin(
        decoder_norm[finite],
        paired_delta[finite],
        gridsize=85,
        xscale="log",
        bins="log",
        mincnt=1,
        cmap="Greys",
        linewidths=0,
    )
    fig.colorbar(density, ax=ax, label="log10 feature count")
    for feature_id, color, marker, label in (
        (selected_ids, "#DC2626", "*", "Selected candidates"),
        (control_ids, "#2563EB", "s", "Matched controls"),
    ):
        rows = [row for row in feature_rows if int(row["feature_id"]) in feature_id]
        ax.scatter(
            [row["decoder_norm"] for row in rows],
            [row["paired_mean_delta"] for row in rows],
            color=color,
            marker=marker,
            s=95 if marker == "*" else 48,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=4,
        )
        for row in rows:
            ax.annotate(
                str(row["feature_id"]),
                (row["decoder_norm"], row["paired_mean_delta"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("SAE decoder norm")
    ax.set_ylabel("Paired full − sham activation delta")
    ax.set_title("Discovery-only SAE feature map at residual-post layer 50")
    ax.legend(frameon=False)
    stem = output_dir / "E05-sae-candidate-map"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    receipt = _receipt(
        figure_id="E05",
        title="Discovery-only SAE candidate map",
        description=(
            "All 65,536 features shown as density over decoder norm and paired "
            "activation delta, with selected candidates and matched controls retained."
        ),
        alt_text=(
            "Hexbin density plot of 65,536 layer-50 SAE features. Four selected "
            "features have positive full-minus-sham activation deltas and are marked "
            "with red stars; three norm/frequency-matched controls are blue squares."
        ),
        source_path=analysis_path,
        generator_path=generator_path,
        stem=stem,
        hashes=hashes,
        derived_data={
            "feature_count": len(feature_rows),
            "selected_diagnostics": analysis["sae_discovery"]["selected_diagnostics"],
            "matched_controls": analysis["sae_discovery"]["matched_controls"],
            "selection_rule": analysis["sae_discovery"]["selection_rule"],
            "matched_control_rule": analysis["sae_discovery"]["matched_control_rule"],
            "feature_map_rows_sha256": sha256_file(
                Path(analysis["source_files"]["sae_discovery"]["path"])
            ),
        },
        non_claims=[
            "feature labels are hypotheses, not mechanisms",
            "candidate and control selection are discovery-only",
            "no held-out causal outcome is shown",
        ],
    )
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    provenance = {
        "schema_version": "1.0",
        "source_analysis": {
            "path": str(analysis_path),
            "sha256": sha256_file(analysis_path),
        },
        "figures": receipts,
    }
    write_json_atomic(output_dir / "provenance.json", provenance)
    return {"figures": len(receipts), "provenance": str(output_dir / "provenance.json")}


def verify_mechanism_figures(analysis_path: Path, output_dir: Path) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_mechanism_figures(analysis_path, temporary)
        expected = json.loads((output_dir / "provenance.json").read_text())
        actual = json.loads((temporary / "provenance.json").read_text())
        comparisons = []
        for expected_figure, actual_figure in zip(
            expected["figures"], actual["figures"], strict=True
        ):
            expected_receipt = json.loads(Path(expected_figure["path"]).read_text())
            actual_receipt = json.loads(Path(actual_figure["path"]).read_text())
            for output_type in ("svg", "png", "pdf"):
                comparisons.append(
                    {
                        "figure_id": expected_receipt["figure_id"],
                        "output_type": output_type,
                        "byte_identical": (
                            expected_receipt["outputs"][output_type]["sha256"]
                            == actual_receipt["outputs"][output_type]["sha256"]
                        ),
                    }
                )
        if not all(item["byte_identical"] for item in comparisons):
            raise ValueError(f"mechanism figure byte verification failed: {comparisons}")
        verified_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        for figure in expected["figures"]:
            receipt_path = Path(figure["path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["verification"] = {
                "status": "verified",
                "verified_utc": verified_utc,
                "byte_identity": [
                    item
                    for item in comparisons
                    if item["figure_id"] == receipt["figure_id"]
                ],
            }
            write_json_atomic(receipt_path, receipt)
            figure["sha256"] = sha256_file(receipt_path)
        write_json_atomic(output_dir / "provenance.json", expected)
    return {"status": "verified", "comparisons": comparisons}
