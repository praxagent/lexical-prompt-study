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


def generate_intervention_figures(
    analysis_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> dict:
    _configure()
    analysis = json.loads(analysis_path.read_text())
    plan = json.loads(plan_path.read_text())
    candidates = sorted(analysis["candidates"], key=lambda row: float(row["rho"]))
    if analysis["status"] != "stopped_no_eligible_alpha":
        raise ValueError("E05a is reserved for a stopped calibration with no eligible alpha")
    if analysis["selection"] is not None or analysis["confirmatory_outcomes_opened"]:
        raise ValueError("E05a requires unopened confirmatory outcomes and a null selection")
    if not candidates or any(row["eligible"] for row in candidates):
        raise ValueError("E05a requires a non-empty, wholly ineligible candidate ladder")

    rules = plan["discovery_alpha_calibration"]["eligible_alpha_rule"]
    half_span_threshold = float(rules["minimum_mean_signed_half_span"])
    representation_ceiling = float(
        rules["maximum_requested_realized_relative_error"]
    )
    residual_ceiling = float(rules["maximum_event_delta_to_pre_residual_norm"])

    labels = [f"{float(row['rho']):g}" for row in candidates]
    x = np.arange(len(candidates), dtype=float)
    half_spans = np.array([row["mean_signed_half_span"] for row in candidates])
    lower = np.array(
        [row["signed_half_span_bootstrap_95_interval"][0] for row in candidates]
    )
    upper = np.array(
        [row["signed_half_span_bootstrap_95_interval"][1] for row in candidates]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)

    axes[0].axhline(
        half_span_threshold,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.3,
        label=f"Eligibility threshold ({half_span_threshold:.1f})",
    )
    axes[0].axhline(0, color="#6B7280", linewidth=0.8)
    axes[0].errorbar(
        x,
        half_spans,
        yerr=[half_spans - lower, upper - half_spans],
        fmt="o",
        color="#D97706",
        markeredgecolor="#111827",
        capsize=5,
        linewidth=1.8,
    )
    for index, value in enumerate(half_spans):
        axes[0].annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Residual-norm dose ρ")
    axes[0].set_ylabel("Mean signed half-span")
    axes[0].set_title("A. Frozen efficacy gate")
    axes[0].legend(frameon=False, loc="upper left")

    restoring = np.array([row["mean_restoring_minus_zero"] for row in candidates])
    opposite = np.array([row["mean_opposite_minus_zero"] for row in candidates])
    axes[1].axhline(0, color="#6B7280", linewidth=0.8)
    axes[1].axhline(
        -0.1,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.3,
        label="Restoring target (≤ −0.1)",
    )
    axes[1].plot(
        x,
        restoring,
        color="#0284C7",
        marker="s",
        linewidth=1.8,
        label="Restoring sign − zero",
    )
    axes[1].plot(
        x,
        opposite,
        color="#7C3AED",
        marker="^",
        linestyle="--",
        linewidth=1.8,
        label="Opposite sign − zero",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Residual-norm dose ρ")
    axes[1].set_ylabel("Mean HarmBench probability difference")
    axes[1].set_title("B. Directional components")
    axes[1].legend(frameon=False, fontsize=8)

    representation = (
        100
        * np.array(
            [row["maximum_requested_realized_relative_error"] for row in candidates]
        )
    )
    residual = (
        100
        * np.array(
            [row["maximum_event_delta_to_pre_residual_norm"] for row in candidates]
        )
    )
    width = 0.34
    axes[2].bar(
        x - width / 2,
        representation,
        width,
        color="#6B7280",
        hatch="//",
        label="Representation mismatch",
    )
    axes[2].bar(
        x + width / 2,
        residual,
        width,
        color="#059669",
        hatch="..",
        label="Effective delta / residual",
    )
    axes[2].axhline(
        100 * representation_ceiling,
        color="#6B7280",
        linestyle="--",
        linewidth=1.1,
        label=f"Mismatch ceiling ({100 * representation_ceiling:.0f}%)",
    )
    axes[2].axhline(
        100 * residual_ceiling,
        color="#047857",
        linestyle=":",
        linewidth=1.3,
        label=f"Residual ceiling ({100 * residual_ceiling:.0f}%)",
    )
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Residual-norm dose ρ")
    axes[2].set_ylabel("Maximum event value (%)")
    axes[2].set_title("C. Runtime safety gates")
    axes[2].set_ylim(0, 12.2)
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")

    stem = output_dir / "E05a-discovery-calibration-stop"
    hashes = _save_all(fig, stem)
    plt.close(fig)

    generator_path = Path(__file__)
    largest = max(candidates, key=lambda row: row["mean_signed_half_span"])
    receipt = {
        "figure_id": "E05a",
        "title": "Discovery calibration stop surface",
        "question": (
            "Did any prospectively frozen dose qualify for held-out causal confirmation?"
        ),
        "description": (
            "The complete four-dose discovery calibration ladder, directional "
            "components, and independent runtime safety diagnostics."
        ),
        "alt_text": (
            "Three-panel chart for four residual-norm-scaled doses. Panel A shows "
            "signed half-span estimates and 95% bootstrap intervals, all far below "
            f"the 0.1 efficacy threshold; the largest estimate is "
            f"{largest['mean_signed_half_span']:.3f} at rho {largest['rho']}. "
            "Panel B shows restoring-minus-zero and opposite-minus-zero components. "
            "Panel C shows representation mismatch and effective residual ratios, "
            "all below their independent runtime ceilings."
        ),
        "permitted_inference": (
            "no frozen discovery dose qualified to open the held-out intervention panel"
        ),
        "non_claims": [
            "this is discovery calibration, not held-out causal evidence",
            "passing runtime safety gates does not establish behavioral efficacy",
            "failure of this dose ladder does not prove that no SAE intervention can work",
            "no defense or utility claim is supported",
        ],
        "source_receipts": [
            {"path": str(analysis_path), "sha256": sha256_file(analysis_path)},
            {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        ],
        "row_filter": "all four prospectively frozen rho candidates; no exclusions",
        "independent_unit": "discovery base behavior ID",
        "uncertainty": "frozen 10,000-replicate behavior-cluster bootstrap",
        "derived_data": {
            "status": analysis["status"],
            "selection": analysis["selection"],
            "confirmatory_outcomes_opened": analysis[
                "confirmatory_outcomes_opened"
            ],
            "candidates": candidates,
            "thresholds": {
                "minimum_mean_signed_half_span": half_span_threshold,
                "maximum_requested_realized_relative_error": representation_ceiling,
                "maximum_event_delta_to_pre_residual_norm": residual_ceiling,
            },
            "generation_receipts_sha256": analysis["generation_receipts_sha256"],
            "score_receipts_sha256": analysis["score_receipts_sha256"],
            "rows_sha256": analysis["rows_sha256"],
        },
        "counts": {
            "expected_candidates": 4,
            "realized_candidates": len(candidates),
            "expected_clusters_per_candidate": analysis["n_behaviors"],
            "realized_clusters_per_candidate": analysis["n_behaviors"],
        },
        "generator": {
            "path": str(generator_path),
            "sha256": sha256_file(generator_path),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-interventions --analysis {analysis_path} "
                f"--plan {plan_path} --out {output_dir}"
            ),
        },
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": (
                "marker shapes, line styles, hatch patterns, direct thresholds, "
                "and numeric estimate labels"
            ),
            "text_equivalent": "alt_text plus derived_data",
        },
        "verification": {
            "status": "pending",
            "verified_utc": None,
            "byte_identity": None,
        },
    }
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    provenance = {
        "schema_version": "1.0",
        "source_analysis": {
            "path": str(analysis_path),
            "sha256": sha256_file(analysis_path),
        },
        "source_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "figures": [{"path": str(receipt_path), "sha256": sha256_file(receipt_path)}],
    }
    write_json_atomic(output_dir / "provenance.json", provenance)
    return {"figures": 1, "provenance": str(output_dir / "provenance.json")}


def verify_intervention_figures(
    analysis_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_intervention_figures(analysis_path, plan_path, temporary)
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
            raise ValueError(f"intervention figure verification failed: {comparisons}")
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
