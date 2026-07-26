from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .figures import ARM_LABELS, COLORS, _configure, _save_all
from .hashing import sha256_file, write_json_atomic

ARM_ORDER = ("base", "inert_length", "structural_sham", "full")
NEGATIVE_ARMS = ("base", "inert_length", "structural_sham")


def _primary_feature(result: dict) -> dict:
    matches = [row for row in result["features"] if row["feature_id"] == 10146]
    if len(matches) != 1:
        raise ValueError("expected exactly one feature-10146 row")
    return matches[0]


def generate_replay_figure(result_path: Path, output_dir: Path) -> dict:
    _configure()
    result = json.loads(result_path.read_text())
    feature = _primary_feature(result)
    gate = result["candidate_gate"]
    if gate["passed"]:
        raise ValueError("this figure contract is for the frozen candidate-gate stop")

    prevalences = [feature["arms"][arm]["prevalence"] for arm in ARM_ORDER]
    positive_counts = [feature["arms"][arm]["positive_count"] for arm in ARM_ORDER]
    arm_counts = [feature["arms"][arm]["count"] for arm in ARM_ORDER]
    contrast_lookup = {
        row["contrast"]: row for row in feature["paired_contrasts"]
    }
    contrasts = [contrast_lookup[f"full-minus-{arm}"] for arm in NEGATIVE_ARMS]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    bars = axes[0].bar(
        np.arange(len(ARM_ORDER)),
        prevalences,
        color=[COLORS[arm] for arm in ARM_ORDER],
        edgecolor="white",
        linewidth=0.8,
        hatch=("", "..", "//", "xx"),
    )
    axes[0].axhspan(0, 0.1, color="#16A34A", alpha=0.08)
    axes[0].axhline(0.1, color="#16A34A", linestyle="--", linewidth=1.1)
    axes[0].axhline(0.9, color="#D97706", linestyle=":", linewidth=1.1)
    axes[0].set_xticks(
        np.arange(len(ARM_ORDER)),
        [ARM_LABELS[arm] for arm in ARM_ORDER],
        rotation=18,
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Feature 10146 prevalence")
    axes[0].set_title("Inert length activates the feature")
    for bar, positive, count in zip(bars, positive_counts, arm_counts, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{positive}/{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x = np.arange(len(contrasts))
    means = np.array([row["paired_mean"] for row in contrasts])
    lower = np.array([row["bootstrap_95_interval"][0] for row in contrasts])
    upper = np.array([row["bootstrap_95_interval"][1] for row in contrasts])
    axes[1].errorbar(
        x,
        means,
        yerr=np.vstack([means - lower, upper - means]),
        fmt="D",
        color="#111827",
        markerfacecolor="#D97706",
        markeredgecolor="#111827",
        capsize=5,
        linewidth=1.8,
    )
    axes[1].axhline(0, color="#6B7280", linewidth=1, linestyle="--")
    axes[1].set_xticks(
        x,
        [f"Full − {ARM_LABELS[arm].lower()}" for arm in NEGATIVE_ARMS],
        rotation=18,
    )
    axes[1].set_ylim(0, max(upper) * 1.12)
    axes[1].set_ylabel("Paired mean activation difference")
    axes[1].set_title("Full remains stronger than every control")

    stem = output_dir / "E05b-feature-10146-four-arm-replay"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    hashes["svg"] = sha256_file(svg_path)

    receipt = {
        "figure_id": "E05b",
        "title": "Feature 10146 four-arm discovery replay",
        "question": (
            "Does harmful content, inert formatting or length, matched structure, "
            "or the full scaffold activate feature 10146?"
        ),
        "description": (
            "Four-arm activation prevalence and paired full-minus-control activation "
            "contrasts for the 20 preserved discovery behavior IDs."
        ),
        "alt_text": (
            "Two-panel chart. Feature 10146 activates in zero of 20 base cases, "
            "14 of 20 inert-length cases, zero of 20 structural-sham cases, and "
            "20 of 20 full-scaffold cases. Full activation remains substantially "
            "higher than every control, but the inert-length prevalence violates "
            "the frozen non-full ceiling."
        ),
        "permitted_inference": (
            "discovery-only four-arm feature fingerprint with demonstrated "
            "formatting or length sensitivity"
        ),
        "non_claims": [
            "not an attack-family-specific detector",
            "not independent of formatting or length",
            "not held-out evidence",
            "not a causal mechanism or defense",
        ],
        "source_receipts": [
            {"path": str(result_path), "sha256": sha256_file(result_path)}
        ],
        "independent_unit": "behavior ID",
        "uncertainty": "10,000-replicate paired bootstrap over 20 behavior IDs",
        "counts": {
            "expected_clusters": 20,
            "realized_clusters": result["behavior_count"],
            "expected_primary_observations": 80,
            "realized_primary_observations": sum(arm_counts),
        },
        "derived_data": {
            "feature_id": 10146,
            "arm_summaries": feature["arms"],
            "full_minus_negative_contrasts": contrasts,
            "candidate_gate": gate,
        },
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-replay --result {result_path} "
                f"--out {output_dir}"
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
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    provenance = {
        "schema_version": "1.0",
        "source_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "figures": [{"path": str(receipt_path), "sha256": sha256_file(receipt_path)}],
    }
    write_json_atomic(output_dir / "provenance.replay.json", provenance)
    return {"figures": 1, "provenance": str(output_dir / "provenance.replay.json")}


def verify_replay_figure(result_path: Path, output_dir: Path) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_replay_figure(result_path, temporary)
        expected_path = output_dir / "E05b-feature-10146-four-arm-replay.receipt.json"
        actual_path = temporary / expected_path.name
        expected = json.loads(expected_path.read_text())
        actual = json.loads(actual_path.read_text())
        comparisons = [
            {
                "output_type": output_type,
                "byte_identical": (
                    expected["outputs"][output_type]["sha256"]
                    == actual["outputs"][output_type]["sha256"]
                ),
            }
            for output_type in ("svg", "png", "pdf")
        ]
        if not all(row["byte_identical"] for row in comparisons):
            raise ValueError(f"replay figure byte verification failed: {comparisons}")
        expected["verification"] = {
            "status": "verified",
            "verified_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "byte_identity": comparisons,
        }
        write_json_atomic(expected_path, expected)
        provenance_path = output_dir / "provenance.replay.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["figures"][0]["sha256"] = sha256_file(expected_path)
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
