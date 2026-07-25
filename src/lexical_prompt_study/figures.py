from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .hashing import sha256_file, write_json_atomic

ARM_ORDER = ("base", "inert_length", "structural_sham", "full")
ARM_LABELS = {
    "base": "Base",
    "inert_length": "Inert length",
    "structural_sham": "Structural sham",
    "full": "Full scaffold",
}
COLORS = {
    "base": "#6B7280",
    "inert_length": "#7C3AED",
    "structural_sham": "#0284C7",
    "full": "#D97706",
}


def _configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "lexical-prompt-study-v1",
        }
    )


def _save_all(fig, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "pdf": stem.with_suffix(".pdf"),
    }
    fig.savefig(paths["svg"], metadata={"Date": None})
    fig.savefig(paths["png"], dpi=300, metadata={"Software": "lexical-prompt-study"})
    fig.savefig(
        paths["pdf"],
        metadata={
            "Creator": "lexical-prompt-study",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    return {name: sha256_file(path) for name, path in paths.items()}


def _bootstrap_arm_interval(rows: list[dict], arm: str, field: str, seed: int) -> list[float]:
    values = np.array([float(row[field]) for row in rows if row["arm"] == arm])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def generate_behavior_figures(gate_path: Path, output_dir: Path) -> dict:
    _configure()
    gate = json.loads(gate_path.read_text())
    rows = gate["source_receipts"]
    source_hash = sha256_file(gate_path)
    generator_hash = sha256_file(Path(__file__))
    seed = gate["primary_contrast"]["bootstrap_seed"]
    receipts = []

    # E01: arm distributions and paired focal contrast.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    rng = np.random.default_rng(20260725)
    plotted = []
    for index, arm in enumerate(ARM_ORDER):
        arm_rows = [row for row in rows if row["arm"] == arm]
        values = np.array([row["score"] for row in arm_rows])
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axes[0].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[arm],
            marker=("o", "s", "^", "D")[index],
            alpha=0.55,
            s=22,
            edgecolor="white",
            linewidth=0.3,
        )
        interval = _bootstrap_arm_interval(rows, arm, "score", seed)
        mean = float(values.mean())
        axes[0].errorbar(
            index,
            mean,
            yerr=[[mean - interval[0]], [interval[1] - mean]],
            fmt="_",
            markersize=18,
            color="black",
            capsize=4,
            linewidth=1.7,
        )
        plotted.append({"arm": arm, "mean": mean, "bootstrap_95_interval": interval})
    axes[0].set_xticks(range(len(ARM_ORDER)), [ARM_LABELS[arm] for arm in ARM_ORDER], rotation=20)
    axes[0].set_ylabel("HarmBench affirmative probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Turn-2 score by arm")

    paired = gate["paired_values"]
    paired_values = np.array([row["full_minus_sham"] for row in paired])
    axes[1].axhline(0, color="#6B7280", linewidth=1, linestyle="--")
    axes[1].scatter(
        np.ones(len(paired_values)) + rng.uniform(-0.08, 0.08, len(paired_values)),
        paired_values,
        color="#111827",
        marker="o",
        alpha=0.55,
        s=22,
    )
    estimate = gate["primary_contrast"]["estimate"]
    interval = gate["primary_contrast"]["bootstrap_95_interval"]
    axes[1].errorbar(
        1,
        estimate,
        yerr=[[estimate - interval[0]], [interval[1] - estimate]],
        fmt="D",
        color="#D97706",
        markeredgecolor="black",
        capsize=5,
        linewidth=2,
    )
    axes[1].set_xlim(0.5, 1.5)
    axes[1].set_xticks([1], ["Full − sham"])
    axes[1].set_ylabel("Paired probability difference")
    axes[1].set_title("Behavior-level paired effect")
    stem = output_dir / "E01-full-vs-sham"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    receipt = {
        "figure_id": "E01",
        "title": "Full scaffold versus structural sham",
        "question": "Does the scaffold change turn-2 harmful compliance?",
        "alt_text": (
            f"Two-panel chart across {gate['n_behaviors']} behavior IDs. The left panel "
            "shows every arm score and its cluster-bootstrap mean interval. The right "
            f"shows paired full-minus-sham values; mean {estimate:.3f}, 95% interval "
            f"[{interval[0]:.3f}, {interval[1]:.3f}]."
        ),
        "permitted_inference": "paired behavioral effect on the pinned split and evaluator",
        "non_claims": ["no mechanism inference", "no cross-model generalization"],
        "source_receipts": [{"path": str(gate_path), "sha256": source_hash}],
        "row_filter": f"split={gate['split']}, turn=2, all four arms",
        "independent_unit": "base behavior ID",
        "uncertainty": "10,000-replicate cluster bootstrap over behavior IDs",
        "derived_data": {"arm_summaries": plotted, "primary_contrast": gate["primary_contrast"]},
        "generator": {"path": str(Path(__file__)), "sha256": generator_hash},
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": "distinct marker shapes, direct arm labels, zero reference line",
            "text_equivalent": "alt_text plus derived_data",
        },
    }
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})

    # E02: response phase outcomes.
    metrics = ("refusal", "divider", "post_divider", "truncated")
    metric_labels = ("Refusal", "Divider", "Post-divider body", "Truncated")
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    width = 0.18
    x = np.arange(len(metrics))
    derived = []
    for arm_index, arm in enumerate(ARM_ORDER):
        arm_rows = [row for row in rows if row["arm"] == arm]
        values = [float(np.mean([row[metric] for row in arm_rows])) for metric in metrics]
        positions = x + (arm_index - 1.5) * width
        ax.bar(
            positions,
            values,
            width,
            label=ARM_LABELS[arm],
            color=COLORS[arm],
            hatch=("", "..", "//", "xx")[arm_index],
            edgecolor="white",
            linewidth=0.7,
        )
        derived.extend(
            {"arm": arm, "metric": metric, "proportion": value}
            for metric, value in zip(metrics, values, strict=True)
        )
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Proportion of behavior IDs")
    ax.set_title("Turn-2 response phases by arm")
    ax.legend(frameon=False, ncol=2)
    stem = output_dir / "E02-response-phases"
    hashes = _save_all(fig, stem)
    plt.close(fig)
    receipt = {
        "figure_id": "E02",
        "title": "Response-phase outcomes",
        "question": "Is the effect refusal, divider production, a post-divider body, or truncation?",
        "alt_text": (
            f"Grouped bar chart for {gate['n_behaviors']} behavior IDs showing refusal, "
            "divider, post-divider-body, and truncation proportions for all four arms."
        ),
        "permitted_inference": "descriptive phase rates on the pinned split",
        "non_claims": ["divider production is not harmful compliance"],
        "source_receipts": [{"path": str(gate_path), "sha256": source_hash}],
        "row_filter": f"split={gate['split']}, turn=2, all four arms",
        "independent_unit": "base behavior ID",
        "derived_data": derived,
        "generator": {"path": str(Path(__file__)), "sha256": generator_hash},
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": "distinct hatch patterns and direct metric labels",
            "text_equivalent": "alt_text plus derived_data",
        },
    }
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    receipts.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path)})
    provenance = {
        "schema_version": "1.0",
        "source_gate": {"path": str(gate_path), "sha256": source_hash},
        "figures": receipts,
    }
    write_json_atomic(output_dir / "provenance.json", provenance)
    return {"figures": len(receipts), "provenance": str(output_dir / "provenance.json")}


def verify_behavior_figures(gate_path: Path, output_dir: Path) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_behavior_figures(gate_path, temporary)
        expected = json.loads((output_dir / "provenance.json").read_text())
        actual = json.loads((temporary / "provenance.json").read_text())
        comparisons = []
        for expected_figure, actual_figure in zip(
            expected["figures"], actual["figures"], strict=True
        ):
            expected_receipt = json.loads(Path(expected_figure["path"]).read_text())
            actual_receipt = json.loads(Path(actual_figure["path"]).read_text())
            for output_type in ("svg", "png", "pdf"):
                expected_hash = expected_receipt["outputs"][output_type]["sha256"]
                actual_hash = actual_receipt["outputs"][output_type]["sha256"]
                comparisons.append(
                    {
                        "figure_id": expected_receipt["figure_id"],
                        "output_type": output_type,
                        "byte_identical": expected_hash == actual_hash,
                    }
                )
        if not all(item["byte_identical"] for item in comparisons):
            raise ValueError(f"figure byte verification failed: {comparisons}")
    return {"status": "verified", "comparisons": comparisons}
