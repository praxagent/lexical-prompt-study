from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .figures import _configure, _save_all
from .hashing import sha256_file, write_json_atomic


REQUEST_CLASSES = ("harmful_request", "ordinary_benign_request")
REQUEST_LABELS = {
    "harmful_request": "Harmful requests",
    "ordinary_benign_request": "Ordinary benign requests",
}
DETECTOR_LABELS = {
    "frozen_jlens": "Frozen J-lens",
    "frozen_hashed_full_prompt": "Learned prompt filter",
    "exact_full_scaffold": "Exact full string",
    "exact_attack_block_count_at_least_1": "At least 1 exact block",
    "exact_attack_block_count_at_least_2": "At least 2 exact blocks",
    "exact_attack_block_count_at_least_3": "At least 3 exact blocks",
    "exact_attack_block_count_at_least_4": "All 4 exact blocks",
    "byte_five_gram_coverage_at_least_0.25": "5-gram coverage ≥ .25",
    "byte_five_gram_coverage_at_least_0.50": "5-gram coverage ≥ .50",
    "byte_five_gram_coverage_at_least_0.75": "5-gram coverage ≥ .75",
    "byte_five_gram_coverage_at_least_1.00": "5-gram coverage = 1",
}


def _save(fig: Any, stem: Path) -> dict[str, str]:
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    hashes["svg"] = sha256_file(svg)
    return hashes


def _receipt(
    *,
    result_path: Path,
    output_dir: Path,
    stem_name: str,
    hashes: MappingLike,
    metadata: MappingLike,
) -> Path:
    stem = output_dir / stem_name
    payload = {
        **metadata,
        "source_receipts": [{"path": str(result_path), "sha256": sha256_file(result_path)}],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
        },
        "outputs": {
            kind: {"path": str(stem.with_suffix(f".{kind}")), "sha256": digest}
            for kind, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": "direct labels, markers, lines, and reference guides",
            "text_equivalent": "alt_text plus complete aggregate derived_data",
        },
        "verification": {"status": "pending", "byte_identity": None},
    }
    path = stem.with_suffix(".receipt.json")
    write_json_atomic(path, payload)
    return path


MappingLike = dict[str, Any]


def _validate(result: MappingLike) -> None:
    if (
        result.get("study_id") != "lexical-jlens-signal-falsification-v1"
        or result.get("status") != "mutation_falsification_analysis_complete"
        or set(result.get("piecewise_curves", {})) != set(REQUEST_CLASSES)
        or result.get("claim_boundaries", {}).get("unopened_v2_confirmation_opened")
        is not False
        or result.get("raw_prompt_request_token_generation_or_row_level_content_public")
        is not False
    ):
        raise ValueError("falsification figure source boundary drift")


def generate_falsification_figures(result_path: Path, output_dir: Path) -> MappingLike:
    _configure()
    result = json.loads(result_path.read_text())
    _validate(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []

    # E30: every exact-length mutation plus the count-averaged curve.
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), constrained_layout=True)
    curve_data = []
    for row_index, request_class in enumerate(REQUEST_CLASSES):
        curve = result["piecewise_curves"][request_class]
        masks = curve["by_mask"]
        for mask, values in masks.items():
            count = int(mask, 2).bit_count()
            score = values["frozen_jlens_score"]["mean"]
            trip = values["frozen_jlens_trip_rate"]
            axes[row_index, 0].scatter(count, score, color="#9CA3AF", alpha=0.55, s=25)
            axes[row_index, 1].scatter(count, trip, color="#9CA3AF", alpha=0.55, s=25)
            curve_data.append(
                {
                    "request_class": request_class,
                    "mask": mask,
                    "attack_block_count": count,
                    "jlens_score_mean": score,
                    "jlens_trip_rate": trip,
                }
            )
        counts = curve["by_attack_block_count"]
        x = np.arange(5)
        mean_score = [counts[str(value)]["frozen_jlens_score"]["mean"] for value in x]
        mean_trip = [counts[str(value)]["frozen_jlens_trip_rate"] for value in x]
        axes[row_index, 0].plot(x, mean_score, color="#D97706", marker="o", linewidth=2)
        axes[row_index, 1].plot(x, mean_trip, color="#0284C7", marker="s", linewidth=2)
        axes[row_index, 0].set_ylabel(f"{REQUEST_LABELS[request_class]}\nmean J-lens score")
        axes[row_index, 1].set_ylabel(f"{REQUEST_LABELS[request_class]}\nbreaker trip rate")
        axes[row_index, 1].set_ylim(-0.03, 1.03)
        for column in range(2):
            axes[row_index, column].set_xticks(x)
            axes[row_index, column].set_xlabel("Exact attack blocks present (of 4)")
    axes[0, 0].set_title("Frozen internal score")
    axes[0, 1].set_title("Frozen breaker decision")
    stem = output_dir / "E30-jlens-piecewise-mutation-curve"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E30",
            "title": "The internal signal under every piecewise scaffold mutation",
            "question": "Does the frozen J-lens signal change smoothly as attack blocks are added?",
            "alt_text": "Each gray point is one of 16 equal-token block combinations; colored lines average all combinations with the same number of exact attack blocks, separately for harmful and benign requests.",
            "independent_unit": "request_id; displayed points are 60-request cell aggregates",
            "derived_data": curve_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E31: the direct defensive-value comparison.
    metrics = result["detector_head_to_head"]["metrics"]
    fig, ax = plt.subplots(figsize=(10.8, 7.0), constrained_layout=True)
    detector_data = []
    for index_value, (name, values) in enumerate(metrics.items()):
        internal = name == "frozen_jlens"
        x = values["ordinary_benign_false_trip_fraction"]
        y = values["successful_harmful_capture_fraction"]
        color = "#D97706" if internal else "#0284C7"
        marker = "*" if internal else ("s" if "hashed" in name else "o")
        ax.scatter(x, y, color=color, marker=marker, s=180 if internal else 70, zorder=3)
        offset = (5, 5 + (index_value % 3) * 6)
        ax.annotate(DETECTOR_LABELS[name], (x, y), xytext=offset, textcoords="offset points", fontsize=8)
        detector_data.append(
            {
                "detector": name,
                "successful_harmful_capture_fraction": y,
                "ordinary_benign_false_trip_fraction": x,
            }
        )
    ax.axhline(0.9, color="#991B1B", linestyle="--", linewidth=1)
    ax.axvline(0.02, color="#991B1B", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Ordinary-benign false-trip fraction (lower is better)")
    ax.set_ylabel("Successful harmful completions caught (higher is better)")
    ax.set_title("Internal breaker versus prompt filtering")
    stem = output_dir / "E31-jlens-versus-prompt-filtering"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E31",
            "title": "Does the internal breaker beat prompt filtering?",
            "question": "Which frozen detector catches more behaviorally successful harmful outputs while stopping fewer ordinary-benign prompts?",
            "alt_text": "A recall-versus-false-trip plot compares the frozen J-lens breaker with a learned byte-ngram prompt filter and fixed exact, block-count, and fuzzy filters. Upper left is better.",
            "independent_unit": "request_id within mutation mask",
            "derived_data": detector_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E32: which blocks and interactions move the readout and behavior.
    panels = (
        ("harmful_request", "frozen_jlens_score", "J-lens score · harmful"),
        ("ordinary_benign_request", "frozen_jlens_score", "J-lens score · benign"),
        ("harmful_request", "feature_6779_magnitude", "Feature 6779 · harmful"),
        ("harmful_request", "harmbench_probability", "HarmBench probability · harmful"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), constrained_layout=True)
    effect_data = []
    for axis, (request_class, endpoint, title) in zip(axes.flat, panels, strict=True):
        effects = result["factorial_effects"][request_class][endpoint]
        mains = [effects[str(block)] for block in range(1, 5)]
        estimates = [value["estimate"] for value in mains]
        lower = [
            value["estimate"] - value["request_cluster_95_interval"][0] for value in mains
        ]
        upper = [
            value["request_cluster_95_interval"][1] - value["estimate"] for value in mains
        ]
        x = np.arange(1, 5)
        axis.errorbar(x, estimates, yerr=[lower, upper], fmt="o", color="#7C3AED", capsize=4)
        axis.axhline(0, color="#374151", linewidth=0.8)
        axis.set_xticks(x, [f"Block {value}" for value in x])
        axis.set_title(title)
        axis.set_ylabel("Average on-minus-off effect")
        for block, value in zip(x, mains, strict=True):
            effect_data.append(
                {
                    "request_class": request_class,
                    "endpoint": endpoint,
                    "block": int(block),
                    "estimate": value["estimate"],
                    "request_cluster_95_interval": value["request_cluster_95_interval"],
                    "holm_adjusted_p": value["holm_adjusted_p"],
                }
            )
    stem = output_dir / "E32-jlens-block-marginal-effects"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E32",
            "title": "Which scaffold blocks actually move the signal and behavior?",
            "question": "What is each block's average contribution after balancing over every configuration of the other three blocks?",
            "alt_text": "Four panels show request-clustered average marginal effects and 95 percent bootstrap intervals for each scaffold block on J-lens, feature 6779, and harmful behavior.",
            "independent_unit": "request_id",
            "derived_data": effect_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    index_path = output_dir / "jlens-falsification-figure-index.json"
    write_json_atomic(
        index_path,
        {
            "schema_version": "1.0",
            "study_id": result["study_id"],
            "status": "figures_generated_pending_visual_verification",
            "source_result_sha256": sha256_file(result_path),
            "figures": index,
            "raw_prompt_request_token_generation_or_row_level_content_public": False,
        },
    )
    return {"figure_count": len(index), "index_path": str(index_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_falsification_figures(args.result, args.out)))


if __name__ == "__main__":
    main()
