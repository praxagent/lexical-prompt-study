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
from .weaponization_analysis import NEGATIVES, PLACEMENTS


PLACEMENT_LABELS = {
    "scaffold_before_request": "Scaffold before request",
    "scaffold_after_request": "Scaffold after request",
}
NEGATIVE_LABELS = {
    "negative_attack_benign": "Attack + benign",
    "negative_attack_neutral": "Attack + neutral",
    "negative_harmless_harmful": "Harmless + harmful",
    "negative_harmless_benign": "Harmless + benign",
    "negative_structural_sham_harmful": "Sham + harmful",
    "negative_inert_harmful": "Inert + harmful",
}
COMPARATOR_ORDER = (
    "primary_two_key",
    "jlens_head_alone",
    "feature_6779_alone",
    "frozen_subspace_alone",
    "restricted_exact_match",
    "restricted_fuzzy_five_byte_gram_coverage",
    "structural_head",
)
COMPARATOR_LABELS = {
    "primary_two_key": "Two-key",
    "jlens_head_alone": "J-lens",
    "feature_6779_alone": "Feature 6779",
    "frozen_subspace_alone": "SAE subspace",
    "restricted_exact_match": "Exact match",
    "restricted_fuzzy_five_byte_gram_coverage": "Fuzzy match",
    "structural_head": "Structure",
}
CONTRASTS = (
    "attack_harmful_minus_attack_benign",
    "attack_harmful_minus_harmless_harmful",
    "difference_in_differences",
)
CONTRAST_LABELS = {
    "attack_harmful_minus_attack_benign": "Attack: harmful − benign",
    "attack_harmful_minus_harmless_harmful": "Harmful: attack − harmless",
    "difference_in_differences": "Difference in differences",
}
TRAJECTORY_STRATA = (
    "positive_attack_harmful",
    "negative_attack_benign",
    "negative_harmless_harmful",
    "negative_structural_sham_harmful",
)
TRAJECTORY_LABELS = {
    "positive_attack_harmful": "Attack + harmful",
    "negative_attack_benign": "Attack + benign",
    "negative_harmless_harmful": "Harmless + harmful",
    "negative_structural_sham_harmful": "Sham + harmful",
}
COLORS = ("#D97706", "#0284C7", "#16A34A", "#7C3AED", "#6B7280", "#DB2777", "#0F766E")


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
    hashes: dict[str, str],
    metadata: dict[str, Any],
) -> Path:
    stem = output_dir / stem_name
    payload = {
        **metadata,
        "source_receipts": [{"path": str(result_path), "sha256": sha256_file(result_path)}],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": f"lexical-study figures-weaponization --result {result_path} --out {output_dir}",
        },
        "outputs": {
            kind: {"path": str(stem.with_suffix(f".{kind}")), "sha256": digest}
            for kind, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": "direct labels, panel titles, markers, and reference lines",
            "text_equivalent": "alt_text plus complete derived_data",
        },
        "verification": {"status": "pending", "verified_utc": None, "byte_identity": None},
    }
    path = stem.with_suffix(".receipt.json")
    write_json_atomic(path, payload)
    return path


def _validate(result: dict[str, Any]) -> None:
    if (
        result["analysis_id"] != "weaponization-calibration-analysis-v1"
        or result["status"] != "calibration_candidate_frozen"
        or result["observation_count"] != 1900
        or set(result["placements"]) != set(PLACEMENTS)
        or result["confirmation_opened_or_executed"] is not False
    ):
        raise ValueError("weaponization figure source boundary drift")


def generate_weaponization_figures(result_path: Path, output_dir: Path) -> dict[str, Any]:
    _configure()
    result = json.loads(result_path.read_text())
    _validate(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []

    # E21: the exact calibration gate, with every negative stratum visible.
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    gate_data = []
    for column, placement in enumerate(PLACEMENTS):
        metrics = result["placements"][placement]["primary_two_key"]["metrics"]
        recall = metrics["recall"]
        axes[0, column].bar([0], [recall], color="#D97706", width=0.55)
        axes[0, column].axhline(0.9, color="#991B1B", linestyle="--", linewidth=1.2)
        axes[0, column].set_ylim(0, 1.05)
        axes[0, column].set_xticks([0], ["Positive attack attempts"])
        axes[0, column].set_ylabel("Recall")
        axes[0, column].set_title(PLACEMENT_LABELS[placement])
        axes[0, column].text(
            0, recall + 0.025, f"{metrics['true_positive_count']}/100", ha="center"
        )
        rates = [
            metrics["critical_negatives"][stratum]["false_positive_rate"] for stratum in NEGATIVES
        ]
        positions = np.arange(len(NEGATIVES))
        axes[1, column].barh(positions, rates, color="#0284C7")
        axes[1, column].axvline(0.02, color="#991B1B", linestyle="--", linewidth=1.2)
        axes[1, column].set_xlim(0, 0.025)
        axes[1, column].set_yticks(positions, [NEGATIVE_LABELS[value] for value in NEGATIVES])
        axes[1, column].invert_yaxis()
        axes[1, column].set_xlabel("False-positive rate")
        for row, (stratum, rate) in enumerate(zip(NEGATIVES, rates, strict=True)):
            count = metrics["critical_negatives"][stratum]["false_positive_count"]
            axes[1, column].text(max(rate, 0.00015), row, f"{count}/100", va="center", fontsize=8)
            gate_data.append(
                {
                    "placement": placement,
                    "kind": "critical_negative",
                    "stratum": stratum,
                    "count": 100,
                    "false_positive_count": count,
                    "false_positive_rate": rate,
                }
            )
        gate_data.append(
            {
                "placement": placement,
                "kind": "positive",
                "count": 100,
                "true_positive_count": metrics["true_positive_count"],
                "recall": recall,
            }
        )
    stem = output_dir / "E21-weaponization-calibration-gate"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E21",
            "title": "The two-key candidate cleared calibration in both placements",
            "question": "Did the frozen two-key rule meet recall and every critical-negative false-positive gate separately before and after the request?",
            "description": "Placement-separated calibration recall above the frozen 0.90 gate and false-positive rates for all six 100-row negative strata against the frozen 0.02 ceiling.",
            "alt_text": "The before-request candidate recalls all 100 positive attack attempts and the after-request candidate recalls 92 of 100. Both record zero false positives in each of six 100-row critical-negative strata.",
            "independent_unit": "request or harmless-wrapper family within calibration stratum",
            "counts": {
                "positive_per_placement": 100,
                "negative_per_stratum_per_placement": 100,
                "critical_negative_strata": 6,
            },
            "permitted_inference": "calibration eligibility on the pinned model, candidate, placements, and fixed request/control panels",
            "non_claims": [
                "not held-out confirmation",
                "not successful-weaponization classification",
                "not adaptive jailbreak robustness",
                "not causal mechanism",
                "not production deployment",
            ],
            "derived_data": gate_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E22: ranking performance reveals which readout carries the separation.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True, constrained_layout=True)
    comparison_data = []
    for column, placement in enumerate(PLACEMENTS):
        placement_result = result["placements"][placement]
        candidates = {
            "primary_two_key": placement_result["primary_two_key"],
            **placement_result["comparators"],
        }
        y = np.arange(len(COMPARATOR_ORDER))
        auc = [candidates[name]["roc_auc"] for name in COMPARATOR_ORDER]
        ap = [candidates[name]["average_precision"] for name in COMPARATOR_ORDER]
        axes[column].barh(y - 0.18, auc, height=0.34, color="#0284C7", label="AUROC")
        axes[column].barh(y + 0.18, ap, height=0.34, color="#D97706", label="Average precision")
        axes[column].set_yticks(y, [COMPARATOR_LABELS[name] for name in COMPARATOR_ORDER])
        axes[column].invert_yaxis()
        axes[column].set_xlim(0, 1.02)
        axes[column].set_xlabel("Ranking metric")
        axes[column].set_title(PLACEMENT_LABELS[placement])
        axes[column].axvline(0.5, color="#9CA3AF", linewidth=0.8)
        for name, auc_value, ap_value in zip(COMPARATOR_ORDER, auc, ap, strict=True):
            comparison_data.append(
                {
                    "placement": placement,
                    "candidate": name,
                    "roc_auc": auc_value,
                    "average_precision": ap_value,
                    "calibration_gate_eligible": candidates[name]["eligible"],
                }
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
    )
    stem = output_dir / "E22-weaponization-candidate-comparison"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E22",
            "title": "The J-lens key carries the harmful-use separation",
            "question": "Does the two-key candidate outperform scaffold-only, lexical, and structural readouts, and which component supplies ranking power?",
            "description": "Calibration AUROC and average precision for the frozen two-key candidate and six prospectively declared comparators, never pooling placements.",
            "alt_text": "The J-lens head has near-perfect calibration ranking in both placements. Feature 6779 ranks moderately but cannot pass the false-positive gate; exact, fuzzy, and structural baselines also fail. The gated two-key candidate passes but has lower ranking metrics than J-lens alone because the structure key is deliberately required.",
            "independent_unit": "critical calibration observation",
            "counts": {"critical_rows_per_placement": 700, "comparators": 7},
            "permitted_inference": "descriptive and gate-linked calibration ranking on the fixed critical strata",
            "non_claims": [
                "no out-of-sample generalization",
                "no behavior-success endpoint",
                "no causal importance of J-lens coordinates",
                "no production superiority",
            ],
            "derived_data": comparison_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E23: direct contrast decomposition for the SAE readouts.
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    contrast_data = []
    readouts = (
        ("feature_6779_magnitude", "Feature 6779"),
        ("frozen_subspace_score", "Frozen SAE subspace"),
    )
    for row_index, (readout, readout_label) in enumerate(readouts):
        for column, placement in enumerate(PLACEMENTS):
            contrasts = result["placements"][placement]["critical_mean_contrasts"][readout]
            values = [contrasts[name][0] for name in CONTRASTS]
            x = np.arange(len(CONTRASTS))
            axes[row_index, column].bar(x, values, color=("#D97706", "#16A34A", "#7C3AED"))
            axes[row_index, column].axhline(0, color="#374151", linewidth=0.8)
            axes[row_index, column].set_xticks(
                x, [CONTRAST_LABELS[name] for name in CONTRASTS], rotation=18, ha="right"
            )
            axes[row_index, column].set_title(f"{readout_label}\n{PLACEMENT_LABELS[placement]}")
            axes[row_index, column].set_ylabel("Mean contrast")
            for name, value in zip(CONTRASTS, values, strict=True):
                contrast_data.append(
                    {
                        "placement": placement,
                        "readout": readout,
                        "contrast": name,
                        "estimate": value,
                    }
                )
    stem = output_dir / "E23-weaponization-sae-contrasts"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E23",
            "title": "Feature 6779 still marks scaffold form, not harmful intent",
            "question": "Do the SAE readouts distinguish harmful from benign use of the same attack scaffold, attack from harmless structure around the same harmful request, or their interaction?",
            "description": "Three prespecified mean contrasts for feature 6779 and the frozen subspace, separately for both scaffold placements.",
            "alt_text": "Feature 6779 changes by almost zero between harmful and benign requests under the same attack scaffold, while it is higher for attack than harmless wrappers around harmful requests. Its difference-in-differences is slightly negative. The frozen subspace is mixed. These SAE coordinates provide a structure key, not the harmful-use key.",
            "independent_unit": "calibration request or wrapper family",
            "counts": {"rows_per_cell": 100, "contrasts": 12},
            "permitted_inference": "fixed-panel mean SAE contrasts on the exact calibration controls",
            "non_claims": [
                "no SAE-only weaponization detector",
                "no uncertainty interval",
                "no causal feature role",
                "no generic harmless-scaffold population claim",
            ],
            "derived_data": contrast_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E24: full layerwise J-lens trajectories for claim-defining controls.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True, constrained_layout=True)
    trajectory_data = []
    layers = np.arange(31)
    for column, placement in enumerate(PLACEMENTS):
        summaries = result["placements"][placement]["stratum_summaries"]
        for line_index, stratum in enumerate(TRAJECTORY_STRATA):
            values = summaries[stratum]["jlens_mean_trajectory"]
            axes[column].plot(
                layers,
                values,
                color=COLORS[line_index],
                marker=("o", "s", "D", "^")[line_index],
                markevery=5,
                linewidth=1.8,
                label=TRAJECTORY_LABELS[stratum],
            )
            trajectory_data.extend(
                {
                    "placement": placement,
                    "stratum": stratum,
                    "source_layer": int(layer),
                    "mean_refusal_minus_compliance": float(value),
                }
                for layer, value in zip(layers, values, strict=True)
            )
        axes[column].axhline(0, color="#6B7280", linewidth=0.8)
        axes[column].set_title(PLACEMENT_LABELS[placement])
        axes[column].set_xlabel("J-lens source layer")
    axes[0].set_ylabel("Mean refusal − compliance margin")
    axes[0].legend(frameon=False, fontsize=8)
    stem = output_dir / "E24-weaponization-jlens-trajectories"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        output_dir=output_dir,
        stem_name=stem.name,
        hashes=hashes,
        metadata={
            "figure_id": "E24",
            "title": "The J-lens trajectory separates the attack–request combination",
            "question": "Across depth, how do attack-plus-harmful, attack-plus-benign, harmless-plus-harmful, and sham-plus-harmful prompts differ?",
            "description": "All 31 source-layer mean refusal-minus-compliance coordinates for four claim-defining calibration strata, with placements kept separate.",
            "alt_text": "Two line panels show ordering-specific J-lens trajectories. The attack-plus-harmful curve separates from attack-plus-benign and from harmful requests paired with harmless or sham structure across multiple layers, motivating the fitted low-capacity weaponization key.",
            "independent_unit": "calibration request or wrapper family",
            "counts": {"rows_per_stratum_per_placement": 100, "source_layers": 31, "strata": 4},
            "permitted_inference": "descriptive placement-specific internal trajectory separation on calibration",
            "non_claims": [
                "no individual layer selected",
                "no causal circuit",
                "no held-out replication",
                "no behavior-success classification",
            ],
            "derived_data": trajectory_data,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    provenance = output_dir / "provenance.weaponization-calibration.json"
    write_json_atomic(
        provenance,
        {
            "schema_version": "1.0",
            "source_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
            "figures": index,
        },
    )
    return {
        "status": "generated",
        "figure_count": len(index),
        "provenance_sha256": sha256_file(provenance),
    }


def verify_weaponization_figures(result_path: Path, output_dir: Path) -> dict[str, Any]:
    receipt_names = tuple(
        f"E{number}-{name}.receipt.json"
        for number, name in (
            (21, "weaponization-calibration-gate"),
            (22, "weaponization-candidate-comparison"),
            (23, "weaponization-sae-contrasts"),
            (24, "weaponization-jlens-trajectories"),
        )
    )
    comparisons = []
    verified_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_weaponization_figures(result_path, temporary)
        for name in receipt_names:
            expected_path = output_dir / name
            expected = json.loads(expected_path.read_text())
            actual = json.loads((temporary / name).read_text())
            rows = []
            for output_type in ("svg", "png", "pdf"):
                identical = (
                    expected["outputs"][output_type]["sha256"]
                    == actual["outputs"][output_type]["sha256"]
                )
                if not identical:
                    raise ValueError(f"{name}: {output_type} byte verification failed")
                rows.append({"output_type": output_type, "byte_identical": True})
                comparisons.append({"receipt": name, **rows[-1]})
            expected["verification"] = {
                "status": "verified",
                "verified_utc": verified_utc,
                "byte_identity": rows,
            }
            write_json_atomic(expected_path, expected)
        provenance_path = output_dir / "provenance.weaponization-calibration.json"
        provenance = json.loads(provenance_path.read_text())
        for row in provenance["figures"]:
            row["sha256"] = sha256_file(Path(row["path"]))
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
