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
LAYERS = (16, 20, 24, 28, 31)
PLACEMENT_LABELS = {
    "ep_before_request": "Before request",
    "ep_after_request": "After request",
}
GATES = (
    ("identity_and_noop_pass", "Identity + no-op"),
    ("negative_controls_pass", "Negative controls"),
    ("primary_pass", "Restoring patch"),
    ("reciprocal_pass", "Reciprocal patch"),
)


def _save_figure(fig, stem: Path) -> dict[str, str]:
    hashes = _save_all(fig, stem)
    plt.close(fig)
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    hashes["svg"] = sha256_file(svg_path)
    return hashes


def generate_followup_patch_figures(
    result_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> dict:
    _configure()
    result = json.loads(result_path.read_text())
    plan = json.loads(plan_path.read_text())
    if (
        result["status"] != "stopped_no_eligible_layer"
        or result["selected_common_layer"] is not None
        or result["eligible_common_layers"]
        or result["pooled_placement_estimate_reported"] is not False
    ):
        raise ValueError("patch stop result boundary drift")
    analysis = plan["causal_localization"]["execution"]["analysis"]
    if (
        analysis["primary_and_reciprocal_require_absolute_mean_at_least"] != 0.1
        or analysis["primary_requires_upper_95_bound_below_zero"] is not True
        or analysis["reciprocal_requires_lower_95_bound_above_zero"] is not True
    ):
        raise ValueError("patch figure threshold drift")

    rows = []
    for placement in PLACEMENTS:
        for layer in LAYERS:
            payload = result["ordering_results"][placement][str(layer)]
            rows.append(
                {
                    "placement": placement,
                    "layer": layer,
                    "restoring": payload["sham_into_full"],
                    "reciprocal": payload["full_into_sham"],
                    "eligibility": payload["eligibility"],
                }
            )
    if len(rows) != 10:
        raise ValueError("patch figure topology drift")

    fig = plt.figure(figsize=(10.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.0))
    forest = fig.add_subplot(grid[0, 0])
    gate = fig.add_subplot(grid[1, 0])

    labels = [
        f"{PLACEMENT_LABELS[row['placement']]} · L{row['layer']}" for row in rows
    ]
    y = np.arange(len(rows))
    for offset, key, label, marker, color in (
        (-0.13, "restoring", "Sham → full (restoring)", "o", "#D97706"),
        (0.13, "reciprocal", "Full → sham (reciprocal)", "s", "#0284C7"),
    ):
        means = np.asarray([row[key]["mean"] for row in rows])
        lower = np.asarray([row[key]["interval"][0] for row in rows])
        upper = np.asarray([row[key]["interval"][1] for row in rows])
        forest.errorbar(
            means,
            y + offset,
            xerr=np.vstack((means - lower, upper - means)),
            fmt=marker,
            color=color,
            label=label,
            capsize=3,
            linewidth=1.3,
            markersize=5,
        )
    forest.axvline(0, color="#6B7280", linewidth=0.9)
    forest.axvline(-0.1, color="#D97706", linewidth=1.0, linestyle="--")
    forest.axvline(0.1, color="#0284C7", linewidth=1.0, linestyle="--")
    forest.set_yticks(y, labels)
    forest.invert_yaxis()
    forest.set_xlabel("Patched HarmBench probability − matched unpatched baseline")
    forest.set_title(
        "A. Frozen bidirectional causal effects and 95% intervals", pad=30
    )
    forest.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        borderaxespad=0,
        fontsize=8.5,
    )

    matrix = np.asarray(
        [
            [bool(row["eligibility"][key]) for key, _ in GATES]
            for row in rows
        ],
        dtype=int,
    )
    gate.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            passed = bool(matrix[row_index, column_index])
            gate.text(
                column_index,
                row_index,
                "PASS" if passed else "FAIL",
                ha="center",
                va="center",
                color="white" if passed else "#111827",
                fontsize=8,
                fontweight="bold",
            )
    gate.set_xticks(np.arange(len(GATES)), [label for _, label in GATES])
    gate.set_yticks(y, labels)
    gate.set_title("B. Predeclared layer-by-placement eligibility gates")
    gate.tick_params(axis="x", labelrotation=0)

    stem = output_dir / "E12-followup-causal-localization-stop"
    hashes = _save_figure(fig, stem)
    receipt = {
        "figure_id": "E12",
        "title": "Llama 3.1 8B causal localization stopped before calibration",
        "question": "Did any residual-post layer causally transfer behavior in both scaffold placements?",
        "description": (
            "Ordering-specific restoring and reciprocal patch effects with frozen "
            "eligibility gates for all five instrument-valid layers."
        ),
        "alt_text": (
            "The upper forest plot shows restoring and reciprocal patch effects "
            "for layers 16, 20, 24, 28, and 31 separately for scaffold-before and "
            "scaffold-after. No restoring effect reaches the required minus 0.10 "
            "threshold and no reciprocal effect reaches plus 0.10. The lower gate "
            "matrix shows that every layer fails both causal-direction gates; some "
            "layers also fail identity, no-op, or negative-control checks. No common "
            "layer advances to calibration."
        ),
        "permitted_inference": (
            "none of the five independently instrument-valid coarse residual-post "
            "sites passed the frozen bidirectional discovery gate in both orderings"
        ),
        "non_claims": [
            "not evidence that no causal circuit exists",
            "not evidence against finer component or tokenwise localization",
            "not a detector or deployable circuit breaker",
            "no placement pooling or held-out causal confirmation",
        ],
        "row_filter": (
            "all 1,800 discovery patch scores; five candidate layers; nine "
            "conditions; placements shown separately"
        ),
        "uncertainty": "10,000-replicate paired behavior bootstrap",
        "counts": {
            "patch_receipts": result["patch_receipt_count"],
            "score_receipts": result["patch_score_receipt_count"],
            "layers": len(LAYERS),
            "placements": len(PLACEMENTS),
            "behavior_pairs_per_cell": 20,
        },
        "derived_data": {
            "effect_thresholds": {
                "restoring_mean_maximum": -0.1,
                "reciprocal_mean_minimum": 0.1,
                "restoring_upper_95_bound_below_zero": True,
                "reciprocal_lower_95_bound_above_zero": True,
                "minimum_directional_concordance": 0.7,
            },
            "selected_common_layer": result["selected_common_layer"],
            "eligible_common_layers": result["eligible_common_layers"],
            "rows": rows,
        },
        "source_receipts": [
            {"path": str(result_path), "sha256": sha256_file(result_path)},
            {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        ],
        "generator": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "plotting_library": f"matplotlib {matplotlib.__version__}",
            "command": (
                f"lexical-study figures-followup-patch --result {result_path} "
                f"--plan {plan_path} --out {output_dir}"
            ),
        },
        "outputs": {
            name: {"path": str(stem.with_suffix(f".{name}")), "sha256": digest}
            for name, digest in hashes.items()
        },
        "accessibility": {
            "non_color_encodings": (
                "circle versus square markers, dashed thresholds, direct PASS/FAIL "
                "labels, and full row labels"
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
        "source_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "source_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "figures": [{"path": str(receipt_path), "sha256": sha256_file(receipt_path)}],
    }
    provenance_path = output_dir / "provenance.followup-patch.json"
    write_json_atomic(provenance_path, provenance)
    return {"figures": 1, "provenance": str(provenance_path)}


def verify_followup_patch_figures(
    result_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> dict:
    receipt_name = "E12-followup-causal-localization-stop.receipt.json"
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_followup_patch_figures(result_path, plan_path, temporary)
        expected_path = output_dir / receipt_name
        expected = json.loads(expected_path.read_text())
        actual = json.loads((temporary / receipt_name).read_text())
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
            raise ValueError("follow-up patch figure byte verification failed")
        expected["verification"] = {
            "status": "verified",
            "verified_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "byte_identity": comparisons,
        }
        write_json_atomic(expected_path, expected)
        provenance_path = output_dir / "provenance.followup-patch.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["figures"][0]["sha256"] = sha256_file(expected_path)
        write_json_atomic(provenance_path, provenance)
    return {"status": "verified", "comparisons": comparisons}
