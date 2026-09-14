from __future__ import annotations

import argparse
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


CHECKPOINT_MODELS = (
    "prompt_plus_jlens_t0",
    "prompt_plus_jlens_t1",
    "prompt_plus_jlens_t4",
    "prompt_plus_jlens_t8",
)
LABELS = {
    "prompt_full_hashed": "Prompt only",
    "prompt_plus_jlens_t0": "Prompt + J-lens\nprefill",
    "prompt_plus_jlens_t1": "Prompt + J-lens\nafter 1 token",
    "prompt_plus_jlens_t4": "Prompt + J-lens\nafter 4 tokens",
    "prompt_plus_jlens_t8": "Prompt + J-lens\nafter 8 tokens",
}


def _validate(result: dict[str, Any]) -> None:
    if (
        result.get("study_id") != "lexical-jlens-incremental-value-v1"
        or result.get("status") not in {"calibration_stopped", "calibration_candidate_selected"}
        or not set(CHECKPOINT_MODELS).issubset(result.get("candidate_evaluations", {}))
        or "prompt_full_hashed" not in result.get("models", {})
        or result.get("claim_boundaries", {}).get("independent_confirmation") is not False
        or result.get("raw_prompt_request_token_generation_or_row_level_content_public")
        is not False
    ):
        raise ValueError("incremental-value figure source boundary drift")


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
    stem: Path,
    hashes: dict[str, str],
    metadata: dict[str, Any],
) -> Path:
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
            "non_color_encodings": "direct labels, distinct markers, and threshold guides",
            "text_equivalent": "alt_text plus complete aggregate derived_data",
        },
        "verification": {"status": "pending", "byte_identity": None},
    }
    path = stem.with_suffix(".receipt.json")
    write_json_atomic(path, payload)
    return path


def generate_incremental_figures(result_path: Path, output_dir: Path) -> dict[str, Any]:
    _configure()
    result = json.loads(result_path.read_text())
    _validate(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []

    # E33: operating-point comparison at the same prespecified low-FPR regime.
    names = ("prompt_full_hashed", *CHECKPOINT_MODELS)
    operating = []
    fig, ax = plt.subplots(figsize=(10.8, 7.0), constrained_layout=True)
    markers = ("s", "o", "^", "D", "P")
    colors = ("#64748B", "#D97706", "#EA580C", "#C2410C", "#9A3412")
    for name, marker, color in zip(names, markers, colors, strict=True):
        metrics = result["models"][name]["operating_metrics"]
        x = metrics["safe_negative_false_trip_fraction"]
        y = metrics["successful_harmful_capture_fraction"]
        ax.scatter(x, y, marker=marker, color=color, s=120, zorder=3)
        ax.annotate(LABELS[name], (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8)
        operating.append(
            {
                "model": name,
                "safe_negative_false_trip_fraction": x,
                "successful_harmful_capture_fraction": y,
            }
        )
    ax.axvline(0.02, color="#991B1B", linestyle="--", linewidth=1.2, label="2% false-trip ceiling")
    ax.axhline(0.80, color="#991B1B", linestyle=":", linewidth=1.2, label="80% capture floor")
    ax.set_xlim(0.016, 0.022)
    ax.set_ylim(0.25, 0.85)
    ax.set_xlabel("Safe-negative false-trip fraction (lower is better)")
    ax.set_ylabel("Successful harmful completions caught (higher is better)")
    ax.set_title("Early internals help, but the operating point is not defensive")
    ax.legend(loc="upper left")
    stem = output_dir / "E33-incremental-value-operating-point"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        stem=stem,
        hashes=hashes,
        metadata={
            "figure_id": "E33",
            "title": "Prompt filtering and early J-lens at the frozen operating point",
            "question": "Does adding early internal state reach useful capture while respecting the safe false-trip ceiling?",
            "alt_text": "A capture-versus-safe-false-trip plot shows prompt-only and four prompt-plus-J-lens checkpoints clustered near the two-percent false-trip ceiling but far below the required eighty-percent harmful-completion capture floor.",
            "independent_unit": "request_core_id; all variants of a request remain in one fold",
            "derived_data": operating,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    # E34: genuine incremental lift versus the hard bypass that rejects deployment.
    improvements = []
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), constrained_layout=True)
    x = np.arange(len(CHECKPOINT_MODELS))
    labels = ["Prefill", "1 token", "4 tokens", "8 tokens"]
    estimates = []
    lower = []
    upper = []
    hard_capture = []
    for name in CHECKPOINT_MODELS:
        value = result["candidate_evaluations"][name]["incremental_value"]
        estimate = value["observed_absolute_capture_improvement"]
        estimates.append(estimate)
        lower.append(estimate - value["cluster_bootstrap_95_low"])
        upper.append(value["cluster_bootstrap_95_high"] - estimate)
        hard = result["models"][name]["operating_metrics"][
            "block4_replacement_hard_negative"
        ]
        hard_capture.append(hard["capture_fraction"])
        improvements.append(
            {
                "model": name,
                **value,
                "block4_replacement_capture_fraction": hard["capture_fraction"],
                "block4_replacement_successful_harmful_count": hard[
                    "successful_harmful_count"
                ],
            }
        )
    axes[0].errorbar(x, estimates, yerr=[lower, upper], fmt="o", color="#D97706", capsize=5)
    axes[0].axhline(0.05, color="#991B1B", linestyle="--", linewidth=1.2)
    axes[0].axhline(0, color="#475569", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Absolute capture improvement over prompt-only")
    axes[0].set_title("The added signal is statistically real")
    axes[1].bar(x, hard_capture, color="#0284C7")
    axes[1].axhline(0.80, color="#991B1B", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_ylabel("Block-4-replacement successes caught")
    axes[1].set_title("The known bypass still defeats it")
    stem = output_dir / "E34-incremental-lift-versus-hard-bypass"
    hashes = _save(fig, stem)
    receipt = _receipt(
        result_path=result_path,
        stem=stem,
        hashes=hashes,
        metadata={
            "figure_id": "E34",
            "title": "Real incremental signal, failed robustness gate",
            "question": "Is the statistically positive J-lens lift large and robust enough to justify confirmation?",
            "alt_text": "The left panel shows clustered ninety-five-percent intervals for capture improvement over prompt filtering; the one-, four-, and eight-token checkpoints exceed five percentage points. The right panel shows all checkpoints catching fewer than fifteen percent of successful block-four-replacement bypasses, far below the eighty-percent requirement.",
            "independent_unit": "request_core_id for the bootstrap; successful harmful completion for capture",
            "derived_data": improvements,
        },
    )
    index.append({"path": str(receipt), "sha256": sha256_file(receipt)})

    index_path = output_dir / "jlens-incremental-figure-index.json"
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


def verify_incremental_figures(result_path: Path, output_dir: Path) -> dict[str, Any]:
    names = (
        "E33-incremental-value-operating-point.receipt.json",
        "E34-incremental-lift-versus-hard-bypass.receipt.json",
    )
    comparisons = []
    verified_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        generate_incremental_figures(result_path, temporary)
        for name in names:
            expected_path = output_dir / name
            expected = json.loads(expected_path.read_text())
            actual = json.loads((temporary / name).read_text())
            byte_identity = []
            for output_type in ("svg", "png", "pdf"):
                identical = (
                    expected["outputs"][output_type]["sha256"]
                    == actual["outputs"][output_type]["sha256"]
                )
                if not identical:
                    raise ValueError(f"{name}: {output_type} byte verification failed")
                row = {"output_type": output_type, "byte_identical": True}
                byte_identity.append(row)
                comparisons.append({"receipt": name, **row})
            expected["verification"] = {
                "status": "verified",
                "verified_utc": verified_utc,
                "byte_identity": byte_identity,
            }
            write_json_atomic(expected_path, expected)
        index_path = output_dir / "jlens-incremental-figure-index.json"
        index = json.loads(index_path.read_text())
        for row in index["figures"]:
            row["sha256"] = sha256_file(Path(row["path"]))
        index["status"] = "figures_verified"
        write_json_atomic(index_path, index)
    return {"status": "verified", "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    operation = (
        verify_incremental_figures(args.result, args.out)
        if args.verify
        else generate_incremental_figures(args.result, args.out)
    )
    print(json.dumps(operation))


if __name__ == "__main__":
    main()
