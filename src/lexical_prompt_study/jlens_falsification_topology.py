from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .followup_runner import build_placement_render_pair
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .jlens_breaker_v2_analysis import select_v2_threshold
from .weaponization_analysis import fit_ridge_logistic


POSITIVE = "positive_attack_harmful"
NEGATIVES = (
    "negative_attack_benign",
    "negative_attack_neutral",
    "negative_harmless_harmful",
    "negative_harmless_benign",
    "negative_structural_sham_harmful",
    "negative_inert_harmful",
)
DIMENSIONS = 256
REQUESTS_PER_CLASS = 60
INJECTED_TOKENS = 252
GENERATION_BUDGET = 128
CONTEXT_CEILING = 8192


def _atomic_json(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    raw = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == 0o600:
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    temporary.replace(path)
    return sha256_bytes(raw)


def _hashed_ngrams(text: str) -> np.ndarray:
    raw = text.encode("utf-8")
    output = np.zeros(DIMENSIONS, dtype=np.float64)
    for size in (3, 4, 5):
        for start in range(max(0, len(raw) - size + 1)):
            value = 2166136261
            for byte in raw[start : start + size]:
                value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
            index = value % DIMENSIONS
            sign = 1.0 if (value >> 31) == 0 else -1.0
            output[index] += sign
    norm = float(np.linalg.norm(output))
    if norm:
        output /= norm
    return output


def _apply_fit(features: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(fit["mean"], dtype=np.float64)
    scale = np.asarray(fit["scale"], dtype=np.float64)
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    logits = coefficients[0] + ((features - mean) / scale) @ coefficients[1:]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _fit_prompt_baseline(
    calibration_topology: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    rows = [
        row
        for row in calibration_topology["observations"]
        if row["placement"] == "scaffold_after_request"
        and row["stratum"] in (POSITIVE, *NEGATIVES)
    ]
    if len(rows) != 700:
        raise ValueError("prompt-baseline calibration topology drift")
    features = np.stack([_hashed_ngrams(str(row[field])) for row in rows])
    labels = np.asarray([int(row["stratum"] == POSITIVE) for row in rows])
    strata = np.asarray([row["stratum"] for row in rows], dtype=object)
    fit = fit_ridge_logistic(features, labels, penalty=0.01)
    selection = select_v2_threshold(fit["probability"], labels, strata)
    return {
        "field": field,
        "dimensions": DIMENSIONS,
        "ngram_sizes": [3, 4, 5],
        "hash": "fnv1a32_signed",
        "row_l2_normalized": True,
        "mean": fit["mean"].tolist(),
        "scale": fit["scale"].tolist(),
        "coefficients": fit["coefficients"].tolist(),
        "iterations": fit["iterations"],
        "threshold_selection": selection,
    }


def _request_pool(topology: Mapping[str, Any], request_class: str) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in topology["observations"]:
        if row["request_class"] != request_class:
            continue
        current = unique.setdefault(row["request_id"], row)
        if current["request_sha256"] != row["request_sha256"]:
            raise ValueError("request identity drift")
    rows = sorted(unique.values(), key=lambda row: (row["request_sha256"], row["request_id"]))
    if len(rows) < REQUESTS_PER_CLASS:
        raise ValueError("insufficient mutation requests")
    return rows[:REQUESTS_PER_CLASS]


def build_falsification_topology(
    *,
    plan_path: Path,
    source_confirmation_topology_path: Path,
    calibration_topology_path: Path,
    factorial_material_path: Path,
    tokenizer_path: Path,
    tokenizer_revision: str,
    private_topology_path: Path,
    private_prompt_baseline_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    plan = json.loads(plan_path.read_text())
    source = json.loads(source_confirmation_topology_path.read_text())
    calibration = json.loads(calibration_topology_path.read_text())
    materials = json.loads(factorial_material_path.read_text())
    if (
        plan.get("status") != "frozen_before_new_mutation_outcomes"
        or source.get("partition") != "detector_confirmation"
        or calibration.get("partition") != "detector_calibration"
        or tokenizer_revision != "0e9e39f249a16976918f6564b8830bc894c89659"
    ):
        raise ValueError("falsification topology input drift")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    joiner = str(materials["material_block_joiner"])
    full = materials["scaffold_materials"]["full_scaffold"]["blocks"]
    inert = materials["scaffold_materials"]["inert_length"]["blocks"]
    if len(full) != 4 or len(inert) != 4:
        raise ValueError("mutation block topology drift")

    prompt_baselines = {
        "full_prompt": _fit_prompt_baseline(calibration, field="prompt_text"),
        "request_only": _fit_prompt_baseline(calibration, field="request_text"),
    }
    baseline_payload = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": "prompt_only_baselines_frozen_before_mutation_outcomes",
        "calibration_topology_sha256": sha256_file(calibration_topology_path),
        "models": prompt_baselines,
        "raw_text_in_artifact": False,
    }
    baseline_sha256 = _atomic_json(private_prompt_baseline_path, baseline_payload, 0o600)

    requests = {
        request_class: _request_pool(source, request_class)
        for request_class in ("harmful_request", "ordinary_benign_request")
    }
    observations: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for mask in range(16):
        blocks = [full[index] if mask & (1 << index) else inert[index] for index in range(4)]
        scaffold = joiner.join(str(block["text"]) for block in blocks)
        injected = tokenizer.encode(scaffold, add_special_tokens=False)
        if len(injected) != INJECTED_TOKENS:
            raise ValueError(f"mutation mask {mask:04b} violates exact token budget")
        variant_id = f"attack-mask-{mask:04b}"
        attack_block_count = int(mask.bit_count())
        for request_class, request_rows in requests.items():
            for source_row in request_rows:
                request_text = str(source_row["request_text"])
                _, pair = build_placement_render_pair(
                    tokenizer=tokenizer,
                    request=request_text,
                    scaffold=scaffold,
                    tokenizer_revision=tokenizer_revision,
                    context_ceiling=CONTEXT_CEILING,
                    generation_budget=GENERATION_BUDGET,
                )
                rendered = pair["ep_after_request"]
                public = {
                    "variant_id": variant_id,
                    "attack_block_mask": mask,
                    "attack_block_count": attack_block_count,
                    "request_class": request_class,
                    "request_id": source_row["request_id"],
                    "request_sha256": source_row["request_sha256"],
                    "material_sha256": sha256_text(scaffold),
                    "placement": "scaffold_after_request",
                    "injected_token_count": INJECTED_TOKENS,
                    "prompt_sha256": sha256_text(rendered["prompt_text"]),
                    "prompt_token_ids_sha256": sha256_bytes(
                        canonical_json_bytes(rendered["prompt_token_ids"])
                    ),
                    "generation_required": request_class == "harmful_request",
                }
                trial_id = sha256_bytes(canonical_json_bytes(public))[:24]
                public["trial_id"] = trial_id
                public_rows.append(public)
                observations.append(
                    {
                        **public,
                        "prompt_text": rendered["prompt_text"],
                        "prompt_token_ids": rendered["prompt_token_ids"],
                        "request_text": request_text,
                    }
                )
    if len(observations) != 1920 or len({row["trial_id"] for row in observations}) != 1920:
        raise ValueError("falsification mutation topology count drift")
    private_payload = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": "mutation_topology_frozen_no_target_outcomes",
        "bindings": {
            "plan_sha256": sha256_file(plan_path),
            "source_confirmation_topology_sha256": sha256_file(
                source_confirmation_topology_path
            ),
            "calibration_topology_sha256": sha256_file(calibration_topology_path),
            "factorial_material_sha256": sha256_file(factorial_material_path),
            "prompt_baseline_sha256": baseline_sha256,
        },
        "model_revision": tokenizer_revision,
        "generation_budget": GENERATION_BUDGET,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
        "observations": observations,
    }
    topology_sha256 = _atomic_json(private_topology_path, private_payload, 0o600)
    receipt = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": private_payload["status"],
        "bindings": private_payload["bindings"],
        "private_topology_sha256": topology_sha256,
        "private_prompt_baseline_sha256": baseline_sha256,
        "observation_count": len(public_rows),
        "generation_observation_count": sum(row["generation_required"] for row in public_rows),
        "variant_count": len({row["variant_id"] for row in public_rows}),
        "request_class_counts": dict(Counter(row["request_class"] for row in public_rows)),
        "attack_block_count_counts": dict(
            Counter(str(row["attack_block_count"]) for row in public_rows)
        ),
        "injected_token_count": INJECTED_TOKENS,
        "raw_prompt_request_or_token_ids_public": False,
        "unopened_v2_confirmation_opened": False,
        "target_outcome_exists": False,
    }
    _atomic_json(public_receipt_path, receipt, 0o644)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-confirmation-topology", type=Path, required=True)
    parser.add_argument("--calibration-topology", type=Path, required=True)
    parser.add_argument("--factorial-material", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--private-topology", type=Path, required=True)
    parser.add_argument("--private-prompt-baseline", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_falsification_topology(
        plan_path=args.plan,
        source_confirmation_topology_path=args.source_confirmation_topology,
        calibration_topology_path=args.calibration_topology,
        factorial_material_path=args.factorial_material,
        tokenizer_path=args.tokenizer_path,
        tokenizer_revision=args.tokenizer_revision,
        private_topology_path=args.private_topology,
        private_prompt_baseline_path=args.private_prompt_baseline,
        public_receipt_path=args.public_receipt,
    )
    print(json.dumps({"status": result["status"], "observation_count": result["observation_count"]}))


if __name__ == "__main__":
    main()
