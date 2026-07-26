from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import EVALUATOR_REVISION
from .followup_patch import analyze_coarse_patch_rows
from .followup_patch_runner import load_frozen_safe_positive_control
from .followup_plan import validate_followup_plan
from .hashing import sha256_file, write_json_atomic
from .models import FollowupPatchReceipt, FollowupTrialReceipt


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _score_by_trial(score_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result = {}
    for path in sorted((score_root / "trials").glob("*.json")):
        row = json.loads(path.read_text())
        trial_id = str(row["trial_id"])
        if trial_id in result:
            raise ValueError("duplicate score trial")
        yes = float(row["yes_probability"])
        no = float(row["no_probability"])
        if (
            row["evaluator_revision"] != EVALUATOR_REVISION
            or not row["parse_ok"]
            or not math.isfinite(yes)
            or not math.isfinite(no)
            or abs(yes + no - 1.0) > 1e-5
        ):
            raise ValueError("invalid patch or baseline HarmBench score")
        result[trial_id] = (path, row)
    return result


def _baseline_scores(
    *,
    generation_root: Path,
    score_root: Path,
    partition: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    scores = _score_by_trial(score_root)
    result = {}
    seen = set()
    for path in sorted((generation_root / "receipts/trials").glob("*.json")):
        receipt = FollowupTrialReceipt.model_validate_json(path.read_text())
        if (
            receipt.partition != partition
            or receipt.arm not in {"full", "structural_sham"}
        ):
            continue
        if receipt.trial_id not in scores:
            raise ValueError("baseline generation lacks score")
        score_path, score = scores[receipt.trial_id]
        if score["generation_receipt_sha256"] != sha256_file(path):
            raise ValueError("baseline score linkage drift")
        key = (str(receipt.placement), receipt.behavior_id, receipt.arm)
        if key in result:
            raise ValueError("duplicate baseline score key")
        result[key] = {
            "score": float(score["yes_probability"]),
            "generation_receipt_sha256": sha256_file(path),
            "score_receipt_sha256": sha256_file(score_path),
        }
        seen.add(receipt.trial_id)
    if len(result) != 80:
        raise ValueError("baseline patch matrix requires 80 full/sham rows")
    return result


def _strip_private_pairs(result: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(result))
    for placement in public["ordering_results"].values():
        for layer in placement.values():
            for condition, summary in layer.items():
                if condition == "eligibility":
                    continue
                summary.pop("behavior_ids", None)
                summary.pop("paired_effects", None)
    return public


def analyze_followup_coarse_patch(
    *,
    public_plan_path: Path,
    patch_root: Path,
    patch_score_root: Path,
    baseline_generation_root: Path,
    baseline_score_root: Path,
    partition: str,
    public_output_path: Path,
    private_output_path: Path,
) -> dict[str, Any]:
    plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(plan)
    if partition not in {"discovery", "calibration"}:
        raise ValueError("patch analysis partition drift")
    load_frozen_safe_positive_control(
        path=patch_root / "safe-positive-control.public.json",
        plan=plan,
    )
    patch_scores = _score_by_trial(patch_score_root)
    baseline = _baseline_scores(
        generation_root=baseline_generation_root,
        score_root=baseline_score_root,
        partition=partition,
    )
    patch_paths = sorted((patch_root / "receipts/trials").glob("*.json"))
    expected_layers = (
        len(
            plan["causal_localization"]["instrument_strength_calibration"][
                "target_candidate_layers"
            ]
        )
        if partition == "discovery"
        else 1
    )
    expected = (
        2
        * expected_layers
        * len(plan["causal_localization"]["execution"]["condition_kinds"])
        * 20
    )
    if len(patch_paths) != expected or len(patch_scores) != expected:
        raise ValueError("patch generation/score count drift")
    rows = []
    receipt_hashes = []
    score_hashes = []
    for path in patch_paths:
        receipt = FollowupPatchReceipt.model_validate_json(path.read_text())
        if receipt.partition != partition:
            raise ValueError("patch receipt partition drift")
        if receipt.trial_id not in patch_scores:
            raise ValueError("patch receipt lacks score")
        score_path, score = patch_scores[receipt.trial_id]
        receipt_sha = sha256_file(path)
        if score["generation_receipt_sha256"] != receipt_sha:
            raise ValueError("patch score linkage drift")
        baseline_key = (
            receipt.placement,
            receipt.behavior_id,
            receipt.baseline_arm,
        )
        if baseline_key not in baseline:
            raise ValueError("patch baseline linkage missing")
        rows.append(
            {
                "partition": partition,
                "placement": receipt.placement,
                "candidate_layer": receipt.candidate_layer,
                "condition": receipt.condition,
                "behavior_id": receipt.behavior_id,
                "patched_score": float(score["yes_probability"]),
                "baseline_score": baseline[baseline_key]["score"],
                "patch_receipt_sha256": receipt_sha,
                "patch_score_receipt_sha256": sha256_file(score_path),
                "baseline_generation_receipt_sha256": baseline[baseline_key][
                    "generation_receipt_sha256"
                ],
                "baseline_score_receipt_sha256": baseline[baseline_key][
                    "score_receipt_sha256"
                ],
            }
        )
        receipt_hashes.append(receipt_sha)
        score_hashes.append(sha256_file(score_path))
    private_result = analyze_coarse_patch_rows(
        rows,
        plan=plan,
        partition=partition,
    )
    private_result.update(
        {
            "study_id": plan["study_id"],
            "plan_sha256": sha256_file(public_plan_path),
            "source_commit": _source_commit(),
            "analysis_implementation_sha256": sha256_file(Path(__file__)),
            "safe_positive_control_sha256": sha256_file(
                patch_root / "safe-positive-control.public.json"
            ),
            "patch_receipt_count": len(patch_paths),
            "patch_score_receipt_count": len(patch_scores),
            "patch_receipt_hashes": receipt_hashes,
            "patch_score_receipt_hashes": score_hashes,
            "rows": rows,
            "raw_prompts_generations_token_ids_or_replay_tensors_public": False,
        }
    )
    private_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(private_output_path, private_result)
    private_output_path.chmod(0o600)
    public = _strip_private_pairs(private_result)
    public.pop("rows", None)
    public.pop("patch_receipt_hashes", None)
    public.pop("patch_score_receipt_hashes", None)
    public["private_result_sha256"] = sha256_file(private_output_path)
    write_json_atomic(public_output_path, public)
    return public
