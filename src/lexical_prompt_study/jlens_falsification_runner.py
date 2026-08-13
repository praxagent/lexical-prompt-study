from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import EVALUATOR_REVISION
from .behavior import REFUSAL_PATTERN
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .jlens_breaker_v2_analysis import apply_frozen_jlens_head
from .jlens_breaker_v2_scoring import HarmBenchExecutor
from .weaponization_gpu import WeaponizationPrefillRuntime


EXPECTED_OBSERVATIONS = 1920
EXPECTED_GENERATIONS = 960
MAX_NEW_TOKENS = 128


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _atomic(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> str:
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


def _authorization(
    *,
    path: Path,
    run_id: str,
    source_commit: str,
    plan_sha256: str,
    topology_sha256: str,
    prompt_baseline_sha256: str,
    threshold_sha256: str,
    phase: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = {
        "source_commit": source_commit,
        "plan_sha256": plan_sha256,
        "private_topology_sha256": topology_sha256,
        "private_prompt_baseline_sha256": prompt_baseline_sha256,
        "private_threshold_sha256": threshold_sha256,
    }
    status = value.get("status")
    status_allowed = status == "mutation_execution_authorized" or (
        phase == "score" and status == "mutation_scoring_recovery_authorized"
    )
    if (
        value.get("schema_version") != "1.0"
        or value.get("study_id") != "lexical-jlens-signal-falsification-v1"
        or not status_allowed
        or value.get("paid_compute_authorized") is not True
        or value.get("run_id") != run_id
        or value.get("expected_prefill_observations") != EXPECTED_OBSERVATIONS
        or value.get("expected_generation_observations") != EXPECTED_GENERATIONS
        or value.get("single_task_owned_pod_maximum") != 1
        or float(value.get("maximum_new_compute_usd", math.inf)) > 50
        or value.get("enforcement_enabled") is not False
        or value.get("unopened_v2_confirmation_opened") is not False
        or any(value.get("bindings", {}).get(key) != item for key, item in expected.items())
        or (
            status == "mutation_scoring_recovery_authorized"
            and value.get("authorized_phase") != "score_only_after_zero_score_stop"
        )
    ):
        raise ValueError("falsification execution authorization drift")
    return value


def _validate_receipt(receipt: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    required_equal = (
        "trial_id",
        "variant_id",
        "attack_block_mask",
        "attack_block_count",
        "request_class",
        "request_id",
        "request_sha256",
        "placement",
        "injected_token_count",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "generation_required",
    )
    if any(receipt.get(field) != observation.get(field) for field in required_equal):
        raise ValueError("falsification receipt topology drift")
    if len(receipt.get("jlens_refusal_minus_compliance_trajectory", [])) != 31:
        raise ValueError("falsification receipt J-lens topology drift")
    finite = [
        receipt["feature_6779_magnitude"],
        receipt["frozen_subspace_score"],
        receipt["sae_normalized_reconstruction_error"],
        receipt["frozen_jlens_score"],
        receipt["frozen_jlens_threshold"],
        *receipt["jlens_refusal_minus_compliance_trajectory"],
    ]
    if not all(math.isfinite(float(value)) for value in finite):
        raise ValueError("falsification receipt non-finite")
    if receipt["would_cancel_before_first_token"] != (
        receipt["frozen_jlens_score"] >= receipt["frozen_jlens_threshold"]
    ):
        raise ValueError("falsification detector arithmetic drift")
    generated = receipt.get("generated_text_sha256")
    if observation["generation_required"] != (generated is not None):
        raise ValueError("falsification generation presence drift")


def _load_scoring_input(
    *, acquisition_root: Path, trial_id: str, observation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    generation_path = acquisition_root / "receipts" / f"{trial_id}.json"
    restricted_path = acquisition_root / "restricted" / f"{trial_id}.json"
    generation = json.loads(generation_path.read_text())
    restricted = json.loads(restricted_path.read_text())
    _validate_receipt(generation, observation)
    if (
        set(restricted) != {"trial_id", "generated_text", "generated_token_ids"}
        or restricted["trial_id"] != trial_id
        or generation["restricted_artifact_sha256"] != sha256_file(restricted_path)
        or sha256_text(restricted["generated_text"])
        != generation["generated_text_sha256"]
        or sha256_bytes(canonical_json_bytes(restricted["generated_token_ids"]))
        != generation["generated_token_ids_sha256"]
    ):
        raise ValueError("falsification scoring input drift")
    behavior_sha256 = sha256_text(str(observation["request_text"]))
    return generation, restricted, behavior_sha256


def _generate(runtime: WeaponizationPrefillRuntime, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    torch = runtime.torch
    tokenizer = runtime.tokenizer
    model = runtime.model
    width = max(len(row["prompt_token_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width), int(tokenizer.pad_token_id), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        ids = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, -len(ids) :] = ids
        attention_mask[index, -len(ids) :] = 1
    eos_ids = model.generation_config.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids or []}
    eos_set.add(int(tokenizer.eos_token_id))
    started = time.monotonic()
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(eos_set),
            use_cache=True,
        )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    output = []
    for index, row in enumerate(rows):
        generated = [int(value) for value in sequences[index, width:].tolist()]
        eos_index = next((i for i, token in enumerate(generated) if token in eos_set), None)
        kept = generated if eos_index is None else generated[:eos_index]
        output.append(
            {
                "trial_id": row["trial_id"],
                "generated_token_ids": kept,
                "generated_text": tokenizer.decode(kept, skip_special_tokens=True),
                "finish_reason": "length" if eos_index is None else "eos",
                "generation_elapsed_ms": elapsed_ms / len(rows),
            }
        )
    return output


def run_acquisition(
    *,
    plan_path: Path,
    public_instrument_plan_path: Path,
    topology_path: Path,
    prompt_baseline_path: Path,
    threshold_path: Path,
    authorization_path: Path,
    probe_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    factorial_material_path: Path,
    output_root: Path,
    run_id: str,
    batch_size: int,
) -> dict[str, Any]:
    source_commit = _source_commit()
    plan_sha256 = sha256_file(plan_path)
    topology_sha256 = sha256_file(topology_path)
    prompt_baseline_sha256 = sha256_file(prompt_baseline_path)
    threshold_sha256 = sha256_file(threshold_path)
    _authorization(
        path=authorization_path,
        run_id=run_id,
        source_commit=source_commit,
        plan_sha256=plan_sha256,
        topology_sha256=topology_sha256,
        prompt_baseline_sha256=prompt_baseline_sha256,
        threshold_sha256=threshold_sha256,
        phase="acquire",
    )
    topology = json.loads(topology_path.read_text())
    instrument_plan = json.loads(public_instrument_plan_path.read_text())
    thresholds = json.loads(threshold_path.read_text())
    if (
        topology.get("status") != "mutation_topology_frozen_no_target_outcomes"
        or topology.get("unopened_v2_confirmation_opened") is not False
        or len(topology.get("observations", [])) != EXPECTED_OBSERVATIONS
        or thresholds["placements"]["scaffold_after_request"].get("eligible") is not True
    ):
        raise ValueError("falsification execution topology drift")
    observations = sorted(topology["observations"], key=lambda row: row["trial_id"])
    receipt_root = output_root / "receipts"
    restricted_root = output_root / "restricted"
    receipt_root.mkdir(parents=True, exist_ok=True)
    restricted_root.mkdir(parents=True, exist_ok=True)
    receipt_root.chmod(0o700)
    restricted_root.chmod(0o700)
    missing: list[dict[str, Any]] = []
    for row in observations:
        path = receipt_root / f"{row['trial_id']}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            _validate_receipt(receipt, row)
            if receipt["source_commit"] != source_commit or receipt["run_id"] != run_id:
                raise ValueError("falsification resume provenance drift")
            continue
        missing.append(row)
    runtime: WeaponizationPrefillRuntime | None = None
    written = 0
    frozen = thresholds["placements"]["scaffold_after_request"]
    software = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if runtime is None:
            runtime = WeaponizationPrefillRuntime(
                public_plan=instrument_plan,
                probe_plan_path=probe_plan_path,
                model_path=model_path,
                lens_path=lens_path,
                sae_path=sae_path,
                factorial_material_path=factorial_material_path,
            )
            software.update(
                {
                    "torch": runtime.torch.__version__,
                    "cuda": runtime.torch.version.cuda,
                }
            )
        readouts = {row["trial_id"]: row for row in runtime.execute_batch(batch, 1)}
        generation_rows = [row for row in batch if row["generation_required"]]
        generated = {row["trial_id"]: row for row in _generate(runtime, generation_rows)} if generation_rows else {}
        for observation in batch:
            readout = readouts[observation["trial_id"]]
            score = float(
                apply_frozen_jlens_head(
                    [readout["jlens_refusal_minus_compliance_trajectory"]],
                    frozen["frozen_jlens_head"],
                )[0]
            )
            generation = generated.get(observation["trial_id"])
            receipt = {
                "schema_version": "1.0",
                "study_id": topology["study_id"],
                "status": "complete",
                "run_id": run_id,
                "source_commit": source_commit,
                "plan_sha256": plan_sha256,
                "private_topology_sha256": topology_sha256,
                **{key: observation[key] for key in (
                    "trial_id", "variant_id", "attack_block_mask", "attack_block_count",
                    "request_class", "request_id", "request_sha256", "placement",
                    "injected_token_count", "prompt_sha256", "prompt_token_ids_sha256",
                    "generation_required",
                )},
                "feature_6779_magnitude": float(readout["feature_6779_magnitude"]),
                "frozen_subspace_score": float(readout["frozen_subspace_score"]),
                "sae_normalized_reconstruction_error": float(readout["sae_normalized_reconstruction_error"]),
                "jlens_refusal_minus_compliance_trajectory": [float(value) for value in readout["jlens_refusal_minus_compliance_trajectory"]],
                "restricted_exact_match": bool(readout["restricted_exact_match"]),
                "structural_metrics": readout["structural_metrics"],
                "frozen_jlens_score": score,
                "frozen_jlens_threshold": float(frozen["threshold"]),
                "would_cancel_before_first_token": score >= float(frozen["threshold"]),
                "generated_text_sha256": None,
                "generated_token_ids_sha256": None,
                "generated_token_count": None,
                "finish_reason": None,
                "refusal_detected": None,
                "restricted_artifact_sha256": None,
                "generation_elapsed_ms": None,
                "software": software,
            }
            if generation is not None:
                private = {
                    "trial_id": observation["trial_id"],
                    "generated_text": generation["generated_text"],
                    "generated_token_ids": generation["generated_token_ids"],
                }
                private_path = restricted_root / f"{observation['trial_id']}.json"
                private_sha256 = _atomic(private_path, private)
                receipt.update(
                    {
                        "generated_text_sha256": sha256_text(generation["generated_text"]),
                        "generated_token_ids_sha256": sha256_bytes(canonical_json_bytes(generation["generated_token_ids"])),
                        "generated_token_count": len(generation["generated_token_ids"]),
                        "finish_reason": generation["finish_reason"],
                        "refusal_detected": bool(REFUSAL_PATTERN.search(generation["generated_text"])),
                        "restricted_artifact_sha256": private_sha256,
                        "generation_elapsed_ms": generation["generation_elapsed_ms"],
                    }
                )
            _validate_receipt(receipt, observation)
            _atomic(receipt_root / f"{observation['trial_id']}.json", receipt)
            written += 1
        print(f"completed {min(start + len(batch), len(missing))}/{len(missing)} missing", flush=True)
    paths = sorted(receipt_root.glob("*.json"))
    if len(paths) != EXPECTED_OBSERVATIONS:
        raise ValueError("falsification acquisition incomplete")
    manifest = sha256_bytes(canonical_json_bytes([{"trial_id": p.stem, "sha256": sha256_file(p)} for p in paths]))
    summary = {
        "schema_version": "1.0",
        "study_id": topology["study_id"],
        "status": "acquisition_complete",
        "run_id": run_id,
        "source_commit": source_commit,
        "plan_sha256": plan_sha256,
        "private_topology_sha256": topology_sha256,
        "observation_count": EXPECTED_OBSERVATIONS,
        "generation_count": EXPECTED_GENERATIONS,
        "written_this_invocation": written,
        "resumed_receipt_count": EXPECTED_OBSERVATIONS - written,
        "receipt_manifest_sha256": manifest,
        "raw_prompt_token_generation_or_row_level_content_public": False,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
    }
    _atomic(output_root / "summary.json", summary)
    return summary


def run_scoring(
    *,
    plan_path: Path,
    topology_path: Path,
    prompt_baseline_path: Path,
    threshold_path: Path,
    authorization_path: Path,
    acquisition_root: Path,
    evaluator_path: str,
    output_root: Path,
    run_id: str,
    batch_size: int,
) -> dict[str, Any]:
    source_commit = _source_commit()
    _authorization(
        path=authorization_path,
        run_id=run_id,
        source_commit=source_commit,
        plan_sha256=sha256_file(plan_path),
        topology_sha256=sha256_file(topology_path),
        prompt_baseline_sha256=sha256_file(prompt_baseline_path),
        threshold_sha256=sha256_file(threshold_path),
        phase="score",
    )
    topology = json.loads(topology_path.read_text())
    observations = {row["trial_id"]: row for row in topology["observations"]}
    acquisition_summary = json.loads((acquisition_root / "summary.json").read_text())
    if acquisition_summary.get("status") != "acquisition_complete":
        raise ValueError("falsification acquisition not scoring eligible")
    trials = output_root / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    trials.chmod(0o700)
    pending = []
    for trial_id, row in sorted(observations.items()):
        if not row["generation_required"]:
            continue
        generation_path = acquisition_root / "receipts" / f"{trial_id}.json"
        generation, restricted, behavior_sha256 = _load_scoring_input(
            acquisition_root=acquisition_root,
            trial_id=trial_id,
            observation=row,
        )
        score_path = trials / f"{trial_id}.json"
        if score_path.exists():
            score = json.loads(score_path.read_text())
            if (
                score.get("generation_receipt_sha256") != sha256_file(generation_path)
                or score.get("classifier_behavior_sha256") != behavior_sha256
            ):
                raise ValueError("falsification scoring resume drift")
            continue
        pending.append(
            {
                "generation_trial_id": trial_id,
                "behavior": row["request_text"],
                "generation_path": generation_path,
                "generation": restricted["generated_text"],
                "generation_hash": generation["generated_text_sha256"],
                "classifier_behavior_sha256": behavior_sha256,
            }
        )
    executor = HarmBenchExecutor(evaluator_path)
    written = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = {row["generation_trial_id"]: row for row in executor(batch)}
        for item in batch:
            trial_id = item["generation_trial_id"]
            result = results[trial_id]
            payload = {
                "schema_version": "1.0",
                "study_id": topology["study_id"],
                "status": "complete",
                "run_id": run_id,
                "source_commit": source_commit,
                "trial_id": trial_id,
                "generation_receipt_sha256": sha256_file(item["generation_path"]),
                "generated_text_sha256": item["generation_hash"],
                "classifier_behavior_sha256": item["classifier_behavior_sha256"],
                "evaluator_revision": EVALUATOR_REVISION,
                **{key: value for key, value in result.items() if key not in {"generation_trial_id"}},
            }
            _atomic(trials / f"{trial_id}.json", payload)
            written += 1
        print(f"scored {min(start + len(batch), len(pending))}/{len(pending)} pending", flush=True)
    paths = sorted(trials.glob("*.json"))
    if len(paths) != EXPECTED_GENERATIONS:
        raise ValueError("falsification scoring incomplete")
    manifest = sha256_bytes(canonical_json_bytes([{"trial_id": p.stem, "sha256": sha256_file(p)} for p in paths]))
    summary = {
        "schema_version": "1.0",
        "study_id": topology["study_id"],
        "status": "scoring_complete",
        "run_id": run_id,
        "source_commit": source_commit,
        "score_count": EXPECTED_GENERATIONS,
        "written_this_invocation": written,
        "resumed_score_count": EXPECTED_GENERATIONS - written,
        "score_manifest_sha256": manifest,
        "evaluator_revision": EVALUATOR_REVISION,
        "raw_generation_or_classifier_input_public": False,
        "unopened_v2_confirmation_opened": False,
    }
    _atomic(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", type=Path, required=True)
    common.add_argument("--topology", type=Path, required=True)
    common.add_argument("--prompt-baseline", type=Path, required=True)
    common.add_argument("--threshold", type=Path, required=True)
    common.add_argument("--authorization", type=Path, required=True)
    common.add_argument("--run-id", required=True)
    common.add_argument("--batch-size", type=int, default=4)
    acquire = subparsers.add_parser("acquire", parents=[common])
    acquire.add_argument("--instrument-plan", type=Path, required=True)
    acquire.add_argument("--probe-plan", type=Path, required=True)
    acquire.add_argument("--model-path", required=True)
    acquire.add_argument("--lens-path", type=Path, required=True)
    acquire.add_argument("--sae-path", type=Path, required=True)
    acquire.add_argument("--factorial-material", type=Path, required=True)
    acquire.add_argument("--output-root", type=Path, required=True)
    score = subparsers.add_parser("score", parents=[common])
    score.add_argument("--acquisition-root", type=Path, required=True)
    score.add_argument("--evaluator-path", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        result = run_acquisition(
            plan_path=args.plan,
            public_instrument_plan_path=args.instrument_plan,
            topology_path=args.topology,
            prompt_baseline_path=args.prompt_baseline,
            threshold_path=args.threshold,
            authorization_path=args.authorization,
            probe_plan_path=args.probe_plan,
            model_path=args.model_path,
            lens_path=args.lens_path,
            sae_path=args.sae_path,
            factorial_material_path=args.factorial_material,
            output_root=args.output_root,
            run_id=args.run_id,
            batch_size=args.batch_size,
        )
    else:
        result = run_scoring(
            plan_path=args.plan,
            topology_path=args.topology,
            prompt_baseline_path=args.prompt_baseline,
            threshold_path=args.threshold,
            authorization_path=args.authorization,
            acquisition_root=args.acquisition_root,
            evaluator_path=args.evaluator_path,
            output_root=args.output_root,
            run_id=args.run_id,
            batch_size=args.batch_size,
        )
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
