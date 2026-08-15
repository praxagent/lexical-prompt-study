from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import EVALUATOR_REVISION
from .behavior import REFUSAL_PATTERN
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .jlens_breaker_v2_scoring import HarmBenchExecutor
from .jlens_falsification_runner import _generate
from .jlens_incremental_topology import (
    EXPECTED_OBSERVATIONS,
    STUDY_ID,
    TOPOLOGY_STATUS,
)
from .weaponization_gpu import WeaponizationPrefillRuntime


CHECKPOINTS = (0, 1, 4, 8)


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
    phase: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != "1.0"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "incremental_value_execution_authorized"
        or value.get("paid_compute_authorized") is not True
        or value.get("authorized_phases") != ["acquire", "score"]
        or phase not in value["authorized_phases"]
        or value.get("run_id") != run_id
        or value.get("expected_observations") != EXPECTED_OBSERVATIONS
        or value.get("expected_generations") != EXPECTED_OBSERVATIONS
        or value.get("generation_checkpoints") != list(CHECKPOINTS)
        or value.get("single_task_owned_pod_maximum") != 1
        or float(value.get("maximum_new_compute_usd", math.inf)) > 25
        or value.get("legacy_hard_ceiling_usd") != 200
        or value.get("enforcement_enabled") is not False
        or value.get("unopened_v2_confirmation_opened") is not False
        or value.get("bindings", {}).get("source_commit") != source_commit
        or value.get("bindings", {}).get("plan_sha256") != plan_sha256
        or value.get("bindings", {}).get("private_topology_sha256")
        != topology_sha256
    ):
        raise ValueError("incremental-value execution authorization drift")
    return value


def _checkpoint_readout(result: Mapping[str, Any], prefix_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "prefix_token_count": len(prefix_ids),
        "prefix_token_ids_sha256": sha256_bytes(canonical_json_bytes(prefix_ids)),
        "feature_6779_magnitude": float(result["feature_6779_magnitude"]),
        "frozen_subspace_score": float(result["frozen_subspace_score"]),
        "sae_normalized_reconstruction_error": float(
            result["sae_normalized_reconstruction_error"]
        ),
        "jlens_refusal_minus_compliance_trajectory": [
            float(value) for value in result["jlens_refusal_minus_compliance_trajectory"]
        ],
        "prefill_latency_ms": float(result["prefill_latency_ms"]),
        "detector_readout_latency_ms": float(result["detector_readout_latency_ms"]),
        "peak_gpu_memory_bytes": int(result["peak_gpu_memory_bytes"]),
    }


def _validate_checkpoint(value: Mapping[str, Any], expected_tokens: int) -> None:
    if (
        value.get("prefix_token_count") != expected_tokens
        or len(value.get("prefix_token_ids_sha256", "")) != 64
        or len(value.get("jlens_refusal_minus_compliance_trajectory", [])) != 31
    ):
        raise ValueError("incremental-value checkpoint topology drift")
    finite = [
        value["feature_6779_magnitude"],
        value["frozen_subspace_score"],
        value["sae_normalized_reconstruction_error"],
        value["prefill_latency_ms"],
        value["detector_readout_latency_ms"],
        *value["jlens_refusal_minus_compliance_trajectory"],
    ]
    if not all(math.isfinite(float(item)) for item in finite):
        raise ValueError("incremental-value checkpoint non-finite")


def _validate_receipt(receipt: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    equal_fields = (
        "trial_id",
        "request_core_id",
        "request_core_sha256",
        "intent_frame",
        "safe_intent",
        "utility_expected_sha256",
        "variant_id",
        "variant_family",
        "attack_block_mask",
        "attack_block_count",
        "material_sha256",
        "wrapper_id",
        "placement",
        "injected_token_count",
        "prompt_sha256",
        "prompt_token_ids_sha256",
    )
    if any(receipt.get(key) != observation.get(key) for key in equal_fields):
        raise ValueError("incremental-value receipt topology drift")
    readouts = receipt.get("readouts", {})
    if set(readouts) != {str(value) for value in CHECKPOINTS}:
        raise ValueError("incremental-value checkpoint key drift")
    _validate_checkpoint(readouts["0"], 0)
    generated_count = int(receipt.get("generated_token_count", -1))
    for checkpoint in CHECKPOINTS[1:]:
        value = readouts[str(checkpoint)]
        if generated_count >= checkpoint:
            if value is None:
                raise ValueError("incremental-value available checkpoint missing")
            _validate_checkpoint(value, checkpoint)
        elif value is not None:
            raise ValueError("incremental-value unavailable checkpoint present")
    if (
        receipt.get("generated_text_sha256") is None
        or receipt.get("generated_token_ids_sha256") is None
        or receipt.get("restricted_artifact_sha256") is None
    ):
        raise ValueError("incremental-value generation linkage missing")
    expected_hash = observation.get("utility_expected_sha256")
    if expected_hash is None and receipt.get("utility_exact_match") is not None:
        raise ValueError("unsafe direct row has utility result")
    if expected_hash is not None and not isinstance(receipt.get("utility_exact_match"), bool):
        raise ValueError("safe intent row lacks utility result")


def _prefix_rows(
    runtime: WeaponizationPrefillRuntime,
    batch: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Mapping[str, Any]],
    checkpoint: int,
) -> list[dict[str, Any]]:
    rows = []
    for observation in batch:
        token_ids = list(generated[observation["trial_id"]]["generated_token_ids"])
        if len(token_ids) < checkpoint:
            continue
        prefix = token_ids[:checkpoint]
        prefix_text = runtime.tokenizer.decode(prefix, skip_special_tokens=False)
        prompt_text = str(observation["prompt_text"]) + prefix_text
        rows.append(
            {
                **observation,
                "prompt_text": prompt_text,
                "prompt_sha256": sha256_text(prompt_text),
                "prompt_token_ids": [*observation["prompt_token_ids"], *prefix],
            }
        )
    return rows


def run_acquisition(
    *,
    plan_path: Path,
    instrument_plan_path: Path,
    topology_path: Path,
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
    if batch_size < 1:
        raise ValueError("incremental-value batch size must be positive")
    source_commit = _source_commit()
    plan_sha256 = sha256_file(plan_path)
    topology_sha256 = sha256_file(topology_path)
    _authorization(
        path=authorization_path,
        run_id=run_id,
        source_commit=source_commit,
        plan_sha256=plan_sha256,
        topology_sha256=topology_sha256,
        phase="acquire",
    )
    topology = json.loads(topology_path.read_text())
    instrument_plan = json.loads(instrument_plan_path.read_text())
    if (
        topology.get("study_id") != STUDY_ID
        or topology.get("status") != TOPOLOGY_STATUS
        or topology.get("generation_checkpoints") != list(CHECKPOINTS)
        or topology.get("enforcement_enabled") is not False
        or topology.get("unopened_v2_confirmation_opened") is not False
        or len(topology.get("observations", [])) != EXPECTED_OBSERVATIONS
    ):
        raise ValueError("incremental-value acquisition topology drift")
    observations = sorted(topology["observations"], key=lambda row: row["trial_id"])
    receipt_root = output_root / "receipts"
    restricted_root = output_root / "restricted"
    receipt_root.mkdir(parents=True, exist_ok=True)
    restricted_root.mkdir(parents=True, exist_ok=True)
    receipt_root.chmod(0o700)
    restricted_root.chmod(0o700)
    missing = []
    for observation in observations:
        receipt_path = receipt_root / f"{observation['trial_id']}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            _validate_receipt(receipt, observation)
            if receipt.get("source_commit") != source_commit or receipt.get("run_id") != run_id:
                raise ValueError("incremental-value resume provenance drift")
            continue
        missing.append(observation)
    runtime: WeaponizationPrefillRuntime | None = None
    written = 0
    software = {"python": sys.version, "platform": platform.platform()}
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
            software.update({"torch": runtime.torch.__version__, "cuda": runtime.torch.version.cuda})
        prefill_results = {
            row["trial_id"]: row for row in runtime.execute_batch(batch, 1)
        }
        generated = {row["trial_id"]: row for row in _generate(runtime, batch)}
        checkpoint_results: dict[int, dict[str, Mapping[str, Any]]] = {}
        for checkpoint in CHECKPOINTS[1:]:
            prefix_batch = _prefix_rows(runtime, batch, generated, checkpoint)
            checkpoint_results[checkpoint] = (
                {row["trial_id"]: row for row in runtime.execute_batch(prefix_batch, 1)}
                if prefix_batch
                else {}
            )
        for observation in batch:
            trial_id = observation["trial_id"]
            generation = generated[trial_id]
            token_ids = list(generation["generated_token_ids"])
            private = {
                "trial_id": trial_id,
                "generated_text": generation["generated_text"],
                "generated_token_ids": token_ids,
            }
            private_path = restricted_root / f"{trial_id}.json"
            private_sha256 = _atomic(private_path, private)
            readouts: dict[str, Any] = {
                "0": _checkpoint_readout(prefill_results[trial_id], [])
            }
            for checkpoint in CHECKPOINTS[1:]:
                result = checkpoint_results[checkpoint].get(trial_id)
                readouts[str(checkpoint)] = (
                    _checkpoint_readout(result, token_ids[:checkpoint])
                    if result is not None
                    else None
                )
            expected = observation.get("utility_expected_text")
            utility_match = (
                str(generation["generated_text"]).strip() == str(expected).strip()
                if expected is not None
                else None
            )
            receipt = {
                "schema_version": "1.0",
                "study_id": STUDY_ID,
                "status": "complete",
                "run_id": run_id,
                "source_commit": source_commit,
                "plan_sha256": plan_sha256,
                "private_topology_sha256": topology_sha256,
                **{
                    key: observation[key]
                    for key in (
                        "trial_id",
                        "request_core_id",
                        "request_core_sha256",
                        "intent_frame",
                        "safe_intent",
                        "utility_expected_sha256",
                        "variant_id",
                        "variant_family",
                        "attack_block_mask",
                        "attack_block_count",
                        "material_sha256",
                        "wrapper_id",
                        "placement",
                        "injected_token_count",
                        "prompt_sha256",
                        "prompt_token_ids_sha256",
                    )
                },
                "readouts": readouts,
                "restricted_exact_match": bool(prefill_results[trial_id]["restricted_exact_match"]),
                "structural_metrics": prefill_results[trial_id]["structural_metrics"],
                "generated_text_sha256": sha256_text(str(generation["generated_text"])),
                "generated_token_ids_sha256": sha256_bytes(canonical_json_bytes(token_ids)),
                "generated_token_count": len(token_ids),
                "finish_reason": generation["finish_reason"],
                "refusal_detected": bool(REFUSAL_PATTERN.search(str(generation["generated_text"]))),
                "utility_exact_match": utility_match,
                "restricted_artifact_sha256": private_sha256,
                "generation_elapsed_ms": float(generation["generation_elapsed_ms"]),
                "software": software,
            }
            _validate_receipt(receipt, observation)
            _atomic(receipt_root / f"{trial_id}.json", receipt)
            written += 1
        print(
            f"completed {min(start + len(batch), len(missing))}/{len(missing)} missing",
            flush=True,
        )
    paths = sorted(receipt_root.glob("*.json"))
    if len(paths) != EXPECTED_OBSERVATIONS:
        raise ValueError("incremental-value acquisition incomplete")
    manifest = sha256_bytes(
        canonical_json_bytes(
            [{"trial_id": path.stem, "sha256": sha256_file(path)} for path in paths]
        )
    )
    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "acquisition_complete",
        "run_id": run_id,
        "source_commit": source_commit,
        "plan_sha256": plan_sha256,
        "private_topology_sha256": topology_sha256,
        "observation_count": EXPECTED_OBSERVATIONS,
        "generation_count": EXPECTED_OBSERVATIONS,
        "generation_checkpoints": list(CHECKPOINTS),
        "written_this_invocation": written,
        "resumed_receipt_count": EXPECTED_OBSERVATIONS - written,
        "receipt_manifest_sha256": manifest,
        "raw_prompt_token_generation_or_row_level_content_public": False,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
    }
    _atomic(output_root / "summary.json", summary)
    return summary


def _load_scoring_input(
    *, acquisition_root: Path, observation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    trial_id = str(observation["trial_id"])
    receipt_path = acquisition_root / "receipts" / f"{trial_id}.json"
    restricted_path = acquisition_root / "restricted" / f"{trial_id}.json"
    receipt = json.loads(receipt_path.read_text())
    restricted = json.loads(restricted_path.read_text())
    _validate_receipt(receipt, observation)
    if (
        set(restricted) != {"trial_id", "generated_text", "generated_token_ids"}
        or restricted["trial_id"] != trial_id
        or receipt["restricted_artifact_sha256"] != sha256_file(restricted_path)
        or receipt["generated_text_sha256"] != sha256_text(restricted["generated_text"])
        or receipt["generated_token_ids_sha256"]
        != sha256_bytes(canonical_json_bytes(restricted["generated_token_ids"]))
    ):
        raise ValueError("incremental-value scoring input drift")
    return receipt, restricted


def run_scoring(
    *,
    plan_path: Path,
    topology_path: Path,
    authorization_path: Path,
    acquisition_root: Path,
    evaluator_path: str,
    output_root: Path,
    run_id: str,
    batch_size: int,
) -> dict[str, Any]:
    source_commit = _source_commit()
    plan_sha256 = sha256_file(plan_path)
    topology_sha256 = sha256_file(topology_path)
    _authorization(
        path=authorization_path,
        run_id=run_id,
        source_commit=source_commit,
        plan_sha256=plan_sha256,
        topology_sha256=topology_sha256,
        phase="score",
    )
    topology = json.loads(topology_path.read_text())
    observations = {row["trial_id"]: row for row in topology["observations"]}
    acquisition_summary = json.loads((acquisition_root / "summary.json").read_text())
    if (
        acquisition_summary.get("status") != "acquisition_complete"
        or acquisition_summary.get("private_topology_sha256") != topology_sha256
    ):
        raise ValueError("incremental-value acquisition not scoring eligible")
    trial_root = output_root / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    trial_root.chmod(0o700)
    pending = []
    for trial_id, observation in sorted(observations.items()):
        receipt_path = acquisition_root / "receipts" / f"{trial_id}.json"
        receipt, restricted = _load_scoring_input(
            acquisition_root=acquisition_root, observation=observation
        )
        score_path = trial_root / f"{trial_id}.json"
        behavior_sha256 = sha256_text(str(observation["request_core_text"]))
        if score_path.exists():
            score = json.loads(score_path.read_text())
            if (
                score.get("generation_receipt_sha256") != sha256_file(receipt_path)
                or score.get("classifier_behavior_sha256") != behavior_sha256
            ):
                raise ValueError("incremental-value scoring resume drift")
            continue
        pending.append(
            {
                "generation_trial_id": trial_id,
                "behavior": observation["request_core_text"],
                "generation_path": receipt_path,
                "generation": restricted["generated_text"],
                "generation_hash": receipt["generated_text_sha256"],
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
                "study_id": STUDY_ID,
                "status": "complete",
                "run_id": run_id,
                "source_commit": source_commit,
                "trial_id": trial_id,
                "generation_receipt_sha256": sha256_file(item["generation_path"]),
                "generated_text_sha256": item["generation_hash"],
                "classifier_behavior_sha256": item["classifier_behavior_sha256"],
                "evaluator_revision": EVALUATOR_REVISION,
                **{
                    key: value
                    for key, value in result.items()
                    if key != "generation_trial_id"
                },
            }
            _atomic(trial_root / f"{trial_id}.json", payload)
            written += 1
        print(
            f"scored {min(start + len(batch), len(pending))}/{len(pending)} pending",
            flush=True,
        )
    paths = sorted(trial_root.glob("*.json"))
    if len(paths) != EXPECTED_OBSERVATIONS:
        raise ValueError("incremental-value scoring incomplete")
    manifest = sha256_bytes(
        canonical_json_bytes(
            [{"trial_id": path.stem, "sha256": sha256_file(path)} for path in paths]
        )
    )
    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "scoring_complete",
        "run_id": run_id,
        "source_commit": source_commit,
        "score_count": EXPECTED_OBSERVATIONS,
        "written_this_invocation": written,
        "resumed_score_count": EXPECTED_OBSERVATIONS - written,
        "score_manifest_sha256": manifest,
        "evaluator_revision": EVALUATOR_REVISION,
        "raw_generation_classifier_input_or_row_level_score_public": False,
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
            instrument_plan_path=args.instrument_plan,
            topology_path=args.topology,
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
