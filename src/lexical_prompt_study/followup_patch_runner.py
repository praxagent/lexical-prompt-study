from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .behavior import _peak_memory
from .followup_design import PLACEMENTS
from .followup_patch import (
    NoOpResidualHook,
    PatchUnit,
    ResidualStatePatch,
    magnitude_matched_random_deltas,
    paired_interval,
    select_cross_behavior_donors,
    stable_patch_seed,
)
from .followup_plan import validate_followup_plan
from .hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from .models import FollowupPatchReceipt, FollowupTrialReceipt


@dataclass(frozen=True)
class InputObservation:
    behavior_id: str
    category: str
    placement: str
    arm: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    receipt_path: Path
    receipt_sha256: str
    state_bundle_path: Path
    state_bundle_sha256: str
    states: dict[int, Any]
    state_source_sha256: dict[int, str]


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def validate_patch_run_authorization(
    *,
    plan: dict[str, Any],
    patch_private_plan: dict[str, Any],
    patch_private_plan_sha256: str,
    source_commit: str,
    partition: str,
    qualification_only: bool,
    run_id: str,
) -> dict[str, Any]:
    if qualification_only:
        matches = [
            (name, row)
            for name, row in plan["compute"]["scientific_runs"].items()
            if name.startswith("g4_patch_")
            and row.get("qualification_only") is True
            and row.get("run_id") == run_id
        ]
        if len(matches) != 1:
            raise ValueError("qualification run is not uniquely prospectively authorized")
        binding_name, authorization = matches[0]
        expected_statuses = {
            "safe_only_authorized_target_closed",
            "throughput_only_authorized_target_closed",
        }
    elif partition == "discovery":
        expected_statuses = {
            "authorized_after_safe_qualification",
            "authorized_partial_resume_after_missing_state_stop",
        }
        matches = [
            (name, row)
            for name, row in plan["compute"]["scientific_runs"].items()
            if name.startswith("g4_patch_discovery")
            and row.get("qualification_only") is False
            and row.get("run_id") == run_id
            and row.get("status") in expected_statuses
        ]
        if len(matches) != 1:
            raise ValueError("discovery run is not uniquely prospectively authorized")
        binding_name, authorization = matches[0]
    else:
        binding_name = "g4_patch_calibration"
        expected_statuses = {"authorized_after_discovery_selection"}
    if not qualification_only and partition != "discovery":
        try:
            authorization = plan["compute"]["scientific_runs"][binding_name]
        except KeyError as exc:
            raise ValueError(f"{binding_name} is not prospectively authorized") from exc
    input_binding = authorization["input_binding"]
    private_input_plan_sha = input_binding.get("patch_private_input_plan_sha256")
    if private_input_plan_sha is None:
        private_input_plan_sha = input_binding["patch_private_scientific_plan_sha256"]
    if (
        authorization["status"] not in expected_statuses
        or authorization["runner_source_commit"] != source_commit
        or authorization["partition"] != partition
        or authorization["qualification_only"] is not qualification_only
        or authorization["run_id"] != run_id
        or input_binding["patch_private_plan_sha256"] != patch_private_plan_sha256
        or patch_private_plan["public_plan_sha256"] != private_input_plan_sha
        or authorization["target_generation_authorized"] is qualification_only
    ):
        raise ValueError("patch run authorization binding drift")
    return authorization


def _save_private_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_bytes(encoded)


def _tensor_sha256(torch, tensor) -> str:
    value = tensor.detach().to("cpu", dtype=torch.bfloat16).contiguous()
    raw = value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _save_replay_bundle(
    torch,
    path: Path,
    *,
    recipient,
    realized_delta,
    provenance: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "provenance": provenance,
            "recipient_pre_patch": recipient.to("cpu", dtype=torch.bfloat16),
            "realized_delta": realized_delta.to("cpu", dtype=torch.bfloat16),
        },
        temporary,
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_file(path)


def _save_private_torch(torch, path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_file(path)


def _stable_trial_id(
    *,
    study_id: str,
    run_id: str,
    partition: str,
    placement: str,
    layer: int,
    condition: str,
    behavior_id: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [
                study_id,
                run_id,
                partition,
                placement,
                int(layer),
                condition,
                behavior_id,
            ]
        )
    ).hexdigest()[:24]


def _pad_prompts(torch, prompts: list[tuple[int, ...]], *, pad_token_id: int, device):
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("patch batch contains an empty prompt")
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(prompts), width),
        dtype=torch.long,
        device=device,
    )
    for row_index, prompt in enumerate(prompts):
        input_ids[row_index, width - len(prompt) :] = torch.tensor(
            prompt,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, width - len(prompt) :] = 1
    return input_ids, attention_mask


def _capture_states_and_logits(
    torch,
    model,
    *,
    input_ids,
    attention_mask,
    layers: list[int],
):
    captured = {}
    handles = []
    for layer in layers:
        def hook(_module, _inputs, output, *, layer_index=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[:, -1, :].detach().to(
                "cpu", dtype=torch.bfloat16
            )

        handles.append(model.model.layers[layer].register_forward_hook(hook))
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, -1, :].float()
    for handle in handles:
        handle.remove()
    if sorted(captured) != sorted(layers):
        raise ValueError("safe positive-control state capture incomplete")
    return captured, logits


def _logits_with_hook(
    torch,
    model,
    *,
    input_ids,
    attention_mask,
    layer: int,
    hook,
):
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :].float()
    finally:
        handle.remove()
    if not hook.applied or hook.replay is None:
        raise ValueError("safe positive-control hook was not applied exactly once")
    return logits


def _safe_margin_shift(
    logits,
    baseline_logits,
    *,
    donor_ids: list[int],
    recipient_ids: list[int],
):
    rows = list(range(len(donor_ids)))
    patched = (
        logits[rows, donor_ids] - logits[rows, recipient_ids]
    ).detach().cpu().numpy()
    baseline = (
        baseline_logits[rows, donor_ids] - baseline_logits[rows, recipient_ids]
    ).detach().cpu().numpy()
    return patched.astype(np.float64) - baseline.astype(np.float64)


def _safe_metric(
    values: np.ndarray,
    *,
    level: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    return {
        "mean_shift": float(values.mean()),
        "positive_concordance": float(np.mean(values > 0)),
        "interval_level": level,
        "interval": paired_interval(
            values,
            level=level,
            replicates=replicates,
            seed=seed,
        ),
    }


def run_safe_positive_control(
    *,
    torch,
    model,
    tokenizer,
    plan: dict[str, Any],
    public_plan_sha256: str,
    patch_private_plan: dict[str, Any],
    patch_private_plan_sha256: str,
    output_root: Path,
    run_id: str,
    source_commit: str,
) -> dict[str, Any]:
    public_path = output_root / "safe-positive-control.public.json"
    if public_path.exists():
        existing = json.loads(public_path.read_text())
        if (
            existing.get("public_plan_sha256") != public_plan_sha256
            or existing.get("patch_private_plan_sha256")
            != patch_private_plan_sha256
            or existing.get("source_commit") != source_commit
            or existing.get("run_id") != run_id
        ):
            raise ValueError("safe positive-control result provenance drift")
        return existing

    layers = [int(value) for value in plan["causal_localization"]["coarse_residual_post_layers"]]
    pairs = patch_private_plan["pairs"]
    recipient_prompts = [
        tuple(int(value) for value in pair["recipient"]["prompt_token_ids"])
        for pair in pairs
    ]
    donor_prompts = [
        tuple(int(value) for value in pair["donor"]["prompt_token_ids"])
        for pair in pairs
    ]
    recipient_ids = [int(pair["recipient"]["answer_token_id"]) for pair in pairs]
    donor_ids = [int(pair["donor"]["answer_token_id"]) for pair in pairs]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    device = next(model.parameters()).device
    recipient_input, recipient_mask = _pad_prompts(
        torch, recipient_prompts, pad_token_id=pad_id, device=device
    )
    donor_input, donor_mask = _pad_prompts(
        torch, donor_prompts, pad_token_id=pad_id, device=device
    )
    recipient_states, recipient_logits = _capture_states_and_logits(
        torch,
        model,
        input_ids=recipient_input,
        attention_mask=recipient_mask,
        layers=layers,
    )
    donor_states, _ = _capture_states_and_logits(
        torch,
        model,
        input_ids=donor_input,
        attention_mask=donor_mask,
        layers=layers,
    )
    specification = plan["causal_localization"]["safe_positive_control"]
    analysis = plan["causal_localization"]["execution"]["analysis"]
    replicates = int(analysis["bootstrap_replicates"])
    base_seed = int(analysis["bootstrap_seed"])
    private_rows = []
    public_layers = []
    all_passed = True
    for layer in layers:
        patch = ResidualStatePatch(
            torch,
            replacement=donor_states[layer],
        )
        patched_logits = _logits_with_hook(
            torch,
            model,
            input_ids=recipient_input,
            attention_mask=recipient_mask,
            layer=layer,
            hook=patch,
        )
        identity = ResidualStatePatch(
            torch,
            replacement=recipient_states[layer],
        )
        identity_logits = _logits_with_hook(
            torch,
            model,
            input_ids=recipient_input,
            attention_mask=recipient_mask,
            layer=layer,
            hook=identity,
        )
        noop = NoOpResidualHook(torch, batch_size=len(pairs))
        noop_logits = _logits_with_hook(
            torch,
            model,
            input_ids=recipient_input,
            attention_mask=recipient_mask,
            layer=layer,
            hook=noop,
        )
        shifts = {
            "donor_into_recipient": _safe_margin_shift(
                patched_logits,
                recipient_logits,
                donor_ids=donor_ids,
                recipient_ids=recipient_ids,
            ),
            "recipient_identity": _safe_margin_shift(
                identity_logits,
                recipient_logits,
                donor_ids=donor_ids,
                recipient_ids=recipient_ids,
            ),
            "no_op_hook": _safe_margin_shift(
                noop_logits,
                recipient_logits,
                donor_ids=donor_ids,
                recipient_ids=recipient_ids,
            ),
        }
        metrics = {
            name: _safe_metric(
                values,
                level=0.95 if name == "donor_into_recipient" else 0.9,
                seed=stable_patch_seed(
                    base_seed=base_seed,
                    partition="safe_positive_control",
                    placement="shared",
                    layer=layer,
                    condition=name,
                    behavior_id="aggregate",
                ),
                replicates=replicates,
            )
            for name, values in shifts.items()
        }
        primary = metrics["donor_into_recipient"]
        identity_passed = all(
            abs(metrics[name]["mean_shift"]) <= 0.02
            and metrics[name]["interval"][0] >= -0.05
            and metrics[name]["interval"][1] <= 0.05
            for name in ("recipient_identity", "no_op_hook")
        )
        layer_passed = (
            primary["mean_shift"] >= float(specification["minimum_mean_shift"])
            and primary["positive_concordance"]
            >= float(specification["minimum_sign_concordance"])
            and primary["interval"][0] > 0
            and identity_passed
        )
        all_passed &= layer_passed
        public_layers.append(
            {
                "layer": layer,
                "metrics": metrics,
                "identity_and_noop_passed": identity_passed,
                "gate_passed": layer_passed,
            }
        )
        private_rows.extend(
            {
                "pair_id": pairs[index]["pair_id"],
                "layer": layer,
                **{name: float(values[index]) for name, values in shifts.items()},
            }
            for index in range(len(pairs))
        )
    private_sha = _save_private_json(
        output_root / "private" / "safe-positive-control.rows.json",
        {
            "schema_version": "1.0",
            "public_plan_sha256": public_plan_sha256,
            "patch_private_plan_sha256": patch_private_plan_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "rows": private_rows,
        },
    )
    result = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "status": "passed" if all_passed else "failed_invalidates_causal_arm",
        "public_plan_sha256": public_plan_sha256,
        "patch_private_plan_sha256": patch_private_plan_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "pair_count": len(pairs),
        "candidate_layer_count": len(layers),
        "all_candidate_layers_passed": all_passed,
        "layers": public_layers,
        "private_rows_sha256": private_sha,
        "raw_prompts_or_token_ids_public": False,
    }
    write_json_atomic(public_path, result)
    return result


def load_frozen_safe_positive_control(
    *,
    path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    instrument = plan["causal_localization"]["instrument_strength_calibration"]
    if sha256_file(path) != instrument["source_result_sha256"]:
        raise ValueError("frozen safe positive-control result hash drift")
    result = json.loads(path.read_text())
    if (
        result["run_id"] != instrument["source_run_id"]
        or result["source_commit"] != instrument["source_commit"]
        or result["public_plan_sha256"] != instrument["source_public_plan_sha256"]
        or result["patch_private_plan_sha256"]
        != instrument["source_private_plan_sha256"]
        or result["pair_count"] != instrument["source_pair_count"]
        or result["raw_prompts_or_token_ids_public"] is not False
    ):
        raise ValueError("frozen safe positive-control provenance drift")
    by_layer = {int(row["layer"]): row for row in result["layers"]}
    all_layers = plan["causal_localization"]["coarse_residual_post_layers"]
    eligible = instrument["target_candidate_layers"]
    excluded = instrument["excluded_instrument_weak_layers"]
    if (
        sorted(by_layer) != sorted(all_layers)
        or [layer for layer in all_layers if by_layer[layer]["gate_passed"]] != eligible
        or [layer for layer in all_layers if not by_layer[layer]["gate_passed"]]
        != excluded
        or not all(row["identity_and_noop_passed"] for row in by_layer.values())
    ):
        raise ValueError("frozen safe positive-control layer partition drift")
    return result


def projected_discovery_maximum_tokens(plan: dict[str, Any]) -> int:
    return (
        2
        * len(
            plan["causal_localization"]["instrument_strength_calibration"][
                "target_candidate_layers"
            ]
        )
        * len(plan["causal_localization"]["execution"]["condition_kinds"])
        * 20
        * int(plan["replication"]["decoding"]["max_new_tokens"])
    )


def run_safe_throughput_qualification(
    *,
    torch,
    model,
    tokenizer,
    plan: dict[str, Any],
    public_plan_sha256: str,
    patch_private_plan: dict[str, Any],
    patch_private_plan_sha256: str,
    output_root: Path,
    run_id: str,
    source_commit: str,
    allowed_provenance: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    public_path = output_root / "safe-throughput-qualification.public.json"
    if public_path.exists():
        existing = json.loads(public_path.read_text())
        provenance = (
            existing.get("public_plan_sha256"),
            existing.get("source_commit"),
        )
        valid_provenance = allowed_provenance or {
            (public_plan_sha256, source_commit)
        }
        if (
            provenance not in valid_provenance
            or existing.get("patch_private_plan_sha256")
            != patch_private_plan_sha256
            or existing.get("run_id") != run_id
        ):
            raise ValueError("safe throughput result provenance drift")
        return existing
    prompts = [
        tuple(int(value) for value in row["prompt_token_ids"])
        for row in patch_private_plan["qualification_prompts"]
    ]
    if len(prompts) != 20:
        raise ValueError("safe throughput prompt-count drift")
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, attention_mask = _pad_prompts(
        torch,
        prompts,
        pad_token_id=pad_id,
        device=device,
    )
    hook = NoOpResidualHook(torch, batch_size=len(prompts))
    handle = model.model.layers[0].register_forward_hook(hook)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=int(plan["replication"]["decoding"]["max_new_tokens"]),
                pad_token_id=pad_id,
                use_cache=True,
            )
    finally:
        handle.remove()
    elapsed = time.monotonic() - started
    if not hook.applied or hook.replay is None:
        raise ValueError("safe throughput no-op hook was not applied")
    generated = output[:, input_ids.shape[1] :].tolist()
    aggregate_tokens = sum(len(row) for row in generated)
    if aggregate_tokens < 1 or elapsed <= 0:
        raise ValueError("safe throughput qualification produced no work")
    private_sha = _save_private_json(
        output_root / "private" / "safe-throughput.outputs.json",
        {
            "schema_version": "1.0",
            "public_plan_sha256": public_plan_sha256,
            "patch_private_plan_sha256": patch_private_plan_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "generated_token_ids": generated,
            "generated_text": [
                tokenizer.decode(
                    row,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for row in generated
            ],
        },
    )
    maximum_target_tokens = projected_discovery_maximum_tokens(plan)
    result = {
        "schema_version": "1.0",
        "status": "passed",
        "public_plan_sha256": public_plan_sha256,
        "patch_private_plan_sha256": patch_private_plan_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "batch_size": len(prompts),
        "maximum_new_tokens_per_prompt": int(
            plan["replication"]["decoding"]["max_new_tokens"]
        ),
        "aggregate_generated_tokens": aggregate_tokens,
        "elapsed_seconds": elapsed,
        "aggregate_generated_tokens_per_second": aggregate_tokens / elapsed,
        "peak_memory_bytes": _peak_memory(torch),
        "projected_discovery_maximum_generated_tokens": maximum_target_tokens,
        "projected_discovery_generation_seconds_at_measured_rate": (
            maximum_target_tokens / (aggregate_tokens / elapsed)
        ),
        "private_outputs_sha256": private_sha,
        "raw_output_public": False,
    }
    write_json_atomic(public_path, result)
    return result


def _load_partition_inputs(
    torch,
    *,
    plan: dict[str, Any],
    partition: str,
    generation_root: Path,
) -> dict[str, dict[str, dict[str, InputObservation]]]:
    binding_name = (
        "g2_discovery"
        if partition == "discovery"
        else "g2_calibration_generation"
    )
    binding = plan["compute"]["scientific_runs"][binding_name]["result_binding"]
    result: dict[str, dict[str, dict[str, InputObservation]]] = {
        placement: {} for placement in PLACEMENTS
    }
    for receipt_path in sorted((generation_root / "receipts/trials").glob("*.json")):
        receipt = FollowupTrialReceipt.model_validate_json(receipt_path.read_text())
        if receipt.partition != partition or receipt.arm not in {
            "full",
            "structural_sham",
        }:
            continue
        if (
            receipt.source_commit != binding["source_commit"]
            or receipt.plan_sha256 != binding["public_plan_sha256"]
            or receipt.private_plan_sha256 != binding["private_plan_sha256"]
            or receipt.run_id != binding["run_id"]
        ):
            raise ValueError("patch input generation provenance drift")
        if receipt.placement not in PLACEMENTS:
            raise ValueError("patch input lacks placement")
        restricted_path = Path(receipt.restricted_artifact_path)
        state_path = Path(receipt.state_bundle_path)
        if (
            sha256_file(restricted_path) != receipt.restricted_artifact_sha256
            or sha256_file(state_path) != receipt.state_bundle_sha256
        ):
            raise ValueError("patch input artifact hash drift")
        raw = json.loads(restricted_path.read_text())
        prompt_ids = tuple(int(value) for value in raw["prompt_token_ids"])
        if (
            sha256_bytes(canonical_json_bytes(list(prompt_ids)))
            != receipt.prompt_token_ids_sha256
        ):
            raise ValueError("patch input prompt-token hash drift")
        state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
        states = {
            int(layer): value.to("cpu", dtype=torch.bfloat16)
            for layer, value in state_payload["states"].items()
        }
        observation = InputObservation(
            behavior_id=receipt.behavior_id,
            category=receipt.category,
            placement=receipt.placement,
            arm=receipt.arm,
            prompt_token_ids=prompt_ids,
            prompt_token_ids_sha256=receipt.prompt_token_ids_sha256,
            receipt_path=receipt_path,
            receipt_sha256=sha256_file(receipt_path),
            state_bundle_path=state_path,
            state_bundle_sha256=receipt.state_bundle_sha256,
            states=states,
            state_source_sha256={
                layer: receipt.state_bundle_sha256 for layer in states
            },
        )
        by_behavior = result[receipt.placement].setdefault(
            receipt.behavior_id, {}
        )
        if receipt.arm in by_behavior:
            raise ValueError("duplicate patch input arm")
        by_behavior[receipt.arm] = observation
    for placement in PLACEMENTS:
        if len(result[placement]) != 20 or any(
            set(arms) != {"full", "structural_sham"}
            for arms in result[placement].values()
        ):
            raise ValueError(f"{placement}: incomplete patch input topology")
    return result


def ensure_candidate_states(
    torch,
    model,
    tokenizer,
    *,
    inputs: dict[str, dict[str, dict[str, InputObservation]]],
    required_layers: list[int],
    output_root: Path,
    public_plan_sha256: str,
    patch_private_plan_sha256: str,
    source_commit: str,
    run_id: str,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    receipts = []
    for placement in PLACEMENTS:
        for arm in ("full", "structural_sham"):
            observations = [
                inputs[placement][behavior_id][arm]
                for behavior_id in sorted(inputs[placement])
            ]
            for layer in required_layers:
                missing = [row for row in observations if layer not in row.states]
                if not missing:
                    continue
                if len(missing) != len(observations):
                    raise ValueError("partial derived-state topology")
                path = (
                    output_root
                    / "derived-states"
                    / f"{placement}.{arm}.layer-{layer}.pt"
                )
                expected = {
                    "schema_version": "1.0",
                    "public_plan_sha256": public_plan_sha256,
                    "patch_private_plan_sha256": patch_private_plan_sha256,
                    "source_commit": source_commit,
                    "run_id": run_id,
                    "placement": placement,
                    "arm": arm,
                    "layer": layer,
                    "behavior_ids": [row.behavior_id for row in observations],
                    "prompt_token_ids_sha256": [
                        row.prompt_token_ids_sha256 for row in observations
                    ],
                }
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if payload.get("provenance") != expected:
                        raise ValueError("derived candidate-state provenance drift")
                    states = payload["states"].to("cpu", dtype=torch.bfloat16)
                else:
                    input_ids, attention_mask = _pad_prompts(
                        torch,
                        [row.prompt_token_ids for row in observations],
                        pad_token_id=pad_id,
                        device=device,
                    )
                    captured: dict[str, Any] = {}

                    def capture(_module, _args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        captured["states"] = hidden[:, -1, :].detach()
                        return output

                    handle = model.model.layers[layer].register_forward_hook(capture)
                    try:
                        with torch.inference_mode():
                            model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                            )
                    finally:
                        handle.remove()
                    if "states" not in captured:
                        raise ValueError("derived candidate-state capture missing")
                    states = captured["states"].to("cpu", dtype=torch.bfloat16)
                    if states.shape != (len(observations), int(model.config.hidden_size)):
                        raise ValueError("derived candidate-state shape drift")
                    _save_private_torch(
                        torch,
                        path,
                        {
                            "provenance": expected,
                            "states": states,
                        },
                    )
                state_sha = sha256_file(path)
                for index, observation in enumerate(observations):
                    observation.states[layer] = states[index]
                    observation.state_source_sha256[layer] = state_sha
                receipts.append(
                    {
                        "placement": placement,
                        "arm": arm,
                        "layer": layer,
                        "path": str(path),
                        "sha256": state_sha,
                        "observation_count": len(observations),
                    }
                )
    return receipts


def _condition_batch(
    torch,
    *,
    plan: dict[str, Any],
    partition: str,
    placement: str,
    candidate_layer: int,
    condition: str,
    inputs: dict[str, dict[str, InputObservation]],
    cross_donors: dict[str, str],
):
    behavior_ids = sorted(inputs)
    full = [inputs[behavior_id]["full"] for behavior_id in behavior_ids]
    sham = [inputs[behavior_id]["structural_sham"] for behavior_id in behavior_ids]
    applied_layer = candidate_layer
    token_offset = -1
    donor_arm: str | None = None
    donor_behavior_ids: list[str | None] = behavior_ids.copy()
    donor_state_hashes: list[str | None]
    if condition == "sham_into_full":
        recipients, donors = full, sham
        donor_arm = "structural_sham"
        replacement = torch.stack([row.states[candidate_layer] for row in donors])
        hook_kind = "replacement"
    elif condition == "full_into_sham":
        recipients, donors = sham, full
        donor_arm = "full"
        replacement = torch.stack([row.states[candidate_layer] for row in donors])
        hook_kind = "replacement"
    elif condition == "full_into_full_identity":
        recipients, donors = full, full
        donor_arm = "full"
        replacement = torch.stack([row.states[candidate_layer] for row in donors])
        hook_kind = "replacement"
    elif condition == "sham_into_sham_identity":
        recipients, donors = sham, sham
        donor_arm = "structural_sham"
        replacement = torch.stack([row.states[candidate_layer] for row in donors])
        hook_kind = "replacement"
    elif condition == "no_op_hook":
        recipients, donors = full, []
        donor_behavior_ids = [None] * len(behavior_ids)
        replacement = None
        hook_kind = "noop"
    elif condition == "same_site_magnitude_matched_seeded_random_delta":
        recipients, donors = full, sham
        donor_behavior_ids = [None] * len(behavior_ids)
        reference = torch.stack(
            [
                sham_row.states[candidate_layer] - full_row.states[candidate_layer]
                for full_row, sham_row in zip(full, sham, strict=True)
            ]
        )
        seeds = [
            stable_patch_seed(
                base_seed=int(
                    plan["causal_localization"]["execution"]["residual_post_hook"][
                        "random_seed"
                    ]
                ),
                partition=partition,
                placement=placement,
                layer=candidate_layer,
                condition=condition,
                behavior_id=behavior_id,
            )
            for behavior_id in behavior_ids
        ]
        replacement = magnitude_matched_random_deltas(
            torch, reference, seeds=seeds
        )
        hook_kind = "delta"
    elif condition == "irrelevant_layer":
        recipients, donors = full, sham
        donor_arm = "structural_sham"
        applied_layer = int(
            plan["causal_localization"]["execution"]["residual_post_hook"][
                "irrelevant_layer"
            ]
        )
        replacement = torch.stack([row.states[applied_layer] for row in donors])
        hook_kind = "replacement"
    elif condition == "irrelevant_token_position":
        recipients, donors = full, sham
        donor_arm = "structural_sham"
        token_offset = -2
        replacement = torch.stack(
            [
                sham_row.states[candidate_layer] - full_row.states[candidate_layer]
                for full_row, sham_row in zip(full, sham, strict=True)
            ]
        )
        hook_kind = "delta"
    elif condition == "cross_behavior_category_and_length_matched_donor":
        recipients = full
        donors = [
            inputs[cross_donors[behavior_id]]["structural_sham"]
            for behavior_id in behavior_ids
        ]
        donor_arm = "structural_sham"
        donor_behavior_ids = [row.behavior_id for row in donors]
        replacement = torch.stack([row.states[candidate_layer] for row in donors])
        hook_kind = "replacement"
    else:
        raise ValueError(f"unknown patch condition: {condition}")
    state_layer = applied_layer if condition == "irrelevant_layer" else candidate_layer
    donor_state_hashes = (
        [row.state_source_sha256[state_layer] for row in donors]
        if donors
        else [None] * len(behavior_ids)
    )
    return {
        "behavior_ids": behavior_ids,
        "recipients": recipients,
        "donor_behavior_ids": donor_behavior_ids,
        "donor_state_hashes": donor_state_hashes,
        "donor_arm": donor_arm,
        "applied_layer": applied_layer,
        "token_offset": token_offset,
        "hook_kind": hook_kind,
        "patch_tensor": replacement,
    }


def _trim_generated(ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[: index + 1], True
    return ids, False


def _load_completed_patch_receipt(
    path: Path,
    *,
    patch_private_plan_sha256: str,
    run_id: str,
    allowed_provenance: set[tuple[str, str]],
) -> FollowupPatchReceipt | None:
    if not path.exists():
        return None
    receipt = FollowupPatchReceipt.model_validate_json(path.read_text())
    if (
        (receipt.public_plan_sha256, receipt.source_commit) not in allowed_provenance
        or receipt.patch_private_plan_sha256 != patch_private_plan_sha256
        or receipt.run_id != run_id
    ):
        raise ValueError("completed patch receipt provenance drift")
    for artifact, expected in (
        (Path(receipt.restricted_artifact_path), receipt.restricted_artifact_sha256),
        (Path(receipt.replay_bundle_path), receipt.replay_bundle_sha256),
    ):
        if not artifact.exists() or sha256_file(artifact) != expected:
            raise ValueError("completed patch artifact hash drift")
    return receipt


def run_followup_coarse_patch_generation(
    *,
    public_plan_path: Path,
    patch_private_plan_path: Path,
    generation_root: Path,
    model_path: str,
    output_root: Path,
    partition: str,
    run_id: str,
    selected_layer: int | None = None,
    qualification_only: bool = False,
    safe_positive_control_result_path: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(plan)
    if partition not in {"discovery", "calibration"}:
        raise ValueError("coarse patch partition must be discovery or calibration")
    layers = [
        int(value)
        for value in plan["causal_localization"][
            "instrument_strength_calibration"
        ]["target_candidate_layers"]
    ]
    if partition == "discovery":
        if selected_layer is not None:
            raise ValueError("discovery cannot receive a selected layer")
        candidate_layers = layers
    else:
        if selected_layer not in layers:
            raise ValueError("calibration requires one frozen coarse layer")
        candidate_layers = [int(selected_layer)]

    public_sha = sha256_file(public_plan_path)
    patch_private_sha = sha256_file(patch_private_plan_path)
    patch_private = json.loads(patch_private_plan_path.read_text())
    if (
        patch_private["study_id"] != plan["study_id"]
        or patch_private["pair_count"] != 20
        or patch_private["tokenizer_revision"]
        != plan["artifacts"]["llama31_model"]["revision"]
    ):
        raise ValueError("patch private plan drift")
    source_commit = _source_commit()
    validate_patch_run_authorization(
        plan=plan,
        patch_private_plan=patch_private,
        patch_private_plan_sha256=patch_private_sha,
        source_commit=source_commit,
        partition=partition,
        qualification_only=qualification_only,
        run_id=run_id,
    )
    expected_revision = plan["artifacts"]["llama31_model"]["revision"]
    if expected_revision not in str(Path(model_path).resolve()):
        raise ValueError("patch model path is not the frozen snapshot")

    import torch
    import transformers

    if torch.cuda.device_count() != 1:
        raise ValueError("coarse patch runner requires exactly one visible GPU")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer.padding_side = "left"
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or expected_revision
    if tokenizer_revision != expected_revision:
        raise ValueError("patch tokenizer revision drift")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().to("cuda")
    if (
        int(model.config.hidden_size) != 4096
        or int(model.config.num_hidden_layers) != 32
        or int(model.config.vocab_size) != 128256
    ):
        raise ValueError("patch model topology drift")
    model_revision = getattr(model.config, "_commit_hash", None) or expected_revision
    if model_revision != expected_revision:
        raise ValueError("patch model revision drift")

    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    resume = plan["causal_localization"]["execution"]["partial_resume"]
    resume_active = (
        resume["enabled"]
        and partition == resume["partition"]
        and run_id == resume["run_id"]
    )
    allowed_provenance = {(public_sha, source_commit)}
    if resume_active:
        allowed_provenance.add(
            (
                resume["predecessor_public_plan_sha256"],
                resume["predecessor_source_commit"],
            )
        )
    if safe_positive_control_result_path is None:
        raise ValueError("layer-specific protocol requires frozen safe-control result")
    safe = load_frozen_safe_positive_control(
        path=safe_positive_control_result_path,
        plan=plan,
    )
    safe_copy_path = output_root / "safe-positive-control.public.json"
    if safe_copy_path.exists():
        if sha256_file(safe_copy_path) != sha256_file(
            safe_positive_control_result_path
        ):
            raise ValueError("copied frozen safe positive-control result drift")
    else:
        write_json_atomic(safe_copy_path, safe)
        if sha256_file(safe_copy_path) != sha256_file(
            safe_positive_control_result_path
        ):
            raise ValueError("frozen safe positive-control byte copy drift")
    qualification = run_safe_throughput_qualification(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        plan=plan,
        public_plan_sha256=public_sha,
        patch_private_plan=patch_private,
        patch_private_plan_sha256=patch_private_sha,
        output_root=output_root,
        run_id=run_id,
        source_commit=source_commit,
        allowed_provenance=allowed_provenance,
    )
    if qualification_only:
        summary = {
            "schema_version": "1.0",
            "status": "qualification_complete_no_target_outcome",
            "partition": partition,
            "run_id": run_id,
            "source_commit": source_commit,
            "public_plan_sha256": public_sha,
            "patch_private_plan_sha256": patch_private_sha,
            "trials": 0,
            "safe_positive_control_status": "passed_layer_specific_instrument_gate",
            "safe_positive_control_source_status": safe["status"],
            "safe_throughput_status": qualification["status"],
            "elapsed_seconds": time.monotonic() - started,
        }
        _save_private_json(output_root / "summary.json", summary)
        return summary

    inputs = _load_partition_inputs(
        torch,
        plan=plan,
        partition=partition,
        generation_root=generation_root,
    )
    derived_state_receipts = ensure_candidate_states(
        torch,
        model,
        tokenizer,
        inputs=inputs,
        required_layers=sorted(
            set(candidate_layers)
            | {
                int(
                    plan["causal_localization"]["execution"]["residual_post_hook"][
                        "irrelevant_layer"
                    ]
                )
            }
        ),
        output_root=output_root,
        public_plan_sha256=public_sha,
        patch_private_plan_sha256=patch_private_sha,
        source_commit=source_commit,
        run_id=run_id,
    )
    write_json_atomic(
        output_root / "derived-state-index.public.json",
        {
            "schema_version": "1.0",
            "study_id": plan["study_id"],
            "public_plan_sha256": public_sha,
            "patch_private_plan_sha256": patch_private_sha,
            "source_commit": source_commit,
            "run_id": run_id,
            "derived_state_receipts": derived_state_receipts,
            "raw_prompts_token_ids_and_tensors_public": False,
        },
    )
    conditions = plan["causal_localization"]["execution"]["condition_kinds"]
    expected_trials = len(PLACEMENTS) * len(candidate_layers) * len(conditions) * 20
    receipt_root = output_root / "receipts" / "trials"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_root.chmod(0o700)
    existing_paths = sorted(receipt_root.glob("*.json"))
    expected_existing = (
        int(resume["predecessor_completed_trial_count"]) if resume_active else 0
    )
    if len(existing_paths) != expected_existing:
        raise ValueError("partial-resume receipt count drift")
    for existing_path in existing_paths:
        _load_completed_patch_receipt(
            existing_path,
            patch_private_plan_sha256=patch_private_sha,
            run_id=run_id,
            allowed_provenance=allowed_provenance,
        )
    decoding = plan["replication"]["decoding"]
    max_new_tokens = int(decoding["max_new_tokens"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_ids = {
        int(value)
        for value in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
        if value is not None and value != tokenizer.unk_token_id
    }
    device = next(model.parameters()).device
    written = 0
    completed_count = 0
    batch_index = 0
    total_batches = len(PLACEMENTS) * len(candidate_layers) * len(conditions)
    for placement in PLACEMENTS:
        units = [
            PatchUnit(
                behavior_id=behavior_id,
                category=arms["full"].category,
                prompt_token_count=len(arms["full"].prompt_token_ids),
            )
            for behavior_id, arms in sorted(inputs[placement].items())
        ]
        cross_donors = select_cross_behavior_donors(units)
        for candidate_layer in candidate_layers:
            for condition in conditions:
                batch_index += 1
                batch = _condition_batch(
                    torch,
                    plan=plan,
                    partition=partition,
                    placement=placement,
                    candidate_layer=candidate_layer,
                    condition=condition,
                    inputs=inputs[placement],
                    cross_donors=cross_donors,
                )
                paths = [
                    receipt_root
                    / (
                        _stable_trial_id(
                            study_id=plan["study_id"],
                            run_id=run_id,
                            partition=partition,
                            placement=placement,
                            layer=candidate_layer,
                            condition=condition,
                            behavior_id=behavior_id,
                        )
                        + ".json"
                    )
                    for behavior_id in batch["behavior_ids"]
                ]
                completed = [
                    _load_completed_patch_receipt(
                        path,
                        patch_private_plan_sha256=patch_private_sha,
                        run_id=run_id,
                        allowed_provenance=allowed_provenance,
                    )
                    for path in paths
                ]
                if all(item is not None for item in completed):
                    completed_count += len(completed)
                    continue
                if any(item is not None for item in completed):
                    raise ValueError("partial patch batch requires exact whole-batch resume")
                recipients = batch["recipients"]
                input_ids, attention_mask = _pad_prompts(
                    torch,
                    [row.prompt_token_ids for row in recipients],
                    pad_token_id=pad_id,
                    device=device,
                )
                if batch["hook_kind"] == "noop":
                    hook = NoOpResidualHook(torch, batch_size=len(recipients))
                elif batch["hook_kind"] == "replacement":
                    hook = ResidualStatePatch(
                        torch,
                        replacement=batch["patch_tensor"],
                        token_offset=batch["token_offset"],
                    )
                else:
                    hook = ResidualStatePatch(
                        torch,
                        delta=batch["patch_tensor"],
                        token_offset=batch["token_offset"],
                    )
                handle = model.model.layers[batch["applied_layer"]].register_forward_hook(
                    hook
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=pad_id,
                            use_cache=True,
                        )
                finally:
                    handle.remove()
                elapsed = time.monotonic() - before
                if not hook.applied or hook.replay is None:
                    raise ValueError("target patch hook was not applied")
                generated_batch = output[:, input_ids.shape[1] :].tolist()
                per_trial_elapsed = elapsed / len(recipients)
                for row_index, (
                    behavior_id,
                    recipient,
                    receipt_path,
                    generated_ids,
                ) in enumerate(
                    zip(
                        batch["behavior_ids"],
                        recipients,
                        paths,
                        generated_batch,
                        strict=True,
                    )
                ):
                    generated_ids, ended = _trim_generated(generated_ids, eos_ids)
                    generated_text = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    trial_id = receipt_path.stem
                    restricted_path = output_root / "restricted" / f"{trial_id}.json"
                    restricted_sha = _save_private_json(
                        restricted_path,
                        {
                            "trial_id": trial_id,
                            "behavior_id": behavior_id,
                            "partition": partition,
                            "placement": placement,
                            "candidate_layer": candidate_layer,
                            "condition": condition,
                            "generated_token_ids": generated_ids,
                            "generated_text": generated_text,
                        },
                    )
                    recipient_tensor = hook.replay["recipient"][row_index]
                    delta_tensor = hook.replay["realized_delta"][row_index]
                    replay_path = output_root / "replay" / f"{trial_id}.pt"
                    replay_provenance = {
                        "trial_id": trial_id,
                        "public_plan_sha256": public_sha,
                        "patch_private_plan_sha256": patch_private_sha,
                        "source_commit": source_commit,
                        "run_id": run_id,
                        "partition": partition,
                        "placement": placement,
                        "candidate_layer": candidate_layer,
                        "applied_layer": batch["applied_layer"],
                        "condition": condition,
                        "behavior_id": behavior_id,
                    }
                    replay_sha = _save_replay_bundle(
                        torch,
                        replay_path,
                        recipient=recipient_tensor,
                        realized_delta=delta_tensor,
                        provenance=replay_provenance,
                    )
                    recipient_arm = recipient.arm
                    baseline_arm = (
                        "structural_sham"
                        if condition in {
                            "full_into_sham",
                            "sham_into_sham_identity",
                        }
                        else "full"
                    )
                    payload = {
                        "schema_version": "1.0",
                        "study_id": plan["study_id"],
                        "public_plan_sha256": public_sha,
                        "patch_private_plan_sha256": patch_private_sha,
                        "source_commit": source_commit,
                        "run_id": run_id,
                        "stage": (
                            "coarse_discovery"
                            if partition == "discovery"
                            else "coarse_calibration"
                        ),
                        "partition": partition,
                        "trial_id": trial_id,
                        "behavior_id": behavior_id,
                        "category": recipient.category,
                        "placement": placement,
                        "candidate_layer": candidate_layer,
                        "applied_layer": batch["applied_layer"],
                        "condition": condition,
                        "recipient_arm": recipient_arm,
                        "donor_arm": batch["donor_arm"],
                        "donor_behavior_id": batch["donor_behavior_ids"][row_index],
                        "baseline_arm": baseline_arm,
                        "token_offset": batch["token_offset"],
                        "recipient_generation_receipt_sha256": recipient.receipt_sha256,
                        "donor_state_bundle_sha256": batch["donor_state_hashes"][
                            row_index
                        ],
                        "prompt_token_ids_sha256": recipient.prompt_token_ids_sha256,
                        "recipient_pre_patch_sha256": _tensor_sha256(
                            torch, recipient_tensor
                        ),
                        "realized_delta_sha256": _tensor_sha256(
                            torch, delta_tensor
                        ),
                        "recipient_pre_patch_norm": float(
                            recipient_tensor.float().norm().item()
                        ),
                        "realized_delta_norm": float(
                            delta_tensor.float().norm().item()
                        ),
                        "tensor_shape": list(recipient_tensor.shape),
                        "tensor_dtype": "torch.bfloat16",
                        "replay_bundle_path": str(replay_path),
                        "replay_bundle_sha256": replay_sha,
                        "generated_text_sha256": sha256_text(generated_text),
                        "generated_token_ids_sha256": sha256_bytes(
                            canonical_json_bytes(generated_ids)
                        ),
                        "generated_token_count": len(generated_ids),
                        "restricted_artifact_path": str(restricted_path),
                        "restricted_artifact_sha256": restricted_sha,
                        "finish_reason": "eos" if ended else "length",
                        "truncated": not ended,
                        "elapsed_seconds": per_trial_elapsed,
                        "peak_memory_bytes": _peak_memory(torch),
                        "model_revision": model_revision,
                        "tokenizer_revision": tokenizer_revision,
                        "software": {
                            "python": sys.version,
                            "platform": platform.platform(),
                            "torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "cuda": torch.version.cuda,
                        },
                    }
                    validated = FollowupPatchReceipt.model_validate(payload)
                    _save_private_json(
                        receipt_path,
                        validated.model_dump(mode="json"),
                    )
                    written += 1
                completed_count += len(recipients)
                print(
                    f"completed patch batch {batch_index}/{total_batches} "
                    f"placement={placement} layer={candidate_layer} "
                    f"condition={condition}",
                    flush=True,
                )
    if completed_count != expected_trials:
        raise ValueError("coarse patch completion count drift")
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "partition": partition,
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_sha,
        "patch_private_plan_sha256": patch_private_sha,
        "candidate_layers": candidate_layers,
        "condition_count": len(conditions),
        "trials": expected_trials,
        "trials_written_this_call": written,
        "trials_resumed_from_predecessor": expected_existing,
        "derived_state_checkpoint_count": len(derived_state_receipts),
        "safe_positive_control_status": "passed_layer_specific_instrument_gate",
        "safe_positive_control_source_status": safe["status"],
        "safe_throughput_status": qualification["status"],
        "model_loaded_this_call": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    _save_private_json(output_root / "summary.json", summary)
    return summary
