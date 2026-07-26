from __future__ import annotations

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

from .artifacts import MODEL_REVISION
from .behavior import (
    REFUSAL_PATTERN,
    _as_token_ids,
    _peak_memory,
    _save_restricted,
)
from .hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from .intervention_analysis import (
    calibration_condition_id,
    planned_calibration_conditions,
)
from .intervention_plan import validate_intervention_plan
from .interventions import (
    MAXIMUM_DELTA_TO_RESIDUAL_NORM,
    REQUESTED_REALIZED_TOLERANCE,
    direction_sha256,
    stable_intervention_trial_id,
    unit_direction,
    validate_intervention_receipt,
)
from .mechanism_runner import _raw_generation


@dataclass(frozen=True)
class CalibrationCondition:
    condition_id: str
    sign: int
    rho: float | None


def calibration_conditions(plan: dict, *, max_rhos: int | None = None) -> list[CalibrationCondition]:
    rho_values = plan["discovery_alpha_calibration"]["rho_ladder"]
    if max_rhos is not None:
        if max_rhos < 1:
            raise ValueError("max_rhos must be positive")
        rho_values = rho_values[:max_rhos]
    conditions = [CalibrationCondition("zero", 0, None)]
    for rho_value in rho_values:
        rho = float(rho_value)
        conditions.extend(
            [
                CalibrationCondition(calibration_condition_id(-1, rho), -1, rho),
                CalibrationCondition(calibration_condition_id(1, rho), 1, rho),
            ]
        )
    expected = (
        planned_calibration_conditions(plan)
        if max_rhos is None
        else [item.condition_id for item in conditions]
    )
    if [item.condition_id for item in conditions] != expected:
        raise ValueError("calibration condition construction drift")
    return conditions


class InterventionReceiptStore:
    def __init__(self, root: Path):
        self.root = root
        self.trials = root / "trials"
        self.attempts = root / "attempts.jsonl"
        self.trials.mkdir(parents=True, exist_ok=True)

    def load_validated(
        self,
        trial_id: str,
        *,
        plan_sha256: str,
        source_commit: str,
        run_id: str,
    ) -> dict:
        path = self.trials / f"{trial_id}.json"
        payload = json.loads(path.read_text())
        receipt = validate_intervention_receipt(payload)
        if receipt.trial_id != trial_id:
            raise ValueError(f"{trial_id}: receipt trial ID drift")
        if receipt.intervention_plan_sha256 != plan_sha256:
            raise ValueError(f"{trial_id}: intervention plan drift")
        if receipt.source_commit != source_commit or receipt.run_id != run_id:
            raise ValueError(f"{trial_id}: source or run drift")
        restricted = Path(receipt.restricted_text_path)
        if sha256_file(restricted) != receipt.restricted_artifact_sha256:
            raise ValueError(f"{trial_id}: restricted artifact hash mismatch")
        raw = json.loads(restricted.read_text())
        if (
            raw["generated_token_ids"] != receipt.generated_token_ids
            or sha256_text(raw["generated_text"]) != receipt.generated_text_sha256
        ):
            raise ValueError(f"{trial_id}: restricted generation drift")
        return payload

    def write(self, payload: dict) -> str:
        receipt = validate_intervention_receipt(payload)
        encoded = canonical_json_bytes(receipt.model_dump(mode="json"))
        final = self.trials / f"{receipt.trial_id}.json"
        temporary = self.trials / f".{receipt.trial_id}.{os.getpid()}.tmp"
        temporary.write_bytes(encoded)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(final)
        with self.attempts.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return sha256_bytes(encoded)


class ResidualPostIntervention:
    def __init__(self, torch, direction, *, sign: int, alpha: float):
        self.torch = torch
        self.direction = direction.float()
        if self.direction.ndim != 1 or not bool(torch.isfinite(self.direction).all()):
            raise ValueError("hook direction must be a finite vector")
        direction_norm = float(self.direction.norm().item())
        if not np.isclose(direction_norm, 1.0, rtol=1e-4, atol=0):
            raise ValueError("hook direction must be unit normalized")
        if sign not in {-1, 0, 1} or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid hook sign or alpha")
        if (sign == 0) != (alpha == 0):
            raise ValueError("zero hook sign and alpha must occur together")
        self.sign = sign
        self.alpha = float(alpha)
        self.steps: list[dict[str, Any]] = []

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("intervention hook requires batch size one")
        current = hidden[:, -1, :]
        pre_float = current.float()
        pre_norm = float(pre_float.norm().item())
        if not np.isfinite(pre_norm) or pre_norm <= 0:
            raise ValueError("invalid pre-intervention residual norm")
        direction = self.direction.to(device=current.device, dtype=current.dtype)
        requested = direction * (self.sign * self.alpha)
        proposed = current + requested
        effective = proposed.float() - pre_float
        realized = float(effective.norm().item())
        relative_error = (
            0.0
            if self.alpha == 0
            else abs(realized - self.alpha) / self.alpha
        )
        ratio = realized / pre_norm
        if relative_error > REQUESTED_REALIZED_TOLERANCE:
            raise ValueError("requested/realized intervention norm mismatch")
        if ratio > MAXIMUM_DELTA_TO_RESIDUAL_NORM:
            raise ValueError("intervention exceeds residual norm budget")
        post_norm = float(proposed.float().norm().item())
        if not np.isfinite(post_norm) or post_norm <= 0:
            raise ValueError("invalid post-intervention residual norm")
        modified = hidden.clone()
        modified[:, -1, :] = proposed
        self.steps.append(
            {
                "generated_token_index": len(self.steps),
                "requested_delta_norm": self.alpha,
                "realized_delta_norm": realized,
                "pre_residual_norm": pre_norm,
                "post_residual_norm": post_norm,
                "requested_realized_relative_error": relative_error,
                "delta_to_pre_residual_norm": ratio,
                "clipped": False,
            }
        )
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prompt_for_attack(
    *,
    tokenizer,
    behavior: dict,
    private_plan: dict,
    public_plan: dict,
    public_plan_sha256: str,
    generation_root: Path,
) -> tuple[list[int], str, str]:
    first_receipt, first_raw = _raw_generation(
        generation_root=generation_root,
        public_plan_sha256=public_plan_sha256,
        study_id=public_plan["study_id"],
        behavior_id=behavior["behavior_id"],
        arm="full",
        turn=1,
        split="discovery",
    )
    second_receipt, _ = _raw_generation(
        generation_root=generation_root,
        public_plan_sha256=public_plan_sha256,
        study_id=public_plan["study_id"],
        behavior_id=behavior["behavior_id"],
        arm="full",
        turn=2,
        split="discovery",
    )
    messages = [
        {"role": "user", "content": behavior["rendered_arms"]["full"]["text"]},
        {"role": "assistant", "content": first_raw["generated_text"]},
        {"role": "user", "content": private_plan["followup"]},
    ]
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    prompt_token_sha = sha256_bytes(canonical_json_bytes(prompt_ids))
    if prompt_token_sha != second_receipt.prompt_token_ids_sha256:
        raise ValueError(f"{behavior['behavior_id']}: reconstructed prompt drift")
    if sha256_text(first_raw["generated_text"]) != first_receipt.generated_text_sha256:
        raise ValueError(f"{behavior['behavior_id']}: parent generation text drift")
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_ids, prompt_text, sha256_file(
        generation_root
        / "receipts"
        / "trials"
        / f"{first_receipt.trial_id}.json"
    )


def _reference_norm(receipts: list[dict]) -> float:
    norms = [
        float(step["pre_residual_norm"])
        for receipt in receipts
        for step in receipt["intervention_steps"]
    ]
    if not norms or not all(np.isfinite(norm) and norm > 0 for norm in norms):
        raise ValueError("invalid zero-intervention residual norm reference")
    return float(np.median(np.asarray(norms, dtype=float)))


def run_intervention_calibration(
    *,
    private_plan_path: Path,
    public_plan_path: Path,
    intervention_plan_path: Path,
    gate3_analysis_path: Path,
    generation_root: Path,
    model_path: str,
    sae_path: Path,
    output_root: Path,
    run_id: str,
    max_behaviors: int | None = None,
    max_rhos: int | None = None,
    max_new_tokens: int | None = None,
) -> dict:
    call_started = time.monotonic()
    validate_intervention_plan(
        intervention_plan_path,
        public_plan_path,
        gate3_analysis_path,
        private_study_path=private_plan_path,
    )
    private_plan = json.loads(private_plan_path.read_text())
    public_plan = json.loads(public_plan_path.read_text())
    intervention_plan = json.loads(intervention_plan_path.read_text())
    plan_sha = sha256_file(intervention_plan_path)
    public_sha = sha256_file(public_plan_path)
    source_commit = _source_commit()
    if sha256_file(sae_path) != intervention_plan["source_bindings"]["sae_sha256"]:
        raise ValueError("SAE artifact hash mismatch")
    behaviors = [
        item
        for item in private_plan["harmful_behaviors"]
        if item["split"] == "discovery"
    ]
    if max_behaviors is not None:
        if max_behaviors < 1:
            raise ValueError("max_behaviors must be positive")
        behaviors = behaviors[:max_behaviors]
    if not behaviors:
        raise ValueError("no discovery behaviors")
    conditions = calibration_conditions(intervention_plan, max_rhos=max_rhos)
    planned_max = int(public_plan["decoding"]["max_new_tokens_per_turn"])
    if max_new_tokens is None:
        max_new_tokens = planned_max
    qualification = max_behaviors is not None or max_rhos is not None
    if not qualification and max_new_tokens != planned_max:
        raise ValueError("full calibration must use the frozen token ceiling")

    store = InterventionReceiptStore(output_root / "receipts")
    restricted_root = output_root / "restricted"
    restricted_root.mkdir(parents=True, exist_ok=True)
    planned = [
        stable_intervention_trial_id(
            public_plan["study_id"],
            run_id,
            behavior["behavior_id"],
            condition.condition_id,
        )
        for behavior in behaviors
        for condition in conditions
    ]
    existing: dict[str, dict] = {}
    unexpected = sorted(
        path.stem
        for path in store.trials.glob("*.json")
        if path.stem not in set(planned)
    )
    if unexpected:
        raise ValueError(f"unexpected calibration receipts: {unexpected[:5]}")
    for trial_id in planned:
        path = store.trials / f"{trial_id}.json"
        if path.exists():
            existing[trial_id] = store.load_validated(
                trial_id,
                plan_sha256=plan_sha,
                source_commit=source_commit,
                run_id=run_id,
            )
    if len(existing) == len(planned):
        zero_receipts = [
            receipt
            for receipt in existing.values()
            if receipt["condition_id"] == "zero"
        ]
        reference_norm = _reference_norm(zero_receipts)
        reference_payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "intervention_plan_sha256": plan_sha,
            "source_commit": source_commit,
            "statistic": (
                "median finite layer-50 residual-post L2 norm across all "
                "current-final-token zero-intervention decoding events"
            ),
            "reference_norm": reference_norm,
            "zero_receipts": sorted(
                (
                    {
                        "trial_id": receipt["trial_id"],
                        "receipt_sha256": sha256_file(
                            store.trials / f"{receipt['trial_id']}.json"
                        ),
                    }
                    for receipt in zero_receipts
                ),
                key=lambda item: item["trial_id"],
            ),
        }
        reference_path = output_root / "calibration-reference.json"
        if reference_path.exists():
            if json.loads(reference_path.read_text()) != reference_payload:
                raise ValueError("calibration reference provenance drift")
        else:
            write_json_atomic(reference_path, reference_payload)
        summary = {
            "schema_version": "1.0",
            "status": "complete",
            "run_id": run_id,
            "trials": len(planned),
            "trials_written_this_call": 0,
            "model_loaded_this_call": False,
            "elapsed_this_call_seconds": time.monotonic() - call_started,
            "reference_norm": reference_norm,
            "intervention_plan_sha256": plan_sha,
            "source_commit": source_commit,
        }
        write_json_atomic(output_root / "summary.json", summary)
        return summary

    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    runtime_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"dtype": runtime_dtype, "attn_implementation": "eager"}
    max_memory = None
    if torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"
        max_memory = {
            device_index: "125GiB"
            for device_index in range(torch.cuda.device_count())
        }
        kwargs["max_memory"] = max_memory
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    if torch.cuda.device_count() <= 1:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    input_device = next(model.parameters()).device
    runtime_model_revision = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    decoder = state_dict["decoder_linear.weight"].float()
    feature_id = int(intervention_plan["features"]["primary_feature_id"])
    if decoder.ndim != 2 or feature_id >= decoder.shape[1]:
        raise ValueError("SAE decoder topology mismatch")
    direction_numpy = unit_direction(decoder[:, feature_id].numpy(), expected_dimension=8192)
    direction = torch.from_numpy(direction_numpy)
    direction_hash = direction_sha256(direction_numpy)
    layer = int(intervention_plan["intervention"]["layer"])
    if layer >= len(model.model.layers):
        raise ValueError("intervention layer outside model topology")
    layer_module = model.model.layers[layer]
    eos_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    }
    eos_ids.discard(None)
    eos_ids.discard(tokenizer.unk_token_id)
    software = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    prompts = {
        behavior["behavior_id"]: _prompt_for_attack(
            tokenizer=tokenizer,
            behavior=behavior,
            private_plan=private_plan,
            public_plan=public_plan,
            public_plan_sha256=public_sha,
            generation_root=generation_root,
        )
        for behavior in behaviors
    }
    trials_written = 0
    zero_receipts = [
        receipt
        for receipt in existing.values()
        if receipt["condition_id"] == "zero"
    ]
    reference_path = output_root / "calibration-reference.json"
    for condition in conditions:
        if condition.sign != 0 and len(zero_receipts) != len(behaviors):
            raise ValueError("all zero conditions must complete before nonzero doses")
        reference_norm = _reference_norm(zero_receipts) if condition.sign != 0 else None
        if condition.sign != 0:
            zero_bindings = sorted(
                (
                    {
                        "trial_id": receipt["trial_id"],
                        "receipt_sha256": sha256_file(
                            store.trials / f"{receipt['trial_id']}.json"
                        ),
                    }
                    for receipt in zero_receipts
                ),
                key=lambda item: item["trial_id"],
            )
            reference_payload = {
                "schema_version": "1.0",
                "run_id": run_id,
                "intervention_plan_sha256": plan_sha,
                "source_commit": source_commit,
                "statistic": (
                    "median finite layer-50 residual-post L2 norm across all "
                    "current-final-token zero-intervention decoding events"
                ),
                "reference_norm": reference_norm,
                "zero_receipts": zero_bindings,
            }
            if reference_path.exists():
                if json.loads(reference_path.read_text()) != reference_payload:
                    raise ValueError("calibration reference provenance drift")
            else:
                write_json_atomic(reference_path, reference_payload)
        alpha = 0.0 if condition.rho is None else condition.rho * reference_norm
        for behavior in behaviors:
            trial_id = stable_intervention_trial_id(
                public_plan["study_id"],
                run_id,
                behavior["behavior_id"],
                condition.condition_id,
            )
            if trial_id in existing:
                continue
            prompt_ids, prompt_text, parent_sha = prompts[behavior["behavior_id"]]
            tensor = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            hook = ResidualPostIntervention(
                torch,
                direction,
                sign=condition.sign,
                alpha=alpha,
            )
            handle = layer_module.register_forward_hook(hook)
            before = time.monotonic()
            try:
                with torch.inference_mode():
                    output = model.generate(
                        tensor,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                handle.remove()
            elapsed = time.monotonic() - before
            generated = output[0, tensor.shape[1] :].tolist()
            if len(hook.steps) != len(generated):
                raise ValueError("generation/intervention step topology mismatch")
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            restricted_path = restricted_root / f"{trial_id}.json"
            restricted_sha = _save_restricted(
                restricted_path,
                {
                    "behavior_id": behavior["behavior_id"],
                    "condition_id": condition.condition_id,
                    "generated_text": generated_text,
                    "generated_token_ids": generated,
                },
            )
            payload = {
                "schema_version": "1.0",
                "study_id": public_plan["study_id"],
                "intervention_plan_sha256": plan_sha,
                "public_plan_sha256": public_sha,
                "source_commit": source_commit,
                "run_id": run_id,
                "phase": "discovery_calibration",
                "trial_id": trial_id,
                "attempt": 1,
                "behavior_id": behavior["behavior_id"],
                "category": behavior["category"],
                "split": "discovery",
                "arm": "full",
                "turn": 2,
                "condition_id": condition.condition_id,
                "direction_kind": "zero" if condition.sign == 0 else "primary",
                "feature_id": None if condition.sign == 0 else feature_id,
                "isotropic_seed": None,
                "direction_sha256": None if condition.sign == 0 else direction_hash,
                "requested_sign": condition.sign,
                "requested_alpha": alpha,
                "calibration_rho": condition.rho,
                "calibration_reference_norm": reference_norm,
                "layer": layer,
                "hook": "residual_post",
                "positions": "current_final_token_on_every_decoding_step",
                "prompt_sha256": sha256_text(prompt_text),
                "prompt_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(prompt_ids)
                ),
                "parent_generation_receipt_sha256": parent_sha,
                "generated_token_ids": generated,
                "generated_text_sha256": sha256_text(generated_text),
                "restricted_text_path": str(restricted_path),
                "restricted_artifact_sha256": restricted_sha,
                "finish_reason": (
                    "eos" if generated and generated[-1] in eos_ids else "length"
                ),
                "generated_token_count": len(generated),
                "refusal_detected": bool(REFUSAL_PATTERN.search(generated_text)),
                "truncated": not bool(generated and generated[-1] in eos_ids),
                "evaluator_score": None,
                "evaluator_parse_ok": False,
                "intervention_steps": hook.steps,
                "model_revision": runtime_model_revision,
                "tokenizer_revision": runtime_model_revision,
                "sae_sha256": sha256_file(sae_path),
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": _peak_memory(torch),
                "software": software,
                "runtime": {
                    "device_count": torch.cuda.device_count(),
                    "device_map": str(getattr(model, "hf_device_map", None)),
                    "max_memory": max_memory,
                    "dtype": str(runtime_dtype),
                    "attention": "eager",
                    "qualification": qualification,
                },
                "error": None,
            }
            store.write(payload)
            existing[trial_id] = payload
            trials_written += 1
            if condition.sign == 0:
                zero_receipts.append(payload)
            print(
                f"completed {behavior['behavior_id']} {condition.condition_id} "
                f"tokens={len(generated)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if set(existing) != set(planned):
        raise ValueError("calibration receipt topology mismatch")
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "run_id": run_id,
        "trials": len(planned),
        "trials_written_this_call": trials_written,
        "model_loaded_this_call": True,
        "elapsed_this_call_seconds": time.monotonic() - call_started,
        "reference_norm": _reference_norm(zero_receipts),
        "intervention_plan_sha256": plan_sha,
        "source_commit": source_commit,
    }
    write_json_atomic(output_root / "summary.json", summary)
    return summary
