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

from .behavior import REFUSAL_PATTERN, _as_token_ids, _peak_memory
from .followup_design import PLACEMENTS
from .followup_plan import validate_followup_plan
from .followup_runner import build_placement_render_pair
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .models import FollowupTrialReceipt
from .scaffolds import DIVIDER, render_arms

HARMFUL_PARTITIONS = {"discovery", "calibration", "confirmatory", "adaptive_stress"}
UTILITY_PARTITIONS = {"utility_calibration", "utility_confirmatory"}
STRUCTURED_ARMS = ("inert_length", "structural_sham", "full")


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def stable_followup_trial_id(
    study_id: str,
    behavior_id: str,
    arm: str,
    placement: str | None,
    partition: str,
) -> str:
    value = f"{study_id}|{partition}|{behavior_id}|{arm}|{placement or 'shared'}|0"
    return sha256_text(value)[:24]


@dataclass(frozen=True)
class PlannedObservation:
    behavior_id: str
    category: str
    partition: str
    request: str
    request_sha256: str
    arm: str
    placement: str | None
    shared_reference: bool
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    render_validation: dict[str, Any] | None

    @property
    def stratum(self) -> str:
        return self.arm if self.placement is None else f"{self.arm}:{self.placement}"

    def trial_id(self, study_id: str) -> str:
        return stable_followup_trial_id(
            study_id,
            self.behavior_id,
            self.arm,
            self.placement,
            self.partition,
        )


def _split_scaffold(rendered_text: str, request: str) -> str:
    if rendered_text.count(request) != 1:
        raise ValueError("request is not uniquely recoverable from pinned scaffold render")
    scaffold = rendered_text.replace(request, "", 1)
    if not scaffold:
        raise ValueError("derived scaffold block is empty")
    return scaffold


def _chat_prompt(tokenizer, text: str) -> tuple[str, tuple[int, ...]]:
    messages = [{"role": "user", "content": text}]
    prompt_ids = tuple(
        _as_token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, prompt_ids


def build_partition_observations(
    *,
    tokenizer,
    private_plan: dict[str, Any],
    public_plan: dict[str, Any],
    partition: str,
    tokenizer_revision: str,
) -> tuple[list[PlannedObservation], list[dict[str, Any]]]:
    if partition not in HARMFUL_PARTITIONS | UTILITY_PARTITIONS:
        raise ValueError(f"unknown follow-up partition: {partition}")
    rows = private_plan["partitions"][partition]
    decoding = public_plan["replication"]["decoding"]
    observations: list[PlannedObservation] = []
    validations: list[dict[str, Any]] = []
    for behavior in rows:
        request = behavior["request"]
        if sha256_text(request) != behavior["request_sha256"]:
            raise ValueError(f"{behavior['behavior_id']}: request hash drift")
        if partition in HARMFUL_PARTITIONS:
            prompt_text, prompt_ids = _chat_prompt(tokenizer, request)
            observations.append(
                PlannedObservation(
                    behavior_id=behavior["behavior_id"],
                    category=behavior["category"],
                    partition=partition,
                    request=request,
                    request_sha256=behavior["request_sha256"],
                    arm="base",
                    placement=None,
                    shared_reference=True,
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_ids,
                    render_validation=None,
                )
            )
            rendered = render_arms(request, tokenizer)
            structured_lengths = {
                len(rendered[name].token_ids) for name in STRUCTURED_ARMS
            }
            if len(structured_lengths) != 1:
                raise ValueError(
                    f"{behavior['behavior_id']}: 8B structured-arm length drift"
                )
            arm_names = STRUCTURED_ARMS
        else:
            prompt_text, prompt_ids = _chat_prompt(tokenizer, request)
            observations.append(
                PlannedObservation(
                    behavior_id=behavior["behavior_id"],
                    category=behavior["category"],
                    partition=partition,
                    request=request,
                    request_sha256=behavior["request_sha256"],
                    arm="ordinary_benign",
                    placement=None,
                    shared_reference=True,
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_ids,
                    render_validation=None,
                )
            )
            rendered = render_arms(request, tokenizer)
            arm_names = ("full",)
        for rendered_arm in arm_names:
            public_validation, private_renders = build_placement_render_pair(
                tokenizer=tokenizer,
                request=request,
                scaffold=_split_scaffold(rendered[rendered_arm].text, request),
                tokenizer_revision=tokenizer_revision,
                context_ceiling=int(decoding["context_ceiling"]),
                generation_budget=int(decoding["max_new_tokens"]),
            )
            public_arm = (
                "structured_benign"
                if partition in UTILITY_PARTITIONS
                else rendered_arm
            )
            validation_row = {
                "behavior_id": behavior["behavior_id"],
                "arm": public_arm,
                **public_validation,
            }
            validations.append(validation_row)
            for placement in PLACEMENTS:
                private_render = private_renders[placement]
                observations.append(
                    PlannedObservation(
                        behavior_id=behavior["behavior_id"],
                        category=behavior["category"],
                        partition=partition,
                        request=request,
                        request_sha256=behavior["request_sha256"],
                        arm=public_arm,
                        placement=placement,
                        shared_reference=False,
                        prompt_text=private_render["prompt_text"],
                        prompt_token_ids=tuple(private_render["prompt_token_ids"]),
                        render_validation=validation_row,
                    )
                )
    trial_ids = [item.trial_id(public_plan["study_id"]) for item in observations]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("follow-up observation IDs are not unique")
    expected_per_behavior = 7 if partition in HARMFUL_PARTITIONS else 3
    if len(observations) != len(rows) * expected_per_behavior:
        raise ValueError("follow-up observation topology drift")
    return observations, validations


class FollowupReceiptStore:
    def __init__(self, root: Path):
        self.root = root
        self.trials = root / "trials"
        self.attempts = root / "attempts.jsonl"
        self.trials.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> str:
        receipt = FollowupTrialReceipt.model_validate(payload)
        _validate_followup_trial_topology(receipt)
        encoded = canonical_json_bytes(receipt.model_dump(mode="json"))
        path = self.trials / f"{receipt.trial_id}.json"
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError(f"{receipt.trial_id}: refusing receipt overwrite")
            return sha256_bytes(encoded)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        with self.attempts.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return sha256_bytes(encoded)

    def load_validated(
        self,
        *,
        observation: PlannedObservation,
        study_id: str,
        plan_sha256: str,
        private_plan_sha256: str,
        source_commit: str,
        run_id: str,
    ) -> dict[str, Any] | None:
        trial_id = observation.trial_id(study_id)
        path = self.trials / f"{trial_id}.json"
        if not path.exists():
            return None
        receipt = FollowupTrialReceipt.model_validate_json(path.read_text())
        _validate_followup_trial_topology(receipt)
        expected = {
            "trial_id": trial_id,
            "plan_sha256": plan_sha256,
            "private_plan_sha256": private_plan_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "partition": observation.partition,
            "behavior_id": observation.behavior_id,
            "arm": observation.arm,
            "placement": observation.placement,
            "stratum": observation.stratum,
            "shared_reference": observation.shared_reference,
            "request_sha256": observation.request_sha256,
            "prompt_sha256": sha256_text(observation.prompt_text),
            "prompt_token_ids_sha256": sha256_bytes(
                canonical_json_bytes(list(observation.prompt_token_ids))
            ),
        }
        for key, value in expected.items():
            if getattr(receipt, key) != value:
                raise ValueError(f"{trial_id}: receipt provenance drift for {key}")
        restricted = Path(receipt.restricted_artifact_path)
        states = Path(receipt.state_bundle_path)
        if sha256_file(restricted) != receipt.restricted_artifact_sha256:
            raise ValueError(f"{trial_id}: restricted artifact hash drift")
        if sha256_file(states) != receipt.state_bundle_sha256:
            raise ValueError(f"{trial_id}: state bundle hash drift")
        raw = json.loads(restricted.read_text())
        if (
            raw["trial_id"] != trial_id
            or raw["behavior_id"] != observation.behavior_id
            or raw["partition"] != observation.partition
            or raw["arm"] != observation.arm
            or raw["placement"] != observation.placement
            or sha256_text(raw["request"]) != receipt.request_sha256
            or sha256_text(raw["prompt_text"]) != receipt.prompt_sha256
            or sha256_bytes(canonical_json_bytes(raw["prompt_token_ids"]))
            != receipt.prompt_token_ids_sha256
            or sha256_text(raw["generated_text"]) != receipt.generated_text_sha256
            or sha256_bytes(canonical_json_bytes(raw["generated_token_ids"]))
            != receipt.generated_token_ids_sha256
        ):
            raise ValueError(f"{trial_id}: restricted content drift")
        import torch

        state_payload = torch.load(states, map_location="cpu", weights_only=True)
        state_provenance = state_payload.get("provenance", {})
        expected_state_provenance = {
            "trial_id": trial_id,
            "public_plan_sha256": plan_sha256,
            "private_plan_sha256": private_plan_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "prompt_token_ids_sha256": receipt.prompt_token_ids_sha256,
            "model_revision": receipt.model_revision,
            "lens_sha256": receipt.lens_sha256,
            "sae_sha256": receipt.sae_sha256,
        }
        if state_provenance != expected_state_provenance:
            raise ValueError(f"{trial_id}: state bundle provenance drift")
        if sorted(state_payload.get("states", {})) != receipt.capture_layers:
            raise ValueError(f"{trial_id}: state bundle layer topology drift")
        for layer, state in state_payload["states"].items():
            if (
                tuple(state.shape) != (receipt.state_shape[1],)
                or state.dtype != torch.bfloat16
            ):
                raise ValueError(f"{trial_id}: state tensor drift at layer {layer}")
        return receipt.model_dump(mode="json")


def _validate_followup_trial_topology(receipt: FollowupTrialReceipt) -> None:
    shared_arm = receipt.arm in {"base", "ordinary_benign"}
    if receipt.shared_reference != shared_arm:
        raise ValueError("follow-up shared-reference topology drift")
    if shared_arm:
        if receipt.placement is not None or receipt.render_validation_sha256 is not None:
            raise ValueError("shared reference cannot carry placement metadata")
    elif receipt.placement not in PLACEMENTS or receipt.render_validation_sha256 is None:
        raise ValueError("ordering-specific receipt lacks placement metadata")
    expected_stratum = (
        receipt.arm
        if receipt.placement is None
        else f"{receipt.arm}:{receipt.placement}"
    )
    if receipt.stratum != expected_stratum:
        raise ValueError("follow-up stratum label drift")
    if (
        not receipt.capture_layers
        or receipt.capture_layers != sorted(set(receipt.capture_layers))
        or 19 not in receipt.capture_layers
        or receipt.state_shape != [len(receipt.capture_layers), 4096]
    ):
        raise ValueError("follow-up capture topology drift")


def _save_json_private(path: Path, payload: dict[str, Any]) -> str:
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


def _capture_states(torch, model, prompt_ids: tuple[int, ...], layers: list[int]):
    captured: dict[int, Any] = {}
    handles = []
    for layer in layers:
        def hook(_module, _inputs, output, *, layer_index=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[0, -1].detach().to("cpu", dtype=torch.bfloat16)

        handles.append(model.model.layers[layer].register_forward_hook(hook))
    tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    with torch.inference_mode():
        model(input_ids=tensor, use_cache=False)
    for handle in handles:
        handle.remove()
    if sorted(captured) != layers:
        raise ValueError("follow-up state capture incomplete")
    return captured


def _save_states_private(
    torch,
    path: Path,
    *,
    states: dict[int, Any],
    provenance: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"provenance": provenance, "states": states}, temporary)
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_file(path)


def _write_preflight(
    output_root: Path,
    *,
    public_plan_sha256: str,
    private_plan_sha256: str,
    source_commit: str,
    run_id: str,
    partition: str,
    observations: list[PlannedObservation],
    validations: list[dict[str, Any]],
) -> str:
    payload = {
        "schema_version": "1.0",
        "status": "passed",
        "public_plan_sha256": public_plan_sha256,
        "private_plan_sha256": private_plan_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "partition": partition,
        "observations": len(observations),
        "shared_references": sum(item.shared_reference for item in observations),
        "ordering_specific": sum(not item.shared_reference for item in observations),
        "validations": validations,
    }
    path = output_root / "render-preflight.public.json"
    encoded = canonical_json_bytes(payload)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("render preflight resume drift")
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    return sha256_bytes(encoded)


def run_followup_generation(
    *,
    private_plan_path: Path,
    public_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    partition: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    private_plan = json.loads(private_plan_path.read_text())
    public_plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(public_plan)
    public_sha = sha256_file(public_plan_path)
    private_sha = sha256_file(private_plan_path)
    if private_plan["source_bindings"]["public_plan_sha256"] != public_sha:
        raise ValueError("follow-up private/public plan hash drift")
    source_commit = _source_commit()
    expected_revision = public_plan["artifacts"]["llama31_model"]["revision"]
    if expected_revision not in str(Path(model_path).resolve()):
        raise ValueError("model path is not the frozen snapshot")
    lens_sha = sha256_file(lens_path)
    sae_sha = sha256_file(sae_path)
    if lens_sha != public_plan["artifacts"]["llama31_lens"]["sha256"]:
        raise ValueError("follow-up J-lens hash drift")
    if sae_sha != public_plan["artifacts"]["llama31_sae"]["sha256"]:
        raise ValueError("follow-up SAE hash drift")

    import torch
    import transformers
    import jlens

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or expected_revision
    observations, validations = build_partition_observations(
        tokenizer=tokenizer,
        private_plan=private_plan,
        public_plan=public_plan,
        partition=partition,
        tokenizer_revision=tokenizer_revision,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_sha = _write_preflight(
        output_root,
        public_plan_sha256=public_sha,
        private_plan_sha256=private_sha,
        source_commit=source_commit,
        run_id=run_id,
        partition=partition,
        observations=observations,
        validations=validations,
    )
    store = FollowupReceiptStore(output_root / "receipts")
    completed = {
        item.trial_id(public_plan["study_id"]): store.load_validated(
            observation=item,
            study_id=public_plan["study_id"],
            plan_sha256=public_sha,
            private_plan_sha256=private_sha,
            source_commit=source_commit,
            run_id=run_id,
        )
        for item in observations
    }
    if all(value is not None for value in completed.values()):
        summary = {
            "schema_version": "1.0",
            "status": "complete",
            "partition": partition,
            "run_id": run_id,
            "source_commit": source_commit,
            "public_plan_sha256": public_sha,
            "private_plan_sha256": private_sha,
            "render_preflight_sha256": preflight_sha,
            "trials": len(observations),
            "trials_written_this_call": 0,
            "model_loaded_this_call": False,
        }
        _save_json_private(output_root / "summary.json", summary)
        return summary

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if torch.cuda.device_count() != 1:
        raise ValueError("follow-up generation requires exactly one visible GPU")
    model = model.to("cuda")
    if (
        int(model.config.hidden_size) != 4096
        or int(model.config.num_hidden_layers) != 32
        or int(model.config.vocab_size) != 128256
    ):
        raise ValueError("Llama 3.1 8B topology drift")
    model_revision = getattr(model.config, "_commit_hash", None) or expected_revision
    if model_revision != expected_revision or tokenizer_revision != expected_revision:
        raise ValueError("follow-up model/tokenizer revision drift")
    lens = jlens.JacobianLens.load(lens_path)
    capture_layers = sorted({int(value) for value in lens.source_layers} | {19})
    if not capture_layers or capture_layers[-1] >= 32:
        raise ValueError("follow-up capture layer topology drift")
    decoding = public_plan["replication"]["decoding"]
    max_new_tokens = int(decoding["max_new_tokens"])
    eos_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    }
    eos_ids.discard(None)
    eos_ids.discard(tokenizer.unk_token_id)
    written = 0
    for index, observation in enumerate(observations, start=1):
        trial_id = observation.trial_id(public_plan["study_id"])
        if completed[trial_id] is not None:
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        capture_started = time.monotonic()
        states = _capture_states(
            torch,
            model,
            observation.prompt_token_ids,
            capture_layers,
        )
        capture_elapsed = time.monotonic() - capture_started
        tensor = torch.tensor(
            [observation.prompt_token_ids],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        generation_started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generation_elapsed = time.monotonic() - generation_started
        generated_ids = output[0, tensor.shape[1] :].tolist()
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        restricted_path = output_root / "restricted" / f"{trial_id}.json"
        restricted_sha = _save_json_private(
            restricted_path,
            {
                "trial_id": trial_id,
                "behavior_id": observation.behavior_id,
                "partition": partition,
                "arm": observation.arm,
                "placement": observation.placement,
                "request": observation.request,
                "prompt_text": observation.prompt_text,
                "prompt_token_ids": list(observation.prompt_token_ids),
                "generated_token_ids": generated_ids,
                "generated_text": generated_text,
            },
        )
        state_path = output_root / "states" / f"{trial_id}.pt"
        state_sha = _save_states_private(
            torch,
            state_path,
            states=states,
            provenance={
                "trial_id": trial_id,
                "public_plan_sha256": public_sha,
                "private_plan_sha256": private_sha,
                "source_commit": source_commit,
                "run_id": run_id,
                "prompt_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(list(observation.prompt_token_ids))
                ),
                "model_revision": model_revision,
                "lens_sha256": lens_sha,
                "sae_sha256": sae_sha,
            },
        )
        divider_present = DIVIDER in generated_text
        post_divider = (
            generated_text.split(DIVIDER, 1)[1].strip()
            if divider_present
            else ""
        )
        payload = {
            "schema_version": "1.0",
            "study_id": public_plan["study_id"],
            "plan_sha256": public_sha,
            "private_plan_sha256": private_sha,
            "source_commit": source_commit,
            "run_id": run_id,
            "trial_id": trial_id,
            "partition": partition,
            "behavior_id": observation.behavior_id,
            "category": observation.category,
            "arm": observation.arm,
            "placement": observation.placement,
            "stratum": observation.stratum,
            "shared_reference": observation.shared_reference,
            "request_sha256": observation.request_sha256,
            "prompt_sha256": sha256_text(observation.prompt_text),
            "prompt_token_ids_sha256": sha256_bytes(
                canonical_json_bytes(list(observation.prompt_token_ids))
            ),
            "render_validation_sha256": (
                sha256_bytes(canonical_json_bytes(observation.render_validation))
                if observation.render_validation is not None
                else None
            ),
            "generated_text_sha256": sha256_text(generated_text),
            "generated_token_ids_sha256": sha256_bytes(
                canonical_json_bytes(generated_ids)
            ),
            "generated_token_count": len(generated_ids),
            "finish_reason": (
                "eos" if generated_ids and generated_ids[-1] in eos_ids else "length"
            ),
            "truncated": not bool(generated_ids and generated_ids[-1] in eos_ids),
            "refusal_detected": bool(REFUSAL_PATTERN.search(generated_text)),
            "divider_detected": divider_present,
            "post_divider_present": bool(post_divider),
            "restricted_artifact_path": str(restricted_path),
            "restricted_artifact_sha256": restricted_sha,
            "state_bundle_path": str(state_path),
            "state_bundle_sha256": state_sha,
            "capture_layers": capture_layers,
            "state_shape": [len(capture_layers), int(model.config.hidden_size)],
            "state_dtype": "torch.bfloat16",
            "generation_elapsed_seconds": generation_elapsed,
            "capture_elapsed_seconds": capture_elapsed,
            "peak_memory_bytes": _peak_memory(torch),
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "lens_sha256": lens_sha,
            "sae_sha256": sae_sha,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "jlens": getattr(jlens, "__version__", "unknown"),
                "cuda": torch.version.cuda,
            },
        }
        store.write(payload)
        written += 1
        print(
            f"completed follow-up observation {index}/{len(observations)} "
            f"trial={trial_id}",
            flush=True,
        )
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "partition": partition,
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_sha,
        "private_plan_sha256": private_sha,
        "render_preflight_sha256": preflight_sha,
        "trials": len(observations),
        "trials_written_this_call": written,
        "model_loaded_this_call": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    _save_json_private(output_root / "summary.json", summary)
    return summary
