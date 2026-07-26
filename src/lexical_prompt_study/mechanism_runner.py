from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import MODEL_REVISION
from .behavior import _as_token_ids
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from .mechanisms import (
    PRIMARY_POSITION,
    RANDOM_TRANSPORT_SEED,
    SAE_HOOK_LAYER,
    SECONDARY_GENERATED_POSITIONS,
    deterministic_transport_seed,
    sae_feature_diagnostics,
    select_sae_candidates,
    validate_mechanism_receipt,
)
from .models import TrialReceipt
from .receipts import stable_trial_id

ARMS = ("base", "full", "structural_sham", "inert_length")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    split: str
    behavior_id: str
    category: str
    arm: str
    turn: int
    position: str
    position_token_index: int | None
    position_available: bool
    missing_position_reason: str | None
    prompt_token_ids_sha256: str
    prefix_token_ids_sha256: str
    prefix_token_ids: tuple[int, ...]

    def public_metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("prefix_token_ids")
        return value


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _raw_generation(
    *,
    generation_root: Path,
    public_plan_sha256: str,
    study_id: str,
    behavior_id: str,
    arm: str,
    turn: int,
    split: str,
) -> tuple[TrialReceipt, dict[str, Any]]:
    trial_id = stable_trial_id(study_id, behavior_id, arm, turn, 0)
    receipt_path = generation_root / "receipts" / "trials" / f"{trial_id}.json"
    raw_path = generation_root / "restricted" / f"{trial_id}.json"
    receipt = TrialReceipt.model_validate_json(receipt_path.read_text())
    if receipt.trial_id != trial_id:
        raise ValueError(f"{trial_id}: trial ID drift")
    if receipt.plan_sha256 != public_plan_sha256:
        raise ValueError(f"{trial_id}: public plan hash drift")
    if (
        receipt.study_id != study_id
        or receipt.behavior_id != behavior_id
        or receipt.arm != arm
        or receipt.turn != turn
        or receipt.split != split
    ):
        raise ValueError(f"{trial_id}: generation topology drift")
    if receipt.restricted_text_path != str(raw_path):
        raise ValueError(f"{trial_id}: restricted path drift")
    if receipt.restricted_artifact_sha256 != sha256_file(raw_path):
        raise ValueError(f"{trial_id}: restricted artifact hash mismatch")
    raw = json.loads(raw_path.read_text())
    if (
        raw.get("behavior_id") != behavior_id
        or raw.get("arm") != arm
        or raw.get("turn") != turn
        or raw.get("generated_token_ids") != receipt.generated_token_ids
    ):
        raise ValueError(f"{trial_id}: restricted generation content drift")
    return receipt, raw


def build_observations(
    *,
    tokenizer,
    private_plan_path: Path,
    public_plan_path: Path,
    generation_root: Path,
    split: str,
    max_behaviors: int | None = None,
) -> list[Observation]:
    private_plan = json.loads(private_plan_path.read_text())
    public_plan = json.loads(public_plan_path.read_text())
    public_sha = sha256_file(public_plan_path)
    if private_plan["public_plan_sha256"] != public_sha:
        raise ValueError("private/public plan hash mismatch")
    behaviors = [
        behavior for behavior in private_plan["harmful_behaviors"] if behavior["split"] == split
    ]
    if max_behaviors is not None:
        if max_behaviors < 1:
            raise ValueError("max_behaviors must be positive")
        behaviors = behaviors[:max_behaviors]
    if not behaviors:
        raise ValueError(f"no harmful behaviors for split={split}")
    observations: list[Observation] = []
    for behavior in behaviors:
        for arm in ARMS:
            first_receipt, first_raw = _raw_generation(
                generation_root=generation_root,
                public_plan_sha256=public_sha,
                study_id=public_plan["study_id"],
                behavior_id=behavior["behavior_id"],
                arm=arm,
                turn=1,
                split=split,
            )
            second_receipt, second_raw = _raw_generation(
                generation_root=generation_root,
                public_plan_sha256=public_sha,
                study_id=public_plan["study_id"],
                behavior_id=behavior["behavior_id"],
                arm=arm,
                turn=2,
                split=split,
            )
            messages = [
                {"role": "user", "content": behavior["rendered_arms"][arm]["text"]},
                {"role": "assistant", "content": first_raw["generated_text"]},
                {"role": "user", "content": private_plan["followup"]},
            ]
            prompt_ids = tuple(
                _as_token_ids(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            )
            prompt_sha = sha256_bytes(canonical_json_bytes(list(prompt_ids)))
            if prompt_sha != second_receipt.prompt_token_ids_sha256:
                raise ValueError(
                    f"{behavior['behavior_id']} {arm}: reconstructed turn-2 prompt drift"
                )
            if first_receipt.generated_text_sha256 != sha256_bytes(
                first_raw["generated_text"].encode()
            ):
                raise ValueError(
                    f"{behavior['behavior_id']} {arm}: turn-1 restricted text hash mismatch"
                )
            generated_ids = tuple(int(item) for item in second_raw["generated_token_ids"])
            position_specs = [(PRIMARY_POSITION, None)] + [
                ("generated", index) for index in SECONDARY_GENERATED_POSITIONS
            ]
            for position, token_index in position_specs:
                available = token_index is None or len(generated_ids) > token_index
                prefix = (
                    prompt_ids
                    if token_index is None
                    else prompt_ids + generated_ids[: token_index + 1]
                    if available
                    else ()
                )
                missing_reason = (
                    None
                    if available
                    else f"generation_length_{len(generated_ids)}_does_not_reach_index_{token_index}"
                )
                metadata = {
                    "split": split,
                    "behavior_id": behavior["behavior_id"],
                    "arm": arm,
                    "turn": 2,
                    "position": position,
                    "position_token_index": token_index,
                    "position_available": available,
                    "prompt_token_ids_sha256": prompt_sha,
                    "prefix_token_ids_sha256": (
                        sha256_bytes(canonical_json_bytes(list(prefix))) if available else ""
                    ),
                }
                observation_id = sha256_bytes(canonical_json_bytes(metadata))[:24]
                observations.append(
                    Observation(
                        observation_id=observation_id,
                        split=split,
                        behavior_id=behavior["behavior_id"],
                        category=behavior["category"],
                        arm=arm,
                        turn=2,
                        position=position,
                        position_token_index=token_index,
                        position_available=available,
                        missing_position_reason=missing_reason,
                        prompt_token_ids_sha256=prompt_sha,
                        prefix_token_ids_sha256=metadata["prefix_token_ids_sha256"],
                        prefix_token_ids=prefix,
                    )
                )
    ids = [item.observation_id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("observation IDs are not unique")
    return observations


def _observation_manifest(observations: list[Observation]) -> dict[str, Any]:
    rows = [item.public_metadata() for item in observations]
    return {
        "schema_version": "1.0",
        "secondary_position_semantics": (
            "index k includes generated token IDs 0..k and reads the residual at token k"
        ),
        "observations": rows,
        "observations_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _capture_states(
    *,
    torch,
    hf_model,
    observations: list[Observation],
    capture_layers: list[int],
    output_root: Path,
    provenance: dict[str, Any],
) -> tuple[dict[int, Any], str]:
    manifest = _observation_manifest(observations)
    manifest["provenance"] = provenance
    manifest_path = output_root / "observation-manifest.json"
    state_path = output_root / "captured-states.pt"
    if manifest_path.exists() or state_path.exists():
        if not manifest_path.exists() or not state_path.exists():
            raise ValueError("partial capture checkpoint")
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("capture observation manifest drift")
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if payload["observations_sha256"] != manifest["observations_sha256"]:
            raise ValueError("captured state observation hash drift")
        if payload.get("provenance") != provenance:
            raise ValueError("captured state provenance drift")
        if sorted(payload["states"]) != capture_layers:
            raise ValueError("captured state layer topology drift")
        return payload["states"], sha256_file(state_path)

    output_root.mkdir(parents=True, exist_ok=True)
    layer_modules = hf_model.model.layers
    if max(capture_layers) >= len(layer_modules):
        raise ValueError("capture layer outside model topology")
    captured: dict[int, list[Any]] = {layer: [] for layer in capture_layers}
    input_device = next(hf_model.parameters()).device
    for completed, observation in enumerate(observations, start=1):
        if not observation.position_available:
            continue
        row: dict[int, Any] = {}
        handles = []
        for layer in capture_layers:
            def hook(_module, _inputs, output, *, layer_index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                row[layer_index] = hidden[0, -1].detach().to("cpu", dtype=torch.bfloat16)

            handles.append(layer_modules[layer].register_forward_hook(hook))
        ids = torch.tensor([observation.prefix_token_ids], dtype=torch.long, device=input_device)
        with torch.inference_mode():
            hf_model(input_ids=ids, use_cache=False)
        for handle in handles:
            handle.remove()
        if sorted(row) != capture_layers:
            raise ValueError(f"{observation.observation_id}: incomplete activation capture")
        for layer in capture_layers:
            captured[layer].append(row[layer])
        if completed % 10 == 0:
            print(f"captured observation {completed}/{len(observations)}", flush=True)
    states = {layer: torch.stack(values) for layer, values in captured.items()}
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    torch.save(
        {
            "observations_sha256": manifest["observations_sha256"],
            "provenance": provenance,
            "states": states,
        },
        temporary,
    )
    temporary.replace(state_path)
    write_json_atomic(manifest_path, manifest)
    return states, sha256_file(state_path)


def _torch_margin_batch(
    torch,
    vectors,
    mean_embedding,
    second_moment,
    probe_embeddings,
    norm_epsilon,
):
    normalized = vectors * torch.rsqrt(
        vectors.square().mean(dim=1, keepdim=True) + norm_epsilon
    )
    means = normalized @ mean_embedding
    second = ((normalized @ second_moment) * normalized).sum(dim=1)
    variance = torch.clamp(second - means.square(), min=1e-12)
    std = variance.sqrt()
    probe_logits = normalized @ probe_embeddings.T
    probe_z = (probe_logits - means[:, None]) / std[:, None]
    return means, std, probe_z


def _validate_layer_checkpoint(
    payload: dict[str, Any],
    *,
    layer: int,
    public_plan_sha256: str,
    source_commit: str,
    run_id: str,
    capture_sha256: str,
    observations: list[Observation],
) -> None:
    expected_header = {
        "layer": layer,
        "public_plan_sha256": public_plan_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "capture_sha256": capture_sha256,
    }
    for key, value in expected_header.items():
        if payload.get(key) != value:
            raise ValueError(f"layer {layer}: resume provenance drift for {key}")
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(observations) * 3:
        raise ValueError(f"layer {layer}: resume receipt count mismatch")
    expected = {
        (item.observation_id, transport)
        for item in observations
        for transport in ("identity", "jacobian_lens", "random_gaussian")
    }
    realized = set()
    for receipt in receipts:
        validate_mechanism_receipt(receipt)
        if receipt["layer"] != layer:
            raise ValueError(f"layer {layer}: receipt layer drift")
        realized.add((receipt["observation_id"], receipt["transport"]))
    if realized != expected or len(realized) != len(receipts):
        raise ValueError(f"layer {layer}: resume receipt topology mismatch")


def _run_transports(
    *,
    torch,
    hf_model,
    lens,
    observations: list[Observation],
    states: dict[int, Any],
    output_root: Path,
    public_plan: dict[str, Any],
    public_plan_sha256: str,
    run_id: str,
    source_commit: str,
    lens_sha256: str,
    sae_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    capture_sha256: str,
) -> None:
    available = [item for item in observations if item.position_available]
    available_index = {
        item.observation_id: index for index, item in enumerate(available)
    }
    refusal_ids = [item["token_id"] for item in public_plan["probes"]["primary_refusal"]]
    compliance_ids = [
        item["token_id"] for item in public_plan["probes"]["primary_compliance"]
    ]
    probe_ids = refusal_ids + compliance_ids
    device = hf_model.lm_head.weight.device
    output_weight = hf_model.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    norm_weight = hf_model.model.norm.weight.detach().to(device=device, dtype=torch.float32)
    norm_epsilon = float(hf_model.model.norm.variance_epsilon)
    effective = output_weight * norm_weight[None, :]
    mean_embedding = effective.mean(dim=0)
    second_moment = effective.T @ effective / effective.shape[0]
    probe_embeddings = effective[probe_ids]
    del effective, output_weight

    # Runtime equivalence gate: the moment shortcut must reproduce complete-vocabulary logits.
    fixture = states[int(lens.source_layers[-1])][0].to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        analytic_normalized = fixture * torch.rsqrt(
            fixture.square().mean() + norm_epsilon
        )
        analytic_normalized = analytic_normalized * norm_weight
        full_logits = hf_model.lm_head.weight.float() @ analytic_normalized
        full_mean = full_logits.mean()
        full_std = full_logits.std(unbiased=False)
        full_probe_z = (full_logits[probe_ids] - full_mean) / full_std
        module_logits = hf_model.lm_head(
            hf_model.model.norm(
                fixture.to(dtype=hf_model.model.norm.weight.dtype)
            )
        ).float()
        module_probe_z = (
            module_logits[probe_ids] - module_logits.mean()
        ) / module_logits.std(unbiased=False)
        moment_mean, moment_std, moment_probe_z = _torch_margin_batch(
            torch,
            fixture[None, :],
            mean_embedding,
            second_moment,
            probe_embeddings,
            norm_epsilon,
        )
    equivalence = {
        "mean_abs_error": float(abs(full_mean - moment_mean[0]).item()),
        "std_abs_error": float(abs(full_std - moment_std[0]).item()),
        "probe_max_abs_error": float(
            (full_probe_z - moment_probe_z[0]).abs().max().item()
        ),
        "module_probe_max_abs_error": float(
            (module_probe_z - moment_probe_z[0]).abs().max().item()
        ),
        "analytic_tolerance": 2e-3,
        "module_tolerance": 2e-2,
    }
    if (
        max(
        equivalence["mean_abs_error"],
        equivalence["std_abs_error"],
        equivalence["probe_max_abs_error"],
        )
        > equivalence["analytic_tolerance"]
        or equivalence["module_probe_max_abs_error"] > equivalence["module_tolerance"]
    ):
        raise ValueError(f"vocabulary moment equivalence gate failed: {equivalence}")
    write_json_atomic(output_root / "vocabulary-moment-validation.json", equivalence)

    layer_root = output_root / "layers"
    layer_root.mkdir(parents=True, exist_ok=True)
    runtime = {
        "capture_sha256": capture_sha256,
        "moment_equivalence": equivalence,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    n_refusal = len(refusal_ids)
    for layer in lens.source_layers:
        final_path = layer_root / f"layer-{int(layer):03d}.json"
        if final_path.exists():
            existing = json.loads(final_path.read_text())
            _validate_layer_checkpoint(
                existing,
                layer=int(layer),
                public_plan_sha256=public_plan_sha256,
                source_commit=source_commit,
                run_id=run_id,
                capture_sha256=capture_sha256,
                observations=observations,
            )
            continue
        hidden = states[int(layer)].to(device=device, dtype=torch.float32)
        jacobian = lens.jacobians[int(layer)].to(device=device, dtype=torch.float32)
        if jacobian.shape != (hidden.shape[1], hidden.shape[1]):
            raise ValueError(f"layer {layer}: Jacobian shape mismatch")
        fitted_norm = float(jacobian.norm().item())
        seed = deterministic_transport_seed(RANDOM_TRANSPORT_SEED, int(layer))
        generator = torch.Generator(device=device).manual_seed(seed)
        random_matrix = torch.randn(
            jacobian.shape, generator=generator, device=device, dtype=torch.float32
        )
        random_matrix.mul_(fitted_norm / random_matrix.norm())
        realized_norm = float(random_matrix.norm().item())
        transported = {
            "identity": hidden,
            "jacobian_lens": hidden @ jacobian.T,
            "random_gaussian": hidden @ random_matrix.T,
        }
        rows = []
        for transport, vectors in transported.items():
            means, stds, probe_z = _torch_margin_batch(
                torch,
                vectors,
                mean_embedding,
                second_moment,
                probe_embeddings,
                norm_epsilon,
            )
            refusal = probe_z[:, :n_refusal].mean(dim=1)
            compliance = probe_z[:, n_refusal:].mean(dim=1)
            for observation in observations:
                index = available_index.get(observation.observation_id)
                margin = (
                    {
                        "vocabulary_logit_mean": float(means[index].item()),
                        "vocabulary_logit_std": float(stds[index].item()),
                        "refusal_probe_mean_z": float(refusal[index].item()),
                        "compliance_probe_mean_z": float(compliance[index].item()),
                        "refusal_minus_compliance_margin": float(
                            (refusal[index] - compliance[index]).item()
                        ),
                    }
                    if index is not None
                    else None
                )
                receipt = {
                    "schema_version": "1.0",
                    "study_id": public_plan["study_id"],
                    "public_plan_sha256": public_plan_sha256,
                    "source_commit": source_commit,
                    "run_id": run_id,
                    **observation.public_metadata(),
                    "transport": transport,
                    "layer": int(layer),
                    "random_seed": seed if transport == "random_gaussian" else None,
                    "fitted_frobenius_norm": (
                        fitted_norm if transport == "random_gaussian" else None
                    ),
                    "realized_frobenius_norm": (
                        realized_norm if transport == "random_gaussian" else None
                    ),
                    "refusal_probe_token_ids": refusal_ids,
                    "compliance_probe_token_ids": compliance_ids,
                    "margin": margin,
                    "model_revision": model_revision,
                    "tokenizer_revision": tokenizer_revision,
                    "lens_sha256": lens_sha256,
                    "sae_sha256": sae_sha256,
                    "runtime": runtime,
                }
                validate_mechanism_receipt(receipt)
                rows.append(receipt)
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "layer": int(layer),
            "public_plan_sha256": public_plan_sha256,
            "source_commit": source_commit,
            "capture_sha256": capture_sha256,
            "receipts": rows,
        }
        _validate_layer_checkpoint(
            payload,
            layer=int(layer),
            public_plan_sha256=public_plan_sha256,
            source_commit=source_commit,
            run_id=run_id,
            capture_sha256=capture_sha256,
            observations=observations,
        )
        write_json_atomic(final_path, payload)
        print(f"completed transport layer {layer}", flush=True)


def _run_sae_discovery(
    *,
    torch,
    sae_path: Path,
    observations: list[Observation],
    states: dict[int, Any],
    output_root: Path,
    provenance: dict[str, Any],
) -> None:
    final_path = output_root / "sae-discovery.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        for key, value in provenance.items():
            if existing.get(key) != value:
                raise ValueError(f"SAE resume provenance drift: {key}")
        return
    available = [
        item
        for item in observations
        if item.position_available and item.position == PRIMARY_POSITION
    ]
    hidden = states[SAE_HOOK_LAYER].float()
    if hidden.shape[0] != len([item for item in observations if item.position_available]):
        raise ValueError("SAE state/observation topology mismatch")
    all_available = [item for item in observations if item.position_available]
    state_indices = {
        item.observation_id: index for index, item in enumerate(all_available)
    }
    assistant_hidden = torch.stack(
        [hidden[state_indices[item.observation_id]] for item in available]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    encoder = state_dict["encoder_linear.weight"].float().to(device)
    encoder_bias = state_dict["encoder_linear.bias"].float().to(device)
    decoder = state_dict["decoder_linear.weight"].float().to(device)
    decoder_bias = state_dict.get("decoder_linear.bias")
    if decoder_bias is not None:
        decoder_bias = decoder_bias.float().to(device)
    with torch.inference_mode():
        acts = torch.relu(assistant_hidden.to(device) @ encoder.T + encoder_bias)
        decoder_norms = decoder.norm(dim=0)
        reconstruction = acts @ decoder.T
        if decoder_bias is not None:
            reconstruction = reconstruction + decoder_bias
        reconstruction_error = (
            (reconstruction - assistant_hidden.to(device)).norm(dim=1)
            / assistant_hidden.to(device).norm(dim=1).clamp_min(1e-12)
        )
        sparsity = (acts > 0).sum(dim=1)
    row_index = {
        (item.behavior_id, item.arm): index for index, item in enumerate(available)
    }
    behavior_ids = sorted({item.behavior_id for item in available})
    full = torch.stack([acts[row_index[(behavior_id, "full")]] for behavior_id in behavior_ids])
    sham = torch.stack(
        [acts[row_index[(behavior_id, "structural_sham")]] for behavior_id in behavior_ids]
    )
    diagnostics = sae_feature_diagnostics(
        full.cpu().numpy(),
        sham.cpu().numpy(),
        decoder_norms.cpu().numpy(),
    )
    selected = select_sae_candidates(diagnostics)
    payload = {
        "schema_version": "1.0",
        **provenance,
        "split": "discovery",
        "hook_layer": SAE_HOOK_LAYER,
        "selection_rule": {
            "contrast": "paired full minus structural_sham at turn-2 assistant boundary",
            "minimum_full_prevalence": 0.10,
            "ranking": "descending paired standardized delta, then mean delta, then feature ID",
            "maximum_candidates": 4,
        },
        "selected_feature_ids": [item.feature_id for item in selected],
        "diagnostics": [asdict(item) for item in diagnostics],
        "activation_sparsity": {
            "mean_active_features": float(sparsity.float().mean().item()),
            "minimum_active_features": int(sparsity.min().item()),
            "maximum_active_features": int(sparsity.max().item()),
        },
        "reconstruction_relative_error": {
            "mean": float(reconstruction_error.mean().item()),
            "maximum": float(reconstruction_error.max().item()),
        },
        "assistant_observation_ids": [item.observation_id for item in available],
    }
    write_json_atomic(final_path, payload)


def run_mechanism_discovery(
    *,
    private_plan_path: Path,
    public_plan_path: Path,
    artifacts_manifest_path: Path,
    generation_root: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
    max_behaviors: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    import torch
    import transformers
    import jlens

    public_plan = json.loads(public_plan_path.read_text())
    public_sha = sha256_file(public_plan_path)
    if public_plan["artifacts_manifest_sha256"] != sha256_file(artifacts_manifest_path):
        raise ValueError("public plan/artifact manifest hash mismatch")
    artifacts_manifest = json.loads(artifacts_manifest_path.read_text())
    expected_hashes = {}
    for artifact in artifacts_manifest["artifacts"]:
        if artifact["role"] in {"jacobian_lens", "sae"}:
            files = artifact["files"]
            if len(files) != 1 or not files[0].get("sha256"):
                raise ValueError(f"{artifact['role']}: expected one SHA-pinned artifact")
            expected_hashes[artifact["role"]] = files[0]["sha256"]
    lens_sha = sha256_file(lens_path)
    sae_sha = sha256_file(sae_path)
    if lens_sha != expected_hashes.get("jacobian_lens"):
        raise ValueError("Jacobian-lens artifact hash mismatch")
    if sae_sha != expected_hashes.get("sae"):
        raise ValueError("SAE artifact hash mismatch")
    source_commit = _source_commit()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    observations = build_observations(
        tokenizer=tokenizer,
        private_plan_path=private_plan_path,
        public_plan_path=public_plan_path,
        generation_root=generation_root,
        split="discovery",
        max_behaviors=max_behaviors,
    )
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    if torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            index: "125GiB" for index in range(torch.cuda.device_count())
        }
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, **load_kwargs
    ).eval()
    if (
        int(hf_model.config.hidden_size) != 8192
        or int(hf_model.config.num_hidden_layers) != 80
        or int(hf_model.config.vocab_size) != 128256
    ):
        raise ValueError("target model topology mismatch")
    if torch.cuda.device_count() <= 1:
        hf_model = hf_model.to("cuda" if torch.cuda.is_available() else "cpu")
    lens = jlens.JacobianLens.load(lens_path)
    source_layers = [int(layer) for layer in lens.source_layers]
    if len(source_layers) != len(set(source_layers)) or not source_layers:
        raise ValueError("invalid lens source-layer declaration")
    capture_layers = sorted(set(source_layers + [SAE_HOOK_LAYER]))
    model_revision = getattr(hf_model.config, "_commit_hash", None) or MODEL_REVISION
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or MODEL_REVISION
    capture_provenance = {
        "study_id": public_plan["study_id"],
        "run_id": run_id,
        "public_plan_sha256": public_sha,
        "source_commit": source_commit,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "lens_sha256": lens_sha,
        "sae_sha256": sae_sha,
    }
    states, capture_sha = _capture_states(
        torch=torch,
        hf_model=hf_model,
        observations=observations,
        capture_layers=capture_layers,
        output_root=output_root,
        provenance=capture_provenance,
    )
    _run_transports(
        torch=torch,
        hf_model=hf_model,
        lens=lens,
        observations=observations,
        states=states,
        output_root=output_root,
        public_plan=public_plan,
        public_plan_sha256=public_sha,
        run_id=run_id,
        source_commit=source_commit,
        lens_sha256=lens_sha,
        sae_sha256=sae_sha,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        capture_sha256=capture_sha,
    )
    provenance = {
        "study_id": public_plan["study_id"],
        "run_id": run_id,
        "public_plan_sha256": public_sha,
        "source_commit": source_commit,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "lens_sha256": lens_sha,
        "sae_sha256": sae_sha,
        "capture_sha256": capture_sha,
    }
    _run_sae_discovery(
        torch=torch,
        sae_path=sae_path,
        observations=observations,
        states=states,
        output_root=output_root,
        provenance=provenance,
    )
    summary = {
        "schema_version": "1.0",
        **provenance,
        "status": "complete",
        "source_layers": source_layers,
        "observations": len(observations),
        "max_behaviors": max_behaviors,
        "available_observations": sum(item.position_available for item in observations),
        "layer_receipts": len(source_layers),
        "elapsed_seconds": time.monotonic() - started,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "jlens": getattr(jlens, "__version__", "unknown"),
            "cuda": torch.version.cuda,
        },
    }
    write_json_atomic(output_root / "summary.json", summary)
    return summary
