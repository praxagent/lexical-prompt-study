from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .behavior import _as_token_ids, _peak_memory
from .followup_design import PLACEMENTS, validate_placement_render_pair
from .followup_plan import validate_followup_plan
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .models import FollowupQualificationReceipt

PLACEMENT_TEMPLATE_VERSION = "placement-blocks-v1"
PLACEMENT_PROLOGUE = "PLACEMENT-CONTROLLED INPUT\n"
PLACEMENT_SEPARATOR = "\n<PLACEMENT-BLOCK-BOUNDARY>\n"
PLACEMENT_EPILOGUE = "\nEND PLACEMENT-CONTROLLED INPUT"
SAFE_REQUEST = "Explain why checksum verification is useful in reproducible research."
SAFE_SCAFFOLD = (
    "Answer concisely, preserve the factual meaning, and mention one practical example."
)


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _token_ids(tokenizer, text: str) -> list[int]:
    return _as_token_ids(tokenizer.encode(text, add_special_tokens=False))


def _unique_subsequence_offset(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("placement component token sequence cannot be empty")
    starts = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    ]
    if len(starts) != 1:
        raise ValueError(
            "placement component token sequence is not uniquely recoverable "
            f"(matches={len(starts)})"
        )
    return starts[0], starts[0] + len(subsequence)


def _render_text(request: str, scaffold: str, placement: str) -> str:
    if placement == "ep_before_request":
        first, second = scaffold, request
    elif placement == "ep_after_request":
        first, second = request, scaffold
    else:
        raise ValueError(f"unknown placement: {placement}")
    return f"{PLACEMENT_PROLOGUE}{first}{PLACEMENT_SEPARATOR}{second}{PLACEMENT_EPILOGUE}"


def build_placement_render_pair(
    *,
    tokenizer,
    request: str,
    scaffold: str,
    tokenizer_revision: str,
    context_ceiling: int,
    generation_budget: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build private renders and return a non-reconstructive public validation."""

    if not request or not scaffold or request == scaffold:
        raise ValueError("placement components must be distinct non-empty strings")
    if context_ceiling < 1 or generation_budget < 1:
        raise ValueError("context and generation ceilings must be positive")
    component_text = {"request": request, "scaffold": scaffold}
    component_bytes_sha = {
        name: sha256_text(text) for name, text in component_text.items()
    }
    component_ids = {
        name: _token_ids(tokenizer, text) for name, text in component_text.items()
    }
    component_token_sha = {
        name: sha256_bytes(canonical_json_bytes(ids))
        for name, ids in component_ids.items()
    }
    component_counts = {name: len(ids) for name, ids in component_ids.items()}
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_sha = sha256_text(chat_template or "")
    template_sha = sha256_text(
        PLACEMENT_TEMPLATE_VERSION
        + PLACEMENT_PROLOGUE
        + PLACEMENT_SEPARATOR
        + PLACEMENT_EPILOGUE
    )
    private: dict[str, dict[str, Any]] = {}
    public_inputs: list[dict[str, Any]] = []
    for placement in PLACEMENTS:
        rendered = _render_text(request, scaffold, placement)
        messages = [{"role": "user", "content": rendered}]
        prompt_ids = _as_token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(prompt_ids) + generation_budget > context_ceiling:
            raise ValueError("placement render exceeds frozen context ceiling")
        offsets = {
            name: list(_unique_subsequence_offset(prompt_ids, ids))
            for name, ids in component_ids.items()
        }
        occupied: set[int] = set()
        for start, end in offsets.values():
            occupied.update(range(start, end))
        delimiter_ids = [
            token for index, token in enumerate(prompt_ids) if index not in occupied
        ]
        suffix_length = min(8, len(prompt_ids))
        receipt = {
            "placement": placement,
            "template_sha256": template_sha,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_sha256": chat_template_sha,
            "component_bytes_sha256": component_bytes_sha,
            "component_token_sha256": component_token_sha,
            "component_token_counts": component_counts,
            "delimiter_special_tokens_sha256": sha256_bytes(
                canonical_json_bytes(delimiter_ids)
            ),
            "assistant_boundary_suffix_sha256": sha256_bytes(
                canonical_json_bytes(prompt_ids[-suffix_length:])
            ),
            "component_offsets": offsets,
            "offset_map_sha256": sha256_bytes(canonical_json_bytes(offsets)),
            "total_tokens": len(prompt_ids),
            "context_ceiling": context_ceiling,
            "generation_budget": generation_budget,
            "truncated": False,
            "padding_applied": False,
            "context_shifted": False,
        }
        public_inputs.append(receipt)
        private[placement] = {
            "rendered_text": rendered,
            "prompt_text": prompt_text,
            "prompt_token_ids": prompt_ids,
            "component_token_ids": component_ids,
            "render_receipt": receipt,
        }
    public = validate_placement_render_pair(public_inputs[0], public_inputs[1])
    return public, private


class PrivateCheckpointStore:
    """Atomic, hash-indexed private stages with exact provenance on resume."""

    def __init__(self, root: Path, provenance: dict[str, Any]):
        self.root = root
        self.stage_root = root / "stages"
        self.index_path = root / "index.json"
        self.provenance = provenance
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.chmod(0o700)

    def _index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"schema_version": "1.0", "provenance": self.provenance, "stages": {}}
        value = json.loads(self.index_path.read_text())
        if value.get("provenance") != self.provenance:
            raise ValueError("private checkpoint provenance drift")
        return value

    def load(self, stage: str) -> dict[str, Any] | None:
        index = self._index()
        expected = index["stages"].get(stage)
        if expected is None:
            return None
        path = self.stage_root / f"{stage}.json"
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"private checkpoint hash drift: {stage}")
        return json.loads(path.read_text())

    def write(self, stage: str, payload: dict[str, Any]) -> str:
        if not stage or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in stage
        ):
            raise ValueError("invalid checkpoint stage name")
        path = self.stage_root / f"{stage}.json"
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        index = self._index()
        existing = index["stages"].get(stage)
        if existing is not None:
            if existing != digest:
                raise ValueError(
                    f"refusing to overwrite completed private stage: {stage}"
                )
            if not path.exists() or sha256_file(path) != existing:
                raise ValueError(f"private checkpoint hash drift: {stage}")
            return digest
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
        index["stages"][stage] = digest
        encoded_index = canonical_json_bytes(index)
        temporary_index = self.index_path.with_name(
            f".{self.index_path.name}.{os.getpid()}.tmp"
        )
        temporary_index.write_bytes(encoded_index)
        with temporary_index.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_index.chmod(0o600)
        temporary_index.replace(self.index_path)
        return digest


def _public_receipt_is_complete(
    path: Path,
    *,
    plan_sha256: str,
    source_commit: str,
    run_id: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    receipt = FollowupQualificationReceipt.model_validate_json(path.read_text())
    if (
        receipt.plan_sha256 != plan_sha256
        or receipt.source_commit != source_commit
        or receipt.run_id != run_id
    ):
        raise ValueError("qualification receipt provenance drift")
    private_path = Path(receipt.private_bundle_path)
    if not private_path.exists() or sha256_file(private_path) != receipt.private_bundle_sha256:
        raise ValueError("qualification private bundle drift")
    return receipt.model_dump(mode="json")


def _assert_no_raw_public_fields(value: Any) -> None:
    forbidden = {
        "prompt",
        "prompt_text",
        "prompt_token_ids",
        "generated_text",
        "generated_token_ids",
        "rendered_text",
        "component_token_ids",
        "captured_states",
    }
    if isinstance(value, dict):
        overlap = forbidden.intersection(value)
        if overlap:
            raise ValueError(f"raw fields cannot enter public receipt: {sorted(overlap)}")
        for child in value.values():
            _assert_no_raw_public_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_raw_public_fields(child)


def _atomic_public_receipt(path: Path, payload: dict[str, Any]) -> str:
    receipt = FollowupQualificationReceipt.model_validate(payload)
    encoded = canonical_json_bytes(receipt.model_dump(mode="json"))
    _assert_no_raw_public_fields(receipt.model_dump(mode="json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_bytes(encoded)


def _capture_assistant_boundary(torch, model, prompt_ids: list[int], layers: list[int]):
    captured: dict[int, Any] = {}
    handles = []
    for layer in layers:
        def hook(_module, _inputs, output, *, layer_index=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[0, -1].detach().to("cpu", dtype=torch.bfloat16)

        handles.append(model.model.layers[layer].register_forward_hook(hook))
    device = next(model.parameters()).device
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=tensor, use_cache=False).logits[0, -1].float().cpu()
    for handle in handles:
        handle.remove()
    if sorted(captured) != sorted(layers):
        raise ValueError("qualification activation capture incomplete")
    return captured, logits


def _mechanism_checks(torch, model, lens, sae_path: Path, states_by_placement: dict):
    source_layers = [int(value) for value in lens.source_layers]
    if not source_layers or len(source_layers) != len(set(source_layers)):
        raise ValueError("invalid Jacobian-lens source-layer topology")
    for layer in source_layers:
        jacobian = lens.jacobians[layer]
        if tuple(jacobian.shape) != (model.config.hidden_size, model.config.hidden_size):
            raise ValueError(f"Jacobian-lens shape drift at layer {layer}")
    first_layer = source_layers[0]
    fixture = states_by_placement[PLACEMENTS[0]][first_layer].float()
    transported = fixture @ lens.jacobians[first_layer].float().T
    if not bool(torch.isfinite(transported).all()) or float(transported.norm()) <= 0:
        raise ValueError("Jacobian-lens qualification produced invalid transport")

    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    encoder = state_dict["encoder_linear.weight"].float()
    encoder_bias = state_dict["encoder_linear.bias"].float()
    decoder = state_dict["decoder_linear.weight"].float()
    hook_layer = 19
    if (
        encoder.shape[1] != model.config.hidden_size
        or decoder.shape[0] != model.config.hidden_size
    ):
        raise ValueError("SAE/model hidden-size mismatch")
    fixture_sae = states_by_placement[PLACEMENTS[0]][hook_layer].float()
    acts = torch.relu(fixture_sae @ encoder.T + encoder_bias)
    reconstruction = acts @ decoder.T
    if "decoder_linear.bias" in state_dict:
        reconstruction += state_dict["decoder_linear.bias"].float()
    relative_error = float(
        ((reconstruction - fixture_sae).norm() / fixture_sae.norm().clamp_min(1e-12)).item()
    )
    if not np.isfinite(relative_error):
        raise ValueError("SAE qualification reconstruction is non-finite")
    return {
        "jlens_load_and_transport": True,
        "jlens_source_layer_count": len(source_layers),
        "sae_load_and_encode": True,
        "sae_active_features": int((acts > 0).sum().item()),
        "sae_reconstruction_relative_error": relative_error,
        "hook_layer": hook_layer,
    }


def _patch_checks(torch, model, prompt_ids: list[int], layer: int, donor):
    device = next(model.parameters()).device
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        baseline = model(input_ids=tensor, use_cache=False).logits[0, -1].float()

    def run_hook(replacement):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            modified[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            result = model(input_ids=tensor, use_cache=False).logits[0, -1].float()
        handle.remove()
        return result

    identity_state: dict[str, Any] = {}

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        identity_state["value"] = hidden[0, -1].detach()

    handle = model.model.layers[layer].register_forward_hook(capture)
    with torch.inference_mode():
        model(input_ids=tensor, use_cache=False)
    handle.remove()
    identity = run_hook(identity_state["value"])
    identity_error = float((identity - baseline).abs().max().item())
    patched = run_hook(donor)
    patch_delta = float((patched - baseline).norm().item())
    if identity_error > 0.1 or not np.isfinite(patch_delta) or patch_delta <= 0:
        raise ValueError(
            "state-patch qualification failed "
            f"(identity_error={identity_error}, patch_delta={patch_delta})"
        )
    return {
        "identity_patch_max_logit_error": identity_error,
        "cross_placement_patch_logit_delta_norm": patch_delta,
        "state_patch_hook": True,
    }


def run_followup_qualification(
    *,
    public_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    """Exercise every expensive 8B pipeline using only a synthetic-safe request."""

    started = time.monotonic()
    plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(plan)
    plan_sha = sha256_file(public_plan_path)
    source_commit = _source_commit()
    output_root.mkdir(parents=True, exist_ok=True)
    public_path = output_root / "qualification.public.json"
    completed = _public_receipt_is_complete(
        public_path,
        plan_sha256=plan_sha,
        source_commit=source_commit,
        run_id=run_id,
    )
    if completed is not None:
        return completed

    expected_lens_sha = plan["artifacts"]["llama31_lens"]["sha256"]
    expected_sae_sha = plan["artifacts"]["llama31_sae"]["sha256"]
    lens_sha = sha256_file(lens_path)
    sae_sha = sha256_file(sae_path)
    if lens_sha != expected_lens_sha or sae_sha != expected_sae_sha:
        raise ValueError("qualification mechanism artifact hash mismatch")
    expected_revision = plan["artifacts"]["llama31_model"]["revision"]
    resolved_model_path = str(Path(model_path).resolve())
    if expected_revision not in resolved_model_path:
        raise ValueError("model path is not the frozen Hugging Face snapshot")

    provenance = {
        "study_id": plan["study_id"],
        "plan_sha256": plan_sha,
        "source_commit": source_commit,
        "run_id": run_id,
        "model_revision": expected_revision,
        "lens_sha256": lens_sha,
        "sae_sha256": sae_sha,
    }
    checkpoints = PrivateCheckpointStore(output_root / "private", provenance)

    import torch
    import transformers
    import jlens

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or expected_revision
    render_public, render_private = build_placement_render_pair(
        tokenizer=tokenizer,
        request=SAFE_REQUEST,
        scaffold=SAFE_SCAFFOLD,
        tokenizer_revision=tokenizer_revision,
        context_ceiling=4096,
        generation_budget=8,
    )
    render_stage = {
        "public_validation": render_public,
        "private_renders": render_private,
    }
    existing_render = checkpoints.load("render")
    if existing_render is None:
        checkpoints.write("render", render_stage)
    elif existing_render != render_stage:
        raise ValueError("qualification render checkpoint drift")

    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, **load_kwargs
    ).eval()
    if torch.cuda.device_count() != 1:
        raise ValueError("qualification requires exactly one visible GPU")
    model = model.to("cuda")
    if (
        int(model.config.hidden_size) != 4096
        or int(model.config.num_hidden_layers) != 32
        or int(model.config.vocab_size) != 128256
    ):
        raise ValueError("Llama 3.1 8B topology mismatch")
    runtime_revision = getattr(model.config, "_commit_hash", None) or expected_revision
    if runtime_revision != expected_revision or tokenizer_revision != expected_revision:
        raise ValueError("model/tokenizer revision drift")
    lens = jlens.JacobianLens.load(lens_path)
    layers = sorted({int(value) for value in lens.source_layers} | {19})

    placements_public = []
    states_by_placement = {}
    logits_by_placement = {}
    for placement in PLACEMENTS:
        stage_name = f"generation-{placement}"
        private_stage = checkpoints.load(stage_name)
        if private_stage is None:
            prompt_ids = render_private[placement]["prompt_token_ids"]
            tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            generation_started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(
                    tensor,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.monotonic() - generation_started
            generated_ids = output[0, tensor.shape[1] :].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            private_stage = {
                "placement": placement,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "generated_text": generated_text,
                "elapsed_seconds": elapsed,
            }
            checkpoints.write(stage_name, private_stage)
        prompt_ids = private_stage["prompt_token_ids"]
        states, logits = _capture_assistant_boundary(torch, model, prompt_ids, layers)
        states_by_placement[placement] = states
        logits_by_placement[placement] = logits
        placements_public.append(
            {
                "placement": placement,
                "prompt_sha256": sha256_text(
                    render_private[placement]["prompt_text"]
                ),
                "prompt_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(prompt_ids)
                ),
                "generated_text_sha256": sha256_text(private_stage["generated_text"]),
                "generated_token_count": len(private_stage["generated_token_ids"]),
                "elapsed_seconds": float(private_stage["elapsed_seconds"]),
                "assistant_boundary_state_sha256": sha256_bytes(
                    states[19].view(torch.uint16).numpy().tobytes()
                ),
            }
        )

    mechanism = _mechanism_checks(
        torch,
        model,
        lens,
        sae_path,
        states_by_placement,
    )
    patch = _patch_checks(
        torch,
        model,
        render_private[PLACEMENTS[0]]["prompt_token_ids"],
        19,
        states_by_placement[PLACEMENTS[1]][19],
    )
    private_bundle = output_root / "private" / "index.json"
    payload = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "plan_sha256": plan_sha,
        "run_id": run_id,
        "source_commit": source_commit,
        "qualification_kind": "synthetic_safe",
        "status": "complete",
        "model_revision": runtime_revision,
        "tokenizer_revision": tokenizer_revision,
        "lens_sha256": lens_sha,
        "sae_sha256": sae_sha,
        "render_validation": render_public,
        "placements": placements_public,
        "pipeline_checks": {
            "generation": True,
            "assistant_boundary_capture": True,
            **mechanism,
            **patch,
        },
        "model_topology": {
            "hidden_size": int(model.config.hidden_size),
            "layers": int(model.config.num_hidden_layers),
            "vocabulary": int(model.config.vocab_size),
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": _peak_memory(torch),
        "private_bundle_path": str(private_bundle),
        "private_bundle_sha256": sha256_file(private_bundle),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "jlens": getattr(jlens, "__version__", "unknown"),
            "cuda": torch.version.cuda,
        },
    }
    digest = _atomic_public_receipt(public_path, payload)
    payload["qualification_receipt_sha256"] = digest
    return payload
