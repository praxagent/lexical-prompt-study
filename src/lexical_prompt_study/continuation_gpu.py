"""Exact-token continuation of previously capped A142 generations.

This module never renders a chat template or decodes a prefix for generation.
The caller is responsible for the source receipt/authorization checks before
constructing the runtime. Token lists and captured tensors are restricted data;
only the numeric/hash readouts returned by ``readout`` are release candidates.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file


READOUT_CHECKPOINTS = (16, 32, 64, 128, 256, 512, 1024)
RESIDUAL_LAYERS = (8, 16, 19, 24, 30)
MAX_TOTAL_TOKENS = 1024


def _token_hash(tokens: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(tokens)))


def _validate_ids(tokens: Sequence[int], *, vocabulary_size: int) -> list[int]:
    value = list(tokens)
    if any(type(token) is not int or token < 0 or token >= vocabulary_size for token in value):
        raise ValueError("continuation token IDs invalid")
    return value


def _eos_ids(runtime: Any) -> set[int]:
    configured = runtime.model.generation_config.eos_token_id
    result = {configured} if isinstance(configured, int) else set(configured or ())
    tokenizer_eos = getattr(runtime.tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        result.add(tokenizer_eos)
    if not result or any(
        type(token) is not int or not 0 <= token < runtime.model.config.vocab_size
        for token in result
    ):
        raise ValueError("continuation EOS configuration invalid")
    return result


def _require_plain_greedy_config(model: Any) -> None:
    # A142 explicitly used do_sample=False. All nontrivial logits processors
    # must be absent before direct argmax is allowed to reproduce that policy.
    expected = {
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "force_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "sequence_bias": None,
        "exponential_decay_length_penalty": None,
        "min_length": 0,
        "min_new_tokens": None,
        "watermarking_config": None,
        "guidance_scale": None,
        "constraints": None,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "token_healing": False,
        "remove_invalid_values": False,
        "renormalize_logits": False,
        "penalty_alpha": None,
        "diversity_penalty": 0.0,
    }
    if any(
        getattr(model.generation_config, key, default) not in (None, default)
        for key, default in expected.items()
    ):
        raise ValueError("continuation requires the frozen plain-greedy generation policy")


class PrefixGenerator:
    """Single-row greedy generator retaining its KV cache between horizons.

    ``runtime`` is the pinned ``WeaponizationPrefillRuntime``. A compatible
    small CPU runtime may be supplied by tests. ``residual_root`` must be a
    private per-trial directory; existing tensors are verified, never replaced.
    A new instance can resume any verified prefix at least 128 tokens long.
    """

    def __init__(
        self,
        runtime: Any,
        prompt_token_ids: Sequence[int],
        *,
        residual_root: Path | None = None,
        deadline_monotonic: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        preserved_readouts: Mapping[str, Mapping[str, Any]] | None = None,
        readout_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.torch = runtime.torch
        self.model = runtime.model
        self.device = next(self.model.parameters()).device
        self.vocabulary_size = int(self.model.config.vocab_size)
        self.prompt_token_ids = _validate_ids(
            prompt_token_ids, vocabulary_size=self.vocabulary_size
        )
        if not self.prompt_token_ids:
            raise ValueError("continuation prompt is empty")
        self.eos_token_ids = _eos_ids(runtime)
        _require_plain_greedy_config(self.model)
        if list(runtime.source_layers) != list(range(31)):
            raise ValueError("continuation requires the pinned 31-layer lens")
        if int(runtime.feature_id) != 6779:
            raise ValueError("continuation frozen SAE feature drift")
        self.residual_root = residual_root
        self.deadline_monotonic = deadline_monotonic
        self.clock = clock
        self.readout_sink = readout_sink
        self._created_at = clock()
        self._forward_count = 0
        self._forward_elapsed_seconds = 0.0
        self._readout_elapsed_seconds = 0.0
        self._generated_new_token_count = 0
        self._prefix: list[int] | None = None
        self._cache: Any = None
        self._next_logits: Any = None
        self._stopped = False
        self._readouts: dict[int, dict[str, Any]] = {}
        self._preserved_readouts = dict(preserved_readouts or {})

    def _check_deadline(self) -> None:
        if self.deadline_monotonic is not None and self.clock() >= self.deadline_monotonic:
            raise TimeoutError("continuation monotonic execution deadline reached")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _capture(
        self, token_ids: list[int], targets: Mapping[int, int], *, cache: Any = None
    ) -> Any:
        """Forward once; target maps generated counts to indices in this input."""
        self._check_deadline()
        torch = self.torch
        captured: dict[int, dict[int, Any]] = {count: {} for count in targets}
        handles = []
        for layer in self.runtime.source_layers:

            def hook(_module: Any, _inputs: Any, output: Any, *, layer_index: int = layer) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                for count, index in targets.items():
                    captured[count][layer_index] = hidden[:, index, :].detach().clone()

            if targets:
                handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        total = len(self.prompt_token_ids) + len(self._prefix or [])
        self._synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model(
                    input_ids=torch.tensor([token_ids], dtype=torch.long, device=self.device),
                    attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        self._synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._forward_count += 1
        self._forward_elapsed_seconds += elapsed_ms / 1000
        self._check_deadline()
        for count, states in captured.items():
            self._record_readout(count, states, elapsed_ms)
        return output

    def _persist_residual(self, count: int, states: Mapping[int, Any]) -> str | None:
        if self.residual_root is None:
            return None
        torch = self.torch
        root = self.residual_root
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        path = root / f"residual-{count:04d}.pt"
        payload = {
            "schema_version": "1.0",
            "generated_token_count": count,
            "prompt_token_ids_sha256": _token_hash(self.prompt_token_ids),
            "prefix_token_ids_sha256": _token_hash((self._prefix or [])[:count]),
            "residual_post_layers": list(RESIDUAL_LAYERS),
            "residuals": {
                layer: states[layer].squeeze(0).to(device="cpu", dtype=torch.bfloat16)
                for layer in RESIDUAL_LAYERS
            },
        }
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            expected_meta = {key: value for key, value in payload.items() if key != "residuals"}
            actual_meta = {key: value for key, value in previous.items() if key != "residuals"}
            if actual_meta != expected_meta or set(previous["residuals"]) != set(RESIDUAL_LAYERS):
                raise ValueError("continuation residual resume metadata drift")
            if any(
                not torch.equal(previous["residuals"][layer], payload["residuals"][layer])
                for layer in RESIDUAL_LAYERS
            ):
                raise ValueError("continuation residual resume numerical drift")
            return sha256_file(path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        # Open exclusively with restrictive permissions before tensor bytes exist.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return sha256_file(path)

    def _record_readout(self, count: int, states: Mapping[int, Any], forward_ms: float) -> None:
        # A crash can leave the next horizon's diagnostic receipt on disk just
        # before its exact-token artifact is committed. Reuse that measurement
        # only after deterministic continuation reaches the same prefix hash.
        if str(count) in self._preserved_readouts:
            self._restore_readouts(self._prefix or [])
            if count in self._readouts:
                return
        torch = self.torch
        runtime = self.runtime
        if sorted(states) != list(range(31)):
            raise ValueError("continuation residual capture incomplete")
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            hidden = states[19].float()
            activations = torch.relu(hidden @ runtime.encoder.T + runtime.encoder_bias)
            reconstruction = activations @ runtime.decoder.T
            if runtime.decoder_bias is not None:
                reconstruction = reconstruction + runtime.decoder_bias
            reconstruction_error = (reconstruction - hidden).norm(dim=1) / hidden.norm(
                dim=1
            ).clamp_min(1e-12)
            feature = activations[:, runtime.feature_id]
            subspace = (
                activations.index_select(1, runtime.subspace_ids)
                * runtime.subspace_weights[None, :]
            ).sum(dim=1)
            trajectory = [
                float(
                    runtime._probe_margin(states[layer].float() @ runtime.jacobians[layer].T)[
                        0
                    ].item()
                )
                for layer in runtime.source_layers
            ]
        self._synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._readout_elapsed_seconds += elapsed_ms / 1000
        value = {
            "prefix_token_count": count,
            "prefix_token_ids_sha256": _token_hash((self._prefix or [])[:count]),
            "feature_6779_magnitude": float(feature[0].item()),
            "frozen_subspace_score": float(subspace[0].item()),
            "sae_normalized_reconstruction_error": float(reconstruction_error[0].item()),
            "jlens_refusal_minus_compliance_trajectory": trajectory,
            "prefill_latency_ms": forward_ms,
            "detector_readout_latency_ms": elapsed_ms,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0,
        }
        finite = [
            value["feature_6779_magnitude"],
            value["frozen_subspace_score"],
            value["sae_normalized_reconstruction_error"],
            *trajectory,
        ]
        if not all(math.isfinite(item) for item in finite):
            raise ValueError("continuation nonfinite internal readout")
        residual_hash = self._persist_residual(count, states)
        if residual_hash is not None:
            value["residual_artifact_sha256"] = residual_hash
        self._readouts[count] = value
        if self.readout_sink is not None:
            self.readout_sink(dict(value))

    def _restore_readouts(self, prefix: list[int]) -> None:
        """Reuse receipt-bound earlier measurements, not numerical replays.

        Cached decoding and a full-prefix cache rebuild need not be bitwise
        equal in BF16. Earlier readouts remain immutable and are verified by
        token-prefix and residual-file hashes before they are carried forward.
        """
        required = {
            "prefix_token_count",
            "prefix_token_ids_sha256",
            "feature_6779_magnitude",
            "frozen_subspace_score",
            "sae_normalized_reconstruction_error",
            "jlens_refusal_minus_compliance_trajectory",
            "prefill_latency_ms",
            "detector_readout_latency_ms",
            "peak_gpu_memory_bytes",
        }
        for key, supplied in self._preserved_readouts.items():
            value = dict(supplied)
            if set(value) not in (required, required | {"residual_artifact_sha256"}):
                raise ValueError("continuation preserved readout schema drift")
            count = value["prefix_token_count"]
            if type(count) is not int or str(count) != key or not 1 <= count <= MAX_TOTAL_TOKENS:
                raise ValueError("continuation preserved readout position drift")
            if count > len(prefix):
                # Preserve this immutable orphan diagnostic for verification
                # when/if the regenerated continuation reaches its position.
                continue
            if value["prefix_token_ids_sha256"] != _token_hash(prefix[:count]):
                raise ValueError("continuation preserved readout prefix drift")
            trajectory = value["jlens_refusal_minus_compliance_trajectory"]
            numbers = [
                value[field]
                for field in required
                - {"prefix_token_ids_sha256", "jlens_refusal_minus_compliance_trajectory"}
            ]
            if len(trajectory) != 31 or not all(
                type(item) in (int, float) and math.isfinite(item)
                for item in [*numbers, *trajectory]
            ):
                raise ValueError("continuation preserved readout values invalid")
            residual_hash = value.get("residual_artifact_sha256")
            if residual_hash is not None:
                if self.residual_root is None:
                    raise ValueError("continuation preserved residual root missing")
                path = self.residual_root / f"residual-{count:04d}.pt"
                if not path.is_file() or sha256_file(path) != residual_hash:
                    raise ValueError("continuation preserved residual hash drift")
                payload = self.torch.load(path, map_location="cpu", weights_only=True)
                if (
                    payload.get("prompt_token_ids_sha256") != _token_hash(self.prompt_token_ids)
                    or payload.get("prefix_token_ids_sha256") != _token_hash(prefix[:count])
                    or payload.get("generated_token_count") != count
                    or payload.get("residual_post_layers") != list(RESIDUAL_LAYERS)
                ):
                    raise ValueError("continuation preserved residual metadata drift")
            self._readouts[count] = value

    def prepare(self, prefix_token_ids: Sequence[int]) -> None:
        prefix = _validate_ids(prefix_token_ids, vocabulary_size=self.vocabulary_size)
        if not 128 <= len(prefix) <= MAX_TOTAL_TOKENS:
            raise ValueError("continuation prefix length outside frozen range")
        if any(token in self.eos_token_ids for token in prefix):
            raise ValueError("continuation prefix unexpectedly contains EOS")
        if self._prefix is not None:
            if prefix != self._prefix:
                raise ValueError("continuation cache/prefix identity drift")
            return
        maximum_context = int(self.model.config.max_position_embeddings)
        if len(self.prompt_token_ids) + MAX_TOTAL_TOKENS > maximum_context:
            raise ValueError("continuation would exceed model context window")
        self._prefix = prefix
        self._restore_readouts(prefix)
        targets = {
            count: len(self.prompt_token_ids) + count - 1
            for count in READOUT_CHECKPOINTS
            if count <= len(prefix) and count not in self._readouts
        }
        output = self._capture([*self.prompt_token_ids, *prefix], targets)
        self._cache = output.past_key_values
        self._next_logits = output.logits[:, -1, :].detach().clone()

    def generate_chunk(
        self, prefix_token_ids: Sequence[int], max_new_tokens: int
    ) -> dict[str, Any]:
        self.prepare(prefix_token_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("continuation chunk length invalid")
        assert self._prefix is not None
        if len(self._prefix) + max_new_tokens > MAX_TOTAL_TOKENS:
            raise ValueError("continuation exceeds 1024-token ceiling")
        delta: list[int] = []
        for _ in range(max_new_tokens):
            self._check_deadline()
            if self._stopped:
                break
            if not bool(self.torch.isfinite(self._next_logits).all().item()):
                raise ValueError("continuation nonfinite next-token logits")
            token = int(self._next_logits.argmax(dim=-1).item())
            if token in self.eos_token_ids:
                self._stopped = True
                break
            self._prefix.append(token)
            delta.append(token)
            self._generated_new_token_count += 1
            count = len(self._prefix)
            targets = {count: 0} if count in READOUT_CHECKPOINTS else {}
            output = self._capture([token], targets, cache=self._cache)
            self._cache = output.past_key_values
            self._next_logits = output.logits[:, -1, :].detach().clone()
        return {"delta_token_ids": delta, "eos": self._stopped}

    __call__ = generate_chunk

    def readout(self, prefix_token_ids: Sequence[int]) -> dict[str, Any]:
        self.prepare(prefix_token_ids)
        assert self._prefix is not None
        count = len(self._prefix)
        if count not in self._readouts:
            # An EOS stop may occur between scheduled sites. Capture that actual
            # position once; do not label it as a later nominal horizon.
            all_ids = [*self.prompt_token_ids, *self._prefix]
            output = self._capture(all_ids, {count: len(all_ids) - 1})
            self._cache = output.past_key_values
            self._next_logits = output.logits[:, -1, :].detach().clone()
        return dict(self._readouts[count])

    @property
    def readouts_by_generated_tokens(self) -> dict[str, dict[str, Any]]:
        return {str(count): dict(value) for count, value in sorted(self._readouts.items())}

    @property
    def timing_summary(self) -> dict[str, int | float]:
        return {
            "elapsed_seconds": max(0.0, self.clock() - self._created_at),
            "generated_new_token_count": self._generated_new_token_count,
            "model_forward_count": self._forward_count,
            "model_forward_elapsed_seconds": self._forward_elapsed_seconds,
            "detector_readout_elapsed_seconds": self._readout_elapsed_seconds,
        }


def verify_continuation_parity(
    runtime: Any,
    prompt_token_ids: Sequence[int],
    original_prefix_token_ids: Sequence[int],
    *,
    extension_tokens: int = 8,
    deadline_monotonic: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Compare exact-prefix conditional continuation with trusted HF generation.

    The saved 128 IDs are fixed conditioning input for both implementations.
    They are never regenerated or compared with a new sampling of the original
    prompt. The adapter crosses one chunk boundary while the trusted path uses
    one uninterrupted ``generate`` call. Returned diagnostics are hashes and
    numeric/status metadata only; any conditional mismatch stops the gate.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    if len(original_prefix_token_ids) != 128 or not 1 <= extension_tokens <= 16:
        raise ValueError("continuation parity pilot shape invalid")
    generator = PrefixGenerator(
        runtime,
        prompt_token_ids,
        deadline_monotonic=deadline_monotonic,
        clock=clock,
    )
    original = _validate_ids(original_prefix_token_ids, vocabulary_size=generator.vocabulary_size)
    original_hash = _token_hash(original)
    if any(token in generator.eos_token_ids for token in original):
        raise ValueError("continuation parity prefix unexpectedly contains EOS")
    generator._check_deadline()
    torch = runtime.torch

    class Deadline(StoppingCriteria):
        def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
            return deadline_monotonic is not None and clock() >= deadline_monotonic

    conditional_input = [*generator.prompt_token_ids, *original]
    input_ids = torch.tensor([conditional_input], dtype=torch.long, device=generator.device)
    with torch.inference_mode():
        sequence = runtime.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=extension_tokens,
            pad_token_id=runtime.tokenizer.pad_token_id,
            eos_token_id=sorted(generator.eos_token_ids),
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([Deadline()]),
        )
    generator._check_deadline()
    trusted_input_unchanged = sequence[0, : input_ids.shape[1]].tolist() == conditional_input
    uninterrupted = [int(value) for value in sequence[0, input_ids.shape[1] :].tolist()]
    eos_index = next(
        (index for index, token in enumerate(uninterrupted) if token in generator.eos_token_ids),
        None,
    )
    uninterrupted_eos = eos_index is not None
    if eos_index is not None:
        uninterrupted = uninterrupted[:eos_index]
    split = max(1, extension_tokens // 2)
    first = generator(original, split)
    continued = list(first["delta_token_ids"])
    continued_eos = first["eos"]
    if split < extension_tokens and not continued_eos:
        second = generator([*original, *continued], extension_tokens - split)
        continued.extend(second["delta_token_ids"])
        continued_eos = second["eos"]
    original_immutable = _token_hash(original_prefix_token_ids) == original_hash
    continuation_matches = continued == uninterrupted and continued_eos == uninterrupted_eos
    passed = trusted_input_unchanged and original_immutable and continuation_matches
    return {
        "status": "pass" if passed else "stop_conditional_parity_mismatch",
        "parity_scope": "exact_saved_prefix_conditional",
        "original_prefix_token_count": len(original),
        "original_prefix_token_ids_sha256": original_hash,
        "conditional_input_token_ids_sha256": _token_hash(conditional_input),
        "extension_token_budget": extension_tokens,
        "split_boundary_new_tokens": split,
        "trusted_continuation_token_count": len(uninterrupted),
        "trusted_continuation_token_ids_sha256": _token_hash(uninterrupted),
        "adapter_continuation_token_count": len(continued),
        "adapter_continuation_token_ids_sha256": _token_hash(continued),
        "trusted_eos": uninterrupted_eos,
        "adapter_eos": continued_eos,
        "trusted_input_prefix_unchanged": trusted_input_unchanged,
        "original_prefix_immutable": original_immutable,
        "conditional_continuation_matches": continuation_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact-token A142 continuation GPU adapter; use the authorized continuation runner."
    )
    parser.parse_args()
    parser.error("This adapter has no standalone paid-execution entry point")


if __name__ == "__main__":
    main()
