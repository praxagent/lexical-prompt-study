from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .breaker import RestrictedExactMatcher
from .factorial_gpu import _probe_manifest, _verify_probe_tokens
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .weaponization_plan import validate_weaponization_plan
from .weaponization_runner import run_weaponization_prefill_loop


def structural_prompt_metrics(prompt: str) -> dict[str, float]:
    return {
        "prompt_bytes": float(len(prompt.encode("utf-8"))),
        "line_count": float(prompt.count("\n") + 1),
        "dash_divider_count": float(prompt.count("---")),
        "equals_divider_count": float(prompt.count("===")),
        "markdown_heading_count": float(prompt.count("###")),
        "bracket_count": float(prompt.count("[") + prompt.count("]")),
    }


class WeaponizationPrefillRuntime:
    def __init__(
        self,
        *,
        public_plan: Mapping[str, Any],
        probe_plan_path: Path,
        model_path: str,
        lens_path: Path,
        sae_path: Path,
        factorial_material_path: Path,
    ) -> None:
        import jlens
        import torch
        import transformers

        self.torch = torch
        self.started = time.monotonic()
        artifacts = public_plan["pinned_instrument"]
        expected_revision = artifacts["model_revision"]
        snapshot = Path(model_path)
        if not snapshot.is_dir() or snapshot.resolve().name != expected_revision:
            raise ValueError("weaponization model snapshot revision path drift")
        if sha256_file(lens_path) != artifacts["jacobian_lens_sha256"]:
            raise ValueError("weaponization J-lens hash drift")
        if sha256_file(sae_path) != artifacts["sae_sha256"]:
            raise ValueError("weaponization SAE hash drift")
        if torch.cuda.device_count() != 1:
            raise ValueError("weaponization runtime requires exactly one visible GPU")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        observed_tokenizer_revision = getattr(self.tokenizer, "_commit_hash", None)
        if observed_tokenizer_revision not in (None, expected_revision):
            raise ValueError("weaponization tokenizer revision drift")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        self.model = self.model.to("cuda")
        if (
            int(self.model.config.hidden_size) != 4096
            or int(self.model.config.num_hidden_layers) != 32
            or int(self.model.config.vocab_size) != 128256
        ):
            raise ValueError("weaponization Llama 3.1 8B topology drift")
        observed_model_revision = getattr(self.model.config, "_commit_hash", None)
        if observed_model_revision not in (None, expected_revision):
            raise ValueError("weaponization model revision drift")

        probe_plan = json.loads(probe_plan_path.read_text())
        if sha256_file(probe_plan_path) != artifacts[
            "jacobian_lens_probe_source_plan_sha256"
        ]:
            raise ValueError("weaponization probe source-plan hash drift")
        manifest = _probe_manifest(probe_plan)
        if sha256_bytes(canonical_json_bytes(manifest)) != artifacts[
            "jacobian_lens_probe_manifest_sha256"
        ]:
            raise ValueError("weaponization probe manifest hash drift")
        refusal_ids = _verify_probe_tokens(self.tokenizer, manifest["refusal"])
        compliance_ids = _verify_probe_tokens(self.tokenizer, manifest["compliance"])
        self.n_refusal = len(refusal_ids)
        probe_ids = refusal_ids + compliance_ids

        lens = jlens.JacobianLens.load(lens_path)
        self.source_layers = [int(layer) for layer in lens.source_layers]
        if self.source_layers != list(range(31)):
            raise ValueError("weaponization J-lens source-layer topology drift")
        self.jacobians = {
            layer: lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
            for layer in self.source_layers
        }
        if any(tuple(value.shape) != (4096, 4096) for value in self.jacobians.values()):
            raise ValueError("weaponization J-lens matrix shape drift")

        state = torch.load(sae_path, map_location="cpu", weights_only=True)
        self.encoder = state["encoder_linear.weight"].to(
            device="cuda", dtype=torch.float32
        )
        self.encoder_bias = state["encoder_linear.bias"].to(
            device="cuda", dtype=torch.float32
        )
        self.decoder = state["decoder_linear.weight"].to(
            device="cuda", dtype=torch.float32
        )
        self.decoder_bias = (
            state["decoder_linear.bias"].to(device="cuda", dtype=torch.float32)
            if "decoder_linear.bias" in state
            else None
        )
        if self.encoder.shape[1] != 4096 or self.decoder.shape[0] != 4096:
            raise ValueError("weaponization SAE/model shape drift")
        self.feature_id = int(artifacts["frozen_feature_id"])
        self.subspace_ids = torch.tensor(
            artifacts["frozen_subspace_feature_ids"], device="cuda", dtype=torch.long
        )
        self.subspace_weights = torch.tensor(
            artifacts["frozen_subspace_weights"],
            device="cuda",
            dtype=torch.float32,
        )

        output_weight = self.model.lm_head.weight.detach().to(
            device="cuda", dtype=torch.float32
        )
        norm_weight = self.model.model.norm.weight.detach().to(
            device="cuda", dtype=torch.float32
        )
        effective = output_weight * norm_weight[None, :]
        self.mean_embedding = effective.mean(dim=0)
        self.second_moment = effective.T @ effective / effective.shape[0]
        self.probe_embeddings = effective[probe_ids]
        self.norm_epsilon = float(self.model.model.norm.variance_epsilon)
        del effective, output_weight, norm_weight, lens, state

        factorial_material = json.loads(factorial_material_path.read_text())
        joiner = str(factorial_material["material_block_joiner"])
        full_text = joiner.join(
            str(block["text"])
            for block in factorial_material["scaffold_materials"]["full_scaffold"][
                "blocks"
            ]
        )
        self.exact_matcher = RestrictedExactMatcher([full_text.encode("utf-8")])

    def _capture_batch(self, observations: Sequence[Mapping[str, Any]]) -> dict[int, Any]:
        torch = self.torch
        lengths = torch.tensor(
            [len(row["prompt_token_ids"]) for row in observations],
            device="cuda",
            dtype=torch.long,
        )
        maximum = int(lengths.max().item())
        batch = len(observations)
        input_ids = torch.full(
            (batch, maximum),
            int(self.tokenizer.pad_token_id),
            device="cuda",
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch, maximum), device="cuda", dtype=torch.long)
        for index, row in enumerate(observations):
            ids = torch.tensor(row["prompt_token_ids"], device="cuda", dtype=torch.long)
            input_ids[index, : ids.numel()] = ids
            attention_mask[index, : ids.numel()] = 1
        row_indices = torch.arange(batch, device="cuda")
        captured: dict[int, Any] = {}
        handles = []
        for layer in self.source_layers:
            def hook(_module, _inputs, output, *, layer_index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_index] = hidden[row_indices, lengths - 1].detach().float()

            handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        if sorted(captured) != self.source_layers:
            raise ValueError("weaponization assistant-boundary capture incomplete")
        return captured

    def _probe_margin(self, vectors: Any) -> Any:
        torch = self.torch
        normalized = vectors * torch.rsqrt(
            vectors.square().mean(dim=1, keepdim=True) + self.norm_epsilon
        )
        mean = normalized @ self.mean_embedding
        second = (normalized @ self.second_moment * normalized).sum(dim=1)
        std = torch.clamp(second - mean.square(), min=1e-12).sqrt()
        probe_z = (
            normalized @ self.probe_embeddings.T - mean[:, None]
        ) / std[:, None]
        refusal = probe_z[:, : self.n_refusal].mean(dim=1)
        compliance = probe_z[:, self.n_refusal :].mean(dim=1)
        return refusal - compliance

    def execute_batch(
        self, observations: Sequence[dict[str, Any]], attempt: int
    ) -> Sequence[Mapping[str, Any]]:
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        prefill_started = time.perf_counter()
        states = self._capture_batch(observations)
        torch.cuda.synchronize()
        prefill_elapsed_ms = (time.perf_counter() - prefill_started) * 1000

        readout_started = time.perf_counter()
        with torch.inference_mode():
            hidden = states[19]
            activations = torch.relu(hidden @ self.encoder.T + self.encoder_bias)
            reconstruction = activations @ self.decoder.T
            if self.decoder_bias is not None:
                reconstruction = reconstruction + self.decoder_bias
            reconstruction_error = (reconstruction - hidden).norm(dim=1) / hidden.norm(
                dim=1
            ).clamp_min(1e-12)
            feature = activations[:, self.feature_id]
            subspace = (
                activations.index_select(1, self.subspace_ids)
                * self.subspace_weights[None, :]
            ).sum(dim=1)
            margins = []
            for layer in self.source_layers:
                transported = states[layer] @ self.jacobians[layer].T
                margins.append(self._probe_margin(transported))
            trajectory = torch.stack(margins, dim=1)
        torch.cuda.synchronize()
        readout_elapsed_ms = (time.perf_counter() - readout_started) * 1000
        peak = int(torch.cuda.max_memory_allocated())
        batch_size = len(observations)
        results = []
        for index, row in enumerate(observations):
            match = self.exact_matcher.inspect(str(row["prompt_text"]))
            if match.prompt_sha256 != row["prompt_sha256"]:
                raise ValueError("weaponization exact matcher prompt hash drift")
            results.append(
                {
                    "trial_id": row["trial_id"],
                    "attempt": attempt,
                    "feature_6779_magnitude": float(feature[index].item()),
                    "frozen_subspace_score": float(subspace[index].item()),
                    "sae_normalized_reconstruction_error": float(
                        reconstruction_error[index].item()
                    ),
                    "jlens_refusal_minus_compliance_trajectory": [
                        float(value) for value in trajectory[index].tolist()
                    ],
                    "restricted_exact_match": match.matched,
                    "structural_metrics": structural_prompt_metrics(str(row["prompt_text"])),
                    "prefill_latency_ms": prefill_elapsed_ms / batch_size,
                    "detector_readout_latency_ms": readout_elapsed_ms / batch_size,
                    "peak_gpu_memory_bytes": peak,
                }
            )
        return results


def run_weaponization_prefill_gpu(
    *,
    public_plan_path: Path,
    private_topology_path: Path,
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
    public_plan = json.loads(public_plan_path.read_text())
    validate_weaponization_plan(public_plan, root=public_plan_path.resolve().parents[1])
    runtime: WeaponizationPrefillRuntime | None = None

    def execute_batch(
        observations: Sequence[dict[str, Any]], attempt: int
    ) -> Sequence[Mapping[str, Any]]:
        nonlocal runtime
        if runtime is None:
            runtime = WeaponizationPrefillRuntime(
                public_plan=public_plan,
                probe_plan_path=probe_plan_path,
                model_path=model_path,
                lens_path=lens_path,
                sae_path=sae_path,
                factorial_material_path=factorial_material_path,
            )
        return runtime.execute_batch(observations, attempt)

    return run_weaponization_prefill_loop(
        public_plan_path=public_plan_path,
        private_topology_path=private_topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id=run_id,
        batch_size=batch_size,
        execute_batch=execute_batch,
    )
