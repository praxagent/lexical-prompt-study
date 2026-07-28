from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .behavior import REFUSAL_PATTERN, _as_token_ids, _peak_memory
from .factorial_authorization import validate_factorial_execution_authorization
from .factorial_plan import validate_factorial_plan
from .factorial_runner import (
    run_factorial_canonical,
    run_factorial_sentinel_repair,
)
from .followup_runner import build_placement_render_pair
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .models import FactorialAssayReceipt

SAE_LAYER = 19
JLENS_LAYER = 30
REFUSAL_PARSER_VERSION = "refusal-regex-v1"


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _atomic_json(path: Path, value: Any, *, mode: int) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == 0o600:
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    temporary.replace(path)
    return sha256_bytes(payload)


def _probe_manifest(probe_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "refusal": probe_plan["probes"]["primary_refusal"],
        "compliance": probe_plan["probes"]["primary_compliance"],
    }


def _verify_probe_tokens(tokenizer: Any, rows: list[dict[str, Any]]) -> list[int]:
    token_ids = []
    for row in rows:
        token_id = int(row["token_id"])
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if sha256_text(token_text) != row["text_sha256"]:
            raise ValueError(f"probe token semantic hash drift for ID {token_id}")
        token_ids.append(token_id)
    return token_ids


def _state_sha256(torch: Any, state: Any) -> str:
    raw = (
        state.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _parser_result(
    generated_text: str,
    *,
    generated_token_count: int,
    finish_reason: str,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "status": "parsed",
        "parser_version": REFUSAL_PARSER_VERSION,
        "nonempty": bool(generated_text.strip()),
        "refusal_detected": bool(REFUSAL_PATTERN.search(generated_text)),
        "generated_token_count": generated_token_count,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "semantic_task_completion_judged": False,
    }


class FactorialCoreRuntime:
    def __init__(
        self,
        *,
        public_plan: dict[str, Any],
        probe_plan_path: Path,
        model_path: str,
        lens_path: Path,
        sae_path: Path,
    ):
        import jlens
        import torch
        import transformers

        self.torch = torch
        self.transformers = transformers
        self.started = time.monotonic()
        artifacts = public_plan["pinned_artifacts"]
        expected_revision = artifacts["model_revision"]
        snapshot = Path(model_path)
        if not snapshot.is_dir() or snapshot.resolve().name != expected_revision:
            raise ValueError("factorial model snapshot revision path drift")
        self.lens_sha256 = sha256_file(lens_path)
        self.sae_sha256 = sha256_file(sae_path)
        if self.lens_sha256 != artifacts["jacobian_lens_sha256"]:
            raise ValueError("factorial J-lens hash drift")
        if self.sae_sha256 != artifacts["sae_sha256"]:
            raise ValueError("factorial SAE hash drift")

        probe_plan = json.loads(probe_plan_path.read_text())
        core = public_plan["core_readout_implementation"]
        if sha256_file(probe_plan_path) != core[
            "jacobian_lens_probe_source_plan_sha256"
        ]:
            raise ValueError("factorial probe source-plan hash drift")
        manifest = _probe_manifest(probe_plan)
        if sha256_bytes(canonical_json_bytes(manifest)) != core[
            "jacobian_lens_probe_manifest_sha256"
        ]:
            raise ValueError("factorial probe manifest hash drift")

        if torch.cuda.device_count() != 1:
            raise ValueError("factorial runtime requires exactly one visible GPU")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        observed_tokenizer_revision = getattr(self.tokenizer, "_commit_hash", None)
        if observed_tokenizer_revision not in (None, expected_revision):
            raise ValueError("factorial tokenizer revision drift")
        self.tokenizer_revision = expected_revision
        refusal_ids = _verify_probe_tokens(
            self.tokenizer, manifest["refusal"]
        )
        compliance_ids = _verify_probe_tokens(
            self.tokenizer, manifest["compliance"]
        )
        self.n_refusal = len(refusal_ids)
        probe_ids = refusal_ids + compliance_ids
        self.probe_ids = torch.tensor(
            probe_ids,
            device="cuda",
            dtype=torch.long,
        )

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
            raise ValueError("factorial Llama 3.1 8B topology drift")
        observed_model_revision = getattr(self.model.config, "_commit_hash", None)
        if observed_model_revision not in (None, expected_revision):
            raise ValueError("factorial model revision drift")
        self.model_revision = expected_revision

        lens = jlens.JacobianLens.load(lens_path)
        source_layers = [int(layer) for layer in lens.source_layers]
        if source_layers != list(range(31)):
            raise ValueError("factorial J-lens source-layer topology drift")
        self.jacobian = lens.jacobians[JLENS_LAYER].to(
            device="cuda", dtype=torch.float32
        )
        if tuple(self.jacobian.shape) != (4096, 4096):
            raise ValueError("factorial layer-30 Jacobian shape drift")

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
            raise ValueError("factorial SAE/model shape drift")
        self.feature_id = int(artifacts["selected_feature_id"])
        self.subspace_ids = torch.tensor(
            artifacts["frozen_subspace_feature_ids"],
            device="cuda",
            dtype=torch.long,
        )
        self.subspace_weights = torch.tensor(
            artifacts["frozen_subspace_weights"],
            device="cuda",
            dtype=torch.float32,
        )

        output_weight = self.model.lm_head.weight.detach().to(
            device="cuda", dtype=torch.float32
        )
        self.norm_weight = self.model.model.norm.weight.detach().to(
            device="cuda", dtype=torch.float32
        )
        effective = output_weight * self.norm_weight[None, :]
        self.mean_embedding = effective.mean(dim=0)
        self.second_moment = effective.T @ effective / effective.shape[0]
        self.probe_embeddings = effective[probe_ids]
        self.norm_epsilon = float(self.model.model.norm.variance_epsilon)
        del effective, output_weight, lens, state

    def _capture_states(self, prompt_token_ids: list[int]) -> dict[int, Any]:
        torch = self.torch
        captured: dict[int, Any] = {}
        handles = []
        for layer in (SAE_LAYER, JLENS_LAYER):
            def hook(_module, _inputs, output, *, layer_index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_index] = hidden[0, -1].detach().float()

            handles.append(
                self.model.model.layers[layer].register_forward_hook(hook)
            )
        tensor = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        try:
            with torch.inference_mode():
                self.model(input_ids=tensor, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if sorted(captured) != [SAE_LAYER, JLENS_LAYER]:
            raise ValueError("factorial assistant-boundary capture incomplete")
        return captured

    def capture_core(self, prompt_token_ids: list[int]) -> dict[str, Any]:
        torch = self.torch
        states = self._capture_states(prompt_token_ids)
        hidden = states[SAE_LAYER]
        with torch.inference_mode():
            activations = torch.relu(hidden @ self.encoder.T + self.encoder_bias)
            reconstruction = activations @ self.decoder.T
            if self.decoder_bias is not None:
                reconstruction = reconstruction + self.decoder_bias
            reconstruction_error = float(
                (
                    (reconstruction - hidden).norm()
                    / hidden.norm().clamp_min(1e-12)
                ).item()
            )
            feature = float(activations[self.feature_id].item())
            subspace = float(
                (
                    activations.index_select(0, self.subspace_ids)
                    * self.subspace_weights
                )
                .sum()
                .item()
            )
            transported = states[JLENS_LAYER] @ self.jacobian.T
            normalized = transported * torch.rsqrt(
                transported.square().mean() + self.norm_epsilon
            )
            mean = normalized @ self.mean_embedding
            second = (normalized @ self.second_moment * normalized).sum()
            std = torch.clamp(second - mean.square(), min=1e-12).sqrt()
            probe_z = (
                normalized @ self.probe_embeddings.T - mean
            ) / std
            refusal = probe_z[: self.n_refusal].mean()
            compliance = probe_z[self.n_refusal :].mean()
            jlens_margin = float((refusal - compliance).item())
        return {
            "feature_6779_magnitude": feature,
            "feature_6779_positive": feature > 0,
            "frozen_subspace_score": subspace,
            "sae_reconstruction_relative_error": reconstruction_error,
            "assistant_boundary_jlens_margin": jlens_margin,
            "state_sha256": {
                str(layer): _state_sha256(torch, value)
                for layer, value in states.items()
            },
        }

    def verify_moment_equivalence(
        self, prompt_token_ids: list[int]
    ) -> dict[str, float]:
        torch = self.torch
        fixture = self._capture_states(prompt_token_ids)[JLENS_LAYER]
        with torch.inference_mode():
            normalized = fixture * torch.rsqrt(
                fixture.square().mean() + self.norm_epsilon
            )
            full_logits = self.model.lm_head.weight.float() @ (
                normalized * self.norm_weight
            )
            full_mean = full_logits.mean()
            full_std = full_logits.std(unbiased=False)
            full_probe_z = (
                full_logits.index_select(0, self.probe_ids) - full_mean
            ) / full_std
            moment_mean = normalized @ self.mean_embedding
            moment_second = (
                normalized @ self.second_moment * normalized
            ).sum()
            moment_std = torch.clamp(
                moment_second - moment_mean.square(), min=1e-12
            ).sqrt()
            moment_probe_z = (
                normalized @ self.probe_embeddings.T - moment_mean
            ) / moment_std
            module_logits = self.model.lm_head(
                self.model.model.norm(
                    fixture.to(dtype=self.model.model.norm.weight.dtype)
                )
            ).float()
            module_probe_z = (
                module_logits.index_select(0, self.probe_ids)
                - module_logits.mean()
            ) / module_logits.std(unbiased=False)
        result = {
            "mean_abs_error": float(abs(full_mean - moment_mean).item()),
            "std_abs_error": float(abs(full_std - moment_std).item()),
            "probe_max_abs_error": float(
                (full_probe_z - moment_probe_z).abs().max().item()
            ),
            "module_probe_max_abs_error": float(
                (module_probe_z - moment_probe_z).abs().max().item()
            ),
            "analytic_tolerance": 0.002,
            "module_tolerance": 0.02,
        }
        result["passed"] = bool(
            max(
                result["mean_abs_error"],
                result["std_abs_error"],
                result["probe_max_abs_error"],
            )
            <= result["analytic_tolerance"]
            and result["module_probe_max_abs_error"]
            <= result["module_tolerance"]
        )
        return result

    def execute_observation(
        self,
        observation: dict[str, Any],
        attempt: int,
        *,
        output_root: Path,
    ) -> dict[str, Any]:
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        readout_started = time.monotonic()
        core = self.capture_core(list(observation["prompt_token_ids"]))
        readout_elapsed = time.monotonic() - readout_started
        tensor = torch.tensor(
            [observation["prompt_token_ids"]],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        generation_started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                tensor,
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generation_elapsed = time.monotonic() - generation_started
        generated_ids = output[0, tensor.shape[1] :].tolist()
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        eos_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        }
        eos_ids.discard(None)
        eos_ids.discard(self.tokenizer.unk_token_id)
        finish_reason = (
            "eos" if generated_ids and generated_ids[-1] in eos_ids else "length"
        )
        truncated = finish_reason == "length"
        parser = _parser_result(
            generated_text,
            generated_token_count=len(generated_ids),
            finish_reason=finish_reason,
            truncated=truncated,
        )
        trial_id = observation["trial_id"]
        restricted_path = (
            output_root
            / "restricted"
            / f"{trial_id}.attempt-{attempt}.private.json"
        )
        restricted_sha256 = _atomic_json(
            restricted_path,
            {
                "schema_version": "1.0",
                "trial_id": trial_id,
                "attempt": attempt,
                "request_class": observation["request_class"],
                "request_id": observation["request_id"],
                "prompt_sha256": observation["prompt_sha256"],
                "prompt_token_ids_sha256": observation[
                    "prompt_token_ids_sha256"
                ],
                "generated_text": generated_text,
                "generated_token_ids": generated_ids,
            },
            mode=0o600,
        )
        print(f"completed factorial trial={trial_id} attempt={attempt}", flush=True)
        return {
            "generated_text_sha256": sha256_text(generated_text),
            "generated_token_ids_sha256": sha256_bytes(
                canonical_json_bytes(generated_ids)
            ),
            "generated_token_count": len(generated_ids),
            "finish_reason": finish_reason,
            "truncated": truncated,
            "feature_6779_magnitude": core["feature_6779_magnitude"],
            "feature_6779_positive": core["feature_6779_positive"],
            "frozen_subspace_score": core["frozen_subspace_score"],
            "sae_reconstruction_relative_error": core[
                "sae_reconstruction_relative_error"
            ],
            "assistant_boundary_jlens_margin": core[
                "assistant_boundary_jlens_margin"
            ],
            "refusal_detected": parser["refusal_detected"],
            "task_completion": None,
            "utility_score": None,
            "phase_parser_result": parser,
            "restricted_artifact_path": str(restricted_path),
            "restricted_artifact_sha256": restricted_sha256,
            "generation_elapsed_seconds": generation_elapsed,
            "readout_elapsed_seconds": readout_elapsed,
            "peak_memory_bytes": _peak_memory(torch),
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "lens_sha256": self.lens_sha256,
            "sae_sha256": self.sae_sha256,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": self.transformers.__version__,
                "cuda": torch.version.cuda,
                "attention": "eager",
                "storage_dtype": "torch.bfloat16",
                "readout_dtype": "torch.float32",
            },
        }


def _legacy_rows(
    *,
    public_plan: dict[str, Any],
    legacy_private: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {
        row["behavior_id"]: row
        for row in legacy_private["partitions"]["discovery"]
    }
    selected = []
    for frozen in public_plan["assay_validity_gate"]["legacy_request_selection"]:
        row = dict(by_id[frozen["behavior_id"]])
        if (
            row["request_sha256"] != frozen["request_sha256"]
            or sha256_text(row["request"]) != frozen["request_sha256"]
        ):
            raise ValueError("factorial legacy canary request hash drift")
        selected.append(row)
    if len(selected) != 2:
        raise ValueError("factorial legacy canary topology drift")
    return selected


def run_factorial_assay(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    legacy_private_path: Path,
    authorization_path: Path,
    probe_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    public_plan = json.loads(public_plan_path.read_text())
    validate_factorial_plan(public_plan)
    private_plan = json.loads(private_plan_path.read_text())
    public_sha256 = sha256_file(public_plan_path)
    private_sha256 = sha256_file(private_plan_path)
    source_commit = _source_commit()
    authorization = json.loads(authorization_path.read_text())
    validate_factorial_execution_authorization(
        authorization,
        expected_public_plan_sha256=public_sha256,
        expected_private_plan_sha256=private_sha256,
        expected_source_commit=source_commit,
        expected_stage="assay_canary",
    )
    if authorization["run_id"] != run_id:
        raise ValueError("factorial assay authorization run-ID drift")

    public_receipt_path = output_root / "assay.public.json"
    if public_receipt_path.exists():
        receipt = FactorialAssayReceipt.model_validate_json(
            public_receipt_path.read_text()
        )
        if (
            receipt.public_plan_sha256 != public_sha256
            or receipt.private_plan_sha256 != private_sha256
            or receipt.source_commit != source_commit
            or receipt.run_id != run_id
        ):
            raise ValueError("factorial assay resume provenance drift")
        private_bundle = Path(receipt.private_bundle_path)
        if sha256_file(private_bundle) != receipt.private_bundle_sha256:
            raise ValueError("factorial assay private bundle drift")
        return receipt.model_dump(mode="json")

    legacy_private = json.loads(legacy_private_path.read_text())
    legacy_rows = _legacy_rows(
        public_plan=public_plan,
        legacy_private=legacy_private,
    )
    full_scaffold = private_plan["material_block_joiner"].join(
        block["text"] for block in private_plan["material_blocks"]["full_scaffold"]
    )
    runtime = FactorialCoreRuntime(
        public_plan=public_plan,
        probe_plan_path=probe_plan_path,
        model_path=model_path,
        lens_path=lens_path,
        sae_path=sae_path,
    )
    records = []
    render_checks = []
    moment_equivalence = None
    for row in legacy_rows:
        request = row["request"]
        messages = [{"role": "user", "content": request}]
        base_ids = _as_token_ids(
            runtime.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        render_public, render_private = build_placement_render_pair(
            tokenizer=runtime.tokenizer,
            request=request,
            scaffold=full_scaffold,
            tokenizer_revision=runtime.tokenizer_revision,
            context_ceiling=8192,
            generation_budget=1024,
        )
        render_checks.append(render_public)
        if moment_equivalence is None:
            moment_equivalence = runtime.verify_moment_equivalence(base_ids)
        base_a = runtime.capture_core(base_ids)
        before = runtime.capture_core(
            render_private["ep_before_request"]["prompt_token_ids"]
        )
        base_b = runtime.capture_core(base_ids)
        after = runtime.capture_core(
            render_private["ep_after_request"]["prompt_token_ids"]
        )
        records.append(
            {
                "behavior_id": row["behavior_id"],
                "request_sha256": row["request_sha256"],
                "base_a": base_a,
                "full_before": before,
                "base_b": base_b,
                "full_after": after,
            }
        )

    deterministic = all(
        row["base_a"]["state_sha256"] == row["base_b"]["state_sha256"]
        and row["base_a"]["feature_6779_magnitude"]
        == row["base_b"]["feature_6779_magnitude"]
        and row["base_a"]["assistant_boundary_jlens_margin"]
        == row["base_b"]["assistant_boundary_jlens_margin"]
        for row in records
    )
    qualitative = {
        "ep_before_request": (
            sum(row["full_before"]["feature_6779_magnitude"] for row in records)
            > sum(row["base_a"]["feature_6779_magnitude"] for row in records)
        ),
        "ep_after_request": (
            sum(row["full_after"]["feature_6779_magnitude"] for row in records)
            > sum(row["base_b"]["feature_6779_magnitude"] for row in records)
        ),
    }
    reconstruction_errors = [
        metrics["sae_reconstruction_relative_error"]
        for row in records
        for metrics in (
            row["base_a"],
            row["full_before"],
            row["base_b"],
            row["full_after"],
        )
    ]
    assay = public_plan["assay_validity_gate"]
    maximum_reconstruction = max(reconstruction_errors)
    reconstruction_range = max(reconstruction_errors) - min(reconstruction_errors)
    render_passed = all(
        check["status"] == "passed"
        and check["component_token_counts"]["scaffold"] == 252
        and check["tokenizer_revision"] == runtime.tokenizer_revision
        and check["total_tokens"] > 0
        and all(len(value) == 64 for value in check["offset_map_sha256"].values())
        for check in render_checks
    )
    status = (
        "passed"
        if deterministic
        and moment_equivalence is not None
        and moment_equivalence["passed"]
        and all(qualitative.values())
        and render_passed
        and maximum_reconstruction
        <= assay["reconstruction_absolute_tolerance"]
        and reconstruction_range
        <= assay["reconstruction_cross_condition_tolerance"]
        else "failed"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    private_bundle_path = output_root / "private" / "assay-bundle.private.json"
    private_bundle_sha256 = _atomic_json(
        private_bundle_path,
        {
            "schema_version": "1.0",
            "study_id": public_plan["study_id"],
            "public_plan_sha256": public_sha256,
            "private_plan_sha256": private_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
            "records": records,
            "moment_equivalence": moment_equivalence,
            "render_validation_sha256": [
                sha256_bytes(canonical_json_bytes(check))
                for check in render_checks
            ],
        },
        mode=0o600,
    )
    payload = {
        "schema_version": "1.0",
        "study_id": public_plan["study_id"],
        "status": status,
        "qualification_kind": "noninferential_legacy_canary",
        "public_plan_sha256": public_sha256,
        "private_plan_sha256": private_sha256,
        "source_commit": source_commit,
        "run_id": run_id,
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "lens_sha256": runtime.lens_sha256,
        "sae_sha256": runtime.sae_sha256,
        "selected_feature_id": 6779,
        "frozen_subspace_feature_ids": public_plan["pinned_artifacts"][
            "frozen_subspace_feature_ids"
        ],
        "planned_canary_conditions": 8,
        "completed_canary_conditions": 8,
        "exact_identity_checks": {
            name: (
                bool(moment_equivalence and moment_equivalence["passed"])
                if name == "vocabulary_moment_equivalence"
                else True
            )
            for name in assay["exact_identity_checks"]
        },
        "final_render_checks": {
            name: render_passed for name in assay["final_render_checks"]
        },
        "deterministic_rerun_passed": deterministic,
        "reconstruction_metric": assay["reconstruction_metric"],
        "reconstruction_absolute_tolerance": assay[
            "reconstruction_absolute_tolerance"
        ],
        "reconstruction_cross_condition_tolerance": assay[
            "reconstruction_cross_condition_tolerance"
        ],
        "maximum_reconstruction_error": maximum_reconstruction,
        "reconstruction_error_range": reconstruction_range,
        "qualitative_ordering_passed_by_placement": qualitative,
        "private_bundle_path": str(private_bundle_path),
        "private_bundle_sha256": private_bundle_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": _peak_memory(runtime.torch),
        "target_factorial_outcome_generated": False,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": runtime.torch.__version__,
            "transformers": runtime.transformers.__version__,
            "cuda": runtime.torch.version.cuda,
            "attention": "eager",
        },
    }
    validated = FactorialAssayReceipt.model_validate(payload)
    _atomic_json(
        public_receipt_path,
        validated.model_dump(mode="json"),
        mode=0o644,
    )
    return validated.model_dump(mode="json")


def run_factorial_canonical_gpu(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    assay_receipt_path: Path,
    authorization_path: Path,
    probe_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_factorial_plan(public_plan)
    runtime: FactorialCoreRuntime | None = None

    def execute(observation: dict[str, Any], attempt: int) -> dict[str, Any]:
        nonlocal runtime
        if runtime is None:
            runtime = FactorialCoreRuntime(
                public_plan=public_plan,
                probe_plan_path=probe_plan_path,
                model_path=model_path,
                lens_path=lens_path,
                sae_path=sae_path,
            )
        return runtime.execute_observation(
            observation,
            attempt,
            output_root=output_root,
        )

    summary = run_factorial_canonical(
        public_plan_path=public_plan_path,
        private_plan_path=private_plan_path,
        assay_receipt_path=assay_receipt_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id=run_id,
        execute_observation=execute,
    )
    summary["model_loaded_this_call"] = runtime is not None
    _atomic_json(output_root / "summary.json", summary, mode=0o600)
    return summary


def run_factorial_sentinel_repair_gpu(
    *,
    public_plan_path: Path,
    private_plan_path: Path,
    assay_receipt_path: Path,
    matrix_receipt_root: Path,
    authorization_path: Path,
    probe_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_factorial_plan(public_plan)
    runtime: FactorialCoreRuntime | None = None

    def execute(observation: dict[str, Any], attempt: int) -> dict[str, Any]:
        nonlocal runtime
        if runtime is None:
            runtime = FactorialCoreRuntime(
                public_plan=public_plan,
                probe_plan_path=probe_plan_path,
                model_path=model_path,
                lens_path=lens_path,
                sae_path=sae_path,
            )
        return runtime.execute_observation(
            observation,
            attempt,
            output_root=output_root,
        )

    summary = run_factorial_sentinel_repair(
        public_plan_path=public_plan_path,
        private_plan_path=private_plan_path,
        assay_receipt_path=assay_receipt_path,
        matrix_receipt_root=matrix_receipt_root,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id=run_id,
        execute_observation=execute,
    )
    summary["model_loaded_this_call"] = runtime is not None
    _atomic_json(output_root / "summary.json", summary, mode=0o600)
    return summary
