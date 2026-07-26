from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lexical_prompt_study.factorial_authorization import (
    validate_factorial_execution_authorization,
)
from lexical_prompt_study.factorial_runner import run_factorial_canonical
from lexical_prompt_study.hashing import sha256_file, write_json_atomic
from lexical_prompt_study.models import FactorialAssayReceipt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PLAN = ROOT / "plans" / "factorial_8b_v1.public.json"


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _private_plan(tmp_path: Path) -> Path:
    observations = []
    for class_index, request_class in enumerate(
        (
            "minimal_neutral_carrier",
            "ordinary_benign_request",
            "harmful_request",
        )
    ):
        for request_index in range(20):
            common = {
                "request_class": request_class,
                "request_id": f"R{class_index}-{request_index:02d}",
                "prompt_family_id": f"F{class_index}-{request_index:02d}",
                "request_sha256": f"{class_index + 1:x}" * 64,
            }
            observations.append(
                {
                    **common,
                    "trial_id": f"base-{class_index}-{request_index:02d}",
                    "material": "no_scaffold",
                    "placement": None,
                    "size_id": "natural-base",
                    "injected_token_count": 0,
                    "shared_reference": True,
                    "prompt_sha256": "a" * 64,
                    "prompt_token_ids_sha256": "b" * 64,
                }
            )
            for material_index, material in enumerate(
                ("inert_length", "structural_sham", "full_scaffold")
            ):
                for placement_index, placement in enumerate(
                    ("ep_before_request", "ep_after_request")
                ):
                    observations.append(
                        {
                            **common,
                            "trial_id": (
                                f"inj-{class_index}-{request_index:02d}-"
                                f"{material_index}-{placement_index}"
                            ),
                            "material": material,
                            "placement": placement,
                            "size_id": "blocks-004",
                            "injected_token_count": 19,
                            "shared_reference": False,
                            "render_group_sha256": "c" * 64,
                            "prompt_sha256": "d" * 64,
                            "prompt_token_ids_sha256": "e" * 64,
                        }
                    )
    for placement_index, placement in enumerate(
        ("ep_before_request", "ep_after_request")
    ):
        observations.append(
            {
                "request_class": "literal_sentinel",
                "request_id": "literal-sentinel",
                "prompt_family_id": "literal-sentinel-descriptive-n1",
                "request_sha256": "f" * 64,
                "trial_id": f"sentinel-{placement_index}",
                "material": "full_scaffold",
                "placement": placement,
                "size_id": "blocks-004",
                "injected_token_count": 19,
                "shared_reference": False,
                "render_group_sha256": "1" * 64,
                "prompt_sha256": "2" * 64,
                "prompt_token_ids_sha256": "3" * 64,
            }
        )
    assert len(observations) == 422
    path = tmp_path / "factorial.private.json"
    write_json_atomic(
        path,
        {
            "schema_version": "1.0",
            "study_id": "lexical-scaffold-8b-factorial-v1",
            "source_commit": _source_commit(),
            "public_plan_sha256": sha256_file(PUBLIC_PLAN),
            "doses": [{"size_id": "blocks-004", "canonical": True}],
            "observations": observations,
        },
    )
    return path


def _assay_receipt(tmp_path: Path, private_path: Path) -> Path:
    bundle = tmp_path / "assay.private.json"
    bundle_sha256 = write_json_atomic(bundle, {"safe": True})
    payload = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-8b-factorial-v1",
        "status": "passed",
        "qualification_kind": "noninferential_legacy_canary",
        "public_plan_sha256": sha256_file(PUBLIC_PLAN),
        "private_plan_sha256": sha256_file(private_path),
        "source_commit": _source_commit(),
        "run_id": "assay-safe",
        "model_revision": "1" * 40,
        "tokenizer_revision": "1" * 40,
        "lens_sha256": "2" * 64,
        "sae_sha256": "3" * 64,
        "selected_feature_id": 6779,
        "frozen_subspace_feature_ids": [
            1980,
            6779,
            11954,
            20449,
            35705,
            43596,
            53185,
            58843,
        ],
        "planned_canary_conditions": 8,
        "completed_canary_conditions": 8,
        "exact_identity_checks": {"all": True},
        "final_render_checks": {"all": True},
        "deterministic_rerun_passed": True,
        "reconstruction_metric": "normalized_l2",
        "reconstruction_absolute_tolerance": 0.5,
        "reconstruction_cross_condition_tolerance": 0.05,
        "maximum_reconstruction_error": 0.2,
        "reconstruction_error_range": 0.01,
        "qualitative_ordering_passed_by_placement": {
            "ep_before_request": True,
            "ep_after_request": True,
        },
        "private_bundle_path": str(bundle),
        "private_bundle_sha256": bundle_sha256,
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 1024,
        "target_factorial_outcome_generated": False,
        "software": {"python": "test"},
    }
    FactorialAssayReceipt.model_validate(payload)
    path = tmp_path / "assay.public.json"
    write_json_atomic(path, payload)
    return path


def _authorization(
    tmp_path: Path,
    private_path: Path,
    assay_path: Path,
    *,
    soft_gate_approved: bool = True,
) -> tuple[Path, dict]:
    payload = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-8b-factorial-v1",
        "status": "prospective_factorial_canonical_authorization",
        "stage": "canonical_factorial",
        "run_id": "canonical-safe",
        "bindings": {
            "public_plan_sha256": sha256_file(PUBLIC_PLAN),
            "private_plan_sha256": sha256_file(private_path),
            "assay_receipt_sha256": sha256_file(assay_path),
            "canonical_result_sha256": None,
            "source_commit": _source_commit(),
        },
        "scope": {
            "target_factorial_outcomes_authorized": True,
            "planned_conditions": 422,
            "placement_pooling": False,
            "size_pooling": False,
            "detector_threshold_fitting": False,
            "completed_receipt_overwrite": False,
        },
        "provider": {
            "maximum_task_owned_pods": 1,
            "gpu_count": 1,
            "secure_cloud": True,
            "persistent_volume_id": "u85xfo0aue",
            "fallback_allowed": False,
        },
        "cost": {
            "maximum_live_rate_usd_per_hour": 5.98,
            "wall_time_minutes": 60,
            "maximum_compute_usd": 5.98,
            "conservative_pre_run_ceiling_usd": 99.0,
            "conservative_post_run_ceiling_usd": 104.98,
            "renewed_human_soft_gate_approval": soft_gate_approved,
            "no_progress_stop_minutes": 10,
        },
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_factorial_authorization_rejects_soft_gate_crossing_without_approval(
    tmp_path: Path,
) -> None:
    private_path = _private_plan(tmp_path)
    assay_path = _assay_receipt(tmp_path, private_path)
    _, payload = _authorization(
        tmp_path,
        private_path,
        assay_path,
        soft_gate_approved=False,
    )
    with pytest.raises(ValueError, match="soft gate"):
        validate_factorial_execution_authorization(
            payload,
            expected_public_plan_sha256=sha256_file(PUBLIC_PLAN),
            expected_private_plan_sha256=sha256_file(private_path),
            expected_source_commit=_source_commit(),
            expected_stage="canonical_factorial",
        )


def test_factorial_runner_writes_atomic_receipts_and_resumes(
    tmp_path: Path,
) -> None:
    private_path = _private_plan(tmp_path)
    assay_path = _assay_receipt(tmp_path, private_path)
    authorization_path, _ = _authorization(tmp_path, private_path, assay_path)
    calls = 0

    def execute(observation, attempt):
        nonlocal calls
        calls += 1
        restricted = tmp_path / "restricted" / f"{observation['trial_id']}.json"
        restricted.parent.mkdir(exist_ok=True)
        restricted_sha256 = write_json_atomic(
            restricted,
            {"generated_text": "safe", "generated_token_ids": [1]},
        )
        benign = observation["request_class"] == "ordinary_benign_request"
        return {
            "generated_text_sha256": "4" * 64,
            "generated_token_ids_sha256": "5" * 64,
            "generated_token_count": 1,
            "finish_reason": "eos",
            "truncated": False,
            "feature_6779_magnitude": 0.1,
            "feature_6779_positive": True,
            "frozen_subspace_score": 0.2,
            "sae_reconstruction_relative_error": 0.1,
            "assistant_boundary_jlens_margin": -0.1,
            "refusal_detected": False,
            "task_completion": True if benign else None,
            "utility_score": 1.0 if benign else None,
            "phase_parser_result": {"status": "safe"},
            "restricted_artifact_path": str(restricted),
            "restricted_artifact_sha256": restricted_sha256,
            "generation_elapsed_seconds": 0.01,
            "readout_elapsed_seconds": 0.01,
            "peak_memory_bytes": 1,
            "model_revision": "6" * 40,
            "tokenizer_revision": "6" * 40,
            "lens_sha256": "7" * 64,
            "sae_sha256": "8" * 64,
            "software": {"python": "test"},
        }

    first = run_factorial_canonical(
        public_plan_path=PUBLIC_PLAN,
        private_plan_path=private_path,
        assay_receipt_path=assay_path,
        authorization_path=authorization_path,
        output_root=tmp_path / "run",
        run_id="canonical-safe",
        execute_observation=execute,
    )
    assert first["receipts_written_this_call"] == 422
    assert calls == 422
    second = run_factorial_canonical(
        public_plan_path=PUBLIC_PLAN,
        private_plan_path=private_path,
        assay_receipt_path=assay_path,
        authorization_path=authorization_path,
        output_root=tmp_path / "run",
        run_id="canonical-safe",
        execute_observation=execute,
    )
    assert second["preexisting_receipts"] == 422
    assert second["receipts_written_this_call"] == 0
    assert calls == 422


def test_factorial_runner_rejects_executor_provenance_override(
    tmp_path: Path,
) -> None:
    private_path = _private_plan(tmp_path)
    assay_path = _assay_receipt(tmp_path, private_path)
    authorization_path, _ = _authorization(tmp_path, private_path, assay_path)

    def execute(_observation, _attempt):
        return {"run_id": "drift"}

    with pytest.raises(ValueError, match="override provenance"):
        run_factorial_canonical(
            public_plan_path=PUBLIC_PLAN,
            private_plan_path=private_path,
            assay_receipt_path=assay_path,
            authorization_path=authorization_path,
            output_root=tmp_path / "run",
            run_id="canonical-safe",
            execute_observation=execute,
        )
