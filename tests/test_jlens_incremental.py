from __future__ import annotations

import json

import numpy as np
import pytest

from lexical_prompt_study.hashing import sha256_file, sha256_text
from lexical_prompt_study import jlens_incremental_runner as runner
from lexical_prompt_study.jlens_incremental_analysis import (
    _candidate_gate,
    _folds,
    _nested_candidate,
    _select_threshold,
)
from lexical_prompt_study.jlens_incremental_runner import (
    _authorization,
    _validate_receipt,
)
from lexical_prompt_study.jlens_incremental_topology import (
    EXPECTED_OBSERVATIONS,
    SAFE_EXPECTED,
    frame_request,
)


def test_public_topology_receipt_has_complete_paired_factorial() -> None:
    receipt = json.loads(
        open("validation/jlens-incremental-a139.topology.public.json").read()
    )
    assert receipt["observation_count"] == EXPECTED_OBSERVATIONS == 8880
    assert receipt["generation_count"] == 8880
    assert receipt["request_core_count"] == 60
    assert receipt["intent_frame_counts"] == {
        "safe_acknowledge_exact": 2220,
        "safe_classify_exact": 2220,
        "safe_refuse_exact": 2220,
        "unsafe_direct": 2220,
    }
    assert receipt["variant_family_counts"] == {
        "attack_block_mask": 7680,
        "harmless_structured_wrapper": 480,
        "no_scaffold": 240,
        "structural_sham": 480,
    }
    assert receipt["injected_token_counts"] == {"0": 240, "252": 8640}
    assert receipt["all_mutated_injections_exactly_252_tokens"] is True
    assert receipt["target_outcome_exists"] is False


@pytest.mark.parametrize("intent", sorted(SAFE_EXPECTED))
def test_safe_frames_preserve_core_once_and_bind_exact_utility(intent: str) -> None:
    core = "Synthetic restricted request placeholder 314159."
    framed, expected = frame_request(core, intent)
    assert framed.count(core) == 1
    assert expected == SAFE_EXPECTED[intent]
    assert "Do not" in framed
    assert framed.endswith(expected)


def test_unsafe_frame_changes_no_request_bytes() -> None:
    core = "Synthetic request bytes."
    framed, expected = frame_request(core, "unsafe_direct")
    assert framed == core
    assert expected is None


def _checkpoint(token_count: int) -> dict[str, object]:
    return {
        "prefix_token_count": token_count,
        "prefix_token_ids_sha256": "a" * 64,
        "feature_6779_magnitude": 0.1,
        "frozen_subspace_score": 0.2,
        "sae_normalized_reconstruction_error": 0.3,
        "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        "prefill_latency_ms": 1.0,
        "detector_readout_latency_ms": 0.5,
        "peak_gpu_memory_bytes": 1024,
    }


def _observation() -> dict[str, object]:
    expected = SAFE_EXPECTED["safe_classify_exact"]
    return {
        "trial_id": "trial",
        "request_core_id": "core",
        "request_core_sha256": "b" * 64,
        "intent_frame": "safe_classify_exact",
        "safe_intent": True,
        "utility_expected_sha256": sha256_text(expected),
        "variant_id": "attack-mask-0111",
        "variant_family": "attack_block_mask",
        "attack_block_mask": 7,
        "attack_block_count": 3,
        "material_sha256": "c" * 64,
        "wrapper_id": None,
        "placement": "scaffold_after_request",
        "injected_token_count": 252,
        "prompt_sha256": "d" * 64,
        "prompt_token_ids_sha256": "e" * 64,
    }


def _receipt() -> dict[str, object]:
    return {
        **_observation(),
        "readouts": {
            "0": _checkpoint(0),
            "1": _checkpoint(1),
            "4": _checkpoint(4),
            "8": None,
        },
        "generated_token_count": 5,
        "generated_text_sha256": "f" * 64,
        "generated_token_ids_sha256": "1" * 64,
        "restricted_artifact_sha256": "2" * 64,
        "utility_exact_match": True,
    }


def test_incremental_receipt_requires_every_available_early_checkpoint() -> None:
    _validate_receipt(_receipt(), _observation())
    broken = _receipt()
    broken["readouts"]["4"] = None
    with pytest.raises(ValueError, match="available checkpoint missing"):
        _validate_receipt(broken, _observation())


def test_request_core_folds_never_split_variants() -> None:
    ids = [core for core in ("a", "b", "c", "d", "e", "f") for _ in range(37)]
    hashes = [f"{ord(core):064x}" for core in ("a", "b", "c", "d", "e", "f") for _ in range(37)]
    folds = _folds(ids, hashes, 5)
    for core in set(ids):
        assert len(set(folds[np.asarray(ids) == core])) == 1


def test_threshold_respects_safe_false_trip_constraint() -> None:
    predictions = np.asarray([0.9, 0.8, 0.7, 0.6, 0.1])
    labels = np.asarray([True, True, False, False, False])
    safe_negative = np.asarray([False, False, True, True, True])
    selected = _select_threshold(predictions, labels, safe_negative)
    assert selected["successful_harmful_capture_fraction"] == 1.0
    assert selected["safe_negative_false_trip_fraction"] == 0.0


def test_nested_candidate_selects_threshold_without_request_leakage() -> None:
    core_ids = [f"core-{index}" for index in range(10) for _ in range(2)]
    core_hashes = [f"{index:064x}" for index in range(10) for _ in range(2)]
    labels = np.asarray([value for _ in range(10) for value in (True, False)])
    safe_negative = ~labels
    matrix = labels.astype(float)[:, None]
    folds = _folds(core_ids, core_hashes, 5)
    result = _nested_candidate(
        matrix,
        labels,
        safe_negative,
        folds,
        np.ones(len(labels), dtype=bool),
    )
    assert np.all(result["trips"] == labels)


def test_incremental_gate_requires_material_prompt_advantage() -> None:
    metrics = {
        "successful_harmful_capture_fraction": 0.9,
        "safe_negative_false_trip_fraction": 0.01,
        "by_placement": {
            "before": {"successful_harmful_capture_fraction": 0.85},
            "after": {"successful_harmful_capture_fraction": 0.9},
        },
        "by_safe_intent_frame": {
            "a": {"false_trip_fraction": 0.01},
            "b": {"false_trip_fraction": 0.02},
            "c": {"false_trip_fraction": 0.01},
        },
        "block4_replacement_hard_negative": {"capture_fraction": 0.85},
    }
    weak = {
        "observed_absolute_capture_improvement": 0.01,
        "cluster_bootstrap_95_low": -0.01,
    }
    assert _candidate_gate(metrics, weak)["eligible"] is False
    strong = {
        "observed_absolute_capture_improvement": 0.08,
        "cluster_bootstrap_95_low": 0.02,
    }
    assert _candidate_gate(metrics, strong)["eligible"] is True


def test_authorization_rejects_excess_compute(tmp_path) -> None:
    payload = {
        "schema_version": "1.0",
        "study_id": "lexical-jlens-incremental-value-v1",
        "status": "incremental_value_execution_authorized",
        "paid_compute_authorized": True,
        "authorized_phases": ["acquire", "score"],
        "run_id": "run",
        "expected_observations": EXPECTED_OBSERVATIONS,
        "expected_generations": EXPECTED_OBSERVATIONS,
        "generation_checkpoints": [0, 1, 4, 8],
        "single_task_owned_pod_maximum": 1,
        "maximum_new_compute_usd": 25.01,
        "legacy_hard_ceiling_usd": 200,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
        "bindings": {
            "source_commit": "a" * 40,
            "plan_sha256": "b" * 64,
            "private_topology_sha256": "c" * 64,
        },
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="authorization drift"):
        _authorization(
            path=path,
            run_id="run",
            source_commit="a" * 40,
            plan_sha256="b" * 64,
            topology_sha256="c" * 64,
            phase="acquire",
        )


class _FakeTokenizer:
    @staticmethod
    def decode(tokens, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(f"<{token}>" for token in tokens)


class _FakeTorch:
    __version__ = "fake"

    class version:
        cuda = "fake"


class _FakeRuntime:
    def __init__(self, **kwargs):
        del kwargs
        self.tokenizer = _FakeTokenizer()
        self.torch = _FakeTorch()

    @staticmethod
    def execute_batch(rows, attempt):
        del attempt
        return [
            {
                "trial_id": row["trial_id"],
                "feature_6779_magnitude": 0.1,
                "frozen_subspace_score": 0.2,
                "sae_normalized_reconstruction_error": 0.3,
                "jlens_refusal_minus_compliance_trajectory": [0.01] * 31,
                "restricted_exact_match": row["variant_family"] == "attack_block_mask",
                "structural_metrics": {
                    "prompt_bytes": 100.0,
                    "line_count": 2.0,
                    "dash_divider_count": 0.0,
                    "equals_divider_count": 0.0,
                    "markdown_heading_count": 0.0,
                    "bracket_count": 0.0,
                },
                "prefill_latency_ms": 1.0,
                "detector_readout_latency_ms": 0.5,
                "peak_gpu_memory_bytes": 1024,
            }
            for row in rows
        ]


def _runner_observation(index: int) -> dict[str, object]:
    safe = index == 0
    expected = SAFE_EXPECTED["safe_classify_exact"] if safe else None
    public = {
        "trial_id": f"trial-{index}",
        "request_core_id": f"core-{index}",
        "request_core_sha256": f"{index + 3:064x}",
        "intent_frame": "safe_classify_exact" if safe else "unsafe_direct",
        "safe_intent": safe,
        "utility_expected_sha256": sha256_text(expected) if expected else None,
        "variant_id": "attack-mask-1111",
        "variant_family": "attack_block_mask",
        "attack_block_mask": 15,
        "attack_block_count": 4,
        "material_sha256": "c" * 64,
        "wrapper_id": None,
        "placement": "scaffold_after_request",
        "injected_token_count": 252,
        "prompt_sha256": sha256_text(f"prompt-{index}"),
        "prompt_token_ids_sha256": "e" * 64,
    }
    return {
        **public,
        "request_core_text": f"core text {index}",
        "framed_request_text": f"frame text {index}",
        "utility_expected_text": expected,
        "prompt_text": f"prompt-{index}",
        "prompt_token_ids": [10, 11, index + 12],
    }


def test_acquisition_replays_early_prefixes_and_resumes_atomically(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_OBSERVATIONS", 2)
    monkeypatch.setattr(runner, "WeaponizationPrefillRuntime", _FakeRuntime)
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)

    def fake_generate(runtime, rows):
        del runtime
        output = []
        for row in rows:
            expected = row["utility_expected_text"]
            text = expected if expected is not None else "synthetic refusal"
            output.append(
                {
                    "trial_id": row["trial_id"],
                    "generated_token_ids": list(range(9)),
                    "generated_text": text,
                    "finish_reason": "eos",
                    "generation_elapsed_ms": 2.0,
                }
            )
        return output

    monkeypatch.setattr(runner, "_generate", fake_generate)
    plan = tmp_path / "plan.json"
    topology = tmp_path / "topology.json"
    authorization = tmp_path / "authorization.json"
    instrument = tmp_path / "instrument.json"
    for path in (plan, instrument):
        path.write_text("{}")
    topology.write_text(
        json.dumps(
            {
                "study_id": "lexical-jlens-incremental-value-v1",
                "status": "incremental_value_topology_frozen_no_target_outcomes",
                "generation_checkpoints": [0, 1, 4, 8],
                "enforcement_enabled": False,
                "unopened_v2_confirmation_opened": False,
                "observations": [_runner_observation(0), _runner_observation(1)],
            }
        )
    )
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "lexical-jlens-incremental-value-v1",
                "status": "incremental_value_execution_authorized",
                "paid_compute_authorized": True,
                "authorized_phases": ["acquire", "score"],
                "run_id": "synthetic-run",
                "expected_observations": 2,
                "expected_generations": 2,
                "generation_checkpoints": [0, 1, 4, 8],
                "single_task_owned_pod_maximum": 1,
                "maximum_new_compute_usd": 1,
                "legacy_hard_ceiling_usd": 200,
                "enforcement_enabled": False,
                "unopened_v2_confirmation_opened": False,
                "bindings": {
                    "source_commit": "a" * 40,
                    "plan_sha256": sha256_file(plan),
                    "private_topology_sha256": sha256_file(topology),
                },
            }
        )
    )
    kwargs = {
        "plan_path": plan,
        "instrument_plan_path": instrument,
        "topology_path": topology,
        "authorization_path": authorization,
        "probe_plan_path": tmp_path / "probe.json",
        "model_path": "unused",
        "lens_path": tmp_path / "lens",
        "sae_path": tmp_path / "sae",
        "factorial_material_path": tmp_path / "material.json",
        "output_root": tmp_path / "output",
        "run_id": "synthetic-run",
        "batch_size": 2,
    }
    first = runner.run_acquisition(**kwargs)
    assert first["written_this_invocation"] == 2
    receipt = json.loads(
        (tmp_path / "output" / "receipts" / "trial-0.json").read_text()
    )
    assert list(receipt["readouts"]) == ["0", "1", "4", "8"]
    assert receipt["utility_exact_match"] is True
    second = runner.run_acquisition(**kwargs)
    assert second["written_this_invocation"] == 0
    assert second["resumed_receipt_count"] == 2
