from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lexical_prompt_study import continuation_runner as runner
from lexical_prompt_study.continuation import STUDY_ID, token_hash, write_immutable_json
from lexical_prompt_study.hashing import sha256_file, sha256_text
from lexical_prompt_study.hashing import canonical_json_bytes, sha256_bytes


def _auth(scope="pilot"):
    return {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "continuation_execution_authorized",
        "paid_compute_authorized": True,
        "authorized_phases": ["acquire", "score"],
        "scope": scope,
        "run_id": "synthetic",
        "source_commit": "a" * 40,
        "bindings": {"plan": "b" * 64},
        "single_task_owned_pod_maximum": 1,
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
        "raw_content_public": False,
        "budget_authority_id": "a148-user-70",
        "execution_protocol": "a148-batch4",
        "batch_size": 4,
        "round_spent_upper_usd": 2,
        "pilot_spent_upper_usd": 2,
        "maximum_new_compute_usd": 68,
        "maximum_pilot_compute_usd": 6,
        "hourly_rate_usd": 1.5,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }


def test_authorization_pilot_cannot_execute_full():
    kwargs = dict(
        phase="acquire", run_id="synthetic", source_commit="a" * 40, bindings={"plan": "b" * 64}
    )
    assert runner.validate_authorization(_auth(), scope="pilot", **kwargs) > time.monotonic()
    with pytest.raises(ValueError, match="authorization drift"):
        runner.validate_authorization(_auth(), scope="full", **kwargs)


def test_authorization_requires_budget_and_full_pilot_gate():
    value = _auth("full")
    kwargs = dict(
        phase="acquire",
        scope="full",
        run_id="synthetic",
        source_commit="a" * 40,
        bindings={"plan": "b" * 64},
    )
    with pytest.raises(ValueError, match="accepted budgeted pilot"):
        runner.validate_authorization(value, **kwargs)
    value["pilot_gate"] = {
        "accepted": True,
        "projected_remaining_compute_usd": 5,
        "remaining_authorized_compute_usd": 13,
        "pilot_acquisition_summary_sha256": "c" * 64,
        "pilot_scoring_summary_sha256": "d" * 64,
        "cost_projection_sha256": "e" * 64,
        "pilot_completed_count": 64,
        "pilot_scored_count": 64,
        "batch_size": 4,
    }
    assert runner.validate_authorization(value, **kwargs) > time.monotonic()
    value["maximum_new_compute_usd"] = 69
    with pytest.raises(ValueError, match="approved ceiling"):
        runner.validate_authorization(value, **kwargs)


def test_expired_deadline_rejected():
    with pytest.raises(TimeoutError):
        runner._deadline("2020-01-01T00:00:00Z")


def test_full_deadline_must_fit_budget_remaining_after_pilot():
    value = _auth("full")
    value["pilot_gate"] = {
        "accepted": True,
        "projected_remaining_compute_usd": 0.01,
        "remaining_authorized_compute_usd": 0.01,
        "pilot_acquisition_summary_sha256": "c" * 64,
        "pilot_scoring_summary_sha256": "d" * 64,
        "cost_projection_sha256": "e" * 64,
        "pilot_completed_count": 64,
        "pilot_scored_count": 64,
        "batch_size": 4,
    }
    with pytest.raises(ValueError, match="remaining round budget"):
        runner.validate_authorization(
            value,
            phase="acquire",
            scope="full",
            run_id="synthetic",
            source_commit="a" * 40,
            bindings={"plan": "b" * 64},
        )


def test_scorer_refuses_context_overflow_without_truncating():
    guarded = runner.ContextGuardedHarmBench("synthetic-unused")
    guarded.executor = SimpleNamespace(
        model_loaded=True,
        _model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5)),
        _tokenizer=lambda text, truncation: {"input_ids": [1] * 6},
    )
    with pytest.raises(ValueError, match="context overflow"):
        guarded([{"behavior": "synthetic", "generation": "harmless"}])


def _fixture(root: Path):
    acquisition = root / "original-acquisition"
    scoring = root / "original-scoring"
    original = {
        "trial_id": "synthetic",
        "generated_token_ids": [5] * 128,
        "generated_text": "synthetic-128",
    }
    private_hash = write_immutable_json(acquisition / "restricted" / "synthetic.json", original)
    row = {
        "trial_id": "synthetic",
        "finish_reason": "length",
        "generated_token_count": 128,
        "generated_token_ids_sha256": token_hash(original["generated_token_ids"]),
        "generated_text_sha256": sha256_text(original["generated_text"]),
        "restricted_artifact_sha256": private_hash,
    }
    row["original_receipt_sha256"] = write_immutable_json(
        acquisition / "receipts" / "synthetic.json", row
    )
    prior = {
        "generation_receipt_sha256": row["original_receipt_sha256"],
        "generated_text_sha256": row["generated_text_sha256"],
        "classifier_behavior_sha256": sha256_text("synthetic safe core"),
        "parse_ok": True,
        "binary_prediction": False,
        "yes_probability": 0.1,
    }
    prior_hash = write_immutable_json(scoring / "trials" / "synthetic.json", prior)
    context = {
        "plan": {"rows": [row], "pilot_trial_ids": ["synthetic"]},
        "public": {"model_revision": "synthetic-pinned-model"},
        "bindings": {"private_plan_sha256": "a" * 64},
        "original_score_hashes": {"synthetic": prior_hash},
        "observations": {
            "synthetic": {"prompt_token_ids": [1, 2], "request_core_text": "synthetic safe core"}
        },
    }
    return context, acquisition, scoring


class _Generator:
    def __init__(self, *args, **kwargs):
        self.readouts_by_generated_tokens = {}

    def __call__(self, prefix, max_new_tokens):
        return {"delta_token_ids": [7, 8], "eos": True}

    def readout(self, prefix):
        value = {
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": token_hash(prefix),
            "feature_6779_magnitude": 0.0,
            "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
        }
        self.readouts_by_generated_tokens[str(len(prefix))] = value
        return value


def _passed_parity(*args, **kwargs):
    return {
        "status": "pass",
        "parity_scope": "exact_saved_prefix_conditional",
        "original_prefix_immutable": True,
        "trusted_input_prefix_unchanged": True,
        "conditional_continuation_matches": True,
    }


def test_exact_prefix_acquisition_score_reuse_and_resume(tmp_path, capsys):
    context, acquisition, scoring = _fixture(tmp_path)
    output = tmp_path / "continuation"
    original_hash = sha256_file(acquisition / "restricted" / "synthetic.json")
    kwargs = dict(
        context=context,
        acquisition_root=acquisition,
        output_root=output,
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        runtime_factory=lambda: object(),
        generator_factory=_Generator,
        parity=_passed_parity,
    )
    acquired = runner.acquire(**kwargs)
    assert acquired["eos_count"] == 1
    assert runner.acquire(**kwargs) == acquired
    calls = []

    def execute(rows):
        calls.append(rows)
        return [
            {
                "generation_trial_id": "synthetic",
                "parse_ok": True,
                "binary_prediction": True,
                "yes_probability": 0.9,
            }
        ]

    scored = runner.score(
        context=context,
        acquisition_root=acquisition,
        original_scoring_root=scoring,
        output_root=output,
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        decode=lambda ids: f"synthetic-{len(ids)}",
        execute=execute,
    )
    assert len(calls) == 1  # Original128 and EOS carryforward do not rescore.
    assert scored["counts"]["horizon_128_censored_negative"] == 1
    assert scored["counts"]["horizon_1024_positive"] == 1
    assert sha256_file(acquisition / "restricted" / "synthetic.json") == original_hash
    assert "synthetic safe core" not in capsys.readouterr().out


def test_original_token_artifact_tampering_fails_closed(tmp_path):
    context, acquisition, _ = _fixture(tmp_path)
    path = acquisition / "restricted" / "synthetic.json"
    path.write_text(json.dumps({"generated_token_ids": [4] * 128}))
    with pytest.raises(ValueError, match="artifact drift"):
        runner.load_original_tokens(context["plan"]["rows"][0], acquisition)


@pytest.mark.parametrize(
    "parity_result",
    [
        {"status": "stop_conditional_parity_mismatch"},
        {"status": "pass"},  # Historical parity semantics are not accepted.
        {**_passed_parity(), "original_prefix_immutable": False},
        {**_passed_parity(), "trusted_input_prefix_unchanged": False},
        {**_passed_parity(), "conditional_continuation_matches": False},
    ],
)
def test_parity_failure_stops_before_continuation(tmp_path, parity_result):
    context, acquisition, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="parity gate failed"):
        runner.acquire(
            context=context,
            acquisition_root=acquisition,
            output_root=tmp_path / "continuation",
            scope="pilot",
            source_commit="b" * 40,
            deadline_monotonic=time.monotonic() + 60,
            runtime_factory=lambda: object(),
            generator_factory=_Generator,
            parity=lambda *args, **kw: parity_result,
        )
    assert not (tmp_path / "continuation" / "trials").exists()


def test_completed_acquisition_manifest_is_verified_on_resume(tmp_path):
    context, acquisition, _ = _fixture(tmp_path)
    output = tmp_path / "continuation"
    kwargs = dict(
        context=context,
        acquisition_root=acquisition,
        output_root=output,
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        runtime_factory=lambda: object(),
        generator_factory=_Generator,
        parity=_passed_parity,
    )
    result = runner.acquire(**kwargs)
    assert result["trial_bundle_file_count"] > 0
    path = output / "trials" / "synthetic" / "restricted" / "tokens-0256.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bundle manifest drift"):
        runner.acquire(**kwargs)


def _source_fixture(tmp_path, monkeypatch):
    paths = [
        "src/lexical_prompt_study/__init__.py",
        "src/lexical_prompt_study/continuation_runner.py",
        *runner._SOURCE_CONFIG_PATHS,
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic safe fixture\n")
    manifest = {
        "source_commit": "a" * 40,
        "files": {relative: sha256_file(tmp_path / relative) for relative in paths},
    }
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "src/lexical_prompt_study/continuation_runner.py")
    )
    return path, manifest


def test_gitless_manifest_verifies_exact_package_and_pinned_hash(tmp_path, monkeypatch):
    path, manifest = _source_fixture(tmp_path, monkeypatch)
    result = runner.verify_source_manifest(
        path, source_root=tmp_path, expected_sha256=sha256_file(path)
    )
    assert result["source_commit"] == manifest["source_commit"]
    assert result["source_file_count"] == 8
    with pytest.raises(ValueError, match="authorization hash drift"):
        runner.verify_source_manifest(path, source_root=tmp_path, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    "unsafe",
    ["../escape.py", "/escape.py", "src/../escape.py", "src\\escape.py", "private/result.json"],
)
def test_gitless_manifest_rejects_unsafe_paths(tmp_path, monkeypatch, unsafe):
    path, manifest = _source_fixture(tmp_path, monkeypatch)
    manifest["files"][unsafe] = "a" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        runner.verify_source_manifest(path, source_root=tmp_path)


@pytest.mark.parametrize("mutation", ["extra", "missing", "changed", "symlink", "missing_config"])
def test_gitless_manifest_rejects_package_drift(tmp_path, monkeypatch, mutation):
    path, manifest = _source_fixture(tmp_path, monkeypatch)
    package = tmp_path / "src/lexical_prompt_study"
    if mutation == "extra":
        (package / "extra.py").write_text("# unexpected\n")
    elif mutation == "missing":
        (package / "__init__.py").unlink()
    elif mutation == "changed":
        (package / "__init__.py").write_text("# changed\n")
    elif mutation == "symlink":
        original = package / "__init__.py"
        original.rename(package / "renamed.txt")
        original.symlink_to(package / "renamed.txt")
    else:
        del manifest["files"]["uv.lock"]
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.verify_source_manifest(path, source_root=tmp_path)


def test_gitless_acquire_uses_verified_manifest_commit_not_git(tmp_path, monkeypatch):
    path, _ = _source_fixture(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    digest = sha256_file(path)
    auth = _auth()
    auth["source_manifest_sha256"] = digest
    auth["bindings"]["source_manifest_sha256"] = digest
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(auth))
    monkeypatch.setattr(runner, "load_context", lambda **kwargs: {"bindings": {"plan": "b" * 64}})
    monkeypatch.setattr(runner, "_source_commit", lambda: pytest.fail("Git must not be fabricated"))
    calls = []

    def acquired(**kwargs):
        calls.append(kwargs)
        return {"status": "acquisition_complete"}

    monkeypatch.setattr(runner, "acquire_batch", acquired)
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "acquire",
            "--source-manifest",
            str(path),
            "--authorization",
            str(authorization),
            "--run-id",
            "synthetic",
        ],
    )
    runner.main()
    assert calls[0]["source_commit"] == "a" * 40
    assert calls[0]["context"]["bindings"]["source_manifest_sha256"] == digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget_authority_id", None),
        ("budget_authority_id", "unknown"),
        ("maximum_new_compute_usd", 69),
        ("maximum_pilot_compute_usd", 6.01),
        ("round_spent_upper_usd", float("nan")),
        ("pilot_spent_upper_usd", -1),
        ("batch_size", 1),
        ("execution_protocol", "a146-singleton"),
    ],
)
def test_a148_authorization_rejects_legacy_or_uncapped_approvals(field, value):
    auth = _auth()
    auth[field] = value
    with pytest.raises(ValueError):
        runner.validate_authorization(
            auth,
            phase="acquire",
            scope="pilot",
            run_id="synthetic",
            source_commit="a" * 40,
            bindings={"plan": "b" * 64},
        )


def _batch_fixture(root, count=68):
    acquisition = root / "original-acquisition"
    context = {
        "plan": {"rows": [], "pilot_trial_ids": [f"row-{i:03d}" for i in range(64)]},
        "public": {
            "model_revision": "synthetic",
            "execution_protocol": "a148-batch4",
            "batch_size": 4,
        },
        "bindings": {"private_plan_sha256": "a" * 64},
        "observations": {},
    }
    for index in range(count):
        trial_id = f"row-{index:03d}"
        private = {
            "trial_id": trial_id,
            "generated_token_ids": [5] * 128,
            "generated_text": "synthetic-128",
        }
        row = {
            "trial_id": trial_id,
            "finish_reason": "length",
            "generated_token_count": 128,
            "generated_token_ids_sha256": token_hash(private["generated_token_ids"]),
            "generated_text_sha256": sha256_text(private["generated_text"]),
            "restricted_artifact_sha256": write_immutable_json(
                acquisition / "restricted" / f"{trial_id}.json", private
            ),
            "intent_frame": "unsafe_direct",
            "placement": "scaffold_before_request",
            "variant_family": "attack_block_mask",
            "attack_block_count": index % 5,
        }
        row["original_receipt_sha256"] = write_immutable_json(
            acquisition / "receipts" / f"{trial_id}.json", row
        )
        context["plan"]["rows"].append(row)
        context["observations"][trial_id] = {
            "prompt_token_ids": [1, 2, index % 7],
            "request_core_text": "synthetic safe core",
        }
    return context, acquisition


class _BatchGenerator:
    constructed = []
    fail_once = False

    def __init__(self, runtime, members, **kwargs):
        self.members = members
        self.kwargs = kwargs
        self.constructed.append([item["trial_id"] for item in members])
        self.timing_summary = {
            "elapsed_seconds": 0.1,
            "model_forward_count": 3,
            "model_forward_elapsed_seconds": 0.05,
            "detector_readout_elapsed_seconds": 0.01,
            "generated_new_token_count": 2 * len(members),
            "new_tokens_by_trial": {item["trial_id"]: 2 for item in members},
        }

    def restore(self, snapshot):
        assert snapshot["member_ids"] == [item["trial_id"] for item in self.members]

    def advance_to_horizon(self, horizon, checkpoint_sink):
        if horizon == 256 and type(self).fail_once:
            type(self).fail_once = False
            raise TimeoutError("synthetic interruption")
        members = [
            {
                "trial_id": item["trial_id"],
                "prompt_token_ids_sha256": token_hash(item["prompt_token_ids"]),
                "original_prefix_token_ids_sha256": token_hash(item["original_token_ids"]),
                "generated_token_ids": [
                    *item["original_token_ids"],
                    *([7, 8] if horizon > 128 else []),
                ],
                "eos": horizon > 128,
            }
            for item in self.members
        ]
        snapshot = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "status": "batch_horizon_tokens_complete",
            "cohort_id": self.kwargs["cohort_id"],
            "runtime_sha256": self.kwargs["runtime_sha256"],
            "plan_sha256": self.kwargs["plan_sha256"],
            "layout_sha256": "9" * 64,
            "batch_size": 4,
            "member_ids": [item["trial_id"] for item in self.members],
            "requested_horizon": horizon,
            "members": members,
        }
        checkpoint_sink(snapshot)
        readouts = {
            item["trial_id"]: {
                str(len(item["generated_token_ids"])): {
                    "prefix_token_count": len(item["generated_token_ids"]),
                    "prefix_token_ids_sha256": token_hash(item["generated_token_ids"]),
                    "feature_6779_magnitude": 0.0,
                    "jlens_refusal_minus_compliance_trajectory": [0.0] * 31,
                }
            }
            for item in members
        }
        return {"snapshot": snapshot, "readouts": readouts}


def _batch_args(tmp_path, *, count=68):
    context, acquisition = _batch_fixture(tmp_path, count=count)
    return dict(
        context=context,
        acquisition_root=acquisition,
        output_root=tmp_path / "a148",
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        runtime_factory=lambda: object(),
        generator_factory=_BatchGenerator,
        parity=_passed_batch_parity,
    )


def _passed_batch_parity(runtime, members, **kwargs):
    return {
        "status": "pass",
        "parity_scope": "fixed_four_slot_saved_prefix_conditional",
        "batch_size": 4,
        "observation_count": len(members),
        "trusted_input_prefix_unchanged": True,
        "split_boundary_input_prefix_unchanged": True,
        "split_boundary_members": [
            {"trial_id": item["trial_id"], "conditional_token_ids_match": True, "eos_match": True}
            for item in members
        ],
        "members": [
            {"trial_id": item["trial_id"], "conditional_token_ids_match": True, "eos_match": True}
            for item in members
        ],
    }


def test_a148_layout_pilot_full_fixed_slots_and_tail(tmp_path):
    context, _ = _batch_fixture(tmp_path, count=69)
    layout = runner.batch_layout(context["plan"])
    assert len(layout["cohorts"]) == 18
    assert all(item["subset"] == "pilot" for item in layout["cohorts"][:16])
    assert layout["cohorts"][-1]["member_ids"] == ["row-068"]
    assert layout["cohorts"][-1]["virtual_slot_count"] == 3
    context["plan"]["rows"].reverse()
    assert layout == runner.batch_layout(context["plan"])


def test_a148_pilot_full_resume_reuses_exact_completed_cohorts(tmp_path):
    kwargs = _batch_args(tmp_path)
    _BatchGenerator.constructed = []
    result = runner.acquire_batch(**kwargs)
    assert result["completed_count"] == result["eos_count"] == 64
    assert result["completed_cohort_count"] == 16
    assert len(_BatchGenerator.constructed) == 16
    assert runner.acquire_batch(**kwargs) == result
    assert len(_BatchGenerator.constructed) == 16
    full = runner.acquire_batch(**{**kwargs, "scope": "full"})
    assert full["completed_count"] == 68
    assert len(_BatchGenerator.constructed) == 17
    assert _BatchGenerator.constructed[-1] == [f"row-{i:03d}" for i in range(64, 68)]
    assert len(list((kwargs["output_root"] / "batch-timings").glob("*/*.json"))) == 17


def test_a148_rejects_historical_singleton_namespace(tmp_path):
    kwargs = _batch_args(tmp_path)
    (kwargs["output_root"] / "trials").mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh namespace"):
        runner.acquire_batch(**kwargs)


def test_a148_collective_checkpoint_resume_keeps_membership(tmp_path):
    kwargs = _batch_args(tmp_path)
    _BatchGenerator.constructed = []
    _BatchGenerator.fail_once = True
    result = runner.acquire_batch(**kwargs)
    assert result["status"] == "paused_deadline" and result["completed_count"] == 0
    snapshot_paths = list((kwargs["output_root"] / "cohorts").glob("*/restricted/tokens-0128.json"))
    assert len(snapshot_paths) == 1
    first = _BatchGenerator.constructed[0]
    completed = runner.acquire_batch(**kwargs)
    assert completed["completed_count"] == 64
    assert _BatchGenerator.constructed[1] == first


def test_a148_token_snapshot_tamper_is_rejected_before_resume(tmp_path):
    kwargs = _batch_args(tmp_path)
    _BatchGenerator.fail_once = True
    runner.acquire_batch(**kwargs)
    path = next((kwargs["output_root"] / "cohorts").glob("*/restricted/tokens-0128.json"))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checkpoint hash drift"):
        runner.acquire_batch(**kwargs)


def test_batch_status_only_parity_receipt_is_insufficient(tmp_path):
    kwargs = _batch_args(tmp_path)
    with pytest.raises(ValueError, match="parity gate failed"):
        runner.acquire_batch(**{**kwargs, "parity": lambda *args, **kw: {"status": "pass"}})


class _LengthGenerator(_Generator):
    def __call__(self, prefix, max_new_tokens):
        return {"delta_token_ids": [7] * max_new_tokens, "eos": False}


def test_scoring_resume_preserves_all_timing_and_separates_horizon_from_ever_positive(tmp_path):
    context, acquisition, scoring = _fixture(tmp_path)
    context["observations"]["synthetic"]["utility_expected_text"] = "synthetic-256"
    output = tmp_path / "continuation"
    runner.acquire(
        context=context,
        acquisition_root=acquisition,
        output_root=output,
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        runtime_factory=lambda: object(),
        generator_factory=_LengthGenerator,
        parity=_passed_parity,
    )
    calls = []

    def first_attempt(rows):
        calls.append(rows[0]["generation"])
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return [
            {
                "generation_trial_id": "synthetic",
                "parse_ok": True,
                "binary_prediction": True,
                "yes_probability": 0.9,
            }
        ]

    kwargs = dict(
        context=context,
        acquisition_root=acquisition,
        original_scoring_root=scoring,
        output_root=output,
        scope="pilot",
        source_commit="b" * 40,
        deadline_monotonic=time.monotonic() + 60,
        decode=lambda ids: f"synthetic-{len(ids)}",
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.score(**kwargs, execute=first_attempt)

    def resumed(rows):
        calls.append(rows[0]["generation"])
        return [
            {
                "generation_trial_id": "synthetic",
                "parse_ok": True,
                "binary_prediction": False,
                "yes_probability": 0.1,
            }
        ]

    result = runner.score(**kwargs, execute=resumed)
    assert result["new_score_call_count"] == 3
    assert result["new_score_call_count_this_invocation"] == 2
    assert result["score_receipt_count"] == 4
    assert result["scoring_timing_complete"] is True
    assert calls == ["synthetic-256", "synthetic-512", "synthetic-512", "synthetic-1024"]
    assert result["counts"]["horizon_512_positive"] == 0
    assert result["counts"]["horizon_512_ever_positive"] == 1
    assert result["counts"]["horizon_1024_ever_positive"] == 1
    value = json.loads((output / "scores" / "synthetic" / "score-0256.json").read_text())
    assert value["utility_exact_match"] is True
    assert value["fresh_score_elapsed_seconds"] > 0
    assert (
        runner.score(**kwargs, execute=lambda _: pytest.fail("completed scoring reran")) == result
    )

    path = output / "scores" / "synthetic" / "score-0256.json"
    value["yes_probability"] = 0.8
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scoring summary drift"):
        runner.score(**kwargs, execute=resumed)


def test_full_projection_requires_measured_complete_scored_batch_evidence(tmp_path):
    context, _ = _batch_fixture(tmp_path)
    auth = _auth("full")
    auth["pilot_gate"] = {
        "pilot_acquisition_summary_sha256": "c" * 64,
        "pilot_scoring_summary_sha256": "d" * 64,
        "remaining_authorized_compute_usd": 10,
        "projected_remaining_compute_usd": 5,
    }
    projection = {
        "schema_version": "2.0",
        "study_id": STUDY_ID,
        "execution_protocol": "a148-batch4",
        "status": "complete_pilot_projection",
        "fits_budget_planning_gate": True,
        "complete_pilot_gate_passed": True,
        "runtime_sha256": runner._runtime_identity(context, "b" * 40),
        "batch_layout_sha256": sha256_bytes(
            canonical_json_bytes(runner.batch_layout(context["plan"]))
        ),
        "batch_size": 4,
        "pilot_selected_count": 64,
        "pilot_completed_count": 64,
        "pilot_scored_count": 64,
        "round_hard_cap_usd": 70,
        "bindings": {
            "private_plan_sha256": context["bindings"]["private_plan_sha256"],
            "pilot_acquisition_summary_sha256": "c" * 64,
            "pilot_scoring_summary_sha256": "d" * 64,
        },
        "projected_remaining_acquisition_usd": 1,
        "projected_remaining_scoring_usd": 1,
        "reserves": {"loading_and_parity_usd": 1, "scoring_additional_usd": 1, "storage_usd": 1},
        "full_hourly_usd": 1.5,
        "projected_round_total_upper_planning_usd": 7,
    }
    kwargs = dict(context=context, authorization=auth, source_commit="b" * 40)
    runner.validate_full_projection(projection, **kwargs)
    for key, value in (
        ("pilot_scored_count", 63),
        ("full_hourly_usd", 1.49),
        ("fits_budget_planning_gate", False),
        ("runtime_sha256", "f" * 64),
        ("projected_remaining_scoring_usd", 2),
        ("projected_round_total_upper_planning_usd", 70.01),
    ):
        with pytest.raises(ValueError):
            runner.validate_full_projection({**projection, key: value}, **kwargs)
