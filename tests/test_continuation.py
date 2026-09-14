from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from lexical_prompt_study.continuation import (
    A142_PLAN_SHA256,
    A142_SOURCE_COMMIT,
    A142_TOPOLOGY_SHA256,
    CHECKPOINTS,
    build_continuation_plan,
    public_plan_summary,
    resume_trial,
    select_pilot,
    token_hash,
    trial_lock,
    write_immutable_json,
)
from lexical_prompt_study.hashing import canonical_json_bytes, sha256_bytes, sha256_file


def _row(index: int = 0, *, family: str = "attack_block_mask", eos: bool = False):
    return {
        "trial_id": f"synthetic-{index:04}",
        "request_core_id": f"core-{index % 10}",
        "request_core_sha256": f"{index + 1:064x}",
        "intent_frame": "unsafe_direct",
        "safe_intent": False,
        "variant_family": family,
        "attack_block_count": 0 if family != "attack_block_mask" else index % 5,
        "attack_block_mask": 0 if family != "attack_block_mask" else (1 << (index % 5)) - 1,
        "placement": None if family == "no_scaffold" else "scaffold_before_request",
        "prompt_sha256": "a" * 64,
        "prompt_token_ids_sha256": "b" * 64,
        "generated_token_ids_sha256": token_hash([42] * (1 if eos else 128)),
        "generated_text_sha256": "c" * 64,
        "restricted_artifact_sha256": "d" * 64,
        "generated_token_count": 1 if eos else 128,
        "finish_reason": "eos" if eos else "length",
        "original_receipt_sha256": f"{index + 100:064x}",
    }


def _original_acquisition(tmp_path: Path, rows: list[dict]):
    acquisition = tmp_path / "acquisition"
    receipts = acquisition / "receipts"
    receipts.mkdir(parents=True)
    for row in rows:
        payload = {
            **{key: value for key, value in row.items() if key != "original_receipt_sha256"},
            "source_commit": A142_SOURCE_COMMIT,
            "plan_sha256": A142_PLAN_SHA256,
            "private_topology_sha256": A142_TOPOLOGY_SHA256,
            "run_id": "synthetic-run",
            "status": "complete",
            # Outcome values must never affect the plan or leave the receipts.
            "readouts": {"0": {"feature_6779_magnitude": 1e20}},
            "refusal_detected": True,
            "utility_exact_match": False,
        }
        (receipts / f"{row['trial_id']}.json").write_bytes(canonical_json_bytes(payload))
    manifest = [
        {"trial_id": path.stem, "sha256": sha256_file(path)}
        for path in sorted(receipts.glob("*.json"))
    ]
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    summary = {
        "status": "acquisition_complete",
        "source_commit": A142_SOURCE_COMMIT,
        "plan_sha256": A142_PLAN_SHA256,
        "private_topology_sha256": A142_TOPOLOGY_SHA256,
        "run_id": "synthetic-run",
        "observation_count": len(rows),
        "generation_count": len(rows),
        "receipt_manifest_sha256": manifest_hash,
        "generation_checkpoints": [0, 1, 4, 8],
        "enforcement_enabled": False,
        "unopened_v2_confirmation_opened": False,
    }
    (acquisition / "summary.json").write_bytes(canonical_json_bytes(summary))
    kwargs = {
        "expected_summary_sha256": sha256_file(acquisition / "summary.json"),
        "expected_manifest_sha256": manifest_hash,
        "expected_observations": len(rows),
        "expected_capped": sum(row["finish_reason"] == "length" for row in rows),
    }
    return acquisition, kwargs


def test_plan_only_includes_all_capped_rows_and_keeps_eos_denominator(tmp_path):
    rows = [_row(index) for index in range(5)] + [_row(5, eos=True)]
    acquisition, kwargs = _original_acquisition(tmp_path, rows)
    plan = build_continuation_plan(acquisition, pilot_size=4, **kwargs)
    assert len(plan["original_receipts"]) == 6
    assert plan["continuation_row_count"] == 5
    assert plan["eos_carryforward_count"] == 1
    assert all(row["finish_reason"] == "length" for row in plan["rows"])
    assert all("readouts" not in row and "utility_exact_match" not in row for row in plan["rows"])
    public = public_plan_summary(plan)
    assert public["pilot_count"] == 4
    assert not any(row["trial_id"] in json.dumps(public) for row in rows)
    assert "rows" not in public and "pilot_trial_ids" not in public


def test_plan_rejects_modified_original_receipt(tmp_path):
    acquisition, kwargs = _original_acquisition(tmp_path, [_row()])
    target = next((acquisition / "receipts").glob("*.json"))
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        build_continuation_plan(acquisition, **kwargs)


def test_plan_rejects_modified_summary(tmp_path):
    acquisition, kwargs = _original_acquisition(tmp_path, [_row()])
    summary = acquisition / "summary.json"
    summary.write_text(summary.read_text() + " ")
    with pytest.raises(ValueError, match="summary hash mismatch"):
        build_continuation_plan(acquisition, **kwargs)


def test_plan_rejects_non128_capped_rows_even_with_matching_manifest(tmp_path):
    row = _row()
    row["generated_token_count"] = 127
    acquisition, kwargs = _original_acquisition(tmp_path, [row])
    with pytest.raises(ValueError, match="not exactly 128"):
        build_continuation_plan(acquisition, **kwargs)


def test_pilot_is_deterministic_outcome_blind_balanced_and_family_complete():
    rows = []
    for index in range(120):
        row = _row(index)
        row["intent_frame"] = (
            "unsafe_direct",
            "safe_classify_exact",
            "safe_refuse_exact",
            "safe_acknowledge_exact",
        )[index % 4]
        row["placement"] = ("scaffold_before_request", "scaffold_after_request")[index % 2]
        rows.append(row)
    rows += [
        _row(120, family="structural_sham"),
        _row(121, family="harmless_structured_wrapper"),
        _row(122, family="no_scaffold"),
    ]
    after_control = _row(123, family="harmless_structured_wrapper")
    after_control["placement"] = "scaffold_after_request"
    rows.append(after_control)
    selection = select_pilot(rows)
    assert len(selection) == len(set(selection)) == 64
    assert selection == select_pilot(list(reversed(rows)))
    for row in rows:
        row["arbitrary_outcome"] = object()
        row["original_receipt_sha256"] = "8" * 64
    assert selection == select_pilot(rows)
    selected_rows = [row for row in rows if row["trial_id"] in selection]
    assert {row["variant_family"] for row in selected_rows} == {
        "attack_block_mask",
        "structural_sham",
        "harmless_structured_wrapper",
        "no_scaffold",
    }
    assert {(row["variant_family"], row["placement"]) for row in selected_rows} == {
        (row["variant_family"], row["placement"]) for row in rows
    }
    assert {
        (row["intent_frame"], row["placement"], row["attack_block_count"]) for row in selected_rows
    } == {(row["intent_frame"], row["placement"], row["attack_block_count"]) for row in rows}


def _run(tmp_path, generator, **overrides):
    return resume_trial(
        **{
            "row": _row(),
            "original_token_ids": [42] * 128,
            "output_root": tmp_path,
            "generator": generator,
            "deadline_monotonic": 100.0,
            "clock": lambda: 0.0,
            "plan_sha256": "e" * 64,
            "runtime_sha256": "f" * 64,
            **overrides,
        }
    )


def _fill(prefix, maximum):
    return {"delta_token_ids": [43] * maximum, "eos": False}


def test_resume_complete_ceiling_never_reexecutes_generator(tmp_path):
    calls = []

    def generator(prefix, maximum):
        calls.append((len(prefix), maximum))
        return _fill(prefix, maximum)

    result = _run(tmp_path, generator)
    assert calls == [(128, 128), (256, 256), (512, 512)]
    assert result["status"] == "complete_censored_at_ceiling"
    assert result["last_checkpoint"]["right_censored"] is True
    assert result["last_checkpoint"]["classifier_status"] == "not_scored"
    assert result == _run(tmp_path, lambda *_: pytest.fail("regenerated complete checkpoint"))
    for path in tmp_path.rglob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for path in tmp_path.rglob("checkpoint-*.json"):
        assert 'generated_token_ids"' not in path.read_text()


@pytest.mark.parametrize("new_tokens", [0, 7, 128])
def test_eos_finishes_at_actual_length_and_does_not_fabricate_later_horizons(tmp_path, new_tokens):
    result = _run(tmp_path, lambda *_: {"delta_token_ids": [43] * new_tokens, "eos": True})
    assert result["status"] == "complete_eos"
    assert result["last_checkpoint"]["generated_token_count"] == 128 + new_tokens
    assert result["last_checkpoint"]["requested_horizon"] == 256
    assert result["last_checkpoint"]["right_censored"] is False
    assert len(list(tmp_path.rglob("checkpoint-*.json"))) == 2
    assert result == _run(tmp_path, lambda *_: pytest.fail("regenerated EOS checkpoint"))


def test_pause_before_first_chunk_is_resumable(tmp_path):
    result = _run(tmp_path, lambda *_: pytest.fail("deadline ignored"), clock=lambda: 100.0)
    assert result["status"] == "paused_deadline"
    assert result["last_checkpoint"] is None
    assert _run(tmp_path, _fill)["status"] == "complete_censored_at_ceiling"


def test_pause_after_checkpoint_resumes_from_saved_prefix(tmp_path):
    times = iter([0.0, 0.0, 100.0])
    result = _run(tmp_path, _fill, clock=lambda: next(times))
    assert result["status"] == "paused_deadline"
    assert result["last_checkpoint"]["generated_token_count"] == 256
    calls = []

    def generator(prefix, maximum):
        calls.append(len(prefix))
        return _fill(prefix, maximum)

    assert _run(tmp_path, generator)["status"] == "complete_censored_at_ceiling"
    assert calls == [256, 512]


def test_crash_between_tokens_and_receipt_recovers_without_model(tmp_path, monkeypatch):
    import lexical_prompt_study.continuation as module

    original_write = module.write_immutable_json

    def fail_receipt(path, payload):
        if path.name == "checkpoint-0256.json":
            raise RuntimeError("simulated interrupted commit")
        return original_write(path, payload)

    monkeypatch.setattr(module, "write_immutable_json", fail_receipt)
    with pytest.raises(RuntimeError, match="interrupted commit"):
        _run(tmp_path, _fill)
    monkeypatch.setattr(module, "write_immutable_json", original_write)
    calls = []

    def generator(prefix, maximum):
        calls.append(len(prefix))
        return _fill(prefix, maximum)

    _run(tmp_path, generator)
    assert calls == [256, 512]


def test_changed_original_prefix_is_rejected_before_any_generation(tmp_path):
    with pytest.raises(ValueError, match="original exact capped token prefix"):
        _run(tmp_path, lambda *_: pytest.fail("bad prefix executed"), original_token_ids=[44] * 128)


def test_modified_checkpoint_tokens_or_receipts_are_rejected(tmp_path):
    _run(tmp_path, _fill)
    token_path = next(tmp_path.rglob("tokens-0256.json"))
    artifact = json.loads(token_path.read_text())
    artifact["generated_token_ids"][-1] = 99
    token_path.write_bytes(canonical_json_bytes(artifact))
    with pytest.raises(ValueError, match="different content"):
        _run(tmp_path, lambda *_: pytest.fail("tampered state executed"))


def test_runtime_or_plan_hash_change_cannot_resume(tmp_path):
    _run(tmp_path, _fill)
    with pytest.raises(ValueError, match="provenance drift"):
        _run(tmp_path, _fill, runtime_sha256="1" * 64)


def test_readout_schema_forbids_decoded_text_and_token_lists(tmp_path):
    def readout(prefix):
        return {
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": token_hash(prefix),
            "generated_text": "synthetic but forbidden",
        }

    with pytest.raises(ValueError, match="unexpected control-plane"):
        _run(tmp_path, _fill, readout=readout)


def test_numeric_readouts_are_prefix_bound(tmp_path):
    def readout(prefix):
        return {
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": token_hash(prefix),
            "jlens_refusal_minus_compliance_trajectory": [0.25] * 31,
            "feature_6779_magnitude": 0.0,
            "residual_artifact_sha256": "9" * 64,
        }

    result = _run(tmp_path, _fill, readout=readout)
    assert result["last_checkpoint"]["readout"]["prefix_token_count"] == 1024


def test_failed_diagnostic_does_not_lose_generated_chunk(tmp_path):
    def failing_readout(prefix):
        if len(prefix) == 256:
            raise RuntimeError("synthetic readout failure")
        return {"prefix_token_count": len(prefix), "prefix_token_ids_sha256": token_hash(prefix)}

    with pytest.raises(RuntimeError, match="readout failure"):
        _run(tmp_path, _fill, readout=failing_readout)
    assert len(list(tmp_path.rglob("tokens-0256.json"))) == 1
    calls = []

    def generator(prefix, maximum):
        calls.append(len(prefix))
        return _fill(prefix, maximum)

    _run(
        tmp_path,
        generator,
        readout=lambda prefix: {
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": token_hash(prefix),
        },
    )
    assert calls == [256, 512]


def test_duplicate_live_trial_is_blocked(tmp_path):
    with trial_lock(tmp_path / _row()["trial_id"]):
        with pytest.raises(RuntimeError, match="already running"):
            _run(tmp_path, _fill)


def test_immutable_json_never_overwrites_and_retains_permissions(tmp_path):
    target = tmp_path / "private" / "receipt.json"
    first = write_immutable_json(target, {"count": 1})
    assert first == write_immutable_json(target, {"count": 1})
    with pytest.raises(ValueError, match="different content"):
        write_immutable_json(target, {"count": 2})
    assert json.loads(target.read_text()) == {"count": 1}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "result",
    [
        {"delta_token_ids": [], "eos": False},
        {"delta_token_ids": [1] * 129, "eos": False},
        {"delta_token_ids": [1], "eos": False},
        {"delta_token_ids": [1], "eos": 1},
        {"delta_token_ids": [1], "eos": True, "decoded_text": "forbidden"},
    ],
)
def test_invalid_generator_contract_is_rejected(tmp_path, result):
    with pytest.raises(ValueError):
        _run(tmp_path, lambda *_: result)


def test_checkpoints_are_prospectively_frozen(tmp_path):
    with pytest.raises(ValueError, match="horizons changed"):
        _run(tmp_path, _fill, checkpoints=(128, 2048))
    assert CHECKPOINTS == (128, 256, 512, 1024)
