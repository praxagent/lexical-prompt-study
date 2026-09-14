from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from lexical_prompt_study.continuation_batch import (
    BatchPrefixGenerator,
    verify_batch_conditional_parity,
)


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = transformers.LlamaConfig(
        vocab_size=97,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=96,
        pad_token_id=0,
        bos_token_id=1,
        attention_dropout=0.0,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        model.lm_head.weight[96].zero_()
    yield SimpleNamespace(
        torch=torch,
        model=model,
        tokenizer=SimpleNamespace(eos_token_id=96, pad_token_id=0),
        source_layers=list(range(31)),
        feature_id=6779,
        encoder=torch.randn(6780, 8) * 0.01,
        encoder_bias=torch.zeros(6780),
        decoder=torch.randn(8, 6780) * 0.01,
        decoder_bias=None,
        subspace_ids=torch.tensor([1, 2, 3]),
        subspace_weights=torch.ones(3) / 3,
        jacobians={layer: torch.eye(8) for layer in range(31)},
        _probe_margin=lambda vectors: vectors.mean(dim=1),
    )
    torch.set_num_threads(previous_threads)


def members(count=4):
    return [
        {
            "trial_id": f"trial-{index}",
            "prompt_token_ids": [1, *([2 + index] * (index + 1))],
            "original_token_ids": [4 + index] * 128,
        }
        for index in range(count)
    ]


def adapter(runtime, rows=None, **kwargs):
    return BatchPrefixGenerator(
        runtime,
        members() if rows is None else rows,
        cohort_id="cohort-0",
        runtime_sha256="a" * 64,
        plan_sha256="b" * 64,
        **kwargs,
    )


def test_real_transformers_conditional_batch_parity_and_short_cohort(runtime):
    for count in (4, 2):
        rows = members(count)
        saved = copy.deepcopy(rows)
        result = verify_batch_conditional_parity(runtime, rows, extension_tokens=8)
        assert result["status"] == "pass"
        assert result["batch_size"] == 4
        assert result["trusted_input_prefix_unchanged"]
        assert result["split_boundary_input_prefix_unchanged"]
        assert all(row["conditional_token_ids_match"] for row in result["split_boundary_members"])
        assert result["observation_count"] == count
        assert rows == saved
        assert all(row["continued_token_count"] == 8 for row in result["members"])
        assert "generated_token_ids" not in repr(result)


def test_layout_exact_prefix_and_restore_tamper_guards(runtime):
    first = adapter(runtime)
    assert first.layout_sha256 == adapter(runtime).layout_sha256
    assert first.layout_sha256 != adapter(runtime, list(reversed(members()))).layout_sha256
    snapshot = first.snapshot()
    adapter(runtime).restore(snapshot)
    for mutate in (
        lambda x: x.update(runtime_sha256="c" * 64),
        lambda x: x["members"].reverse(),
        lambda x: x["members"][0]["generated_token_ids"].__setitem__(0, 5),
        lambda x: x["members"][0].update(eos=True),
    ):
        changed = copy.deepcopy(snapshot)
        mutate(changed)
        with pytest.raises(ValueError, match="drift"):
            adapter(runtime).restore(changed)
    short = members(1)
    short[0]["original_token_ids"].pop()
    with pytest.raises(ValueError, match="128-token"):
        adapter(runtime, short)


def test_checkpoint_precedes_all_readouts_and_historical_sites_share_one_forward(runtime, tmp_path):
    committed, received = [], []

    def readout_sink(trial, value):
        assert committed
        received.append((trial, value))

    first = adapter(runtime, members(2), residual_root=tmp_path, readout_sink=readout_sink)
    result = first.advance_to_horizon(128, committed.append)
    assert first.timing_summary["diagnostic_forward_calls"] == 1
    assert len(received) == 8
    for trial, history in result["readouts"].items():
        assert set(history) == {"16", "32", "64", "128"}
        assert all(
            len(row["jlens_refusal_minus_compliance_trajectory"]) == 31 for row in history.values()
        )
        payload = runtime.torch.load(tmp_path / trial / "residual-0128.pt", weights_only=True)
        assert set(payload["residuals"]) == {8, 16, 19, 24, 30}
        assert all(value.dtype == runtime.torch.bfloat16 for value in payload["residuals"].values())
    result = first.advance_to_horizon(256, committed.append)
    assert first.timing_summary["diagnostic_forward_calls"] == 2
    assert first.timing_summary["generation_calls"] == 1
    assert first.timing_summary["generation_step_count"] == 128
    assert first.timing_summary["model_forward_count"] == 130
    assert first.timing_summary["generated_new_token_count"] == 256
    assert all(len(row["generated_token_ids"]) == 256 for row in result["snapshot"]["members"])
    assert all("256" in history for history in result["readouts"].values())


def force_eos_by_slot(runtime, monkeypatch, after):
    from transformers import LogitsProcessor, LogitsProcessorList

    actual_generate = runtime.model.generate
    calls = []

    def generate(*args, **kwargs):
        inputs = kwargs["input_ids"]
        calls.append(tuple(inputs.shape))
        width = inputs.shape[1]

        class ForceEOS(LogitsProcessor):
            def __call__(self, input_ids, scores):
                step = input_ids.shape[1] - width
                for slot, threshold in enumerate(after):
                    if step >= threshold:
                        scores[slot, :] = -runtime.torch.inf
                        scores[slot, 96] = 0
                return scores

        kwargs["logits_processor"] = LogitsProcessorList(
            [*kwargs.get("logits_processor", []), ForceEOS()]
        )
        return actual_generate(*args, **kwargs)

    monkeypatch.setattr(runtime.model, "generate", generate)
    return calls


def test_mixed_eos_keeps_physical_slots_and_actual_terminal_readouts(runtime, monkeypatch):
    calls = force_eos_by_slot(runtime, monkeypatch, [0, 3, 7, 200])
    first = adapter(runtime)
    sink = []
    result = first.advance_to_horizon(256, sink.append)
    rows = result["snapshot"]["members"]
    assert [len(row["generated_token_ids"]) for row in rows] == [128, 131, 135, 256]
    assert [row["eos"] for row in rows] == [True, True, True, False]
    for index, count in enumerate((128, 131, 135)):
        history = result["readouts"][f"trial-{index}"]
        assert str(count) in history
        assert "256" not in history
    restored = adapter(runtime, preserved_readouts=result["readouts"])
    restored.restore(result["snapshot"])
    later = restored.advance_to_horizon(512, sink.append)
    assert calls[0][0] == calls[1][0] == 4
    assert later["snapshot"]["members"][:3] == rows[:3]
    assert len(later["snapshot"]["members"][3]["generated_token_ids"]) == 456
    assert later["snapshot"]["members"][3]["eos"]
    assert "456" in later["readouts"]["trial-3"]
    assert "512" not in later["readouts"]["trial-3"]


def test_batch_parity_exercises_immediate_and_mixed_eos(runtime, monkeypatch):
    force_eos_by_slot(runtime, monkeypatch, [0, 3, 7, 200])
    result = verify_batch_conditional_parity(runtime, members(), extension_tokens=8)
    assert result["status"] == "pass"
    assert [row["continued_token_count"] for row in result["members"]] == [0, 3, 7, 8]
    assert all(row["eos_match"] for row in result["split_boundary_members"])


def test_multisite_readouts_index_actual_tokens_not_right_padding(runtime):
    from lexical_prompt_study.continuation_gpu import PrefixGenerator

    batch = adapter(runtime)
    result = batch.advance_to_horizon(128, lambda _: None)
    for row in members():
        single = PrefixGenerator(runtime, row["prompt_token_ids"])
        single.prepare(row["original_token_ids"])
        for count, expected in single.readouts_by_generated_tokens.items():
            actual = result["readouts"][row["trial_id"]][count]
            for key in (
                "feature_6779_magnitude",
                "frozen_subspace_score",
                "sae_normalized_reconstruction_error",
                "jlens_refusal_minus_compliance_trajectory",
            ):
                assert actual[key] == pytest.approx(expected[key], abs=1e-6)


def test_crash_after_whole_group_tokens_resumes_without_regeneration(
    runtime, tmp_path, monkeypatch
):
    snapshots, history = [], {}

    def crash_after_readout(trial, value):
        history.setdefault(trial, {})[str(value["prefix_token_count"])] = value
        raise RuntimeError("synthetic diagnostic crash")

    first = adapter(runtime, members(2), residual_root=tmp_path, readout_sink=crash_after_readout)
    with pytest.raises(RuntimeError, match="synthetic"):
        first.advance_to_horizon(256, snapshots.append)
    assert len(snapshots) == 1
    assert all(len(row["generated_token_ids"]) == 256 for row in snapshots[0]["members"])
    resumed = adapter(runtime, members(2), residual_root=tmp_path, preserved_readouts=history)
    resumed.restore(snapshots[0])

    def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("completed cohort tokens must not be regenerated")

    monkeypatch.setattr(runtime.model, "generate", forbidden_generate)
    complete = resumed.advance_to_horizon(256, snapshots.append)
    assert complete["snapshot"] == snapshots[0]
    assert all(
        set(values) == {"16", "32", "64", "128", "256"} for values in complete["readouts"].values()
    )
    assert resumed.timing_summary["generation_calls"] == 0


def test_conditional_parity_detects_bad_adapter_tokens(runtime, monkeypatch):
    original = BatchPrefixGenerator._generate

    def wrong(self, count):
        result = original(self, count)
        result[0]["generated_token_ids"][128] = (result[0]["generated_token_ids"][128] + 1) % 95
        return result

    monkeypatch.setattr(BatchPrefixGenerator, "_generate", wrong)
    result = verify_batch_conditional_parity(runtime, members(2), extension_tokens=2)
    assert result["status"] == "stop_batch_conditional_parity_mismatch"


def test_deadline_prevents_generation_and_invalid_horizons_fail(runtime):
    first = adapter(runtime, deadline_monotonic=0, clock=lambda: 1)
    with pytest.raises(TimeoutError, match="deadline"):
        first.advance_to_horizon(256, lambda _: pytest.fail("no checkpoint without tokens"))
    assert first.timing_summary["generation_calls"] == 0
    with pytest.raises(ValueError, match="consecutive"):
        adapter(runtime).advance_to_horizon(512, lambda _: None)


def test_actual_cpu_adapter_materializes_runner_scorer_artifacts(runtime, tmp_path, monkeypatch):
    from lexical_prompt_study.continuation import token_hash, write_immutable_json
    from lexical_prompt_study.continuation_runner import _materialize_batch_horizon
    from lexical_prompt_study.hashing import sha256_file

    force_eos_by_slot(runtime, monkeypatch, [0, 3, 7, 9])
    original = members()
    rows = [
        {
            "trial_id": row["trial_id"],
            "generated_token_ids_sha256": token_hash(row["original_token_ids"]),
            "original_receipt_sha256": "c" * 64,
            "restricted_artifact_sha256": "d" * 64,
        }
        for row in original
    ]
    batch = adapter(runtime, original, residual_root=tmp_path / "batch-residuals")

    def checkpoint(snapshot):
        write_immutable_json(
            tmp_path / "cohort" / "restricted" / f"tokens-{snapshot['requested_horizon']:04d}.json",
            snapshot,
        )

    for horizon in (128, 256):
        result = batch.advance_to_horizon(horizon, checkpoint_sink=checkpoint)
        _materialize_batch_horizon(
            snapshot=result["snapshot"],
            readouts=result["readouts"],
            output_root=tmp_path,
            rows=rows,
            runtime_hash="a" * 64,
            plan_hash="b" * 64,
            layout_hash="e" * 64,
        )
        for item in result["snapshot"]["members"]:
            trial = tmp_path / "trials" / item["trial_id"]
            token_path = trial / "restricted" / f"tokens-{horizon:04d}.json"
            receipt_path = trial / "receipts" / f"checkpoint-{horizon:04d}.json"
            artifact = json.loads(token_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            count = len(item["generated_token_ids"])
            assert artifact["generated_token_ids"] == item["generated_token_ids"]
            assert receipt["restricted_artifact_sha256"] == sha256_file(token_path)
            assert receipt["generated_token_ids_sha256"] == token_hash(item["generated_token_ids"])
            assert receipt["readout"]["prefix_token_count"] == count
            assert receipt["readout"]["residual_artifact_sha256"] == sha256_file(
                tmp_path / "batch-residuals" / item["trial_id"] / f"residual-{count:04d}.pt"
            )
            assert receipt["classifier_status"] == "not_scored"
            assert len(receipt["readout"]["jlens_refusal_minus_compliance_trajectory"]) == 31
            assert receipt["finish_reason"] == ("eos" if horizon == 256 else "length")
