from __future__ import annotations

from types import SimpleNamespace

import pytest

from lexical_prompt_study.continuation_gpu import PrefixGenerator, verify_continuation_parity


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)
    # Real causal attention and DynamicCache, small dimensions; 32 layers retain
    # the production readout topology without requiring weights or network.
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
    # Ensure the random model's ordinary path does not hit EOS. Tests explicitly
    # exercise EOS by substituting its next logits below.
    with torch.no_grad():
        model.lm_head.weight[96].zero_()
    value = SimpleNamespace(
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
    return value


def test_exact_prefix_cached_chunks_match_real_transformers_generate(runtime):
    torch = runtime.torch
    prompt = [1, 2, 3]
    original = [4] * 128
    input_ids = torch.tensor([[*prompt, *original]])
    with torch.inference_mode():
        expected = runtime.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=7,
            eos_token_id=96,
            pad_token_id=0,
            use_cache=True,
        )[0, input_ids.shape[1] :].tolist()
    generator = PrefixGenerator(runtime, prompt)
    first = generator(original, 3)
    second = generator([*original, *first["delta_token_ids"]], 4)
    actual = [*first["delta_token_ids"], *second["delta_token_ids"]]
    assert actual == expected
    assert original == [4] * 128
    assert set(generator.readouts_by_generated_tokens) == {"16", "32", "64", "128"}
    assert (
        len(generator.readout([*original, *actual])["jlens_refusal_minus_compliance_trajectory"])
        == 31
    )


def test_resume_eos_excluded_and_actual_terminal_readout(runtime, tmp_path):
    torch = runtime.torch
    prefix = [4] * 129
    generator = PrefixGenerator(runtime, [1, 2], residual_root=tmp_path / "restricted")
    generator.prepare(prefix)
    generator._next_logits = torch.full((1, 97), -100.0)
    generator._next_logits[0, 96] = 100.0
    result = generator(prefix, 127)
    assert result == {"delta_token_ids": [], "eos": True}
    readout = generator.readout(prefix)
    assert readout["prefix_token_count"] == 129
    assert len(readout["residual_artifact_sha256"]) == 64
    assert "256" not in generator.readouts_by_generated_tokens
    assert (tmp_path / "restricted" / "residual-0129.pt").stat().st_mode & 0o777 == 0o600
    payload = torch.load(tmp_path / "restricted" / "residual-0129.pt", weights_only=True)
    assert set(payload["residuals"]) == {8, 16, 19, 24, 30}
    assert all(tensor.dtype == torch.bfloat16 for tensor in payload["residuals"].values())
    assert "prompt_token_ids" not in payload


def test_reaches_256_captures_last_consumed_token_and_resumes(runtime):
    prefix = [4] * 128
    generator = PrefixGenerator(runtime, [1, 2])
    chunk = generator(prefix, 128)
    extended = [*prefix, *chunk["delta_token_ids"]]
    assert len(extended) == 256
    cached_readout = generator.readout(extended)
    resumed = PrefixGenerator(runtime, [1, 2])
    replayed = resumed.readout(extended)
    assert cached_readout["prefix_token_count"] == 256
    assert cached_readout["prefix_token_ids_sha256"] == replayed["prefix_token_ids_sha256"]
    assert cached_readout["jlens_refusal_minus_compliance_trajectory"] == pytest.approx(
        replayed["jlens_refusal_minus_compliance_trajectory"], abs=1e-6
    )
    assert generator(extended, 2) == resumed(extended, 2)


def test_fails_closed_on_changed_prefix_ceiling_eos_and_deadline(runtime):
    prefix = [4] * 128
    generator = PrefixGenerator(runtime, [1, 2])
    generator.prepare(prefix)
    with pytest.raises(ValueError, match="identity drift"):
        generator([5] * 128, 1)
    with pytest.raises(ValueError, match="1024-token"):
        generator(prefix, 897)
    with pytest.raises(ValueError, match="EOS"):
        PrefixGenerator(runtime, [1]).prepare([96] * 128)
    with pytest.raises(ValueError, match="outside frozen range"):
        PrefixGenerator(runtime, [1]).prepare([4] * 127)
    with pytest.raises(TimeoutError, match="deadline"):
        PrefixGenerator(runtime, [1], deadline_monotonic=0, clock=lambda: 1).prepare(prefix)


def test_residual_resume_rejects_different_prefix(runtime, tmp_path):
    root = tmp_path / "restricted"
    PrefixGenerator(runtime, [1], residual_root=root).prepare([4] * 128)
    with pytest.raises(ValueError, match="resume metadata drift"):
        PrefixGenerator(runtime, [1], residual_root=root).prepare([5] * 128)


def test_preserved_readouts_are_hash_checked_and_not_recomputed(runtime, tmp_path):
    root = tmp_path / "restricted"
    sink = []
    prefix = [4] * 128
    first = PrefixGenerator(runtime, [1], residual_root=root, readout_sink=sink.append)
    first.prepare(prefix)
    assert len(sink) == 4
    history = first.readouts_by_generated_tokens
    resumed_sink = []
    resumed = PrefixGenerator(
        runtime,
        [1],
        residual_root=root,
        preserved_readouts=history,
        readout_sink=resumed_sink.append,
    )
    resumed.prepare(prefix)
    assert resumed.readouts_by_generated_tokens == history
    assert not resumed_sink
    assert resumed.timing_summary["model_forward_count"] == 1
    assert resumed.timing_summary["generated_new_token_count"] == 0
    assert resumed.timing_summary["detector_readout_elapsed_seconds"] == 0
    bad = {key: dict(value) for key, value in history.items()}
    bad["128"]["residual_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preserved residual hash drift"):
        PrefixGenerator(runtime, [1], residual_root=root, preserved_readouts=bad).prepare(prefix)
    with pytest.raises(ValueError, match="preserved readout prefix drift"):
        PrefixGenerator(runtime, [1], residual_root=root, preserved_readouts=history).prepare(
            [5] * 128
        )


def test_orphan_future_readout_waits_for_exact_regenerated_prefix(runtime, tmp_path):
    root = tmp_path / "restricted"
    prefix = [4] * 128
    first = PrefixGenerator(runtime, [1], residual_root=root)
    original_chunk = first(prefix, 128)
    history = first.readouts_by_generated_tokens
    assert "256" in history
    # Simulate a crash after the 256 readout, before its token checkpoint.
    resumed = PrefixGenerator(runtime, [1], residual_root=root, preserved_readouts=history)
    resumed.prepare(prefix)
    assert "256" not in resumed.readouts_by_generated_tokens
    assert resumed(prefix, 128) == original_chunk
    assert resumed.readouts_by_generated_tokens["256"] == history["256"]
    bad_history = {key: dict(value) for key, value in history.items()}
    bad_history["256"]["prefix_token_ids_sha256"] = "0" * 64
    bad = PrefixGenerator(runtime, [1], residual_root=root, preserved_readouts=bad_history)
    bad.prepare(prefix)
    with pytest.raises(ValueError, match="preserved readout prefix drift"):
        bad(prefix, 128)


def test_pilot_parity_conditions_on_saved_prefix_without_regenerating_it(runtime, monkeypatch):
    torch = runtime.torch
    prompt = [1, 2, 3]
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        original = runtime.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=128,
            eos_token_id=96,
            pad_token_id=0,
            use_cache=True,
        )[0, len(prompt) :].tolist()
    assert len(original) == 128
    original_generate = runtime.model.generate
    trusted_calls = []

    def record_trusted_generate(*args, **kwargs):
        trusted_calls.append((kwargs["input_ids"].shape[1], kwargs["max_new_tokens"]))
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(runtime.model, "generate", record_trusted_generate)
    result = verify_continuation_parity(runtime, prompt, original, extension_tokens=2)
    assert result["status"] == "pass"
    assert result["trusted_input_prefix_unchanged"] and result["conditional_continuation_matches"]
    assert result["original_prefix_immutable"]
    assert result["parity_scope"] == "exact_saved_prefix_conditional"
    assert all("text" not in key for key in result)
    assert "original_prefix_token_ids" not in result
    different_saved_prefix = [*original]
    different_saved_prefix[0] = 5 if different_saved_prefix[0] != 5 else 6
    retained = list(different_saved_prefix)
    result = verify_continuation_parity(runtime, prompt, different_saved_prefix, extension_tokens=4)
    assert result["status"] == "pass"
    assert different_saved_prefix == retained
    assert result["adapter_continuation_token_count"] == 4
    assert result["split_boundary_new_tokens"] == 2
    assert trusted_calls == [(len(prompt) + 128, 2), (len(prompt) + 128, 4)]


def test_conditional_parity_detects_corrupt_continuation(runtime, monkeypatch):
    original_call = PrefixGenerator.__call__
    calls = 0

    def corrupt_second_chunk(self, prefix, count):
        nonlocal calls
        result = original_call(self, prefix, count)
        calls += 1
        if calls == 2:
            result = dict(result)
            result["delta_token_ids"] = list(result["delta_token_ids"])
            result["delta_token_ids"][0] = (result["delta_token_ids"][0] + 1) % 96
        return result

    monkeypatch.setattr(PrefixGenerator, "__call__", corrupt_second_chunk)
    result = verify_continuation_parity(runtime, [1, 2], [4] * 128, extension_tokens=4)
    assert result["status"] == "stop_conditional_parity_mismatch"
    assert not result["conditional_continuation_matches"]
    assert result["original_prefix_immutable"]


def test_conditional_parity_preserves_immediate_eos_semantics(runtime):
    torch = runtime.torch
    old_head = runtime.model.lm_head
    head = torch.nn.Linear(8, 97, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[96] = 100
    runtime.model.lm_head = head
    try:
        result = verify_continuation_parity(runtime, [1, 2], [4] * 128, extension_tokens=4)
        assert result["status"] == "pass"
        assert result["trusted_eos"] and result["adapter_eos"]
        assert result["trusted_continuation_token_count"] == 0
        assert result["adapter_continuation_token_count"] == 0
        assert result["original_prefix_immutable"]
    finally:
        runtime.model.lm_head = old_head


def test_nontrivial_generation_policy_rejected(runtime):
    old = runtime.model.generation_config.repetition_penalty
    runtime.model.generation_config.repetition_penalty = 1.1
    try:
        with pytest.raises(ValueError, match="plain-greedy"):
            PrefixGenerator(runtime, [1])
    finally:
        runtime.model.generation_config.repetition_penalty = old
