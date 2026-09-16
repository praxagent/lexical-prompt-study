from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from lexical_prompt_study import text_challenger as base
from lexical_prompt_study import text_challenger_semantic as sem


def records():
    result = []
    for core in range(15):
        for placement in base.PLACEMENTS:
            for family in base.FAMILIES:
                for label in (False, True):
                    result.append({"core": f"invented-core-{core}", "fold": core % 5,
                        "placement": placement, "family": family, "intent": "unsafe_direct",
                        "origin": "original_eos", "prompt": f"Invented {'blue' if label else 'green'} task {core}.",
                        "prefix_hash": [float(label)] + [0.] * 255,
                        "internal": [float(label)] * 31, "structural": [float(core)] * 6,
                        "scores": [{"requested_horizon": horizon, "binary_prediction": label,
                                    "right_censored": False, "observed_token_count": 12}
                                   for horizon in base.HORIZONS]})
    return result


def unit(index=0):
    value = np.zeros(384, dtype=np.float32)
    value[index] = 1.
    return value


def embeddings(data):
    return {base.sha(row["prompt"].encode()): unit(int(row["scores"][-1]["binary_prediction"]))
            for row in data}


def config(tmp):
    model = tmp / sem.REVISION
    model.mkdir()
    files = {}
    for name in sem.MODEL_FILES:
        (model / name).write_bytes(b"invented model metadata; never loaded")
        files[name] = sem.digest_file(model / name)
    return {"schema_version": "prompt-semantic-encoder-v1", "encoding": sem.ENCODING,
            "model_path": str(model), "model_files_sha256": files,
            "runtime": {"python": "test", "executable": "/invented/python", "versions": {
                key: "test" for key in ("torch", "transformers", "tokenizers", "numpy")}},
            "source_pins": sem.extraction_sources()}


def receipt():
    return {"schema_version": "prompt-semantic-receipt-v1", "prompt_sha256": "a" * 64,
            "packet_sha256": "b" * 64, "encoder_config_sha256": "c" * 64,
            "content_tokens": 511, "chunk_lengths": [510, 1], "embedding": unit().tolist()}


def validate(value):
    return sem.validate_receipt(value, prompt_sha="a" * 64, packet_sha="b" * 64, config_sha="c" * 64)


class Tokenizer:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append(kwargs)
        return self.ids


class FakeEncoder:
    calls = []

    def __init__(self, config):
        pass

    def encode(self, prompt):
        self.calls.append(prompt)
        return unit(), [len(prompt)]


def make_cache(tmp, data, *, benchmark=False):
    packet = sem.packet_from_records(data)
    conf = config(tmp)
    packet_path, config_path = tmp / "prompts.json", tmp / "config.json"
    packet_sha = base.immutable(packet_path, packet)
    config_sha = base.immutable(config_path, conf)
    with patch.object(sem, "encoding_runtime", return_value=conf["runtime"]):
        cache = sem.encode_packet(packet_path, packet_sha, config_path, config_sha,
                                 tmp / "cache", benchmark=benchmark, encoder_factory=FakeEncoder)
    return cache, {"packet_sha256": packet_sha}, (packet_path, packet_sha, config_path, config_sha)


def test_packet_has_only_sorted_deduplicated_prompts_and_fixed_provenance():
    packet = sem.packet_from_records(records())
    sem.validate_packet(packet)
    assert len(packet["prompts"]) == 30
    assert all(set(row) == {"prompt_sha256", "prompt"} for row in packet["prompts"])
    serialized = json.dumps(packet)
    assert '"scores"' not in serialized and '"internal"' not in serialized
    assert '"prefix_hash"' not in serialized and '"core"' not in serialized


@pytest.mark.parametrize("mutation", ("label", "prompt", "order", "source"))
def test_packet_rejects_labels_tampering_order_and_source(mutation):
    packet = sem.packet_from_records(records())
    if mutation == "label":
        packet["prompts"][0]["label"] = True
    elif mutation == "prompt":
        packet["prompts"][0]["prompt"] += " changed"
    elif mutation == "order":
        packet["prompts"].reverse()
    else:
        packet["source_pins"] = {}
    with pytest.raises(base.ChallengerError):
        sem.validate_packet(packet)


def test_full_token_coverage_has_no_truncation_overlap_or_loss():
    tokenizer = Tokenizer(list(range(1021)))
    chunks = sem.content_chunks(tokenizer, "Invented long text")
    assert list(map(len, chunks)) == [510, 510, 1]
    assert [value for chunk in chunks for value in chunk] == list(range(1021))
    assert tokenizer.calls == [{"add_special_tokens": False, "truncation": False}]


@pytest.mark.parametrize("ids", ([], [True], [-1], ["1"]))
def test_invalid_content_tokenization_is_rejected(ids):
    with pytest.raises(base.ChallengerError):
        sem.content_chunks(Tokenizer(ids), "Invented")


def test_pooling_is_chunk_normalized_token_weighted_then_document_normalized():
    actual = sem.pool_chunks([unit(0) * 9, unit(1) * 3], [510, 1])
    expected = unit(0) * 510 + unit(1)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("vectors,lengths", [
    ([np.zeros(384)], [1]), ([unit(), -unit()], [1, 1]),
    ([np.full(384, np.nan)], [1]), ([np.zeros(383)], [1]), ([unit()], [0]),
])
def test_pooling_rejects_empty_zero_nonfinite_and_wrong_dimension(vectors, lengths):
    with pytest.raises(base.ChallengerError):
        sem.pool_chunks(vectors, lengths)


def test_receipt_accepts_only_unit_finite_complete_coverage_vector():
    np.testing.assert_array_equal(validate(receipt()), unit())


@pytest.mark.parametrize("key,value", [
    ("prompt_sha256", "d" * 64), ("packet_sha256", "d" * 64),
    ("encoder_config_sha256", "d" * 64), ("content_tokens", 512),
    ("chunk_lengths", [509, 2]), ("embedding", [0.] * 384),
    ("embedding", [float("nan")] * 384), ("embedding", [1.] * 383),
])
def test_receipt_rejects_cross_input_binding_or_corrupt_vector(key, value):
    bad = receipt()
    bad[key] = value
    with pytest.raises(base.ChallengerError):
        validate(bad)


def test_model_manifest_and_exact_offline_revision_are_validated(tmp_path):
    conf = config(tmp_path)
    sem.validate_encoder_config(conf, check_runtime=False)
    (Path(conf["model_path"]) / "model.safetensors").write_bytes(b"different bytes")
    with pytest.raises(base.ChallengerError, match="model_file_drift"):
        sem.validate_encoder_config(conf, check_runtime=False)


def test_model_config_rejects_pooling_or_version_changes(tmp_path):
    conf = config(tmp_path)
    conf["encoding"] = dict(sem.ENCODING, pooling="mean")
    with pytest.raises(base.ChallengerError, match="config"):
        sem.validate_encoder_config(conf, check_runtime=False)


@pytest.mark.parametrize("extra", ("added_tokens.json", "adapter_config.json"))
def test_model_directory_rejects_unpinned_loader_inputs(tmp_path, extra):
    conf = config(tmp_path)
    (Path(conf["model_path"]) / extra).write_text("{}")
    with pytest.raises(base.ChallengerError, match="extra_or_missing_model_files"):
        sem.validate_encoder_config(conf, check_runtime=False)


def test_benchmark_is_first32_sorted_unique_hashes_and_contains_no_labels(tmp_path):
    data = [{"prompt": f"Invented phrase number {index}"} for index in range(40)]
    FakeEncoder.calls = []
    cache, _, _ = make_cache(tmp_path, data, benchmark=True)
    expected = sorted(data, key=lambda row: base.sha(row["prompt"].encode()))[:32]
    assert FakeEncoder.calls == [row["prompt"] for row in expected]
    assert cache["summary"]["encoded_unique_prompts"] == 32
    assert cache["summary"]["labels_or_outcomes_supplied_to_encoder"] is False
    assert cache["scope"] == "benchmark"


def test_complete_cache_joins_exact_prompt_population_and_resumes_without_model(tmp_path):
    data = records()
    cache, prep, args = make_cache(tmp_path, data)
    vectors = sem.load_cache(cache, prep, data)
    assert len(vectors) == 30
    with patch.object(sem, "encoding_runtime", return_value=cache["encoder_config"]["runtime"]):
        resumed = sem.encode_packet(*args, tmp_path / "cache", encoder_factory=lambda _: pytest.fail("model loaded"))
    assert resumed["summary"]["computed_this_call"] == 0
    assert resumed["receipt_sha256"] == cache["receipt_sha256"]


def test_partial_cache_never_fits(tmp_path):
    cache, prep, _ = make_cache(tmp_path, records(), benchmark=True)
    with pytest.raises(base.ChallengerError, match="cache"):
        sem.load_cache(cache, prep, records())


@pytest.mark.parametrize("mutation", ("missing", "extra", "record", "config", "header"))
def test_cache_rejects_missing_extra_or_tampered_receipt_lineage(tmp_path, mutation):
    data = records()
    cache, prep, args = make_cache(tmp_path, data)
    key = next(iter(cache["receipt_sha256"]))
    if mutation == "missing":
        del cache["receipt_sha256"][key]
    elif mutation == "extra":
        cache["receipt_sha256"]["0" * 64] = "0" * 64
    elif mutation == "record":
        (tmp_path / "cache/embeddings" / (key + ".json")).write_text("{}")
    elif mutation == "config":
        args[2].write_text("{}")
    else:
        (tmp_path / "cache/run.json").write_text("{}")
    with pytest.raises(base.ChallengerError):
        sem.load_cache(cache, prep, data)


def test_semantic_block_stays_unit_scaled_and_numeric_scalers_are_train_only():
    data = records()
    train_mask, test_mask = base.cell_masks(data, base.PLACEMENTS[0], 0)
    train = [r for r, keep in zip(data, train_mask, strict=True) if keep]
    test = deepcopy([r for r, keep in zip(data, test_mask, strict=True) if keep])
    for row in test:
        row["structural"] = [10**9] * 6
        row["prompt"] += " EXCLUSIVELYHELDOUTSEMANTICWORD"
    embed = {**embeddings(train), **embeddings(test)}
    matrices, fitted, scalers = sem.matrices_for_cell(train, test, embed)
    np.testing.assert_allclose(scalers["structural"].mean_, np.mean([r["structural"] for r in train], axis=0))
    assert all("EXCLUSIVELYHELDOUTSEMANTICWORD" not in v.vocabulary_ for v in fitted)
    assert matrices[sem.MODELS[1]][0].shape[1] - matrices[sem.MODELS[0]][0].shape[1] == 31
    np.testing.assert_array_equal(matrices[sem.MODELS[0]][0][:, -384:].toarray(),
                                  [embed[base.sha(r["prompt"].encode())] for r in train])


def test_two_new_matched_fits_emit_aggregates_only():
    data = records()
    result = sem.evaluate_cell(data, embeddings(data), base.PLACEMENTS[0], 0)
    assert result["status"] == "complete" and set(result["models"]) == set(sem.MODELS)
    assert result["test_rows"] == int(base.cell_masks(data, base.PLACEMENTS[0], 0)[1].sum())
    assert result["semantic_response_prefix_available"] is False
    raw = json.dumps(result)
    assert "Invented blue" not in raw and "invented-core" not in raw
    assert '"embedding"' not in raw and '"coefficients"' not in raw and '"predictions"' not in raw


def test_structural_stress_and_unknown_labels_use_original_masks():
    data = records()
    data[0]["scores"][-1]["binary_prediction"] = None
    result = sem.evaluate_cell(data, embeddings(records()), base.PLACEMENTS[0], 0, "structural_sham")
    train, test = base.cell_masks(data, base.PLACEMENTS[0], 0, "structural_sham")
    assert result["training_rows"] == int(train.sum()) and result["test_rows"] == int(test.sum())


def test_deadline_and_single_class_do_not_add_search_or_fallback_fits():
    data = records()
    embed = embeddings(data)
    result = sem.evaluate_cell(data, embed, base.PLACEMENTS[0], 0, max_seconds=0)
    assert all(row["status"] == "unavailable_deadline" for row in result["models"].values())
    for row in data:
        row["scores"][-1]["binary_prediction"] = True
    result = sem.evaluate_cell(data, embed, base.PLACEMENTS[0], 0)
    assert result["models"] == {} and result["status"] == "unavailable_training_classes_or_test_support"


def test_raw_outputs_cannot_be_written_inside_worktree():
    with pytest.raises(base.ChallengerError, match="inside_worktree"):
        sem.private_path(base.REPO / "synthetic-private.json")


def test_cli_rejects_bad_packet_hash_without_loading_encoder(capsys):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "packet.json").write_text('{"prompt":"private synthetic sentinel"}')
        with patch.object(sem, "environment"), patch.object(sem, "FrozenEncoder", side_effect=AssertionError):
            code = sem.main(["encode", "--packet", str(root / "packet.json"), "--packet-sha256", "0" * 64,
                "--encoder-config", str(root / "config.json"), "--encoder-config-sha256", "1" * 64,
                "--cache-root", str(root / "cache"), "--output", str(root / "result.json")])
        assert code == 1
        assert capsys.readouterr().out == '{"status":"prompt_semantic_rejected"}\n'
        assert not (root / "result.json").exists()
