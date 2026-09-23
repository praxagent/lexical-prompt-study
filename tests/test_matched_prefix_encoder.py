"""Invented label-free documents and untrained encoder qualification only."""

from types import SimpleNamespace
import struct

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

from lexical_prompt_study import matched_prefix_encoder as subject


@pytest.fixture(autouse=True)
def deterministic_cpu():
    before = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])


def packets(unavailable=False):
    common = {
        "rendered_prompt_utf8": b"Invented prompt",
        "prompt_token_ids": [1, 2, 3],
        "prefix_utf8": b"12, 34,",
        "prefix_token_ids": list(range(8)),
        "completed_fields": 2,
        "prefix_known_error": False,
    }
    return [
        {
            "schema_version": "a186-common-prefix-v1",
            "observation_id": "invented-row",
            "common": None if unavailable else common,
            "features_sha256": "a" * 64,
        }
    ]


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    padding_side, vocab_size = "right", 64

    def __init__(self, count=4):
        self.count = count

    def num_special_tokens_to_add(self, *, pair):
        assert pair is False
        return 2

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return [3] * (4 if text == "A short invented sentence." else self.count)

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": False,
            "return_attention_mask": True,
            "return_token_type_ids": True,
        }
        ids = [1, *self.encode(text, add_special_tokens=False, truncation=False), 2]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "token_type_ids": [0] * len(ids),
        }


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(384))
        self.config = SimpleNamespace(
            model_type="bert",
            hidden_size=384,
            max_position_embeddings=512,
            _attn_implementation="eager",
            output_hidden_states=False,
            output_attentions=False,
        )
        self.inputs, self.mode, self.error = [], None, None
        self.eval()

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        assert kwargs == {
            "output_hidden_states": False,
            "output_attentions": False,
            "return_dict": True,
        }
        assert torch.is_inference_mode_enabled()
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert torch.equal(token_type_ids, torch.zeros_like(input_ids))
        self.inputs.append(input_ids.detach().clone())
        if self.error is not None:
            raise self.error
        hidden = (
            torch.arange(1, 385, dtype=torch.float32)[None, None, :]
            .expand(1, input_ids.shape[1], 384)
            .clone()
        )
        if self.mode == "nan":
            hidden[0, 0, 0] = float("nan")
        elif self.mode == "dtype":
            hidden = hidden.double()
        elif self.mode == "shape":
            hidden = hidden[:, :, :-1]
        elif self.mode == "mutate":
            input_ids[0, 0] += 1
        return SimpleNamespace(last_hidden_state=hidden)


def run(root, *, model=None, tokenizer=None, packet=None):
    model = Encoder() if model is None else model
    result = subject.encode_packets(
        model,
        Tokenizer() if tokenizer is None else tokenizer,
        packets() if packet is None else packet,
        binding="b" * 64,
        directory=root,
    )
    return result, model


def clean(model):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_exact_two_views_and_missing_slot_retained(tmp_path):
    source = packets()
    unavailable = packets(True)[0]
    unavailable["observation_id"] = "invented-missing"
    source.append(unavailable)
    planned = subject.document_plan(source)
    assert [d["text"] for d in planned] == [
        "Prompt:\nInvented prompt",
        "Assistant prefix:\n12, 34,",
        None,
        None,
    ]
    result, model = run(tmp_path / "run", packet=source)
    assert result["entries"] == len(model.inputs) == 2
    assert result["embeddings"]["invented-missing"] == {"prompt": None, "prefix": None}
    assert all(len(x) == 1536 for x in result["embeddings"]["invented-row"].values())
    assert subject.load_encoding(tmp_path / "run", source, binding="b" * 64) == result
    clean(model)


def test_empty_decoded_prefix_retains_explicit_header():
    source = packets()
    source[0]["common"]["prefix_utf8"] = b""
    assert subject.document_plan(source)[1]["text"] == "Assistant prefix:\n"


@pytest.mark.parametrize(
    "mutation",
    [
        "future",
        "label",
        "internal",
        "long_prefix",
        "bool_id",
        "bool_count",
        "bad_utf8",
        "duplicate",
    ],
)
def test_common_packet_rejects_forbidden_or_malformed_fields(mutation):
    source = packets()
    common = source[0]["common"]
    if mutation in ("future", "label", "internal"):
        common[mutation] = 1
    elif mutation == "long_prefix":
        common["prefix_token_ids"].append(9)
    elif mutation == "bool_id":
        common["prefix_token_ids"][0] = False
    elif mutation == "bool_count":
        common["completed_fields"] = True
    elif mutation == "bad_utf8":
        common["prefix_utf8"] = b"\xff"
    else:
        source.append(source[0])
    with pytest.raises((ValueError, UnicodeDecodeError)):
        subject.wire_packets(source)


def test_chunk_coverage_and_no_truncation_or_cross_chunk_padding(tmp_path):
    result, model = run(tmp_path / "run", tokenizer=Tokenizer(511))
    assert result["entries"] == 4
    assert [x.shape[1] for x in model.inputs] == [512, 3, 512, 3]
    assert all(x[0, 0] == 1 and x[0, -1] == 2 for x in model.inputs)


def test_oversize_stops_before_any_forward_with_unattempted_plan(tmp_path):
    model = Encoder()
    with pytest.raises(ValueError, match="chunk_bound"):
        run(tmp_path / "run", model=model, tokenizer=Tokenizer(4081))
    assert model.inputs == []
    failure = subject._read(tmp_path / "run/failure.json")
    assert failure["entries"] == 0
    assert [r["status"] for r in failure["records"]] == ["unattempted", "unattempted"]


@pytest.mark.parametrize("mode", ["nan", "dtype", "shape", "mutate"])
def test_invalid_return_never_becomes_embedding(tmp_path, mode):
    model = Encoder()
    model.mode = mode
    with pytest.raises(ValueError):
        run(tmp_path / "run", model=model)
    assert len(model.inputs) == 1
    failure = subject._read(tmp_path / "run/document_000_prompt/chunk_00/failure.json")
    assert failure["entries"] == 1 and failure["model_call_returned"] is True
    assert not (tmp_path / "run/terminal.json").exists()
    clean(model)


@pytest.mark.parametrize("error", [RuntimeError("invented"), KeyboardInterrupt(), SystemExit(9)])
def test_interrupt_preserves_original_exception_and_owned_hook_cleanup(tmp_path, error):
    model = Encoder()
    model.error = error
    with pytest.raises(type(error)) as caught:
        run(tmp_path / "run", model=model)
    assert caught.value is error and len(model.inputs) == 1
    failure = subject._read(tmp_path / "run/failure.json")
    assert failure["entries"] == 1
    clean(model)


def test_normalized_weighted_pool_is_scale_independent():
    a, b = np.zeros(384, np.float32), np.zeros(384, np.float32)
    a[0], b[1] = 2, 9
    raw = subject.pool_chunks([a.tobytes(), b.tobytes()], [1, 3])
    value = np.asarray(struct.unpack("<384f", raw))
    assert np.allclose(value[:2], np.asarray([1, 3]) / np.sqrt(10), atol=1e-7)
    assert np.count_nonzero(value[2:]) == 0


@pytest.mark.parametrize("site", ["pooled", "cls", "entry", "extra", "bool_chunk"])
def test_replay_rejects_tampering(tmp_path, site):
    root = tmp_path / "run"
    run(root)
    if site == "pooled":
        (root / "document_000_prompt/embedding.fp32").write_bytes(b"\0" * 1536)
    elif site == "cls":
        (root / "document_000_prompt/chunk_00/cls.fp32").write_bytes(b"\0" * 1536)
    elif site == "entry":
        p = root / "document_000_prompt/chunk_00/entry.json"
        value = subject._read(p)
        value["entry_number"] = True
        p.write_bytes(subject.e.canonical(value))
    elif site == "bool_chunk":
        p = root / "prepared.json"
        value = subject._read(p)
        value["documents"][0]["content_token_ids"][0] = 1
        value["documents"][0]["chunks"][0][0] = True
        p.write_bytes(subject.e.canonical(value))
    else:
        (root / "unclaimed").write_text("invented")
    with pytest.raises(ValueError):
        subject.load_encoding(root, packets(), binding="b" * 64)


def test_publication_after_terminal_does_not_prove_normal_return(tmp_path, monkeypatch):
    publish = subject.e._publish

    def fail(path, raw):
        publish(path, raw)
        if path.name == "terminal.json":
            raise OSError("invented publication interruption")

    monkeypatch.setattr(subject.e, "_publish", fail)
    with pytest.raises(OSError):
        run(tmp_path / "run")
    with pytest.raises(ValueError):
        subject.load_encoding(tmp_path / "run", packets(), binding="b" * 64)


def test_one_shot_directory_is_never_retried(tmp_path):
    model = Encoder()
    run(tmp_path / "run", model=model)
    with pytest.raises(FileExistsError):
        run(tmp_path / "run", model=model)
    assert len(model.inputs) == 2


def test_untrained_bert_cls_bytes_match_saved_reference(tmp_path):
    torch.manual_seed(186)
    config = BertConfig(
        vocab_size=64,
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    config._attn_implementation = "eager"
    model = BertModel(config).float().eval()
    result, _ = run(tmp_path / "run", model=model)
    assert result["entries"] == 2
    inputs = torch.tensor([[1, 3, 3, 3, 3, 2]])
    with torch.inference_mode():
        reference = (
            model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                token_type_ids=torch.zeros_like(inputs),
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            .last_hidden_state[0, 0]
            .numpy()
            .tobytes()
        )
    assert (tmp_path / "run/document_000_prompt/chunk_00/cls.fp32").read_bytes() == reference
    clean(model)
