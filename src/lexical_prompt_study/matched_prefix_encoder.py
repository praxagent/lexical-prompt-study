"""A186 label-free semantic encoding with counted, retained chunk forwards.

This module never receives labels, generated suffixes or internal vectors. Native
asset loading and process supervision belong to the frozen outer caller.
"""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
import struct
import sys

from . import landmark_evidence as e
from . import prefix_only_capture as capture

SCHEMA = "a186-semantic-v1"
POLICY = {
    "model_id": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "dimensions": 384,
    "content_tokens_per_chunk": 510,
    "maximum_chunks_per_document": 8,
    "maximum_documents": 224,
    "maximum_forward_entries": 1792,
    "batch_size": 1,
    "pooling": "cls_l2_content_weighted_mean_l2",
    "dtype": "float32",
    "device": "cpu",
    "attention": "eager",
    "threads": 4,
    "seed": 20260915,
    "deduplication": False,
}
COMMON = {
    "rendered_prompt_utf8",
    "prompt_token_ids",
    "prefix_utf8",
    "prefix_token_ids",
    "completed_fields",
    "prefix_known_error",
}


def require(ok, reason):
    if not ok:
        raise ValueError("a186_encoder_" + reason)


def source_sha256():
    return e.sha256(Path(__file__).read_bytes())


def runtime_descriptor():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "numpy",
                "safetensors",
                "huggingface-hub",
            )
        },
    }


def wire_packets(packets):
    """Snapshot only the strict allowed common-time fields into JSON values."""
    require(type(packets) is list and 1 <= len(packets) <= 112, "packet_count")
    result, seen = [], set()
    for packet in packets:
        require(
            type(packet) is dict
            and set(packet) == {"schema_version", "observation_id", "common", "features_sha256"}
            and packet["schema_version"] == "a186-common-prefix-v1",
            "packet_schema",
        )
        key = packet["observation_id"]
        e._str(key)
        require(key not in seen, "duplicate_observation")
        seen.add(key)
        e._hash(packet["features_sha256"])
        common = packet["common"]
        if common is not None:
            require(type(common) is dict and set(common) == COMMON, "common_fields")
            common = dict(common)
            for field in ("rendered_prompt_utf8", "prefix_utf8"):
                value = common[field]
                require(type(value) in (str, bytes), "common_text")
                common[field] = value.decode("utf-8") if type(value) is bytes else value
                common[field].encode("utf-8")
            require(
                0 < len(common["rendered_prompt_utf8"]) <= 65536
                and len(common["prefix_utf8"]) <= 65536,
                "text_bound",
            )
            for field in ("prompt_token_ids", "prefix_token_ids"):
                require(type(common[field]) in (list, tuple), "id_sequence")
                common[field] = list(common[field])
                e._ids(common[field], 128256, nonempty=True)
            require(
                len(common["prompt_token_ids"]) <= 256 and len(common["prefix_token_ids"]) == 8,
                "common_token_boundary",
            )
            e._int(common["completed_fields"])
            require(type(common["prefix_known_error"]) is bool, "prefix_flag")
        result.append({**packet, "common": common})
    return e._snapshot(result)


def document_plan(packets):
    packets = wire_packets(packets)
    result = []
    for i, packet in enumerate(packets):
        for kind, field, header in (
            ("prompt", "rendered_prompt_utf8", "Prompt:\n"),
            ("prefix", "prefix_utf8", "Assistant prefix:\n"),
        ):
            common = packet["common"]
            result.append(
                {
                    "document_id": f"document_{i:03d}_{kind}",
                    "observation_id": packet["observation_id"],
                    "kind": kind,
                    "text": None if common is None else header + common[field],
                    "features_sha256": packet["features_sha256"],
                }
            )
    return result


def tokenizer_layout(tokenizer):
    ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    require(
        all(type(x) is int and x >= 0 for x in ids)
        and len(set(ids)) == 3
        and tokenizer.padding_side == "right"
        and tokenizer.num_special_tokens_to_add(pair=False) == 2,
        "native_layout",
    )
    text = "A short invented sentence."
    content = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    e._ids(content, tokenizer.vocab_size, nonempty=True)
    native = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    require(
        native["input_ids"] == [ids[0], *content, ids[1]]
        and native["attention_mask"] == [1] * (len(content) + 2)
        and native["token_type_ids"] == [0] * (len(content) + 2),
        "native_layout_probe",
    )
    return ids


def prepare_documents(packets, tokenizer):
    special = tokenizer_layout(tokenizer)
    documents = document_plan(packets)
    for document in documents:
        text = document["text"]
        if text is None:
            document.update(content_token_ids=None, chunks=None)
            continue
        values = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        e._ids(values, tokenizer.vocab_size, nonempty=True)
        chunks = [values[i : i + 510] for i in range(0, len(values), 510)]
        require(len(chunks) <= 8 and sum(map(len, chunks)) == len(values), "chunk_bound")
        document.update(content_token_ids=values, chunks=chunks)
    return {
        "schema_version": SCHEMA,
        "policy": POLICY,
        "packets_sha256": e.object_hash(wire_packets(packets)),
        "special_token_ids": special,
        "documents": documents,
    }


def pool_chunks(raw_vectors, lengths):
    import numpy as np

    require(
        bool(lengths)
        and len(lengths) == len(raw_vectors)
        and all(type(n) is int and 1 <= n <= 510 for n in lengths),
        "pool_lengths",
    )
    values = np.asarray([struct.unpack("<384f", raw) for raw in raw_vectors], dtype=np.float32)
    require(np.isfinite(values).all(), "finite_chunks")
    norms = np.linalg.norm(values, axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), "nonzero_chunks")
    pooled = np.average(
        values / norms[:, None], axis=0, weights=np.asarray(lengths, dtype=np.float32)
    )
    norm = np.linalg.norm(pooled)
    require(np.isfinite(pooled).all() and np.isfinite(norm) and norm > 0, "finite_pool")
    return np.asarray(pooled / norm, dtype="<f4").tobytes()


def _preflight(model, torch):
    require(isinstance(model, torch.nn.Module) and not hasattr(model, "_orig_mod"), "model")
    require(all(not m.training for m in model.modules()), "evaluation_mode")
    require(
        sys.byteorder == "little"
        and torch.get_num_threads() == 4
        and torch.are_deterministic_algorithms_enabled()
        and not torch.cuda.is_initialized()
        and not torch.is_autocast_enabled("cpu"),
        "runtime",
    )
    require(
        all(
            x.device.type == "cpu"
            and not x.is_complex()
            and (not x.is_floating_point() or x.dtype == torch.float32)
            for x in [*model.parameters(), *model.buffers()]
        ),
        "model_tensors",
    )
    config = model.config
    require(
        config.model_type == "bert"
        and config.hidden_size == 384
        and config.max_position_embeddings == 512
        and config._attn_implementation == "eager"
        and config.output_hidden_states is False
        and config.output_attentions is False,
        "model_geometry",
    )
    capture._hook_audit(torch, model)


def _chunk(model, torch, root, attempt, entries):
    e._publish(root / "attempt.json", e.canonical(attempt))
    entered, returned, handle = 0, False, None
    inputs = {
        name: torch.tensor([attempt[name]], dtype=torch.long)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }

    def mark(_module, _args, kwargs):
        nonlocal entered
        entered += 1
        entries[0] += 1
        require(entered == 1 and entries[0] <= 1792, "entry_ceiling")
        for name, value in inputs.items():
            require(torch.equal(kwargs[name], value), "entered_input")
        e._publish(
            root / "entry.json",
            e.canonical({"attempt_sha256": e.object_hash(attempt), "entry_number": entries[0]}),
        )

    original = None
    try:
        _preflight(model, torch)
        handle = model.register_forward_pre_hook(mark, with_kwargs=True)
        with torch.inference_mode():
            output = model(
                **inputs, output_hidden_states=False, output_attentions=False, return_dict=True
            )
        returned = True
        for name, value in inputs.items():
            require(value.tolist() == [attempt[name]], "returned_input")
        require(entered == 1, "single_entry")
        value = output.last_hidden_state
        require(
            value.dtype == torch.float32
            and value.device.type == "cpu"
            and tuple(value.shape) == (1, len(attempt["input_ids"]), 384)
            and bool(torch.isfinite(value).all()),
            "returned_hidden",
        )
        raw = value[0, 0].detach().contiguous().numpy().tobytes()
    except BaseException as exc:
        original = exc
        raise
    finally:
        cleanup = None
        try:
            if handle is not None:
                handle.remove()
            capture._hook_audit(torch, model)
        except BaseException as exc:
            cleanup = exc
        if original is not None or cleanup is not None:
            try:
                e._publish(
                    root / "failure.json",
                    e.canonical(
                        {
                            "attempt_sha256": e.object_hash(attempt),
                            "entries": entered,
                            "model_call_returned": returned,
                            "failure_type": None if original is None else type(original).__name__,
                            "cleanup_failure_type": None
                            if cleanup is None
                            else type(cleanup).__name__,
                        }
                    ),
                )
            except BaseException:
                pass
        if original is None and cleanup is not None:
            raise cleanup
    e._publish(root / "cls.fp32", raw)
    result = {
        "attempt_sha256": e.object_hash(attempt),
        "entry_sha256": e.sha256(e._read_file(root / "entry.json")),
        "cls_sha256": e.sha256(raw),
        "model_call_returned": True,
    }
    e._publish(root / "result.json", e.canonical(result))
    return raw, e.object_hash(result)


def encode_packets(model, tokenizer, packets, *, binding, directory):
    import torch

    packets = wire_packets(packets)
    e._hash(binding)
    _preflight(model, torch)
    root = e._directory(directory, create=True)
    entries = [0]
    records = [
        {"document_id": d["document_id"], "status": "unattempted", "result_sha256": None}
        for d in document_plan(packets)
    ]
    header = {
        "schema_version": SCHEMA,
        "source_sha256": source_sha256(),
        "binding_sha256": binding,
        "packets_sha256": e.object_hash(packets),
        "policy": POLICY,
    }
    e._publish(root / "run.json", e.canonical(header))
    e._publish(root / "packets.json", e.canonical(packets))
    prepared = None
    try:
        prepared = prepare_documents(packets, tokenizer)
        e._publish(root / "prepared.json", e.canonical(prepared))
        records = [
            {"document_id": d["document_id"], "status": "unattempted", "result_sha256": None}
            for d in prepared["documents"]
        ]
        for document, record in zip(prepared["documents"], records, strict=True):
            directory = e._directory(root / document["document_id"], create=True)
            attempt = {
                "run_sha256": e.object_hash(header),
                "prepared_sha256": e.object_hash(prepared),
                "document_sha256": e.object_hash(document),
            }
            e._publish(directory / "attempt.json", e.canonical(attempt))
            hashes, vectors = [], []
            if document["chunks"] is not None:
                for i, chunk in enumerate(document["chunks"]):
                    chunk_root = e._directory(directory / f"chunk_{i:02d}", create=True)
                    ids = [
                        prepared["special_token_ids"][0],
                        *chunk,
                        prepared["special_token_ids"][1],
                    ]
                    call = {
                        "document_attempt_sha256": e.object_hash(attempt),
                        "chunk_index": i,
                        "input_ids": ids,
                        "attention_mask": [1] * len(ids),
                        "token_type_ids": [0] * len(ids),
                    }
                    raw, digest = _chunk(model, torch, chunk_root, call, entries)
                    vectors.append(raw)
                    hashes.append(digest)
                pooled = pool_chunks(vectors, list(map(len, document["chunks"])))
                e._publish(directory / "embedding.fp32", pooled)
            else:
                pooled = None
            result = {
                "attempt_sha256": e.object_hash(attempt),
                "chunk_result_sha256": hashes,
                "embedding_sha256": None if pooled is None else e.sha256(pooled),
                "status": "unavailable" if pooled is None else "completed",
            }
            e._publish(directory / "result.json", e.canonical(result))
            record.update(status=result["status"], result_sha256=e.object_hash(result))
        terminal = {
            "schema_version": SCHEMA,
            "run_sha256": e.object_hash(header),
            "status": "completed",
            "records": records,
            "entries": entries[0],
            "failure_type": None,
        }
        e._publish(root / "terminal.json", e.canonical(terminal))
        return load_encoding(root, packets, binding=binding)
    except BaseException as exc:
        try:
            e._publish(
                root / "failure.json",
                e.canonical(
                    {
                        "run_sha256": e.object_hash(header),
                        "entries": entries[0],
                        "records": records,
                        "failure_type": type(exc).__name__,
                    }
                ),
            )
        except BaseException:
            pass
        raise


def _read(path):
    return e._read_json(e._read_file(path))


def load_encoding(directory, packets, *, binding):
    """Replay saved inputs, CLS bytes and pooling; never call a tokenizer/model."""
    packets = wire_packets(packets)
    root = e._directory(directory)
    header = {
        "schema_version": SCHEMA,
        "source_sha256": source_sha256(),
        "binding_sha256": binding,
        "packets_sha256": e.object_hash(packets),
        "policy": POLICY,
    }
    require(e.canonical(_read(root / "run.json")) == e.canonical(header), "header")
    require(e.canonical(_read(root / "packets.json")) == e.canonical(packets), "packet_binding")
    prepared = _read(root / "prepared.json")
    require(
        set(prepared)
        == {"schema_version", "policy", "packets_sha256", "special_token_ids", "documents"}
        and prepared["schema_version"] == SCHEMA
        and e.canonical(prepared["policy"]) == e.canonical(POLICY)
        and prepared["packets_sha256"] == e.object_hash(packets),
        "prepared",
    )
    special = prepared["special_token_ids"]
    require(
        type(special) is list
        and len(special) == len(set(special)) == 3
        and all(type(x) is int and x >= 0 for x in special),
        "prepared_specials",
    )
    planned = document_plan(packets)
    require(len(prepared["documents"]) == len(planned), "prepared_count")
    require(
        {p.name for p in root.iterdir()}
        == {"run.json", "packets.json", "prepared.json", "terminal.json"}
        | {d["document_id"] for d in planned},
        "root_inventory",
    )
    entries, records, embeddings = 0, [], {}
    for document, expected in zip(prepared["documents"], planned, strict=True):
        require(
            set(document) == set(expected) | {"content_token_ids", "chunks"}
            and e.canonical({k: document[k] for k in expected}) == e.canonical(expected),
            "document",
        )
        values, chunks = document["content_token_ids"], document["chunks"]
        if document["text"] is None:
            require(values is None and chunks is None, "unavailable_content")
        else:
            e._ids(values, 30522, nonempty=True)
            require(
                e.canonical(chunks)
                == e.canonical([values[i : i + 510] for i in range(0, len(values), 510)])
                and len(chunks) <= 8,
                "chunk_coverage",
            )
        directory = e._directory(root / document["document_id"])
        attempt = {
            "run_sha256": e.object_hash(header),
            "prepared_sha256": e.object_hash(prepared),
            "document_sha256": e.object_hash(document),
        }
        require(
            e.canonical(_read(directory / "attempt.json")) == e.canonical(attempt),
            "document_attempt",
        )
        chunk_names = set() if chunks is None else {f"chunk_{i:02d}" for i in range(len(chunks))}
        require(
            {p.name for p in directory.iterdir()}
            == {"attempt.json", "result.json"}
            | chunk_names
            | (set() if chunks is None else {"embedding.fp32"}),
            "document_inventory",
        )
        hashes, vectors = [], []
        for i, chunk in enumerate(chunks or []):
            child = e._directory(directory / f"chunk_{i:02d}")
            require(
                {p.name for p in child.iterdir()}
                == {"attempt.json", "entry.json", "cls.fp32", "result.json"},
                "chunk_inventory",
            )
            ids = [special[0], *chunk, special[1]]
            call = {
                "document_attempt_sha256": e.object_hash(attempt),
                "chunk_index": i,
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "token_type_ids": [0] * len(ids),
            }
            require(
                e.canonical(_read(child / "attempt.json")) == e.canonical(call), "chunk_attempt"
            )
            entries += 1
            entry = {"attempt_sha256": e.object_hash(call), "entry_number": entries}
            require(e.canonical(_read(child / "entry.json")) == e.canonical(entry), "chunk_entry")
            raw = e._read_file(child / "cls.fp32")
            require(len(raw) == 384 * 4, "cls_shape")
            result = {
                "attempt_sha256": e.object_hash(call),
                "entry_sha256": e.object_hash(entry),
                "cls_sha256": e.sha256(raw),
                "model_call_returned": True,
            }
            require(
                e.canonical(_read(child / "result.json")) == e.canonical(result), "chunk_result"
            )
            vectors.append(raw)
            hashes.append(e.object_hash(result))
        pooled = None if chunks is None else pool_chunks(vectors, list(map(len, chunks)))
        if pooled is not None:
            require(e._read_file(directory / "embedding.fp32") == pooled, "pooled_bytes")
        result = {
            "attempt_sha256": e.object_hash(attempt),
            "chunk_result_sha256": hashes,
            "embedding_sha256": None if pooled is None else e.sha256(pooled),
            "status": "unavailable" if pooled is None else "completed",
        }
        require(
            e.canonical(_read(directory / "result.json")) == e.canonical(result), "document_result"
        )
        records.append(
            {
                "document_id": document["document_id"],
                "status": result["status"],
                "result_sha256": e.object_hash(result),
            }
        )
        embeddings.setdefault(document["observation_id"], {})[document["kind"]] = pooled
    terminal = {
        "schema_version": SCHEMA,
        "run_sha256": e.object_hash(header),
        "status": "completed",
        "records": records,
        "entries": entries,
        "failure_type": None,
    }
    require(
        entries <= 1792 and e.canonical(_read(root / "terminal.json")) == e.canonical(terminal),
        "terminal",
    )
    return {
        "terminal_sha256": e.object_hash(terminal),
        "entries": entries,
        "embeddings": embeddings,
    }
