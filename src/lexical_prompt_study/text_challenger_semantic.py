"""Frozen prompt-semantic augmentation with separate private extraction and fits.

Import is inert. Encoding consumes a prompt-only packet and never receives
labels, response text, prefix hashes, internal readouts, or fitted classifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

from . import text_challenger as base

PROTOCOL = "plans/text_challenger_semantic_20260915.md"
MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt"}
MODELS = ("prompt_prefix_hash_semantic", "prompt_prefix_hash_semantic_and_jlens")
ENCODING = {
    "schema_version": "prompt-semantic-encoding-v1", "model_id": MODEL_ID,
    "revision": REVISION, "dimensions": 384, "content_tokens_per_chunk": 510,
    "overlap_tokens": 0, "special_tokens_per_chunk": 2, "pooling": "cls",
    "chunk_l2_normalize": True, "pool_weight": "content_token_count",
    "document_l2_normalize": True, "retrieval_instruction": None,
    "full_prompt_coverage": True, "dtype": "float32", "device": "cpu",
    "attention_implementation": "eager", "deterministic_algorithms": True,
    "eval_mode": True, "gradient_enabled": False, "batch_size": 4,
    "threads": 4, "benchmark_prompt_count": 32, "seed": 20260915,
}


def require(ok, code):
    if not ok:
        raise base.ChallengerError("semantic_" + code)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def extraction_sources():
    paths = [Path(__file__), Path(base.__file__), base.REPO / PROTOCOL, base.REPO / base.PROTOCOL,
             base.REPO / "tests/test_text_challenger_semantic.py"]
    return {str(path.relative_to(base.REPO)): digest_file(path) for path in paths}


def encoding_runtime():
    return {"python": platform.python_version(), "executable": os.path.realpath(sys.executable),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "numpy")}}


def private_path(path):
    path = Path(path).absolute()
    require(not path.is_relative_to(base.REPO), "private_path_inside_worktree")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "private_symlink")
    return path


def packet_from_records(records):
    prompts = {}
    for record in records:
        prompt = record["prompt"]
        require(type(prompt) is str and 0 < len(prompt) <= base.CONTRACT["max_document_characters"],
                "prompt_bounds")
        key = base.sha(prompt.encode())
        require(prompts.setdefault(key, prompt) == prompt, "prompt_hash_collision")
    require(bool(prompts), "empty_packet")
    return {"schema_version": "prompt-semantic-packet-v1", "encoding": ENCODING,
            "source_pins": extraction_sources(),
            "prompts": [{"prompt_sha256": key, "prompt": prompts[key]} for key in sorted(prompts)]}


def validate_packet(packet):
    require(type(packet) is dict and set(packet) == {"schema_version", "encoding", "source_pins", "prompts"}
            and packet["schema_version"] == "prompt-semantic-packet-v1"
            and packet["encoding"] == ENCODING and packet["source_pins"] == extraction_sources(), "packet")
    require(type(packet["prompts"]) is list and bool(packet["prompts"]), "packet_prompts")
    keys = []
    for row in packet["prompts"]:
        require(type(row) is dict and set(row) == {"prompt_sha256", "prompt"}, "packet_row")
        text = row["prompt"]
        require(type(text) is str and 0 < len(text) <= base.CONTRACT["max_document_characters"]
                and row["prompt_sha256"] == base.sha(text.encode()), "packet_prompt_binding")
        keys.append(row["prompt_sha256"])
    require(keys == sorted(set(keys)), "packet_order")


def validate_encoder_config(config, *, check_runtime=True, check_files=True):
    require(type(config) is dict and set(config) == {
        "schema_version", "encoding", "model_path", "model_files_sha256", "runtime", "source_pins"
    } and config["schema_version"] == "prompt-semantic-encoder-v1"
        and config["encoding"] == ENCODING and config["source_pins"] == extraction_sources(), "config")
    path = Path(config["model_path"])
    require(path.is_absolute() and path.name == REVISION, "model_path_revision")
    files = config["model_files_sha256"]
    require(type(files) is dict and set(files) == MODEL_FILES
            and all(type(v) is str and len(v) == 64 and set(v) <= set("0123456789abcdef")
                    for v in files.values()), "model_file_manifest")
    runtime = config["runtime"]
    require(type(runtime) is dict and set(runtime) == {"python", "executable", "versions"}
            and type(runtime["versions"]) is dict
            and set(runtime["versions"]) == {"torch", "transformers", "tokenizers", "numpy"}
            and all(type(v) is str and bool(v) for v in runtime["versions"].values())
            and type(runtime["python"]) is str and type(runtime["executable"]) is str, "runtime_schema")
    if check_runtime:
        require(runtime == encoding_runtime(), "encoding_runtime_drift")
    if check_files:
        require(path.is_dir() and {item.name for item in path.iterdir()} == MODEL_FILES,
                "extra_or_missing_model_files")
        require(all((path / name).is_file() and digest_file(path / name) == value
                    for name, value in files.items()), "model_file_drift")


def content_chunks(tokenizer, prompt):
    require(type(prompt) is str and 0 < len(prompt) <= base.CONTRACT["max_document_characters"], "prompt_bounds")
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    require(type(ids) is list and bool(ids) and all(type(i) is int and i >= 0 for i in ids), "content_tokens")
    chunks = [ids[start:start + 510] for start in range(0, len(ids), 510)]
    require(sum(map(len, chunks)) == len(ids), "full_coverage")
    return chunks


def pool_chunks(vectors, lengths):
    import numpy as np
    values = np.asarray(vectors, dtype=np.float32)
    require(values.shape == (len(lengths), 384) and bool(lengths)
            and all(type(v) is int and 1 <= v <= 510 for v in lengths)
            and np.isfinite(values).all(), "chunk_vectors")
    norms = np.linalg.norm(values, axis=1)
    require(np.isfinite(norms).all() and np.all(norms > 0), "zero_chunk_vector")
    pooled = np.average(values / norms[:, None], axis=0, weights=np.asarray(lengths, dtype=np.float32))
    norm = np.linalg.norm(pooled)
    require(np.isfinite(pooled).all() and np.isfinite(norm) and norm > 0, "zero_document_vector")
    return np.asarray(pooled / norm, dtype=np.float32)


class FrozenEncoder:
    def __init__(self, config):
        validate_encoder_config(config)
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        torch.manual_seed(ENCODING["seed"])
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = AutoModel.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, attn_implementation="eager").to("cpu").eval()
        require(self.model.config.hidden_size == 384 and self.model.config.max_position_embeddings == 512,
                "model_dimensions")
        require(self.tokenizer.num_special_tokens_to_add(pair=False) == 2
                and self.tokenizer.cls_token_id is not None, "special_tokens")
        require(not self.model.training and all(p.dtype == torch.float32 and p.device.type == "cpu"
                                               for p in self.model.parameters()), "model_execution")

    def encode(self, prompt):
        chunks = content_chunks(self.tokenizer, prompt)
        vectors = []
        for start in range(0, len(chunks), 4):
            batch = []
            for chunk in chunks[start:start + 4]:
                item = self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, padding=False,
                    truncation=False, return_attention_mask=True, return_token_type_ids=True)
                require(len(item["input_ids"]) == len(chunk) + 2
                        and item["input_ids"][0] == self.tokenizer.cls_token_id
                        and item["input_ids"][1:-1] == chunk, "chunk_special_token_layout")
                batch.append(item)
            inputs = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                output = self.model(**inputs).last_hidden_state[:, 0, :]
            vectors.extend(output.float().cpu().numpy())
        lengths = list(map(len, chunks))
        return pool_chunks(vectors, lengths), lengths


def validate_receipt(receipt, *, prompt_sha, packet_sha, config_sha):
    import numpy as np
    require(type(receipt) is dict and set(receipt) == {"schema_version", "prompt_sha256",
        "packet_sha256", "encoder_config_sha256", "content_tokens", "chunk_lengths", "embedding"}
        and receipt["schema_version"] == "prompt-semantic-receipt-v1"
        and receipt["prompt_sha256"] == prompt_sha and receipt["packet_sha256"] == packet_sha
        and receipt["encoder_config_sha256"] == config_sha, "receipt_binding")
    lengths = receipt["chunk_lengths"]
    require(type(lengths) is list and bool(lengths) and all(type(v) is int and 1 <= v <= 510 for v in lengths)
            and all(v == 510 for v in lengths[:-1]) and type(receipt["content_tokens"]) is int
            and receipt["content_tokens"] == sum(lengths), "receipt_coverage")
    vector = np.asarray(receipt["embedding"], dtype=np.float32)
    require(vector.shape == (384,) and np.isfinite(vector).all()
            and abs(float(np.linalg.norm(vector)) - 1) < 1e-5, "receipt_embedding")
    return vector


def encode_packet(packet_path, packet_sha, config_path, config_sha, output_root, *, benchmark=False,
                  encoder_factory=FrozenEncoder):
    packet = base.document(packet_path, packet_sha)
    config = base.document(config_path, config_sha)
    validate_packet(packet)
    validate_encoder_config(config)
    root = private_path(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    binding = {"schema_version": "prompt-semantic-cache-run-v1", "packet_sha256": packet_sha,
               "encoder_config_sha256": config_sha, "source_pins": extraction_sources()}
    header = root / "run.json"
    if header.exists():
        require(base.document(header) == binding, "cache_run_drift")
    else:
        base.immutable(header, binding)
    selected = packet["prompts"][:32] if benchmark else packet["prompts"]
    started, encoder, computed, index, chunk_counts, token_counts = time.monotonic(), None, 0, {}, Counter(), []
    for row in selected:
        key = row["prompt_sha256"]
        path = root / "embeddings" / (key + ".json")
        if path.exists():
            receipt = base.document(path)
        else:
            if encoder is None:
                encoder = encoder_factory(config)
            vector, lengths = encoder.encode(row["prompt"])
            receipt = {"schema_version": "prompt-semantic-receipt-v1", "prompt_sha256": key,
                "packet_sha256": packet_sha, "encoder_config_sha256": config_sha,
                "content_tokens": sum(lengths), "chunk_lengths": lengths, "embedding": vector.tolist()}
            validate_receipt(receipt, prompt_sha=key, packet_sha=packet_sha, config_sha=config_sha)
            base.immutable(path, receipt)
            computed += 1
        validate_receipt(receipt, prompt_sha=key, packet_sha=packet_sha, config_sha=config_sha)
        index[key] = digest_file(path)
        chunk_counts[str(len(receipt["chunk_lengths"]))] += 1
        token_counts.append(receipt["content_tokens"])
    return {"schema_version": "prompt-semantic-cache-v1", "encoding": ENCODING,
        "scope": "benchmark" if benchmark else "complete", "source_pins": extraction_sources(),
        "packet_sha256": packet_sha, "encoder_config_sha256": config_sha,
        "encoder_config": config, "encoder_config_path": str(Path(config_path).absolute()),
        "receipt_root": str(root), "receipt_sha256": index,
        "summary": {"expected_unique_prompts": len(packet["prompts"]), "encoded_unique_prompts": len(index),
            "computed_this_call": computed, "chunk_count_histogram": dict(chunk_counts),
            "total_content_tokens": sum(token_counts), "max_content_tokens": max(token_counts),
            "elapsed_seconds": time.monotonic() - started, "labels_or_outcomes_supplied_to_encoder": False}}


def load_cache(cache, preparation, records):
    require(cache.get("schema_version") == "prompt-semantic-cache-v1" and cache.get("scope") == "complete"
            and cache.get("encoding") == ENCODING and cache.get("source_pins") == extraction_sources()
            and cache.get("packet_sha256") == preparation["packet_sha256"], "cache")
    config = base.document(cache["encoder_config_path"], cache["encoder_config_sha256"])
    require(config == cache["encoder_config"], "cache_config_binding")
    validate_encoder_config(config, check_runtime=False, check_files=False)
    expected = {base.sha(r["prompt"].encode()) for r in records}
    index = cache["receipt_sha256"]
    require(type(index) is dict and set(index) == expected
            and cache["summary"]["expected_unique_prompts"] == len(expected)
            and cache["summary"]["encoded_unique_prompts"] == len(expected), "cache_prompt_population")
    root = private_path(cache["receipt_root"])
    header = base.document(root / "run.json")
    require(header == {"schema_version": "prompt-semantic-cache-run-v1", "packet_sha256": cache["packet_sha256"],
        "encoder_config_sha256": cache["encoder_config_sha256"], "source_pins": extraction_sources()}, "cache_header")
    vectors = {}
    for key, receipt_sha in index.items():
        receipt = base.document(root / "embeddings" / (key + ".json"), receipt_sha)
        vectors[key] = validate_receipt(receipt, prompt_sha=key, packet_sha=cache["packet_sha256"],
                                       config_sha=cache["encoder_config_sha256"])
    return vectors


def preparation_inventory(records, evidence, packet_sha):
    inventory = base.inventory(records, evidence)
    # Bind this helper identically under `python -m` and ordinary module import.
    inventory["source_pins"][str(Path(__file__).relative_to(base.REPO))] = digest_file(__file__)
    return {"schema_version": "prompt-semantic-preparation-v1", "encoding": ENCODING,
            "models": list(MODELS), "packet_sha256": packet_sha,
            "base_inventory": inventory, "source_pins": extraction_sources(),
            "unique_prompts": len({base.sha(r["prompt"].encode()) for r in records}),
            "semantic_response_prefix_available": False, "new_fits_in_full_matrix": 80}


def matrices_for_cell(train, test, embeddings):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.preprocessing import StandardScaler
    text, vectorizers = base.vectorize_text(train, test)
    scalers = {}
    def scaled(field):
        scaler = StandardScaler()
        values = scaler.fit_transform([r[field] for r in train]), scaler.transform([r[field] for r in test])
        scalers[field] = scaler
        return values
    structural, internal = scaled("structural"), scaled("internal")
    joined = []
    for index, rows in enumerate((train, test)):
        semantic = [embeddings[base.sha(r["prompt"].encode())] for r in rows]
        joined.append(hstack((text[index], csr_matrix([r["prefix_hash"] for r in rows]),
                              csr_matrix(structural[index]), csr_matrix(semantic)), format="csr"))
    return {MODELS[0]: tuple(joined), MODELS[1]: tuple(hstack((x, csr_matrix(j)), format="csr")
            for x, j in zip(joined, internal, strict=True))}, vectorizers, scalers


def evaluate_cell(records, embeddings, placement, fold, held_family=None, *, max_seconds=900):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    started = time.monotonic()
    train_mask, test_mask = base.cell_masks(records, placement, fold, held_family)
    train = [r for r, keep in zip(records, train_mask, strict=True) if keep]
    test = [r for r, keep in zip(records, test_mask, strict=True) if keep]
    labels = np.asarray([r["scores"][-1]["binary_prediction"] for r in train], dtype=int)
    result = {"schema_version": "prompt-semantic-cell-v1", "placement": placement, "fold": fold,
              "held_family": held_family, "target_horizon": 1024, "training_rows": len(train),
              "test_rows": len(test), "models": {}, "interpretation": "retrospective_classifier_proxy",
              "confirmation_opened": False, "semantic_response_prefix_available": False}
    if not train or not test or len(set(labels)) != 2:
        return dict(result, status="unavailable_training_classes_or_test_support")
    matrices, fitted, _ = matrices_for_cell(train, test, embeddings)
    result["vocabulary_sizes"] = [len(v.vocabulary_) for v in fitted]
    for name in MODELS:
        if time.monotonic() - started >= max_seconds:
            result["models"][name] = {"status": "unavailable_deadline"}
            continue
        model = LogisticRegression(C=1.0, solver="liblinear", tol=1e-6, max_iter=1000,
                                   class_weight=None, random_state=20260915)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(matrices[name][0], labels)
        except ConvergenceWarning:
            result["models"][name] = {"status": "unavailable_convergence"}
            continue
        probability = model.predict_proba(matrices[name][1])[:, 1]
        result["models"][name] = {"status": "available", "feature_dimensions": matrices[name][0].shape[1],
                                  "metrics": base.metrics(test, probability)}
    result["status"] = "complete" if all(v["status"] == "available" for v in result["models"].values()) else "incomplete"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def environment():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        require(os.environ.get(key) in ("1", "2", "3", "4"), "threads")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("TOKENIZERS_PARALLELISM") == "false"
            and os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline_cpu_environment")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "encode", "inventory", "run"))
    for name in ("output", "packet", "packet-output", "encoder-config", "cache-root", "cache", "preparation", "inventory"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    for name in ("packet", "encoder-config", "cache", "preparation", "inventory"):
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--placement", choices=base.PLACEMENTS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--held-family", choices=base.FAMILIES)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        environment()
        private_path(args.output)
        require(not args.output.exists() and 1 <= args.max_seconds <= 1800, "output_or_deadline")
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            if args.action == "encode":
                require(all((args.packet, args.packet_sha256, args.encoder_config,
                             args.encoder_config_sha256, args.cache_root)), "encoding_arguments")
                result = encode_packet(args.packet, args.packet_sha256, args.encoder_config,
                    args.encoder_config_sha256, args.cache_root, benchmark=args.benchmark)
            else:
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=4):
                    records, evidence = base.load_retained()
                    packet = packet_from_records(records)
                    packet_sha = base.sha(base.canonical(packet))
                    current = preparation_inventory(records, evidence, packet_sha)
                    if args.action == "prepare":
                        require(args.packet_output is not None, "packet_output")
                        private_path(args.packet_output)
                        base.immutable(args.packet_output, packet)
                        result = current
                    else:
                        require(all((args.preparation, args.preparation_sha256, args.cache, args.cache_sha256)),
                                "fit_arguments")
                        prepared = base.document(args.preparation, args.preparation_sha256)
                        require(base.canonical(current) == base.canonical(prepared), "preparation_drift")
                        cache = base.document(args.cache, args.cache_sha256)
                        embeddings = load_cache(cache, prepared, records)
                        inventory = {"schema_version": "prompt-semantic-fit-inventory-v1",
                            "preparation_sha256": args.preparation_sha256, "cache_sha256": args.cache_sha256,
                            "preparation": current, "cache_encoder_config_sha256": cache["encoder_config_sha256"],
                            "fit_runtime": base.runtime(), "actual_fitting_performed": False}
                        if args.action == "inventory":
                            result = inventory
                        else:
                            require(all((args.inventory, args.inventory_sha256, args.placement))
                                    and args.fold is not None, "cell_arguments")
                            require(base.canonical(base.document(args.inventory, args.inventory_sha256))
                                    == base.canonical(inventory), "fit_inventory_drift")
                            result = evaluate_cell(records, embeddings, args.placement, args.fold,
                                                   args.held_family, max_seconds=args.max_seconds)
                            result.update(inventory_sha256=args.inventory_sha256,
                                          preparation_sha256=args.preparation_sha256, cache_sha256=args.cache_sha256)
        output_sha = base.immutable(args.output, result)
        print(json.dumps({"status": result.get("status", args.action + "_complete"), "sha256": output_sha}))
        return 0
    except Exception:
        print('{"status":"prompt_semantic_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
