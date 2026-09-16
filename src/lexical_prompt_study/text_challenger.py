"""Private, bounded CPU text challenger on explicitly unsealed retained data.

Only the pinned calibration paths are opened. Raw text stays inside the process;
stdout and immutable outputs contain aggregates, fixed enums and hashes only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings
import zipfile

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "plans/text_challenger_option3_20260915.md"
PINS = {
    "private/jlens-incremental/a139.calibration-topology.private.json": "52aabffe282219c9e39fce2f24890a606c2db22044d3e8be38712ba1837ff4bc",
    "private/jlens-incremental/a145.candidate.private.json": "932bf9b2aef298fd1dd75ebc3806426c650fdffb764ec8484affdf7b457c4088",
    "private/continuation/a146.plan.private.json": "be578ad04f40ac18e8b89b35779b9f70f17c25a48b075d199201e59b07215fe5",
    "private/runs/jlens-incremental-a142/acquisition/summary.json": "bc0038ba514598d9c836181f3ccbb41c0437e6b433855e2b0fbcc299757c38fc",
    "private/runs/jlens-incremental-a142/scoring/summary.json": "1e68657c5f124fe9cd4b403885b0e84ecc1d4e7a19fd9f27c1467109ac7d3302",
    "private/continuation/a148-verification-17/scoring-receipts.private.zip": "721b4a40f8acecbed68b75697181e7bf413344be032eab586e279716af4134fa",
    "private/continuation/analysis18_a148.py": "f6c95e8cb355e283664ab29049744e43d07b65d987caf2b61d75b22a3a349040",
    "private/continuation/analysis18-freeze.private.json": "5af99717d8db170149b3e1c81ef35a8217a6bf09982a7e1de580c96c80d345c6",
    "private/continuation/refit20-preparation.private.json": "72a07b2b6adcabb472f8af2fb73a1418109d99df6e63720c086629d97a9c78dd",
    "private/continuation/fit22-preparation.private.json": "40880a566870255904b55c158f0c78a597c3b0fb8f294e8fc7b22a1e49845d63",
    "private/continuation/features28_a150.py": "2797e3c9b04acea26336898f8fc0ef248c97c104d7407d3611274d7021f063b4",
    "private/continuation/recover19_a150.py": "3baa7247d1150485bcc154b1ed60ac869d6e81a0c3c5c4af9fb72c8ee4e134b1",
    "private/continuation/a150-features-28/features21.private.zip": "6b072eb2c825678c7f259afe50cf8a0353af92d3d5914cf4788eff326003a574",
}
TOKENIZER_PINS = {
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
}
PLACEMENTS = ("scaffold_before_request", "scaffold_after_request")
FAMILIES = ("attack_block_mask", "structural_sham", "harmless_structured_wrapper")
INTENTS = ("unsafe_direct", "safe_classify_exact", "safe_refuse_exact", "safe_acknowledge_exact")
HORIZONS = (128, 256, 512, 1024)
MODELS = ("training_prevalence", "prompt_text", "prompt_plus_prefix_hash", "jlens_t8", "prompt_plus_prefix_hash_and_jlens")
TEXT_BRANCHES = (("prompt", "word", (1, 2), 10000), ("prompt", "char", (3, 5), 20000))
CONTRACT = {
    "schema_version": "text-challenger-contract-v1", "target_horizon": 1024,
    "prefix_tokens": 8, "outer_core_folds": 5, "C": 1.0, "solver": "liblinear",
    "tolerance": 1e-6, "max_iterations": 1000, "class_weight": None, "seed": 20260915,
    "min_df": 2, "sublinear_tf": True, "lowercase": False, "models": list(MODELS),
    "branches": [[field, analyzer, list(ngram), cap] for field, analyzer, ngram, cap in TEXT_BRANCHES],
    "max_document_characters": 100000, "max_threads": 4,
    "representation_stage": "learned_prompt_plus_retained_prefix_hash",
    "prefix_hash_dimensions": 256, "prefix_hash_skip_special_tokens": True,
    "learned_prefix_vocabulary_available": False, "pretrained_semantic_baseline": False,
    "confirmation_opened": False, "retrospective_exploratory": True,
}


class ChallengerError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise ChallengerError(code)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink")
    require(path.is_file() and path.stat().st_size <= 512 * 1024**2, "input_bounds")
    raw = path.read_bytes()
    require(expected is None or sha(raw) == expected, "input_hash")
    return raw


def document(path, expected=None):
    return json.loads(read(path, expected))


def immutable(path, value):
    raw = canonical(value)
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "output_symlink")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha(raw)


def source_pins():
    paths = [Path(__file__), REPO / PROTOCOL, REPO / "tests/test_text_challenger.py"]
    # Called after loading retained inputs: bind loaded transitive helpers, not
    # unrelated research modules being developed concurrently. The independently
    # frozen private A149 loader binds its own dynamic helper sources.
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if (name == "lexical_prompt_study" or name.startswith("lexical_prompt_study.")) and location:
            path = Path(location).resolve()
            require(path.is_relative_to(REPO / "src/lexical_prompt_study") and path.suffix == ".py",
                    "helper_source_location")
            paths.append(path)
    return {str(path.relative_to(REPO)): sha(read(path)) for path in sorted(set(paths))}


def runtime():
    return {"python": platform.python_version(), "executable": os.path.realpath(os.sys.executable),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "transformers", "tokenizers", "threadpoolctl")}}


def load_numeric():
    """Verify the retained numeric chain without opening restricted generations."""
    for name, digest in PINS.items():
        read(REPO / name, digest)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec = importlib.util.spec_from_file_location("challenger_verified_a149", REPO / "private/continuation/analysis18_a148.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.verify_frozen()
        numeric, candidate, _ = loader.load_inputs()
    return numeric, candidate


def load_prefix_features():
    """Read the immutable numeric export; never open original or continued text."""
    path = REPO / "private/continuation/a150-features-28/features21.private.zip"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec = importlib.util.spec_from_file_location("challenger_features28", REPO / "private/continuation/features28_a150.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        verified = helper.verify_bundle(path)
    require(verified["bundle_sha256"] == PINS[str(path.relative_to(REPO))], "feature_bundle")
    with zipfile.ZipFile(io.BytesIO(read(path, PINS[str(path.relative_to(REPO))]))) as archive:
        metadata = json.loads(archive.read("metadata.private.json"))
        matrix = helper.safe_npy(archive.read("prefix_features.npy"), len(metadata["prefix_rows"]), 256)
    indexed = {}
    for meta, vector in zip(metadata["prefix_rows"], matrix, strict=True):
        if meta["regime"] == "original" and meta["prefix_token_count"] == 8:
            require(meta["trial_id"] not in indexed, "duplicate_prefix")
            indexed[meta["trial_id"]] = (meta, vector)
    return indexed


def validate_prefix(meta, vector, receipt_sha, readout):
    import numpy as np
    require(meta["regime"] == "original" and meta["prefix_token_count"] == 8
            and meta["source_generation_receipt_sha256"] == receipt_sha
            and meta["prefix_token_ids_sha256"] == readout["prefix_token_ids_sha256"], "prefix_binding")
    require(np.asarray(vector).shape == (256,) and np.isfinite(vector).all(), "prefix_features")
    return np.asarray(vector).tolist()


def load_retained():
    """Privately join original prompts, token-eight numeric features and labels."""
    import numpy as np
    from .jlens_incremental_runner import _validate_receipt
    from .weaponization_analysis import STRUCTURAL_FIELDS
    numeric, candidate = load_numeric()
    prefixes = load_prefix_features()
    topology = document(REPO / "private/jlens-incremental/a139.calibration-topology.private.json")
    require(topology["unopened_v2_confirmation_opened"] is False, "sealed_topology")
    observations = {row["trial_id"]: row for row in topology["observations"]}
    require(len(observations) == len(numeric) == 8880, "population")
    plan = document(REPO / "private/continuation/a146.plan.private.json")
    receipt_pins = {row["trial_id"]: row["sha256"] for row in plan["original_receipts"]}
    preparation = document(REPO / "private/continuation/refit20-preparation.private.json")
    require(candidate["row_order"] == preparation["row_order"] and candidate["folds"] == preparation["folds"], "fold_binding")
    records, unavailable = [], Counter()
    input_records = []
    acq = REPO / "private/runs/jlens-incremental-a142/acquisition"
    for index, row in enumerate(numeric):
        identity = row["trial_id"]
        observation = observations[identity]
        require(identity == candidate["row_order"][index], "row_order")
        receipt = document(acq / "receipts" / (identity + ".json"), receipt_pins[identity])
        _validate_receipt(receipt, observation)
        require(row["variant_family"] in (*FAMILIES, "no_scaffold") and row["intent_frame"] in INTENTS,
                "metadata_enum")
        if row["placement"] is None:
            unavailable["unplaced"] += 1
            continue
        require(row["placement"] in PLACEMENTS, "placement")
        if row["readouts"]["8"] is None:
            unavailable["ended_before_eight"] += 1
            continue
        require(identity in prefixes, "missing_prefix_features")
        meta, vector = prefixes[identity]
        prefix = validate_prefix(meta, vector, receipt_pins[identity], row["readouts"]["8"])
        prompt = observation["prompt_text"]
        require(sha(prompt.encode()) == receipt["prompt_sha256"], "prompt_binding")
        require(len(prompt) <= CONTRACT["max_document_characters"], "document_bound")
        internal = row["readouts"]["8"]["jlens_refusal_minus_compliance_trajectory"]
        structural = [receipt["structural_metrics"][name] for name in STRUCTURAL_FIELDS]
        require(len(internal) == 31 and len(structural) == 6
                and np.isfinite(internal).all() and np.isfinite(structural).all(), "numeric_features")
        record = {"prompt": prompt, "prefix_hash": prefix, "internal": internal, "structural": structural,
                  "core": row["request_core_sha256"], "fold": candidate["folds"][index],
                  "placement": row["placement"], "family": row["variant_family"], "intent": row["intent_frame"],
                  "origin": row["origin"], "scores": row["scores"]}
        records.append(record)
        input_records.append({"trial_sha256": sha(identity.encode()), "prompt_sha256": sha(prompt.encode()),
            "prefix_sha256": meta["prefix_token_ids_sha256"], "receipt_sha256": receipt_pins[identity],
            "numeric_sha256": sha(canonical({k: record[k] for k in record if k != "prompt"}))})
    require(len({r["core"] for r in records}) == 60, "core_count")
    return records, {"original_rows": 8880, "excluded": dict(unavailable),
                     "ordered_inputs_sha256": sha(canonical(input_records)),
                     "prefix_representation": "original_t8_fixed_byte_ngram_hash_256d",
                     "raw_generation_read_performed": False,
                     "original_fold_map_sha256": sha(canonical(candidate["folds"]))}


def cell_masks(records, placement, fold, held_family=None):
    import numpy as np
    require(placement in PLACEMENTS and type(fold) is int and 0 <= fold < 5
            and held_family in (None, *FAMILIES), "cell")
    assignments = {}
    for record in records:
        require(assignments.setdefault(record["core"], record["fold"]) == record["fold"], "core_leakage")
    train = np.asarray([r["placement"] == placement and r["fold"] != fold
                        and (held_family is None or r["family"] != held_family)
                        and r["scores"][-1]["binary_prediction"] is not None for r in records])
    test = np.asarray([r["placement"] == placement and r["fold"] == fold
                       and (held_family is None or r["family"] == held_family) for r in records])
    require(not np.any(train & test), "row_leakage")
    require(not ({r["core"] for r, keep in zip(records, train, strict=True) if keep}
                 & {r["core"] for r, keep in zip(records, test, strict=True) if keep}), "core_leakage")
    return train, test


def inventory(records, evidence):
    cells = []
    for placement in PLACEMENTS:
        for family in (None, *FAMILIES):
            for fold in range(5):
                train, test = cell_masks(records, placement, fold, family)
                labels = [r["scores"][-1]["binary_prediction"] for r, keep in zip(records, train, strict=True) if keep]
                cells.append({"placement": placement, "held_family": family, "fold": fold,
                    "training_rows": int(train.sum()), "test_rows": int(test.sum()),
                    "training_cores": len({r["core"] for r, keep in zip(records, train, strict=True) if keep}),
                    "test_cores": len({r["core"] for r, keep in zip(records, test, strict=True) if keep}),
                    "both_training_classes": len(set(labels)) == 2})
    return {"schema_version": "text-challenger-inventory-v1", "contract": CONTRACT, "input_pins": PINS,
            "tokenizer_pins": TOKENIZER_PINS, "source_pins": source_pins(), "runtime": runtime(),
            "evidence": evidence, "eligible_rows": len(records), "request_cores": len({r["core"] for r in records}),
            "fold_rows": {str(f): sum(r["fold"] == f for r in records) for f in range(5)},
            "placement_rows": {p: sum(r["placement"] == p for r in records) for p in PLACEMENTS},
            "family_rows": {f: sum(r["family"] == f for r in records) for f in FAMILIES},
            "origin_rows": dict(Counter(r["origin"] for r in records)), "cells": cells,
            "actual_fitting_performed": False, "confirmation_opened": False}


def vectorize_text(train, test):
    """Fit every vocabulary and IDF solely on training documents."""
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    fitted, train_blocks, test_blocks = [], [], []
    for field, analyzer, ngram, cap in TEXT_BRANCHES:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=cap,
            min_df=2, sublinear_tf=True, norm="l2", lowercase=False, dtype=np.float64)
        matrix = vectorizer.fit_transform([r[field] for r in train])
        train_blocks.append(matrix)
        test_blocks.append(vectorizer.transform([r[field] for r in test]))
        fitted.append(vectorizer)
    return (hstack(train_blocks, format="csr"), hstack(test_blocks, format="csr")), fitted


def metrics(records, probabilities):
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    require(len(records) == len(probabilities) and np.isfinite(probabilities).all()
            and np.all((probabilities >= 0) & (probabilities <= 1)), "predictions")
    results = {}
    for h_index, horizon in enumerate(HORIZONS):
        groups = {}
        for group in ("all", *INTENTS):
            selected = [i for i, r in enumerate(records) if group == "all" or r["intent"] == group]
            known = [i for i in selected if records[i]["scores"][h_index]["binary_prediction"] is not None]
            labels = np.asarray([records[i]["scores"][h_index]["binary_prediction"] for i in known], dtype=int)
            p = np.asarray(probabilities)[known]
            two = len(set(labels)) == 2
            groups[group] = {"selected": len(selected), "known": len(known), "unknown": len(selected) - len(known),
                "positive": int(labels.sum()), "request_cores": len({records[i]["core"] for i in selected}),
                "right_censored": sum(records[i]["scores"][h_index]["right_censored"] for i in selected),
                "capped_negative": sum(records[i]["scores"][h_index]["right_censored"]
                    and records[i]["scores"][h_index]["binary_prediction"] is False for i in selected),
                "log_loss": float(log_loss(labels, p, labels=[0, 1])) if known else None,
                "brier": float(brier_score_loss(labels, p)) if known else None,
                "roc_auc": float(roc_auc_score(labels, p)) if two else None,
                "average_precision": float(average_precision_score(labels, p)) if two else None}
        results[str(horizon)] = groups
    return results


def evaluate_cell(records, placement, fold, held_family=None, *, max_seconds=900):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    started = time.monotonic()
    train_mask, test_mask = cell_masks(records, placement, fold, held_family)
    train = [r for r, keep in zip(records, train_mask, strict=True) if keep]
    test = [r for r, keep in zip(records, test_mask, strict=True) if keep]
    labels = np.asarray([r["scores"][-1]["binary_prediction"] for r in train], dtype=int)
    result = {"schema_version": "text-challenger-cell-v1", "placement": placement, "fold": fold,
              "held_family": held_family, "target_horizon": 1024, "training_rows": len(train),
              "test_rows": len(test), "models": {}, "interpretation": "retrospective_classifier_proxy",
              "confirmation_opened": False}
    if not train or not test or len(set(labels)) != 2:
        result["status"] = "unavailable_training_classes_or_test_support"
        return result
    text, fitted = vectorize_text(train, test)
    result["vocabulary_sizes"] = [len(v.vocabulary_) for v in fitted]
    def numeric(field):
        scaler = StandardScaler()
        return scaler.fit_transform([r[field] for r in train]), scaler.transform([r[field] for r in test])
    internal, structural = numeric("internal"), numeric("structural")
    prefix = (csr_matrix([r["prefix_hash"] for r in train]), csr_matrix([r["prefix_hash"] for r in test]))
    matrices = {"prompt_text": tuple(hstack((x, csr_matrix(s)), format="csr") for x, s in zip(text, structural, strict=True)),
                "prompt_plus_prefix_hash": tuple(hstack((x, p, csr_matrix(s)), format="csr") for x, p, s in zip(text, prefix, structural, strict=True)),
                "jlens_t8": internal,
                "prompt_plus_prefix_hash_and_jlens": tuple(hstack((x, p, csr_matrix(s), csr_matrix(j)), format="csr")
                    for x, p, s, j in zip(text, prefix, structural, internal, strict=True))}
    result["models"]["training_prevalence"] = {"status": "available", "metrics": metrics(test, np.full(len(test), labels.mean()))}
    for name in MODELS[1:]:
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
                                   "metrics": metrics(test, probability)}
    result["status"] = "complete" if all(v["status"] == "available" for v in result["models"].values()) else "incomplete"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--placement", choices=PLACEMENTS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--held-family", choices=FAMILIES)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            require(os.environ.get(key) in ("1", "2", "3", "4"), "thread_environment")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("TOKENIZERS_PARALLELISM") == "false",
                "cpu_only_environment")
        require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline_environment")
        require(1 <= args.max_seconds <= 1800 and not args.output.exists(), "output_or_deadline")
        if args.action == "run":
            require(args.inventory is not None and args.inventory_sha256 is not None
                    and args.placement is not None and args.fold is not None, "run_arguments")
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            records, evidence = load_retained()
            current = inventory(records, evidence)
            if args.action == "inventory":
                result = current
            else:
                frozen = document(args.inventory, args.inventory_sha256)
                require(canonical(frozen) == canonical(current), "inventory_drift")
                result = evaluate_cell(records, args.placement, args.fold, args.held_family, max_seconds=args.max_seconds)
                result["inventory_sha256"] = args.inventory_sha256
                result["eligible_rows"] = len(records)
        output_sha = immutable(args.output, result)
        print(json.dumps({"status": result.get("status", "inventory_complete"), "sha256": output_sha,
                          "eligible_rows": result["eligible_rows"]}, sort_keys=True))
        return 0
    except Exception:
        print('{"status":"text_challenger_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
