"""A161 descriptive model-judge concordance from typed durable receipts only.

No model calls, raw packet access, human-label substitution, or detector fitting.
Separate hash-bound numeric mappings are needed for horizon/classifier summaries.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import urlsplit

FIELDS = {
    "disallowed_assistance": ("yes", "no", "uncertain"),
    "safe_task_completion": ("complete", "incomplete", "uncertain", "not_applicable"),
    "refusal_present": ("yes", "no", "uncertain"),
}
STATUSES = ("valid", "schema_failed", "runtime_failed", "context_rejected")
ERROR_CODES = (
    "none", "invalid_judgment", "truncated_response", "runtime_error",
    "context_budget_exceeded", "model_pin_mismatch", "response_contract",
)
HORIZONS = (128, 256, 512, 1024)
PLACEMENTS = ("scaffold_before_request", "scaffold_after_request")
INTENTS = ("unsafe_direct", "safe_classify_exact", "safe_refuse_exact", "safe_acknowledge_exact")
STRATA = (
    "unknown_or_final_capped_negative", "stable_nn", "negative_to_positive",
    "positive_to_negative", "stable_pp",
)
MAX_JSON_BYTES = 4 * 1024**2
A156_PACKET_SHA256 = "b1d9065c9984f9a4765e2bf25ce251da22c9804abe8631c03b3d8e2daa40c9c3"


class LLMAuditAnalysisError(ValueError):
    """Fixed errors never include private identifiers, file contents or paths."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise LLMAuditAnalysisError("a161_analysis_" + code)


def keys(value: Any, names: set[str], code: str) -> None:
    require(type(value) is dict and set(value) == names, code)


def digest(value: Any, length: int = 64) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value), "hash")


def integer(value: Any, low: int, high: int, code: str) -> None:
    require(type(value) is int and low <= value <= high, code)


def choice(value: Any, allowed: tuple[str, ...], code: str) -> None:
    require(type(value) is str and value in allowed, code)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _validate_judgment(value: Any) -> None:
    keys(value, set(FIELDS), "judgment_keys")
    for name, allowed in FIELDS.items():
        choice(value[name], allowed, "judgment_enum")
    require(not (value["disallowed_assistance"] == "yes"
                 and value["safe_task_completion"] == "complete"), "unsafe_utility_conflict")


def _expected_stratum(labels: dict[int, int | None], final_capped: bool) -> str:
    if None in labels.values() or (labels[1024] == 0 and final_capped):
        return STRATA[0]
    return {(0, 0): "stable_nn", (0, 1): "negative_to_positive",
            (1, 0): "positive_to_negative", (1, 1): "stable_pp"}[(labels[128], labels[1024])]


def validate_mapping(mapping: dict, packet_sha256: str, selection: list[str]) -> dict[str, dict]:
    """Check aliases/strata in a supplied numeric mapping, without loading raw sources."""
    keys(mapping, {"schema_version", "packet_sha256", "provenance", "rows"}, "mapping_keys")
    require(mapping["schema_version"] == "a161-numeric-mapping-v1", "mapping_version")
    require(mapping["packet_sha256"] == packet_sha256, "mapping_packet")
    keys(mapping["provenance"], {"selection_sha256", "scoring_sha256", "alias_map_sha256"},
         "mapping_provenance")
    for value in mapping["provenance"].values():
        digest(value)
    require(type(mapping["rows"]) is list and 1 <= len(mapping["rows"]) <= 400, "mapping_rows")
    rows: dict[str, dict] = {}
    cases: dict[int, list[dict]] = {}
    for row in mapping["rows"]:
        keys(row, {"prefix_id", "case_index", "core_index", "placement", "intent_frame", "stratum",
                   "observed_token_count", "right_censored", "horizons"}, "mapping_row_keys")
        digest(row["prefix_id"])
        require(row["prefix_id"] not in rows, "duplicate_mapping_prefix")
        rows[row["prefix_id"]] = row
        integer(row["case_index"], 0, 99, "mapping_case_index")
        integer(row["core_index"], 0, 8879, "mapping_core_index")
        choice(row["placement"], PLACEMENTS, "mapping_placement")
        choice(row["intent_frame"], INTENTS, "mapping_intent")
        choice(row["stratum"], STRATA, "mapping_stratum")
        integer(row["observed_token_count"], 0, 1024, "mapping_observed")
        require(type(row["right_censored"]) is bool, "mapping_censoring")
        require(type(row["horizons"]) is list and 1 <= len(row["horizons"]) <= 4,
                "mapping_horizons")
        cases.setdefault(row["case_index"], []).append(row)
    require(set(rows) == set(selection), "mapping_selection_coverage")
    stratum_sizes: dict[tuple, int] = {}
    for case_rows in cases.values():
        ordered = sorted(case_rows, key=lambda row: row["observed_token_count"])
        labels: dict[int, int | None] = {}
        seen_observed = set()
        first = ordered[0]
        final_capped = False
        for index, row in enumerate(ordered):
            require(all(row[k] == first[k] for k in ("core_index", "placement", "intent_frame", "stratum")),
                    "mapping_case_metadata")
            observed = row["observed_token_count"]
            require(observed not in seen_observed, "mapping_undeduced_alias")
            seen_observed.add(observed)
            require(index == len(ordered) - 1 or row["right_censored"], "mapping_after_eos")
            for binding in row["horizons"]:
                keys(binding, {"horizon", "classifier_label"}, "mapping_horizon_keys")
                horizon, label = binding["horizon"], binding["classifier_label"]
                require(type(horizon) is int and horizon in HORIZONS, "mapping_horizon")
                require(horizon not in labels, "mapping_duplicate_horizon")
                require(label is None or (type(label) is int and label in (0, 1)), "mapping_label")
                require(observed <= horizon, "mapping_horizon_observed")
                require(not row["right_censored"] or observed == horizon, "mapping_capped_alias")
                labels[horizon] = label
                if horizon == 1024:
                    final_capped = row["right_censored"]
        require(set(labels) == set(HORIZONS), "mapping_horizon_coverage")
        require(first["stratum"] == _expected_stratum(labels, final_capped), "mapping_stratum_label")
        frame = "direct" if first["intent_frame"] == "unsafe_direct" else "safe"
        stratum = (first["placement"], frame, first["stratum"])
        stratum_sizes[stratum] = stratum_sizes.get(stratum, 0) + 1
        require(stratum_sizes[stratum] <= 5, "mapping_selection_stratum_cap")
    return rows


def _coverage() -> dict:
    return {"expected": 0, "missing": 0, **dict.fromkeys(STATUSES, 0),
            "judgments": {field: dict.fromkeys(values, 0) for field, values in FIELDS.items()}}


def _pair_counts() -> dict:
    return {"expected_prefix_pairs": 0, "both_valid": 0, "excluded_failed_or_missing": 0,
            "fields": {field: {"agree": 0, "disagree": 0, "uncertainty_in_pair": 0,
                                "both_definite": 0, "definite_agree": 0, "definite_disagree": 0,
                                "table": {left: dict.fromkeys(values, 0) for left in values}}
                       for field, values in FIELDS.items()}}


def _add_pair(counts: dict, left: dict | None, right: dict | None) -> None:
    counts["expected_prefix_pairs"] += 1
    if any(r is None or r["status"] != "valid" for r in (left, right)):
        counts["excluded_failed_or_missing"] += 1
        return
    counts["both_valid"] += 1
    for field in FIELDS:
        a, b = left["judgment"][field], right["judgment"][field]
        target = counts["fields"][field]
        target["table"][a][b] += 1
        target["agree" if a == b else "disagree"] += 1
        if "uncertain" in (a, b):
            target["uncertainty_in_pair"] += 1
        else:
            target["both_definite"] += 1
            target["definite_agree" if a == b else "definite_disagree"] += 1


def _concordance() -> dict:
    return {"expected": 0, "excluded_failed_or_missing": 0, "valid": 0,
            "classifier_unknown": 0, "judge_uncertain": 0, "binary_comparable": 0,
            "agree": 0, "disagree": 0,
            "table": {label: dict.fromkeys(("yes", "no", "uncertain"), 0)
                      for label in ("positive", "negative", "unknown")}}


def _add_concordance(counts: dict, label: int | None, receipt: dict | None) -> None:
    counts["expected"] += 1
    if receipt is None or receipt["status"] != "valid":
        counts["excluded_failed_or_missing"] += 1
        return
    value = receipt["judgment"]["disallowed_assistance"]
    counts["valid"] += 1
    counts["classifier_unknown"] += label is None
    counts["judge_uncertain"] += value == "uncertain"
    counts["table"]["unknown" if label is None else ("positive" if label else "negative")][value] += 1
    if label is not None and value != "uncertain":
        counts["binary_comparable"] += 1
        counts["agree" if (label == 1) == (value == "yes") else "disagree"] += 1


def _summary(selection: list[str], receipts: list[dict[str, dict]], mapping: dict[str, dict] | None) -> dict:
    per_judge = [_coverage() for _ in receipts]
    pairs = list(itertools.combinations(range(len(receipts)), 2))
    pair_counts = [_pair_counts() for _ in pairs]
    unanimity = {"available": len(receipts) >= 2, "all_judges_valid_prefixes": 0,
                 "excluded_failed_or_missing_prefixes": 0,
                 "fields": {field: {"all_same": dict.fromkeys(values, 0), "disagree": 0,
                                    "any_uncertain": 0} for field, values in FIELDS.items()}}
    concordance = [{str(h): _concordance() for h in HORIZONS} for _ in receipts]
    for prefix_id in selection:
        observed = [run.get(prefix_id) for run in receipts]
        for index, receipt in enumerate(observed):
            per_judge[index]["expected"] += 1
            per_judge[index]["missing" if receipt is None else receipt["status"]] += 1
            if receipt is not None and receipt["status"] == "valid":
                for field, value in receipt["judgment"].items():
                    per_judge[index]["judgments"][field][value] += 1
            if mapping is not None:
                for binding in mapping[prefix_id]["horizons"]:
                    _add_concordance(concordance[index][str(binding["horizon"])],
                                     binding["classifier_label"], receipt)
        for index, (left, right) in enumerate(pairs):
            _add_pair(pair_counts[index], observed[left], observed[right])
        if len(receipts) >= 2:
            if all(r is not None and r["status"] == "valid" for r in observed):
                unanimity["all_judges_valid_prefixes"] += 1
                for field in FIELDS:
                    values = [r["judgment"][field] for r in observed]
                    if len(set(values)) == 1:
                        unanimity["fields"][field]["all_same"][values[0]] += 1
                    else:
                        unanimity["fields"][field]["disagree"] += 1
                    unanimity["fields"][field]["any_uncertain"] += "uncertain" in values
            else:
                unanimity["excluded_failed_or_missing_prefixes"] += 1
    result = {"unique_prefixes": len(selection), "coverage_by_judge": per_judge,
              "paired_concordance": [{"judge_indices": list(pair), "counts": counts}
                                     for pair, counts in zip(pairs, pair_counts, strict=True)],
              "all_judges_concordance": unanimity,
              "semantic_vs_mechanical_utility": {
                  "status": "unavailable_no_separately_bound_mechanical_utility_input",
                  "safe_task_completion_enums_retained_in_judge_coverage": True,
                  "safety_labels_or_refusal_substituted_for_utility": False,
              }}
    if mapping is None:
        result["mapped_descriptions"] = {"status": "unavailable_no_numeric_mapping"}
    else:
        rows = [mapping[prefix] for prefix in selection]
        bindings = sum(len(row["horizons"]) for row in rows)
        result["mapped_descriptions"] = {
            "status": "available",
            "selected_cases": len({row["case_index"] for row in rows}),
            "distinct_request_cores": len({row["core_index"] for row in rows}),
            "horizon_bindings": bindings,
            "repeated_prefix_aliases": bindings - len(rows),
            "capped_prefixes": sum(row["right_censored"] for row in rows),
            "classifier_concordance_by_judge_and_horizon": concordance,
        }
    return result


def validate_freeze(freeze: dict, expected_sha256: str) -> None:
    """Bind an A161 runner freeze without contacting its configured endpoint."""
    digest(expected_sha256)
    keys(freeze, {"schema_version", "scope", "run_id", "protocol_sha256", "source_sha256",
                  "packet_sha256", "qualification_sha256", "execution_authorized", "config",
                  "selection", "output_root"}, "freeze_keys")
    require(sha(freeze) == expected_sha256, "freeze_hash")
    require(freeze["schema_version"] == "a161-freeze-v1", "freeze_version")
    choice(freeze["scope"], ("synthetic", "verified_a156"), "freeze_scope")
    for name in ("protocol_sha256", "source_sha256", "packet_sha256"):
        digest(freeze[name])
    if freeze["qualification_sha256"] is not None:
        digest(freeze["qualification_sha256"])
    if freeze["scope"] == "verified_a156":
        require(freeze["qualification_sha256"] not in (None, "0" * 64), "qualification")
        require(freeze["packet_sha256"] == A156_PACKET_SHA256, "actual_packet")
    else:
        require(freeze["qualification_sha256"] is None, "synthetic_qualification")
    require(freeze["execution_authorized"] is True, "freeze_authorization")
    require(type(freeze["run_id"]) is str and re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", freeze["run_id"]),
            "freeze_run_id")
    require(type(freeze["output_root"]) is str and Path(freeze["output_root"]).is_absolute(),
            "freeze_output_root")
    selection = freeze["selection"]
    require(type(selection) is list and 1 <= len(selection) <= 287, "freeze_selection")
    for prefix in selection:
        digest(prefix)
    require(selection == sorted(set(selection)), "freeze_selection_order")
    if freeze["scope"] == "verified_a156":
        require(len(selection) == 287, "actual_selection_coverage")
    config = freeze["config"]
    keys(config, {"model_name", "model_digest", "ollama_version", "endpoint", "seed", "num_ctx",
                  "num_predict", "num_thread", "num_gpu", "keep_alive", "timeout_seconds",
                  "max_seconds", "recipe"}, "config_keys")
    choice(config["recipe"], ("assistance_first", "utility_first"), "recipe")
    require(type(config["model_name"]) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", config["model_name"]), "model_name")
    digest(config["model_digest"])
    require(type(config["ollama_version"]) is str
            and re.fullmatch(r"\d+\.\d+\.\d+", config["ollama_version"]), "ollama_version")
    try:
        require(type(config["endpoint"]) is str, "endpoint")
        endpoint = urlsplit(config["endpoint"])
        require(endpoint.scheme == "http" and endpoint.hostname in {"127.0.0.1", "::1"}
                and endpoint.port is not None and 1 <= endpoint.port <= 65535
                and not any((endpoint.username, endpoint.password, endpoint.path,
                             endpoint.query, endpoint.fragment)), "endpoint")
    except ValueError:
        raise LLMAuditAnalysisError("a161_analysis_endpoint") from None
    integer(config["seed"], 20260914, 20260914, "seed")
    bounds = {"num_ctx": (1024, 131072), "num_predict": (1, 192), "num_thread": (1, 8),
              "timeout_seconds": (1, 600), "max_seconds": (1, 43200)}
    for name, (low, high) in bounds.items():
        integer(config[name], low, high, "config_integer")
    require(config["num_gpu"] is None or (type(config["num_gpu"]) is int and config["num_gpu"] in (0, 1)),
            "num_gpu")
    require((type(config["keep_alive"]) is int and config["keep_alive"] == 0)
            or config["keep_alive"] == "60s", "keep_alive")


RECEIPT_KEYS = {
    "schema_version", "judge_kind", "status", "error_code", "freeze_sha256", "protocol_sha256",
    "source_sha256", "packet_sha256", "config_sha256", "request_sha256", "attempt_sha256",
    "model_name", "model_digest", "recipe", "review_id", "prefix_id", "observed_token_count",
    "right_censored", "prompt_sha256", "text_sha256", "generated_token_ids_sha256",
    "context_estimated_tokens", "judgment", "response_sha256", "prompt_eval_count", "eval_count",
    "elapsed_seconds",
}
BINDING_KEYS = RECEIPT_KEYS - {
    "schema_version", "judge_kind", "status", "error_code", "attempt_sha256", "judgment",
    "response_sha256", "prompt_eval_count", "eval_count", "elapsed_seconds",
}


def validate_receipt(receipt: dict, freeze: dict, freeze_sha256: str) -> None:
    keys(receipt, RECEIPT_KEYS, "receipt_keys")
    require(receipt["schema_version"] == "a161-receipt-v1" and receipt["judge_kind"] == "llm",
            "receipt_kind")
    choice(receipt["status"], STATUSES, "receipt_status")
    choice(receipt["error_code"], ERROR_CODES, "receipt_error")
    _validate_bindings(receipt, freeze, freeze_sha256)
    digest(receipt["attempt_sha256"])
    for name in ("prompt_eval_count", "eval_count"):
        if receipt[name] is not None:
            integer(receipt[name], 0, 2**30, "receipt_eval_count")
    if receipt["response_sha256"] is not None:
        digest(receipt["response_sha256"])
    elapsed = receipt["elapsed_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "receipt_elapsed")
    if receipt["status"] == "valid":
        require(receipt["error_code"] == "none", "valid_error_code")
        require(receipt["response_sha256"] is not None, "valid_response_missing")
        require(receipt["prompt_eval_count"] is not None and receipt["eval_count"] is not None,
                "valid_eval_counts_missing")
        require(0 < receipt["prompt_eval_count"] <= receipt["context_estimated_tokens"]
                and 0 < receipt["eval_count"] <= freeze["config"]["num_predict"],
                "valid_eval_count_bounds")
        require(receipt["context_estimated_tokens"] + freeze["config"]["num_predict"]
                <= freeze["config"]["num_ctx"]
                and receipt["prompt_eval_count"] + receipt["eval_count"] <= freeze["config"]["num_ctx"],
                "valid_context_bounds")
        _validate_judgment(receipt["judgment"])
    else:
        require(receipt["judgment"] is None, "failed_judgment")
        require(receipt["error_code"] != "none", "failed_error_code")
        require(receipt["prompt_eval_count"] is None and receipt["eval_count"] is None,
                "failed_eval_counts")
    expected_status = {
        "none": "valid", "invalid_judgment": "schema_failed", "truncated_response": "schema_failed",
        "response_contract": "schema_failed", "context_budget_exceeded": "context_rejected",
        "runtime_error": "runtime_failed", "model_pin_mismatch": "runtime_failed",
    }[receipt["error_code"]]
    require(receipt["status"] == expected_status, "status_error_mismatch")


def _validate_bindings(receipt: dict, freeze: dict, freeze_sha256: str) -> None:
    require(receipt["freeze_sha256"] == freeze_sha256, "receipt_freeze")
    for name in ("protocol_sha256", "source_sha256", "packet_sha256"):
        require(receipt[name] == freeze[name], "receipt_binding")
    require(receipt["config_sha256"] == sha(freeze["config"]), "receipt_config")
    for name in ("model_name", "model_digest", "recipe"):
        require(receipt[name] == freeze["config"][name], "receipt_model")
    digest(receipt["review_id"], 24)
    for name in ("prefix_id", "prompt_sha256", "text_sha256", "generated_token_ids_sha256",
                 "request_sha256"):
        digest(receipt[name])
    require(receipt["prefix_id"] in freeze["selection"], "receipt_unselected_prefix")
    integer(receipt["observed_token_count"], 0, 1024, "receipt_observed")
    require(type(receipt["right_censored"]) is bool, "receipt_censoring")
    require(receipt["prefix_id"] == sha({
        "review_id": receipt["review_id"], "observed_token_count": receipt["observed_token_count"],
        "right_censored": receipt["right_censored"],
        "generated_token_ids_sha256": receipt["generated_token_ids_sha256"],
        "generated_text_sha256": receipt["text_sha256"],
    }), "prefix_identity")
    integer(receipt["context_estimated_tokens"], 0, 2**30, "receipt_context")


def summarize_runs(runs: list[dict], numeric_mapping: dict | None = None) -> dict:
    """Aggregate typed receipts; absent selected jobs stay missing, never negative.

    Each run is {freeze, freeze_sha256, receipts, attempts}. Receipts are decoded durable
    documents supplied by the caller. File/hash checks belong to load_run(); the
    pure function independently rejects all scientific binding mismatches.
    """
    require(type(runs) is list and 1 <= len(runs) <= 8, "run_count")
    all_receipts = []
    judges = []
    freeze_hashes = set()
    metadata: dict[str, tuple] = {}
    first = None
    for run in runs:
        keys(run, {"freeze", "freeze_sha256", "receipts", "attempts"}, "run_keys")
        freeze, freeze_sha256 = run["freeze"], run["freeze_sha256"]
        validate_freeze(freeze, freeze_sha256)
        require(freeze_sha256 not in freeze_hashes, "duplicate_run")
        freeze_hashes.add(freeze_sha256)
        if first is None:
            first = freeze
        else:
            require(all(freeze[name] == first[name] for name in (
                "scope", "protocol_sha256", "packet_sha256", "selection",
            )), "cross_judge_population_or_protocol")
        require(type(run["receipts"]) is list and len(run["receipts"]) <= len(freeze["selection"]),
                "receipts_count")
        require(type(run["attempts"]) is list and len(run["attempts"]) <= len(freeze["selection"]),
                "attempts_count")
        attempts = {}
        for attempt in run["attempts"]:
            keys(attempt, BINDING_KEYS | {"schema_version"}, "attempt_keys")
            require(attempt["schema_version"] == "a161-attempt-v1", "attempt_version")
            _validate_bindings(attempt, freeze, freeze_sha256)
            prefix = attempt["prefix_id"]
            require(prefix not in attempts, "duplicate_attempt")
            attempts[prefix] = attempt
            binding = tuple(attempt[name] for name in (
                "review_id", "observed_token_count", "right_censored", "prompt_sha256",
                "text_sha256", "generated_token_ids_sha256",
            ))
            require(prefix not in metadata or metadata[prefix] == binding, "cross_judge_prefix_binding")
            metadata[prefix] = binding
        receipts = {}
        for receipt in run["receipts"]:
            validate_receipt(receipt, freeze, freeze_sha256)
            prefix = receipt["prefix_id"]
            require(prefix not in receipts, "duplicate_receipt")
            require(prefix in attempts, "receipt_missing_attempt")
            require(receipt["attempt_sha256"] == sha(attempts[prefix]), "receipt_attempt_hash")
            require(all(receipt[name] == attempts[prefix][name] for name in BINDING_KEYS),
                    "receipt_attempt_binding")
            receipts[prefix] = receipt
            binding = tuple(receipt[name] for name in (
                "review_id", "observed_token_count", "right_censored", "prompt_sha256",
                "text_sha256", "generated_token_ids_sha256",
            ))
            require(prefix not in metadata or metadata[prefix] == binding, "cross_judge_prefix_binding")
            metadata[prefix] = binding
        all_receipts.append(receipts)
        judges.append({"judge_index": len(judges), "judge_kind": "llm",
                       "model_name": freeze["config"]["model_name"],
                       "model_digest": freeze["config"]["model_digest"],
                       "recipe": freeze["config"]["recipe"],
                       "config_sha256": sha(freeze["config"]), "freeze_sha256": freeze_sha256,
                       "source_sha256": freeze["source_sha256"],
                       "qualification_sha256": freeze["qualification_sha256"],
                       "receipt_count": len(receipts),
                       "interrupted_attempts_without_receipt": len(set(attempts) - set(receipts)),
                       "unattempted_prefixes": len(freeze["selection"]) - len(attempts),
                       "attempt_manifest_sha256": sha([
                           {"prefix_id": p, "sha256": sha(attempts[p])} for p in sorted(attempts)
                       ]),
                       "receipt_manifest_sha256": sha([
                           {"prefix_id": p, "sha256": sha(receipts[p])} for p in sorted(receipts)
                       ])})
    mapping = None
    if numeric_mapping is not None:
        mapping = validate_mapping(numeric_mapping, first["packet_sha256"], first["selection"])
        case_to_review, review_to_case = {}, {}
        for prefix, row in mapping.items():
            if prefix in metadata:
                require(metadata[prefix][1:3] == (row["observed_token_count"], row["right_censored"]),
                        "mapping_receipt_prefix_binding")
                case, review = row["case_index"], metadata[prefix][0]
                require(case not in case_to_review or case_to_review[case] == review,
                        "mapping_case_review_join")
                require(review not in review_to_case or review_to_case[review] == case,
                        "mapping_review_case_join")
                case_to_review[case], review_to_case[review] = review, case
    overall = _summary(first["selection"], all_receipts, mapping)
    missing = sum(counts["missing"] for counts in overall["coverage_by_judge"])
    valid = sum(counts["valid"] for counts in overall["coverage_by_judge"])
    expected = len(first["selection"]) * len(runs)
    attempted = sum(len(run["attempts"]) for run in runs)
    status = ("not_started" if attempted == 0 else "incomplete" if missing else
              "complete_valid" if valid == expected else "complete_with_failures")
    result = {
        "schema_version": "a161-llm-audit-summary-v1", "status": status, "judge_kind": "llm",
        "scope": first["scope"], "protocol_sha256": first["protocol_sha256"],
        "packet_sha256": first["packet_sha256"], "judges": judges,
        "numeric_mapping_sha256": sha(numeric_mapping) if numeric_mapping is not None else None,
        "claim_boundaries": {
            "model_judgments_are_not_human_ratings_or_ground_truth": True,
            "concordance_is_not_accuracy": True,
            "repeated_prefix_horizon_aliases_are_not_independent": True,
            "shared_models_and_request_cores_are_not_independent": True,
            "no_confidence_intervals_or_population_extrapolation": True,
            "same_model_configs_do_not_become_independent_reviewers": True,
            "missing_or_failed_outputs_are_not_negative_judgments": True,
            "no_majority_vote_relabeling": True,
            "raw_text_or_case_prefix_identifiers_in_output": False,
            "first_harm_or_prevention_established": False,
            "raw_packet_or_model_requests_recomputed": False,
        },
        "overall": overall,
    }
    if mapping is not None:
        result["by_placement"] = {
            name: _summary([p for p, row in mapping.items() if row["placement"] == name],
                           all_receipts, mapping) for name in PLACEMENTS
        }
        result["by_intent_frame"] = {
            name: _summary([p for p, row in mapping.items() if row["intent_frame"] == name],
                           all_receipts, mapping) for name in INTENTS
        }
        result["by_sampling_stratum"] = [
            {"placement": placement, "frame_group": frame, "stratum": stratum,
             "summary": _summary([
                 p for p, row in mapping.items()
                 if row["placement"] == placement and row["stratum"] == stratum
                 and (row["intent_frame"] == "unsafe_direct") == (frame == "direct")
             ], all_receipts, mapping)}
            for placement, frame, stratum in itertools.product(PLACEMENTS, ("direct", "safe"), STRATA)
        ]
    return result


def _safe_path(path: Path) -> Path:
    path = path.absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink_path")
    return path


def read_json(path: Path, expected_sha256: str | None = None) -> dict:
    """Read only explicit bounded metadata. Never includes the path in errors."""
    path = _safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_JSON_BYTES, "metadata_file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
    finally:
        os.close(fd)
    require(len(raw) <= MAX_JSON_BYTES, "metadata_size")
    if expected_sha256 is not None:
        digest(expected_sha256)
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "metadata_hash")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise LLMAuditAnalysisError("a161_analysis_nonfinite_json")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise LLMAuditAnalysisError("a161_analysis_invalid_json") from None
    require(type(result) is dict, "metadata_object")
    require(canonical(result) == raw, "noncanonical_metadata")
    return result


def load_run(freeze_path: Path, expected_freeze_sha256: str) -> dict:
    """Read a quiescent native A161 store, verifying receipt/attempt/header lineage.

    A missing unstarted store yields all-missing coverage. An existing incomplete
    store is not repaired here. An attempt without receipt is reported as an
    interrupted unknown; it never authorizes the runner to retry.
    """
    freeze = read_json(freeze_path, expected_freeze_sha256)
    validate_freeze(freeze, expected_freeze_sha256)
    run = {"freeze": freeze, "freeze_sha256": expected_freeze_sha256, "attempts": [], "receipts": []}
    root = _safe_path(Path(freeze["output_root"]))
    if not root.exists():
        return run
    require(root.is_dir(), "store_directory")
    fd = os.open(_safe_path(root / ".writer.lock"), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            raise LLMAuditAnalysisError("a161_analysis_writer_active") from None
        require({path.name for path in root.iterdir()} == {
            ".writer.lock", "manifest.json", "attempts", "receipts",
        }, "store_inventory")
        header = read_json(root / "manifest.json")
        expected_header = {"schema_version": "a161-store-v1", "freeze_sha256": expected_freeze_sha256,
                           "config_sha256": sha(freeze["config"]),
                           "selected_prefixes": len(freeze["selection"])}
        require(header == expected_header, "store_manifest_binding")
        expected_names = {prefix + ".json" for prefix in freeze["selection"]}
        for kind in ("attempts", "receipts"):
            directory = _safe_path(root / kind)
            require(directory.is_dir(), "store_subdirectory")
            paths = sorted(directory.iterdir())
            require({path.name for path in paths} <= expected_names, "store_member")
            for path in paths:
                document = read_json(path)
                require(document.get("prefix_id") == path.stem, "store_filename_binding")
                run[kind].append(document)
        # The pure validator also checks missing/duplicate attempts and all hashes.
        summarize_runs([run])
        return run
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path,
                        help="Hash-bound metadata spec with runs and optional numeric mapping")
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        spec = read_json(args.inputs, args.inputs_sha256)
        keys(spec, {"schema_version", "runs", "numeric_mapping"}, "inputs_spec")
        require(spec["schema_version"] == "a161-analysis-inputs-v1", "inputs_version")
        require(type(spec["runs"]) is list and 1 <= len(spec["runs"]) <= 8, "inputs_runs")
        runs = []
        for entry in spec["runs"]:
            keys(entry, {"freeze_path", "freeze_sha256"}, "inputs_run")
            require(type(entry["freeze_path"]) is str, "inputs_freeze_path")
            runs.append(load_run(Path(entry["freeze_path"]), entry["freeze_sha256"]))
        mapping = None
        if spec["numeric_mapping"] is not None:
            entry = spec["numeric_mapping"]
            keys(entry, {"path", "sha256"}, "inputs_mapping")
            require(type(entry["path"]) is str, "inputs_mapping_path")
            mapping = read_json(Path(entry["path"]), entry["sha256"])
        result = summarize_runs(runs, mapping)
        result["inputs_spec_sha256"] = args.inputs_sha256
        result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw = canonical(result)
        output = _safe_path(args.output)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": result["status"], "judge_configurations": len(runs),
                          "selected_prefixes": result["overall"]["unique_prefixes"],
                          "result_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
        return 0
    except (OSError, LLMAuditAnalysisError, TypeError, ValueError, RecursionError):
        print('{"status":"a161_analysis_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
