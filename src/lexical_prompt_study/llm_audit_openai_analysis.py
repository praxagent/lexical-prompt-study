"""OpenAI-specific A161 receipt validation and numeric model-judge concordance.

Provider envelopes are privately replayed through the frozen runner parser before
loaded receipts leave this module. Only typed judgments and hashes are returned.
No API requests, raw study packet reads, or human-label claims occur here.
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

from .llm_audit_analysis import (
    A156_PACKET_SHA256,
    INTENTS,
    PLACEMENTS,
    STRATA,
    _safe_path,
    _summary,
    _validate_judgment,
    canonical,
    choice,
    digest,
    integer,
    keys,
    read_json,
    require,
    sha,
    validate_mapping,
)

FREEZE_FIELDS = {
    "schema_version", "scope", "run_id", "protocol_sha256", "source_sha256", "legacy_source_sha256",
    "packet_sha256", "fixture_sha256", "qualification_sha256", "qualification_path",
    "execution_authorized", "config", "selection", "output_root", "budget_root", "budget_micro_usd",
    "input_token_cap", "timeout_seconds", "max_seconds", "concurrency",
}
CONFIG_FIELDS = {
    "provider", "model_name", "endpoint", "recipe", "temperature", "max_output_tokens",
    "input_token_limit", "input_micro_usd_per_token", "output_micro_usd_per_token",
}
BINDING_FIELDS = {
    "freeze_sha256", "protocol_sha256", "source_sha256", "legacy_source_sha256", "packet_sha256",
    "fixture_sha256", "config_sha256", "request_sha256", "provider", "model_name", "recipe",
    "review_id", "prefix_id", "observed_token_count", "right_censored", "prompt_sha256",
    "text_sha256", "generated_token_ids_sha256", "context_estimated_tokens",
}
ATTEMPT_FIELDS = BINDING_FIELDS | {"schema_version", "budget_entry_id", "reserved_micro_usd"}
RECEIPT_FIELDS = BINDING_FIELDS | {
    "schema_version", "judge_kind", "status", "error_code", "attempt_sha256", "response_sha256",
    "http_status", "judgment", "input_tokens", "output_tokens", "charged_micro_usd",
    "billing_status", "elapsed_seconds", "budget_entry_id", "reserved_micro_usd",
}
LEGACY_SHA256 = "5ff1708b287e1d43954bb0266862607a169d28f76b27f412c22b0ec4be8f5a0a"
FIXTURE_SHA256 = "1dd2518ce872902bc1c4e4b20be0e2bfffe92fb8513466c4537adbb51ffcb3a8"
ERROR_CODES = (
    "none", "invalid_judgment", "provider_refusal", "incomplete_response", "response_contract",
    "usage_contract", "model_pin_mismatch", "pricing_tier_mismatch", "http_error", "runtime_error",
)


def validate_freeze(freeze: dict, expected_sha256: str) -> None:
    keys(freeze, FREEZE_FIELDS, "openai_freeze_fields")
    digest(expected_sha256)
    require(sha(freeze) == expected_sha256, "openai_freeze_hash")
    require(freeze["schema_version"] == "a161-openai-freeze-v1", "openai_freeze_version")
    choice(freeze["scope"], ("synthetic", "verified_a156"), "openai_scope")
    for field in ("protocol_sha256", "source_sha256", "legacy_source_sha256", "packet_sha256", "fixture_sha256"):
        digest(freeze[field])
    require(freeze["legacy_source_sha256"] == LEGACY_SHA256 and freeze["fixture_sha256"] == FIXTURE_SHA256,
            "openai_dependency_pins")
    require(freeze["execution_authorized"] is True and type(freeze["concurrency"]) is int
            and freeze["concurrency"] == 1, "openai_authority")
    for field in ("output_root", "budget_root"):
        require(type(freeze[field]) is str and Path(freeze[field]).is_absolute(), "openai_absolute_root")
    require(type(freeze["run_id"]) is str and re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", freeze["run_id"]),
            "openai_run_id")
    output_root, budget_root = Path(freeze["output_root"]), Path(freeze["budget_root"])
    require(output_root != budget_root and output_root not in budget_root.parents
            and budget_root not in output_root.parents, "openai_shared_root_overlap")
    integer(freeze["budget_micro_usd"], 1, 10_000_000, "openai_budget_cap")
    integer(freeze["input_token_cap"], 1, 5_000_000, "openai_input_cap")
    integer(freeze["timeout_seconds"], 1, 600, "openai_timeout")
    integer(freeze["max_seconds"], 1, 43200, "openai_duration")
    selection = freeze["selection"]
    require(type(selection) is list and 1 <= len(selection) <= 287, "openai_selection")
    for prefix in selection:
        digest(prefix)
    require(selection == sorted(set(selection)), "openai_selection_order")
    if freeze["scope"] == "synthetic":
        require(freeze["qualification_sha256"] is None and freeze["qualification_path"] is None,
                "openai_synthetic_qualification")
    else:
        digest(freeze["qualification_sha256"])
        require(freeze["qualification_sha256"] != "0" * 64
                and type(freeze["qualification_path"]) is str
                and Path(freeze["qualification_path"]).is_absolute(), "openai_qualification_binding")
        require(freeze["packet_sha256"] == A156_PACKET_SHA256 and len(selection) == 287,
                "openai_actual_selection")
    config = freeze["config"]
    keys(config, CONFIG_FIELDS, "openai_config_fields")
    require(config["provider"] == "openai" and config["model_name"] == "gpt-4.1-2025-04-14"
            and config["endpoint"] == "https://api.openai.com/v1/responses", "openai_provider_model")
    choice(config["recipe"], ("assistance_first", "utility_first"), "openai_recipe")
    require(type(config["temperature"]) is int and config["temperature"] == 0,
            "openai_temperature")
    integer(config["max_output_tokens"], 192, 192, "openai_output_cap")
    integer(config["input_token_limit"], 1024, 131072, "openai_context_limit")
    integer(config["input_micro_usd_per_token"], 2, 2, "openai_input_price")
    integer(config["output_micro_usd_per_token"], 8, 8, "openai_output_price")


def validate_bindings(value: dict, freeze: dict, freeze_sha256: str) -> None:
    require(value["freeze_sha256"] == freeze_sha256, "openai_receipt_freeze")
    for field in ("protocol_sha256", "source_sha256", "legacy_source_sha256", "packet_sha256", "fixture_sha256"):
        require(value[field] == freeze[field], "openai_receipt_source")
    require(value["config_sha256"] == sha(freeze["config"]), "openai_receipt_config")
    for field in ("provider", "model_name", "recipe"):
        require(value[field] == freeze["config"][field], "openai_receipt_provider")
    digest(value["review_id"], 24)
    for field in ("request_sha256", "prefix_id", "prompt_sha256", "text_sha256", "generated_token_ids_sha256"):
        digest(value[field])
    require(value["prefix_id"] in freeze["selection"], "openai_unselected_prefix")
    integer(value["observed_token_count"], 0, 1024, "openai_observed")
    require(type(value["right_censored"]) is bool, "openai_censoring")
    integer(value["context_estimated_tokens"], 1, freeze["config"]["input_token_limit"], "openai_context_estimate")
    require(value["prefix_id"] == sha({
        "review_id": value["review_id"], "observed_token_count": value["observed_token_count"],
        "right_censored": value["right_censored"], "generated_token_ids_sha256": value["generated_token_ids_sha256"],
        "generated_text_sha256": value["text_sha256"],
    }), "openai_prefix_identity")


def validate_attempt(attempt: dict, freeze: dict, freeze_sha256: str) -> None:
    keys(attempt, ATTEMPT_FIELDS, "openai_attempt_fields")
    require(attempt["schema_version"] == "a161-openai-attempt-v1", "openai_attempt_version")
    validate_bindings(attempt, freeze, freeze_sha256)
    digest(attempt["budget_entry_id"])
    integer(attempt["reserved_micro_usd"], 1, freeze["budget_micro_usd"], "openai_reservation")
    require(attempt["budget_entry_id"] == sha({"freeze_sha256": freeze_sha256,
                                              "prefix_id": attempt["prefix_id"]})
            and attempt["reserved_micro_usd"] == attempt["context_estimated_tokens"] * 2 + 192 * 8,
            "openai_reservation_binding")


def validate_receipt(receipt: dict, freeze: dict, freeze_sha256: str, attempt: dict) -> None:
    keys(receipt, RECEIPT_FIELDS, "openai_receipt_fields")
    require(receipt["schema_version"] == "a161-openai-receipt-v1" and receipt["judge_kind"] == "llm",
            "openai_receipt_version")
    validate_bindings(receipt, freeze, freeze_sha256)
    require(all(receipt[field] == attempt[field] for field in BINDING_FIELDS | {"budget_entry_id", "reserved_micro_usd"})
            and receipt["attempt_sha256"] == sha(attempt), "openai_receipt_attempt")
    choice(receipt["status"], ("valid", "schema_failed", "runtime_failed"), "openai_receipt_status")
    choice(receipt["error_code"], ERROR_CODES, "openai_error_code")
    if receipt["response_sha256"] is not None:
        digest(receipt["response_sha256"])
    if receipt["http_status"] is not None:
        integer(receipt["http_status"], 100, 599, "openai_http_status")
    require((receipt["response_sha256"] is None) == (receipt["http_status"] is None),
            "openai_response_http_binding")
    elapsed = receipt["elapsed_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "openai_elapsed")
    choice(receipt["billing_status"], ("metered", "reserved_unknown"), "openai_billing_status")
    integer(receipt["charged_micro_usd"], 1, receipt["reserved_micro_usd"], "openai_charge_bound")
    if receipt["billing_status"] == "metered":
        integer(receipt["input_tokens"], 1, receipt["context_estimated_tokens"], "openai_input_tokens")
        integer(receipt["output_tokens"], 0, freeze["config"]["max_output_tokens"], "openai_output_tokens")
        require(receipt["charged_micro_usd"] == receipt["input_tokens"] * 2 + receipt["output_tokens"] * 8,
                "openai_metered_arithmetic")
        require(receipt["response_sha256"] is not None, "openai_metered_response")
    else:
        require(receipt["input_tokens"] is None and receipt["output_tokens"] is None
                and receipt["charged_micro_usd"] == receipt["reserved_micro_usd"], "openai_unknown_usage")
    if receipt["status"] == "valid":
        _validate_judgment(receipt["judgment"])
        require(receipt["error_code"] == "none" and receipt["http_status"] == 200
                and receipt["response_sha256"] is not None and receipt["billing_status"] == "metered"
                and receipt["output_tokens"] > 0, "openai_valid_receipt")
    else:
        require(receipt["judgment"] is None and receipt["error_code"] != "none", "openai_failed_receipt")
        allowed = {"schema_failed": ("invalid_judgment", "provider_refusal", "incomplete_response",
                                      "response_contract", "usage_contract"),
                   "runtime_failed": ("http_error", "runtime_error", "model_pin_mismatch", "pricing_tier_mismatch")}
        choice(receipt["error_code"], allowed[receipt["status"]], "openai_error_status_binding")


def summarize_runs(runs: list[dict], numeric_mapping: dict | None = None,
                   utility_mapping: dict | None = None) -> dict:
    require(type(runs) is list and 1 <= len(runs) <= 8, "openai_runs")
    baseline = None
    run_hashes, metadata, judges, receipt_lists = set(), {}, [], []
    for run in runs:
        keys(run, {"freeze", "freeze_sha256", "attempts", "receipts", "response_hashes"}, "openai_run_fields")
        freeze, freeze_sha256 = run["freeze"], run["freeze_sha256"]
        validate_freeze(freeze, freeze_sha256)
        require(freeze_sha256 not in run_hashes, "openai_duplicate_run")
        run_hashes.add(freeze_sha256)
        if baseline is None:
            baseline = freeze
        else:
            require(all(freeze[field] == baseline[field] for field in (
                "scope", "packet_sha256", "fixture_sha256", "protocol_sha256", "selection",
            )), "openai_cross_run_population")
        require(type(run["attempts"]) is list and type(run["receipts"]) is list, "openai_run_lists")
        require(len(run["attempts"]) <= len(freeze["selection"])
                and len(run["receipts"]) <= len(freeze["selection"]), "openai_run_counts")
        require(type(run["response_hashes"]) is dict
                and set(run["response_hashes"]) <= set(freeze["selection"]), "openai_response_inventory")
        for response_sha in run["response_hashes"].values():
            digest(response_sha)
        attempts, receipts = {}, {}
        for attempt in run["attempts"]:
            validate_attempt(attempt, freeze, freeze_sha256)
            prefix = attempt["prefix_id"]
            require(prefix not in attempts, "openai_duplicate_attempt")
            attempts[prefix] = attempt
            binding = tuple(attempt[field] for field in (
                "review_id", "observed_token_count", "right_censored", "prompt_sha256",
                "text_sha256", "generated_token_ids_sha256",
            ))
            require(prefix not in metadata or metadata[prefix] == binding, "openai_cross_run_prefix")
            metadata[prefix] = binding
        require(set(run["response_hashes"]) <= set(attempts), "openai_response_missing_attempt")
        require(sum(a["context_estimated_tokens"] for a in attempts.values()) <= freeze["input_token_cap"],
                "openai_attempted_input_cap")
        for receipt in run["receipts"]:
            require(type(receipt) is dict and receipt.get("prefix_id") in attempts, "openai_receipt_missing_attempt")
            prefix = receipt["prefix_id"]
            require(prefix not in receipts, "openai_duplicate_receipt")
            validate_receipt(receipt, freeze, freeze_sha256, attempts[prefix])
            require(receipt["response_sha256"] == run["response_hashes"].get(prefix), "openai_response_file_hash")
            receipts[prefix] = receipt
        receipt_lists.append(receipts)
        interrupted = set(attempts) - set(receipts)
        judges.append({
            "judge_index": len(judges), "provider": "openai", "judge_kind": "llm",
            "model_name": freeze["config"]["model_name"], "recipe": freeze["config"]["recipe"],
            "freeze_sha256": freeze_sha256, "config_sha256": sha(freeze["config"]),
            "source_sha256": freeze["source_sha256"], "legacy_source_sha256": freeze["legacy_source_sha256"],
            "qualification_sha256": freeze["qualification_sha256"],
            "receipt_count": len(receipts), "unattempted_prefixes": len(freeze["selection"]) - len(attempts),
            "interrupted_attempts_without_receipt": len(interrupted),
            "attempt_manifest_sha256": sha([{ "prefix_id": p, "sha256": sha(attempts[p])} for p in sorted(attempts)]),
            "receipt_manifest_sha256": sha([{ "prefix_id": p, "sha256": sha(receipts[p])} for p in sorted(receipts)]),
            "billing": {
                "interpretation": "token_price_estimate_or_conservative_reserve_not_settled_invoice",
                "shared_budget_ledger_verified_by_this_aggregate": False,
                "metered_micro_usd": sum(r["charged_micro_usd"] for r in receipts.values() if r["billing_status"] == "metered"),
                "reserved_unknown_micro_usd": sum(r["charged_micro_usd"] for r in receipts.values() if r["billing_status"] == "reserved_unknown"),
                "interrupted_reserved_micro_usd": sum(attempts[p]["reserved_micro_usd"] for p in interrupted),
            },
        })
    mapping = None
    if numeric_mapping is not None:
        mapping = validate_mapping(numeric_mapping, baseline["packet_sha256"], baseline["selection"])
        case_to_review, review_to_case = {}, {}
        for prefix, row in mapping.items():
            if prefix in metadata:
                require(metadata[prefix][1:3] == (row["observed_token_count"], row["right_censored"]),
                        "openai_mapping_prefix")
                case, review = row["case_index"], metadata[prefix][0]
                require(case not in case_to_review or case_to_review[case] == review, "openai_mapping_case")
                require(review not in review_to_case or review_to_case[review] == case, "openai_mapping_review")
                case_to_review[case], review_to_case[review] = review, case
    overall = _summary(baseline["selection"], receipt_lists, mapping)
    attempts = sum(len(run["attempts"]) for run in runs)
    missing = sum(row["missing"] for row in overall["coverage_by_judge"])
    valid = sum(row["valid"] for row in overall["coverage_by_judge"])
    expected = len(runs) * len(baseline["selection"])
    status = ("not_started" if not attempts else "incomplete" if missing else
              "complete_valid" if valid == expected else "complete_with_failures")
    result = {
        "schema_version": "a161-openai-audit-summary-v1", "provider": "openai", "judge_kind": "llm",
        "status": status, "scope": baseline["scope"], "protocol_sha256": baseline["protocol_sha256"],
        "packet_sha256": baseline["packet_sha256"], "fixture_sha256": baseline["fixture_sha256"],
        "judges": judges, "overall": overall,
        "numeric_mapping_sha256": sha(numeric_mapping) if numeric_mapping is not None else None,
        "claim_boundaries": {
            "model_judgments_are_not_human_ratings_or_ground_truth": True,
            "concordance_is_not_accuracy": True, "missing_or_failed_outputs_are_not_negative": True,
            "shared_models_recipes_request_cores_and_aliases_are_not_independent": True,
            "raw_provider_envelope_content_not_in_summary": True,
            "load_run_replays_response_bodies_with_frozen_runner": True,
            "api_requests_or_raw_packet_not_recomputed": True,
            "first_harm_or_prevention_established": False,
            "raw_text_or_case_prefix_identifiers_in_output": False,
        },
    }
    if mapping is not None:
        result["by_placement"] = {
            name: _summary([p for p, row in mapping.items() if row["placement"] == name], receipt_lists, mapping)
            for name in PLACEMENTS
        }
        result["by_intent_frame"] = {
            name: _summary([p for p, row in mapping.items() if row["intent_frame"] == name], receipt_lists, mapping)
            for name in INTENTS
        }
        result["by_sampling_stratum"] = [
            {"placement": placement, "frame_group": frame, "stratum": stratum,
             "summary": _summary([p for p, row in mapping.items()
                                  if row["placement"] == placement and row["stratum"] == stratum
                                  and (row["intent_frame"] == "unsafe_direct") == (frame == "direct")],
                                 receipt_lists, mapping)}
            for placement, frame, stratum in itertools.product(PLACEMENTS, ("direct", "safe"), STRATA)
        ]
    if utility_mapping is not None:
        require(numeric_mapping is not None, "openai_utility_requires_numeric_mapping")
        from .llm_audit_utility import summarize_utility
        result["semantic_vs_mechanical_utility"] = summarize_utility(
            utility_mapping, numeric_mapping, [list(receipts.values()) for receipts in receipt_lists])
        def mark_utility_section(value):
            if type(value) is dict:
                if "semantic_vs_mechanical_utility" in value:
                    value["semantic_vs_mechanical_utility"] = {
                        "status": "available_in_top_level_semantic_vs_mechanical_utility",
                        "subgroup_tables_not_computed": value is not overall,
                    }
                for child in value.values():
                    mark_utility_section(child)
            elif type(value) is list:
                for child in value:
                    mark_utility_section(child)
        for section in (overall, result.get("by_placement"), result.get("by_intent_frame"),
                        result.get("by_sampling_stratum")):
            mark_utility_section(section)
    return result


def _response_file_sha(path: Path) -> str:
    """Hash the opaque response envelope without parsing body_hex or provider text."""
    path = _safe_path(path)
    require(path.is_file() and path.stat().st_size <= 4 * 1024**2, "openai_response_file_bound")
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def load_run(freeze_path: Path, expected_freeze_sha256: str) -> dict:
    freeze = read_json(freeze_path, expected_freeze_sha256)
    validate_freeze(freeze, expected_freeze_sha256)
    run = {"freeze": freeze, "freeze_sha256": expected_freeze_sha256,
           "attempts": [], "receipts": [], "response_hashes": {}}
    root = _safe_path(Path(freeze["output_root"]))
    if not root.exists():
        return run
    require(root.is_dir(), "openai_store_directory")
    from . import llm_audit_runner as legacy_runner
    from . import llm_audit_openai as provider_runner
    require(hashlib.sha256(Path(provider_runner.__file__).read_bytes()).hexdigest()
            == freeze["source_sha256"], "openai_replay_runner_source")
    require(hashlib.sha256(Path(legacy_runner.__file__).read_bytes()).hexdigest()
            == freeze["legacy_source_sha256"], "openai_replay_legacy_source")
    fd = os.open(_safe_path(root / ".writer.lock"), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        require({path.name for path in root.iterdir()} == {
            ".writer.lock", "manifest.json", "attempts", "receipts", "responses",
        }, "openai_store_inventory")
        require(read_json(root / "manifest.json") == {
            "schema_version": "a161-openai-store-v1", "freeze_sha256": expected_freeze_sha256,
            "config_sha256": sha(freeze["config"]), "selected_prefixes": len(freeze["selection"]),
        }, "openai_store_manifest")
        expected_names = {prefix + ".json" for prefix in freeze["selection"]}
        for kind in ("attempts", "receipts", "responses"):
            directory = _safe_path(root / kind)
            require(directory.is_dir(), "openai_store_subdirectory")
            paths = sorted(directory.iterdir())
            require({path.name for path in paths} <= expected_names, "openai_store_member")
            for path in paths:
                if kind == "responses":
                    run["response_hashes"][path.stem] = _response_file_sha(path)
                else:
                    value = read_json(path)
                    require(value.get("prefix_id") == path.stem, "openai_store_filename")
                    run[kind].append(value)
        summarize_runs([run])
        attempts = {attempt["prefix_id"]: attempt for attempt in run["attempts"]}
        for receipt in run["receipts"]:
            prefix = receipt["prefix_id"]
            response_path = root / "responses" / (prefix + ".json") if prefix in run["response_hashes"] else None
            reconstructed = provider_runner._receipt(
                attempts[prefix], sha(attempts[prefix]), response_path, receipt["elapsed_seconds"])
            require(canonical(reconstructed) == canonical(receipt), "openai_body_receipt_replay")
        return run
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        spec = read_json(args.inputs, args.inputs_sha256)
        keys(spec, {"schema_version", "runs", "numeric_mapping", "utility_mapping"}, "openai_analysis_inputs")
        require(spec["schema_version"] == "a161-openai-analysis-inputs-v1", "openai_analysis_inputs_version")
        require(type(spec["runs"]) is list and 1 <= len(spec["runs"]) <= 8, "openai_analysis_runs")
        runs = []
        for entry in spec["runs"]:
            keys(entry, {"freeze_path", "freeze_sha256"}, "openai_analysis_run")
            runs.append(load_run(Path(entry["freeze_path"]), entry["freeze_sha256"]))
        mappings = {}
        for name in ("numeric_mapping", "utility_mapping"):
            mappings[name] = None
            if spec[name] is not None:
                keys(spec[name], {"path", "sha256"}, "openai_analysis_mapping")
                mappings[name] = read_json(Path(spec[name]["path"]), spec[name]["sha256"])
        result = summarize_runs(runs, **mappings)
        result["inputs_spec_sha256"] = args.inputs_sha256
        result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["shared_counting_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("llm_audit_analysis.py").read_bytes()).hexdigest()
        raw = canonical(result)
        fd = os.open(_safe_path(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": result["status"], "provider": "openai", "judge_configurations": len(runs),
                          "selected_prefixes": result["overall"]["unique_prefixes"],
                          "result_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
        return 0
    except Exception:
        print('{"status":"a161_openai_analysis_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
